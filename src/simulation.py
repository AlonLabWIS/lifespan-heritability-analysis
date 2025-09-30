import numpy as np
from numba import njit
from sympy import symbols, sympify, lambdify
from multiprocessing import Pool, cpu_count
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from scipy.stats import gaussian_kde, gennorm, norm, gamma, beta as beta_dist
import collections # Import callable


class SimulationParams:
    """
    A class to manage simulation parameters for the SR model.
    
    This class handles parameter processing, drift function setup, and parameter validation
    for simulating stochastic resilience (SR) paths.

    Attributes:
        raw_* : Original input parameters before processing
        n (int): Number of simulations to run
        tmin (float): Start time of simulation
        tmax (float): End time of simulation
        dt (float): Time step size
        save_times (float): Interval for saving simulation states
        parallel (bool): Whether to run simulation in parallel
        drift_expr (str/expr): Custom drift function expression
        drift_mode (str): How to apply custom drift ('replace' or 'add')
        extra_params (dict): Additional parameters for custom drift
    """
    
    def __init__(self, eta, beta, kappa, epsilon, Xc, n=10000, tmin=0, tmax=1000, x0=1e-10,
                 dt=1, save_times=None, h_ext=None, units='days', parallel=False, break_early=True,
                 drift_expr=None, drift_mode='replace', extra_params=None, start_idx=0):
        """
        Initialize simulation parameters.

        Args:
            eta (float/array): Production rate
            beta (float/array): Removal rate
            kappa (float/array): Half-saturation constant
            epsilon (float/array): Noise intensity
            Xc (float/array): Critical threshold
            n (int): Number of simulations
            tmin (float): Start time
            tmax (float): End time
            x0 (float): Initial condition
            dt (float): Time step
            save_times (float): Save interval
            h_ext (float/callable/array/list/None): External hazard rate.
                - float: Constant hazard rate for all agents.
                - callable: Function h(t) returning hazard rate at time t for all agents.
                - np.ndarray: Array of length n specifying a constant hazard rate for each agent.
                - list of callables: List of length n with a specific h(t) for each agent.
                - None: No external hazard.
            units (str): Time units ('days' or 'years')
            parallel (bool): Enable parallel processing
            break_early (bool): Stop if all paths cross threshold
            drift_expr (str/expr): Custom drift expression
            drift_mode (str): How to apply custom drift
            extra_params (dict): Additional drift parameters
            start_idx (int): Starting index for parallel chunks
        """
        # Store raw parameters
        self.raw_eta = eta
        self.raw_beta = beta
        self.raw_kappa = kappa
        self.raw_epsilon = epsilon
        self.raw_Xc = Xc
        
        # Store other parameters
        self.n = n
        self.tmin = tmin
        self.tmax = tmax
        self.x0 = x0
        self.dt = dt
        self.save_times = save_times if save_times is not None else dt
        
        # Validate and store h_ext
        is_list_of_callables = isinstance(h_ext, list) and all(callable(f) for f in h_ext)
        if h_ext is not None and not isinstance(h_ext, (float, np.ndarray)) and not callable(h_ext) and not is_list_of_callables:
            raise TypeError("h_ext must be None, a float, a callable, a NumPy array, or a list of callables.")

        if isinstance(h_ext, np.ndarray):
            if h_ext.ndim != 1 or len(h_ext) != n:
                raise ValueError(f"If h_ext is an array, it must be 1D and have length n={n}, but got shape {h_ext.shape}")
        elif is_list_of_callables:
            if len(h_ext) != n:
                raise ValueError(f"If h_ext is a list of functions, it must have length n={n}, but got length {len(h_ext)}")

        self.h_ext = h_ext # Store h_ext directly

        self.units = units
        self.parallel = parallel
        self.break_early = break_early
        self.start_idx = start_idx
        
        # Drift related - store only configuration, not the function
        self.drift_expr = drift_expr
        self.drift_mode = drift_mode
        self.extra_params = extra_params or {}
        
        # Process parameters for actual simulation
        self._process_all_parameters()
        
        # Setup drift function after processing parameters
        self.drift_func = None  # Will be set up when needed

    def _process_all_parameters(self):
        """Process all raw parameters into arrays of appropriate length."""
        self.eta = self._process_parameter(self.raw_eta)
        self.beta = self._process_parameter(self.raw_beta)
        self.kappa = self._process_parameter(self.raw_kappa)
        self.epsilon = self._process_parameter(self.raw_epsilon)
        self.Xc = self._process_parameter(self.raw_Xc)
        self._calc_derived_params()
    def _process_parameter(self, param):
        """
        Convert parameter to array of appropriate length.

        Args:
            param: Scalar or array parameter value

        Returns:
            np.ndarray: Parameter array of length self.n

        Raises:
            ValueError: If parameter can't be converted to correct length
        """
        if np.isscalar(param) or np.size(param) <= 1:
            return np.full(self.n, param)
        elif len(param) == self.n:
            return np.array(param)
        else:
            raise ValueError(f"{param} must be a scalar or an array of length {self.n} but is sized {len(param)}")

    def _calc_derived_params(self):
        """
        Calculate derived parameters used in analysis.
        
        Computes various combinations of basic parameters that are useful
        for analyzing the system behavior, including characteristic times
        and dimensionless parameters.
        """
        self.p1 = self.beta * self.kappa / self.epsilon
        self.p2 = self.eta * self.kappa / (self.beta**2)
        self.p3 = self.Xc / self.kappa
        self.alpha = self.eta * (self.kappa + self.Xc) / self.epsilon
        self.tau = self.beta / self.eta
        self.t_r = self.kappa / self.beta
        self.t_D = self.kappa**2 / self.epsilon
        self.t_p = np.sqrt(self.kappa / self.eta)

    @staticmethod
    @njit
    def _default_drift(X, tcur, eta, beta, kappa):
        """
        Default SR model drift function.

        Args:
            X (np.ndarray): Current state
            tcur (float): Current time
            eta (np.ndarray): Production rate
            beta (np.ndarray): Removal rate
            kappa (np.ndarray): Half-saturation constant

        Returns:
            np.ndarray: Drift term for SR model
        """
        return eta * tcur - X * (beta / (X + kappa))

    def _setup_drift(self):
        """
        Set up drift function based on configuration.
        
        Creates appropriate drift function based on drift_expr and drift_mode.
        Handles custom expressions, parameter substitution, and compilation.

        Returns:
            callable: Configured drift function
        """
        if self.drift_expr is None:
            return self._default_drift
        
        try:
            # Define symbols with explicit Symbol constructor to avoid conflicts
            X = symbols('X')
            t = symbols('t')
            eta = symbols('eta', real=True)  # real=True tells SymPy these are not functions
            beta = symbols('beta', real=True)
            kappa = symbols('kappa', real=True)
            basic_params = {'X': X, 't': t, 'eta': eta, 'beta': beta, 'kappa': kappa}
            
            
            # Add extra parameters if provided
            extra_syms = {}
            if self.extra_params:
                extra_syms = {k: symbols(k) for k in self.extra_params.keys()}
                basic_params.update(extra_syms)
            
            # Parse the expression
            if isinstance(self.drift_expr, str):
                expr = sympify(self.drift_expr, locals={'beta': beta})
            else:
                expr = self.drift_expr
                
            # If mode is 'add', add to default drift
            if self.drift_mode == 'add':
                default_expr = eta * t - X * (beta / (X + kappa))
                expr = default_expr + expr
            
            # Create a lambda function
            drift_lambda = lambdify(tuple(basic_params.values()), expr)
            
            # Return a wrapper function that matches the expected signature but can use extra params
            def drift_wrapper(X, tcur, eta, beta, kappa, **extra):
                params = {
                    'X': X, 't': tcur,
                    'eta': eta, 
                    'beta': beta, 
                    'kappa': kappa
                }
                params.update(extra)
                return drift_lambda(**params)
                
            return drift_wrapper
            
        except Exception as e:
            print(f"Error setting up custom drift: {e}")
            print("Falling back to default drift")
            return self._default_drift

    def get_drift(self, X, tcur):
        """
        Calculate drift at current state and time.

        Args:
            X (np.ndarray): Current state
            tcur (float): Current time

        Returns:
            np.ndarray: Drift values
        """
        # Setup drift function if not already set up
        if self.drift_func is None:
            self.drift_func = self._setup_drift()
            
        params = {
            'X': X,
            'tcur': tcur,
            'eta': self.eta,
            'beta': self.beta,
            'kappa': self.kappa
        }
        if self.extra_params:
            params.update(self.extra_params)
        return self.drift_func(**params)

    def create_chunk_params(self, start_idx, chunk_size):
        """
        Create parameter object for parallel chunk processing.

        Args:
            start_idx (int): Starting index for this chunk
            chunk_size (int): Number of paths in chunk

        Returns:
            SimulationParams: New parameter object for chunk
        """
        # Slice the raw parameters if they are arrays of length n
        def slice_param(param, n_total):
            if isinstance(param, (np.ndarray, list)):
                if len(param) == n_total: # Only slice if it's an agent-specific array or list
                    return param[start_idx:start_idx + chunk_size]
            # Otherwise (scalar, single callable, or already chunked), return as is
            return param

        # Pre-slice all raw parameters
        raw_eta_chunk = slice_param(self.raw_eta, self.n)
        raw_beta_chunk = slice_param(self.raw_beta, self.n)
        raw_kappa_chunk = slice_param(self.raw_kappa, self.n)
        raw_epsilon_chunk = slice_param(self.raw_epsilon, self.n)
        raw_Xc_chunk = slice_param(self.raw_Xc, self.n)
        # Handle h_ext separately as it can be callable or float
        h_ext_chunk = slice_param(self.h_ext, self.n) # Slice only if it's an agent-specific array


        chunk_params = SimulationParams(
            eta=raw_eta_chunk,
            beta=raw_beta_chunk,
            kappa=raw_kappa_chunk,
            epsilon=raw_epsilon_chunk,
            Xc=raw_Xc_chunk,
            n=chunk_size,
            tmin=self.tmin,
            tmax=self.tmax,
            x0=self.x0,
            dt=self.dt,
            save_times=self.save_times,
            h_ext=h_ext_chunk, # Pass the (potentially sliced) h_ext
            break_early=self.break_early,
            drift_expr=self.drift_expr,
            drift_mode=self.drift_mode,
            extra_params=self.extra_params,
            start_idx=start_idx
        )
        return chunk_params
        
    def print_params(self):
        """
        Print a concise summary of core simulation parameters.
        """
        eta_val = np.mean(self.eta) / 365
        beta_val = np.mean(self.beta) / 365
        kappa_val = np.mean(self.kappa)
        epsilon_val = np.mean(self.epsilon)
        xc_val = np.mean(self.Xc)

        print_str = (
            f"η = {eta_val:.5f} day⁻¹·year⁻¹, "
            f"β = {beta_val:.2f} day⁻¹, "
            f"κ = {kappa_val:.1f}, "
            f"ε = {epsilon_val:.3f} day⁻¹, "
            f"Xc = {int(round(xc_val))}"
        )
        print(print_str)

    def print_full_params_summary(self):
        """
        Print a detailed summary of all simulation parameters.
        
        For array parameters, prints mean and standard deviation if the array
        has more than one unique value.
        """
        print("=== Simulation Parameters ===")
        
        # Core SR model parameters
        print("\nCore SR Model Parameters:")
        for param_name, param_array in [
            ("eta", self.eta),
            ("beta", self.beta),
            ("kappa", self.kappa),
            ("epsilon", self.epsilon),
            ("Xc", self.Xc)
        ]:
            if len(np.unique(param_array)) == 1:
                # Single value parameter
                print(f"  {param_name} = {param_array[0]:.6g}")
            else:
                # Array parameter with multiple values
                mean_val = np.mean(param_array)
                std_val = np.std(param_array)
                min_val = np.min(param_array)
                max_val = np.max(param_array)
                print(f"  {param_name} = {mean_val:.6g} ± {std_val:.6g} (mean ± std)")
                print(f"    range: [{min_val:.6g}, {max_val:.6g}]")
        
        # Derived parameters
        print("\nDerived Parameters:")
        for param_name, param_array in [
            ("p1", self.p1),
            ("p2", self.p2),
            ("p3", self.p3),
            ("alpha", self.alpha),
            ("tau", self.tau),
            ("t_r", self.t_r),
            ("t_D", self.t_D),
            ("t_p", self.t_p)
        ]:
            if len(np.unique(param_array)) == 1:
                print(f"  {param_name} = {param_array[0]:.6g}")
            else:
                mean_val = np.mean(param_array)
                std_val = np.std(param_array)
                print(f"  {param_name} = {mean_val:.6g} ± {std_val:.6g}")
        
        # Simulation settings
        print("\nSimulation Settings:")
        print(f"  Number of simulations (n) = {self.n}")
        print(f"  Time range = [{self.tmin}, {self.tmax}] {self.units}")
        print(f"  Time step (dt) = {self.dt} {self.units}")
        print(f"  Save interval = {self.save_times} {self.units}")
        print(f"  Initial condition (x0) = {self.x0}")
        
        # External hazard
        print("\nExternal Hazard:")
        if self.h_ext is None:
            print("  None")
        elif isinstance(self.h_ext, float):
            print(f"  Constant rate: {self.h_ext}")
        elif isinstance(self.h_ext, np.ndarray):
            mean_h = np.mean(self.h_ext)
            std_h = np.std(self.h_ext)
            if std_h < 1e-10:  # Effectively constant
                print(f"  Constant rate: {mean_h:.6g}")
            else:
                print(f"  Variable rate: {mean_h:.6g} ± {std_h:.6g}")
        elif isinstance(self.h_ext, list):
            print("  List of agent-specific hazard functions")
        else:
            print("  Time-dependent function")
        
        # Other settings
        print("\nOther Settings:")
        print(f"  Parallel processing: {self.parallel}")
        print(f"  Break early: {self.break_early}")
        
        # Custom drift
        print("\nDrift Function:")
        if self.drift_expr is None:
            print("  Default SR model drift")
        else:
            print(f"  Custom drift ({self.drift_mode} mode)")
            print(f"  Expression: {self.drift_expr}")
            if self.extra_params:
                print("  Extra parameters:")
                for k, v in self.extra_params.items():
                    print(f"    {k} = {v}")

class SR_sim:
    """
    Saturated Removal (SR) simulation class.
    
    Handles simulation of SR paths, including parallel processing,
    path generation, and statistical analysis of results.
    """
    
    def __init__(self, params):
        """
        Initialize SR simulation.

        Args:
            params (SimulationParams): Simulation parameters
        """
        self.params = params
        self.paths = None
        self.death_times = None
        self.tspan = None
        self.alive_mask = None
        self.extrinsic_deaths = None  # New attribute
        self.run_simulation()
        self._post_simulation_calculations()

    def run_simulation(self):
        if self.params.parallel:
            self.death_times, self.paths, self.tspan, self.alive_mask, self.extrinsic_deaths = self._create_paths_parallel()
        else:
            self.death_times, self.paths, self.tspan, self.alive_mask, self.extrinsic_deaths = self._create_paths()

    def _create_paths(self):
        """
        Generate simulation paths for serial processing.

        Returns:
            tuple: (death_times, paths, tspan_save_times, alive_mask, extrinsic_deaths)
        """
        tspan = np.arange(self.params.tmin, self.params.tmax + 0.000001, self.params.dt)
        tspan_save_times = np.arange(self.params.tmin, self.params.tmax + 0.000001, self.params.save_times)
        paths = np.ones((self.params.n, len(tspan_save_times)))
        paths[:, 0] = self.params.x0
        
        death_times, paths, tspan_save, alive_mask, extrinsic = self._simulate_paths(paths, tspan, tspan_save_times)
        return death_times, paths, tspan_save, alive_mask, extrinsic  # Return all 5 values

    def _create_paths_parallel(self):
        """
        Generate simulation paths using parallel processing.

        Returns:
            tuple: (death_times, paths, tspan, alive_mask, extrinsic_deaths)
        """
        num_cores = cpu_count()
        chunk_size = self.params.n // num_cores
        chunk_params = [self.params.create_chunk_params(i * chunk_size, chunk_size) 
                       for i in range(num_cores)]
        
        with Pool(num_cores) as pool:
            results = pool.map(self._run_chunk, chunk_params)
        
        return self._merge_chunks(results)

    @staticmethod
    def _run_chunk(params):
        """
        Run simulation for a chunk of parameters.

        Args:
            params (SimulationParams): Parameters for the chunk

        Returns:
            tuple: (start_idx, death_times, paths, alive_mask, extrinsic_deaths)
        """
        # Ensure drift function is set up in the child process
        params.drift_func = params._setup_drift()
        sim = SR_sim(params)
        return params.start_idx, sim.death_times, sim.paths, sim.alive_mask, sim.extrinsic_deaths

    def _merge_chunks(self, results):
        """
        Merge results from parallel chunks while preserving vector structure.

        Args:
            results (list): List of results from parallel chunks

        Returns:
            tuple: (death_times, paths, tspan, alive_mask, extrinsic_deaths)
        """
        # Determine maximum time steps across all chunks
        max_time_steps = max(chunk_paths.shape[1] for _, _, chunk_paths, _, _ in results)
        
        # Initialize arrays for merged results
        death_times = np.full(self.params.n, np.inf)
        paths = np.zeros((self.params.n, max_time_steps))
        alive_mask = np.zeros(self.params.n, dtype=bool)
        extrinsic_deaths = np.zeros(self.params.n, dtype=int)
        
        # Place each chunk's results in the correct position
        for start_idx, chunk_death_times, chunk_paths, chunk_alive_mask, chunk_extrinsic in results:
            end_idx = start_idx + len(chunk_death_times)
            paths[start_idx:end_idx, :chunk_paths.shape[1]] = chunk_paths
            death_times[start_idx:end_idx] = chunk_death_times
            alive_mask[start_idx:end_idx] = chunk_alive_mask
            extrinsic_deaths[start_idx:end_idx] = chunk_extrinsic
        
        # Create corresponding time span
        tspan = np.arange(self.params.tmin, 
                         self.params.tmax + 0.01, 
                         self.params.save_times)[:max_time_steps]
        
        # Ensure death times don't exceed simulation time
        death_times = np.minimum(death_times, self.params.tmax + self.params.dt)
        
        return death_times, paths, tspan, alive_mask, extrinsic_deaths

    def _skorokhod_step(self, X, tcur):
        """
        Perform one step of Skorokhod simulation.

        Implements reflecting boundary condition at zero and
        combines drift and noise terms.

        Args:
            X (np.ndarray): Current state
            tcur (float): Current time

        Returns:
            np.ndarray: Next state
        """
        noise = np.sqrt(2 * self.params.epsilon) * (X > 0)
        Y = noise * np.sqrt(self.params.dt) * np.random.standard_normal(X.shape)
        U = np.random.random(X.shape)
        M = (Y + np.sqrt(Y**2 - 2 * self.params.dt * np.log(U))) / 2
        drift = self.params.get_drift(X, tcur)
        delta_X = self.params.dt * drift
        return np.maximum(M-Y, X + delta_X - Y)

    @staticmethod
    @njit
    def _find_first_crossings_numba(X, Xc, p_death_ext):
        """
        Identifies crossings and potential extrinsic deaths for a subset of agents.
        Numba-jitted for performance.

        Args:
            X (np.ndarray): Current states for the subset of agents being checked.
            Xc (float/np.ndarray): Threshold values (scalar or subset array).
            p_death_ext (np.ndarray): Probability of extrinsic death for each agent in this subset.

        Returns:
            tuple: (died_mask, extrinsic_mask)
                   Boolean masks relative to the input subset X.
        """
        # Handle empty input case (if no agents were alive)
        if X.size == 0:
            return np.array([False]), np.array([False]) # Return dummy non-empty boolean arrays

        crossed = (X > Xc) # Check crossing against threshold
        extrinsic_cause = np.zeros(X.shape[0], dtype=np.bool_) # Use shape[0] for length
        died_mask = crossed.copy() # Start with only intrinsic deaths, use copy

        # Determine if any extrinsic death check is needed for this batch
        # Since p_death_ext is always an array, just check if any element > 0
        check_extrinsic = False
        # Check if the array is not empty before calling np.any
        if p_death_ext.size > 0 and np.any(p_death_ext > 0.0):
             check_extrinsic = True

        if check_extrinsic:
            random_nums = np.random.random(X.shape[0]) # Use X.shape[0] for length
            # Comparison works element-wise because p_death_ext is always an array now
            external_death_event = (random_nums < p_death_ext)
            died_mask = died_mask | external_death_event # Combine intrinsic and extrinsic
            extrinsic_cause[external_death_event] = True # Assign True where event occurred

        # Return masks relative to the input subset X
        return died_mask, extrinsic_cause

    def _simulate_paths(self, paths, tspan, tspan_save_times):
        """
        Core simulation loop for generating paths.

        Args:
            paths (np.ndarray): Array to store paths
            tspan (np.ndarray): Time points
            tspan_save_times (np.ndarray): Times to save results

        Returns:
            tuple: (death_times, paths, tspan_save_times, alive_mask, extrinsic_deaths)
        """
        death_times = np.full(self.params.n, np.inf)
        self.extrinsic_deaths = np.zeros(self.params.n, dtype=int)
        # alive_mask is determined at the end based on death_times
        save_interval = int(self.params.save_times / self.params.dt)
        
        if np.isscalar(self.params.x0):
             X = np.full(self.params.n, self.params.x0, dtype=float)
        elif len(self.params.x0) == self.params.n:
             X = np.array(self.params.x0, dtype=float)
        else:
             raise ValueError(f"x0 must be a scalar or array of length n={self.params.n}")

        paths[:, 0] = X # Store initial state for all particles

        p_death_ext_const_scalar_or_array = None
        if isinstance(self.params.h_ext, float):
            p_death_ext_const_scalar_or_array = 1.0 - np.exp(-self.params.h_ext * self.params.dt)
        elif isinstance(self.params.h_ext, np.ndarray):
            p_death_ext_const_scalar_or_array = 1.0 - np.exp(-self.params.h_ext * self.params.dt)

        _ = self.params.get_drift(X, tspan[0]) # Ensure drift_func is initialized

        for i_t, tcur in enumerate(tspan[1:], 1):
            # Early break: if all particles have a recorded death_time
            if self.params.break_early and np.all(death_times != np.inf):
                num_valid_save_points = (i_t - 1) // save_interval + 1
                paths = paths[:, :num_valid_save_points]
                tspan_save_times = tspan_save_times[:num_valid_save_points]
                break

            # 1. Simulate ALL particles
            # Skorokhod step for all X
            noise_term = np.sqrt(2 * self.params.epsilon) * (X > 0) # epsilon is size n or scalar
            Y_sk = noise_term * np.sqrt(self.params.dt) * np.random.standard_normal(X.shape)
            U_sk = np.random.random(X.shape)
            Y_sq_sk = Y_sk**2
            log_U_term_sk = -2 * self.params.dt * np.log(np.maximum(U_sk, 1e-100))
            inside_sqrt_sk = Y_sq_sk + log_U_term_sk
            M_numerator_sk = Y_sk + np.sqrt(np.maximum(inside_sqrt_sk, 0))
            M_sk = M_numerator_sk / 2.0

            # Drift calculation for all X
            drift_params = {
                 'X': X, 'tcur': tcur,
                 'eta': self.params.eta, 
                 'beta': self.params.beta, 
                 'kappa': self.params.kappa
            }
            if self.params.extra_params:
                 drift_params.update(self.params.extra_params) # Assumes extra_params are scalar or size n
            
            drift = self.params.drift_func(**drift_params)
            delta_X = self.params.dt * drift
            
            X_next = np.maximum(M_sk - Y_sk, X + delta_X - Y_sk)
            X = X_next # Update all X

            # 2. Save state if it's a save time
            if i_t % save_interval == 0:
                save_idx = i_t // save_interval
                if save_idx < paths.shape[1]:
                     paths[:, save_idx] = X

            # 3. Check for FIRST crossings & extrinsic deaths for those not yet crossed
            not_yet_crossed_mask = (death_times == np.inf)
            agents_to_check_indices = np.where(not_yet_crossed_mask)[0]

            if agents_to_check_indices.size > 0:
                X_to_check = X[agents_to_check_indices]
                
                # Prepare Xc for these agents
                Xc_to_check = self.params.Xc[agents_to_check_indices] # self.params.Xc is already array of size n

                # Calculate extrinsic death probability for this step for agents_to_check
                p_death_ext_for_check = np.zeros(agents_to_check_indices.size, dtype=float)

                if p_death_ext_const_scalar_or_array is not None: # Covers float and agent-specific array cases
                    if isinstance(p_death_ext_const_scalar_or_array, np.ndarray):
                        p_death_ext_for_check = p_death_ext_const_scalar_or_array[agents_to_check_indices]
                    else: # it's a scalar from a float h_ext
                        p_death_ext_for_check.fill(p_death_ext_const_scalar_or_array)
                elif callable(self.params.h_ext): # single callable for all agents
                    current_h_ext_val = self.params.h_ext(tcur)
                    if not isinstance(current_h_ext_val, (float, int)):
                         raise TypeError(f"h_ext function must return a number, but got {type(current_h_ext_val)} for t={tcur}")
                    p_death_ext_value = 1.0 - np.exp(-current_h_ext_val * self.params.dt)
                    p_death_ext_for_check.fill(p_death_ext_value)
                elif isinstance(self.params.h_ext, list): # List of callables, one for each agent
                    h_ext_funcs_to_check = [self.params.h_ext[i] for i in agents_to_check_indices]
                    h_vals = np.array([func(tcur) for func in h_ext_funcs_to_check])
                    p_death_ext_for_check = 1.0 - np.exp(-h_vals * self.params.dt)

                died_local_mask, extrinsic_local_mask = self._find_first_crossings_numba(
                    X_to_check, Xc_to_check, p_death_ext_for_check
                )

                if X_to_check.size > 0: # Ensure masks correspond to actual agents checked
                    crossed_this_step_global_indices = agents_to_check_indices[died_local_mask]
                    
                    # Record death times ONLY for the first time they cross
                    death_times[crossed_this_step_global_indices] = tcur
                    
                    # Record if the first passage was extrinsic
                    died_extrinsically_this_step_global_indices = agents_to_check_indices[died_local_mask & extrinsic_local_mask]
                    self.extrinsic_deaths[died_extrinsically_this_step_global_indices] = 1
            
        # Final alive_mask indicates who never had a death_time recorded
        final_alive_mask = (death_times == np.inf)
        
        # Censor remaining "alive" agents at tmax if their death_time is still inf
        # (This is implicitly handled as death_times initialised to inf will remain inf if no crossing)
        # For consistency with KaplanMeierFitter, if simulation ends and death_time is inf,
        # it's a censored observation at tmax. The death_times array already reflects this if it remains inf.
        # The event_observed for KMF will be `~final_alive_mask`.
        # If KMF needs explicit tmax for censored, that's usually handled by its fit method.
        # The current `kmf.fit(self.death_times, event_observed=event_observed)` uses np.inf correctly.

        # Ensure paths and tspan_save_times are correctly sized if loop didn't break early
        # but tmax was not a multiple of save_times.
        # The initial allocation of paths is for max possible saves.
        # If loop finishes, tspan_save_times is already the correct length.
        # Paths might have trailing zeros if tmax not on save interval, but that's fine.
        # Or, we can truncate paths to the number of actual save points made.
        last_actual_save_idx = (i_t) // save_interval # i_t is the last tspan index processed
        # paths should be sliced up to this index + 1 columns.
        if not (self.params.break_early and np.all(death_times != np.inf)): # if not broken early
            num_total_save_points = last_actual_save_idx + 1
            # Ensure it doesn't exceed original allocation if tmax is small
            num_total_save_points = min(num_total_save_points, paths.shape[1])
            paths = paths[:, :num_total_save_points]
            tspan_save_times = tspan_save_times[:num_total_save_points]


        return death_times, paths, tspan_save_times, final_alive_mask, self.extrinsic_deaths

### post-simulation processing ###

    def _post_simulation_calculations(self):
        """
        Calculate statistical measures after simulation.
        
        Computes survival curves, hazard rates, and various
        statistical measures of the simulation results.
        """
        self.survival, self.tspan_survival, self.kmf = self._create_survival()
        self.naf = self._create_naf()
        self.hazard, self.tspan_hazard = self._calc_hazard()
        self.median_t = self.kmf.median_survival_time_
        self.mean_X = self._calc_mean_X()
        self.mean_X_analytical = self._calc_mean_X_analytical()
        self.cv_X = self._calc_cv_X()
        self.std_X = self._calc_std_X()
        self.steepness = self.calc_steepness()

    def _create_survival(self):
        """
        Create Kaplan-Meier survival curve from simulation results.

        Returns:
            tuple: (survival_function, timeline, kmf_object)
        """
        # event_observed is True for individuals who died (intrinsic or extrinsic)
        # and False for individuals who survived until tmax (censored).
        event_observed = ~self.alive_mask

        # Create a copy of death_times to modify for censored individuals
        # Individuals who survived until tmax have death_times == np.inf.
        # For Kaplan-Meier fitting, censored times should be the time of censoring (tmax).
        censored_death_times = np.copy(self.death_times)

        # Replace np.inf with tmax for censored individuals (where alive_mask is True)
        # Note: self.alive_mask is True where death_times was np.inf after simulation.
        # This corresponds to event_observed being False.
        censored_death_times[self.alive_mask] = self.params.tmax

        kmf = KaplanMeierFitter()

        # Fit the KMF using the modified death_times (with tmax for censored)
        # and the event_observed mask.
        kmf.fit(censored_death_times, event_observed=event_observed)

        return kmf.survival_function_, kmf.timeline, kmf

    def _create_naf(self, timeline=None):
        """
        Create Nelson-Aalen cumulative hazard estimator.

        Returns:
            NelsonAalenFitter: Fitted object
        """
        event_observed = ~self.alive_mask
        naf = NelsonAalenFitter()
        
        # Create a copy of death_times to modify for censored individuals
        # Individuals who survived until tmax have death_times == np.inf.
        # For Nelson-Aalen fitting, censored times should be the time of censoring (tmax).
        censored_death_times = np.copy(self.death_times)
        
        # Replace np.inf with tmax for censored individuals (where alive_mask is True)
        # Note: self.alive_mask is True where death_times was np.inf after simulation.
        # This corresponds to event_observed being False.
        censored_death_times[self.alive_mask] = self.params.tmax
        
        if timeline is None:
            naf.fit(censored_death_times, event_observed=event_observed)
        else:
            naf.fit(censored_death_times, event_observed=event_observed, timeline=timeline)
        return naf

    def _calc_hazard(self, timeline=None):
        """
        Calculate smoothed hazard function.

        Returns:
            tuple: (hazard_values, hazard_times)
        """
        if timeline is None:
            sm_haz_table = self.naf.smoothed_hazard_(bandwidth=3)
            tspan_hazard = sm_haz_table.index.values
            hazard = sm_haz_table.values
        else:
            temp_naf = self._create_naf(timeline)
            sm_haz_table = temp_naf.smoothed_hazard_(bandwidth=3)
            tspan_hazard = sm_haz_table.index.values
            hazard = sm_haz_table.values
        return hazard, tspan_hazard

    def _calc_mean_X(self):
        """
        Calculate mean state over time.

        Returns:
            np.ndarray: Mean state values
        """
        means = np.mean(self.paths, axis=0)
        means[-1] = np.inf
        return means
    
    def _calc_mean_X_analytical(self,epsilon=False):
        """
        Calculate mean state over time analytically from SR model.

        Returns:
            np.ndarray: Mean state values
        """
        if epsilon:
            return (self.params.kappa[0]*self.params.eta[0]*self.tspan+self.params.epsilon[0]) / (self.params.beta[0] - self.params.eta[0]*self.tspan)
        else:
            return self.params.kappa[0]*self.params.eta[0]*self.tspan / (self.params.beta[0] - self.params.eta[0]*self.tspan)

    def _calc_std_X(self):
        """
        Calculate standard deviation of state over time.

        Returns:
            np.ndarray: Standard deviation values
        """
        return np.std(self.paths, axis=0)

    def _calc_cv_X(self):
        """
        Calculate coefficient of variation of state over time.

        Returns:
            np.ndarray: Coefficient of variation values
        """
        return self._calc_std_X() / self._calc_mean_X()

    def _create_survival_from_t(self, t):
        """
        Create Kaplan-Meier survival curve for individuals alive at time t.

        Args:
            t (float): The time from which to calculate survival.

        Returns:
            KaplanMeierFitter or None: Fitted KMF object, or None if no individuals alive at t.
        """
        # Identify individuals alive at time t and get their death times
        alive_mask = self.death_times >= t
        if not np.any(alive_mask):
            return None

        death_times_from_t = self.death_times[alive_mask]
        
        # Replace np.inf with tmax for censored individuals
        # For conditional survival from time t, censored times should be (tmax - t)
        censored_death_times_from_t = np.copy(death_times_from_t)
        inf_mask = (death_times_from_t == np.inf)
        censored_death_times_from_t[inf_mask] = self.params.tmax
        
        event_times = np.maximum(0, censored_death_times_from_t - t)
        event_observed = (~inf_mask).astype(int)

        # Create and fit Kaplan-Meier Fitter
        kmf = KaplanMeierFitter()
        kmf.fit(event_times, event_observed=event_observed)
        return kmf

    def find_time_at_survival(self, S, from_t=None, relative = True):
        """
        Find time at which survival probability equals S.

        Args:
            S (float): Survival probability (0-1)
            from_t (float, optional): Start time for conditional survival

        Returns:
            float: Time at survival probability S, or None if not reached
        """
        if from_t is not None:
            kmf = self._create_survival_from_t(from_t)
            if kmf is None:
                return None
            survival_func = kmf.survival_function_.iloc[:, 0]
            timeline = kmf.timeline
        else:
            survival_func = self.survival.iloc[:, 0]
            timeline = self.survival.index
        if np.any(survival_func.values <= S):
            time_at_S = np.interp(S, survival_func.values[::-1], timeline[::-1])
            if not relative and from_t is not None:
                return from_t + time_at_S
            return time_at_S
        return None

    def calc_steepness(self, method = 'IQR',from_t=None, relative = True):
        """
        Calculate steepness of survival curve.
        
        Steepness is defined as median survival time divided by
        the time difference between 75% and 25% survival.

        Args:
            from_t (float, optional): Start time for conditional survival

        Returns:
            float: Steepness value, or None if insufficient mortality
        """
        if from_t is None:
            from_t = 0
        if method == 'IQR':
            t_25 = self.find_time_at_survival(0.25, from_t, relative)
            t_50 = self.find_time_at_survival(0.5, from_t, relative)
            t_75 = self.find_time_at_survival(0.75, from_t, relative)
            
            if all(t is not None for t in [t_25, t_50, t_75]) and t_75 != t_25:
                return -t_50 / (t_75 - t_25)
        elif method == 'CV':
            if from_t is None:
                from_t = 0
            filtered_death_times = self.death_times[(self.death_times >= from_t) & (self.death_times != np.inf)]
            if len(filtered_death_times) > 0:
                if relative:
                    filtered_death_times = filtered_death_times - from_t
                mean_time = np.mean(filtered_death_times)
                std_time = np.std(filtered_death_times)
                if mean_time > 0:
                    cv = std_time / mean_time
                    if cv > 0:
                        return 1 / cv