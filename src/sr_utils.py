"""
SR model utilities: parameters, distributions, and simulation factory.

Provides helpers to load baseline parameters, construct twin-style
distributions (MZ/DZ/uncorrelated), create `SR_sim` instances, and
compose plotting color/marker schemes used across analyses.
"""

### Description: Utility functions for the SR model, including loading baseline parameters, creating parameter dictionaries and running SR
import pandas as pd
import numpy as np
from src.simulation import SR_sim, SimulationParams

# Constants
eta_karin = 0.00135 * 365
beta_karin = 0.15 * 365
kappa_karin = 0.5
eps_karin = 0.142 * 365
Xc_karin = 17

karin_params = {
    'eta': np.array([eta_karin]),
    'beta': np.array([beta_karin]),
    'kappa': np.array([kappa_karin]),
    'epsilon': np.array([eps_karin]),
    'Xc': np.array([Xc_karin])
}
# param colors for plotting
param_colors = {
    'eta': 'blue',
    'beta': 'green',
    'epsilon': 'orange',
    'Xc': 'purple',
    'kappa' : 'grey'
}
# param descriptions
param_descriptions = {
'eta' : 'Production',
'beta' : 'Removal',
'epsilon' : 'Noise',
'Xc' : 'Threshold',
'kappa' : 'Sensitivity'
}
# param_names for plotting
param_names = {
    'eta' : r'$\eta$',
    'beta' : r'$\beta$',
    'epsilon' : r'$\epsilon$',
    'Xc' : r'$X_c$',
    'kappa' : r'$\kappa$'
}

twin_line_styles = {
    'MZ': '--',
    'DZ' : ':',
    'None' : '-'
}

twin_alphas = {
    'MZ': 1,
    'DZ' : 0.3,
    'None' : 1
}

def load_SR_params(species_name):
    """
    Load SR parameters for a specific species from CSV file.
    
    Args:
        species_name (str): Name of the species to load parameters for

    Returns:
        dict: Dictionary containing SR parameters for the species

    Raises:
        ValueError: If species is not found in the database
    """
    import os
    project_root = os.path.dirname(os.path.dirname(__file__))
    df = pd.read_csv(os.path.join(project_root, 'saved_data', 'SR_params.csv'))
    species_data = df[df['species'] == species_name]
    
    if species_data.empty:
        raise ValueError(f"Species '{species_name}' not found in the data.")
    
    species_params = species_data.iloc[0].to_dict()
    relevant_params = {key: value for key, value in species_params.items() 
                       if key not in ['species', 'source', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10']}
    return relevant_params

def load_baseline_human_params_dict(params_dict=karin_params):
    return params_dict.copy()

# create param distribution dictionary for a given parameter, with a given standard deviation, and a given distribution type. specify MZ, DZ twins, or just distribution
def create_param_distribution_dict(params, std, n=40000, dist_type='gaussian', params_dict=None, family='MZ' , corr = None):
    """
    Create parameter distributions for twin studies.
    
    Args:
        params (str/list): Parameter(s) to create distributions for ('eta', 'beta', etc.)
        std (float): Standard deviation for the distribution
        n (int): Number of samples (will be divided by 2 for twins)
        dist_type (str): Type of distribution ('gaussian', 'lognormal', 'lognormal_flipped')
        params_dict (dict): Base parameter dictionary (defaults to karin_params)
        family (str): Type of twins ('MZ', 'DZ', or 'None')

    Returns:
        dict: Dictionary with distributed parameters
    """
    if params_dict is None:
        params_dict = karin_params

    new_params_dict = params_dict.copy()
    n_pairs = int(n/2)

    def generate_base_distribution(n_samples):
        if dist_type == 'gaussian':
            return np.random.normal(1, std, n_samples)
        elif dist_type == 'lognormal':
            return np.random.lognormal(0, std, n_samples)
        elif dist_type == 'lognormal_flipped':
            return 2 - np.random.lognormal(0, std, n_samples)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    def ensure_positive(dist):
        while np.any(dist < 0):
            neg_mask = dist < 0
            dist[neg_mask] = generate_base_distribution(np.sum(neg_mask))
        return dist

    if family == 'MZ':
        # Generate one set of values and repeat for both twins
        dist = ensure_positive(generate_base_distribution(n_pairs))
        dist = np.repeat(dist, 2)
    elif family == 'DZ':
        # Generate correlated values using bivariate normal with rho = 0.5, works for gaussian only
        Z1 = np.random.randn(n_pairs)
        Z2 = np.random.randn(n_pairs)
        dist1 = 1 + std * Z1
        dist2 = 1 + std * (0.5 * Z1 + np.sqrt(1 - 0.5**2) * Z2)
        dist = ensure_positive(np.ravel(np.column_stack((dist1, dist2))))
        
    elif family == 'corr':
        # Generate correlated values using bivariate normal with rho = corr, works for gaussian only
        Z1 = np.random.randn(n_pairs)
        Z2 = np.random.randn(n_pairs)
        dist1 = 1 + std * Z1
        dist2 = 1 + std * (corr * Z1 + np.sqrt(1 - corr**2) * Z2)
        dist = ensure_positive(np.ravel(np.column_stack((dist1, dist2))))
        
    elif family == 'None':
        # Generate independent values for both twins
        dist1 = ensure_positive(generate_base_distribution(n_pairs))
        dist2 = ensure_positive(generate_base_distribution(n_pairs))
        dist = np.ravel(np.column_stack((dist1, dist2)))
    else:
        raise ValueError("family must be one of 'MZ', 'DZ', or 'None'")

    if isinstance(params, str):
        params = [params]
    
    for param in params:
        # Get the base value - needs careful handling for h_ext which might not be in karin_params by default
        if param == 'h_ext':
            # Default h_ext to 0 if not present, assuming no external hazard is the base
            base_value = params_dict.get(param, 0.0)
            # Ensure base_value is scalar for distribution generation
            if hasattr(base_value, '__iter__') and not isinstance(base_value, str):
                 base_value = base_value[0] # Use first element if it's an array
            elif not np.isscalar(base_value):
                 raise ValueError(f"Base value for h_ext must be a scalar or array, got {type(base_value)}")
        else:
             # Original logic for other params
             base_value = params_dict[param][0] if hasattr(params_dict[param], '__iter__') else params_dict[param]

        # Apply distribution: multiply base value by the generated distribution factor
        # Ensure non-negativity for hazard rates
        if param == 'h_ext':
             new_params_dict[param] = ensure_positive(dist * base_value) 
        else:
             new_params_dict[param] = dist * base_value

    # --- Final Expansion and Validation --- 
    target_n = len(dist) # n for the simulation
    # List all parameters expected by SimulationParams + h_ext
    all_expected_params = ['eta', 'beta', 'kappa', 'epsilon', 'Xc', 'h_ext']
    
    for p in all_expected_params:
        if p in new_params_dict:
            current_val = new_params_dict[p]
            param_was_distributed = (p in params) # Check if this param was generated

            if isinstance(current_val, (np.ndarray, list)):
                current_len = len(current_val)
                if current_len == target_n:
                    pass # Length is already correct
                elif current_len == 1 and not param_was_distributed:
                    # Expand length-1 array if it wasn't the distributed param
                    # print(f"Info: Expanding parameter '{p}' from length 1 to {target_n}.") # Optional info
                    new_params_dict[p] = np.full(target_n, current_val[0])
                else:
                    # Length mismatch is an error, either internal (if distributed) or input (if not)
                    error_type = "Internal" if param_was_distributed else "Input"
                    raise ValueError(f"{error_type} error: Parameter '{p}' has unexpected length {current_len}, expected {target_n}")
                    
            elif np.isscalar(current_val):
                # Expand scalar to full array
                new_params_dict[p] = np.full(target_n, current_val)
            else:
                # Handle unexpected types
                raise TypeError(f"Parameter '{p}' has unexpected type: {type(current_val)}")
                
        elif p in params_dict:
            # If parameter wasn't distributed and wasn't even copied initially (e.g., h_ext not in base dict), expand from base.
            base_val = params_dict[p]
            if np.isscalar(base_val):
                 new_params_dict[p] = np.full(target_n, base_val)
            elif isinstance(base_val, (np.ndarray, list)) and len(base_val) == 1:
                 new_params_dict[p] = np.full(target_n, base_val[0])
            else:
                 raise ValueError(f"Base parameter '{p}' (not distributed) cannot be expanded to length {target_n}. Value: {base_val}")
        
        elif p != 'h_ext': # If a core param is missing entirely
             raise ValueError(f"Core parameter '{p}' is missing from parameter dictionaries.")

    return new_params_dict

def create_sr_simulation(species='human', n=1e5, save_times=100, params_dict=None, param_updates=None, 
                        drift_expr=None, drift_mode='replace', extra_params=None, h_ext=None, **kwargs):
    """
    Create and configure an SR simulation.

    Args:
        species (str): Species to simulate ('human' or other species in database)
        n (int): Number of simulations to run
        save_times (int): Interval for saving simulation states
        params_dict (dict): Optional custom parameter dictionary providing base values 
                          for eta, beta, kappa, epsilon, Xc.
        param_updates (dict): Updates to specific parameters (eta, beta, kappa, epsilon, Xc).
                              Values can be scalar or array of length n.
        drift_expr (str): Custom drift expression
        drift_mode (str): How to apply custom drift ('replace' or 'add')
        extra_params (dict): Additional parameters for custom drift
        h_ext (float/callable/array/None): External hazard rate override. 
               Takes precedence over h_ext possibly present in params_dict or param_updates.
               See SimulationParams docstring for details.
        **kwargs: Additional simulation parameters passed directly to SimulationParams 
                  (e.g., tmin, tmax, x0, dt, parallel, break_early, units).

    Returns:
        SR_sim: Configured simulation object
    """
    n = int(n)
    
    # Set defaults based on species
    if species == 'human':
        base_params = load_baseline_human_params_dict()
        defaults = {'tmin': 0, 'tmax': 140, 'x0': 1e-6, 'dt': 0.025, 'save_times': save_times, 
                   'units': 'years', 'parallel': True, 'break_early': True}
    else:
        base_params = load_SR_params(species)
        defaults = {'tmin': 0, 'tmax': 1400, 'x0': 1e-6, 'dt': 1, 'save_times': save_times,
                   'units': 'days', 'parallel': True, 'break_early': True}

    # Build final parameters
    final_params = {**defaults, **kwargs, 'n': n}
    final_params.update(base_params)
    
    if params_dict:
        # Handle parameter array expansion and consistency
        sr_params = ['eta', 'beta', 'kappa', 'epsilon', 'Xc']
        array_lens = {}
        
        # Check lengths of all SR parameters
        for param in sr_params:
            if param in params_dict:
                val = params_dict[param]
                if isinstance(val, (np.ndarray, list)):
                    array_lens[param] = len(val)
                else:
                    array_lens[param] = 1  # scalar or size-1 array
        
        # Find target length
        lengths = list(array_lens.values())
        if lengths:
            if all(l == 1 for l in lengths):
                target_n = n  # All scalars/size-1, use input n
            elif any(l == n for l in lengths):
                target_n = n  # At least one is size n, use input n
            else:
                # Check for inconsistent lengths (not 1 and not n)
                inconsistent = [param for param, length in array_lens.items() 
                              if length != 1 and length != n]
                if inconsistent:
                    raise ValueError(f"Inconsistent array lengths in params_dict: {inconsistent} have lengths {[array_lens[p] for p in inconsistent]}")
                target_n = n
        
        # Expand parameters to target_n
        expanded_params = {}
        for param in sr_params:
            if param in params_dict:
                val = params_dict[param]
                if isinstance(val, (np.ndarray, list)):
                    if len(val) == 1:
                        expanded_params[param] = np.full(target_n, val[0])
                    elif len(val) == target_n:
                        expanded_params[param] = val
                    else:
                        raise ValueError(f"Parameter {param} has length {len(val)}, expected 1 or {target_n}")
                else:
                    expanded_params[param] = np.full(target_n, val)
        
        final_params.update(expanded_params)
    
    if param_updates:
        final_params.update({k: v for k, v in param_updates.items() 
                           if k in ['eta', 'beta', 'kappa', 'epsilon', 'Xc']})
    
    # Handle h_ext precedence
    final_params['h_ext'] = h_ext if h_ext is not None else final_params.get('h_ext')
    
    # Add drift parameters
    if drift_expr:
        final_params.update({'drift_expr': drift_expr, 'drift_mode': drift_mode, 'extra_params': extra_params})

    return SR_sim(SimulationParams(**final_params))

# Define the Gompertz hazard function here
def gompertz_hazard(t, m, a, b):
    """Calculates the Gompertz hazard rate."""
    return m + a * np.exp(b * t)