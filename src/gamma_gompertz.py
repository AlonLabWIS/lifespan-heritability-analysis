"""
Gamma-Gompertz-Makeham (GGM) model utilities.

Supports fitting to HMD data, Kaplan-Meier curves, or custom arrays;
sampling death times; plotting helpers; and optimization constrained by
twin correlations or KS distance to empirical distributions.
"""

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.integrate import cumulative_trapezoid as cumtrapz
import matplotlib.pyplot as plt
import pandas as pd
from src.HMD_lifetables import HMD
import src.twin_analysis as ta
from lifelines import NelsonAalenFitter, KaplanMeierFitter
import copy

class GammaGompertz:
    def __init__(self, params=None):
        # if params is not None then set the parameters
        if params is not None:
            self.set_params(params)
                   
    def set_params(self, params):
        self.a = params['a']
        self.b = params['b']
        self.c = params['c']
        self.m = params.get('m', 0)  # Default m to 0 if not provided
        
    def fit_params(self, country=None, year=None, gender=None, data_type='cohort', haz_type='mx', 
                  filter_from=20, filter_to=110, print_out=True, kmf=None, 
                  initial_guess=None, maxiter=1000, method='L-BFGS-B',
                  time_array=None, log10_hazard_array=None):
        """
        Fit parameters to HMD hazard data, KaplanMeierFitter object, or custom hazard arrays.
        
        Args:
            country, year, gender, data_type, haz_type: HMD data parameters (if kmf and arrays are None)
            filter_from, filter_to: Age range for HMD fitting
            print_out: Whether to print fitted parameters
            kmf: Optional KaplanMeierFitter object to fit survival function to
            initial_guess: Initial parameter guess [a, b, c, m]. If None, uses default.
            maxiter: Maximum iterations for optimization
            method: Optimization method ('L-BFGS-B', 'Nelder-Mead', etc.)
            time_array: Optional array of time/age values
            log10_hazard_array: Optional array of log10(hazard) values
        """
        if time_array is not None and log10_hazard_array is not None:
            # Fit to custom hazard arrays
            self._fit_params_to_arrays(time_array, log10_hazard_array, initial_guess, print_out)
        elif kmf is not None:
            # Fit to KaplanMeierFitter object
            self._fit_params_to_kmf(kmf, initial_guess, maxiter, method, print_out)
        else:
            # Original HMD fitting functionality
            if country is None or year is None or gender is None:
                raise ValueError("country, year, and gender must be provided when kmf and arrays are None")
            
            self._fit_params_to_hmd(country, year, gender, data_type, haz_type, 
                                  filter_from, filter_to, print_out)
    
    def _fit_params_to_kmf(self, kmf, initial_guess=None, maxiter=1000, method='L-BFGS-B', print_out=True):
        """Fit parameters to a KaplanMeierFitter object using survival function"""
        from scipy.optimize import minimize
        
        # Get survival data from KMF
        times = kmf.timeline
        survival_kmf = kmf.survival_function_.values.flatten()
        
        # Remove any NaN values
        valid_mask = ~np.isnan(survival_kmf)
        times = times[valid_mask]
        survival_kmf = survival_kmf[valid_mask]
        
        # Define objective function (mean squared error)
        def objective_function(params):
            a, b, c, m = params
            try:
                # Calculate model survival function
                _, survival_model = self.survival_function(times, a, b, c, m)
                # Calculate MSE
                mse = np.mean((survival_model - survival_kmf) ** 2)
                return mse
            except:
                # Return large value if calculation fails
                return 1e10
        
        # Set initial guess
        if initial_guess is None:
            initial_guess = [5e-5, 0.1, 9, 0.005]  # [a, b, c, m]
        
        # Set bounds for positive parameters
        bounds = [(1e-8, None), (1e-8, None), (1e-8, None), (0, None)]  # a, b, c, m
        
        # Optimize
        result = minimize(
            objective_function,
            x0=initial_guess,
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter}
        )
        
        # Set fitted parameters
        self.a, self.b, self.c, self.m = result.x
        
        # Store KMF data for refitting
        self.kmf_data = {
            'kmf': kmf,
            'initial_guess': initial_guess,
            'maxiter': maxiter,
            'method': method
        }
        
        if print_out:
            print("Fitted to KMF data:")
            self.print_params()
            print(f"Final MSE: {result.fun:.6f}")
            print(f"Optimization success: {result.success}")
        
        return result
    
    def _fit_params_to_arrays(self, time_array, log10_hazard_array, initial_guess=None, print_out=True):
        """Fit parameters to custom time and log10(hazard) arrays"""
        # Convert arrays to numpy and ensure they're finite
        time_array = np.asarray(time_array)
        log10_hazard_array = np.asarray(log10_hazard_array)
        
        # Filter out NaN/inf values and ensure positive hazards
        filter_mask = (np.isfinite(time_array) & np.isfinite(log10_hazard_array) & 
                      (log10_hazard_array > -np.inf))
        time_filtered = time_array[filter_mask]
        log10_hazard_filtered = log10_hazard_array[filter_mask]
        
        # Convert log10 to natural log for fitting (model uses natural log)
        ln_hazard_filtered = log10_hazard_filtered * np.log(10)
        
        # Set initial guess
        if initial_guess is None:
            initial_guess = [5e-5, 0.1, 9, 0.005]  # [a, b, c, m]
        
        # Fit parameters using curve_fit (same as HMD method)
        popt, pcov = curve_fit(
            self.log_hazard_function, 
            time_filtered, 
            ln_hazard_filtered, 
            p0=initial_guess
        )
        
        # Extract fitted parameters
        self.a, self.b, self.c, self.m = popt
        
        # Store array data for refitting
        self.array_data = {
            'time_array': time_array,
            'log10_hazard_array': log10_hazard_array,
            'initial_guess': initial_guess
        }
        
        if print_out:
            print("Fitted to custom hazard arrays:")
            self.print_params()
            
        return popt, pcov
    
    def _fit_params_to_hmd(self, country, year, gender, data_type, haz_type, 
                          filter_from, filter_to, print_out):
        """Original HMD fitting functionality"""
        ages_data, hazard_data = self.load_HMD_hazard_data(
            country=country, year=year, gender=gender, 
            data_type=data_type, haz_type=haz_type
        )
        
        # Filter data and remove NaN values
        filter_mask = (ages_data >= filter_from) & (ages_data <= filter_to) & \
                     (~np.isnan(ages_data)) & (~np.isnan(hazard_data)) & \
                     (hazard_data > 0)
        ages_data_filtered = ages_data[filter_mask]
        hazard_data_filtered = hazard_data[filter_mask]
        
        # Fit all parameters together
        initial_guess = [5e-5, 0.1, 9, 0.005]  # [a, b, c, m]
        popt, pcov = curve_fit(
            self.log_hazard_function, 
            ages_data_filtered, 
            np.log(hazard_data_filtered), 
            p0=initial_guess
        )
        
        # Extract fitted parameters
        self.a, self.b, self.c, self.m = popt
        self.hmd_data = {
            'country': country, 'year': year, 'gender': gender,
            'data_type': data_type, 'haz_type': haz_type
        }
        
        if print_out:
            self.print_params()
        
    def refit_params(self, print_out=False):
        """Refit parameters to the previously used data (HMD, KMF, or arrays)"""
        if hasattr(self, 'hmd_data'):
            # Refit to HMD data
            self.fit_params(
                country=self.hmd_data['country'],
                year=self.hmd_data['year'], 
                gender=self.hmd_data['gender'],
                data_type=self.hmd_data['data_type'],
                haz_type=self.hmd_data['haz_type'],
                filter_from=20, 
                filter_to=110,
                print_out=print_out
            )
        elif hasattr(self, 'kmf_data'):
            # Refit to KMF data
            self.fit_params(
                kmf=self.kmf_data['kmf'],
                initial_guess=self.kmf_data.get('initial_guess'),
                maxiter=self.kmf_data.get('maxiter', 1000),
                method=self.kmf_data.get('method', 'L-BFGS-B'),
                print_out=print_out
            )
        elif hasattr(self, 'array_data'):
            # Refit to array data
            self.fit_params(
                time_array=self.array_data['time_array'],
                log10_hazard_array=self.array_data['log10_hazard_array'],
                initial_guess=self.array_data.get('initial_guess'),
                print_out=print_out
            )
        else:
            raise ValueError("No previous fitting data found. Use fit_params() first.")
   
    def print_params(self):
        print('a: ', self.a)
        print('b: ', self.b)
        print('c: ', self.c)
        print('m: ', self.m)
        
    def hazard_function(self, t, a, b, c, m):
        """Gamma-Gompertz-Makeham hazard function"""
        exp_bt = np.exp(b * t)
        exp_c = np.exp(c)
        return m + a * exp_bt * (exp_c / (exp_c + exp_bt - 1))
    
    def log_hazard_function(self, t, a, b, c, m):
        """Log of the Gamma-Gompertz-Makeham hazard function"""
        return np.log(self.hazard_function(t, a, b, c, m))
    
    def calculate_hazard(self, t):
        """Calculate hazard for given times using current parameters"""
        return t, self.hazard_function(t, self.a, self.b, self.c, self.m)
    
    def sample_death_times(self, n, dt=0.5, min_age = None, params=None):
        """
        Sample n death times from the hazard function using numerical integration.
        
        Args:
            n (int): Number of samples to draw
            dt (float): Time step for numerical integration
            params (dict): Optional dictionary of parameters to use
            
        Returns:
            numpy array: Array of n death times
        """
        # Use provided params if given, otherwise use instance params
        if params is None:
            a, b, c, m = self.a, self.b, self.c, self.m
        else:
            a = params['a']
            b = params['b']
            c = params['c']
            m = params['m']

        # Create time grid for numerical integration
        t = np.arange(0, 200, dt)
        
        # Calculate hazard at each time point
        hazard = self.hazard_function(t, a, b, c, m)
        
        # Calculate cumulative hazard
        cumulative_hazard = np.cumsum(hazard * dt)
        
        # Generate uniform random numbers and convert to survival thresholds
        u = np.random.random(n)
        survival_thresholds = -np.log(u)
        
        # Interpolate to get death times
        ch_full = np.concatenate(([0.0], cumulative_hazard))
        t_full = np.concatenate(([t[0]], t))
        death_times = np.interp(survival_thresholds, ch_full, t_full)
        
        if min_age is not None:
            death_times = death_times[death_times >= min_age]
            return death_times
        else:
            return death_times

    def sample_death_times_with_random_param(self, n, param_name='a', dt=0.5, std=0.15, coupled_ab=False):
        """
        Sample death times with a randomly drawn parameter value for each individual.

        Args:
            n (int): Number of samples to draw
            param_name (str): Which parameter to randomize. One of {'a','b','c','m'}.
                            Default 'a'.
            dt (float): Time step for numerical integration. Default 0.5.
            std (float): Standard deviation for parameter sampling. Default 0.15.
            coupled_ab (bool): If True and param_name='b', will draw a random multiplier r
                             and set b = b*r and a = a^r. Default False.

        Returns:
            tuple: (death_times, param_values) where:
                  death_times is numpy array of n death times
                  param_values is numpy array of n parameter values used
        """
        if param_name not in ['a', 'b', 'c', 'm']:
            raise ValueError("param_name must be one of 'a','b','c','m'.")

        # Get base parameter value
        pop_mean = getattr(self, param_name)
        
        # Draw n parameter values
        param_values = np.random.normal(loc=pop_mean, scale=std * pop_mean, size=n)
        # Ensure all values are positive by resampling negative values
        while np.any(param_values <= 0):
            neg_mask = param_values <= 0
            param_values[neg_mask] = np.random.normal(
                loc=pop_mean, 
                scale=std * pop_mean, 
                size=np.sum(neg_mask)
            )

        death_times = []
        for param_val in param_values:
            # Create parameter dictionary for this individual
            if coupled_ab and param_name == 'b':
                # Calculate multiplier r = param_val/self.b
                r = param_val / self.b
                params = {
                    'a': self.a ** r,  # a^r
                    'b': param_val,    # b*r
                    'c': self.c,
                    'm': self.m
                }
            else:
                params = {
                    'a': self.a if param_name != 'a' else param_val,
                    'b': self.b if param_name != 'b' else param_val,
                    'c': self.c if param_name != 'c' else param_val,
                    'm': self.m if param_name != 'm' else param_val
                }
            
            # Sample single death time with these parameters
            death_time = self.sample_death_times(1, dt=dt, params=params)[0]
            death_times.append(death_time)

        return np.array(death_times), param_values
    
    def create_death_table_for_twins(self, twin_type='MZ', n=1000, param_name='b', 
                                   dt=0.5, std=0.15, coupled_ab=True):
        """
        Create a DataFrame with death times for n pairs of twins (MZ or DZ).
        
        Args:
            twin_type (str): Either 'MZ' or 'DZ'
            n (int): Number of twin pairs to simulate
            param_name (str): Parameter to randomize ('a', 'b', 'c', or 'm')
            dt (float): Time step for numerical integration
            std (float): Standard deviation for parameter sampling
            coupled_ab (bool): If True and param_name='b', will draw a random multiplier r
                             and set b = b*r and a = a^r
            
        Returns:
            pd.DataFrame: DataFrame with columns ['death1','death2']
        """
        if param_name not in ['a', 'b', 'c', 'm']:
            raise ValueError("param_name must be one of 'a','b','c','m'")

        # Get base parameter value
        pop_mean = getattr(self, param_name)
        death_times_list = []

        # Generate correlated parameter values for all pairs at once
        if twin_type == 'MZ':
            # For MZ twins, generate one set and repeat
            param_values = np.random.normal(loc=pop_mean, scale=std * pop_mean, size=n)
            param_values = np.column_stack((param_values, param_values))
        elif twin_type == 'DZ':
            # For DZ twins, use correlation formula with rho = 0.5
            Z1 = np.random.randn(n)
            Z2 = np.random.randn(n)
            param_values1 = pop_mean + (std * pop_mean) * Z1
            param_values2 = pop_mean + (std * pop_mean) * (0.5 * Z1 + np.sqrt(1 - 0.5**2) * Z2)
            param_values = np.column_stack((param_values1, param_values2))
        else:
            raise ValueError("twin_type must be 'MZ' or 'DZ'")

        # Ensure all values are positive
        param_values = np.abs(param_values)

        for param_val1, param_val2 in param_values:
            # Handle coupled parameters if needed
            if coupled_ab and param_name == 'b':
                r1 = param_val1 / self.b
                r2 = param_val2 / self.b
                params1 = {
                    'a': self.a ** r1,
                    'b': param_val1,
                    'c': self.c,
                    'm': self.m
                }
                params2 = {
                    'a': self.a ** r2,
                    'b': param_val2,
                    'c': self.c,
                    'm': self.m
                }
            else:
                params1 = {
                    'a': self.a if param_name != 'a' else param_val1,
                    'b': self.b if param_name != 'b' else param_val1,
                    'c': self.c if param_name != 'c' else param_val1,
                    'm': self.m if param_name != 'm' else param_val1
                }
                params2 = {
                    'a': self.a if param_name != 'a' else param_val2,
                    'b': self.b if param_name != 'b' else param_val2,
                    'c': self.c if param_name != 'c' else param_val2,
                    'm': self.m if param_name != 'm' else param_val2
                }

            # Sample death times for both twins
            death1 = self.sample_death_times(1, dt=dt, params=params1)[0]
            death2 = self.sample_death_times(1, dt=dt, params=params2)[0]
            death_times_list.append([death1, death2])

        return pd.DataFrame(death_times_list, columns=['death1', 'death2'])
    
    def load_HMD_hazard_data(self, country, year, gender, data_type='cohort', 
                            haz_type='mx'):
        """Load hazard data from HMD"""
        country_period_data = HMD(country=country, gender=gender, data_type=data_type)
        ages, hazard = country_period_data.get_hazard(year=year, haz_type=haz_type)
        return ages, hazard

    def load_HMD_survival_data(self, country, year, gender, data_type='cohort', haz_type='mx'):
        country_period_data = HMD(country=country, gender=gender, data_type=data_type)
        ages, survival = country_period_data.calculate_survival(year=year)
        return ages, survival
    
    def get_HMD_death_distribution(self, country, year, gender, data_type='cohort', haz_type='mx'):
        # Get survival data from HMD
        ages, survival = self.load_HMD_survival_data(country, year, gender)
        
        # Calculate death distribution as negative derivative of survival
        death_distribution = -np.gradient(survival, ages)
        
        return ages, death_distribution
    
    def plot_hazard(self, ax=None, t=np.arange(0, 110.5, 0.5), **kwargs):
        """
        Plot the hazard function h(t) using the model's parameters.
        
        Args:
            ax: Optional matplotlib axes object. If None, creates new figure.
            t: Time points to evaluate hazard at (default: 0 to 110 in steps of 0.5)
            **kwargs: Additional arguments passed to plot function
        
        Returns:
            ax: The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate hazard using current parameters
        ages, hazard = self.calculate_hazard(t)
        
        # Plot with log y-scale
        ax.plot(ages, hazard, **kwargs)
        ax.set_xlabel('Age')
        ax.set_ylabel('Hazard [1/year]')
        ax.set_yscale('log')
        ax.legend()
        
        return ax

    def plot_hazard_with_array_comparison(self, time_array, log10_hazard_array, 
                                          t=None, ax=None, plot_model=True, model_color='r', **kwargs):
        """
        Plot model hazard against custom hazard array data.

        Parameters
        ----------
        time_array : array-like
            Array of ages/times for the data points.
        log10_hazard_array : array-like
            Array of log10 hazard values corresponding to time_array.
        t : array-like, optional
            Ages at which to evaluate the model hazard. If None, uses a linspace over the data range.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure and axes.
        plot_model : bool, optional
            Whether to plot the model hazard curve (default: True).
        model_color : str or color, optional
            Color to use for the model hazard curve (default: 'r').
        **kwargs : dict
            Additional keyword arguments passed to ax.scatter for the data points.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert log10 hazard back to linear scale for plotting
        hazard_array = 10**log10_hazard_array
        
        # Plot data
        ax.scatter(time_array, hazard_array, alpha=0.6, **kwargs)
        
        # Plot fitted model if requested
        if plot_model:
            if t is None:
                t = np.linspace(np.min(time_array), np.max(time_array), 200)
            _, h_model = self.calculate_hazard(t)
            ax.plot(t, h_model, color=model_color, linewidth=2, label=None)
        
        ax.set_xlabel('Age [years]')
        ax.set_ylabel('Hazard [1/year]')
        ax.set_yscale('log')
        ax.legend()
        
        return ax

    def plot_hazard_with_data_comparison(self, country, year, gender, 
                                       data_type='cohort', haz_type='mx',
                                       t=np.linspace(0, 120, 200), ax=None):
        """Plot model hazard against data"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        ages, hazard = self.calculate_hazard(t)
        ages_data, hazard_data = self.load_HMD_hazard_data(
            country=country, year=year, gender=gender,
            data_type=data_type, haz_type=haz_type
        )
        
        ax.scatter(ages_data, hazard_data, label='Data')
        ax.plot(ages, hazard, 'r-', label='Gamma-Gompertz-Makeham')
        ax.legend()
        ax.set_xlabel('Age')
        ax.set_yscale('log')
        ax.set_ylabel('Hazard [1/year]')
        
        return ax

    def plot_hazard_from_death_times(self, death_times, ax=None, **kwargs):
        """Plot hazard using Nelson-Aalen estimator from death times"""
        if ax is None:
            ax = plt.gca()
            
        naf = NelsonAalenFitter()
        naf.fit(death_times, event_observed=np.ones(len(death_times)),
                timeline=np.arange(0, 110, 0.5))
        naf.plot_hazard(ax=ax, bandwidth=3, **kwargs)
        ax.set_yscale('log')
        return ax

    def plot_survival_from_death_times(self, death_times, ax=None, **kwargs):
        """Plot survival function using Kaplan-Meier estimator from death times"""
        if ax is None:
            ax = plt.gca()
            
        kmf = KaplanMeierFitter()
        kmf.fit(death_times)
        kmf.plot_survival_function(ax=ax, **kwargs)
        return ax

    def survival_function(self, t, a=None, b=None, c=None, m=None):
        """
        Compute survival function S(t) = exp(-integral_0^t h(s) ds)
        where h(s) is the hazard function.
        
        Args:
            t: Time points
            a, b, c, m: Optional parameters. If None, uses instance parameters.
        """
        t = np.asarray(t)
        
        # Use provided parameters or fall back to instance parameters
        a = a if a is not None else self.a
        b = b if b is not None else self.b
        c = c if c is not None else self.c
        m = m if m is not None else self.m
        hazard = self.hazard_function(t, a, b, c, m)
        # Numerical integration of hazard from 0 to t for each t
        # Use cumulative trapezoidal integration
        cumulative_hazard = cumtrapz(hazard, t, initial=0)
        survival = np.exp(-cumulative_hazard)
        return t, survival

    def plot_survival(self, ax=None, t=np.arange(0, 110.5, 0.5), **kwargs):
        """
        Plot the survival function S(t) using the model's parameters.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        t, survival = self.survival_function(t)
        ax.plot(t, survival, **kwargs)
        ax.set_xlabel('Age')
        ax.set_ylabel('Survival Probability')
        ax.set_ylim(0, 1.05)
        ax.legend()
        return ax

    def plot_survival_from_age_X(self, age_X, t=None, ax=None, **kwargs):
        """
        Plot the survival function S(t) normalized from age_X, so that at age_X survival = 1.
        This shows survival probability conditional on surviving to age_X.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        if t is None:
            t = np.arange(age_X, 110.5, 0.5)
        else:
            t = np.asarray(t)
            # Ensure t starts from at least age_X
            t = t[t >= age_X]
        
        # Calculate full survival function
        t_full, survival_full = self.survival_function(t)
        
        # Find survival at age_X for normalization
        # Get the closest age to age_X in our survival function
        idx_X = np.argmin(np.abs(t_full - age_X))
        survival_at_X = survival_full[idx_X]
        
        # Normalize survival by survival at age_X
        survival_normalized = survival_full / survival_at_X
        
        ax.plot(t_full, survival_normalized, **kwargs)
        ax.set_xlabel('Age')
        ax.set_ylabel(f'Survival Probability from Age {age_X}')
        ax.set_ylim(0, 1.05)
        ax.legend()
        return ax

    def calculate_median_lifespan(self, from_t: float = 0, relative: bool = False):
        """Return the median lifespan (age at 50 % survival).

        By default this returns the **absolute age** at which survival equals 0.5.
        If ``relative`` is True the value returned is the number of **years *after*``from_t``**.
        This mirrors behaviour in other mortality models in ``ageing_packages``.
        """

        # Relative time (years after ``from_t``) at which S=0.5
        t_rel = self.find_time_at_survival(0.5, from_t=from_t, relative=True)
        if t_rel is None:
            return None
        return t_rel if relative else t_rel + from_t

    def find_time_at_survival(self, S: float, from_t: float = 0, *, relative: bool = True):
        """Time at which the *conditional* survival equals ``S``.

        The survival curve is conditioned on being alive at ``from_t`` (i.e. it is
        normalised so that :math:`S(\text{from\_t}) = 1`).  If ``relative`` is
        True the returned value is measured **from ``from_t``**; otherwise an
        absolute age is returned.
        """

        if not (0 <= S <= 1):
            raise ValueError("S must be between 0 and 1")

        t_max = 120  # upper bound
        t = np.arange(from_t, t_max, 0.5)
        _, surv_full = self.survival_function(t)

        # Conditional survival starting at ``from_t``
        surv_from_t = surv_full / surv_full[0]

        # Find time where conditional survival reaches S
        if np.any(surv_from_t <= S):
            t_abs = np.interp(S, surv_from_t[::-1], t[::-1])
            return t_abs - from_t if relative else t_abs
        return None

    def calc_steepness(self, method: str = 'IQR', from_t: float = 0):
        """
        Calculate steepness of survival curve.
        
        Steepness is defined as median survival time divided by
        the time difference between 75% and 25% survival.
        
        Args:
            method (str): Method for calculating steepness ('IQR' or 'CV')
            from_t (float): Starting age for conditional calculation
            
        Returns:
            float: Steepness value, or None if insufficient mortality
        """
        if method.upper() == 'IQR':
            # Use **relative** times for the IQR-based steepness measure
            t_25 = self.find_time_at_survival(0.25, from_t, relative=True)
            t_50 = self.find_time_at_survival(0.5,  from_t, relative=True)
            t_75 = self.find_time_at_survival(0.75, from_t, relative=True)

            if all(t is not None for t in (t_25, t_50, t_75)) and t_75 != t_25:
                return -t_50 / (t_75 - t_25)
        elif method.upper() == 'CV':
            # For CV method, sample death times and calculate coefficient of variation
            death_times = self.sample_death_times(10000)
            filtered_death_times = death_times[death_times >= from_t]
            if len(filtered_death_times) > 0:
                mean_time = np.mean(filtered_death_times)
                std_time = np.std(filtered_death_times)
                if mean_time > 0:
                    cv = std_time / mean_time
                    if cv > 0:
                        return 1 / cv
        return None

    def fit_params_with_std_with_ks(self, HMD_params, std=0.35, min_age=25, max_age=90, maxiter=250, callback=True, method='Nelder-Mead'):
        """
        Runs optimization to find best (a, b, c, m) that minimizes KS distance
        to the HMD-based normalized_distribution in ages min_age..max_age.
        """
        
        def compute_ks_statistic(death_times, ages_filtered, normalized_distribution):
            cdf_theoretical = np.cumsum(normalized_distribution)
            cdf_empirical = np.zeros_like(cdf_theoretical)
            counts, _ = np.histogram(death_times, bins=np.arange(ages_filtered[0], ages_filtered[-1] + 2))
            cdf_empirical = np.cumsum(counts) / sum(counts)
            ks_dist = np.max(np.abs(cdf_empirical - cdf_theoretical))
            return ks_dist

        def objective_function(param_array, model, ages_filtered, normalized_distribution):
            model_copy = copy.deepcopy(model)
            params_dict = {
                'a': param_array[0],
                'b': param_array[1],
                'c': param_array[2],
                'm': param_array[3]
            }
            model_copy.set_params(params_dict)
            n_samples = 10000
            death_times, _ = model_copy.sample_death_times_with_random_param(
                n=n_samples,
                param_name='b',
                dt=1, 
                std=std,
                coupled_ab=True
            )
            ks_dist = compute_ks_statistic(death_times, ages_filtered, normalized_distribution)
            return ks_dist

        ages, death_distribution = self.get_HMD_death_distribution(**HMD_params)
        age_filter = (ages >= min_age) & (ages <= max_age)
        ages_filtered = ages[age_filter]
        ddist_filtered = death_distribution[age_filter]
        normalized_distribution = ddist_filtered / np.sum(ddist_filtered)

        baseline_params = [
            self.a,
            self.b,
            self.c,
            self.m
        ]

        def local_objective(x):
            return objective_function(x, self, ages_filtered, normalized_distribution)

        iteration = [0]  # Use list to allow modification in callback
        result = minimize(local_objective, 
                        x0=baseline_params, 
                        method=method,
                        options={'maxiter': maxiter, 'disp': True},
                        callback=lambda xk: (print(f"Iteration {iteration[0]}") if callback else None, iteration.__setitem__(0, iteration[0] + 1)))

        self.set_params(
            {
                'a': result.x[0],
                'b': result.x[1],
                'c': result.x[2],
                'm': result.x[3],
            }
        )
        
        if result.success:
            best_params = result.x
            print("Optimization succeeded.")
            print("Best-fit parameters (a, b, c, m): ", best_params)
            print("Final KS distance: ", result.fun)
        else:
            print("Optimization failed with message:", result.message)

        return result

    def fit_params_with_corr_constraint(
    self,
    HMD_params,
    target_corr=0.2,
    corr_twin_type='MZ',
    param_name_corr='b',
    coupled_ab=True,
    min_age=25,
    max_age=90,
    n_samples_for_corr=50000,
    n_samples_for_distribution=50000,
    std_initial=0.3,
    maxiter=250,
    method='Nelder-Mead',
    alpha=1.0,
    callback=True,
    **fixed_params
    ):
        """
        Fit model parameters and 'std' so that the death distribution matches
        HMD data (via KS distance) AND the twin correlation matches target_corr.

        A regularized penalty approach is used in the objective function:
            cost = KS distance + alpha * (corr - target_corr)^2

        This method also enables "fixing" some parameters by passing them as
        keyword arguments (e.g., c=90). Any parameter provided will remain
        constant; only the remaining free parameters plus 'std' will be optimized.

        Args:
            HMD_params (dict): Dictionary with the keys required by
                            get_HMD_death_distribution (e.g.,
                            {'country':'Japan','year':1950,'gender':'male',
                                'data_type':'cohort','haz_type':'mx'}).
            target_corr (float): Desired correlation among twins (MZ or DZ).
            corr_twin_type (str): 'MZ' or 'DZ' - the twin type to use for correlation.
            param_name_corr (str): Parameter name to randomize for correlation
                                calculation, default is 'b'.
            coupled_ab (bool): If True and param_name_corr='b', sets b=b*r and a=a^r
                            in the correlation sample generation.
            min_age (int): Minimum age for distribution fitting (i.e., from HMD).
            max_age (int): Maximum age for distribution fitting.
            n_samples_for_corr (int): Number of samples for twin correlation check.
            n_samples_for_distribution (int): Number of samples for the
                                            death distribution KS calculation.
            std_initial (float): Initial guess for std.
            maxiter (int): Maximum iterations for optimizer.
            method (str): Optimization method, e.g. 'Nelder-Mead', 'Powell', etc.
            alpha (float): Weight for the correlation penalty term. Increase if
                        you want to enforce correlation more strictly.
            callback (bool): If True, prints iteration info.
            **fixed_params: Any subset of {'a','b','c','m','std'} to hold constant.

        Returns:
            result (OptimizeResult): The result from scipy.optimize.minimize.
        """

        import copy
        from scipy.optimize import minimize
        import numpy as np

        # ----------------------------------------------------------------------
        # Prepare data for the KS-based (distribution) part: HMD death distribution
        # ----------------------------------------------------------------------
        ages, death_distribution = self.get_HMD_death_distribution(**HMD_params)
        age_filter = (ages >= min_age) & (ages <= max_age)
        ages_filtered = ages[age_filter]
        ddist_filtered = death_distribution[age_filter]
        normalized_distribution = ddist_filtered / np.sum(ddist_filtered)

        # ----------------------------------------------------------------------
        # Decide which parameters are free vs fixed
        # ----------------------------------------------------------------------
        # Possible param names: a, b, c, m, std
        # 'std' is not stored in self, so treat it specially.
        # Meanwhile, the model stores a,b,c,m in self.
        base_params = {
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'm': self.m,
            'std': std_initial
        }

        # Overwrite with any fixed_params that user explicitly wants to keep
        for p_name, p_val in fixed_params.items():
            base_params[p_name] = p_val

        # Identify free parameters
        free_param_names = []
        for key in ['a', 'b', 'c', 'm', 'std']:
            if key not in fixed_params:  # means we want to optimize it
                free_param_names.append(key)

        # Build initial guess from base_params for the free parameters
        x0 = [base_params[p] for p in free_param_names]

        # ----------------------------------------------------------------------
        # Define objective function
        # ----------------------------------------------------------------------
        def compute_ks_statistic(death_times, ages_filt, norm_distr):
            # Compute empirical CDF from the sampled death_times within the age bounds
            cdf_theoretical = np.cumsum(norm_distr)
            # Make histogram from death_times
            counts, _ = np.histogram(death_times, bins=np.arange(ages_filt[0], ages_filt[-1] + 2))
            cdf_empirical = np.cumsum(counts) / sum(counts) if np.sum(counts) > 0 else np.zeros_like(counts)
            # Interpolate to match length of cdf_theoretical
            # (since cdf_theoretical is typically len(ages_filt),
            # so we take the first len(ages_filt) elements from cdf_empirical)
            # adjust if needed
            min_len = min(len(cdf_empirical), len(cdf_theoretical))
            ks_dist = np.max(np.abs(cdf_empirical[:min_len] - cdf_theoretical[:min_len]))
            return ks_dist

        def local_objective(x):
            """
            x = array of free parameters in the order of free_param_names.
            """
            # 1) Create a copy of the model
            model_copy = copy.deepcopy(self)

            # 2) Reconstruct the full set of parameters (including std)
            #    from x and the fixed_params dict
            param_dict = {}
            idx = 0
            for p_name in ['a', 'b', 'c', 'm', 'std']:
                if p_name in free_param_names:
                    param_dict[p_name] = x[idx]
                    idx += 1
                else:
                    # p_name is fixed
                    param_dict[p_name] = fixed_params[p_name]

            # 3) Update the model_copy's a,b,c,m
            model_copy.set_params({
                'a': param_dict['a'],
                'b': param_dict['b'],
                'c': param_dict['c'],
                'm': param_dict['m']
            })

            # 4) Sample death times for the distribution-based KS
            #    We want param_name='a' as in fit_params_with_std_with_ks or
            #    you may do param_name_corr if relevant.
            #    Typically, you might randomize param_name_corr for distribution
            #    as well, but let's keep it consistent with your existing code
            #    or adapt as needed.
            death_times_dist, _ = model_copy.sample_death_times_with_random_param(
                n=n_samples_for_distribution,
                param_name=param_name_corr,
                dt=1,
                std=param_dict['std'],
                coupled_ab=coupled_ab
            )
            ks_dist = compute_ks_statistic(death_times_dist, ages_filtered, normalized_distribution)

            # 5) Compute the correlation (using the desired twin_type) with
            #    param_name_corr, same std, etc.
            #    We'll do that in model_copy as well.
            def corr_func():
                # We create twins, each with param_name_corr randomization
                twin_table = model_copy.create_death_table_for_twins(
                    n=n_samples_for_corr,
                    param_name=param_name_corr,
                    std=param_dict['std'],
                    twin_type=corr_twin_type,
                    coupled_ab=coupled_ab
                )
                # Filter if your ta.filter_death_table is needed:
                twin_table_filtered = ta.filter_death_table(twin_table, 15)
                return twin_table_filtered['death1'].corr(twin_table_filtered['death2'])

            corr_val = corr_func()

            # 6) Combine into an objective with a penalty for dev from target_corr
            penalty = (corr_val - target_corr) ** 2
            # Weighted sum of KS distance + alpha * penalty
            return ks_dist + alpha * penalty

        iteration = [0]  # to track iteration count in callback

        def nm_callback(xk):
            if callback:
                print(f"Iteration {iteration[0]}: x={xk}")
            iteration[0] += 1

        # ----------------------------------------------------------------------
        # Use scipy.optimize.minimize
        # ----------------------------------------------------------------------
        result = minimize(
            local_objective,
            x0=x0,
            method=method,
            options={'maxiter': maxiter, 'disp': True},
            callback=nm_callback
        )

        # ----------------------------------------------------------------------
        # Assign best-fit parameters back to self, if success
        # ----------------------------------------------------------------------
        # Reconstruct final dictionary of parameters
        final_params = {}
        idx = 0
        for p_name in ['a', 'b', 'c', 'm', 'std']:
            if p_name in free_param_names:
                final_params[p_name] = result.x[idx]
                idx += 1
            else:
                final_params[p_name] = fixed_params[p_name]

        # Update the instance
        self.set_params({
            'a': final_params['a'],
            'b': final_params['b'],
            'c': final_params['c'],
            'm': final_params['m']
        })

        print("Best-fit parameters (a, b, c, m, std):",
            (final_params['a'], final_params['b'],
            final_params['c'], final_params['m'],
            final_params['std']))

        # If you want to see final correlation:
        final_corr = self.calc_corr(
            n=n_samples_for_corr,
            std=final_params['std'],
            twin_type=corr_twin_type,
            param_name=param_name_corr,
            coupled_ab=coupled_ab
        )
        print("Final correlation:", final_corr)

        return final_params  # Return parameters as a dictionary regardless of success
    
    
    def calc_corr(self, n=100000, std=0.3, twin_type='MZ', param_name='b', coupled_ab=True, filter_age=15, twin_table = None):
        if twin_table is None:
            twin_table = self.create_death_table_for_twins(n=n, std=std, param_name=param_name, twin_type=twin_type, coupled_ab=coupled_ab)
        twin_table = ta.filter_death_table(twin_table, filter_age)
        # filter so that both twins die before 110
        twin_table = ta.filter_death_table(twin_table, 110, above = False)
        corr = twin_table['death1'].corr(twin_table['death2'])
        return corr
