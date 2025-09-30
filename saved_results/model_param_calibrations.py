"""
Model parameter calibrations for mortality analysis.

This module contains calibrated parameters and utility functions for different mortality models
including Stochastic Repair (SR) and Modified Gamma-Gompertz (MGG) models across various cohorts.
"""

from src import sr_utils as utils
import numpy as np
from scipy.interpolate import RectBivariateSpline
import pickle
import os

# ============================================================================
#  (SR) MODEL CALIBRATIONS
# ============================================================================

# SR calibrations for different countries
sr_calibrations = {
    'denmark': {
        'm_ex': 0.003,
        'eta_factor': 1.33,
        'beta_factor': 1.16,
        'Xc': 17,
        'Xc_std': 0.21,
        'Xc_std_lower': 0.19,
        'Xc_std_upper': 0.23
    },
    'sweden': {
        'm_ex': 10**((-2.4676-2.5463)/2),
        'eta_factor': 1.33,
        'beta_factor': 1.17,
        'Xc': 18.4,
        'Xc_std': 0.21,
        'Xc_std_lower': 0.2,
        'Xc_std_upper': 0.22
    },
    'satsa': {
        'm_ex': 0.002,
        'eta_factor': 1.33,
        'beta_factor': 1.17,
        'Xc': 18.4,
        'Xc_std': 0.2,
        'Xc_std_lower': 0.17,
        'Xc_std_upper': 0.23
    },
    'usa': {
        'm_ex': 0.0027,
        'eta_factor': 0.96*1.33,
        'beta_factor': 0.98*1.16,
        'Xc': 0.96 * 17,
        'Xc_std': 0.23,
        'Xc_std_lower': 0.21,
        'Xc_std_upper': 0.25
    }
}

# Create and calibrate parameter dictionaries for SR model
country_dicts = {}
for country in sr_calibrations:
    country_dict = utils.load_baseline_human_params_dict()
    for param, value in country_dict.items():
        factor_key = f'{param}_factor'
        if factor_key in sr_calibrations[country]:
            country_dict[param] = float(country_dict[param]) * sr_calibrations[country][factor_key]
    # Add m_ex directly from sr_calibrations
    if 'm_ex' in sr_calibrations[country]:
        country_dict['m_ex'] = sr_calibrations[country]['m_ex']
    # Add Xc directly from sr_calibrations
    if 'Xc' in sr_calibrations[country]:
        country_dict['Xc'] = sr_calibrations[country]['Xc']
    country_dicts[country] = country_dict

# Individual country dictionaries for convenience
sr_denmark_dict = country_dicts['denmark']
sr_sweden_dict = country_dicts['sweden']
sr_satsa_dict = country_dicts['satsa']
sr_usa_dict = country_dicts['usa']


def print_calibrated_SR_params(cohort):
    """Print calibrated SR parameters with their units using Unicode superscripts.
    
    Args:
        cohort (str): Either 'denmark', 'sweden', 'satsa', or 'usa'
    """
    if cohort not in sr_calibrations:
        raise ValueError(f"Cohort must be one of {list(sr_calibrations.keys())}")
        
    calibs = sr_calibrations[cohort]
    
    if 'eta_factor' in calibs:
        eta_val = calibs['eta_factor'] * utils.karin_params['eta'][0]
        print(f"η = {eta_val:.2f} year⁻²,")
    elif 'eta' in calibs:
        eta_val = calibs['eta'] * utils.karin_params['eta'][0]
        print(f"η = {eta_val:.2f} year⁻²,")
        
    if 'beta_factor' in calibs:
        beta_val = calibs['beta_factor'] * utils.karin_params['beta'][0]
        print(f"β = {beta_val:.2f} year⁻¹,")
    elif 'beta' in calibs:
        beta_val = calibs['beta'] * utils.karin_params['beta'][0]
        print(f"β = {beta_val:.2f} year⁻¹,")
        
    if 'epsilon' in calibs:
        eps_val = calibs['epsilon'] * utils.karin_params['epsilon'][0]
        print(f"ε = {eps_val:.2f} year⁻¹,")
    else:
        print(f"ε = {utils.karin_params['epsilon'][0]:.2f} year⁻¹,")
        
    print(f"κ = {utils.karin_params['kappa'][0]:.2f},")
        
    if 'Xc_std' in calibs:
        print(f"Xc = {calibs['Xc']:.1f},")
        print(f"{calibs['Xc_std']*100:.0f}% ({calibs['Xc_std_lower']*100:.0f}%,{calibs['Xc_std_upper']*100:.0f}%)")
    
    print(f"m_ex = {calibs['m_ex']:.2e} year⁻¹")


# ============================================================================
# MODIFIED GAMMA-GOMPERTZ (MGG) MODEL CALIBRATIONS
# ============================================================================

mgg_calibrations = {
    'denmark_old': {
        'a': 5.36e-06,
        'b': 0.12, 
        'c': 21.624,
        'm': 0.004,
        'std': 0.35,
        'std_lower': 0.335,
        'std_upper': 0.39
    },
    'denmark': {
        'a': 1e-5,
        'b': 0.115,
        'c': 30,
        'm': 0.0022,
        'std': 0.27,
        'std_lower': 0.23,
        'std_upper': 0.31
    },
    'sweden': {
        'a': 4.7e-06,
        'b': 0.12,
        'c': 40,
        'm': 0.0015,
        'std': 0.31,
        'std_lower': 0.27,
        'std_upper': 0.35
    },
    'satsa': {
        'a': 4.23e-06,
        'b': 0.1235,
        'c': 20,
        'm': 0.001,
        'std': 0.27,
        'std_lower': 0.24,
        'std_upper': 0.31
    },
    'usa': {
        'a': 9.13e-05,
        'b': 0.087,
        'c': 100,
        'm': 0.001,
        'std': 0.27,
        'std_lower': 0.24,
        'std_upper': 0.31
    }
}

def print_calibrated_GG_params(dict_or_country):
    """Print calibrated MGG parameters with their units using Unicode superscripts.
    
    Args:
        dict_or_country: Either a dictionary of parameters or a string ('denmark', 'sweden', 'satsa', or 'usa')
        m: Whether to print mortality rate parameter
    """
    # If string input, get corresponding dictionary from mgg_calibrations
    if isinstance(dict_or_country, str):
        if dict_or_country.lower() not in ['denmark', 'sweden', 'satsa', 'usa']:
            raise ValueError("Country must be one of 'denmark', 'sweden', 'satsa', or 'usa'")
        dict = mgg_calibrations[dict_or_country.lower()]
    else:
        dict = dict_or_country

    print(f"a={dict['a']:.2e} year⁻¹,")
    print(f"b={dict['b']:.2f} year⁻¹,")
    print(f"c = {dict['c']:.0f},")
    print(f"{dict['std']*100:.0f}% ({dict['std_lower']*100:.0f}%,{dict['std_upper']*100:.0f}%)")
    print(f"m = {dict['m']:.2e} year⁻¹")



# ============================================================================
# CORRELATION ANALYSIS FUNCTIONS
# ============================================================================

# Define age and h_ext ranges for correlation analysis÷
h_exts = np.concatenate([[1e-30, 9e-7], np.logspace(-6, -2, 20), [10**(-1.9)]]).flatten()
filtered_ages = np.arange(5, 80, 1)  # 5 to 79 in steps of 1


def get_correlation_value(model, filter_age, log10h_ext, metric, cohort):
    """Get interpolated correlation value for given parameters using RectBivariateSpline.

    Args:
        model (str): Model type ('sr' or 'sm')
        filter_age (float): Age to filter at
        log10h_ext (float): Log10 of h_ext value
        metric (str): Type of metric ('mz', 'dz', or 'h2')
        cohort (str): Either 'danish', 'swedish', or 'satsa'

    Returns:
        str: Interpolated value in 'std_val (lower_val, upper_val)' format
    """
    import numpy as np
    import pickle
    from scipy.interpolate import RectBivariateSpline
    
    # Use global filtered_ages and h_exts
    log_h_exts = np.log10(h_exts)

    # Load the appropriate correlation matrices
    model = model.upper()
    cohort = cohort.lower()
    if cohort not in ['danish', 'swedish', 'satsa', 'usa']:
        raise ValueError("cohort must be either 'danish', 'swedish', or 'satsa'")
        
    # Build path relative to this file's directory (saved_results)
    base_dir = os.path.dirname(__file__)
    filepath = os.path.join(base_dir, f"{cohort}_{model}_correlation_matrices.pkl")
    
    try:
        with open(filepath, 'rb') as f:
            matrices = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find correlation matrices at {filepath}")

    # Set up key mappings based on metric
    metric = metric.lower()
    if metric == 'mz':
        keys = {'std': 'mz', 'upper': 'mz_upper', 'lower': 'mz_lower'}
    elif metric == 'dz':
        keys = {'std': 'dz', 'upper': 'dz_upper', 'lower': 'dz_lower'}
    elif metric == 'h2':
        keys = {'std': 'h2_std', 'upper': 'h2_upper', 'lower': 'h2_lower'}
    else:
        raise ValueError("metric must be one of: 'mz', 'dz', 'h2'")

    # Extract matrices
    try:
        z_std = matrices[keys['std']]
        z_lower = matrices[keys['lower']]
        z_upper = matrices[keys['upper']]
    except KeyError as e:
        raise KeyError(f"Missing key in correlation matrices: {e}")
    
    # Ensure matrices have correct shape - now using global filtered_ages
    if z_std.shape[0] != len(filtered_ages) or z_std.shape[1] != len(log_h_exts):
        # Adjust the grid to match the actual data
        filtered_ages_local = np.linspace(5, 79, z_std.shape[0])  # Adjust to match global range
        
        # Keep original log_h_exts but make sure it has the right length
        if z_std.shape[1] != len(log_h_exts):
            # Create a new log_h_exts with the right length but similar range
            log_h_exts = np.linspace(log_h_exts[0], log_h_exts[-1], z_std.shape[1])
    else:
        filtered_ages_local = filtered_ages
    
    # Make sure log_h_exts is strictly increasing
    if not np.all(np.diff(log_h_exts) > 0):
        # Find unique values and sort them
        log_h_exts_unique = np.unique(log_h_exts)
        
        # Create new matrices with only the unique values
        z_std_new = np.zeros((z_std.shape[0], len(log_h_exts_unique)))
        z_lower_new = np.zeros((z_lower.shape[0], len(log_h_exts_unique)))
        z_upper_new = np.zeros((z_upper.shape[0], len(log_h_exts_unique)))
        
        for i, val in enumerate(log_h_exts_unique):
            idx = np.where(log_h_exts == val)[0][0]
            z_std_new[:, i] = z_std[:, idx]
            z_lower_new[:, i] = z_lower[:, idx]
            z_upper_new[:, i] = z_upper[:, idx]
        
        log_h_exts = log_h_exts_unique
        z_std = z_std_new
        z_lower = z_lower_new
        z_upper = z_upper_new
    
    # Ensure input values are within bounds
    filter_age = np.clip(filter_age, filtered_ages_local.min(), filtered_ages_local.max())
    log10h_ext = np.clip(log10h_ext, log_h_exts.min(), log_h_exts.max())
    
    # Create spline interpolators
    try:
        f_std = RectBivariateSpline(filtered_ages_local, log_h_exts, z_std, kx=1, ky=1)
        f_lower = RectBivariateSpline(filtered_ages_local, log_h_exts, z_lower, kx=1, ky=1)
        f_upper = RectBivariateSpline(filtered_ages_local, log_h_exts, z_upper, kx=1, ky=1)
    except Exception as e:
        # Fallback to nearest neighbor if interpolation fails
        age_idx = np.abs(filtered_ages_local - filter_age).argmin()
        h_ext_idx = np.abs(log_h_exts - log10h_ext).argmin()
        
        std_val = z_std[age_idx, h_ext_idx]
        lower_val = z_lower[age_idx, h_ext_idx]
        upper_val = z_upper[age_idx, h_ext_idx]
        
        return f"{std_val:.2f} ({lower_val:.2f}, {upper_val:.2f})"
    
    # Interpolate at the supplied filter_age and log10h_ext
    std_val = float(f_std(filter_age, log10h_ext)[0][0])
    lower_val = float(f_lower(filter_age, log10h_ext)[0][0])
    upper_val = float(f_upper(filter_age, log10h_ext)[0][0])
    
    # Check for NaN values and use nearest neighbor if needed
    if np.isnan(std_val) or np.isnan(lower_val) or np.isnan(upper_val):
        age_idx = np.abs(filtered_ages_local - filter_age).argmin()
        h_ext_idx = np.abs(log_h_exts - log10h_ext).argmin()
        
        std_val = z_std[age_idx, h_ext_idx]
        lower_val = z_lower[age_idx, h_ext_idx]
        upper_val = z_upper[age_idx, h_ext_idx]
    
    # Return as formatted string
    return f"{std_val:.2f} ({lower_val:.2f}, {upper_val:.2f})"

