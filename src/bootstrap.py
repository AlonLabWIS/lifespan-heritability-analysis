"""
Bootstrap and interval estimates for relative survival probabilities.

Includes non-parametric bootstrap, delta method approximation, and
Wilson-style intervals; plus a helper for multiprocessing.
"""
import numpy as np
from statsmodels.stats.proportion import proportion_effectsize
from src.survival_analysis import relative_prob_to_live_past_age_X, chance_to_live_past_age_X, chance_to_live_past_age_X_conditional
from src.twin_analysis import calc_twin_death_table

def bootstrap_relative_prob(sim, age, n_bootstrap=1000):
    bootstrap_results = []
    death_times = sim.death_times
    n_pairs = len(death_times) // 2    
    pairs = death_times.reshape(n_pairs, 2)
    for _ in range(n_bootstrap):
        sampled_indices = np.random.choice(n_pairs, n_pairs, replace=True)
        sampled_pairs = pairs[sampled_indices]
        sampled_death_times = sampled_pairs.flatten()

        rel_prob = relative_prob_to_live_past_age_X(sim, age, death_times=sampled_death_times)
        bootstrap_results.append(rel_prob)
    
    return np.mean(bootstrap_results), np.std(bootstrap_results)

def delta_method_relative_prob(sim, age):
    N = len(sim.death_times) // 2
    
    p_unconditional = chance_to_live_past_age_X(sim, age)
    p_conditional = chance_to_live_past_age_X_conditional(sim, age)
    
    relative_prob = p_conditional / p_unconditional
    
    var_unconditional = p_unconditional * (1 - p_unconditional) / (2*N)
    var_conditional = p_conditional * (1 - p_conditional) / N
    
    var_relative = (relative_prob**2) * (var_conditional / (p_conditional**2) + var_unconditional / (p_unconditional**2))
    
    return relative_prob, np.sqrt(var_relative)

def wilson_ci_relative_prob(sim, age):
    N = len(sim.death_times) // 2
    
    death_table = calc_twin_death_table(sim)
    survived_after_table = death_table >= age
    
    unconditional_count = np.sum(survived_after_table)
    p_unconditional = unconditional_count / (2*N)
    
    conditional_count = np.sum(survived_after_table['death1'] & survived_after_table['death2'])
    conditional_n = np.sum(survived_after_table['death1'] | survived_after_table['death2'])
    p_conditional = conditional_count / conditional_n if conditional_n > 0 else 0
    
    relative_prob = p_conditional / p_unconditional if p_unconditional > 0 else 0
    
    _, ci_unconditional = proportion_effectsize(unconditional_count, 2*N)
    _, ci_conditional = proportion_effectsize(conditional_count, conditional_n) if conditional_n > 0 else ([0, 0], [0, 0])
    
    if p_unconditional > 0 and conditional_n > 0:
        lower_rel = ci_conditional[0] / ci_unconditional[1]
        upper_rel = ci_conditional[1] / ci_unconditional[0]
        std_error = (upper_rel - lower_rel) / (2 * 1.96)
    else:
        std_error = 0
    
    return relative_prob, std_error

def bootstrap_relative_prob_parallel(args):
    sim, age, n_bootstrap = args
    return bootstrap_relative_prob(sim, age, n_bootstrap)

# At the end of each module file (e.g., sr_utils.py, twin_analysis.py, etc.)
__all__ = [name for name in dir() if not name.startswith('_')]