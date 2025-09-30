"""
Plotting utilities for twin heterogeneity analyses.

Functions to visualize survival, hazard, relative survival probabilities,
error bars via bootstrap/delta/Wilson, and twin correlation summaries.
"""
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from src.sr_utils import create_param_distribution_dict, karin_params , create_sr_simulation
from src.simulation import SR_sim
from src.survival_analysis import relative_prob_to_live_past_age_X, relative_prob_to_live_past_age_X_conditional_on_age_Y
from src.correlation_analysis import calc_phi, calc_r_intracorr
from src.twin_analysis import calc_twin_death_table, filter_death_table, death_times_twins_of_people_past_age_X
from src.bootstrap import bootstrap_relative_prob, delta_method_relative_prob, wilson_ci_relative_prob, bootstrap_relative_prob_parallel
from src.simulation import SR_sim

def plot_survival_for_twins_of_people_past_age_X(sim,age_X, dt=1, from_age = 0, **kwargs):
    """
    Plot Kaplan-Meier survival curves for twins of individuals who survived past a given age.
    This function creates survival curves showing the probability of survival over time for 
    the twins of people who lived past age_X, optionally starting from a specified age.
    """
    death_times = death_times_twins_of_people_past_age_X(sim, age_X)
    # take only death_times that are greater than from_age
    death_times = death_times[death_times > from_age]
    event_observed = np.where(death_times == np.inf, 0, 1)
    kmf = KaplanMeierFitter()
    kmf.fit(death_times, event_observed, timeline=sim.tspan_hazard)
    kmf.plot_survival_function(**kwargs)

def plot_hazard_for_twins_of_people_past_age_X(sim,age_X, dt=1, **kwargs):
    """
    Plot the hazard function for twins of individuals who survived past a given age.
    This function creates a smoothed hazard plot showing the instantaneous risk of death
    over time for twins of people who lived past age_X.
    """
    death_times = death_times_twins_of_people_past_age_X(sim, age_X)
    event_observed = np.where(death_times == np.inf, 0, 1)
    naf = NelsonAalenFitter()
    # Use sim.tspan_hazard for time intervals
    time_intervals = sim.tspan_hazard
    naf.fit(death_times, event_observed, timeline=sim.tspan_hazard)
    naf.plot_hazard(bandwidth=3, **kwargs)
    
def plot_hazard_ratio_twins_of_people_past_age_X(sim, age_X, dt=1, **kwargs):
    """
    Plot the hazard ratio for twins of people who lived past age_X compared to the total population.
    This function compares the hazard (instantaneous risk of death) between twins of long-lived
    individuals and the general population, showing how much higher or lower their risk is at each age.
    
    Parameters:
    -----------
    sim : SR_sim object
        Simulation object containing the population data
    age_X : float
        The age threshold that the twin must have survived past
    dt : float, optional
        Time step for hazard calculation (default=1)
    **kwargs : dict
        Additional keyword arguments to pass to the plot function
    """
    death_times = death_times_twins_of_people_past_age_X(sim, age_X)
    event_observed = np.where(death_times == np.inf, 0, 1)
    
    # Calculate hazard for twins of people past age_X
    naf = NelsonAalenFitter()
    naf.fit(death_times, event_observed, timeline=sim.tspan_hazard)
    twins_hazard = naf.smoothed_hazard_(bandwidth=3)
    
    # Get the total population hazard from the simulation
    total_hazard = sim.hazard
    
    # Calculate the hazard ratio
    hazard_ratio = twins_hazard / total_hazard
    
    # Plot the hazard ratio
    plt.plot(sim.tspan_hazard, hazard_ratio, **kwargs)
    plt.xlabel('Age')
    plt.ylabel('Hazard Ratio')
    plt.title(f'Hazard Ratio for Twins of People Past Age {age_X}')
    
    return hazard_ratio

def plot_relative_survival_prob_to_age(sim=None, params_dict=karin_params, std=False, n=40000, ax=None, ages=range(50, 110, 5), minus_one=True, filter_age=6, v_survival=False, death_table=None, **kwargs):
    """
    Plot the relative survival probability for twins of individuals who lived past various ages.
    This function shows how the probability of a twin surviving past age X changes when conditioned
    on their twin having survived past age X. Can plot against age or survival probability, and
    optionally subtract 1 to show the excess probability above baseline.
    """
    if ax is None:
        fig, ax = plt.subplots()
    if sim is None and death_table is None:
        sim = SR_sim(**params_dict, n=n)
    
    if death_table is not None:
        # If death_table is provided, use it directly to calculate relative probabilities
        relative_probs = np.array([relative_prob_to_live_past_age_X(sim=None, age=age, death_table=death_table, filter_age=filter_age) for age in ages])
    else:
        # Otherwise use the simulation object
        relative_probs = np.array([relative_prob_to_live_past_age_X(sim, age, filter_age=filter_age) for age in ages])
    
    if minus_one:
        relative_probs = relative_probs - 1

    if v_survival == False:
        ax.plot(ages, relative_probs, **kwargs)
        ax.set_xlabel('Age X')
    else:
        if death_table is not None:
            raise ValueError("v_survival=True requires a simulation object, not just a death_table")
        survs = sim.kmf.predict(ages)
        ax.plot(survs, relative_probs, **kwargs)
        ax.set_xlabel('Survival Probability at Age X')

    
    ax.set_ylabel('Relative Probability')
    ax.set_title('Relative Probability to Live Past Age X Given Twin Lived Past Age X')

def plot_relative_survival_prob_to_age_conditional_on_age_Y(sim=None, params_dict=karin_params, std=False, n=40000, ax=None, ages_X=range(50, 110, 5), age_Y=80, minus_one=True, filter_age=6, v_survival=False, death_table=None, **kwargs):
    """
    Plot the relative survival probability for twins to live past age X given their twin lived past age Y.
    This function shows how the probability of a twin surviving past age X changes when conditioned
    on their twin having survived past age Y (which may be different from X). Can plot against age 
    or survival probability, and optionally subtract 1 to show the excess probability above baseline.
    """
    if ax is None:
        fig, ax = plt.subplots()
    if sim is None and death_table is None:
        sim = SR_sim(**params_dict, n=n)
    
    if death_table is not None:
        # If death_table is provided, use it directly to calculate relative probabilities
        relative_probs = np.array([relative_prob_to_live_past_age_X_conditional_on_age_Y(sim=None, age_X=age_X, age_Y=age_Y, death_table=death_table, filter_age=filter_age) for age_X in ages_X])
    else:
        # Otherwise use the simulation object
        relative_probs = np.array([relative_prob_to_live_past_age_X_conditional_on_age_Y(sim, age_X, age_Y, filter_age=filter_age) for age_X in ages_X])
    
    if minus_one:
        relative_probs = relative_probs - 1

    if v_survival == False:
        ax.plot(ages_X, relative_probs, **kwargs)
        ax.set_xlabel('Age X')
    else:
        if death_table is not None:
            raise ValueError("v_survival=True requires a simulation object, not just a death_table")
        survs = sim.kmf.predict(ages_X)
        ax.plot(survs, relative_probs, **kwargs)
        ax.set_xlabel('Survival Probability at Age X')

    
    ax.set_ylabel('Relative Probability')
    ax.set_title(f'Relative Probability to Live Past Age X Given Twin Lived Past Age {age_Y}')

def plot_relative_survival_prob_to_age_diff_stds(params_dict_baseline=karin_params, param='eta', dist_type='gaussian', stds=[0.05, 0.08, 0.1, 0.12, 0.15], n=40000, ax=None, minus_one=False):
    """
    Plot relative survival probabilities for different levels of parameter heterogeneity.
    This function creates multiple curves showing how the relative survival probability changes
    as the standard deviation of a specified parameter (e.g., eta) varies, allowing comparison
    of the effect of different levels of population heterogeneity.
    """
    if ax is None:
        fig, ax = plt.subplots()
    for std in stds:
        params_dict_temp = create_param_distribution_dict(param, std, n=n, dist_type=dist_type, params_dict=params_dict_baseline)
        plot_relative_survival_prob_to_age(params_dict=params_dict_temp, std=std, n=n, ax=ax, minus_one=minus_one)
        print(std)
    ax.legend()
    ax.set_xlabel('Age X')
    ax.set_ylabel('Relative Probability')
    ax.set_title(f'Relative Probability to Live Past Age X Given Twin Lived Past Age X, {param} Distribution')

def plot_phi_survive_to_age(params_dict, std=False, n=40000, ax=None, sim=None):
    """
    Plot the phi correlation coefficient for twin survival past various ages.
    This function calculates and plots the phi correlation (a measure of association for
    binary variables) between whether both twins in a pair survive past different ages,
    showing how twin survival correlation changes with age.
    """
    if ax is None:
        fig, ax = plt.subplots()
    if sim is None:
        from src.simulation import SR_sim
        sim = SR_sim(**params_dict, n=n)
    death_table_temp = calc_twin_death_table(sim)
    phi_corrs = np.array([calc_phi(death_table_temp > age) for age in range(50, 110, 5)])
    ax.plot(range(50, 110, 5), phi_corrs, 'o-', label=rf'$\sigma = {std}$')
    ax.legend()
    ax.set_xlabel('Age X')
    ax.set_ylabel('Phi correlation')
    ax.set_title('Phi correlation for both Twins Living Past Age X')

# plot the correlation for either MZ or DZ twins based on parameter want to distribute, and baseline params - it creates the simulations and then calculates the correlation
def plot_corr_twin_deaths(params_dict_baseline=karin_params, params=['eta'], dist_type = 'gaussian', stds=[0.03, 0.05, 0.07, 0.08, 0.1, 0.12, 0.15], 
                          n=int(1e5), h_ext=None, ax=None, family=['MZ', 'DZ'], filter_from = None,plot_ratio=False,**kwargs):
    """
    Plot intraclass correlations of death times for monozygotic (MZ) and dizygotic (DZ) twins.
    This function generates simulations with varying parameter heterogeneity and calculates
    the correlation in death times between twin pairs. Can plot either the correlations
    themselves or the ratio of MZ to DZ correlations, which is important for heritability analysis.
    """
    ax = ax or plt.subplots(figsize=(8, 6))[1]
    params = [params] if isinstance(params, str) else params
    family = [family] if isinstance(family, str) else family
    
    line_styles = {'MZ': '-', 'DZ': '-'}
    
    corrs = {fam: [] for fam in family}
    for std in stds:
        for fam in family:
            params_dict_temp = create_param_distribution_dict(params, std, n=n, dist_type=dist_type, 
                                                              params_dict=params_dict_baseline.copy(), family=fam)
            sim = create_sr_simulation(params_dict=params_dict_temp ,h_ext=h_ext, n=n , parallel=True)
            death_table = calc_twin_death_table(sim)

            if filter_from is not None:
                death_table = filter_death_table(death_table, filter_from)

            corrs[fam].append(calc_r_intracorr(death_table))
            print(f"Completed: params={params}, family={fam}, std={std}, corr = {round(calc_r_intracorr(death_table),3)}")
    
    if plot_ratio:
        ratio = [mz / dz if dz != 0 else float('inf') for mz, dz in zip(corrs['MZ'], corrs['DZ'])]
        ax.plot(stds, ratio, 'o-', **kwargs)
    else:
        for fam in family:
            ax.plot(np.array(stds)*100, corrs[fam], f'o{line_styles[fam]}', **kwargs)
    
    ax.legend()
    ax.set_xlabel('Parameter Variation (%)')
    
    if plot_ratio:
        ax.set_ylabel('MZ/DZ Correlation Ratio')
        ax.set_title('Ratio of MZ to DZ Intraclass Correlations for Twin Deaths')
        ax.axhline(y=2, color='r', linestyle='--')
        ax.text(stds[0], 2.1, "Observed correlation ratio", color='r', ha='left', va='bottom')
    else:
        ax.set_ylabel('Death Time Correlation')
        ax.set_title('Death Time Correlation of Twin Deaths')
        #ax.axhline(y=0.2, color='g', linestyle='--')
        #ax.text(stds[0], 0.21, "Observed MZ correlation", color='g', ha='left', va='bottom')
        #ax.axhline(y=0.1, color='b', linestyle='--')
        #ax.text(stds[0], 0.11, "Observed DZ correlation", color='b', ha='left', va='bottom')

    return ax

def plot_relative_survival_prob_to_age_with_errors(sim, method='bootstrap', ax=None, n_bootstrap=150, n_cores=None, **kwargs):
    """
    Plot relative survival probabilities with confidence intervals using various statistical methods.
    This function calculates and plots the relative survival probability for twins of long-lived
    individuals along with error bars computed using bootstrap resampling, delta method, or
    Wilson confidence intervals to quantify uncertainty in the estimates.
    """

    if ax is None:
        fig, ax = plt.subplots()
    
    ages = range(50, 110, 5)
    
    if method == 'bootstrap':
        if n_cores is None:
            from multiprocessing import cpu_count
            n_cores = cpu_count()
        
        from multiprocessing import Pool
        with Pool(n_cores) as pool:
            bootstrap_args = [(sim, age, n_bootstrap) for age in ages]
            results = pool.map(bootstrap_relative_prob_parallel, bootstrap_args)
        
        relative_probs, error_bars = zip(*results)
    
    elif method == 'delta':
        relative_probs, error_bars = zip(*[delta_method_relative_prob(sim, age) for age in ages])
    
    elif method == 'wilson':
        relative_probs, error_bars = zip(*[wilson_ci_relative_prob(sim, age) for age in ages])
    
    else:
        raise ValueError("Invalid method. Choose 'bootstrap', 'delta', or 'wilson'.")
    
    ax.errorbar(ages, relative_probs, yerr=error_bars, fmt='o-', capsize=5, **kwargs)
    ax.set_xlabel('Age [years]')
    ax.set_ylabel('Relative Survival Probability')
    ax.set_title(f'Relative Survival Probability: ({method.capitalize()} method)')

    # At the end of each module file (e.g., sr_utils.py, twin_analysis.py, etc.)
__all__ = [name for name in dir() if not name.startswith('_')]