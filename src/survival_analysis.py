"""
Survival and hazard utilities for SR simulations and twin tables.

Functions include unconditional/conditional survival past age X,
relative survival probabilities, excess survival fits, and
hazard calculation for selected subgroups.
"""
import numpy as np
from lifelines import NelsonAalenFitter
from src.twin_analysis import calc_twin_death_table, death_times_twins_of_people_past_age_X

def chance_to_live_past_age_X(sim, age, death_times=None, death_table=None, filter_age=15):
    """
    Calculate the probability that an individual lives past a given age.
    
    This function computes the proportion of individuals in the population who survive
    beyond the specified age threshold. It can work with simulation objects, death tables,
    or direct death time arrays.
    
    Parameters:
    -----------
    sim : SR_sim object
        Simulation object containing population data
    age : float
        The age threshold to calculate survival probability for
    death_times : array-like, optional
        Array of death times. If None, uses sim.death_times
    death_table : DataFrame, optional
        DataFrame with 'death1' and 'death2' columns for twin pairs
    filter_age : float, default=15
        Minimum age to include in the analysis (filters out early deaths)
    
    Returns:
    --------
    float
        Probability of surviving past the given age
    """
    if death_table is not None:
        # If death_table is provided, combine death1 and death2 columns
        death_times = np.concatenate([death_table['death1'].values, death_table['death2'].values])
    elif death_times is None:
        death_times = sim.death_times
    
    death_times = death_times[death_times >= filter_age]
    return np.mean(death_times >= age)

def chance_to_live_past_age_X_conditional(sim, age, death_times=None, death_table=None, filter_age=15):
    """
    Calculate the conditional probability that a twin lives past age X, given their twin lived past age X.
    
    This function computes the probability that one twin survives past a given age, conditioned
    on their twin also surviving past that age. This is a key measure for understanding
    familial clustering of longevity.
    
    Parameters:
    -----------
    sim : SR_sim object
        Simulation object containing population data
    age : float
        The age threshold for the conditional probability calculation
    death_times : array-like, optional
        Array of death times. If None, uses sim.death_times
    death_table : DataFrame, optional
        DataFrame with 'death1' and 'death2' columns for twin pairs
    filter_age : float, default=15
        Minimum age to include in the analysis
    
    Returns:
    --------
    float
        Conditional probability of surviving past age X given twin survived past age X
    """
    if death_table is not None:
        death_table = death_table[
            (death_table['death1'] >= filter_age) &
            (death_table['death2'] >= filter_age)
        ].copy()
    elif death_times is None:
        death_table = calc_twin_death_table(sim, filter_age=filter_age)
    else:
        death_table = calc_twin_death_table(sim, death_times, filter_age=filter_age)
    
    # Create boolean table indicating which twins survived past the age threshold
    survived_after_table = death_table >= age
    
    # Calculate conditional probabilities for both directions
    # P(twin1 survives | twin2 survives)
    mean1 = np.mean(survived_after_table['death1'][survived_after_table['death2']==True])
    # P(twin2 survives | twin1 survives)
    mean2 = np.mean(survived_after_table['death2'][survived_after_table['death1']==True])
    
    # Return the average of both conditional probabilities
    return (mean1 + mean2) / 2

def chance_to_live_past_age_X_conditional_on_age_Y(sim, age_X, age_Y, death_times=None, death_table=None, filter_age=15):
    """
    Calculate the conditional probability that a twin lives past age X, given their twin lived past age Y.
    
    This function computes the probability that one twin survives past age X, conditioned
    on their twin surviving past age Y (which may be different from X). This allows for
    more flexible analysis of familial clustering of longevity.
    
    Parameters:
    -----------
    sim : SR_sim object
        Simulation object containing population data
    age_X : float
        The age threshold for the twin whose survival we're calculating
    age_Y : float
        The age threshold that the conditioning twin must have survived past
    death_times : array-like, optional
        Array of death times. If None, uses sim.death_times
    death_table : DataFrame, optional
        DataFrame with 'death1' and 'death2' columns for twin pairs
    filter_age : float, default=15
        Minimum age to include in the analysis
    
    Returns:
    --------
    float
        Conditional probability of surviving past age X given twin survived past age Y
    """
    if death_table is not None:
        death_table = death_table[
            (death_table['death1'] >= filter_age) &
            (death_table['death2'] >= filter_age)
        ].copy()
    elif death_times is None:
        death_table = calc_twin_death_table(sim, filter_age=filter_age)
    else:
        death_table = calc_twin_death_table(sim, death_times, filter_age=filter_age)
    
    # Create boolean tables for both age thresholds
    survived_X_table = death_table >= age_X
    survived_Y_table = death_table >= age_Y
    
    # Calculate conditional probabilities for both directions
    # P(twin1 survives past X | twin2 survives past Y)
    mean1 = np.mean(survived_X_table['death1'][survived_Y_table['death2']==True])
    # P(twin2 survives past X | twin1 survives past Y)
    mean2 = np.mean(survived_X_table['death2'][survived_Y_table['death1']==True])
    
    # Return the average of both conditional probabilities
    return (mean1 + mean2) / 2

def relative_prob_to_live_past_age_X(sim, age, death_times=None, death_table=None, filter_age=15):
    """
    Calculate the relative survival probability (RSP) for twins of long-lived individuals.
    
    This function computes the ratio of conditional survival probability to unconditional
    survival probability. An RSP > 1 indicates that having a long-lived twin increases
    one's own survival probability, suggesting familial clustering of longevity.
    
    Parameters:
    -----------
    sim : SR_sim object
        Simulation object containing population data
    age : float
        The age threshold for calculating relative survival probability
    death_times : array-like, optional
        Array of death times. If None, uses sim.death_times
    death_table : DataFrame, optional
        DataFrame with 'death1' and 'death2' columns for twin pairs
    filter_age : float, default=15
        Minimum age to include in the analysis
    
    Returns:
    --------
    float
        Relative survival probability (conditional probability / unconditional probability)
    """
    return chance_to_live_past_age_X_conditional(sim, age, death_times, death_table, filter_age) / chance_to_live_past_age_X(sim, age, death_times, death_table, filter_age)

def relative_prob_to_live_past_age_X_conditional_on_age_Y(sim, age_X, age_Y, death_times=None, death_table=None, filter_age=15):
    """
    Calculate the relative survival probability (RSP) for twins to live past age X given their twin lived past age Y.
    
    This function computes the ratio of conditional survival probability (given twin survived past Y)
    to unconditional survival probability (for age X). An RSP > 1 indicates that having a twin who
    survived past age Y increases one's probability of surviving past age X.
    
    Parameters:
    -----------
    sim : SR_sim object
        Simulation object containing population data
    age_X : float
        The age threshold for the twin whose survival we're calculating
    age_Y : float
        The age threshold that the conditioning twin must have survived past
    death_times : array-like, optional
        Array of death times. If None, uses sim.death_times
    death_table : DataFrame, optional
        DataFrame with 'death1' and 'death2' columns for twin pairs
    filter_age : float, default=15
        Minimum age to include in the analysis
    
    Returns:
    --------
    float
        Relative survival probability (conditional probability / unconditional probability)
    """
    return chance_to_live_past_age_X_conditional_on_age_Y(sim, age_X, age_Y, death_times, death_table, filter_age) / chance_to_live_past_age_X(sim, age_X, death_times, death_table, filter_age)

def excess_survival_prob_past_age_X_exponential_fit(sim, ages, filter_age = 15, age_start = 50, age_end = 90):
    """
    Fit an exponential model to the excess survival probability across ages.
    
    This function calculates the relative survival probability across multiple ages,
    converts it to excess probability (RSP - 1), and fits an exponential decay model
    to understand how the familial survival advantage changes with age.
    
    Parameters:
    -----------
    sim : SR_sim object
        Simulation object containing population data
    ages : array-like
        Array of ages to calculate relative survival probabilities for
    filter_age : float, default=15
        Minimum age to include in the analysis
    age_start : float, default=50
        Starting age for the exponential fit
    age_end : float, default=90
        Ending age for the exponential fit
    
    Returns:
    --------
    tuple
        (intercept, slope) of the linear fit to log(excess probability) vs age
        The exponential model is: excess = exp(intercept + slope * age)
    """
    # Convert ages to numpy array if it's a range object
    ages_array = np.array(ages)
    
    # Calculate relative survival probabilities for all ages
    relative_probs = np.array([relative_prob_to_live_past_age_X(sim, age, filter_age = filter_age) for age in ages_array])
    # Convert to excess probability (amount above baseline)
    excess = relative_probs - 1
    
    # Filter ages to the specified range for fitting
    mask = (ages_array >= age_start) & (ages_array <= age_end)
    ages_fit = ages_array[mask]
    
    # Take log of excess only for the filtered age range
    excess_fit = excess[mask]
    log_excess_fit = np.log(excess_fit)
    
    # Fit linear regression to log(excess) vs filtered ages
    # This gives us the exponential decay parameters
    slope, intercept = np.polyfit(ages_fit, log_excess_fit, 1)
    return intercept, slope

def calc_hazard_for_twins_of_people_past_age_X(sim, age_X, dt=1, tspan = None):
    """
    Calculate the hazard function for twins of individuals who survived past a given age.
    
    This function computes the instantaneous risk of death (hazard) over time for twins
    of people who lived past age_X. The hazard function shows how the risk of death
    changes with age for this selected population.
    
    Parameters:
    -----------
    sim : SR_sim object
        Simulation object containing population data
    age_X : float
        The age threshold that the twin must have survived past
    dt : float, default=1
        Time step for hazard calculation (currently unused)
    tspan : array-like, optional
        Time span for hazard calculation. If None, uses sim.tspan_hazard
    
    Returns:
    --------
    pandas.Series
        Smoothed hazard function values over the specified time span
    """
    if tspan is None:
        tspan = sim.tspan_hazard
    
    # Get death times for twins of people who lived past age_X
    death_times = death_times_twins_of_people_past_age_X(sim, age_X)
    # Create event indicator (1 for observed death, 0 for censored)
    event_observed = np.where(death_times == np.inf, 0, 1)
    
    # Fit Nelson-Aalen estimator for cumulative hazard
    naf = NelsonAalenFitter()
    naf.fit(death_times, event_observed, timeline=tspan)
    
    # Return smoothed hazard function
    hazard_for_twins = naf.smoothed_hazard_(bandwidth=3)
    return hazard_for_twins


# At the end of each module file (e.g., sr_utils.py, twin_analysis.py, etc.)
__all__ = [name for name in dir() if not name.startswith('_')]