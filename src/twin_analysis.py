"""
Twin utilities: build twin death tables and index helpers.

Assumes twins are indexed consecutively (i, i+1). Provides helpers
to filter by age, compute absolute differences, and select twins of
individuals who survived past age X.
"""
import pandas as pd
import numpy as np

def filter_death_table(death_table, filter_age, above=True):
    """Filter twin death table based on age threshold.
    
    Parameters:
        death_table (pd.DataFrame): DataFrame with death1 and death2 columns
        filter_age (float): Age threshold to filter by
        above (bool): If True, keep pairs where both died >= filter_age
                     If False, keep pairs where both died <= filter_age
    
    Returns:
        pd.DataFrame: Filtered death table
    """
    if above:
        return death_table[(death_table['death1'] >= filter_age) & 
                          (death_table['death2'] >= filter_age)]
    else:
        return death_table[(death_table['death1'] <= filter_age) & 
                          (death_table['death2'] <= filter_age)]

def calc_twin_death_table(sim, death_times=None, filter_age=None, above=True):
    """Calculate table of twin death times.
    
    Parameters:
        sim: Simulation object containing death times
        death_times (array-like, optional): Death times to use instead of sim.death_times
        filter_age (float, optional): Age threshold to filter by
        above (bool): Direction of age filtering if filter_age provided
    
    Returns:
        pd.DataFrame: Table with columns death1, death2, abs_diff, death1_extrinsic, death2_extrinsic
    """
    if death_times is None:
        death_times = sim.death_times
    death1 = []; death2 = []
    death1_extrinsic = []; death2_extrinsic = []
    for i in range(0, len(death_times), 2):
        death1.append(death_times[i])
        death2.append(death_times[i+1])
        death1_extrinsic.append(bool(sim.extrinsic_deaths[i]))
        death2_extrinsic.append(bool(sim.extrinsic_deaths[i+1]))
        
    twin_death_table = pd.DataFrame({
        'death1': death1, 
        'death2': death2,
        'death1_extrinsic': death1_extrinsic,
        'death2_extrinsic': death2_extrinsic
    })

    # get rid of twin pairs where one of the twins is still alive
    twin_death_table = twin_death_table[(twin_death_table['death1'] != (sim.params.tmax + sim.params.dt)) & 
                                      (twin_death_table['death2'] != (sim.params.tmax + sim.params.dt))]
    
    twin_death_table['abs_diff'] = abs(twin_death_table['death1'] - twin_death_table['death2'])
    
    # filter by age if specified
    if filter_age is not None:
        twin_death_table = filter_death_table(twin_death_table, filter_age, above)
    
    return twin_death_table

# Returns the indices of the twins of people who have lived past age X
def indices_twins_of_people_past_age_X(sim, age_X):
    """
    Calculates the indices of the twins corresponding to individuals 
    whose death times are greater than or equal to age_X.

    Args:
        sim: Simulation object containing death times (sim.death_times). 
             Assumes twins are stored consecutively (index i and i+1).
        age_X (float): The age threshold.

    Returns:
        np.ndarray: An array of integer indices representing the twins.
    """
    # Find indices of individuals who lived past age_X
    lived_past_indices = np.where(sim.death_times >= age_X)[0]
    
    # Calculate twin indices using vectorized operations
    # If index i is even, twin is i+1. If index i is odd, twin is i-1.
    # This can be calculated as: i + 1 - 2 * (i % 2)
    twin_indices = lived_past_indices + 1 - 2 * (lived_past_indices % 2)
    
    # Ensure indices are integers (though the calculation should yield integers)
    return twin_indices.astype(int)

def death_times_twins_of_people_past_age_X(sim, age_X):
    """
    Returns the death times of the twins of individuals who lived past age_X.

    Args:
        sim: Simulation object containing death times (sim.death_times).
        age_X (float): The age threshold.

    Returns:
        np.ndarray: An array of death times for the corresponding twins.
    """
    twin_indices = indices_twins_of_people_past_age_X(sim, age_X)
    # Check if twin_indices is empty to avoid potential errors with empty slicing
    if twin_indices.size == 0:
        return np.array([]) # Return an empty array if no twins found
    return sim.death_times[twin_indices]

def filter_death_times_parameter_condition(sim, param, condition):
  # Determine which attribute to use based on the value of 'param', # example: condition = lambda Xc: (10 < Xc) & (Xc < 20)
  if param == 'Xc':
      data = sim.params.Xc
  elif param == 'epsilon':
      data = sim.params.epsilon
  elif param == 'eta':
      data = sim.params.eta
  elif param == 'beta':
      data = sim.params.beta
  else:
      raise ValueError(f"Unknown parameter: {param}")

  # Apply the condition to the selected data and get the indices where the condition is True. condition as lambda function:
  # example: condition = lambda Xc: (10 < Xc) & (Xc < 20)
  indices = np.where(condition(data))[0]
  # Return the death times corresponding to those indices
  return sim.death_times[indices]


def calc_avg_age_diff_randoms(death_table):
    abs_diffs = 0
    # just a really large number
    for i in range(0,200000):
        # pick random element from human_sim_eta_dist.death_times
        random_death_time1 = np.random.choice(death_table['death1'])
        random_death_time2 = np.random.choice(death_table['death2'])
        # calculate the absolute difference between random_death_time1 and random_death_time2
        abs_diffs = abs_diffs + abs(random_death_time1 - random_death_time2)

    # calculate the mean absolute difference
    mean_abs_diff = abs_diffs / 200000
    return mean_abs_diff

# At the end of each module file (e.g., sr_utils.py, twin_analysis.py, etc.)
__all__ = [name for name in dir() if not name.startswith('_')]