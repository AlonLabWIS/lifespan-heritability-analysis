"""
Correlation metrics for twin death-times and survival outcomes.

Provides:
- calc_phi: Phi coefficient for 2x2 survival contingency
- calc_pearson_corr_twin_deaths: Pearson correlation of death ages
- calc_icc: Intraclass correlation via pingouin
- calc_h2: Variance decomposition-based heritability proxy over parameter bins
"""

import numpy as np
import pandas as pd
import pingouin as pg
from src.twin_analysis import calc_twin_death_table, filter_death_table
from scipy.stats import pearsonr


def calc_phi(survived_after_table):
    contingency_table = pd.crosstab(survived_after_table['death1'], survived_after_table['death2'])
    n11 = contingency_table.loc[True, True]
    n00 = contingency_table.loc[False, False]
    n10 = contingency_table.loc[True, False]
    n01 = contingency_table.loc[False, True]

    n1_plus = contingency_table.sum(axis=1)[True]
    n0_plus = contingency_table.sum(axis=1)[False]
    n_plus1 = contingency_table.sum(axis=0)[True]
    n_plus0 = contingency_table.sum(axis=0)[False]

    phi = (n11 * n00 - n10 * n01) / ((n1_plus * n0_plus * n_plus1 * n_plus0) ** 0.5)
    return phi

def calc_pearson_corr_twin_deaths(sim, filter_age = None):
    death_table = calc_twin_death_table(sim)
    if filter_age is not None:
        death_table = filter_death_table(death_table, filter_age)
    return death_table.corr().iloc[0, 1]

def calc_MSB(death_table):
    death_table_copy = death_table.drop('abs_diff', axis=1)
    overall_mean = death_table_copy.values.mean()
    twin_death_avg = death_table_copy.mean(axis=1).values
    SSB = 2*((twin_death_avg - overall_mean)**2).sum()
    MSB = SSB/(len(death_table)-1)
    return MSB

def calc_MSW(death_table):
    death_table_copy = death_table.drop('abs_diff', axis=1)
    twin_death_avg = death_table_copy.mean(axis=1).values
    deaths_twin1 = death_table_copy['death1'].values
    deaths_twin2 = death_table_copy['death2'].values
    SSW = ((deaths_twin1 - twin_death_avg)**2).sum() + ((deaths_twin2 - twin_death_avg)**2).sum()
    MSW = SSW/(len(death_table))
    return MSW

def calc_r_intracorr(death_table):
    r  = pearsonr(death_table['death1'], death_table['death2'])[0]
    return r

def calc_icc(death_table):
    df_long = death_table.stack().reset_index()
    df_long.columns = ['Pair_ID', 'Twin', 'Death_Time']
    df_long['Pair_ID'] += 1
    df_long['Twin'] = df_long['Twin'].apply(lambda x: x.replace('death1', '1').replace('death2', '2'))
    df_long = df_long[df_long['Twin'] != 'abs_diff']

    icc_results = pg.intraclass_corr(data=df_long, targets='Pair_ID', raters='Twin', ratings='Death_Time')
    icc = icc_results.set_index('Type').loc['ICC1', 'ICC']
    return icc

def calc_h2(sim, filter_age=None, param = 'Xc'):
    if filter_age is None:
        filter_age = 15
    
    death_times = sim.death_times[sim.death_times > filter_age]
    param_array = getattr(sim.params, param)[sim.death_times > filter_age]
    
    # Get min and max values of Xc
    param_min = param_array.min()
    param_max = param_array.max()
    
    # Create 50 evenly spaced bins
    bins = np.linspace(param_min, param_max, 51)  # 51 edges to create 50 bins
    
    # groups: integers 0 … 49 telling you which bin
    groups = np.digitize(param_array, bins) - 1
    N = len(death_times)
    grand = death_times.mean()
    
    SS_within = 0.0           # ∑ (T_ij − T̄_i)²
    SS_between = 0.0           # ∑ n_i (T̄_i − grand)²
    
    for g in np.unique(groups):
        mask = groups == g
        T_g = death_times[mask]
        n_g = T_g.size
        if n_g < 2:      # skip bins with 0/1 obs — they give no variance info
            continue
        mean_g = T_g.mean()
        SS_within += ((T_g - mean_g)**2).sum()
        SS_between += n_g * (mean_g - grand)**2
    
    V_total = death_times.var(ddof=0)        # population variance
    V_noise = SS_within / N                  # E_G[Var_N(T|G)]
    V_genetic = SS_between / N               # Var_G(E_N[T|G])
    
    return V_genetic / V_total

# Public API
__all__ = [
    'calc_phi',
    'calc_pearson_corr_twin_deaths',
    'calc_MSB',
    'calc_MSW',
    'calc_r_intracorr',
    'calc_icc',
    'calc_h2',
]
