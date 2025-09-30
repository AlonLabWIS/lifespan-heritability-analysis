"""
This module contains data from various twin studies on longevity, including correlations,
confidence intervals, and study metadata.
Now includes 67% confidence intervals (±1 SE) for h² as well.
"""
import math

def ci_to_se(lower_ci, upper_ci):
    """
    Convert a 95% confidence interval (lower_ci, upper_ci)
    to an approximate standard error, assuming normal distribution.
    For a two-sided 95% CI, the z-value is ~1.96,
    so se ~ (upper_ci - lower_ci) / (2 * 1.96) = (CI width) / 3.92
    """
    return (upper_ci - lower_ci) / 3.92

def compute_heritability_ci(r_mz, r_dz, ci_mz, ci_dz):
    """
    Computes heritability h² = 2 * (r_mz - r_dz)
    and its 95% CI/SE by propagating errors assuming independence.
    Returns:
    --------
    h2      : point estimate of heritability (float)
    ci_95   : tuple, 95% confidence interval for h²
    se_h2   : standard error of h²
    """
    se_mz = ci_to_se(ci_mz[0], ci_mz[1])
    se_dz = ci_to_se(ci_dz[0], ci_dz[1])
    diff_r = r_mz - r_dz
    se_diff_r = math.sqrt(se_mz**2 + se_dz**2)
    h2 = 2.0 * diff_r
    se_h2 = 2.0 * se_diff_r

    # 95% CI
    lower_h2 = h2 - 1.96 * se_h2
    upper_h2 = h2 + 1.96 * se_h2

    return h2, (lower_h2, upper_h2), se_h2

# -----------------------------------------------------------------------------
# Helper utilities to compute and attach h² to study dicts
# -----------------------------------------------------------------------------

def _compute_h2_package(r_mz, r_dz, ci_mz, ci_dz):
    """
    Convenience wrapper around compute_heritability_ci returning
    (h2, ci_95, se, ci_67) where ci_67 = (h2 - se, h2 + se).
    """
    h2, ci_95, se = compute_heritability_ci(r_mz, r_dz, ci_mz, ci_dz)
    ci_67 = (h2 - se, h2 + se)
    return h2, ci_95, se, ci_67

def attach_h2_for_cohorts(study):
    """
    For a study dict with a 'cohorts' section, compute h² for males and females
    per cohort and attach under key 'h2' using the existing field naming schema.
    """
    cohorts = study.get('cohorts', {})
    for cohort_key, cohort in cohorts.items():
        male_h2, male_ci, male_se, male_ci_67 = _compute_h2_package(
            cohort['male_MZ']['r'],
            cohort['male_DZ']['r'],
            cohort['male_MZ']['ci_95'],
            cohort['male_DZ']['ci_95'],
        )

        female_h2, female_ci, female_se, female_ci_67 = _compute_h2_package(
            cohort['female_MZ']['r'],
            cohort['female_DZ']['r'],
            cohort['female_MZ']['ci_95'],
            cohort['female_DZ']['ci_95'],
        )

        cohort['h2'] = {
            'males': male_h2,
            'males_ci': male_ci,
            'males_se': male_se,
            'males_ci_67': male_ci_67,
            'females': female_h2,
            'females_ci': female_ci,
            'females_se': female_se,
            'females_ci_67': female_ci_67,
        }

def attach_h2_for_combined(study):
    """
    For a study dict with a 'combined' section that includes keys
    male_MZ, male_DZ, female_MZ, female_DZ, all_MZ, all_DZ,
    compute study-level h² (males, females, all) and attach under
    'combined'[ 'h2' ] with the existing field naming schema.
    """
    combined = study['combined']

    male_h2, male_ci, male_se, male_ci_67 = _compute_h2_package(
        combined['male_MZ']['r'],
        combined['male_DZ']['r'],
        combined['male_MZ']['ci_95'],
        combined['male_DZ']['ci_95'],
    )

    female_h2, female_ci, female_se, female_ci_67 = _compute_h2_package(
        combined['female_MZ']['r'],
        combined['female_DZ']['r'],
        combined['female_MZ']['ci_95'],
        combined['female_DZ']['ci_95'],
    )

    all_h2, all_ci, all_se, all_ci_67 = _compute_h2_package(
        combined['all_MZ']['r'],
        combined['all_DZ']['r'],
        combined['all_MZ']['ci_95'],
        combined['all_DZ']['ci_95'],
    )

    combined['h2'] = {
        'males': male_h2,
        'males_ci': male_ci,
        'males_se': male_se,
        'males_ci_67': male_ci_67,
        'females': female_h2,
        'females_ci': female_ci,
        'females_se': female_se,
        'females_ci_67': female_ci_67,
        'all': all_h2,
        'all_ci': all_ci,
        'all_se': all_se,
        'all_ci_67': all_ci_67,
    }

# Herskind study data
herskind_study = {
    'metadata': {
        'cohort_start': 1870,
        'cohort_end': 1900,
        'country': 'denmark',
        'filter_age': 15,
        'mex_male': 3.99e-03,  # Fitted using Makeham-Gamma-Gompertz model
        'mex_std_male': 6.65e-04,
        'mex_female': 4.15e-03,  # Fitted using Makeham-Gamma-Gompertz model
        'mex_std_female': 5.51e-04,
        'mex': 4.07e-03,  # Average of male and female MGG fits
        'mex_std': 6.17e-04  # Average of male and female std
    },
    'cohorts': {
        '1870-1880': {
            'male_MZ': {
                'n': 113,
                'r': 0.251,
                'ci_95': (0.069, 0.416),
                'std_1': (0.160, 0.338)
            },
            'male_DZ': {
                'n': 186,
                'r': 0.081,
                'ci_95': (-0.064, 0.222),
                'std_1': (0.007, 0.154)
            },
            'female_MZ': {
                'n': 126,
                'r': 0.313,
                'ci_95': (0.146, 0.463),
                'std_1': (0.230, 0.392)
            },
            'female_DZ': {
                'n': 215,
                'r': 0.019,
                'ci_95': (-0.115, 0.152),
                'std_1': (-0.050, 0.087)
            }
        },
        '1881-1890': {
            'male_MZ': {
                'n': 168,
                'r': 0.223,
                'ci_95': (0.074, 0.362),
                'std_1': (0.148, 0.296)
            },
            'male_DZ': {
                'n': 306,
                'r': 0.111,
                'ci_95': (-0.001, 0.220),
                'std_1': (0.054, 0.167)
            },
            'female_MZ': {
                'n': 184,
                'r': 0.226,
                'ci_95': (0.084, 0.359),
                'std_1': (0.154, 0.295)
            },
            'female_DZ': {
                'n': 329,
                'r': 0.130,
                'ci_95': (0.022, 0.235),
                'std_1': (0.075, 0.184)
            }
        },
        '1891-1900': {
            'male_MZ': {
                'n': 232,
                'r': 0.189,
                'ci_95': (0.062, 0.310),
                'std_1': (0.125, 0.252)
            },
            'male_DZ': {
                'n': 403,
                'r': 0.088,
                'ci_95': (-0.010, 0.184),
                'std_1': (0.038, 0.137)
            },
            'female_MZ': {
                'n': 210,
                'r': 0.183,
                'ci_95': (0.049, 0.311),
                'std_1': (0.115, 0.249)
            },
            'female_DZ': {
                'n': 400,
                'r': 0.066,
                'ci_95': (-0.032, 0.163),
                'std_1': (0.016, 0.116)
            }
        }
    },
    'combined': {
        'male_MZ': {
            'n': 113 + 168 + 232,  # 
            'r': 0.214,
            'ci_95': (0.129, 0.295),
            'std_1': (0.171, 0.256)
        },
        'male_DZ': {
            'n': 186 + 306 + 403,  # 1112
            'r': 0.094,
            'ci_95': (0.029, 0.159),
            'std_1': (0.061, 0.128)
        },
        'female_MZ': {
            'n': 126 + 184 + 210,  # 604
            'r': 0.230,
            'ci_95': (0.147, 0.310),
            'std_1': (0.188, 0.272)
        },
        'female_DZ': {
            'n': 215 + 329 + 400,  # 1129
            'r': 0.078,
            'ci_95': (0.014, 0.141),
            'std_1': (0.045, 0.110)
        },
        'all_MZ': {
            'n': 113 + 168 + 232 +  126 + 184 + 210,  
            'r': 0.222,
            'ci_95': (0.163, 0.280),
            'std_1': (0.192, 0.252)
        },
        'all_DZ': {
            'n':186 + 306 + 403 + 215 + 329 + 400,  
            'r': 0.086,
            'ci_95': (0.040, 0.131),
            'std_1': (0.063, 0.109)
        }
    }
}

attach_h2_for_cohorts(herskind_study)
attach_h2_for_combined(herskind_study)

# Ljinquist study data
ljinquist_study = {
    'metadata': {
        'cohort_start': 1886,
        'cohort_end': 1925,
        'country': 'sweden',
        'filter_age': 37,
        'loghext_male': -2.4676,
        'loghext_std_male': 0.1809,
        'loghext_female': -2.5463,
        'loghext_std_female': 0.2276,
        'loghext': -2.5070,  # Average of male and female
        'loghext_std': 0.2043,  # Average of male and female std
        'mex_male': 3.43e-03,  # Fitted using Makeham-Gamma-Gompertz model
        'mex_std_male': 1.42e-03,
        'mex_female': 3.09e-03,  # Fitted using Makeham-Gamma-Gompertz model
        'mex_std_female': 1.35e-03,
        'mex': 3.25e-03,  # Average of male and female MGG fits
        'mex_std': 1.39e-03  # Average of male and female std
    },
    'combined': {
        'male_MZ': {
            'r': 0.330,
            'n': 1567,
            'ci_95': (0.260, 0.390),
            'std_1': (0.296, 0.363)
        },
        'male_DZ': {
            'r': 0.110,
            'n': 2814,
            'ci_95': (0.060, 0.150),
            'std_1': (0.087, 0.133)
        },
        'female_MZ': {
            'r': 0.280,
            'n': 1910,
            'ci_95': (0.220, 0.340),
            'std_1': (0.249, 0.310)
        },
        'female_DZ': {
            'r': 0.120,
            'n': 3589,
            'ci_95': (0.080, 0.150),
            'std_1': (0.102, 0.138)
        },
        'all_MZ': {
            'n': 3477,
            'r': 0.303,
            'ci_95': (0.272, 0.333),
            'std_1': (0.287, 0.318)
        },
        'all_DZ': {
            'n': 6403,
            'r': 0.116,
            'ci_95': (0.091, 0.140),
            'std_1': (0.103, 0.128)
        }
    }
}

attach_h2_for_combined(ljinquist_study)
# SATSA study data with updated log h_ext and revised h2 estimations
satsa_study = {
    'metadata': {
        'cohort_start': 1900,
        'cohort_end': 1935,
        'country': 'sweden',
        'filter_age': 50,
        # Convert the weighted average h_ext (1.48e-03) and bounds ([1.01e-03, 2.17e-03])
        # to log10 scale: log10(1.48e-03) ≈ -2.8297 and an average error of ≈ 0.1662.
        'mex': 2e-03,
        'mex_std': 0.0008,
    },
    'combined': {
        'h2': {
            'MZ_reared_apart': {
                'mean': 0.285,
                'std': 0.070,
                'ci_95': (0.144, 0.419)
            },
            'DZ_reared_apart': {
                'value': 0.278, 
                'std': 0.104, 
                'ci': (0.070, 0.476), 
                'label': r'$2r_{dz,apart}$', 
                'color': '#e74c3c'
            },
            'reared_together': {
                'value': 0.307, 
                'std': 0.159, 
                'ci': (-0.007, 0.612), 
                'label': r'$2(r_{mz}-r_{dz})$', 
                'color': '#2ecc71'
            },
            'weighted': {
                'mean': 0.288,
                'std': 0.055
            }
        }
    },
    'birth_cohorts': {
        '1900-1910': {
            'm_ex': 0.002953,
            'heritability': 0.1279,
            'heritability_std': 0.1029,
            'n': 380
        },
        '1910-1920': {
            'm_ex': 0.001905,
            'heritability': 0.3276,
            'heritability_std': 0.0810,
            'n': 493
        },
        '1920-1935': {
            'm_ex': 0.001044,
            'heritability': 0.3676,
            'heritability_std': 0.0873,
            'n': 366
        },
        '1900-1935': {
            'm_ex': 0.001982,
            'heritability': 0.2896,
            'heritability_std': 0.0539,
            'n': 1144
        }
    }
}

# USA study data
usa_study = {
    'metadata': {
        'cohort_start': 1873,
        'cohort_end': 1900,
        'country': 'usa',
        'filter_age': 20,
        'mex': 3.5e-03, 
        'mex_std': 0.0008,
    }
}



# Dictionary containing all studies
twin_studies = {
    'herskind': herskind_study,
    'ljinquist': ljinquist_study,
    'satsa': satsa_study,
    'usa': usa_study
}

def get_study_data(study_name):
    """
    Retrieve data for a specific twin study.

    Parameters:
    -----------
    study_name : str
        Name of the study ('herskind', 'ljinquist', or 'SATSA')

    Returns:
    --------
    dict
        Dictionary containing all data for the specified study
    """
    return twin_studies.get(study_name)

def get_combined_correlations(study_name):
    """
    Get the combined correlations across all cohorts for a specific study.

    Parameters:
    -----------
    study_name : str
        Name of the study ('herskind', 'ljinquist', or 'SATSA')

    Returns:
    --------
    dict
        Dictionary containing combined correlations for all twin types
    """
    study = twin_studies.get(study_name)
    if study:
        return study.get('combined')
    return None