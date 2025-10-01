### Heritability of intrinsic human lifespan is about 50% when confounding factors are addressed

This repository accompanies the paper:

- **Title**: Heritability of intrinsic human lifespan is about 50% when confounding factors are addressed
- **Authors**: Ben Shenhar, Glen Pridham, Thaís Lopes De Oliveira, Naveh Raz, Yifan Yang, Joris Deelen, Sara Hägg, Uri Alon*

## Research Context

This repo investigates how extrinsic mortality affects estimates of lifespan heritability. Traditional twin studies often find heritability estimates around 20-30%, but this work shows that when extrinsic mortality is properly accounted for, the intrinsic heritability of human lifespan is approximately 50%. The analysis uses both the Saturating-Removal (SR) model and Makeham Gamma-Gompertz (MGG) model to simulate different genetic groups with distinct lifespan distributions

## Installation

### Prerequisites
- Python 3.8 or higher
- All data files are included in the repository

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd extrinsic-mortality-code
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify installation by running a simple example (see Quick Start below).

## Quick Start

Here's a minimal example to get started:

```python
# Basic SR model simulation
from src.sr_utils import karin_params, create_sr_simulation
from src.plotting import SR_plotting
import matplotlib.pyplot as plt

# Create and run simulation
sim = create_sr_simulation(params_dict=karin_params, n=10000, tmax=120, save_times=0.5, parallel=False)
plotter = SR_plotting(sim)

# Plot results
ax = plotter.plot_survival(color='black', label='Survival')
plotter.plot_hazard(ax=None, color='red', label='Hazard')
plt.show()
```

For more examples, see `notebooks/examples_model_sim_basics.ipynb`.

## Extended Examples: SR and MGG

The snippets below show how to build SR parameters from scratch, add twin-style heterogeneity (MZ and DZ), run simulations, and plot results. A corresponding `GammaGompertz` example is included as well.

### SR model: build params, add MZ/DZ heterogeneity, simulate, plot

```python
import numpy as np
import matplotlib.pyplot as plt

from src.sr_utils import create_param_distribution_dict, create_sr_simulation
from src.plotting import SR_plotting

# 1) Baseline SR parameters (single-value arrays as expected by the factory)
baseline_params = {
    'eta':     np.array([0.00135 * 365]),  # production (per year)
    'beta':    np.array([0.15    * 365]),  # removal (per year)
    'kappa':   np.array([0.5     ]),       # half-saturation
    'epsilon': np.array([0.142   * 365]),  # noise (per year)
    'Xc':      np.array([17      ])        # threshold
}

# 2) Create MZ heterogeneity by distributing a parameter (e.g., Xc)
#    - family='MZ' duplicates the same draw for both twins in each pair
mz_params = create_param_distribution_dict(
    params='Xc',
    std=0.08,
    n=40000,                 # total individuals (multiple of 2)
    dist_type='gaussian',
    params_dict=baseline_params,
    family='MZ'
)

# 3) Create DZ heterogeneity (correlated, rho≈0.5, for gaussian)
dz_params = create_param_distribution_dict(
    params='Xc',
    std=0.08,
    n=40000,
    dist_type='gaussian',
    params_dict=baseline_params,
    family='DZ'
)

# 4) Run SR simulations (set tmax, dt, etc.)
sim_mz = create_sr_simulation(params_dict=mz_params, n=len(mz_params['Xc']), tmax=120, dt=0.25, save_times=0.5, parallel=False)
sim_dz = create_sr_simulation(params_dict=dz_params, n=len(dz_params['Xc']), tmax=120, dt=0.25, save_times=0.5, parallel=False)

# 5) Plot survival and hazard for MZ vs DZ
plotter_mz = SR_plotting(sim_mz)
plotter_dz = SR_plotting(sim_dz)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plotter_mz.plot_survival(ax=ax1, color='#6a3d9a', label='MZ')
plotter_dz.plot_survival(ax=ax1, color='#1f78b4', label='DZ')
ax1.set_title('SR: Survival')
ax1.legend()

plotter_mz.plot_hazard(ax=ax2, color='#6a3d9a', label='MZ')
plotter_dz.plot_hazard(ax=ax2, color='#1f78b4', label='DZ')
ax2.set_title('SR: Hazard (NAF-smoothed)')
ax2.legend()
plt.tight_layout()
plt.show()
```

Notes:
- Use `params` to select which SR parameter(s) to vary (`'eta'|'beta'|'kappa'|'epsilon'|'Xc'|'h_ext'`).
- `family='MZ'` replicates the same draw for a pair; `family='DZ'` draws correlated values (rho≈0.5; gaussian case).
- You can also pass a constant extrinsic hazard via `h_ext` to `create_sr_simulation(..., h_ext=3e-3)`.

### GammaGompertz (MGG): set params, simulate with/without heterogeneity, plot

```python
import numpy as np
import matplotlib.pyplot as plt

from src.gamma_gompertz import GammaGompertz
from lifelines import NelsonAalenFitter, KaplanMeierFitter

# 1) Define and set MGG parameters directly
gg = GammaGompertz()
gg.set_params({'a': 1.0e-5, 'b': 0.12, 'c': 30.0, 'm': 2.0e-3})

# 2) Sample death times without heterogeneity (homogeneous parameters)
n = 20000
death_times_hom = gg.sample_death_times(n=n, dt=0.5)

# 3) Sample death times with heterogeneity in parameter 'b'
#    - std is relative to the base value (drawn ~ Normal(mean=b, sd=std*b))
#    - coupled_ab=True couples 'a' and 'b' as in the codebase (a -> a^r when b -> b*r)
death_times_het, b_vals = gg.sample_death_times_with_random_param(
    n=n,
    param_name='b',
    dt=0.5,
    std=0.20,
    coupled_ab=True
)

# 4) Build nonparametric estimators for both cases
timeline = np.arange(0, 120, 0.5)

km_hom = KaplanMeierFitter().fit(death_times_hom, event_observed=np.ones(len(death_times_hom)))
naf_hom = NelsonAalenFitter().fit(death_times_hom, event_observed=np.ones(len(death_times_hom)), timeline=timeline)

km_het = KaplanMeierFitter().fit(death_times_het, event_observed=np.ones(len(death_times_het)))
naf_het = NelsonAalenFitter().fit(death_times_het, event_observed=np.ones(len(death_times_het)), timeline=timeline)

# 5) Plot model vs nonparametric, and compare homogeneous vs heterogeneous
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Survival
ages = np.linspace(0, 120, 240)
_, S_model = gg.survival_function(ages)
ax1.plot(ages, S_model, color='black', linewidth=2.0, label='MGG model')
km_hom.plot_survival_function(ax=ax1, ci_show=False, color='#377eb8', label='KM (hom)')
km_het.plot_survival_function(ax=ax1, ci_show=False, color='#4daf4a', label='KM (het, b~N)')
ax1.set_title('MGG: Survival')
ax1.legend()

# Hazard (log-scale)
_, haz_model = gg.calculate_hazard(ages)
ax2.plot(ages, haz_model, color='black', linewidth=2.0, label='MGG model')
naf_hom.plot_hazard(ax=ax2, bandwidth=3, color='#e41a1c', label='NAF (hom)')
naf_het.plot_hazard(ax=ax2, bandwidth=3, color='#984ea3', label='NAF (het, b~N)')
ax2.set_yscale('log')
ax2.set_title('MGG: Hazard')
ax2.legend()

plt.tight_layout()
plt.show()
```

## Repository Layout

- `src/` - Core modules (detailed below)
- `saved_data/` - Input data (HMD lifetables, cohorts, CSVs)
- `saved_results/` - Calibrated parameters and correlation matrices
- `notebooks/` - Figure-generation and analysis notebooks

## Data and Results Folders

### `saved_data/` (inputs)
Contains the raw inputs used by notebooks and modules, including:
- Human Mortality Database (HMD) lifetable extracts (period and cohort) used by `src/HMD_lifetables.py`
- Cohort death-rate tables and auxiliary CSVs used to compute hazards/survival and to plot empirical curves
- Twin-study aggregates (e.g., Danish, Swedish, SATSA, and U.S. siblings datasets) used for comparisons

### `saved_results/` (derived artifacts)
Stores precomputed, heavy-to-recreate outputs that the notebooks read directly, including:
- Correlation matrices (pickles) for SR and MGG models across extrinsic mortality grids and cutoff ages, per cohort
- Calibrated parameter dictionaries for SR/MGG (`model_param_calibrations.py`) used to instantiate models in notebooks
- Convenience result bundles for quick plotting and tables

**Purpose**: Speed, reproducibility, and to avoid lengthy recomputation on clone. You can regenerate these with the calibration notebooks/workflows, but it may take time and CPU.

## Module-by-Module Guide (`src/`) by Theme

### Models
- **`simulation.py`**:
  - Defines `SimulationParams` and `SR_sim` for SR model simulations
  - Supports constant/time-dependent/agent-specific extrinsic hazard `h_ext`
  - Parallel chunking with `multiprocessing.Pool`; produces Kaplan–Meier and Nelson–Aalen estimators, hazards, and summary stats
- **`gamma_gompertz.py`**:
  - `GammaGompertz` class: fit to HMD data, custom hazard arrays, or KM estimators
  - Sample death times (with/without heterogeneity), create twin death tables, and plot survival/hazard

### Data (HMD)
- **`HMD_lifetables.py`**:
  - `HMD` class to load HMD lifetables (period/cohort) and compute hazard, survival, death distributions
  - Plot helpers: survival, hazard, Gompertz/GGM fits, geometric-mean hazard curves, Makeham term trends
- **`HMD_death_rates.py`**:
  - Lightweight reader for cohort death-rate tables from `saved_data/`
  - Quick helpers to compute survival/hazard from age X and plot

### Analysis
- **`twin_analysis.py`**:
  - Build twin death tables from simulated death times (including flags for extrinsic deaths)
  - Filtering helpers (age thresholds, parameter-conditioned subsets)
- **`correlation_analysis.py`**:
  - Pearson/intraclass correlations and phi coefficients
  - Binned variance decomposition proxy for heritability
- **`survival_analysis.py`**:
  - Unconditional/conditional survival past age X; relative survival probabilities (RSP)
  - Excess survival exponential fits; hazard for twins of people past age X
- **`model_free_twins.py`**:
  - Model-free partitioning of total mortality into intrinsic/extrinsic using `HMD`
  - Generate correlated twin lifespans and visualize correlations without invoking the SR/MGG models

### Plotting
- **`plotting.py`**:
  - Visualization helpers for SR simulations: survival, hazards, path summaries, hazard fits
- **`hetero_plotting.py`**:
  - Higher-level figures for relative survival probabilities, error bars (bootstrap/delta/Wilson), and twin correlations

### Utilities
- **`sr_utils.py`**:
  - Base human parameters (`karin_params`) and plotting colors/labels
  - `create_param_distribution_dict`: build MZ/DZ/uncorrelated parameter arrays with specified heterogeneity
  - `create_sr_simulation`: factory combining params with `SimulationParams` to return an `SR_sim`
  - `gompertz_hazard(t, m, a, b)`: helper for extrinsic hazard profiles
- **`bootstrap.py`**:
  - Bootstrap/delta/Wilson interval utilities for relative survival probability curves

## Notebook-by-Notebook Guide (`notebooks/`)

- **`Fig1.ipynb`**:
  - Demonstrates extrinsic plateaus and Gompertz fits on Danish cohorts
  - Illustrative geometry for high vs low twin correlations and how distributions shift
  - Toy Gompertz sampling to show mean/variance effects and correlation changes

- **`Fig2.ipynb`**:
  - Builds twin scatter plots under SR with/without extrinsic mortality
  - Compares empirical twin heritability with SR/MGG predictions vs extrinsic mortality
  - Adds SATSA/Sweden/Danish cohort comparisons and a heritability forest plot

- **`Fig3.ipynb`**:
  - Consolidated multi-panel figure: SATSA cohort trends, heritability vs extrinsic mortality, US siblings excess survival odds
  - Uses precomputed correlation matrices in `saved_results/` and calibrated params in `saved_results/model_param_calibrations.py`

- **`increasing_extrinsic.ipynb`**:
  - Tests increasing extrinsic mortality with age and parameter-specific profiles
  - Shows that parameter-specific rising extrinsic hazards can replicate constant-hazard outcomes in SR

- **`compression_of_morbidity.ipynb`**:
  - Compares historical vs modern Danish period hazards
  - Adjusts SR threshold (`Xc`) to capture modern hazard compression; explores correlation as a function of threshold factor

- **`model_free.ipynb`**:
  - Model-free mortality partition using `model_free_twins.py`
  - Simulates twin correlations vs target intrinsic r and shows mortality distributions

- **`Excess_survival_odds.ipynb`**:
  - Excess survival odds (RSP-1) for Danish twins and U.S. siblings of centenarians
  - Compares SR and MGG model predictions to data and shows mortality alongside empirical curves

- **`results_tables.ipynb`**:
  - Prints calibrated SR/MGG parameters and tabulates h² values at negligible extrinsic mortality

- **`examples_model_sim_basics.ipynb`**:
  - Minimal SR and MGG demonstrations: sampling, survival/hazard plots, basic fits

- **`extrinsic_mortality_calibrations.ipynb`**:
  - Calculates extrinsic mortality levels (m_ex) for each study (Danish, Swedish, SATSA)

All notebooks start with a common setup cell that:
- Locates the project root and adds `src/` to `sys.path`
- Sets deterministic seeds and uses Arial font

## Reproducibility

- Seeds are set via `PYTHONHASHSEED`, `random.seed`, and `numpy.random.seed`
- Many simulations run with large `n` and may use parallel workers; set smaller `n` for quick runs

## Citation

If you use this code or data in your research, please cite:

**Paper** (when published):
```
Shenhar, B., Pridham, G., De Oliveira, T. L., Raz, N., Yang, Y., Deelen, J., Hägg, S., & Alon, U. (Year). Heritability of intrinsic human lifespan is about 50% when confounding factors are addressed. [Journal Name], [Volume], [Pages]. DOI: [DOI]
```

**Repository**:
```
Shenhar, B., Pridham, G., De Oliveira, T. L., Raz, N., Yang, Y., Deelen, J., Hägg, S., & Alon, U. (Year). Heritability of intrinsic human lifespan is about 50% when confounding factors are addressed [Code]. GitHub. https://github.com/[username]/[repository-name]
```

**Data Attribution**:
- Human Mortality Database (HMD): [Human Mortality Database. University of California, Berkeley (USA) and Max Planck Institute for Demographic Research (Germany). Available at www.mortality.org or www.humanmortality.de]
- Twin study data: Various sources as detailed in the paper

## Troubleshooting

- If a notebook cannot find data, verify the files under `saved_data/`
- If fonts look different, ensure Arial is installed or adjust matplotlib rcParams
- For installation issues, ensure you have Python 3.8+ and all dependencies from `requirements.txt`

