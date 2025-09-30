### Heritability of intrinsic human lifespan is about 50% when confounding factors are addressed

This repository accompanies the paper:

- Title: Heritability of intrinsic human lifespan is about 50% when confounding factors are addressed
- Authors: Ben Shenhar, Glen Pridham, Thaís Lopes De Oliveira, Naveh Raz, Yifan Yang, Joris Deelen, Sara Hägg, Uri Alon*

### Heritability Analysis Toolkit

This repository implements two complementary approaches to study the heritability of lifespan from historic twin studies, adjusting for extrinsic mortality:
- SR model (stochastic resilience) simulations for lifespans and twin outcomes
- Modified Gamma–Gompertz–Makeham (MGG) modeling with Human Mortality Database (HMD) lifetables

It includes calibrated parameters, reproducible figures, and utilities to compute heritability proxies, relative survival probabilities, and hazard-based summaries.


### Repository Layout

- `src/` core modules (detailed below)
- `saved_data/` input data (HMD lifetables, cohorts, CSVs)
- `saved_results/` calibrated parameters and correlation matrices
- `notebooks/` figure-generation and analysis notebooks

### Data and results folders

- `saved_data/` (inputs)
  - Contains the raw inputs used by notebooks and modules:
    - Human Mortality Database (HMD) lifetable extracts (period and cohort) used by `src/HMD_lifetables.py`
    - Cohort death-rate tables and auxiliary CSVs used to compute hazards/survival and to plot empirical curves
    - Twin-study aggregates (e.g., Danish, Swedish, SATSA, and U.S. siblings datasets) used for comparisons

- `saved_results/` (derived artifacts)
  - Stores precomputed, heavy-to-recreate outputs that the notebooks read directly:
    - Correlation matrices (pickles) for SR and MGG models across extrinsic mortality grids and cutoff ages, per cohort
    - Calibrated parameter dictionaries for SR/MGG (`model_param_calibrations.py`) used to instantiate models in notebooks
    - Convenience result bundles for quick plotting and tables
  - Purpose: speed, reproducibility, and to avoid lengthy recomputation on clone. You can regenerate these with the calibration notebooks/workflows, but it may take time and CPU.

### Module-by-module guide (`src/`)

- `simulation.py`:
  - Defines `SimulationParams` and `SR_sim` for SR model simulations
  - Supports constant/time-dependent/agent-specific extrinsic hazard `h_ext`
  - Parallel chunking with `multiprocessing.Pool`; produces Kaplan–Meier and Nelson–Aalen estimators, hazards, and summary stats

- `sr_utils.py`:
  - Base human parameters (`karin_params`) and plotting colors/labels
  - `create_param_distribution_dict`: build MZ/DZ/uncorrelated parameter arrays with specified heterogeneity
  - `create_sr_simulation`: factory combining params with `SimulationParams` to return an `SR_sim`
  - `gompertz_hazard(t, m, a, b)`: helper for extrinsic hazard profiles

- `gamma_gompertz.py`:
  - `GammaGompertz` class: fit to HMD data, custom hazard arrays, or KM estimators
  - Sampling death times with parameter heterogeneity; twin death tables for MZ/DZ
  - Hazard/survival plotting utilities and optimization helpers (e.g., KS-constrained fitting)

- `HMD_lifetables.py`:
  - `HMD` class to load HMD lifetables (period/cohort) and compute hazard, survival, death distributions
  - Plot helpers: survival, hazard, Gompertz/GGM fits, geometric-mean hazard curves, Makeham term trends

- `HMD_death_rates.py`:
  - Lightweight reader for cohort death-rate tables from `saved_data/`
  - Quick helpers to compute survival/hazard from age X and plot

- `twin_analysis.py`:
  - Build twin death tables from simulated death times (including flags for extrinsic deaths)
  - Filtering helpers (age thresholds, parameter-conditioned subsets)

- `correlation_analysis.py`:
  - Pearson/intraclass correlations and phi coefficients
  - Binned variance decomposition proxy for heritability

- `survival_analysis.py`:
  - Unconditional/conditional survival past age X; relative survival probabilities (RSP)
  - Excess survival exponential fits; hazard for twins of people past age X

- `plotting.py`:
  - Visualization helpers for SR simulations: survival, hazards, path summaries, hazard fits

- `hetero_plotting.py`:
  - Higher-level figures for relative survival probabilities, error bars (bootstrap/delta/Wilson), and twin correlations

- `bootstrap.py`:
  - Bootstrap/delta/Wilson interval utilities for relative survival probability curves

- `model_free_twins.py`:
  - Model-free partitioning of total mortality into intrinsic/extrinsic using `HMD`
  - Generate correlated twin lifespans and visualize correlations without invoking the SR/MGG models

### Notebook-by-notebook guide (`notebooks/`)

- `Fig1.ipynb`:
  - Demonstrates extrinsic plateaus and Gompertz fits on Danish cohorts
  - Illustrative geometry for high vs low twin correlations and how distributions shift
  - Toy Gompertz sampling to show mean/variance effects and correlation changes

- `Fig2.ipynb`:
  - Builds twin scatter plots under SR with/without extrinsic mortality
  - Compares empirical twin heritability with SR/MGG predictions vs extrinsic mortality
  - Adds SATSA/Sweden/Danish cohort comparisons and a heritability forest plot

- `Fig3.ipynb`:
  - Consolidated multi-panel figure: SATSA cohort trends, heritability vs extrinsic mortality, US siblings excess survival odds
  - Uses precomputed correlation matrices in `saved_results/` and calibrated params in `saved_results/model_param_calibrations.py`

- `increasing_extrinsic.ipynb`:
  - Tests increasing extrinsic mortality with age and parameter-specific profiles
  - Shows that parameter-specific rising extrinsic hazards can replicate constant-hazard outcomes in SR

- `compression_of_morbidity.ipynb`:
  - Compares historical vs modern Danish period hazards
  - Adjusts SR threshold (`Xc`) to capture modern hazard compression; explores correlation as a function of threshold factor

- `model_free.ipynb`:
  - Model-free mortality partition using `model_free_twins.py`
  - Simulates twin correlations vs target intrinsic r and shows mortality distributions

- `Excess_survival_odds.ipynb`:
  - Excess survival odds (RSP-1) for Danish twins and U.S. siblings of centenarians
  - Compares SR and MGG model predictions to data and shows mortality alongside empirical curves

- `results_tables.ipynb`:
  - Prints calibrated SR/MGG parameters and tabulates h² values at negligible extrinsic mortality

- `examples_model_sim_basics.ipynb`:
  - Minimal SR and MGG demonstrations: sampling, survival/hazard plots, basic fits

All notebooks start with a common setup cell that:
- Locates the project root and adds `src/` to `sys.path`
- Sets deterministic seeds and uses Arial font


### Reproducibility

- Seeds are set via `PYTHONHASHSEED`, `random.seed`, and `numpy.random.seed`
- Many simulations run with large `n` and may use parallel workers; set smaller `n` for quick runs

### Troubleshooting

- If a notebook cannot find data, verify the files under `saved_data/`
- If fonts look different, ensure Arial is installed or adjust matplotlib rcParams

