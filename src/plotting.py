"""
General plotting helpers for SR simulations.

Includes survival curves (absolute/scaled), hazards, path visualizations,
and hazard fits (exponential and power-law).
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from lifelines import KaplanMeierFitter
from scipy.optimize import curve_fit


class SR_plotting:
    def __init__(self, simulation):
        self.simulation = simulation

    def plot_some_paths(self, n, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        for i in range(n):
            random_row_index = np.random.choice(self.simulation.paths.shape[0])
            ax.plot(self.simulation.tspan, self.simulation.paths[random_row_index])
        ax.axhline(y=self.simulation.params.Xc[0], color='grey', linestyle=':')
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        ax.set_xlabel(f'time [{self.simulation.params.units}]')
        ax.set_ylabel('X')

    def plot_survival(self, scaled=False, with_median=False, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        if scaled:
            ax.plot(self.simulation.tspan_survival / self.simulation.median_t, self.simulation.survival, **kwargs)
            ax.set_xlabel(r'$\frac{t}{t_m}$')
            ax.set_xlim(0, 2)
        else:
            self.simulation.kmf.plot_survival_function(ax=ax, **kwargs)
        ax.legend()
        ax.set_ylabel('Survival')
        if with_median:
            ax.axhline(y=0.5, color='grey', linestyle=':')
            ax.axvline(x=self.simulation.params.median_t, color='grey', linestyle=':')
            ax.set_xticks(np.append(ax.get_xticks()[ax.get_xticks() != ax.get_xticks()[np.argmin(np.abs(ax.get_xticks() - self.simulation.median_t))]], self.simulation.median_t))
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)


    def plot_survival_from_age_X(self, age_X=40, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        death_times = self.simulation.death_times
        # take only death_times that are greater than from_age
        death_times = death_times[death_times > age_X]
        event_observed = np.where(death_times == np.inf, 0, 1)
        kmf = KaplanMeierFitter()
        kmf.fit(death_times, event_observed, timeline=self.simulation.tspan_survival)
        kmf.plot_survival_function(ax=ax, **kwargs)
        ax.set_xlabel('Age [{}]'.format(self.simulation.params.units))
        ax.set_ylabel('Survival Probability')

    def plot_hazard(self, dt=1, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        self.simulation.naf.plot_hazard(bandwidth=3, ax=ax, **kwargs)
        ax.set_yscale('log')
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        ax.set_xlabel(f'time [{self.simulation.params.units}]')
        ax.set_ylabel(f'Mortality rate [1/{self.simulation.params.units}]')



    def plot_meanX(self, ax=None, x1line=False, alive_only=False, tspan_scale=1, **kwargs):
        """
        Plots the mean of X over time. Optionally filters out "dead" paths at each time step
        to compute a live-only average. This version uses vectorized operations for speed."""
        if ax is None:
            fig, ax = plt.subplots()

        if alive_only:
            # Convert death_times into shape (num_paths, 1)
            death_times = self.simulation.death_times[:, None]   # shape: (N, 1)
            tspan = self.simulation.tspan                        # shape: (T, )

            # Create a boolean mask of shape (N, T), indicating if each path is alive at each time
            alive_mask = death_times > tspan                     # shape: (N, T)

            # Convert the boolean mask to float, so we can multiply directly with paths
            alive_mask_float = alive_mask.astype(float)          # shape: (N, T)

            # Sum of the alive paths for each time step
            sum_alive = np.sum(self.simulation.paths * alive_mask_float, axis=0)  # shape: (T, )
            count_alive = np.sum(alive_mask_float, axis=0)                         # shape: (T, )

            # Compute mean only where count_alive != 0, else 0 (or NaN) if no paths are alive
            mean_X = np.divide(
                sum_alive,
                count_alive,
                out=np.zeros_like(sum_alive, dtype=float),
                where=(count_alive > 0)
            )

            ax.plot(tspan * tspan_scale, mean_X, **kwargs)

        else:
            # Use the precomputed mean_X if we don't need live-only filtering
            ax.plot(self.simulation.tspan * tspan_scale, self.simulation.mean_X, **kwargs)

        if x1line:
            ax.axhline(y=1, color='grey', linestyle=':')

        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        ax.set_xlabel(f'time [{self.simulation.params.units}]')
        ax.set_ylabel('X')

    def plot_STD(self, ax=None, alive_only=False, tspan_scale=1, **kwargs):
        """
        Plots the standard deviation of X over time. Optionally filters out "dead" paths
        at each time step to compute a live-only standard deviation.
        """
        if ax is None:
            fig, ax = plt.subplots()

        if alive_only:
            # Convert death_times into shape (num_paths, 1)
            death_times = self.simulation.death_times[:, None]   # shape: (N, 1)
            tspan = self.simulation.tspan                        # shape: (T, )

            # Create a boolean mask of shape (N, T), indicating if each path is alive at each time
            alive_mask = death_times > tspan                     # shape: (N, T)

            # For each time step, calculate std dev of alive paths
            std_X = np.zeros_like(tspan, dtype=float)
            for i, t in enumerate(tspan):
                alive_at_t = alive_mask[:, i]
                paths_at_t = self.simulation.paths[:, i]
                
                # Calculate std for alive paths, ensuring we have at least 1 path
                if np.sum(alive_at_t) > 1:  # Need at least 2 samples for std dev
                    std_X[i] = np.std(paths_at_t[alive_at_t], ddof=1)
                # If only 1 or 0 paths, std is 0

            ax.plot(tspan * tspan_scale, std_X, **kwargs)
        else:
            # Use the precomputed std_X
            ax.plot(self.simulation.tspan * tspan_scale, self.simulation.std_X, **kwargs)

        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        ax.set_xlabel(f'time [{self.simulation.params.units}]')
        ax.set_ylabel('STD(X)')

    def plot_CV(self, ax=None, alive_only=False, tspan_scale=1, **kwargs):
        """
        Plots the coefficient of variation (CV = STD/Mean) of X over time. 
        Optionally filters out "dead" paths at each time step.
        """
        if ax is None:
            fig, ax = plt.subplots()

        if alive_only:
            # Convert death_times into shape (num_paths, 1)
            death_times = self.simulation.death_times[:, None]   # shape: (N, 1)
            tspan = self.simulation.tspan                        # shape: (T, )

            # Create a boolean mask of shape (N, T), indicating if each path is alive at each time
            alive_mask = death_times > tspan                     # shape: (N, T)

            # For each time step, calculate CV of alive paths
            cv_X = np.zeros_like(tspan, dtype=float)
            for i, t in enumerate(tspan):
                alive_at_t = alive_mask[:, i]
                paths_at_t = self.simulation.paths[:, i]
                
                # Calculate CV for alive paths if there are enough samples
                if np.sum(alive_at_t) > 1:
                    alive_paths = paths_at_t[alive_at_t]
                    mean_alive = np.mean(alive_paths)
                    if mean_alive > 0:  # Avoid division by zero
                        std_alive = np.std(alive_paths, ddof=1)
                        cv_X[i] = std_alive / mean_alive

            ax.plot(tspan * tspan_scale, cv_X, **kwargs)
        else:
            # Use the precomputed cv_X
            ax.plot(self.simulation.tspan * tspan_scale, self.simulation.cv_X, **kwargs)

        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        ax.set_xlabel(f'time [{self.simulation.params.units}]')
        ax.set_ylabel('CV(X)')

    def plot_pdf_at_t(self, t, hist=False, bins=50, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        t_indice = np.where(self.simulation.tspan == t)[0][0]
        kde = self.simulation.paths[np.where(self.simulation.death_times > t)[0], t_indice]
        if hist:
            sns.histplot(kde, bins=bins, stat='count', alpha=0.5, kde=True, label=f't = {t} {self.simulation.params.units}', ax=ax)
        else:
            sns.histplot(kde, bins=bins, stat='density', alpha=0.5, label=f't = {t} {self.simulation.params.units}', ax=ax)
        ax.legend()

    def plot_go_ww_pdf_at_t(self, t, normalized=False):
        pdf, closest_t = self.simulation.pdf_at_t(t)
        if normalized:
            pdf = pdf / self.simulation.survival_area
        plt.stairs(pdf, edges=self.simulation.bins, fill=True, alpha=0.5, label=f't = {closest_t} {self.simulation.params.units}')
        plt.title(f'pdf at time t = {closest_t} {self.simulation.params.units}')
        

    ############# FITTING ############
    def fit_exponential_hazard(self, t_start=60, t_end=80):
        """
        Fits an exponential function A * exp(B * ages) to the hazard data.
        Only considers data from ages_start to ages_end.
        """
        # Filter for the age range
        filtered_indices = (self.simulation.tspan_hazard >= t_start) & (self.simulation.tspan_hazard <= t_end)
        ages_filtered = self.simulation.tspan_hazard[filtered_indices]
        hazards_filtered = np.ravel(self.simulation.hazard)[filtered_indices]

        # Define the exponential function to fit
        def exp_func(ages, A, B):
            return np.log(A) + B*ages

        # Perform the curve fitting

        params, _ = curve_fit(exp_func, ages_filtered, np.log(hazards_filtered), maxfev=10000)

        # Extract the coefficients A and B
        A, B = params
        return A, B

    def plot_hazard_exp_fit(self, t_start=40, t_end=80 , ax=None, show_fit_bounds = False, **kwargs):
        if ax == None:
            fig, ax = plt.subplots()

        # Get the parameters A and B from the fit
        A, B = self.fit_exponential_hazard(t_start=t_start, t_end=t_end)

        # Generate ages for the fit line
        ages_fit = np.linspace(0, self.simulation.tspan_hazard[-1], 100)
        # Calculate the hazard values using the exponential fit
        hazards_fit = A * np.exp(B * ages_fit)
        
        label = f'${A:.2e} e^{{\\frac{{t}}{{{1/B:.2f}}}}}$'
        if show_fit_bounds == True:
            label = f'${A:.2e} e^{{\\frac{{t}}{{{1/B:.2f}}}}}$ from {t_start} to {t_end}'

        ax.plot(ages_fit, hazards_fit, label=label, linestyle='--' , **kwargs)
        
        

    def fit_power_law_hazard(self, t_start=40, t_end=80):
      """
      Fits a power-law function A * ages^B to the hazard data by performing
      a linear fit on log10(ages) and log10(hazards).
      Only considers data from ages_start to ages_end.
      """
      # Filter for the age range
      filtered_indices = (self.simulation.tspan_hazard >= t_start) & (self.simulation.tspan_hazard <= t_end)
      ages_filtered = self.simulation.tspan_hazard[filtered_indices]
      hazards_filtered = np.ravel(self.simulation.hazard)[filtered_indices]

      # Take the logarithm of the data
      log_ages = np.log10(ages_filtered)
      log_hazards = np.log10(hazards_filtered)

      # Define the linear function to fit
      def linear_func(x, a, m):
          return a + m * x

      # Perform the curve fitting
      params, _ = curve_fit(linear_func, log_ages, log_hazards)

      # Extract the coefficients a and m
      a, m = params

      # Convert back to power-law parameters A and B
      A = 10**a
      B = m
      return A, B

    def plot_hazard_power_law_fit(self, t_start=40, t_end=80, ax=None, show_fit_bounds=False, **kwargs):
      if ax is None:
          fig, ax = plt.subplots()

      # Get the parameters A and B from the fit
      A, B = self.fit_power_law_hazard(t_start=t_start, t_end=t_end)

      # Generate ages for the fit line
      ages_fit = np.linspace(t_start, t_end, 100)
      # Calculate the hazard values using the power-law fit
      hazards_fit = A * np.power(ages_fit, B)

      label = f'${A:.2e} t^{{{B:.2f}}}$'
      if show_fit_bounds:
          label = f'${A:.2e} t^{{{B:.2f}}}$ from {t_start} to {t_end}'

      ax.plot(ages_fit, hazards_fit, label=label, linestyle='--', **kwargs)
      ax.set_xlabel('Age')
      ax.set_ylabel('Hazard')
      ax.legend()