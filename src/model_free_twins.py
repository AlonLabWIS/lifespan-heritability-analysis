"""
Model-free twin simulations and mortality partitioning utilities.

Provides a lightweight framework to partition total mortality into
intrinsic/extrinsic components directly from HMD lifetables and to
simulate twin lifespans with a specified intrinsic correlation.

Key capabilities:
- Build total/intrinsic/extrinsic distributions from `src.HMD_lifetables.HMD`
- Generate correlated twin lifespans with mixture of intrinsic/extrinsic deaths
- Plot correlations and compare with empirical twin studies
"""
import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz
from scipy.stats import norm
from src.HMD_lifetables import HMD
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

class ModelFreeTwins:
    def __init__(self, country='dan', gender='both', years=None, 
                 age_start=15, age_end=110,
                 fit_start=70, fit_end=85, data_type='cohort',
                 intrinsic_to_total_ratio=0.85, hmd_sim=None):
        """
        Initialize twin data generator with mortality data from HMD
        
        Parameters:
        country : str 'dan', 'swe', 'nor', 'fin', 'deu', 'ita', 'gbr', 'usa'
        years : array-like of years to use (default: 1950-2019)
        age_start : starting age for distributions
        age_end : maximum age to consider
        fit_start : age to start Gompertz fit for intrinsic mortality
        fit_end : age to end Gompertz fit for intrinsic mortality
        intrinsic_to_total_ratio : float between 0 and 1, ratio of intrinsic to total hazard
        hmd_sim : HMD object, optional
            If provided, will use this HMD instance instead of creating a new one
        """
        self.age_start = age_start
        self.age_end = age_end
        self.fit_start = fit_start
        self.fit_end = fit_end
        self.years = years if years is not None else np.arange(1870, 1900)
        self.intrinsic_to_total_ratio = intrinsic_to_total_ratio
        
        # Initialize distributions
        if hmd_sim is not None:
            self.hmd_sim = hmd_sim
            self.country = hmd_sim.country
            self.gender = hmd_sim.gender
            self.data_type = data_type
        else:
            self.country = country
            self.gender = gender
            self.data_type = data_type
            self.hmd_sim = HMD(country=country, gender=gender, data_type=data_type)
            
        self._compute_distributions()
        
    def set_distributions(self, total_distribution=None, ages=None, intrinsic_distribution=None, 
                         extrinsic_distribution=None, partition_mortality=True):
        """
        Set custom mortality distributions after initialization.
        
        Parameters:
        -----------
        total_distribution : array-like, optional
            Custom total mortality distribution
        ages : array-like, optional
            Age points corresponding to distributions
        intrinsic_distribution : array-like, optional
            Custom intrinsic mortality distribution
        extrinsic_distribution : array-like, optional
            Custom extrinsic mortality distribution
        partition_mortality : bool, optional
            If True and only total_distribution provided, automatically partition into 
            intrinsic and extrinsic components using Gompertz fitting (default: True)
        """
        if total_distribution is not None and ages is not None:
            self.ages = np.asarray(ages)
            self.total_distribution = np.asarray(total_distribution)
            
            # Calculate total hazard
            _, self.total_hazard = self.hazard_from_distribution(
                self.total_distribution, self.ages
            )
            
            if partition_mortality:
                # Partition into intrinsic and extrinsic as before
                min_hazard = self.get_min_hazard(self.ages, self.total_hazard, 
                                            ages_start=self.fit_start, ages_end=self.fit_end)
                self.intrinsic_hazard = self.intrinsic_to_total_ratio * min_hazard
                self.extrinsic_hazard = self.total_hazard - self.intrinsic_hazard
                
                # Convert hazards to distributions
                cum_hazard = cumtrapz(self.intrinsic_hazard, self.ages, initial=0)
                S = np.exp(-cum_hazard)
                self.intrinsic_distribution = self.intrinsic_hazard * S
                
                self.extrinsic_distribution = self.extrinsic_hazard * self.survival_from_distribution(
                    self.total_distribution, self.ages
                )[1]
        
        # Allow manual setting of intrinsic/extrinsic distributions
        if intrinsic_distribution is not None:
            self.intrinsic_distribution = np.asarray(intrinsic_distribution)
        if extrinsic_distribution is not None:
            self.extrinsic_distribution = np.asarray(extrinsic_distribution)
        
    def _compute_distributions(self):
        """Calculate all required distributions from HMD data"""
        # Total mortality distribution
        self.ages, self.total_distribution = self.lifespan_distribution(
            self.years, self.hmd_sim, self.age_start
        )
        
        # Calculate total hazard
        _, self.total_hazard = self.hazard_from_distribution(
            self.total_distribution, self.ages
        )
        
        # Calculate intrinsic hazard as fraction of total
        min_hazard = self.get_min_hazard(self.ages, self.total_hazard, 
                                        ages_start=self.fit_start, ages_end=self.fit_end)
        self.intrinsic_hazard = self.intrinsic_to_total_ratio * min_hazard
        
        # Calculate extrinsic hazard as remainder
        self.extrinsic_hazard = self.total_hazard - self.intrinsic_hazard
        
        # Convert hazards back to distributions
        cum_hazard = cumtrapz(self.intrinsic_hazard, self.ages, initial=0)
        S = np.exp(-cum_hazard)
        self.intrinsic_distribution = self.intrinsic_hazard * self.survival_from_distribution(
            self.total_distribution, self.ages
        )[1]
        
        self.extrinsic_distribution = self.extrinsic_hazard * self.survival_from_distribution(
            self.total_distribution, self.ages
        )[1]

        # Calculate death probabilities
        total_area = np.trapz(self.total_distribution[:-1], self.ages[:-1])
        intrinsic_area = np.trapz(self.intrinsic_distribution[:-1], self.ages[:-1])
        extrinsic_area = np.trapz(self.extrinsic_distribution[:-1], self.ages[:-1])
        
        self.extrinsic_death_prop = extrinsic_area / total_area
        self.intrinsic_death_prop = intrinsic_area / total_area

    def lifespan_distribution(self, years, hmd_sim, age_start=15):
        """Identical to original function but as class method"""
        ages, distributions = hmd_sim.get_lifespan_distribution(year=years, age_start=age_start)
        mean_dist = np.mean(distributions, axis=1)
        return ages, mean_dist
    
    
    @staticmethod
    def hazard_from_distribution(P, t):
        """Static version of original hazard_from_distribution"""
        import numpy as np
        from scipy.integrate import cumtrapz

        # Convert input to numpy arrays of type float for numerical operations.
        P = np.asarray(P, dtype=float)
        t = np.asarray(t, dtype=float)

        # Calculate the cumulative distribution (CDF) via trapezoidal integration.
        cum_pdf = cumtrapz(P, t, initial=0)

        # The total probability mass is assumed to be cum_pdf[-1].
        total_mass = cum_pdf[-1]

        # The survival function is S(t) = total_mass - CDF(t)
        S = total_mass - cum_pdf
        
        # Compute hazard: h(t) = P(t)/S(t).
        # When S is zero, use the last valid hazard value instead of np.nan
        valid_hazards = P[:-1] / S[S > 0]  # Get valid hazard values
        last_valid = valid_hazards[-1] if len(valid_hazards) > 0 else 0
        hazard = np.where(S > 0, P / S, last_valid)

        return t, hazard

    @staticmethod
    def survival_from_distribution(P, t):
        """Static version of original survival_from_distribution"""
        import numpy as np
        from scipy.integrate import cumtrapz

        # Convert input to numpy arrays of type float for numerical operations.
        P = np.asarray(P, dtype=float)
        t = np.asarray(t, dtype=float)

        # Calculate the cumulative distribution (CDF) via trapezoidal integration.
        cum_pdf = cumtrapz(P, t, initial=0)

        # The total probability mass is assumed to be cum_pdf[-1].
        total_mass = cum_pdf[-1]

        # The survival function is S(t) = total_mass - CDF(t)
        survival = total_mass - cum_pdf

        return t, survival

    def generate_twin_data(self, n, r=0.5):
        """
        Generate twin lifespan data using precomputed distributions
        
        Parameters:
        n : number of twin pairs to generate
        r : correlation coefficient for intrinsic lifespans
        
        Returns:
        DataFrame with columns: twin1, twin2, twin1_type, twin2_type
        """
        # Use precomputed distributions from instance
        return self._generate_twin_data_impl(
            n, self.extrinsic_distribution, self.intrinsic_distribution,
            self.total_distribution, self.ages, r
        )

    @staticmethod
    def _generate_twin_data_impl(n, extrinsic_distribution, intrinsic_distribution,
                                total_distribution, ages, r=0.5):

        # Calculate areas under the curves to determine death probabilities.
        total_area = np.trapz(total_distribution[:-1], ages[:-1])
        intrinsic_area = np.trapz(intrinsic_distribution[:-1], ages[:-1])
        extrinsic_area = np.trapz(extrinsic_distribution[:-1], ages[:-1])
        extrinsic_death_prob = extrinsic_area / total_area
        # The intrinsic_death_prob is computed but not used explicitly
        intrinsic_death_prob = intrinsic_area / total_area

        # Normalize the extrinsic and intrinsic distributions (drop the last age point)
        P_e_norm = extrinsic_distribution[:-1] / extrinsic_distribution[:-1].sum()
        P_i_norm = intrinsic_distribution[:-1] / intrinsic_distribution[:-1].sum()
        ages_used = ages[:-1]

        # Precompute the cumulative density function for the intrinsic distribution.
        cdf_i = np.cumsum(P_i_norm)

        # Generate death type decisions for twin1 and twin2.
        twin1_randoms = np.random.random(n)
        twin2_randoms = np.random.random(n)
        twin1_types = np.where(twin1_randoms < extrinsic_death_prob, 'extrinsic', 'intrinsic')
        twin2_types = np.where(twin2_randoms < extrinsic_death_prob, 'extrinsic', 'intrinsic')

        # Set up correlated uniform samples for the intrinsic cases.
        z1 = norm.ppf(np.random.random(n))
        z2 = r * z1 + np.sqrt(1 - r**2) * np.random.normal(0, 1, n)
        u1 = norm.cdf(z1)
        u2 = norm.cdf(z2)

        # Initialize arrays to hold the lifespans.
        twin1_lifespans = np.empty(n)
        twin2_lifespans = np.empty(n)

        # --- Process Twin 1 ---
        # For extrinsic deaths, sample from ages using the normalized extrinsic distribution.
        mask_t1_ex = (twin1_types == 'extrinsic')
        count_t1_ex = np.sum(mask_t1_ex)
        if count_t1_ex > 0:
            twin1_lifespans[mask_t1_ex] = np.random.choice(ages_used, size=count_t1_ex, p=P_e_norm)
        # For intrinsic deaths, invert the intrinsic CDF using the precomputed u1.
        mask_t1_in = ~mask_t1_ex  # equivalent to (twin1_types == 'intrinsic')
        if np.sum(mask_t1_in) > 0:
            twin1_lifespans[mask_t1_in] = np.interp(u1[mask_t1_in], cdf_i, ages_used)

        # --- Process Twin 2 ---
        # For extrinsic deaths, sample similarly.
        mask_t2_ex = (twin2_types == 'extrinsic')
        count_t2_ex = np.sum(mask_t2_ex)
        if count_t2_ex > 0:
            twin2_lifespans[mask_t2_ex] = np.random.choice(ages_used, size=count_t2_ex, p=P_e_norm)
        # For intrinsic deaths, we distinguish two cases:
        #
        # 1. When twin1 is intrinsic, we want the second twin's lifespan to be correlated.
        #    Use the correlated sample u2.
        #
        # 2. When twin1 is extrinsic, sample independently by drawing a new
        #    uniform random value.
        mask_t2_in = ~mask_t2_ex
        mask_corr = mask_t2_in & (twin1_types == 'intrinsic')
        mask_ind = mask_t2_in & (twin1_types == 'extrinsic')

        if np.sum(mask_corr) > 0:
            twin2_lifespans[mask_corr] = np.interp(u2[mask_corr], cdf_i, ages_used)
        if np.sum(mask_ind) > 0:
            twin2_lifespans[mask_ind] = np.interp(np.random.random(np.sum(mask_ind)), cdf_i, ages_used)

        # Return a DataFrame with the results.
        return pd.DataFrame({
            'twin1': twin1_lifespans,
            'twin2': twin2_lifespans,
            'twin1_type': twin1_types,
            'twin2_type': twin2_types
        })

    @staticmethod
    def _generate_twin_data_impl(n, extrinsic_distribution, intrinsic_distribution,
                                total_distribution, ages, r=0.5):

        # Calculate areas under the curves to determine death probabilities.
        total_area = np.trapz(total_distribution[:-1], ages[:-1])
        intrinsic_area = np.trapz(intrinsic_distribution[:-1], ages[:-1])
        extrinsic_area = np.trapz(extrinsic_distribution[:-1], ages[:-1])
        extrinsic_death_prob = extrinsic_area / total_area
        # The intrinsic_death_prob is computed but not used explicitly
        intrinsic_death_prob = intrinsic_area / total_area

        # Normalize the extrinsic and intrinsic distributions (drop the last age point)
        P_e_norm = extrinsic_distribution[:-1] / extrinsic_distribution[:-1].sum()
        P_i_norm = intrinsic_distribution[:-1] / intrinsic_distribution[:-1].sum()
        ages_used = ages[:-1]

        # Precompute the cumulative density function for the intrinsic distribution.
        cdf_i = np.cumsum(P_i_norm)

        # Generate death type decisions for twin1 and twin2.
        twin1_randoms = np.random.random(n)
        twin2_randoms = np.random.random(n)
        twin1_types = np.where(twin1_randoms < extrinsic_death_prob, 'extrinsic', 'intrinsic')
        twin2_types = np.where(twin2_randoms < extrinsic_death_prob, 'extrinsic', 'intrinsic')

        # Set up correlated uniform samples for the intrinsic cases.
        z1 = norm.ppf(np.random.random(n))
        z2 = r * z1 + np.sqrt(1 - r**2) * np.random.normal(0, 1, n)
        u1 = norm.cdf(z1)
        u2 = norm.cdf(z2)

        # Initialize arrays to hold the lifespans.
        twin1_lifespans = np.empty(n)
        twin2_lifespans = np.empty(n)

        # --- Process Twin 1 ---
        # For extrinsic deaths, sample from ages using the normalized extrinsic distribution.
        mask_t1_ex = (twin1_types == 'extrinsic')
        count_t1_ex = np.sum(mask_t1_ex)
        if count_t1_ex > 0:
            twin1_lifespans[mask_t1_ex] = np.random.choice(ages_used, size=count_t1_ex, p=P_e_norm)
        # For intrinsic deaths, invert the intrinsic CDF using the precomputed u1.
        mask_t1_in = ~mask_t1_ex  # equivalent to (twin1_types == 'intrinsic')
        if np.sum(mask_t1_in) > 0:
            twin1_lifespans[mask_t1_in] = np.interp(u1[mask_t1_in], cdf_i, ages_used)

        # --- Process Twin 2 ---
        # For extrinsic deaths, sample similarly.
        mask_t2_ex = (twin2_types == 'extrinsic')
        count_t2_ex = np.sum(mask_t2_ex)
        if count_t2_ex > 0:
            twin2_lifespans[mask_t2_ex] = np.random.choice(ages_used, size=count_t2_ex, p=P_e_norm)
        # For intrinsic deaths, we distinguish two cases:
        #
        # 1. When twin1 is intrinsic, we want the second twin's lifespan to be correlated.
        #    Use the correlated sample u2.
        #
        # 2. When twin1 is extrinsic, sample independently by drawing a new
        #    uniform random value.
        mask_t2_in = ~mask_t2_ex
        mask_corr = mask_t2_in & (twin1_types == 'intrinsic')
        mask_ind = mask_t2_in & (twin1_types == 'extrinsic')

        if np.sum(mask_corr) > 0:
            twin2_lifespans[mask_corr] = np.interp(u2[mask_corr], cdf_i, ages_used)
        if np.sum(mask_ind) > 0:
            twin2_lifespans[mask_ind] = np.interp(np.random.random(np.sum(mask_ind)), cdf_i, ages_used)

        # Return a DataFrame with the results.
        return pd.DataFrame({
            'twin1': twin1_lifespans,
            'twin2': twin2_lifespans,
            'twin1_type': twin1_types,
            'twin2_type': twin2_types
        })


    def plot_twin_correlations(self, table=None, n=1000, r=0.5, ax=None):
        """
        Create scatter plot showing correlations between twin lifespans,
        separated by cause of death.
        
        Parameters:
        table : DataFrame with columns twin1, twin2, twin1_type, twin2_type
                as generated by generate_twin_data(). If None, n and r must be provided.
        n : int, optional
            Number of twin pairs to generate if table not provided
        r : float, optional 
            Correlation coefficient for generating twin data if table not provided
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure and axes.
        
        Returns:
        matplotlib axes object showing the correlations
        """

        # Generate data if table not provided
        if table is None:
            if n is None or r is None:
                raise ValueError("Must provide either table or both n and r")
            table = self.generate_twin_data(n=n, r=r)

        # Split pairs by death type
        intrinsic_pairs = table[
            (table['twin1_type'] == 'intrinsic') & 
            (table['twin2_type'] == 'intrinsic')
        ]
        extrinsic_pairs = table[
            (table['twin1_type'] == 'extrinsic') & 
            (table['twin2_type'] == 'extrinsic')
        ]
        mixed_pairs = table[
            (table['twin1_type'] != table['twin2_type'])
        ]

        # Calculate correlations
        intrinsic_correlation, _ = pearsonr(intrinsic_pairs['twin1'], intrinsic_pairs['twin2'])
        extrinsic_correlation, _ = pearsonr(extrinsic_pairs['twin1'], extrinsic_pairs['twin2'])
        mixed_correlation, _ = pearsonr(mixed_pairs['twin1'], mixed_pairs['twin2'])
        total_correlation, _ = pearsonr(table['twin1'], table['twin2'])

        # Calculate percentages
        total_pairs = len(table)
        intrinsic_pct = len(intrinsic_pairs) / total_pairs * 100
        extrinsic_pct = len(extrinsic_pairs) / total_pairs * 100
        mixed_pct = len(mixed_pairs) / total_pairs * 100

        # Round correlations to 2 decimal places for labels
        intrinsic_corr_str = f"{intrinsic_correlation:.2f}"
        extrinsic_corr_str = f"{extrinsic_correlation:.2f}"
        mixed_corr_str = f"{mixed_correlation:.2f}"
        total_corr_str = f"{total_correlation:.2f}"

        # Create axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Create scatter plot
        ax.scatter(intrinsic_pairs['twin1'], intrinsic_pairs['twin2'], 
                   alpha=0.5, color='blue', 
                   label=f'Intrinsic-Intrinsic (r={intrinsic_corr_str})')
        ax.scatter(extrinsic_pairs['twin1'], extrinsic_pairs['twin2'], 
                   alpha=0.5, color='red',
                   label=f'Extrinsic-Extrinsic (r={extrinsic_corr_str})')
        ax.scatter(mixed_pairs['twin1'], mixed_pairs['twin2'], 
                   alpha=0.5, color='green',
                   label=f'Mixed (r={mixed_corr_str})')

        ax.set_xlabel('Twin 1 lifespan', fontname='Arial')
        ax.set_ylabel('Twin 2 lifespan', fontname='Arial')
        ax.set_title(f'Twin lifespans\nobserved correlation r = {total_corr_str}',
                     fontname='Arial')
        ax.legend(prop={'family': 'Arial'})
        
        # Set font for tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Arial')
        
        return ax


    def simulate_correlation_distribution(self, r, n_twins=1000, n_runs=2000, plot=True):
        """
        Simulate twin correlations multiple times to get distribution of correlation coefficients.
        
        Parameters:
        -----------
        r : float
            Target correlation coefficient for twin pairs
        n_twins : int, optional
            Number of twin pairs per simulation run (default: 1000)
        n_runs : int, optional 
            Number of simulation runs (default: 2000)
        plot : bool, optional
            If True, plots histogram of correlations. If False, returns statistics (default: True)
            
        Returns:
        --------
        If plot=True:
            matplotlib figure object
        If plot=False:
            tuple containing:
            - mean correlation
            - standard deviation
            - 95% confidence interval (tuple of lower and upper bounds)
        """
        correlations = []
        for _ in range(n_runs):
            table = self.generate_twin_data(n=n_twins, r=r)
            total_correlation, _ = pearsonr(table['twin1'], table['twin2'])
            correlations.append(total_correlation)
            
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        ci_95 = np.percentile(correlations, [2.5, 97.5])
        
        if not plot:
            return mean_corr, std_corr, ci_95
            
        fig = plt.figure(figsize=(8,6))
        plt.hist(correlations, bins=30)
        plt.xlabel('Correlation coefficient')
        plt.ylabel('Frequency')
        plt.axvline(mean_corr, color='red', linestyle='--', 
                   label=f'Mean: {mean_corr:.3f}\nStd: {std_corr:.3f}\n95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]')
        plt.axvline(ci_95[0], color='gray', linestyle=':', alpha=0.5)
        plt.axvline(ci_95[1], color='gray', linestyle=':', alpha=0.5)
        plt.title(f'Distribution of twin correlations (n={n_runs})\n'
                 f'Target r = {r}, Mean r = {mean_corr:.3f} ± {std_corr:.3f}')
        plt.legend()
        
        return fig


    @staticmethod
    def fit_gompertz_hazard_from_data(ages, hazards, ages_start=70, ages_end=85):
        """Fit exponential hazard function to data in specified age range"""
        import numpy as np
        
        ages = np.asarray(ages)
        hazards = np.asarray(hazards)

        filtered_indices = (ages >= ages_start) & (ages <= ages_end) & (hazards > 0)
        ages_filtered = ages[filtered_indices]
        hazards_filtered = hazards[filtered_indices]

        if len(ages_filtered) < 2:
            raise ValueError(f"Insufficient valid data points between ages {ages_start} and {ages_end}. Need at least 2 points for fitting.")

        try:
            log_hazards = np.log(hazards_filtered)
            coeffs = np.polyfit(ages_filtered, log_hazards, 1)
            B = coeffs[0]
            A = np.exp(coeffs[1])
            return A, B
            
        except RuntimeError as e:
            raise ValueError(f"Curve fitting failed: {str(e)}")

    @staticmethod
    def get_min_hazard(ages, hazards, ages_start=70, ages_end=85):
        """Get minimum between actual hazard and fitted Gompertz hazard"""
        import numpy as np
        
        A, B = ModelFreeTwins.fit_gompertz_hazard_from_data(ages, hazards, ages_start, ages_end)
        gompertz = A * np.exp(B * ages)
        min_hazard = np.minimum(hazards, gompertz)
        return min_hazard

    def plot_corrs_vs_mex(self, plot_type='h2', study='danish', filter_age=15, 
                          use_both=True, models=['gg', 'sr'], 
                          color_dict={'gg': 'green', 'sr': 'purple'},
                          ax=None):
        """
        Plot twin correlations/heritability vs extrinsic mortality rate (mex)
        
        Parameters:
        plot_type : 'mz', 'dz', or 'h2'
        study : 'danish' or 'swedish'
        filter_age : age cutoff to use for filtering (15, 37, etc.)
        use_both : whether to plot combined genders (True) or separate (False)
        models : list of models to plot ('gg', 'sr', etc.)
        color_dict : color mapping for models
        ax : matplotlib axis to plot on (if None, creates new figure/axis)
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        
        # ... [rest of the code remains the same] ...
        
        plt.tight_layout()
        plt.show()