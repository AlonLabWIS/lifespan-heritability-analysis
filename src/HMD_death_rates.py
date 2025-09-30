"""
Lightweight loader for cohort death rates tables from `saved_data`.

Provides helpers to read country-specific cohort death rates, and to
compute survival/hazard from age X for quick exploratory plots.
"""
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from scipy import interpolate, integrate
from scipy.stats import linregress
from scipy.optimize import curve_fit
import seaborn as sns
from matplotlib.colors import ListedColormap
import os

class HMD_cohort_death_rates:
    def __init__(self, country):

        self.country = country
        project_root = os.path.dirname(os.path.dirname(__file__))
        self.data_folder = os.path.join(project_root, "saved_data")
        self.death_rates_table = self.load_data()

    def load_data(self):
        float_columns = ['Female', 'Male', 'Total']
        # Custom converter for float columns
        def to_float(x):
            return np.nan if x == '.' else float(x)
        def age_to_int(age):
            return 110 if age == '110+' else int(age)

        table = pd.read_csv(f"{self.data_folder}/{self.country}_cohort_death_rates.txt" , sep='\s+' , converters={'Age': age_to_int, **{col: to_float for col in float_columns}})
        table['Age'] = table['Age'].replace('110+', 110).astype(int)
        return table

    def calc_survival_from_age_X(self, X, gender='Total', year=None):
        # Filter data for the specified year and gender
        filtered_data = self.death_rates_table[(self.death_rates_table['Year'] == year) & 
                                     (self.death_rates_table['Age'] >= X)]
        if gender == 'Male' or gender ==  'm' or gender == 'M' or gender == 'male':
            gender = 'Male'
        elif gender == 'Female' or gender == 'f' or gender == 'F' or gender == 'female':
            gender = 'Female'
        else:
            gender = 'Total'
        ages = filtered_data['Age'].values
        mx = filtered_data[gender].values

        # Convert mx to qx (assuming ax = 0.5 for simplicity)
        qx = mx / (1 + 0.5 * mx)

        # Calculate survival probabilities
        lx = np.ones(len(ages))
        for i in range(1, len(lx)):
            lx[i] = lx[i-1] * (1 - qx[i-1])

        return ages, lx

    def calc_hazard_from_age_X(self, X, gender='Total', year=None):
        filtered_data = self.death_rates_table[(self.death_rates_table['Year'] == year) & 
                                     (self.death_rates_table['Age'] >= X)]
        if gender == 'Male' or gender ==  'm' or gender == 'M' or gender == 'male':
            gender = 'Male'
        elif gender == 'Female' or gender == 'f' or gender == 'F' or gender == 'female':
            gender = 'Female'
        else:
            gender = 'Total'
        ages = filtered_data['Age'].values
        mx = filtered_data[gender].values
        # Convert mx to qx (assuming ax = 0.5 for simplicity)
        qx = mx / (1 + 0.5 * mx)

    def plot_survival_from_age_X(self, X, gender='Total', year=None, ax=None, **kwargs):
        ages, lx = self.calc_survival_from_age_X(X, gender, year)
        if ax is None:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
        sns.lineplot(x=ages, y=lx, ax=ax, label = f'{gender} Survival', **kwargs)
        plt.xlabel('Age')
        plt.ylabel('Survival')
        plt.title(f'Survival from Age {X} for {self.country.capitalize()}, {gender.capitalize()}')