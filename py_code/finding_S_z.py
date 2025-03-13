# =============================================================================
#
#                     Part X: Model Setup A
#                   (S_i and zeta_i estimation)
#
#         (Considers only subset including the cleaned data)
#
# =============================================================================

# Importing necessary libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import skew

#Set up working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"
final_monthly_data = pd.read_csv(data_folder + "/preprocessed/final_monthly_data.csv")
model_data_cgo = pd.read_csv(data_folder + "/preprocessed/model_data_cgo.csv")

# ===================================================================    
#                     a. Set up known parameters        
# ===================================================================
nu = 7.5
sigma_m = 0.25
Rf = 1
gamma_hat, b0 = (0.6, 0.6)
alpha, delta, lamb = (0.7, 0.65, 1.5)

# To find theta, we need to know the average number of bonds in our datasat as
# well as the average proportions of each portfolio of bonds.
monthly_total = model_data_cgo.groupby('eom').size().reset_index(name='total_bonds')
avg_bonds = monthly_total['total_bonds'].mean()
print("Average number of bonds per month:", avg_bonds)
monthly_portfolio = (
    model_data_cgo.groupby(['eom', 'portfolio'])
    .size()
    .reset_index(name='portfolio_count')
)
monthly_portfolio = monthly_portfolio.merge(monthly_total, on='eom')
monthly_portfolio['proportion'] = monthly_portfolio['portfolio_count'] / monthly_portfolio['total_bonds']
print("\nMonthly proportions by portfolio (first few rows):")
print(monthly_portfolio.head())
avg_portfolio_props = monthly_portfolio.groupby('portfolio')['proportion'].mean().reset_index()
print("\nAverage proportions across all months:")
print(avg_portfolio_props)

N = 5000
pr_DI = 150
pr_HY = 950
pr_IG = 3900

# For each month and portfolio, sum the market_value_past.
monthly_mv = (
    model_data_cgo.groupby(['eom', 'portfolio'])['market_value_past']
    .sum()
    .reset_index()
)
overall_monthly = (
    model_data_cgo.groupby('eom')['market_value_past']
    .sum()
    .reset_index()
)

avg_portfolio_mv = monthly_mv.groupby('portfolio')['market_value_past'].mean().reset_index()
overall_avg_mv = overall_monthly['market_value_past'].mean()
avg_portfolio_mv['proportion'] = avg_portfolio_mv['market_value_past'] / overall_avg_mv

print("Average monthly portfolio market values and weights:")
print(avg_portfolio_mv)

weight_DI = avg_portfolio_mv.loc[avg_portfolio_mv['portfolio'] == 'DI', 'proportion'].values[0]
weight_HY = avg_portfolio_mv.loc[avg_portfolio_mv['portfolio'] == 'HY', 'proportion'].values[0]
weight_IG = avg_portfolio_mv.loc[avg_portfolio_mv['portfolio'] == 'IG', 'proportion'].values[0]

# Given target market portfolio values (pr_DI, pr_HY, pr_IG)
pr_DI = 150
pr_HY = 950
pr_IG = 3900

# Compute theta for each portfolio as the ratio of the computed weight to the target value
theta_DI = weight_DI / pr_DI
theta_HY = weight_HY / pr_HY
theta_IG = weight_IG / pr_IG

print("Theta_DI:", theta_DI)
print("Theta_HY:", theta_HY)
print("Theta_IG:", theta_IG)
