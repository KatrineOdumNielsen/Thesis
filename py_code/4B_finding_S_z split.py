# =============================================================================
#
#                     Part 4B: Finding S_i and zeta_i
#
#    (Considers cleaned data, but splits distressed by high and low CGO)
#
# =============================================================================

# Importing necessary libraries
import pandas as pd
import numpy as np
import os
from scipy.stats import skew
from scipy.optimize import fsolve
from math import sqrt
import timeit
import matplotlib.pyplot as plt

#Set up working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"
final_monthly_data = pd.read_csv(data_folder + "/preprocessed/final_monthly_data_split.csv")
model_data_cgo = pd.read_csv(data_folder + "/preprocessed/model_data_cgo_split.csv")
average_metrics = pd.read_csv(data_folder + "/preprocessed/average_metrics_split.csv")

# ===================================================================    
#                     a. Set up known parameters        
# ===================================================================
nu = 17
Rf = 1

# To find theta, we need to know the average number of bonds in our datasat as well as the average proportions of each portfolio of bonds.
monthly_total = model_data_cgo.groupby('eom').size().reset_index(name='total_bonds')
avg_bonds = monthly_total['total_bonds'].mean()
print("Average number of bonds per month:", avg_bonds)
monthly_portfolio = (model_data_cgo.groupby(['eom', 'portfolio']).size().reset_index(name='portfolio_count'))
monthly_portfolio = monthly_portfolio.merge(monthly_total, on='eom')
monthly_portfolio['proportion'] = monthly_portfolio['portfolio_count'] / monthly_portfolio['total_bonds']
avg_portfolio_props = monthly_portfolio.groupby('portfolio')['proportion'].mean().reset_index()
print("\nAverage proportions across all months:")
print(avg_portfolio_props)

#Rounds the average proportions to even numbers
N = 1000
pr_DI_low = 20
pr_DI_high = 20
pr_HY = 180
pr_IG = 780

# For each month and portfolio, sum the market_value_past
monthly_mv = (model_data_cgo.groupby(['eom', 'portfolio'])['market_value_start'].sum().reset_index()) 
overall_monthly = (model_data_cgo.groupby('eom')['market_value_start'].sum().reset_index()) 
avg_portfolio_mv = monthly_mv.groupby('portfolio')['market_value_start'].mean().reset_index() 
overall_avg_mv = overall_monthly['market_value_start'].mean() 
avg_portfolio_mv['weight'] = avg_portfolio_mv['market_value_start'] / overall_avg_mv
weight_DI_high = avg_portfolio_mv.loc[avg_portfolio_mv['portfolio'] == 'DI_high', 'weight'].values[0] #Assigns the weight of DI_high
weight_DI_low = avg_portfolio_mv.loc[avg_portfolio_mv['portfolio'] == 'DI_low', 'weight'].values[0] #Assigns the weight of DI_low
weight_HY = avg_portfolio_mv.loc[avg_portfolio_mv['portfolio'] == 'HY', 'weight'].values[0] #Assigns the weight of HY
weight_IG = avg_portfolio_mv.loc[avg_portfolio_mv['portfolio'] == 'IG', 'weight'].values[0] #Assigns the weight of IG
print("Average monthly portfolio market values and weights:")
print(avg_portfolio_mv)

# Compute theta for each portfolio as the ratio of the computed weight to the target value
theta_mi_DI_high = weight_DI_high / pr_DI_high
theta_mi_DI_low = weight_DI_low / pr_DI_low
theta_mi_HY = weight_HY / pr_HY
theta_mi_IG = weight_IG / pr_IG
print("Theta_DI_high:", theta_mi_DI_high)
print("Theta_DI_low:", theta_mi_DI_low)
print("Theta_HY:", theta_mi_HY)
print("Theta_IG:", theta_mi_IG)

# Making thetas into a dataframe for later use
portfolios = ['DI_high', 'DI_low', 'HY', 'IG']
theta_values = [theta_mi_DI_high, theta_mi_DI_low, theta_mi_HY, theta_mi_IG]
thetas_df = pd.DataFrame({'portfolio': portfolios,'theta_mi': theta_values})
thetas_df['theta_i_minus1'] = thetas_df['theta_mi']
# print("thetas_df DataFrame:")
# print(thetas_df)
thetas_df.to_csv(data_folder + '/preprocessed/thetas_df_split.csv')

# =========================================================================================
#                                    Calculating S and zeta
# =========================================================================================
print("Calculating S and zeta")

def equation_system(params, nu, sigma, skew):
    """
    Defines the system of equations to solve for Si and zeta.
    """
    Si, zeta = params
    eq1 = sqrt((nu / (nu - 2)) * Si + (2 * nu**2) / ((nu - 2)**2 * (nu - 4)) * zeta**2) - sigma
    eq2 = ((2 * zeta * sqrt(nu * (nu - 4))) /
           (sqrt(Si) * ((2 * nu * zeta**2) / Si + (nu - 2) * (nu - 4))**(3 / 2))) * \
          (3 * (nu - 2) + (8 * nu * zeta**2) / (Si * (nu - 6))) - skew
    return [eq1, eq2]

def solve_equations(nu, sigma, skew):
    """
    Defines the wrapper function to solve the system of equations.
    It takes the parameters nu, sigma, and skew as inputs and returns the solution.
    Initial guesses for Si and zeta are set to 0.001.
    """
    initial_guess = [0.001, 0.001]

    solution = fsolve(equation_system, initial_guess, args=(nu, sigma, skew))
    
    return solution

# Initialize lists to store Si and zeta values
Si_values = []
zeta_values = []

# Compute Si and zeta for each row
for index, row in average_metrics.iterrows():
    sigma = row["volatility"]
    skewR = row["skewness"]
    Si, zeta = solve_equations(nu, sigma, skewR)
    Si_values.append(Si)
    zeta_values.append(zeta)

# Add the new columns to the dataframe
average_metrics["Si"] = Si_values
average_metrics["zeta"] = zeta_values

# Create the updated dataframe
average_metrics_updated = average_metrics.copy()
print(average_metrics_updated)
average_metrics_updated.to_csv(data_folder + '/preprocessed/average_metrics_split_updated.csv', index=False)

print("Average metrics updated with Si and zeta values")