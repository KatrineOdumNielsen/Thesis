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
from scipy.stats import skew
from scipy.optimize import fsolve
from math import sqrt
import timeit

#Set up working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"
final_monthly_data = pd.read_csv(data_folder + "/preprocessed/final_monthly_data.csv")
model_data_cgo = pd.read_csv(data_folder + "/preprocessed/model_data_cgo.csv")
average_metrics = pd.read_csv(data_folder + "/preprocessed/average_metrics.csv")

# ===================================================================    
#                     a. Set up known parameters        
# ===================================================================
nu = 7.5
sigma_m = 0.25
Rf = 1
gamma_hat, b0 = (0.6, 0.6)
alpha, delta, lamb = (0.7, 0.65, 1.5)

# To find theta, we need to know the average number of bonds in our datasat as well as the average proportions of each portfolio of bonds.
monthly_total = model_data_cgo.groupby('eom').size().reset_index(name='total_bonds') #Counts the number of bonds in each month
avg_bonds = monthly_total['total_bonds'].mean() #Takes the average number of bonds in each month
print("Average number of bonds per month:", avg_bonds)
monthly_portfolio = (model_data_cgo.groupby(['eom', 'portfolio']).size().reset_index(name='portfolio_count')) #Counts the number of bonds in each month by portfolio
monthly_portfolio = monthly_portfolio.merge(monthly_total, on='eom') #Merges monthly total number of bonds with monthly number of bonds in each portfolio
monthly_portfolio['proportion'] = monthly_portfolio['portfolio_count'] / monthly_portfolio['total_bonds'] #Calculates the proportion of bonds in each portfolio for each month
avg_portfolio_props = monthly_portfolio.groupby('portfolio')['proportion'].mean().reset_index() #Takes the average proportion of bonds in each portfolio across all months
print("\nAverage proportions across all months:")
print(avg_portfolio_props)

#Rounds the average proportions to even numbers
N = 5000
pr_DI = 150
pr_HY = 950
pr_IG = 3900

# For each month and portfolio, sum the market_value_past
monthly_mv = (model_data_cgo.groupby(['eom', 'portfolio'])['market_value_past'].sum().reset_index()) #Finds the total market value of each portfolio during that month
overall_monthly = (model_data_cgo.groupby('eom')['market_value_past'].sum().reset_index()) #Finds the total market value of all bonds during that month
avg_portfolio_mv = monthly_mv.groupby('portfolio')['market_value_past'].mean().reset_index() #Takes the average market value of each portfolio across all months
overall_avg_mv = overall_monthly['market_value_past'].mean() #Takes the average market value of all bonds across all months
avg_portfolio_mv['weight'] = avg_portfolio_mv['market_value_past'] / overall_avg_mv #Calculates the weight of each portfolio
weight_DI = avg_portfolio_mv.loc[avg_portfolio_mv['portfolio'] == 'DI', 'weight'].values[0] #Assigns the weight of DI
weight_HY = avg_portfolio_mv.loc[avg_portfolio_mv['portfolio'] == 'HY', 'weight'].values[0] #Assigns the weight of HY
weight_IG = avg_portfolio_mv.loc[avg_portfolio_mv['portfolio'] == 'IG', 'weight'].values[0] #Assigns the weight of IG
print("Average monthly portfolio market values and weights:")
print(avg_portfolio_mv)

# Compute theta for each portfolio as the ratio of the computed weight to the target value
theta_mi_DI = weight_DI / pr_DI
theta_mi_HY = weight_HY / pr_HY
theta_mi_IG = weight_IG / pr_IG
print("Theta_DI:", theta_mi_DI)
print("Theta_HY:", theta_mi_HY)
print("Theta_IG:", theta_mi_IG)

# Making thetas into a dataframe for later use
portfolios = ['DI', 'HY', 'IG']
theta_values = [theta_mi_DI, theta_mi_HY, theta_mi_IG]
thetas_df = pd.DataFrame({'portfolio': portfolios,'theta_mi': theta_values})
thetas_df['theta_i_minus1'] = thetas_df['theta_mi']
print("thetas_df DataFrame:")
print(thetas_df)
thetas_df.to_csv(data_folder + '/preprocessed/thetas_df.csv') # Export for later use

# ======================================================================================================================
# Calculating S and zeta
# ======================================================================================================================
# print("Calculating S and zeta")

# def equation_vol_skew(p,*args): # Defines the equation 
#     Si, zetai = p
#     volatility, skewness = args
#     return[(nu/(nu-2) * Si + ((2 * nu ** 2) / ((nu-2)**2 * (nu-4))) * zetai**2) ** (0.5) - volatility,
#             ((2 * zetai * sqrt(nu * (nu - 4)) /
#               (sqrt(Si) * ((2 * nu * zetai ** 2) / Si + (nu - 2) * (nu - 4)) ** (3 / 2)))
#              * (3 * (nu - 2) + (8 * nu * zetai ** 2) / (Si * (nu - 6))))- skewness]


# def fsolve_si_zetai(volatility, skewness): # Attempts to find values of Si and zetai that satisfy equation_std_skew.
#     Si, zetai = fsolve(equation_vol_skew,(0.1, 0.1),args=(volatility, skewness))
#     # return {'Si':Si, 'zetai':zetai}
#     return Si, zetai


# start = timeit.default_timer()
# average_metrics['Si'], average_metrics['zetai'] = zip(*average_metrics.apply(lambda x: fsolve_si_zetai(x.volatility, x.skewness), axis=1))

# stop = timeit.default_timer()
# execution_time = stop - start
# print(f"Program Executed in {execution_time} seconds")  # Returns time in seconds

# average_metrics.to_csv(data_folder + '/preprocessed/average_metrics_updated.csv', index = False) # Export for later use
# print("done solving S and zetai")


print("Calculating S and zeta")

def equation_vol_skew(p, *args):
    Si, zetai = p
    volatility, skewness = args
    
    # Ensure that Si is strictly positive
    if Si <= 0:
        return [1e6, 1e6]
    
    # Compute the theoretical standard deviation
    eq1 = ((nu/(nu-2) * Si + ((2 * nu ** 2) / ((nu-2)**2 * (nu-4))) * zetai**2) ** 0.5) - volatility
    
    # Compute the expression inside the skewness formula
    expr = ((2 * nu * zetai ** 2) / Si + (nu - 2) * (nu - 4))
    # If this expression is negative, the power (3/2) is not defined for real numbers
    if expr < 0:
        return [1e6, 1e6]
    
    eq2 = ((2 * zetai * sqrt(nu * (nu - 4)) /
            (sqrt(Si) * (expr) ** (3/2))) *
           (3 * (nu - 2) + (8 * nu * zetai ** 2) / (Si * (nu - 6)))) - skewness
    
    return [eq1, eq2]

def fsolve_si_zetai(volatility, skewness):
    # Attempts to solve for Si and zetai, with an initial guess (0.1, 0.1)
    Si, zetai = fsolve(equation_vol_skew, (0.1, 0.1), args=(volatility, skewness))
    return Si, zetai

start = timeit.default_timer()
# Make sure that average_metrics has columns 'volatility' and 'skewness'
average_metrics['Si'], average_metrics['zetai'] = zip(*average_metrics.apply(
    lambda x: fsolve_si_zetai(x.volatility, x.skewness), axis=1))
stop = timeit.default_timer()
execution_time = stop - start
print(f"Program Executed in {execution_time} seconds")  # Returns time in seconds

# Save the updated DataFrame (ensure the file path is correctly formatted)
output_file = os.path.join(data_folder, 'preprocessed/average_metrics_updated.csv')
average_metrics.to_csv(output_file, index=False)
print("done solving S and zeta")