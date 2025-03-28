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
import matplotlib.pyplot as plt

#Set up working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"
final_monthly_data = pd.read_csv(data_folder + "/preprocessed/final_monthly_data.csv")
model_data_cgo = pd.read_csv(data_folder + "/preprocessed/model_data_cgo.csv")
average_metrics = pd.read_csv(data_folder + "/preprocessed/average_metrics.csv")

# ===================================================================    
#                     a. Set up known parameters        
# ===================================================================
nu = 17
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
N = 100
pr_DI = 3
pr_HY = 19
pr_IG = 78
# N = 1000
# pr_DI = 30
# pr_HY = 190
# pr_IG = 780
# N = 5000
# pr_DI = 150
# pr_HY = 950
# pr_IG = 3900

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
print("Calculating S and zeta")

# Define the function for the system of equations
def equation_system(params, nu, sigma, skew):
    Si, zeta = params
    eq1 = sqrt((nu / (nu - 2)) * Si + (2 * nu**2) / ((nu - 2)**2 * (nu - 4)) * zeta**2) - sigma
    eq2 = ((2 * zeta * sqrt(nu * (nu - 4))) /
           (sqrt(Si) * ((2 * nu * zeta**2) / Si + (nu - 2) * (nu - 4))**(3 / 2))) * \
          (3 * (nu - 2) + (8 * nu * zeta**2) / (Si * (nu - 6))) - skew
    return [eq1, eq2]

# Define a wrapper function to solve the system given nu, sigma, and skew
def solve_equations(nu, sigma, skew):
    # Initial guesses for Si and zeta
    initial_guess = [0.001, 0.001]
    
    # Solve the equations using fsolve
    solution = fsolve(equation_system, initial_guess, args=(nu, sigma, skew))
    
    # Return the solution as a tuple (Si, zeta)
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
average_metrics_updated.to_csv(data_folder + '/preprocessed/average_metrics_updated.csv', index=False)

# def equation_vol_skew(p,*args): # Defines the equations 
#     Si, zetai = p
#     volatility, skewness = args
#     if Si <= 0:
#         return [1e6, 1e6]
#     return[(nu/(nu-2) * Si + ((2 * nu ** 2) / ((nu-2)**2 * (nu-4))) * zetai**2) ** (0.5) - volatility,
#                 ((2 * zetai * sqrt(nu * (nu - 4)) /
#                 (sqrt(Si) * ((2 * nu * zetai ** 2) / Si + (nu - 2) * (nu - 4)) ** (3 / 2)))
#                 * (3 * (nu - 2) + (8 * nu * zetai ** 2) / (Si * (nu - 6))))- skewness]


# def fsolve_si_zetai(volatility, skewness): # Attempts to find values of Si and zetai that satisfy equation_std_skew.
#     Si, zetai = fsolve(equation_vol_skew,(0.2, 0.2),args=(volatility, skewness))
#     # return {'Si':Si, 'zetai':zetai}
#     return Si, zetai


# start = timeit.default_timer()
# average_metrics['Si'], average_metrics['zetai'] = zip(*average_metrics.apply(lambda x: fsolve_si_zetai(x.volatility, x.skewness), axis=1))

# stop = timeit.default_timer()
# execution_time = stop - start
# print(f"Program Executed in {execution_time} seconds")  # Returns time in seconds

# average_metrics.to_csv(data_folder + '/preprocessed/average_metrics_updated.csv', index = False) # Export for later use
# print("done solving S and zetai")


# print("Calculating S and zeta")

# def equation_vol_skew(p, *args): # Defines the equations
#     Si, zetai = p
#     volatility, skewness = args
#     if Si <= 0: # Ensure that Si is strictly positive
#         return [1e6, 1e6]
#     eq1 = ((nu/(nu-2) * Si + ((2 * nu ** 2) / ((nu-2)**2 * (nu-4))) * zetai**2) ** 0.5) - volatility
#     expr = ((2 * nu * zetai ** 2) / Si + (nu - 2) * (nu - 4)) #part of the denominator of skewness formula
#     if expr < 0: #must be positive to avoid complex numbers when taking the power of 3/2
#         return [1e6, 1e6]
#     eq2 = ((2 * zetai * sqrt(nu * (nu - 4)) /
#             (sqrt(Si) * (expr) ** (3/2))) *
#            (3 * (nu - 2) + (8 * nu * zetai ** 2) / (Si * (nu - 6)))) - skewness
#     return [eq1, eq2]

# def fsolve_si_zetai(volatility, skewness): # Attempts to solve for Si and zetai, with an initial guess (0.1, 0.1)
#     Si, zetai = fsolve(equation_vol_skew, (0.1, 0.1), args=(volatility, skewness))
#     return Si, zetai

# average_metrics['Si'], average_metrics['zetai'] = zip(*average_metrics.apply(
#     lambda x: fsolve_si_zetai(x.volatility, x.skewness), axis=1))

# average_metrics['portfolio'] = ['DI', 'HY', 'IG']
# average_metrics.to_csv(os.path.join(data_folder, 'preprocessed/average_metrics_updated.csv'), index=False)
# print("done solving S and zeta")




# # Empirical target values for DI bonds (from your CSV)
# vol_target = 0.1011781287870689    # empirical volatility (in decimals)
# skew_target = 0.0193250587179916    # empirical skewness

# # Set the degree-of-freedom parameter nu (as used in the model)

# # Define the function returning the residuals for the two equations
# def equation_vol_skew(p, *args):
#     Si, zetai = p
#     volatility, skewness = args
    
#     # Enforce Si > 0 to avoid invalid square roots
#     if Si <= 0:
#         return [1e6, 1e6]
    
#     # Equation 1: difference between theoretical and empirical volatility
#     theo_std = ((nu/(nu-2) * Si + ((2 * nu**2) / ((nu-2)**2 * (nu-4))) * zetai**2))**0.5
#     eq1 = theo_std - volatility
    
#     # Intermediate expression for skewness calculation
#     expr = ((2 * nu * zetai**2) / Si + (nu - 2) * (nu - 4))
#     if expr < 0:
#         return [1e6, 1e6]
    
#     # Equation 2: difference between theoretical and empirical skewness
#     theo_skew = ((2 * zetai * sqrt(nu * (nu - 4))) /
#                  (sqrt(Si) * (expr)**(3/2))) * (3 * (nu - 2) + (8 * nu * zetai**2) / (Si * (nu - 6)))
#     eq2 = theo_skew - skewness
    
#     return [eq1, eq2]

# # Create a grid of Si and zetai values.
# # Adjust the ranges based on your expected solution region.
# S_values = np.linspace(0.001, 0.2, 200)       # Example range for S_i
# zeta_values = np.linspace(-0.1, 0.1, 200)       # Example range for ζ_i

# S_grid, zeta_grid = np.meshgrid(S_values, zeta_values)

# # Prepare arrays to store the residuals.
# eq1_grid = np.zeros_like(S_grid)
# eq2_grid = np.zeros_like(S_grid)

# # Loop over the grid to compute residuals at each (Si, ζ_i)
# for i in range(S_grid.shape[0]):
#     for j in range(S_grid.shape[1]):
#         Si_val = S_grid[i, j]
#         zeta_val = zeta_grid[i, j]
        
#         if Si_val <= 0:
#             eq1_grid[i, j] = 1e6
#             eq2_grid[i, j] = 1e6
#             continue
        
#         theo_std = ((nu/(nu-2) * Si_val + ((2 * nu**2) / ((nu-2)**2 * (nu-4))) * zeta_val**2))**0.5
#         eq1_val = theo_std - vol_target
        
#         expr_val = ((2 * nu * zeta_val**2) / Si_val + (nu - 2) * (nu - 4))
#         if expr_val < 0:
#             eq1_grid[i, j] = 1e6
#             eq2_grid[i, j] = 1e6
#             continue
        
#         theo_skew = ((2 * zeta_val * sqrt(nu * (nu - 4))) /
#                      (sqrt(Si_val) * (expr_val)**(3/2))) * (3 * (nu - 2) + (8 * nu * zeta_val**2) / (Si_val * (nu - 6)))
#         eq2_val = theo_skew - skew_target
        
#         eq1_grid[i, j] = eq1_val
#         eq2_grid[i, j] = eq2_val

# # --- Plotting the Residuals ---
# plt.figure(figsize=(14, 6))

# # Plot for Equation 1 (Volatility difference)
# plt.subplot(1, 2, 1)
# contour1 = plt.contourf(S_grid, zeta_grid, eq1_grid, levels=50, cmap='coolwarm', vmin=-0.01, vmax=0.01)
# plt.colorbar(contour1)
# # Add an explicit contour line at 0 (eq1 = 0)
# zero_contour1 = plt.contour(S_grid, zeta_grid, eq1_grid, levels=[0], colors='black', linewidths=2)
# plt.clabel(zero_contour1, fmt='eq1=0', fontsize=10)
# plt.title('Residual of Equation 1 (Theoretical Std - Empirical Vol)')
# plt.xlabel('$S_i$')
# plt.ylabel('$\\zeta_i$')

# # Plot for Equation 2 (Skewness difference)
# plt.subplot(1, 2, 2)
# contour2 = plt.contourf(S_grid, zeta_grid, eq2_grid, levels=50, cmap='coolwarm', vmin=-0.01, vmax=0.01)
# plt.colorbar(contour2)
# # Add an explicit contour line at 0 (eq2 = 0)
# zero_contour2 = plt.contour(S_grid, zeta_grid, eq2_grid, levels=[0], colors='black', linewidths=2)
# plt.clabel(zero_contour2, fmt='eq2=0', fontsize=10)
# plt.title('Residual of Equation 2 (Theoretical Skew - Empirical Skew)')
# plt.xlabel('$S_i$')
# plt.ylabel('$\\zeta_i$')

# plt.tight_layout()
# plt.show()