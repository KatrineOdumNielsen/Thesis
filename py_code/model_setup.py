# Ask Harris about the code when we set this up; how to access the model from another py file
# def model():
#     model = Sequential()
#     model.add(Dense(32, input_dim=784))
#     model.add(Activation('relu'))
#     model.add(Dense(10))
#     model.add(Activation('softmax'))

# =============================================================================
#
#                     Part X: Model Setup
#
#         (Considers only subset including the cleaned data)
#
# =============================================================================

# Importing necessary libraries
import pandas as pd
import numpy as np
import os

#Set up working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"
figures_folder = project_dir + "/figures"

# Importing the cleaned data
bond_data = pd.read_csv(data_folder + "/preprocessed/bond_data.csv")
model_data = bond_data[['eom', 'cusip', 'ret_exc', 'credit_spread_past', 'rating_class', 'market_value_past', 'price_eom']]
model_data['eom'] = pd.to_datetime(model_data['eom'])
model_data.to_csv("data/preprocessed/model_data.csv")
## HUSK AT ÆNDRE TIL PRICE_PAST####

# ===================================================================    
#                      Set up portfolios        
# ===================================================================                                                         

bond_data['eom'] = pd.to_datetime(model_data['eom'])
unique_months = model_data['eom'].unique()
portfolio_by_month = {}

for month in sorted(unique_months):
    month_data = model_data[model_data['eom'] == month].copy()
    # Assign portfolios for this month.
    # Order matters if the conditions are not mutually exclusive.
    month_data.loc[month_data['credit_spread_past'] > 0.1, 'portfolio'] = 'DI'
    month_data.loc[(month_data['portfolio'].isnull()) & (month_data['rating_class'] == '0.IG'), 'portfolio'] = 'IG'
    month_data.loc[(month_data['portfolio'].isnull()) & (month_data['rating_class'] == '1.HY'), 'portfolio'] = 'HY'
    month_data.loc[month_data['portfolio'].isnull(), 'portfolio'] = 'Other'
    
    # Store the resulting DataFrame in a dictionary keyed by the month.
    portfolio_by_month[month] = month_data

# Now you have a dictionary where for each month you have three portfolios.

month = sorted(unique_months)[0]
IG_portfolio = portfolio_by_month[month][portfolio_by_month[month]['portfolio'] == 'IG']
HY_portfolio = portfolio_by_month[month][portfolio_by_month[month]['portfolio'] == 'HY']
DI_portfolio = portfolio_by_month[month][portfolio_by_month[month]['portfolio'] == 'DI']

print(portfolio_by_month)

# ==================== Calculate monthly portfolio weighted returns ============================
# Function to calculate weighted return for each bond
def calculate_monthly_weighted_return(month_data):
    monthly_total_portfolio_value_past = month_data.groupby('portfolio')['market_value_past'].sum()
    month_data['portfolio_weight'] = month_data['market_value_past'] / monthly_total_portfolio_value_past
    month_data['weighted_return'] = month_data['portfolio_weight'] * month_data['ret_exc']
    return month_data

#Add portfolio to the bond_data
model_data['portfolio'] = np.nan
model_data.loc[model_data['credit_spread_past'] > 0.1, 'portfolio'] = 'DI'
model_data.loc[model_data['portfolio'].isnull() & (model_data['rating_class'] == '0.IG'), 'portfolio'] = 'IG'
model_data.loc[model_data['portfolio'].isnull() & (model_data['rating_class'] == '1.HY'), 'portfolio'] = 'HY'
model_data.to_csv("data/preprocessed/model_data.csv")

portfolio_counts = model_data['portfolio'].value_counts()
print("Portfolio counts:")
print(portfolio_counts)

# ===================================================================    
#                      Calculate betas        
# ===================================================================   
# Function to calculate weighted return for each bond
def calculate_weighted_return(model_data):
    total_market_value_past = model_data['market_value_past'].sum()
    model_data['weight'] = model_data['market_value_past'] / total_market_value_past
    model_data['weighted_return'] = model_data['weight'] * model_data['ret_exc']
    return model_data

# Apply the function to calculate weighted return for each bond
model_data = calculate_weighted_return(model_data)

# Calculate the average weighted return across all bonds
average_weighted_return = model_data['weighted_return'].mean()
print("Average Weighted Return across all bonds:", average_weighted_return)

portfolio_returns = model_data.groupby(['eom','portfolio'])['weighted_return'].sum()

# Assuming you have a market return series to calculate beta against
# For example, let's assume 'market_return' is a series of market returns
market_return = model_data.groupby('eom')['weighted_return'].sum()

# Calculate the asset beta for each bond
betas = {}
for portfolio in model_data['portfolio'].unique():
    portfolio_returns = model_data[model_data['portfolio'] == portfolio]['weighted_return']
    cov_matrix = np.cov(portfolio_returns, market_return.loc[portfolio_returns.index])
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    betas[portfolio] = beta

print("Asset Betas for each portfolio:")
print(betas)