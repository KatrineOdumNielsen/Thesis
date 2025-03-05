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

# ===================================================================    
#                      Set up portfolios        
# ===================================================================                                                         

bond_data['eom'] = pd.to_datetime(bond_data['eom'])
unique_months = bond_data['eom'].unique()
portfolio_by_month = {}

for month in sorted(unique_months):
    month_data = bond_data[bond_data['eom'] == month].copy()
    # Assign portfolios for this month.
    # Order matters if the conditions are not mutually exclusive.
    month_data.loc[month_data['credit_spread'] > 0.1, 'portfolio'] = 'DI'
    month_data.loc[(month_data['portfolio'].isnull()) & (month_data['rating_class'] == '0.IG'), 'portfolio'] = 'IG'
    month_data.loc[(month_data['portfolio'].isnull()) & (month_data['rating_class'] == '1.HY'), 'portfolio'] = 'HY'
    month_data.loc[month_data['portfolio'].isnull(), 'portfolio'] = 'Other'
    
    # Store the resulting DataFrame in a dictionary keyed by the month.
    portfolio_by_month[month] = month_data

# Now you have a dictionary where for each month you have three (or more) portfolios.

month = sorted(unique_months)[0]
IG_portfolio = portfolio_by_month[month][portfolio_by_month[month]['portfolio'] == 'IG']
HY_portfolio = portfolio_by_month[month][portfolio_by_month[month]['portfolio'] == 'HY']
DI_portfolio = portfolio_by_month[month][portfolio_by_month[month]['portfolio'] == 'DI']

print(portfolio_by_month)