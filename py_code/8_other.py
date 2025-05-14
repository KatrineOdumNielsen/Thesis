# ===================================================================    
#             a. Importing libraries, data and packages  
# ===================================================================

import os
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fredapi import Fred
fred = Fred(api_key='insert_key') # Insert FRED API key here
import statsmodels.api as sm
from datetime import datetime

# Set the working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"

# Loading data
model_data = pd.read_csv(data_folder + "/preprocessed/bond_data.csv")
model_data = model_data[model_data["ret"].abs() <= 0.2] # Aligning with cleaning method of Avramov et al.

# Creating portfolios
model_data['eom'] = pd.to_datetime(model_data['eom'])
model_data['offering_date'] = pd.to_datetime(model_data['offering_date'])
unique_months = model_data['eom'].unique()

# Add portfolio to the model_data 
model_data['portfolio'] = np.nan
#model_data.loc[model_data['credit_spread_start'] > 0.1, 'portfolio'] = 'DI'
model_data.loc[model_data['distressed_rating_start'] == True , 'portfolio'] = 'DI'
model_data.loc[model_data['portfolio'].isnull() & (model_data['rating_class_start'] == '0.IG'), 'portfolio'] = 'IG'
model_data.loc[model_data['portfolio'].isnull() & (model_data['rating_class_start'] == '1.HY'), 'portfolio'] = 'HY'

# =====================================================================    
#             b. Calculate raw returns between start of sample and 2016  
# =====================================================================

# Calculating returns for portfolios
groups = model_data['portfolio'].dropna().unique().tolist()

# Initialize storage for returns
group_ret = {grp: [] for grp in groups}
date_series = []

# Get list of unique month-end dates
dates = sorted(model_data['eom'].dropna().unique())

# Loop through each month
for dt in dates:
    current_period = model_data[model_data['eom'] == dt]

    # Skip if no data
    if current_period.empty:
        continue

    date_series.append(dt)

    # Compute returns raw returns for each group
    for grp in groups:
        group_data = current_period[current_period['portfolio'] == grp]
        total_mv = group_data['market_value_start'].sum()

        if len(group_data) > 0 and total_mv > 0:
            weights = group_data['market_value_start'] / total_mv
            weighted_return = (weights * group_data['ret']).sum()
        else:
            weighted_return = 0

        group_ret[grp].append(weighted_return)

raw_returns_df = pd.DataFrame(group_ret, index=pd.to_datetime(date_series))
raw_returns_subset = raw_returns_df.loc[raw_returns_df.index < '2017-01-01']
print(raw_returns_subset.mean())

# # ===================================================================    
# #             b. Calculate raw returns in Warga  
# # ===================================================================

# warga_data = pd.read_csv(data_folder + "/preprocessed/bond_warga_data.csv")

# # Creating portfolios
# warga_data['eom'] = pd.to_datetime(warga_data['eom'])
# unique_months_warga = warga_data['eom'].unique()

# # Add portfolio to the model_data 
# warga_data['portfolio'] = np.nan
# #warga_data.loc[warga_data['credit_spread_start'] > 0.1, 'portfolio'] = 'DI'
# warga_data.loc[warga_data['distressed_rating_start'] == True , 'portfolio'] = 'DI'
# warga_data.loc[warga_data['portfolio'].isnull() & (warga_data['rating_class_start'] == '0.IG'), 'portfolio'] = 'IG'
# warga_data.loc[warga_data['portfolio'].isnull() & (warga_data['rating_class_start'] == '1.HY'), 'portfolio'] = 'HY'

# # Calculating returns for portfolios
# groups_warga = warga_data['portfolio'].dropna().unique().tolist()

# # Initialize storage for returns
# group_ret_warga = {grp: [] for grp in groups}
# date_series_warga = []

# # Get list of unique month-end dates
# dates_warga = sorted(warga_data['eom'].dropna().unique())

# # Loop through each month
# for dt in dates_warga:
#     current_period_warga = warga_data[warga_data['eom'] == dt]

#     # Skip if no data
#     if current_period_warga.empty:
#         continue

#     date_series_warga.append(dt)

#     # Compute returns raw returns for each group
#     for grp in groups_warga:
#         group_data = current_period_warga[current_period_warga['portfolio'] == grp]
#         total_mv = group_data['market_value_start'].sum()

#         if len(group_data) > 0 and total_mv > 0:
#             weights = group_data['market_value_start'] / total_mv
#             weighted_return = (weights * group_data['ret']).sum()
#         else:
#             weighted_return = 0

#         group_ret_warga[grp].append(weighted_return)

# raw_returns_warga_df = pd.DataFrame(group_ret_warga, index=pd.to_datetime(date_series_warga))
# print(raw_returns_warga_df.mean())
