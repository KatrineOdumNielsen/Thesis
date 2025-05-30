# =============================================================================
#
#                     Part 2: Descriptive statistics
#
#             (Considers different datasets and rating groups)
#
# =============================================================================
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
from scipy import stats

#Set the working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"
figures_folder = project_dir + "/figures"

# ================  Loading the necessary data  ================
bond_data = pd.read_csv("data/preprocessed/bond_data.csv")
bond_data_large = pd.read_csv("data/preprocessed/bond_data_large.csv")
bond_warga_data = pd.read_csv("data/preprocessed/bond_warga_data.csv")
avramov_dataset = pd.read_csv("data/preprocessed/avramov_dataset.csv")
avramov_dataset['eom'] = pd.to_datetime(avramov_dataset['eom'])
bond_data_large['eom'] = pd.to_datetime(bond_data_large['eom'])
bond_data['eom'] = pd.to_datetime(bond_data['eom'])
bond_warga_data['eom'] = pd.to_datetime(bond_warga_data['eom'])

model_data = bond_data[['eom', 'cusip', 'ret', 'ret_exc', 'ret_texc', 'credit_spread_start', 'rating_class_start', 'market_value_start', 'price_eom', 'price_eom_start', 'offering_date', 'distressed_rating_start', 'amount_outstanding']]
model_data['eom'] = pd.to_datetime(model_data['eom'])
model_data['offering_date'] = pd.to_datetime(model_data['offering_date'])
model_data.to_csv("data/preprocessed/model_data.csv") #saving the smaller dataframe

# Creating portfolios
model_data['eom'] = pd.to_datetime(model_data['eom'])
unique_months = model_data['eom'].unique()

# Add portfolio to the model_data 
model_data['portfolio'] = np.nan
model_data.loc[model_data['credit_spread_start'] > 0.1, 'portfolio'] = 'DI'
#model_data.loc[model_data['distressed_rating_start'] == True , 'portfolio'] = 'DI'
model_data.loc[model_data['portfolio'].isnull() & (model_data['rating_class_start'] == '0.IG'), 'portfolio'] = 'IG'
model_data.loc[model_data['portfolio'].isnull() & (model_data['rating_class_start'] == '1.HY'), 'portfolio'] = 'HY'

# =============================================================================     
#               Descriptive statistics for different datasets          
# =============================================================================                                                          
#Load descriptive statistics
print("Descriptive statistics:")
descriptive_variables = ['size', 'ret', 'ret_texc', 'duration', 'market_value_start', 
                         'maturity_years','credit_spread_start', 'rating_num_start','amount_outstanding']

# Descriptive statistics for the clean dataset
summary_stats = bond_data[descriptive_variables].describe(include='all')
#print(summary_stats)
summary_stats.to_csv("data/other/summary_statistics.csv")

# Descriptive statistics for the clean dataset, using only model parameters
model_stats = model_data.describe(include='all')
model_stats.to_csv("data/other/model_summary_statistics.csv")

# Descriptive statistics for the large dataset
summary_stats_large = bond_data_large.describe(include='all')
model_stats.to_csv("data/other/summary_statistics_large.csv")

# Warga data only
descriptive_variables_warga = ['size', 'ret', 'market_value_start', 'maturity_years',
                               'credit_spread_start', 'rating_num_start']
print("summary stats for warga:", bond_warga_data[descriptive_variables_warga].describe(include='all'))

# Full dataset including Warga, clean data, and WRDS
print("summary stats including avramov_dataset:", avramov_dataset[descriptive_variables_warga].describe(include='all'))

# Average bond size across all bonds
grouped = bond_data.groupby('cusip')
unique_sizes = grouped['size'].first()
average_bond_size = unique_sizes.mean()
print("Average bond size (initial offering value):", average_bond_size)

# Proportion senior vs subordinated
proportions = bond_data['security_level'].value_counts(normalize=True)
print(proportions)

# =============================================================================     
#       Descriptive statistics for different rating groups (clean dataset)          
# =============================================================================                                                          
bond_data['portfolio'] = np.nan
bond_data.loc[bond_data['credit_spread_start'] > 0.1, 'portfolio'] = 'DI'
#model_data.loc[model_data['distressed_rating_start'] == True , 'portfolio'] = 'DI'
bond_data.loc[bond_data['portfolio'].isnull() & (bond_data['rating_class_start'] == '0.IG'), 'portfolio'] = 'IG'
bond_data.loc[bond_data['portfolio'].isnull() & (bond_data['rating_class_start'] == '1.HY'), 'portfolio'] = 'HY'

# Divide into groups based on the assigned portfolio
IG = bond_data[bond_data['portfolio'] == 'IG']
HY = bond_data[bond_data['portfolio'] == 'HY']
DI = bond_data[bond_data['portfolio'] == 'DI']

# Descriptive statistics for each group
IG_stats = IG[descriptive_variables].describe()
HY_stats = HY[descriptive_variables].describe()
DI_stats = DI[descriptive_variables].describe()

# Export to CSV
IG_stats.to_csv("data/other/IG_summary_statistics.csv")
HY_stats.to_csv("data/other/HY_summary_statistics.csv")
DI_stats.to_csv("data/other/DI_summary_statistics.csv")

# =============================================================================  
#      Time Series: Compute the monthly average values (clean dataset)
# =============================================================================  
# Listing the columns to average
cols_to_average = ['duration', 'yield', 'market_value_start', 'ret', 
                   'credit_spread_start', 'rating_num_start']

# Group by the end-of-month date and calculate the mean for each group
time_series_avg = bond_data.groupby('eom')[cols_to_average].mean(numeric_only=True)

# Reset the index if you want 'eom' as a column
time_series_avg = time_series_avg.reset_index()
variables = ['duration', 'yield', 'ret', 'market_value_start', 'credit_spread_start', 'rating_num_start']

# Create a 2x3 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharex=True)
axes = axes.flatten()

for i, var in enumerate(variables):
    axes[i].plot(time_series_avg['eom'], time_series_avg[var], linestyle='-', color='cornflowerblue')  
    axes[i].set_title(f'Average {var} over time')
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel(var)
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "time_series_avg.png"))
plt.close()

# =================================================================================================
#     Market-weighted cumulative return for each rating group using the clean dataset      
# =================================================================================================                                               
#Calculating returns for portfolios
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

    # Compute returns (texc) for each group
    for grp in groups:
        group_data = current_period[current_period['portfolio'] == grp]
        total_mv = group_data['market_value_start'].sum()

        if len(group_data) > 0 and total_mv > 0:
            weights = group_data['market_value_start'] / total_mv
            weighted_return = (weights * group_data['ret']).sum()
        else:
            weighted_return = 0

        group_ret[grp].append(weighted_return)

# 2. Create DataFrames for Returns
returns_df = pd.DataFrame(group_ret, index=date_series)  
mean_returns = returns_df.mean()
std_returns = returns_df.std()
print("Mean returns:", mean_returns)
print("Standard deviation of returns:", std_returns)
returns_df.index.name = 'date'

# Add a row on 2002-07-31 with values of 0 for all columns
first_row = pd.DataFrame({'IG': [0], 'HY': [0], 'DI': [0]}, index=[pd.Timestamp("2002-07-31")])
returns_df = pd.concat([first_row, returns_df])
returns_df = returns_df.sort_index()
returns_df.index = pd.to_datetime(returns_df.index)

# Compute cumulative returns: cumulative product of (1 + monthly_return).
cumulative_df = (1 + returns_df).cumprod()  
cumulative_df = 100 * cumulative_df
first_row = pd.DataFrame({'IG': [100], 'HY': [100], 'DI': [100]}, index=[pd.Timestamp("2002-07-31")])
cumulative_df = pd.concat([first_row, cumulative_df])
cumulative_df = cumulative_df.sort_index()

# Combine Monthly and Cumulative Returns
monthly_df = returns_df.reset_index()
cum_df = cumulative_df.reset_index()

# Merge the two DataFrames on 'date'
monthly_df = monthly_df.reset_index().rename(columns={"index": "date"})
cum_df = cum_df.reset_index().rename(columns={"index": "date"})
combined_df = pd.merge(monthly_df, cum_df, on='date')

# Save Combined DataFrame to CSV
output_path = os.path.join(data_folder, "preprocessed", "market_returns_combined.csv") 
combined_df.to_csv(output_path, index=False) 
print("Combined monthly and cumulative returns saved to:", output_path)

cum_df['date'] = pd.to_datetime(cum_df['date'])
returns_df.index = pd.to_datetime(returns_df.index)

# Plot Cumulative Returns
cmap = cm.get_cmap('GnBu', 5).reversed()

plt.figure(figsize=(12, 6))
for i, col in enumerate(['IG', 'HY', 'DI']):
    plt.plot(cum_df['date'], cum_df[col], label=col, color=cmap(i+1))
plt.title("Market-Weighted Cumulative Returns by Bond Class")
plt.ylabel("Cumulative Return Index")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "cumulative_returns.png"))
plt.close()

# 8. Monthly (Simple) Returns
plt.figure(figsize=(12, 10))
for i, col in enumerate(['IG', 'HY', 'DI']):
    plt.plot(returns_df.index, returns_df[col], label=col, linestyle='-', color=cmap(i+1))
plt.title("Monthly Market-Weighted Simple Returns by Rating Class")
plt.xlabel("Date")
plt.ylabel("Monthly Return")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "monthly_returns.png"))
plt.close()

avg_returns = returns_df.mean()

for port, r in avg_returns.items():
    print(f"Average market-weighted return using small dataset for {port}: {r*100:.2f}%")

print("done with clean dataset")

# ========================================================================================     
#       Market-weighted cumulative return for each rating group using the large dataset        
# ========================================================================================                                                         
# Prepare Data
model_data_large = bond_data_large[['eom', 'cusip', 'ret', 'ret_exc', 'ret_texc', 'credit_spread_start',
                                    'rating_class_start', 'market_value_start', 'price_eom', 'price_eom_start',
                                    'offering_date', 'distressed_rating_start']].copy()
model_data_large['eom'] = pd.to_datetime(model_data_large['eom'])
model_data_large['offering_date'] = pd.to_datetime(model_data_large['offering_date'])
model_data_large.to_csv("data/preprocessed/model_data_large.csv", index=False)

# Create Portfolios
model_data_large['portfolio'] = np.nan
model_data_large.loc[model_data_large['credit_spread_start'] > 0.1, 'portfolio'] = 'DI'
model_data_large.loc[model_data_large['portfolio'].isnull() & (model_data_large['rating_class_start'] == '0.IG'), 'portfolio'] = 'IG'
model_data_large.loc[model_data_large['portfolio'].isnull() & (model_data_large['rating_class_start'] == '1.HY'), 'portfolio'] = 'HY'

groups_large = model_data_large['portfolio'].dropna().unique().tolist()
dates_large = sorted(model_data_large['eom'].dropna().unique())

# Calculate Monthly Returns
group_ret_large = {grp: [] for grp in groups_large}
date_series_large = []

for dt in dates_large:
    current_period = model_data_large[model_data_large['eom'] == dt]

    if current_period.empty:
        continue

    date_series_large.append(dt)

    for grp in groups_large:
        group_data = current_period[current_period['portfolio'] == grp]
        total_mv = group_data['market_value_start'].sum()

        if len(group_data) > 0 and total_mv > 0:
            weights = group_data['market_value_start'] / total_mv
            weighted_return = (weights * group_data['ret']).sum()
        else:
            weighted_return = 0

        group_ret_large[grp].append(weighted_return)

returns_df_large = pd.DataFrame(group_ret_large, index=pd.to_datetime(date_series_large))
returns_df_large.index.name = 'date'

first_row = pd.DataFrame({'IG': [0], 'HY': [0], 'DI': [0]}, index=[pd.Timestamp("2002-07-31")])
returns_df_large = pd.concat([first_row, returns_df_large]).sort_index()
cumulative_df_large = (1 + returns_df_large).cumprod() * 100
cumulative_df_large.index.name = 'date'
first_row_cum = pd.DataFrame({'IG': [100], 'HY': [100], 'DI': [100]}, index=[pd.Timestamp("2002-07-31")])
cumulative_df_large = pd.concat([first_row_cum, cumulative_df_large]).sort_index()

# Combine Monthly and Cumulative
monthly_df_large = returns_df_large.reset_index()
cumulative_df_large.index.name = 'date'  # Important to name before reset
cum_df_large = cumulative_df_large.reset_index()

# Merge the two DataFrames on 'date'
combined_df_large = pd.merge(monthly_df_large, cum_df_large, on='date', suffixes=('_monthly', '_cumulative'))

# Save combined DataFrame
output_path_large = os.path.join(data_folder, "preprocessed", "market_returns_large_combined.csv")
combined_df_large.to_csv(output_path_large, index=False)

print("Combined monthly and cumulative returns saved to:", output_path_large)

# Plot Cumulative Returns
cmap_large = cm.get_cmap('GnBu', 5).reversed()

plt.figure(figsize=(12, 6))
for i, col in enumerate(['IG', 'HY', 'DI']):
    plt.plot(cum_df_large['date'], cum_df_large[col], label=col, color=cmap_large(i+1))
plt.title("Market-Weighted Cumulative Returns by Bond Class (Extended Dataset)")
plt.ylabel("Cumulative Return Index")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "cumulative_returns_large.png"))
plt.close()

# Plot Monthly Returns
plt.figure(figsize=(12, 10))
for i, col in enumerate(['IG', 'HY', 'DI']):
    plt.plot(returns_df_large.index, returns_df_large[col], label=col, linestyle='-', color=cmap_large(i+1))
plt.title("Monthly Market-Weighted Simple Returns by Rating Class")
plt.xlabel("Date")
plt.ylabel("Monthly Return")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "monthly_returns_large.png"))
plt.close()

avg_returns = returns_df.mean()

for port, r in avg_returns.items():
    print(f"Average market-weighted return using large dataset for {port}: {r*100:.2f}%")

print("Done with large dataset")

# ======================================================================================     
#     Market-weighted cumulative return for each rating group starting January 2004      
# ======================================================================================                                                          
# Getting returns starting from January 2004
returns_df = pd.DataFrame(group_ret, index=date_series)  
returns_df.index = pd.to_datetime(returns_df.index)
returns_df = returns_df[returns_df.index >= pd.Timestamp("2004-01-01")]

# Add a row on 2003-12-31 with values of 0 for all columns
first_row = pd.DataFrame({'IG': [0], 'HY': [0], 'DI': [0]}, index=[pd.Timestamp("2003-12-31")])
returns_df = pd.concat([first_row, returns_df])
returns_df = returns_df.sort_index()

# Compute cumulative returns: cumulative product of (1 + monthly_return).
cumulative_df = (1 + returns_df).cumprod()  
cumulative_df = 100 * cumulative_df
first_row = pd.DataFrame({'IG': [100], 'HY': [100], 'DI': [100]}, index=[pd.Timestamp("2003-12-31")])
cumulative_df = pd.concat([first_row, cumulative_df])
cumulative_df = cumulative_df.sort_index()

# Combine Monthly and Cumulative Returns
monthly_df = returns_df.reset_index()
cum_df = cumulative_df.reset_index()
monthly_df = monthly_df.reset_index().rename(columns={"index": "date"})
cum_df = cum_df.reset_index().rename(columns={"index": "date"})
combined_df_2004 = pd.merge(monthly_df, cum_df, on='date')
cum_df['date'] = pd.to_datetime(cum_df['date'])
returns_df.index = pd.to_datetime(returns_df.index)

# Plot Cumulative Returns
cmap = cm.get_cmap('GnBu', 5).reversed()

plt.figure(figsize=(12, 6))
for i, col in enumerate(['IG', 'HY', 'DI']):
    plt.plot(cum_df['date'], cum_df[col], label=col, color=cmap(i+1), linestyle='--')
plt.title("Market-Weighted Cumulative Returns by Bond Class (starting 2004)")
plt.ylabel("Cumulative Return Index")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "cumulative_returns_2004.png"))
plt.close()

avg_returns = returns_df.mean()

for port, r in avg_returns.items():
    print(f"Average market-weighted return using dataset starting 2004 for {port}: {r*100:.2f}%")

print("done with 2004 dataset")