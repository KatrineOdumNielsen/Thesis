#Husk de her koder:
#git pull
#git add .
#git commit -m "hvad du har gjort"
#git push

# =============================================================================
#
#                     Part X: Descriptive statistics
#
#         (Considers both the entire dataset and different rating groups)
#
# =============================================================================
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import seaborn as sns
from scipy import stats

#Set the working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"
figures_folder = project_dir + "/figures"

# Read the CSV file
bond_data = pd.read_csv("data/preprocessed/bond_data.csv")
bond_data_large = pd.read_csv("data/preprocessed/bond_data_large.csv")
bond_data['eom'] = pd.to_datetime(bond_data['eom'])

model_data = bond_data[['eom', 'cusip', 'ret', 'ret_exc', 'ret_texc', 'credit_spread_start', 'rating_class_start', 'market_value_start', 'price_eom', 'price_eom_start', 'offering_date']]
model_data['eom'] = pd.to_datetime(model_data['eom'])
model_data['offering_date'] = pd.to_datetime(model_data['offering_date'])
model_data.to_csv("data/preprocessed/model_data.csv") #saving the smaller dataframe

# =============================================================================     
#       Descriptive statistics for entire dataset          
# =============================================================================                                                          

#Load descriptive statistics
print("Descriptive statistics:")
descriptive_variables = ['size', 'ret', 'duration', 'market_value_start', 'maturity_years','credit_spread_start', 'rating_num_start']
summary_stats = bond_data[descriptive_variables].describe(include='all')
summary_stats.to_csv("data/other/summary_statistics.csv")
model_stats = model_data.describe(include='all')
model_stats.to_csv("data/other/model_summary_statistics.csv")

# Average bond size across all bonds
grouped = bond_data.groupby('cusip')
unique_sizes = grouped['size'].first()
average_bond_size = unique_sizes.mean()
print("Average bond size (initial offering value):", average_bond_size)

# Proportion senior vs subordinated
proportions = bond_data['security_level'].value_counts(normalize=True)
print(proportions)

# Listing the columns to average
cols_to_average = ['duration', 'yield', 'market_value_start', 'ret', 
                   'credit_spread_start', 'rating_num_start']

# Compute the 1st (Q1) and 3rd (Q3) quartiles and the IQR
Q1 = bond_data['ret'].quantile(0.25)
Q3 = bond_data['ret'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers_iqr = bond_data[(bond_data['ret_exc'] < lower_bound) | (bond_data['ret_exc'] > upper_bound)]
# print("Number of outliers using IQR method:", len(outliers_iqr))
# print("Outliers (IQR method):")
# print(outliers_iqr)
# print(outliers_iqr.describe())
outliers_iqr.to_csv("data/other/outliers_iqr.csv")

# ---------------------------------------------------
# Time Series: Compute the monthly average values
# ---------------------------------------------------
# Group by the end-of-month date and calculate the mean for each group
time_series_avg = bond_data.groupby('eom')[cols_to_average].mean(numeric_only=True)

# Reset the index if you want 'eom' as a column
time_series_avg = time_series_avg.reset_index()
variables = ['duration', 'yield', 'ret', 'market_value_start', 'credit_spread_start', 'rating_num_start']

# Create a 2x3 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharex=True)
axes = axes.flatten()  # flatten to easily iterate over axes

for i, var in enumerate(variables):
    axes[i].plot(time_series_avg['eom'], time_series_avg[var], linestyle='-', color='cornflowerblue')  
    axes[i].set_title(f'Average {var} over time')
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel(var)
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "time_series_avg.png"))
plt.close()

# =============================================================================     
#       Descriptive statistics for different rating groups          
# =============================================================================                                                          

# Divide into groups based on the rating type
IG = bond_data[bond_data['rating_class'] == '0.IG']
HY = bond_data[bond_data['rating_class'] == '1.HY']
DI = bond_data[bond_data['credit_spread'] > 0.1]


# descriptive statistics for each group
IG_stats = IG[descriptive_variables].describe()
HY_stats = HY[descriptive_variables].describe()
DI_stats = DI[descriptive_variables].describe()
IG_stats.to_csv("data/other/IG_summary_statistics.csv")
HY_stats.to_csv("data/other/HY_summary_statistics.csv")
DI_stats.to_csv("data/other/DI_summary_statistics.csv")

# =================================================================================================
#       Market-weighted cumulative return for each rating group using the small dataset and ret        
# =================================================================================================                                               

# 1. Compute Monthly Market-Weighted Returns Using ret
dates = sorted(bond_data['eom'].unique())
date_series = []  # Store the dates for which we compute returns
groups = ['0.IG', '1.HY']  # Distressed bonds (DI) will be computed separately.

# Initialize dictionaries/lists to store returns
group_returns = {grp: [] for grp in groups}
di_returns = []  # For distressed bonds

# Loop over each month (each date)
for dt in dates:
    date_series.append(dt)
    current_period = bond_data[bond_data['eom'] == dt]
    
    # For each group (IG and HY), compute the market-weighted return.
    for grp in groups:
        group_data = current_period[current_period['rating_class_start'] == grp]
        total_mv = group_data['market_value_start'].sum()
        
        if len(group_data) > 0 and total_mv > 0:
            weights = group_data['market_value_start'] / total_mv
            weighted_return = (weights * group_data['ret']).sum()
        else:
            weighted_return = 0
        
        group_returns[grp].append(weighted_return)
    
    # For distressed bonds: subset of HY where is_distressed == True.
    distressed_data = current_period[current_period['distressed_spread_start'] == True]
    total_mv_di = distressed_data['market_value_start'].sum()
    if len(distressed_data) > 0 and total_mv_di > 0:
        weights_di = distressed_data['market_value_start'] / total_mv_di
        weighted_return_di = (weights_di * distressed_data['ret']).sum()
    else:
        weighted_return_di = 0
    di_returns.append(weighted_return_di)


# 2. Create DataFrames for Returns
returns_df = pd.DataFrame(group_returns, index=date_series)  
returns_df['2.DI'] = di_returns 
returns_df.index.name = 'date'

# Add a row on 2002-07-31 with values of 0 for all columns
first_row = pd.DataFrame({'0.IG': [0], '1.HY': [0], '2.DI': [0]}, index=[pd.Timestamp("2002-07-31")])
returns_df = pd.concat([first_row, returns_df])
returns_df = returns_df.sort_index()
returns_df.index = pd.to_datetime(returns_df.index)

# Compute cumulative returns: cumulative product of (1 + monthly_return).
cumulative_df = (1 + returns_df).cumprod()  
cumulative_df = 100 * cumulative_df
first_row = pd.DataFrame({'0.IG': [100], '1.HY': [100], '2.DI': [100]}, index=[pd.Timestamp("2002-07-31")])
cumulative_df = pd.concat([first_row, cumulative_df])
cumulative_df = cumulative_df.sort_index()

# 3. Combine Monthly and Cumulative Returns
monthly_df = returns_df.reset_index()
cum_df = cumulative_df.reset_index()

# Rename cumulative columns to avoid collision.
cum_df = cum_df.rename(columns={'0.IG': 'IG_cum', '1.HY': 'HY_cum', '2.DI': 'DI_cum'})

# Merge the two DataFrames on 'date'
monthly_df = monthly_df.reset_index().rename(columns={"index": "date"})
cum_df = cum_df.reset_index().rename(columns={"index": "date"})
combined_df = pd.merge(monthly_df, cum_df, on='date')

# 4. Save Combined DataFrame to CSV
output_path = os.path.join(data_folder, "preprocessed", "market_returns_combined.csv") 
combined_df.to_csv(output_path, index=False) 
print("Combined monthly and cumulative returns saved to:", output_path)

cum_df['date'] = pd.to_datetime(cum_df['date'])
returns_df.index = pd.to_datetime(returns_df.index)

# 5. Plot Cumulative Returns
cmap = cm.get_cmap('GnBu', 5).reversed()

plt.figure(figsize=(12, 6))
for i, col in enumerate(['IG_cum', 'HY_cum', 'DI_cum']):
    plt.plot(cum_df['date'], cum_df[col], label=col, color=cmap(i+1))
plt.title("Market-Weighted Cumulative Returns by Bond Class")
plt.ylabel("Cumulative Return Index")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "cumulative_returns.png"))
plt.close()

# 8. Plot Monthly (Simple) Returns
plt.figure(figsize=(12, 10))
for i, col in enumerate(['0.IG', '1.HY', '2.DI']):
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

print("done with small dataset")

# ======================================================================================     
#       Market-weighted cumulative return for each rating group using the large dataset        
# ======================================================================================                                                          

# 1. Compute Monthly Market-Weighted Returns Using ret
# Get sorted unique month-end dates
dates = sorted(bond_data_large['eom'].unique())
date_series = []  # Store the dates for which we compute returns

# Define the groups for which we compute returns
groups = ['0.IG', '1.HY']  # Distressed bonds (DI) will be computed separately.

# Initialize dictionaries/lists to store returns
group_returns = {grp: [] for grp in groups}
di_returns = []  # For distressed bonds

# Loop over each month (each date)
for dt in dates:
    date_series.append(dt)
    current_period = bond_data_large[bond_data_large['eom'] == dt]
    
    # For each group (IG and HY), compute the market-weighted return.
    for grp in groups:
        group_data = current_period[current_period['rating_class_start'] == grp]
        total_mv = group_data['market_value_start'].sum()
        
        if len(group_data) > 0 and total_mv > 0:
            weights = group_data['market_value_start'] / total_mv
            weighted_return = (weights * group_data['ret']).sum()
        else:
            weighted_return = 0
        
        group_returns[grp].append(weighted_return)
    
    # For distressed bonds: subset of HY where is_distressed == True.
    distressed_data = current_period[current_period['distressed_spread_start'] == True]
    total_mv_di = distressed_data['market_value_start'].sum()
    if len(distressed_data) > 0 and total_mv_di > 0:
        weights_di = distressed_data['market_value_start'] / total_mv_di
        weighted_return_di = (weights_di * distressed_data['ret']).sum()
    else:
        weighted_return_di = 0
    di_returns.append(weighted_return_di)

# 2. Create DataFrames for Returns
# Create a DataFrame for monthly returns using the computed lists.
returns_df = pd.DataFrame(group_returns, index=date_series)  
returns_df['2.DI'] = di_returns 
returns_df.index.name = 'date'
returns_df.index = pd.to_datetime(returns_df.index)

# Add a row on 2002-07-31 with values of 0 for all columns
first_row = pd.DataFrame({'0.IG': [0], '1.HY': [0], '2.DI': [0]}, index=[pd.Timestamp("2002-07-31")])
returns_df = pd.concat([first_row, returns_df])
returns_df = returns_df.sort_index()

# Compute cumulative returns: cumulative product of (1 + monthly_return).
cumulative_df = (1 + returns_df).cumprod()  
cumulative_df = 100 * cumulative_df
first_row = pd.DataFrame({'0.IG': [100], '1.HY': [100], '2.DI': [100]}, index=[pd.Timestamp("2002-07-31")])
cumulative_df = pd.concat([first_row, cumulative_df])
cumulative_df = cumulative_df.sort_index()

# 3. Combine Monthly and Cumulative Returns
monthly_df = returns_df.reset_index()
cum_df = cumulative_df.reset_index()
cum_df = cum_df.rename(columns={'0.IG': 'IG_cum', '1.HY': 'HY_cum', '2.DI': 'DI_cum'})
monthly_df = monthly_df.reset_index().rename(columns={"index": "date"})
cum_df = cum_df.reset_index().rename(columns={"index": "date"})
combined_df_large = pd.merge(monthly_df, cum_df, on='date')

# 4. Save Combined DataFrame to CSV
output_path = os.path.join(data_folder, "preprocessed", "market_returns_combined.csv") 
combined_df_large.to_csv(output_path, index=False) 
print("Combined monthly and cumulative returns saved to:", output_path)

cum_df['date'] = pd.to_datetime(cum_df['date'])
returns_df.index = pd.to_datetime(returns_df.index)

# 5. Plot Cumulative Returns
cmap = cm.get_cmap('GnBu', 5).reversed()

plt.figure(figsize=(12, 6))
for i, col in enumerate(['IG_cum', 'HY_cum', 'DI_cum']):
    plt.plot(cum_df['date'], cum_df[col], label=col, color=cmap(i+1))
plt.title("Market-Weighted Cumulative Returns by Bond Class (Large Dataset)")
plt.ylabel("Cumulative Return Index")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "cumulative_returns_large.png"))
plt.close()

# 8. Plot Monthly (Simple) Returns
plt.figure(figsize=(12, 10))
for i, col in enumerate(['0.IG', '1.HY', '2.DI']):
    plt.plot(returns_df.index, returns_df[col], label=col, linestyle='-', color=cmap(i+1))
plt.title("Monthly Market-Weighted Simple Returns by Rating Class (Large Dataset)")
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

print("done with large dataset")

# ======================================================================================     
#       Market-weighted cumulative return for each rating group starting july 2003        
# ======================================================================================                                                          

# 1. Compute Monthly Market-Weighted Returns Using ret
# Get sorted unique month-end dates
bond_data = bond_data[bond_data['eom'] >= pd.Timestamp("2004-01-31")]
dates = sorted(bond_data['eom'].unique())
date_series = []  # Store the dates for which we compute returns

# Define the groups for which we compute returns
groups = ['0.IG', '1.HY']  # Distressed bonds (DI) will be computed separately.

# Initialize dictionaries/lists to store returns
group_returns = {grp: [] for grp in groups}
di_returns = []  # For distressed bonds

# Loop over each month (each date)
for dt in dates:
    date_series.append(dt)
    current_period = bond_data[bond_data['eom'] == dt]
    
    # For each group (IG and HY), compute the market-weighted return.
    for grp in groups:
        group_data = current_period[current_period['rating_class_start'] == grp]
        total_mv = group_data['market_value_start'].sum()
        
        if len(group_data) > 0 and total_mv > 0:
            weights = group_data['market_value_start'] / total_mv
            weighted_return = (weights * group_data['ret']).sum()
        else:
            weighted_return = 0
        
        group_returns[grp].append(weighted_return)
    
    # For distressed bonds: subset of HY where is_distressed == True.
    distressed_data = current_period[current_period['distressed_spread_start'] == True]
    total_mv_di = distressed_data['market_value_start'].sum()
    if len(distressed_data) > 0 and total_mv_di > 0:
        weights_di = distressed_data['market_value_start'] / total_mv_di
        weighted_return_di = (weights_di * distressed_data['ret']).sum()
    else:
        weighted_return_di = 0
    di_returns.append(weighted_return_di)

# 2. Create DataFrames for Returns
# Create a DataFrame for monthly returns using the computed lists.
returns_df = pd.DataFrame(group_returns, index=date_series)  
returns_df['2.DI'] = di_returns 
returns_df.index.name = 'date'
returns_df.index = pd.to_datetime(returns_df.index)

# Add a row on 2002-07-31 with values of 0 for all columns
first_row = pd.DataFrame({'0.IG': [0], '1.HY': [0], '2.DI': [0]}, index=[pd.Timestamp("2004-01-31")])
returns_df = pd.concat([first_row, returns_df])
returns_df = returns_df.sort_index()

# Compute cumulative returns: cumulative product of (1 + monthly_return).
cumulative_df = (1 + returns_df).cumprod()  
cumulative_df = 100 * cumulative_df
first_row = pd.DataFrame({'0.IG': [100], '1.HY': [100], '2.DI': [100]}, index=[pd.Timestamp("2004-01-31")])
cumulative_df = pd.concat([first_row, cumulative_df])
cumulative_df = cumulative_df.sort_index()

# 3. Combine Monthly and Cumulative Returns
monthly_df = returns_df.reset_index()
cum_df = cumulative_df.reset_index()
cum_df = cum_df.rename(columns={'0.IG': 'IG_cum', '1.HY': 'HY_cum', '2.DI': 'DI_cum'})
monthly_df = monthly_df.reset_index().rename(columns={"index": "date"})
cum_df = cum_df.reset_index().rename(columns={"index": "date"})
combined_df_2004 = pd.merge(monthly_df, cum_df, on='date')

# 4. Save Combined DataFrame to CSV
output_path = os.path.join(data_folder, "preprocessed", "market_returns_combined.csv") 
combined_df_2004.to_csv(output_path, index=False) 
print("Combined monthly and cumulative returns saved to:", output_path)

cum_df['date'] = pd.to_datetime(cum_df['date'])
returns_df.index = pd.to_datetime(returns_df.index)

# 5. Plot Cumulative Returns
cmap = cm.get_cmap('GnBu', 5).reversed()

plt.figure(figsize=(12, 6))
for i, col in enumerate(['IG_cum', 'HY_cum', 'DI_cum']):
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