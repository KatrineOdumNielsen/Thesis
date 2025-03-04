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
import seaborn as sns
from scipy import stats

#Set the working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"

# Read the CSV file
bond_data = pd.read_csv("data/preprocessed/bond_data.csv")

print("Checking that it is in a dataframe format:")
print(type(bond_data))

# View the first 5 rows
print("First 5 rows of the data:")
print(bond_data.head())

print("Printing first column:")
print(bond_data["eom"])

# =============================================================================     
#       Descriptive statistics for entire dataset          
# =============================================================================                                                          

#Load descriptive statistics
print("Descriptive statistics:")
summary_stats = bond_data.describe()
print(summary_stats)

# Save the summary statistics to a CSV file
summary_stats.to_csv("data/other/summary_statistics.csv")

# Average bond size across all bonds
grouped = bond_data.groupby('cusip')
unique_sizes = grouped['size'].first()
average_bond_size = unique_sizes.mean()
print("Average bond size (initial offering value):", average_bond_size)

# Average values of variables at two data points (midway and end)
bond_data['eom'] = pd.to_datetime(bond_data['eom'])
earliest_date = pd.to_datetime('2002-07-31')
last_date = bond_data['eom'].max()
midway_date = earliest_date + (last_date - earliest_date) / 2

midway = pd.to_datetime('2012-03-31')
enddate = pd.to_datetime('2021-11-30')

# Filter rows for each snapshot
snapshot_2013 = bond_data[bond_data['eom'] == midway]
snapshot_2023 = bond_data[bond_data['eom'] == enddate]

# List the columns you want to average
cols_to_average = ['duration', 'yield', 'market_value', 'ret_exc', 
                   'ret_texc', 'rating_num']

# Compute the averages at each snapshot
snapshot_2002_avg = snapshot_2013[cols_to_average].mean(numeric_only=True)
snapshot_2023_avg = snapshot_2023[cols_to_average].mean(numeric_only=True)

print("Averages at 2002-07-31:")
print(snapshot_2002_avg)
print("\nAverages at 2023-12-31:")
print(snapshot_2023_avg)

# Compute the 1st (Q1) and 3rd (Q3) quartiles and the IQR
Q1 = bond_data['ret_exc'].quantile(0.25)
Q3 = bond_data['ret_exc'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers_iqr = bond_data[(bond_data['ret_exc'] < lower_bound) | (bond_data['ret_exc'] > upper_bound)]
print("Number of outliers using IQR method:", len(outliers_iqr))
print("Outliers (IQR method):")
print(outliers_iqr)
print(outliers_iqr.describe())
outliers_iqr.to_csv("data/other/outliers_iqr.csv")

# ---------------------------------------------------
# Time Series: Compute the monthly average values
# ---------------------------------------------------
# Group by the end-of-month date and calculate the mean for each group
time_series_avg = bond_data.groupby('eom')[cols_to_average].mean(numeric_only=True)

# Reset the index if you want 'eom' as a column
time_series_avg = time_series_avg.reset_index()

# Optionally, print or plot the time series
print("\nTime series of average values (first 5 rows):")
print(time_series_avg.head())

# If you want to plot one variable (e.g., average duration) over time:
# Assuming time_series_avg is your DataFrame with an 'eom' column and the variables in cols_to_average
variables = ['duration', 'yield', 'market_value', 'ret_exc', 'ret_texc', 'rating_num']

# Create a 2x3 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharex=True)
axes = axes.flatten()  # flatten to easily iterate over axes

for i, var in enumerate(variables):
    axes[i].plot(time_series_avg['eom'], time_series_avg[var], linestyle='-', color='blue')  
    axes[i].set_title(f'Average {var.capitalize()} Over Time')
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel(var.capitalize())
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# =============================================================================     
#       Descriptive statistics for different rating groups          
# =============================================================================                                                          

# Divide into groups based on the rating type
IG = bond_data[bond_data['rating_num'] < 10.5]
HY = bond_data[bond_data['rating_num'] >= 10.5]
DI = bond_data[bond_data['rating_num'] >= 17.5]

# descriptive statistics for each group
IG_stats = IG.describe()
HY_stats = HY.describe()
DI_stats = DI.describe()
IG_stats.to_csv("data/other/IG_summary_statistics.csv")
HY_stats.to_csv("data/other/HY_summary_statistics.csv")
DI_stats.to_csv("data/other/DI_summary_statistics.csv")

# =============================================================================     
#       Market-weighted cumulative return for each rating group        
# =============================================================================                                                          

# Identify the unique dates in the dataset (assumed monthly frequency)
dates = sorted(bond_data['eom'].unique())

# Define the groups we want to analyze:
groups = ['0.IG', '1.HY']
# And we want to show cumulative returns for distressed bonds separately as well.
# For DI, we use only the bonds flagged as distressed.
# We'll calculate three series: IG, HY (all HY bonds) and DI (distressed subset).

# Initialize dictionaries to store monthly market-weighted returns for each group
group_returns = {grp: [] for grp in groups}
di_returns = []  # For distressed bonds
date_series = []

# Loop over each month (each date)
for dt in dates:
    date_series.append(dt)
    current_period = bond_data[bond_data['eom'] == dt]
    
    # For each group (IG and HY), compute the market-weighted return.
    for grp in groups:
        group_data = current_period[current_period['rating_class'] == grp]
        total_mv = group_data['market_value'].sum()
        
        if len(group_data) > 0 and total_mv > 0:
            weights = group_data['market_value'] / total_mv
            weighted_return = (weights * group_data['ret_exc']).sum()
        else:
            weighted_return = 0
        
        group_returns[grp].append(weighted_return)
    
    # For distressed bonds: subset of HY where is_distressed == True.
    distressed_data = current_period[current_period['distressed_rating'] == True]
    total_mv_di = distressed_data['market_value'].sum()
    if len(distressed_data) > 0 and total_mv_di > 0:
        weights_di = distressed_data['market_value'] / total_mv_di
        weighted_return_di = (weights_di * distressed_data['ret_exc']).sum()
    else:
        weighted_return_di = 0
    di_returns.append(weighted_return_di)

# Convert the dictionaries into DataFrames for further analysis.
# Build a DataFrame for monthly returns using the computed lists.
returns_df = pd.DataFrame(group_returns, index=date_series)
returns_df['2.DI'] = di_returns
returns_df.index.name = 'date'
returns_df.reset_index(inplace=True)
returns_df.set_index('date', inplace=True)

# Compute cumulative returns: (1 + monthly_return).cumprod().
cumulative_df = (1 + returns_df).cumprod()
# Multiply by 100 if you wish to start the index at 100.
cumulative_df = 100 * cumulative_df

# Combine Monthly Returns and Cumulative Returns in One DataFrame
monthly_df = returns_df.reset_index()
cum_df = cumulative_df.reset_index()
cum_df = cum_df.rename(columns={'0.IG': 'IG_cum', '1.HY': 'HY_cum', '2.DI': 'DI_cum'})

# Merge the two DataFrames on 'date'
combined_df = pd.merge(monthly_df, cum_df, on='date')

# Save the Combined DataFrame to CSV
output_path = os.path.join(data_folder, "preprocessed", "market_returns_combined.csv")
combined_df.to_csv(output_path, index=False)
print("Combined monthly and cumulative returns saved to:", output_path)

# Plot the Cumulative Returns
#plt.figure(figsize=(12, 6))
#for col in ['IG_cum', 'HY_cum', 'DI_cum']:   #includes distressed bonds
#    plt.plot(cum_df['date'], cum_df[col], label=col)
#plt.title("Market-Weighted Cumulative Returns by Rating Group")
#plt.xlabel("Date")
#plt.ylabel("Cumulative Return Index")
#plt.legend()
#plt.xticks(rotation=45)
#plt.tight_layout()
#plt.show()

print("done")



############################ USING RET ########################################



# 3. Compute Monthly Market-Weighted Returns Using ret
# -------------------------------
# Get sorted unique month-end dates
dates = sorted(bond_data['eom'].unique())
date_series = []  # Store the dates for which we compute returns

# Define the groups for which we compute returns
groups = ['0.IG', '1.HY']  # Distressed bonds (DI) will be computed separately.

# Initialize dictionaries/lists to store returns
group_returns = {grp: [] for grp in groups}
di_returns = []  # For distressed bonds

# Loop over each month, starting from the second period
for i in range(1, len(dates)):
    dt = dates[i]
    prev_dt = dates[i - 1]  # Previous month

    date_series.append(dt)
    current_period = bond_data[bond_data['eom'] == dt]
    prev_period = bond_data[bond_data['eom'] == prev_dt]

    # For each group (IG and HY), compute the market-weighted return using prior period's weights
    for grp in groups:
        current_group_data = current_period[current_period['rating_class'] == grp]
        prev_group_data = prev_period[prev_period['rating_class'] == grp]

        # Get previous month's market values
        prev_mv = prev_group_data.set_index('cusip')['market_value'] if not prev_group_data.empty else pd.Series(dtype=float)

        # Compute weights using previous month's market values
        total_prev_mv = prev_mv.sum()
        if not prev_mv.empty and total_prev_mv > 0:
            weights = prev_mv / total_prev_mv
            weighted_return = (weights * current_group_data.set_index('cusip')['ret']).sum()  
        else:
            weighted_return = 0

        group_returns[grp].append(weighted_return) 

    # For distressed bonds: select only rows where is_distressed == True
    current_distressed_data = current_period[current_period['distressed_rating'] == True]
    prev_distressed_data = prev_period[prev_period['distressed_rating'] == True]

    prev_mv_di = prev_distressed_data.set_index('cusip')['market_value'] if not prev_distressed_data.empty else pd.Series(dtype=float)

    total_prev_mv_di = prev_mv_di.sum()
    if not prev_mv_di.empty and total_prev_mv_di > 0:
        weights_di = prev_mv_di / total_prev_mv_di
        weighted_return_di = (weights_di * current_distressed_data.set_index('cusip')['ret']).sum()  
    else:
        weighted_return_di = 0

    di_returns.append(weighted_return_di)


# -------------------------------
# 4. Create DataFrames for Returns
# -------------------------------
# Create a DataFrame for monthly returns using the computed lists.
returns_df = pd.DataFrame(group_returns, index=date_series)  
returns_df['2.DI'] = di_returns 
returns_df.index.name = 'date'
returns_df.reset_index(inplace=True)
returns_df.set_index('date', inplace=True)

# Compute cumulative returns: cumulative product of (1 + monthly_return).
cumulative_df = (1 + returns_df).cumprod()  
# Multiply by 100 if you want an index starting at 100.
cumulative_df = 100 * cumulative_df 

# -------------------------------
# 5. Combine Monthly and Cumulative Returns
# -------------------------------
monthly_df = returns_df.reset_index()
cum_df = cumulative_df.reset_index()

# Rename cumulative columns to avoid collision.
cum_df = cum_df.rename(columns={'0.IG': 'IG_cum', '1.HY': 'HY_cum', '2.DI': 'DI_cum'})

# Merge the two DataFrames on 'date'
combined_df = pd.merge(monthly_df, cum_df, on='date')

# -------------------------------
# 6. Save Combined DataFrame to CSV
# -------------------------------
output_path = os.path.join(data_folder, "preprocessed", "market_returns_combined.csv") 
combined_df.to_csv(output_path, index=False) 
print("Combined monthly and cumulative returns saved to:", output_path)

# -------------------------------
# 7. Plot Cumulative Returns
# -------------------------------
plt.figure(figsize=(12, 6))
for col in ['IG_cum', 'HY_cum', 'DI_cum']:
    plt.plot(cum_df['date'], cum_df[col], label=col)
plt.title("Market-Weighted Cumulative Returns by Rating Group (ret)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return Index")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# 8. Plot Monthly (Simple) Returns
# -------------------------------
plt.figure(figsize=(12, 6))
for col in ['0.IG', '1.HY', '2.DI']:
    plt.plot(returns_df.index, returns_df[col], label=col, linestyle='-')
plt.title("Monthly Market-Weighted Simple Returns by Rating Group (ret)")
plt.xlabel("Date")
plt.ylabel("Monthly Return")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("done")