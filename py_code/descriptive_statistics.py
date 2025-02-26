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
DI = bond_data[bond_data['rating_num'] >= 18.5]

# descriptive statistics for each group
IG_stats = IG.describe()
HY_stats = HY.describe()
DI_stats = DI.describe()
IG_stats.to_csv("data/other/IG_summary_statistics.csv")
HY_stats.to_csv("data/other/HY_summary_statistics.csv")
DI_stats.to_csv("data/other/DI_summary_statistics.csv")

