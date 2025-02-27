import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Set the working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"

# Read the CSV files
bond_return_data = pd.read_csv(data_folder + "/preprocessed/final_bond_data.csv")
bond_rating_data = pd.read_csv(data_folder + "/raw/full_wrds_data.csv")

# Rename columns
bond_rating_data.rename(columns={'CUSIP': 'cusip'}, inplace=True)
bond_rating_data.rename(columns={'DATE': 'eom'}, inplace=True)

# Create subsets of the dataframes
bond_return_data_subset = bond_return_data.iloc[:, :8]
bond_rating_data_subset = bond_rating_data.iloc[:, [0, 2, 6, 7,8,9,10,11,15,16,19]]

# Convert the date columns to datetime
bond_rating_data_subset['eom'] = pd.to_datetime(bond_rating_data['eom'])
bond_return_data_subset['eom'] = pd.to_datetime(bond_return_data_subset['eom'])

# Merge the two dataframes
print("Merging data")
bond_data = pd.merge(bond_return_data_subset, bond_rating_data_subset, on=["cusip", "eom"])
bond_data.rename(columns={'OFFERING_AMT': 'size'}, inplace=True)
bond_data.rename(columns={'RATING_NUM': 'rating_num'}, inplace=True)
bond_data.rename(columns={'RATING_CAT': 'rating_cat'}, inplace=True)
bond_data.rename(columns={'RATING_CLASS': 'rating_class'}, inplace=True)
bond_data.rename(columns={'BOND_TYPE': 'bond_type'}, inplace=True)
bond_data.rename(columns={'COUPON': 'coupon'}, inplace=True)
bond_data.rename(columns={'COUPAMT': 'coupamt'}, inplace=True)
bond_data.rename(columns={'TMT': 'tmt'}, inplace=True)

# Also create a flag for distressed bonds (DI) (a subset of HY)
bond_data['distressed'] = bond_data['rating_num'] >= 18.5

bond_data.to_csv("data/preprocessed/bond_data.csv", index=False)

print("Data merged and saved to data/preprocessed/bond_data.csv")