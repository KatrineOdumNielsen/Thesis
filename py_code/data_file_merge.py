import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Set the working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"

# Read the CSV files
bond_return_data = pd.read_csv("data/preprocessed/final_bond_data.csv")
bond_rating_data = pd.read_csv(data_folder + "/raw/wrds_data.csv")

# Rename columns
bond_rating_data.rename(columns={'CUSIP': 'cusip'}, inplace=True)
bond_rating_data.rename(columns={'DATE': 'eom'}, inplace=True)

# Create subsets of the dataframes
bond_return_data_subset = bond_return_data.iloc[:, :8]
bond_rating_data_subset = bond_rating_data.iloc[:, [0, 2, 6, 9, 10]]

# Convert the date columns to datetime
bond_rating_data_subset['eom'] = pd.to_datetime(bond_rating_data['eom'])
bond_return_data_subset['eom'] = pd.to_datetime(bond_return_data_subset['eom'])

# Merge the two dataframes
print("Merging data")
bond_data = pd.merge(bond_return_data_subset, bond_rating_data_subset, on=["cusip", "eom"])
bond_data.rename(columns={'OFFERING_AMT': 'size'}, inplace=True)
bond_data.rename(columns={'RATING_NUM': 'rating_num'}, inplace=True)
bond_data.rename(columns={'RATING_CAT': 'rating_cat'}, inplace=True)
bond_data.to_csv("data/preprocessed/bond_data.csv", index=False)
print("Merged data saved to data/preprocessed/bond_data.csv")