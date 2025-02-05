import pandas as pd

# Read the CSV file
bond_return_data = pd.read_csv("data/raw/bond_returns.csv")

# View the first 5 rows
print("First 5 rows of the data:")
print(bond_return_data.head())

print("Printing first column:")
print(bond_return_data["eom"])

#Load descriptive statistics


#Create initial diagrams