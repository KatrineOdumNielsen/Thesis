#Husk de her koder:
#git add .
#git commit -m "hvad du har gjort"
#git push

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
bond_return_data = pd.read_csv("data/raw/bond_returns.csv")

print("Checking that it is in a dataframe format:")
print(type(bond_return_data))

# View the first 5 rows
print("First 5 rows of the data:")
print(bond_return_data.head())

print("Printing first column:")
print(bond_return_data["eom"])

#Load descriptive statistics
print("Descriptive statistics:")
print(bond_return_data.describe())

# Visualize summary statistics using seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(data=bond_return_data, x='market_value')
plt.title('Boxplot of market value')
plt.xlabel('Market Value (Billion USD)')
plt.show()

## Runnning excess return with outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=bond_return_data, x='ret_exc')
plt.title('Boxplot of excess return')
plt.show()

## Runnning excess return without outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=bond_return_data, x='ret_exc', showliers=False)
plt.title('Boxplot of excess return')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=bond_return_data, x='ret_texc')
plt.title('Boxplot of total excess return')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=bond_return_data, x='yield')
plt.title('Boxplot of yield')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=bond_return_data, x='duration')
plt.title('Boxplot of duration')
plt.show()

#JEG TESTER DATA 

plt.figure(figsize=(10, 6))
sns.boxplot(data=bond_return_data, x='rating_group')
plt.title('Boxplot of rating group')
plt.show()

#Create initial diagrams
bond_return_data['ret_exc'].hist()
plt.title('Histogram of ret_exc')
plt.xlabel('ret_exc')
plt.ylabel('Frequency')
plt.show()