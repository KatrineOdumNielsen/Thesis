#Set up directory
import os
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"
print("This is my current directory:\n")
print(os.getcwd())
print(os.getcwd()+"/data")


import sys
sys.path.insert(0,os.getcwd())
import numpy as np
import pandas as pd
#from tabulate import tabulate
import matplotlib.pyplot as plt
#import seaborn as sns


#Importing data
print("importing data")
Data = pd.read_csv(data_folder + "/raw/bond_returns.csv")
Data.rename(columns={"cusip": "COMPLETE_CUSIP"}, inplace=True)
Data['eom'] = pd.to_datetime(Data['eom'])
Ratings = pd.read_csv(data_folder + "/raw/rating_data.csv")
Ratings['RATING_DATE'] = pd.to_datetime(Ratings['RATING_DATE'])

rating_map = {
    'AAA': 1, 'Aaa': 1,
    'AA+': 2, 'Aa1': 2,
    'AA': 3, 'Aa2': 3,
    'AA-': 4, 'Aa3': 4,
    'A+': 5, 'A1': 5,
    'A': 6, 'A2': 6,
    'A-': 7, 'A3': 7,
    'BBB+': 8, 'Baa1': 8,
    'BBB': 9, 'Baa2': 9,
    'BBB-': 10, 'Baa3': 10,
    'BB+': 11, 'Ba1': 11,
    'BB': 12, 'Ba2': 12,
    'BB-': 13, 'Ba3': 13,
    'B+': 14, 'B1': 14,
    'B': 15, 'B2': 15,
    'B-': 16, 'B3': 16,
    'CCC+': 17, 'Caa1': 17,
    'CCC': 18, 'Caa2': 18,
    'CCC-': 19, 'Caa3': 19,
    'CC': 20, 'Ca': 20,
    'C': 21, 'D': 21, 'DD': 21, 'DDD': 21,
}
#Nummerical ratings
Ratings['NUMERICAL_RATING'] = Ratings['RATING'].map(rating_map)

# Creating separate DataFrames for each rating type
ratings_FR = Ratings[Ratings['RATING_TYPE'] == 'FR']
ratings_MR = Ratings[Ratings['RATING_TYPE'] == 'MR']
ratings_SR = Ratings[Ratings['RATING_TYPE'] == 'SPR']

# Create a new column in Data to store the average rating
Data['num_ratings_FR'] = np.nan
Data['num_ratings_MR'] = np.nan
Data['num_ratings_SR'] = np.nan

print("looking up latest rating")
# Function to look up the latest rating for a given COMPLETE_CUSIP
def get_latest_rating_for_cusip(ratings_df, date, cusip):
    # Filter the DataFrame for the specific COMPLETE_CUSIP
    cusip_ratings_df = ratings_df[ratings_df['COMPLETE_CUSIP'] == cusip]
    
    # Check if there are any ratings for the CUSIP
    if cusip_ratings_df.empty:
        print(f"No ratings found for CUSIP {cusip} up to {date}")
        return None
    
    # Sort the DataFrame by RATING_DATE
    cusip_ratings_df = cusip_ratings_df.sort_values(by='RATING_DATE')
    
    # Filter the DataFrame to include only dates up to the specified date
    cusip_ratings_df = cusip_ratings_df[cusip_ratings_df['RATING_DATE'] <= date]
    
    # Check again if the DataFrame has entries after date filtering
    if cusip_ratings_df.empty:
        print(f"No ratings found for CUSIP {cusip} up to {date}")
        return None
    
    # Get the latest rating for each rating agency
    latest_ratings = cusip_ratings_df.iloc[-1].reset_index()

    return latest_ratings

func_test = get_latest_rating_for_cusip(ratings_FR, "2003-06-14", "37042GUT1")
print(func_test)

Data = Data[0:50]


print("merging ratings to return dataset")
def get_rating_for_row(row, rating_type):
    cusip = row['COMPLETE_CUSIP']
    date = row['eom']
    rating = get_latest_rating_for_cusip(rating_type, date, cusip)
    return rating

# Apply the function
Data['num_ratings_FR'] = Data.apply(lambda row: get_rating_for_row(row, ratings_FR), axis=1)
print("showing ratings.fr")
print(Data.head())

#Load price data
Prices = pd.read_csv(data_folder + "/raw/bond_prices.csv")
Prices['eom'] = pd.to_datetime(Prices['DATE'])
Prices.head()



#Function to find price
def get_price_given_cusip(prices_df, date, cusip):
    # Filter the DataFrame for the specific CUSIP
    cusip_prices_df = prices_df[prices_df['CUSIP'] == cusip]
    #Check if there is a price for the CUSIP
    if cusip_prices_df.empty:
        print(f"No price found for CUSIP {cusip}")
        return None
    
    #Filter the DataFrame for the specific date
    price_on_date_df = cusip_prices_df[cusip_prices_df['eom'] == date]
        #Check if there is a price for the CUSIP
    if price_on_date_df.empty:
        print(f"No price found for CUSIP {cusip} on date {date}")
        return None
        
    return price_on_date_df['PRICE_EOM'].iloc[0]

get_price_given_cusip(Prices,'2022-07-31','00036AAB17')


# Apply the function to each row in Data
print("merging prices with return dataset")
Data['price_eom'] = Data.apply(
    lambda row: get_price_given_cusip(Prices, row['eom'], row['cusip']), axis=1
)

Data.head()

print("exporting data")
Data.to_csv(data_folder + "/preprocessed/bond_returns_with_ratings.csv")

# In[137]:


Prices[Prices['CUSIP'] == '031162BQ2']