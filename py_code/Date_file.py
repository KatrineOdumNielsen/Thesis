#Set up directory
import os



project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"
print("This is my current directory:\n")
print(os.getcwd())
print(os.getcwd()+"/data")

# In[3]:


import sys
sys.path.insert(0,os.getcwd())


# In[5]:


# numpy for working with matrices, etc. 
import numpy as np

# import pandas
import pandas as pd

# import tabulate for tables
#from tabulate import tabulate

# plotting libraries
import matplotlib.pyplot as plt
#import seaborn as sns


# In[27]:


#Importing data
Data = pd.read_csv(data_folder + "/raw/bond_returns.csv")
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

# Function to look up the latest rating for a given COMPLETE_CUSIP
def get_average_latest_rating_for_cusip(ratings_df, date, cusip):
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
    latest_ratings = cusip_ratings_df.groupby('RATING_TYPE').last().reset_index()
    
    # Calculate the average numerical rating
    average_rating = latest_ratings['NUMERICAL_RATING'].mean()
    
    return average_rating

func_test = get_average_latest_rating_for_cusip(Ratings, "2003-06-14", "37042GUT1")
print(func_test)

Ratings.head()


# In[50]:


Data = Data[0:100]


# In[52]:


def get_rating_for_row(row):
    cusip = row['cusip']
    date = row['eom']
    average_rating = get_average_latest_rating_for_cusip(Ratings, date, cusip)
    return average_rating

# Apply the function
Data['average_rating'] = Data.apply(get_rating_for_row, axis=1)

# --- Step 4: Review Results ---

Data.head()


# In[61]:


#Load price data
Prices = pd.read_csv(data_folder + "/raw/bond_prices.csv")
Prices['eom'] = pd.to_datetime(Prices['DATE'])
Prices.head()


# In[109]:


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


# In[77]:


# Apply the function to each row in Data
Data['price_eom'] = Data.apply(
    lambda row: get_price_given_cusip(Prices, row['eom'], row['cusip']), axis=1
)

Data.head()

Data.to_csv(data_folder + "/processed/bond_returns_with_ratings.csv")

# In[137]:


Prices[Prices['CUSIP'] == '031162BQ2']