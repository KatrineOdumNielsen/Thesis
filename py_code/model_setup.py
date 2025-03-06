# Ask Harris about the code when we set this up; how to access the model from another py file
# def model():
#     model = Sequential()
#     model.add(Dense(32, input_dim=784))
#     model.add(Activation('relu'))
#     model.add(Dense(10))
#     model.add(Activation('softmax'))

# =============================================================================
#
#                     Part X: Model Setup
#
#         (Considers only subset including the cleaned data)
#
# =============================================================================

# Importing necessary libraries
import pandas as pd
import numpy as np
import os

#Set up working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"
figures_folder = project_dir + "/figures"

# Importing the cleaned data
bond_data = pd.read_csv(data_folder + "/preprocessed/bond_data.csv")
model_data = bond_data[['eom', 'cusip', 'ret_exc', 'credit_spread_past', 'rating_class', 'market_value_past', 'price_eom']]
model_data['eom'] = pd.to_datetime(model_data['eom'])
model_data.to_csv("data/preprocessed/model_data.csv")
## HUSK AT ÆNDRE TIL PRICE_PAST####

# ===================================================================    
#                      Set up portfolios        
# ===================================================================                                                         

bond_data['eom'] = pd.to_datetime(model_data['eom'])
unique_months = model_data['eom'].unique()
portfolio_by_month = {}

for month in sorted(unique_months):
    month_data = model_data[model_data['eom'] == month].copy()
    # Assign portfolios for this month.
    # Order matters if the conditions are not mutually exclusive.
    month_data.loc[month_data['credit_spread_past'] > 0.1, 'portfolio'] = 'DI'
    month_data.loc[(month_data['portfolio'].isnull()) & (month_data['rating_class'] == '0.IG'), 'portfolio'] = 'IG'
    month_data.loc[(month_data['portfolio'].isnull()) & (month_data['rating_class'] == '1.HY'), 'portfolio'] = 'HY'
    month_data.loc[month_data['portfolio'].isnull(), 'portfolio'] = 'Other'
    
    # Store the resulting DataFrame in a dictionary keyed by the month.
    portfolio_by_month[month] = month_data

# Now you have a dictionary where for each month you have three portfolios.

month = sorted(unique_months)[0]
IG_portfolio = portfolio_by_month[month][portfolio_by_month[month]['portfolio'] == 'IG']
HY_portfolio = portfolio_by_month[month][portfolio_by_month[month]['portfolio'] == 'HY']
DI_portfolio = portfolio_by_month[month][portfolio_by_month[month]['portfolio'] == 'DI']

print(portfolio_by_month)

# ================== Add portfolio to the bond_data ===============
model_data['portfolio'] = np.nan
model_data.loc[model_data['credit_spread_past'] > 0.1, 'portfolio'] = 'DI'
model_data.loc[model_data['portfolio'].isnull() & (model_data['rating_class'] == '0.IG'), 'portfolio'] = 'IG'
model_data.loc[model_data['portfolio'].isnull() & (model_data['rating_class'] == '1.HY'), 'portfolio'] = 'HY'
model_data.to_csv("data/preprocessed/model_data.csv")

portfolio_counts = model_data['portfolio'].value_counts()
print("Portfolio counts:")
print(portfolio_counts)

# ===================================================================    
#                      Calculate betas        
# ===================================================================   

# ==================== Calculate monthly portfolio weighted returns ============================
# Function to calculate weighted return for each bond
def calculate_monthly_weighted_return(month_data):
    monthly_total_portfolio_value_past = month_data.groupby('portfolio')['market_value_past'].sum()
    month_data['portfolio_weight'] = month_data['market_value_past'] / monthly_total_portfolio_value_past
    month_data['weighted_return'] = month_data['portfolio_weight'] * month_data['ret_exc']
    return month_data


def calculate_monthly_weighted_return(month_data):
    """
    Given a DataFrame for a single month with columns:
    ['portfolio', 'market_value_past', 'ret_exc'],
    compute each bond's portfolio weight and weighted return,
    then sum over each portfolio to get final monthly returns.
    """
    # 1) For each row, get the sum of 'market_value_past' in that row's portfolio
    sum_portfolio_mv_past = month_data.groupby('portfolio')['market_value_past'].transform('sum')
    
    # 2) Compute each bond's weight within its portfolio
    month_data['portfolio_weight'] = month_data['market_value_past'] / sum_portfolio_mv_past
    
    # 3) Multiply each bond’s weight by its return
    month_data['weighted_return'] = month_data['portfolio_weight'] * month_data['ret_exc']
    
    # 4) Sum weighted returns by portfolio to get the final monthly returns
    monthly_returns = month_data.groupby('portfolio')['weighted_return'].sum()
    
    return monthly_returns

monthly_portfolio_returns = {}
for m, df in portfolio_by_month.items():
    returns_this_month = calculate_monthly_weighted_return(df)
    monthly_portfolio_returns[m] = returns_this_month

# Convert the dictionary to a DataFrame.
# The resulting DataFrame (monthly_port_ret) is in wide format,
# with the index as the month (from the dictionary keys) and columns as portfolio names.
monthly_port_ret = pd.DataFrame(monthly_portfolio_returns).T

# Set the index name to 'eom' and then reset the index to turn it into a column.
monthly_port_ret.index.name = 'eom'
monthly_port_ret = monthly_port_ret.reset_index()

# Convert the wide format DataFrame to long format:
# Each row will then have: 'eom', 'portfolio', 'weighted_return'
monthly_port_ret_long = monthly_port_ret.melt(id_vars='eom', var_name='portfolio', value_name='weighted_return')

# Save monthly portfolio returns to CSV if desired
monthly_port_ret_long.to_csv("data/preprocessed/monthly_portfolio_returns.csv", index=False)

# ==================== Calculate monthly market weighted returns ============================
# Function to calculate weighted return for each bond
def calculate_weighted_return_per_month(df):
    """
    For each month (eom), compute each bond's weight, and then the bond-level weighted return.
    """
    # 1) Sum market_value_past within each month
    monthly_sum_mv = df.groupby('eom')['market_value_past'].transform('sum')
    
    # 2) Each bond's weight is fraction of that month's total
    df['weight'] = df['market_value_past'] / monthly_sum_mv
    
    # 3) Weighted return at bond-level
    df['weighted_return'] = df['weight'] * df['ret_exc']
    
    return df

# Apply the function
model_data = calculate_weighted_return_per_month(model_data)

# Now compute the overall market return per month by summing bond-level weighted returns
market_return_df = (
    model_data
    .groupby('eom', as_index=False)['weighted_return']
    .sum()
    .rename(columns={'weighted_return': 'market_return'})
)

# If you have a separate DataFrame with monthly portfolio returns, 
# merge it on 'eom' to get a column with the overall market return.
monthly_portfolio_returns = monthly_port_ret  # your existing DataFrame with columns [eom, portfolio, ...]
monthly_portfolio_returns = monthly_portfolio_returns.reset_index()  
market_return_df = market_return_df.reset_index()

returns_merged = pd.merge(monthly_port_ret_long, market_return_df, on='eom')

# ==================== Compute betas for each portfolio ============================
betas = {}
for portfolio in returns_merged['portfolio'].unique():
    port_df = returns_merged[returns_merged['portfolio'] == portfolio]
    port_ret = port_df['weighted_return']     # portfolio's monthly return
    mkt_ret  = port_df['market_return']       # overall market return
    
    cov_matrix = np.cov(port_ret, mkt_ret)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    betas[portfolio] = beta

print("Asset Betas for each portfolio:")
print(betas)
