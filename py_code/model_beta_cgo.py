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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import seaborn as sns

#Set up working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"
figures_folder = project_dir + "/figures"

# Importing the cleaned data
bond_data = pd.read_csv(data_folder + "/preprocessed/bond_data.csv")
model_data = bond_data[['eom', 'cusip', 'ret_exc', 'credit_spread_past', 'rating_class_past', 'market_value_past', 'price_eom_past']]
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
    month_data.loc[(month_data['portfolio'].isnull()) & (month_data['rating_class_past'] == '0.IG'), 'portfolio'] = 'IG'
    month_data.loc[(month_data['portfolio'].isnull()) & (month_data['rating_class_past'] == '1.HY'), 'portfolio'] = 'HY'
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
model_data.loc[model_data['portfolio'].isnull() & (model_data['rating_class_past'] == '0.IG'), 'portfolio'] = 'IG'
model_data.loc[model_data['portfolio'].isnull() & (model_data['rating_class_past'] == '1.HY'), 'portfolio'] = 'HY'
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
returns_merged.to_csv("data/preprocessed/returns_merged.csv", index=False)

# ==================== Compute betas for each portfolio ============================
# --- Helper Function: Compute Beta over a Given Window of Monthly Returns ---
def compute_beta(window_df, min_months=12):
    """
    Given a DataFrame window_df with columns:
      'weighted_return' (portfolio returns) and 'market_return' (market returns),
    compute beta as:
         beta = Cov(portfolio returns, market returns) / Var(market returns)
    If there are fewer than min_months observations or if the market variance is zero,
    return np.nan.
    """
    pr = window_df['weighted_return'].values
    mr = window_df['market_return'].values
    if len(mr) < min_months or np.var(mr, ddof=1) == 0:
        return np.nan
    cov = np.cov(pr, mr, ddof=1)[0, 1]
    var_market = np.var(mr, ddof=1)
    return cov / var_market

# --- Main Loop: Compute Rolling Beta for Each Portfolio Using Historical Data ---
# Ensure returns_merged is sorted by 'eom'
returns_merged = returns_merged.sort_values('eom')

beta_records = []
# Loop over each portfolio separately.
for portfolio in returns_merged['portfolio'].unique():
    # Subset data for the current portfolio and sort by eom.
    df_port = returns_merged[returns_merged['portfolio'] == portfolio].copy().sort_values('eom').reset_index(drop=True)
    # Loop over each month (by index and date)
    for i, current_date in enumerate(df_port['eom']):
        if i < 12:
            # Skip the first 12 months (no full 12-month history available).
            continue
        if i < 60:
            # For months 13 up to 60, use all available historical months.
            window_df = df_port.iloc[0:i]
        else:
            # For month 61 and later, use only the previous 60 months.
            window_df = df_port.iloc[i-60:i]
        n_months = len(window_df)
        # Compute beta using the available window (minimum required is 12 months).
        beta_val = compute_beta(window_df, min_months=12)
        beta_records.append({
            'portfolio': portfolio,
            'eom': current_date,
            'beta': beta_val,
            'n_months': n_months
        })

beta_df = pd.DataFrame(beta_records)
print("Rolling Beta DataFrame (variable window up to 60 months):")
print(beta_df)

# --- Plot the Rolling Betas for Each Portfolio ---
cmap = cm.get_cmap('GnBu', 5).reversed()

plt.figure(figsize=(10, 6))
for i, portfolio in enumerate(sorted(beta_df['portfolio'].unique())):
    sub_df = beta_df[beta_df['portfolio'] == portfolio]
    plt.plot(sub_df['eom'], sub_df['beta'], marker='o', label=portfolio, color=cmap(i+1))
plt.xlabel("End-of-Month (eom)")
plt.ylabel("Rolling Beta")
plt.title("Rolling Beta by Portfolio Over Time\n(Beta computed using past data: increasing from 12 to 60 months)")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(figures_folder + "/rolling_beta_by_portfolio.png")
plt.close()

# ===================================================================    
#                      Calculate capital gain overhang        
# ===================================================================   
monthly_turnover = 0.015 #using quarterly turnover of 4.5% from Peter's paper


# ==================== Computing purchase price and CGO ============================
def compute_effective_purchase_price_linear(group):
    """
    For a given bond (grouped by 'cusip'), compute the effective purchase price and capital gain overhang for each month
    using linear decaying weights.
    
    For the first observation, we set the effective price equal to the current price (CGO = 0%).
    For subsequent observations, we compute weights linearly as:
        weight(k) = 1 - monthly_turnover * (k - 1)
    for k = 1,2,..., where k=1 corresponds to the most recent past observation.
    The weights are then reversed (to align with past_prices order) and normalized.
    """
    group = group.sort_values('eom').copy()
    
    # For the first observation, set effective price equal to current price, and CGO = 0%
    effective_prices = [group.iloc[0]['price_eom_past']]
    cgo_values = [0.0]
    
    for i in range(1, len(group)):
        past_prices = group.iloc[:i]['price_eom_past'].values
        n = len(past_prices)
        k_values = np.arange(1, n+1)  # k = 1, 2, ..., n
        # Linear weights: most recent gets weight = 1, then decreasing linearly
        weights = 1 - monthly_turnover * (k_values - 1)
        # Reverse weights so the most recent observation gets the highest weight
        weights = weights[::-1]
        # Normalize so weights sum to 1
        weights = weights / np.sum(weights)
        
        effective_price = np.sum(weights * past_prices)
        effective_prices.append(effective_price)
        
        current_price = group.iloc[i]['price_eom_past']
        cgo = (current_price / effective_price - 1) * 100
        cgo_values.append(cgo)
    
    group['effective_price'] = effective_prices
    group['cap_gain_overhang'] = cgo_values
    return group

## To switch back to the exponential method, use the code below:
# def compute_effective_purchase_price_exponential(group):
#     group = group.sort_values('eom').copy()
#     effective_prices = [group.iloc[0]['price_eom']]
#     cgo_values = [0.0]
#     for i in range(1, len(group)):
#         past_prices = group.iloc[:i]['price_eom'].values
#         n = len(past_prices)
#         k_values = np.arange(1, n+1)
#         weights = monthly_turnover * (1 - monthly_turnover)**(k_values - 1)
#         weights = weights[::-1]
#         weights = weights / np.sum(weights)
#         effective_price = np.sum(weights * past_prices)
#         effective_prices.append(effective_price)
#         current_price = group.iloc[i]['price_eom']
#         cgo = (current_price / effective_price - 1) * 100
#         cgo_values.append(cgo)
#     group['effective_price'] = effective_prices
#     group['cap_gain_overhang'] = cgo_values
#     return group


# Assuming model_data is your DataFrame with columns: ['eom', 'cusip', 'price_eom']
print('Applying the function group-wise by bond (cusip) ...')
model_data['eom'] = pd.to_datetime(model_data['eom'])
model_data = model_data.sort_values(['cusip', 'eom'])
model_data = model_data.groupby('cusip').apply(compute_effective_purchase_price_linear)
model_data.to_csv("data/preprocessed/model_data_cgo.csv")

# Inspect the result for a single bond
first_bond = model_data.iloc[0]['cusip']
print(model_data[model_data['cusip'] == first_bond].head(10))


# ================= CGO at portfolio level, first monthly then yearly ======================
# 1. Monthly Portfolio Averages
# Group by end-of-month and portfolio, then calculate the average CGO
monthly_cgo = (
    model_data.groupby(['eom', 'portfolio'])['cap_gain_overhang']
    .mean()
    .reset_index()
)
print("Monthly Average Capital Gain Overhang by Portfolio:")
print(monthly_cgo.head())

# 2. Annual Portfolio Averages
# Extract the year from the end-of-month dates, and then group by year and portfolio
monthly_cgo['year'] = monthly_cgo['eom'].dt.year
annual_cgo = (
    monthly_cgo.groupby(['year', 'portfolio'])['cap_gain_overhang']
    .mean()
    .reset_index()
)
print("Annual Average Capital Gain Overhang by Portfolio:")
print(annual_cgo.head())
annual_cgo.to_csv(os.path.join("data", "preprocessed", "annual_cap_gain_overhang.csv"), index=False)



# ==================== Checking the accuracy of the calculations ============================
# # -------------------------------
# # 1. Descriptive Statistics
# # -------------------------------
# print("Summary Statistics for Capital Gain Overhang:")
# print(model_data['cap_gain_overhang'].describe())
# print("Median Capital Gain Overhang:", model_data['cap_gain_overhang'].median())

# # -------------------------------
# # 2. Distribution Visualization
# # -------------------------------

# # Histogram with KDE to inspect the overall distribution
# plt.figure(figsize=(10, 6))
# # We can pick a single color from the colormap, e.g. cmap(1)
# sns.histplot(
#     model_data['cap_gain_overhang'].dropna(),
#     bins=50,
#     kde=True,
#     color=cmap(1)  # Using one color from the reversed colormap
# )
# plt.xlabel('Capital Gain Overhang (%)')
# plt.title('Distribution of Capital Gain Overhang (using price_eom_past)')
# plt.tight_layout()
# plt.show()

# # Boxplot by Portfolio to see differences across portfolios
# unique_portfolios = sorted(model_data['portfolio'].dropna().unique())
# palette_colors = [cmap(i+1) for i in range(len(unique_portfolios))]

# plt.figure(figsize=(10, 6))
# sns.boxplot(
#     data=model_data,
#     x='portfolio',
#     y='cap_gain_overhang',
#     order=unique_portfolios,       # ensure consistent order
#     palette=palette_colors         # use our custom palette
# )
# plt.title('Capital Gain Overhang by Portfolio')
# plt.xlabel('Portfolio')
# plt.ylabel('Capital Gain Overhang (%)')
# plt.tight_layout()
# plt.show()

# # -------------------------------
# # 3. Time Series Visualization
# # -------------------------------

# # Calculate the average capital gain overhang for each month and portfolio.
# portfolio_cgo = model_data.groupby(['eom', 'portfolio'])['cap_gain_overhang'].mean().reset_index()
# unique_portfolios = sorted(portfolio_cgo['portfolio'].dropna().unique())
# palette_colors = [cmap(i+1) for i in range(len(unique_portfolios))]

# plt.figure(figsize=(12, 6))
# for i, portfolio in enumerate(unique_portfolios):
#     sub_df = portfolio_cgo[portfolio_cgo['portfolio'] == portfolio]
#     plt.plot(
#         sub_df['eom'], sub_df['cap_gain_overhang'],
#         marker='o',
#         label=portfolio,
#         color=palette_colors[i]
#     )
# plt.xlabel('Date (eom)')
# plt.ylabel('Average Capital Gain Overhang (%)')
# plt.title('Monthly Average Capital Gain Overhang by Portfolio')
# plt.xticks(rotation=45)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # -------------------------------
# # 4. (Optional) Annual Aggregation
# # -------------------------------

# # For annual averages, we can further group by year.
# unique_portfolios = sorted(annual_cgo['portfolio'].dropna().unique())
# palette_colors = [cmap(i+1) for i in range(len(unique_portfolios))]
# plt.figure(figsize=(12, 6))
# sns.barplot(
#     data=annual_cgo,
#     x='year',
#     y='cap_gain_overhang',
#     hue='portfolio',
#     hue_order=unique_portfolios,
#     palette=palette_colors
# )
# plt.xlabel('Year')
# plt.ylabel('Average Capital Gain Overhang (%)')
# plt.title('Annual Average Capital Gain Overhang by Portfolio')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()