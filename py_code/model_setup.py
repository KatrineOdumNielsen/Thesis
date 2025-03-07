# =============================================================================
#
#                     Part X: Model Setup A
#                   (Beta and Capital Gain overhang)
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
from scipy.stats import skew

#Set up working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"
figures_folder = project_dir + "/figures"

# Importing the cleaned data
bond_data = pd.read_csv(data_folder + "/preprocessed/bond_data.csv")
model_data = bond_data[['eom', 'cusip', 'ret_exc', 'credit_spread_past', 'rating_class_past', 'market_value_past', 'price_eom_past']]
model_data['eom'] = pd.to_datetime(model_data['eom'])
model_data.to_csv("data/preprocessed/model_data.csv") #saving the smaller dataframe

# ===================================================================    
#                     a. Set up portfolios by month        
# ===================================================================                                                         
bond_data['eom'] = pd.to_datetime(model_data['eom'])
unique_months = model_data['eom'].unique()
portfolio_by_month = {}

for month in sorted(unique_months):
    month_data = model_data[model_data['eom'] == month].copy()
    # Assign portfolios for this month based on credit spread and rating class
    month_data.loc[month_data['credit_spread_past'] > 0.1, 'portfolio'] = 'DI'
    month_data.loc[(month_data['portfolio'].isnull()) & (month_data['rating_class_past'] == '0.IG'), 'portfolio'] = 'IG'
    month_data.loc[(month_data['portfolio'].isnull()) & (month_data['rating_class_past'] == '1.HY'), 'portfolio'] = 'HY'
    month_data.loc[month_data['portfolio'].isnull(), 'portfolio'] = 'Other'
    portfolio_by_month[month] = month_data # Save the DataFrame for this month

# Add portfolio to the model_data 
model_data['portfolio'] = np.nan
model_data.loc[model_data['credit_spread_past'] > 0.1, 'portfolio'] = 'DI'
model_data.loc[model_data['portfolio'].isnull() & (model_data['rating_class_past'] == '0.IG'), 'portfolio'] = 'IG'
model_data.loc[model_data['portfolio'].isnull() & (model_data['rating_class_past'] == '1.HY'), 'portfolio'] = 'HY'
model_data.to_csv("data/preprocessed/model_data.csv")

# print("Portfolio counts:")
# print(model_data['portfolio'].value_counts())

# ===================================================================    
#             b. Calculate monthly portfolio weighted returns        
# ===================================================================   
# Function to calculate weighted return for each bond
def calculate_monthly_weighted_return(month_data):
    """Compute the monthly weighted return by portfolio for a given month."""
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

# Convert the result in to a DataFrame.
monthly_port_ret = pd.DataFrame(monthly_portfolio_returns).T
monthly_port_ret.index.name = 'eom'
monthly_port_ret = monthly_port_ret.reset_index()
monthly_port_ret_long = monthly_port_ret.melt(id_vars='eom', var_name='portfolio', value_name='weighted_return')

# ===================================================================    
#            c.  Calculate monthly market weighted returns        
# ===================================================================  
# Function to calculate weighted return for each bond
def calculate_weighted_return_per_month(df):
    """
    For each month, compute each bond's weight, and then the bond-level weighted return.
    """
    monthly_sum_mv = df.groupby('eom')['market_value_past'].transform('sum')
    df['weight'] = df['market_value_past'] / monthly_sum_mv
    df['weighted_return'] = df['weight'] * df['ret_exc']
    return df

model_data = calculate_weighted_return_per_month(model_data) # Apply the function

# Now compute the overall market return per month by summing bond-level weighted returns
market_return_df = (
    model_data
    .groupby('eom', as_index=False)['weighted_return']
    .sum()
    .rename(columns={'weighted_return': 'market_return'})
)

returns_merged = pd.merge(monthly_port_ret_long, market_return_df, on='eom')


# ===================================================================    
#           d.   Compute rolling betas for each portfolio        
# ===================================================================  
def compute_beta(window_df, min_months=12):
    """Compute beta as:
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

# Loop over each portfolio and each month to calculate beta
beta_records = []
returns_merged = returns_merged.sort_values('eom')
for portfolio in returns_merged['portfolio'].unique(): # Loop over each portfolio separately.
    df_port = returns_merged[returns_merged['portfolio'] == portfolio].copy().sort_values('eom').reset_index(drop=True)
    for i, current_date in enumerate(df_port['eom']):
        if i < 12:     # Skip the first 12 months (no full 12-month history available).
            continue
        if i < 60:     # For months 13 up to 60, use all available historical months.
            window_df = df_port.iloc[0:i]
        else:          # For month 61 and later, use only the previous 60 months.
            window_df = df_port.iloc[i-60:i]
        n_months = len(window_df)
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

# plt.figure(figsize=(10, 6))
# for i, portfolio in enumerate(sorted(beta_df['portfolio'].unique())):
#     sub_df = beta_df[beta_df['portfolio'] == portfolio]
#     plt.plot(sub_df['eom'], sub_df['beta'], marker='o', label=portfolio, color=cmap(i+1))
# plt.xlabel("End-of-Month (eom)")
# plt.ylabel("Rolling Beta")
# plt.title("Rolling Beta by Portfolio Over Time\n(Beta computed using past data: increasing from 12 to 60 months)")
# plt.legend()
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(figures_folder + "/rolling_beta_by_portfolio.png")
# plt.close()

# ===================================================================    
#           e.  Calculate capital gain overhang  (CGO)      
# ===================================================================   
monthly_turnover = 0.015 #using quarterly turnover of 4.5% from Peter's paper

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
        weights = 1 - monthly_turnover * (k_values - 1) # Linear weights: most recent gets weight = 1
        weights = weights[::-1]  # Reverse weights so the most recent observation gets the highest weight
        weights = weights / np.sum(weights)   # Normalize so weights sum to 1
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


print('Applying the function group-wise by bond (cusip) ...')
model_data = model_data.sort_values(['cusip', 'eom'])
model_data = model_data.groupby('cusip').apply(compute_effective_purchase_price_linear).reset_index(drop=True)
model_data.to_csv("data/preprocessed/model_data_cgo.csv")
model_data.to_csv(os.path.join(data_folder + "/preprocessed/model_data_cgo.csv"), index=False)

# Aggregate CGO at portfolio level, first monthly then yearly
monthly_cgo = (
    model_data.groupby(['eom', 'portfolio'])['cap_gain_overhang']
    .median() # Use median rather than mean, maybe change
    .reset_index()
)
print("Monthly Median Capital Gain Overhang by Portfolio:")
print(monthly_cgo.head())

# Extract the year from the end-of-month dates, and then group by year and portfolio
# monthly_cgo['year'] = monthly_cgo['eom'].dt.year
# annual_cgo = (
#     monthly_cgo.groupby(['year', 'portfolio'])['cap_gain_overhang']
#     .median() # Use median rather than mean, maybe change
#     .reset_index()
# )

# =============================================================================
#               f. Calculate volatility and skewness 
# =============================================================================
model_data = model_data.sort_values(['portfolio', 'cusip', 'eom'])
model_data['ret_exc'] = model_data['ret_exc'].fillna(0)
model_data['log_ret'] = np.log(1 + model_data['ret_exc'])

def compute_annual_return(bond_df):
    """
    For a given bond (grouped by cusip), compute the annual return using a rolling
    12-month window. The annual return is computed as:
        annual_return = exp(sum(log_ret over 12 months)) - 1).
    If fewer than 12 months are available, return NaN.
    """
    bond_df = bond_df.sort_values('eom').reset_index(drop=True)
    annual_returns = []
    for i in range(len(bond_df)):
        if i + 12 <= len(bond_df):
            window = bond_df.loc[i:i+11, 'log_ret']
            annual_ret = np.exp(window.sum()) - 1
        else:
            annual_ret = np.nan
        annual_returns.append(annual_ret)
    bond_df['annual_return'] = annual_returns
    return bond_df

# We'll loop over each portfolio and compute the cross-sectional volatility and skewness.
vol_skew_list = []
portfolios = model_data['portfolio'].unique()
for port in portfolios:
    port_data = model_data[model_data['portfolio'] == port].copy()
    port_data = port_data.groupby('cusip').apply(compute_annual_return).reset_index(drop=True)
    vol_skew = port_data.groupby('eom')['annual_return'].agg(
        volatility = lambda x: np.nanstd(x),
        skewness   = lambda x: skew(x, nan_policy='omit')
    ).reset_index()
    vol_skew['portfolio'] = port # Add the portfolio label for clarity.
    vol_skew_list.append(vol_skew)

final_vol_skew = pd.concat(vol_skew_list, ignore_index=True) # Concatenate the results from all portfolios.

print("Volatility and Skewness for Each Portfolio:")
print(final_vol_skew.head())

# =============================================================================
#                     g.  Merging to one dataset 
# =============================================================================
final_monthly_df = monthly_port_ret_long.copy()
final_monthly_df = final_monthly_df.merge(beta_df[['eom', 'portfolio', 'beta']],
                                          on=['eom', 'portfolio'], how='left')
final_monthly_df = final_monthly_df.merge(monthly_cgo[['eom', 'portfolio', 'cap_gain_overhang']],
                                          on=['eom', 'portfolio'], how='left')
final_monthly_df = final_monthly_df.merge(market_return_df, on='eom', how='left')
final_monthly_df = final_monthly_df.merge(
    final_vol_skew[['eom', 'portfolio', 'volatility', 'skewness']],
    on=['eom', 'portfolio'],
    how='left')
final_monthly_df.to_csv(os.path.join(data_folder, "preprocessed", "final_monthly_data.csv"), index=False)
print("Final monthly dataset created.")


# =============================================================================
#               h. Checking the accuracy of the CGO calculations 
# =============================================================================
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