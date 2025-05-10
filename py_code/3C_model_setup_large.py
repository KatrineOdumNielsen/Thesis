# =============================================================================
#
#                         Part 3C: Model Setup
#    (Calculating Beta, Capital Gain overhang, volatility, and skewness)
#
#           (Considers subset including clean data and WRDS data)
#
# =============================================================================
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import skew
from datetime import datetime

#Set up working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"
figures_folder = project_dir + "/figures"

# Importing the cleaned data
bond_data = pd.read_csv(data_folder + "/preprocessed/bond_data_large.csv")
model_data = pd.read_csv(data_folder + "/preprocessed/model_data.csv")
model_data['eom'] = pd.to_datetime(model_data['eom'])
model_data['offering_date'] = pd.to_datetime(model_data['offering_date'])

# ===================================================================    
#                     a. Set up portfolios by month        
# ===================================================================
print("Setting up portfolios by month...")                                                         
bond_data['eom'] = pd.to_datetime(model_data['eom'])
unique_months = model_data['eom'].unique()
portfolio_by_month = {}

for month in sorted(unique_months):
    month_data = model_data[model_data['eom'] == month].copy()
    # Assign portfolios for this month based on credit spread and rating class
    month_data.loc[month_data['credit_spread_start'] > 0.1, 'portfolio'] = 'DI'
    month_data.loc[(month_data['portfolio'].isnull()) & (month_data['rating_class_start'] == '0.IG'), 'portfolio'] = 'IG'
    month_data.loc[(month_data['portfolio'].isnull()) & (month_data['rating_class_start'] == '1.HY'), 'portfolio'] = 'HY'
    month_data.loc[month_data['portfolio'].isnull(), 'portfolio'] = 'Other'
    portfolio_by_month[month] = month_data # Save the DataFrame for this month

# Add portfolio to the model_data 
model_data['portfolio'] = np.nan
model_data.loc[model_data['credit_spread_start'] > 0.1, 'portfolio'] = 'DI'
model_data.loc[model_data['portfolio'].isnull() & (model_data['rating_class_start'] == '0.IG'), 'portfolio'] = 'IG'
model_data.loc[model_data['portfolio'].isnull() & (model_data['rating_class_start'] == '1.HY'), 'portfolio'] = 'HY'
print("Done setting up portfolios")

# print("Portfolio counts:")
# print(model_data['portfolio'].value_counts())

# ===================================================================    
#             b. Calculate monthly portfolio weighted returns        
# ===================================================================
# Calculating returns for portfolios
groups = model_data['portfolio'].dropna().unique().tolist()

group_returns = {grp: [] for grp in groups}
market_returns = []
date_series = []
dates = sorted(model_data['eom'].dropna().unique())

# Loop through each month
for dt in dates:
    current_period = model_data[model_data['eom'] == dt]

    # Skip if no data
    if current_period.empty:
        continue

    date_series.append(dt)

    # Compute returns (texc) for each group
    for grp in groups:
        group_data = current_period[current_period['portfolio'] == grp]
        total_mv = group_data['market_value_start'].sum()

        if len(group_data) > 0 and total_mv > 0:
            weights = group_data['market_value_start'] / total_mv
            weighted_return = (weights * group_data['ret_texc']).sum()
        else:
            weighted_return = 0

        group_returns[grp].append(weighted_return)

    # Total market return across all portfolios
    total_mv_all = current_period['market_value_start'].sum()
    if total_mv_all > 0:
        weights_all = current_period['market_value_start'] / total_mv_all
        weighted_return_all = (weights_all * current_period['ret_texc']).sum()
    else:
        weighted_return_all = 0
    market_returns.append(weighted_return_all)

# Create a DataFrame for monthly returns using the computed lists.
returns_df = pd.DataFrame(group_returns, index=date_series)  
returns_df['market'] = market_returns
returns_df.index.name = 'date'
returns_df = returns_df.sort_index()
returns_df.index = pd.to_datetime(returns_df.index)
returns_df_reset = returns_df.reset_index()

# Change the DataFrame to long format for the portfolios (excluding 'market') and merge with market returns
long_returns = returns_df_reset.melt(
    id_vars='date',
    value_vars=['IG', 'HY', 'DI'],
    var_name='portfolio',
    value_name='weighted_return'
)

long_returns['market_return'] = long_returns['date'].map(
    returns_df['market']
)

long_returns = long_returns.sort_values(['date', 'portfolio']).reset_index(drop=True)
long_returns['portfolio'] = pd.Categorical(
    long_returns['portfolio'],
    categories=['DI', 'HY', 'IG'],
    ordered=True
)

# Sort first by portfolio, then by date
long_returns = long_returns.sort_values(['portfolio', 'date']).reset_index(drop=True)
long_returns.rename(columns={'date': 'eom'}, inplace=True)
returns_merged = long_returns
monthly_port_ret_long = long_returns.iloc[:, :3]
market_return_df = long_returns[['eom', 'market_return']]

print("Done calculating monthly market weighted returns.")

# ===================================================================    
#           d.   Compute rolling betas for each portfolio        
# ===================================================================  
print("Calculating rolling betas...")
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
# print("Rolling Beta DataFrame (variable window up to 60 months):")
# print(beta_df)

# print("Done calculating rolling betas")

# # Plot the Rolling Betas for Each Portfolio
# cmap = cm.get_cmap('GnBu', 5).reversed()

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
# plt.savefig(figures_folder + "/rolling_beta_by_portfolio_large.png")
# plt.close()

# ===================================================================    
#           e.  Calculate capital gain overhang  (CGO)      
# ===================================================================   
print("Calculating capital gain overhang (CGO)...")
monthly_turnover = 0.015 #using quarterly turnover of 4.5% from Peter's paper

def compute_effective_purchase_price_exponential(group):
    """
    For a given bond (grouped by 'cusip'), compute the effective purchase price and 
    capital gain overhang for each month, using an exponentially decaying approach 
    that assigns the largest weight to the earliest date (k=0).

    Logic:
      - For i data points (i.e., up to index i-1),
        earliest date gets (1 - p)^(i-1),
        date k in [1..i-1] gets p * (1 - p)^(i-1 - k),
        sum of weights = 1.
      - p = monthly_turnover
    """
    group = group.sort_values('eom').copy()
    
    effective_prices = [group.iloc[0]['price_eom']]
    cgo_values = [0.0]

    
    for i in range(1, len(group)):
        # We have i observations so far: 0, 1, ..., i-1
        # The earliest observation is index 0, the newest is index i-1
        n = i  # number of past observations
        p = monthly_turnover
        
        # k=0 => earliest date
        # k=1..n-1 => subsequent dates
        weights = np.zeros(n)
        
        # earliest date
        weights[0] = (1 - p)**(n - 1)
        
        # subsequent dates: for k in [1..n-1], w_k = p * (1 - p)^(n-1 - k)
        for k in range(1, n):
            weights[k] = p * (1 - p)**(n - 1 - k)
        
        past_prices = group.iloc[:i]['price_eom'].values

         # Apply the condition to adjust price_eom for k=0
        if group.iloc[0]['offering_date'] < datetime(2002, 7, 31):
            past_prices[0] = 100
        
        # Weighted sum of the past prices
        effective_price = np.sum(weights * past_prices)
        effective_prices.append(effective_price)
        
        # Calculate capital gain overhang (CGO)
        current_price = group.iloc[i]['price_eom']
        cgo = (current_price / effective_price - 1) * 100
        cgo_values.append(cgo)
    
    group['effective_price'] = effective_prices
    group['cap_gain_overhang'] = cgo_values
    return group

print('Applying the function group-wise by bond (cusip) ...')
model_data = model_data.sort_values(['cusip', 'eom'])
model_data = model_data.groupby('cusip').apply(compute_effective_purchase_price_exponential).reset_index(drop=True)

# Aggregate CGO at portfolio level, first monthly then yearly
monthly_cgo = (
    model_data.groupby(['eom', 'portfolio'])['cap_gain_overhang']
    .median()
    .reset_index()
)

print("Done calculating capital gain overhang (CGO).")

# =============================================================================
#               f. Calculate volatility and skewness 
# =============================================================================
print("Calculating volatility and skewness...")
model_data = model_data.sort_values(['cusip', 'eom'])

#model_data['ret_texc'] = model_data['ret_exc']
model_data['log_texc'] = np.log(1 + model_data['ret_texc'])

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
        if i - 12 <= len(bond_df):
            window = bond_df.loc[i:i+11, 'log_texc']
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
    vol_skew['portfolio'] = port
    vol_skew_list.append(vol_skew)

final_vol_skew = pd.concat(vol_skew_list, ignore_index=True)

print("Done calculating volatility and skewness.")

# =============================================================================
#                     g.  Merging to one dataset 
# =============================================================================
print("Merging all datasets to one final dataset...")
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
print("Final monthly dataset created.")


# =============================================================================
#                     h.  Obtaining average values 
# =============================================================================
print("Obtaining average values for each bond portfolio...")
average_metrics = final_monthly_df.groupby("portfolio")[["beta", "cap_gain_overhang", "volatility", "skewness"]].mean()
average_metrics.to_csv(os.path.join(data_folder, "preprocessed", "average_metrics_large.csv"), index=False)
print("Average metrics per bond portfolio:")
print(average_metrics)

median_metrics = final_monthly_df.groupby("portfolio")[["beta", "cap_gain_overhang", "volatility", "skewness"]].median()
print("Median metrics per bond portfolio:")
print(median_metrics)