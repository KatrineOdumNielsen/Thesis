import os
from fredapi import Fred
import pandas as pd
import statsmodels.api as sm
import numpy as np

#Set the working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"

bond_data = pd.read_csv(data_folder + "/preprocessed/bond_data.csv")

# ===================================================================    
#                     a. Get factor returns        
# ===================================================================
# Initialize FRED API
fred = Fred(api_key='01b2825140d39531d70a035eaf06853d')

# Load TLT returns
tlt_returns = pd.read_excel(data_folder + "/raw/tlt_adj_returns.xlsx")

# Convert 'Date' column to datetime
tlt_returns['Date'] = pd.to_datetime(tlt_returns['Date'])

# Sort by date and filter the desired period
tlt_returns = tlt_returns.sort_values(by='Date', ascending=True)
tlt_returns = tlt_returns[
    (tlt_returns['Date'] >= '2002-08-31') &
    (tlt_returns['Date'] <= '2021-11-30')
]

# Set 'Date' as index for merging
tlt_returns = tlt_returns.set_index('Date')

# Pull T-bill rate (1-month) from FRED and resample to monthly
t1m_daily = fred.get_series('DGS1MO') / 100  # Convert from % to decimal
t1m_daily = t1m_daily.ffill() # filling forward
t1m_monthly = t1m_daily.resample('ME').last() 

# Convert to compounded monthly return and shift
monthly_t1m_monthly = (1 + t1m_monthly) ** (1/12) - 1
monthly_t1m_monthly_shift = monthly_t1m_monthly.shift()
monthly_t1m_monthly_shift = monthly_t1m_monthly_shift.loc['2002-08-30':'2021-11-30']

# Convert to DataFrame
monthly_rf = monthly_t1m_monthly_shift.to_frame(name='1m_risk_free')
monthly_rf.index = pd.to_datetime(monthly_rf.index)

# Merge on the index (Date)
term_returns = tlt_returns[['Return']].join(monthly_rf, how='inner')
term_returns = term_returns.rename(columns={'Return': 'lt_gvt_bond_return'})

# Reset index so 'Date' becomes a column again
term_returns = term_returns.reset_index()

# Display result
term_returns['term'] = term_returns['lt_gvt_bond_return'] - term_returns['1m_risk_free']

# Convert 'Date' to datetime and normalize it (removes time component)
term_returns['Date'] = pd.to_datetime(term_returns['Date']).dt.normalize()

# Set as index
term_returns.set_index('Date', inplace=True)

# Load stock market returns
stock_returns = pd.read_csv(data_folder + "/raw/stock_market_returns.csv")
# Select the first two columns and rename them
stock_returns = stock_returns.iloc[:, :2].rename(columns={
    'caldt': 'date',
    'vwretd': 'stock_return'
})
stock_returns['date'] = pd.to_datetime(stock_returns['date'])
stock_returns['date'] = pd.to_datetime(stock_returns['date']) + pd.offsets.MonthEnd(0)
stock_returns.set_index('date', inplace=True)
stock_returns = stock_returns.loc['2002-08-30':'2021-11-30'].reset_index()
stock_returns = stock_returns.set_index('date').join(monthly_rf, how='left').reset_index()
stock_returns['stock_ret_exc']  = stock_returns['stock_return'] - stock_returns['1m_risk_free']
# Convert 'Date' to datetime and normalize it (removes time component)
stock_returns['date'] = pd.to_datetime(stock_returns['date']).dt.normalize()
stock_returns['date'] = pd.to_datetime(stock_returns['date']) + pd.offsets.MonthEnd(0)

# Set as index
stock_returns.set_index('date', inplace=True)

# ===================================================================    
#                     b. Get portfolio and market returns and merge data        
# ===================================================================

# Get sorted unique month-end dates
dates = sorted(bond_data['eom'].unique())
date_series = []  # Store the dates for which we compute returns

# Define the groups for which we compute returns
groups = ['0.IG', '1.HY']  # Distressed bonds (DI) will be computed separately.

# Initialize dictionaries/lists to store returns
group_returns = {grp: [] for grp in groups}
di_returns = []  # For distressed bonds
market_returns = []
credit_returns = [] # pure credit returns

# Loop over each month (each date)
for dt in dates:
    date_series.append(dt)
    current_period = bond_data[bond_data['eom'] == dt]
    
    # For each group (IG and HY), compute the market-weighted return.
    for grp in groups:
        group_data = current_period[current_period['rating_class_past'] == grp]
        total_mv = group_data['market_value_past'].sum()
        
        if len(group_data) > 0 and total_mv > 0:
            weights = group_data['market_value_past'] / total_mv
            weighted_return = (weights * group_data['ret_exc']).sum()
        else:
            weighted_return = 0
        
        group_returns[grp].append(weighted_return)
    
    # For distressed bonds
    distressed_data = current_period[current_period['distressed_spread_past'] == True]
    total_mv_di = distressed_data['market_value_past'].sum()
    if len(distressed_data) > 0 and total_mv_di > 0:
        weights_di = distressed_data['market_value_past'] / total_mv_di
        weighted_return_di = (weights_di * distressed_data['ret_exc']).sum()
    else:
        weighted_return_di = 0
    di_returns.append(weighted_return_di)
    
    # New section: compute total market-weighted excess return (across all bonds)
    total_mv_all = current_period['market_value_past'].sum()
    if len(current_period) > 0 and total_mv_all > 0:
        weights_all = current_period['market_value_past'] / total_mv_all
        weighted_return_all = (weights_all * current_period['ret_exc']).sum()
    else:
        weighted_return_all = 0
    market_returns.append(weighted_return_all)

    # New section: compute total market-weighted credit return (across all bonds)
    total_mv_all_credit = current_period['market_value_past'].sum()
    if len(current_period) > 0 and total_mv_all_credit > 0:
        weights_all_credit = current_period['market_value_past'] / total_mv_all_credit
        weighted_return_all_credit = (weights_all_credit * current_period['ret_texc']).sum()
    else:
        weighted_return_all_credit = 0
    credit_returns.append(weighted_return_all_credit)

# Create a DataFrame for monthly returns using the computed lists.
returns_df = pd.DataFrame(group_returns, index=date_series)  
returns_df['2.DI'] = di_returns 
returns_df['bond_market'] = market_returns
returns_df['credit_market'] = credit_returns
returns_df.index.name = 'date'
returns_df.index = pd.to_datetime(returns_df.index).normalize()
returns_df = returns_df.join(term_returns[['term']], how='left')
returns_df = returns_df.join(stock_returns[['stock_ret_exc']], how='left')

# ====================================================================
#                     c. Fama MacBeth Regression
# ====================================================================

# Define the portfolios and factors
portfolios = ['0.IG', '1.HY', '2.DI']
factors = ['bond_market']

# Regression
min_window = 12
max_window = 36

rolling_betas = {portfolio: [] for portfolio in portfolios}
beta_dates = []

for i in range(min_window, len(returns_df)):
    # Define the flexible window size
    window_size = min(i, max_window)
    window_data = returns_df.iloc[i - window_size:i]
    beta_dates.append(returns_df.index[i])

    for portfolio in portfolios:
        y = window_data[portfolio]
        X = sm.add_constant(window_data[factors])
        model = sm.OLS(y, X).fit()
        rolling_betas[portfolio].append(model.params)

# Convert to a tidy betas DataFrame with MultiIndex (portfolio, date)
betas_df_list = []

for portfolio in portfolios:
    df = pd.DataFrame(rolling_betas[portfolio], index=beta_dates)
    df['portfolio'] = portfolio
    betas_df_list.append(df)

betas_df = pd.concat(betas_df_list)
betas_df.set_index('portfolio', append=True, inplace=True)
betas_df = betas_df.reorder_levels(['portfolio', betas_df.index.names[0]])  # ['portfolio', 'date']

gamma_list = []
alpha_dict = {portfolio: [] for portfolio in portfolios}
gamma_index = beta_dates

for date in gamma_index:
    y = returns_df.loc[date, portfolios].values
    X = betas_df.loc[(slice(None), date), factors].values  # (portfolio, date) lookup
    model = sm.OLS(y, X).fit()
    gamma_list.append(model.params)

    residuals = y - model.fittedvalues
    for j, portfolio in enumerate(portfolios):
        alpha_dict[portfolio].append(residuals[j])

gamma_df = pd.DataFrame(gamma_list, index=gamma_index, columns=factors)

# Average lambda
avg_lambda = gamma_df.mean()

# Average alpha per asset
avg_alpha = {portfolio: np.mean(alpha_dict[portfolio]) for portfolio in portfolios}
avg_alpha_series = pd.Series(avg_alpha)

# output
results = pd.DataFrame({
    'Average Alpha': avg_alpha_series,
})

print(results * 12)  # Annualize the alpha