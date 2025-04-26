import os
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fredapi import Fred
fred = Fred(api_key='01b2825140d39531d70a035eaf06853d')
import statsmodels.api as sm

#Set the working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"

### Getting data
bond_data = pd.read_csv(data_folder + "/preprocessed/bond_data.csv")
model_data = pd.read_csv(data_folder + "/preprocessed/model_data.csv")
stock_returns = pd.read_csv(data_folder + "/raw/stock_market_returns.csv")
long_term_gvt_bond_data = pd.read_csv(data_folder + "/raw/long_term_gvt_bond_data.csv")
t1m_daily = fred.get_series('DGS1MO') / 100  # Convert from % to decimal

### Creating portfolios
model_data['eom'] = pd.to_datetime(model_data['eom'])
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

### Calculating returns for portfolios
groups = ['DI', 'HY', 'IG']

# Initialize storage for returns
group_ret_texc = {grp: [] for grp in groups}
group_ret_exc = {grp: [] for grp in groups}
market_ret_texc = []
date_series = []

# Get list of unique month-end dates
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

        group_ret_texc[grp].append(weighted_return)

        # Compute returns (exc) for each group
    for grp in groups:
        group_data = current_period[current_period['portfolio'] == grp]
        total_mv = group_data['market_value_start'].sum()

        if len(group_data) > 0 and total_mv > 0:
            weights = group_data['market_value_start'] / total_mv
            weighted_return_exc = (weights * group_data['ret_exc']).sum()
        else:
            weighted_return_exc = 0

        group_ret_exc[grp].append(weighted_return_exc)


    # Total market return across all portfolios (optional, but common)
    total_mv_all = current_period['market_value_start'].sum()
    if total_mv_all > 0:
        weights_all = current_period['market_value_start'] / total_mv_all
        weighted_return_all = (weights_all * current_period['ret_texc']).sum()
    else:
        weighted_return_all = 0
    market_ret_texc.append(weighted_return_all)

### Creating term variable
long_term_gvt_bond_data = long_term_gvt_bond_data[['QTDATE', 'TOTRET']]
long_term_gvt_bond_data.columns = ['date', 'gvt_ret']
long_term_gvt_bond_data['gvt_ret'] = long_term_gvt_bond_data['gvt_ret'] / 100

# Convert 'date' to datetime and align to month-end
long_term_gvt_bond_data['date'] = pd.to_datetime(long_term_gvt_bond_data['date'])
long_term_gvt_bond_data['date'] = long_term_gvt_bond_data['date'].dt.to_period('M').dt.to_timestamp('M')

# Filter date range
long_term_gvt_bond_data = long_term_gvt_bond_data.loc[
    (long_term_gvt_bond_data['date'] >= '2002-08-31') &
    (long_term_gvt_bond_data['date'] <= '2024-07-31')
]

# Set 'date' as index for merging
long_term_gvt_bond_data = long_term_gvt_bond_data.set_index('date')

# Pull T-bill rate (1-month) from FRED and resample to monthly
t1m_daily = t1m_daily.ffill()  # Forward fill missing values

# Resample to month-end
t1m_monthly = t1m_daily.resample('M').last()

# Convert to compounded monthly return and shift
monthly_t1m_monthly = (1 + t1m_monthly) ** (1 / 12) - 1
monthly_t1m_monthly_shift = monthly_t1m_monthly.shift()
monthly_t1m_monthly_shift = monthly_t1m_monthly_shift.loc['2002-08-30':'2024-07-31']
monthly_rf = monthly_t1m_monthly_shift.to_frame(name='1m_risk_free')
monthly_rf.index = pd.to_datetime(monthly_rf.index)

# Merge on the index (Date)
term_returns = long_term_gvt_bond_data.join(monthly_rf, how='inner')

# Reset index so date becomes a column again
term_returns = term_returns.reset_index()
term_returns.rename(columns={'index': 'date'}, inplace=True)

# Calculate term variable
term_returns['term'] = term_returns['gvt_ret'] - term_returns['1m_risk_free']

# Prepare to merge
term_returns['date'] = pd.to_datetime(term_returns['date']).dt.normalize()
term_returns.set_index('date', inplace=True)

### Creating stock market variable
stock_returns = stock_returns.iloc[:, :2].rename(columns={
    'caldt': 'date',
    'vwretd': 'stock_return'
})
# Small adjustments
stock_returns['date'] = pd.to_datetime(stock_returns['date'])
stock_returns['date'] = pd.to_datetime(stock_returns['date']) + pd.offsets.MonthEnd(0)
stock_returns.set_index('date', inplace=True)
stock_returns = stock_returns.loc['2002-08-30':'2024-07-31'].reset_index()
# Getting excess returns
stock_returns = stock_returns.set_index('date').join(monthly_rf, how='left').reset_index()
stock_returns['stock_ret_exc']  = stock_returns['stock_return'] - stock_returns['1m_risk_free']
# Convert 'Date' to datetime and normalize it (removes time component)
stock_returns['date'] = pd.to_datetime(stock_returns['date']).dt.normalize()
stock_returns['date'] = pd.to_datetime(stock_returns['date']) + pd.offsets.MonthEnd(0)
stock_returns.set_index('date', inplace=True)

# Collecting into a single dataframe
texc_returns_df = pd.DataFrame(group_ret_texc, index=pd.to_datetime(date_series))
texc_returns_df['market_ret_texc'] = market_ret_texc
texc_returns_df.index.name = 'date'
texc_regression_df = texc_returns_df.iloc[24:]
texc_regression_df = texc_regression_df * 12

# Define the independent variable (market excess return) and add a constant
X = sm.add_constant(texc_regression_df[['market_ret_texc']])

# Store results in a dictionary
texc_regression_results = {}

# Loop through each portfolio and run the regression
for portfolio in ['IG', 'HY', 'DI']:
    y = texc_regression_df[portfolio]
    model = sm.OLS(y, X).fit()
    texc_regression_results[portfolio] = model
    print(f"Regression results using ret_texc for {portfolio}:\n{model.summary()}\n")

# Collecting into a single dataframe
exc_returns_df = pd.DataFrame(group_ret_exc, index=pd.to_datetime(date_series))
exc_returns_df['market_ret_texc'] = market_ret_texc
exc_returns_df.index.name = 'date'
exc_returns_df = exc_returns_df.join(term_returns[['term']], how='left')
exc_returns_df = exc_returns_df.join(stock_returns[['stock_ret_exc']], how='left')
exc_regression_df = exc_returns_df.iloc[24:]
exc_regression_df = exc_regression_df * 12

# Define the independent variable (market excess return) and add a constant
X = sm.add_constant(exc_regression_df[['market_ret_texc', 'term', 'stock_ret_exc']])

# Store results in a dictionary
exc_regression_results = {}

print(exc_regression_df.isnull().sum())

# Loop through each portfolio and run the regression
for portfolio in ['IG', 'HY', 'DI']:
    y = exc_regression_df[portfolio]
    model = sm.OLS(y, X).fit()
    exc_regression_results[portfolio] = model
    print(f"Regression results using ret_exc for {portfolio}:\n{model.summary()}\n")
