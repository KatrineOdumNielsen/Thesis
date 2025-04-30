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
model_data = pd.read_csv(data_folder + "/preprocessed/model_data.csv")
stock_returns = pd.read_csv(data_folder + "/raw/stock_market_returns.csv")
long_term_gvt_bond_data = pd.read_csv(data_folder + "/raw/long_term_gvt_bond_data.csv")
t1m_daily = fred.get_series('DGS1MO') / 100  # Convert from % to decimal

### Creating portfolios
model_data['eom'] = pd.to_datetime(model_data['eom'])
unique_months = model_data['eom'].unique()

# Add portfolio to the model_data 
model_data['portfolio'] = np.nan
#model_data.loc[model_data['credit_spread_start'] > 0.1, 'portfolio'] = 'DI'
model_data.loc[model_data['distressed_rating_start'] == True , 'portfolio'] = 'DI'
model_data.loc[model_data['portfolio'].isnull() & (model_data['rating_class_start'] == '0.IG'), 'portfolio'] = 'IG'
model_data.loc[model_data['portfolio'].isnull() & (model_data['rating_class_start'] == '1.HY'), 'portfolio'] = 'HY'

# Now split DI into DI_high and DI_low based on price_eom_start
for month in unique_months:
    di_mask = (model_data['eom'] == month) & (model_data['portfolio'] == 'DI')
    di_bonds = model_data.loc[di_mask]
    
    if len(di_bonds) > 0:
        # Sort DI bonds by price_eom_start descending
        di_bonds_sorted = di_bonds.sort_values('price_eom_start', ascending=False)
        
        # Find the number to split
        split_index = len(di_bonds_sorted) // 2
        
        # Assign DI_high and DI_low
        high_indices = di_bonds_sorted.index[:split_index]
        low_indices = di_bonds_sorted.index[split_index:]
        
        model_data.loc[high_indices, 'portfolio'] = 'DI_high'
        model_data.loc[low_indices, 'portfolio'] = 'DI_low'

### Calculating returns for portfolios
groups = model_data['portfolio'].dropna().unique().tolist()

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
texc_regression_df = texc_returns_df.iloc[:]
texc_regression_df = texc_regression_df * 12

# Define the independent variable (market excess return) and add a constant
X = sm.add_constant(texc_regression_df[['market_ret_texc']])

# Store results in a dictionary
texc_regression_results = {}

# Loop through each portfolio and run the regression
for portfolio in groups: 
    y = texc_regression_df[portfolio]
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    texc_regression_results[portfolio] = model
    print(f"Regression results using ret_texc for {portfolio}:\n{model.summary()}\n")

# Collecting into a single dataframe
exc_returns_df = pd.DataFrame(group_ret_exc, index=pd.to_datetime(date_series))
exc_returns_df['market_ret_texc'] = market_ret_texc
exc_returns_df.index.name = 'date'
exc_returns_df = exc_returns_df.join(term_returns[['term']], how='left')
exc_returns_df = exc_returns_df.join(stock_returns[['stock_ret_exc']], how='left')
exc_regression_df = exc_returns_df.iloc[:]
#exc_regression_df = exc_regression_df * 12

# Define the independent variable (market excess return) and add a constant
X = sm.add_constant(exc_regression_df[['market_ret_texc', 'term', 'stock_ret_exc']])

# Store results in a dictionary
exc_regression_results = {}

# Loop through each portfolio and run the regression
for portfolio in groups: 
    y = exc_regression_df[portfolio]
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    exc_regression_results[portfolio] = model
    print(f"Regression results using ret_exc for {portfolio}:\n{model.summary()}\n")

print(exc_regression_df.head())

# # Shape ratios
# print(exc_regression_df.tail())
# print(texc_regression_df.tail())
# print(exc_regression_df.head())
# print(texc_regression_df.head())

# exc_returns_mean = exc_regression_df.mean()
# exc_returns_std = exc_regression_df.std()
# exc_returns_sharpe = exc_returns_mean / exc_returns_std
# print("Mean excess return:\n" + exc_returns_mean.to_string())
# print("Standard deviation of excess return:\n" + exc_returns_std.to_string())
# print("Sharpe ratio:\n" + exc_returns_sharpe.to_string())

# # Cumulative return
# cum_return = (1 + texc_regression_df['market_ret_texc']).prod() - 1

# Number of months
n_months = texc_regression_df.shape[0]

# # Annualized return
# annualized_return = (1 + cum_return) ** (12 / n_months) - 1

# print(f"Cumulative pure credit market return: {cum_return:.4%}")
# print(f"Annualized pure credit market return: {annualized_return:.4%}")
