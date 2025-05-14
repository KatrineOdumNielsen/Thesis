# =============================================================================
#
#                     Part 1A: Creating the datasets
#
#         (Creates the dataset for the WARGA bonds only (1973-1997))
#
# =============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fredapi import Fred
fred = Fred(api_key='insert_key') # Insert FRED API key here

# Setting the working directory
project_dir = os.getcwd()
data_folder = project_dir + "/data"

# ================  Loading the necessary data  ================
# Loading yield curve data (monthly)
series_ids = {
    #'1M': 'DGS1MO',   # 1 Month
    #'3M': 'DGS3MO',   # 3 Month
    #'6M': 'DGS6MO',   # 6 Month
    '1Y': 'DGS1',      # 1 Year
    '2Y': 'DGS2',      # 2 Year
    '3Y': 'DGS3',      # 3 Year
    '5Y': 'DGS5',      # 5 Year
    '7Y': 'DGS7',      # 7 Year
    '10Y': 'DGS10',    # 10 Year
    '20Y': 'DGS20',    # 20 Year
    '30Y': 'DGS30'     # 30 Year
}
yield_data = {name: fred.get_series(series) for name, series in series_ids.items()}
yield_curve = pd.DataFrame(yield_data)
yield_curve = yield_curve[yield_curve.index >= "1973-01-01"] #previous start date: "2002-01-01"
# Some observations are missing due to holidays/weekends. We forward fill to get a complete dataset.
full_date_range = pd.date_range(start="1973-01-01", end="2024-12-31", freq='D')
yield_curve = yield_curve.reindex(full_date_range)
yield_curve.ffill(inplace=True)
# Converting to end of month
yield_curve = yield_curve[yield_curve.index.is_month_end]
yield_curve = yield_curve / 100 # convert to decimal

# Loading WARGA data
bond_warga_data = pd.read_csv(data_folder + "/raw/warga_full.csv")
bond_warga_data['yymmdd'] = pd.to_datetime(bond_warga_data['yymmdd'])
max_date_str = pd.Timestamp.max.normalize().strftime('%Y-%m-%d') # This will be '2262-04-11', which the upper bound of datetime64[ns]
mask = bond_warga_data['matdate'] > max_date_str
bond_warga_data.loc[mask, 'matdate'] = max_date_str # Limit the max date to '2262-04-11', to avoid overflow

# Calculating missing parameters and constructing the dataset
bond_warga_data['matdate'] = pd.to_datetime(bond_warga_data['matdate'])
bond_warga_data['iss_date'] = pd.to_datetime(bond_warga_data['issdate'])
bond_warga_data['tmt'] = (bond_warga_data['matdate'] - bond_warga_data['yymmdd']).dt.days / 365
md = bond_warga_data['matdate'].values.astype('datetime64[D]')
id = bond_warga_data['iss_date'].values.astype('datetime64[D]')
bond_warga_data['maturity_years'] = (md - id).astype('timedelta64[D]') / 365
bond_warga_data['rating_num'] = (bond_warga_data['moodys']+bond_warga_data['snp'])/2
bond_warga_data['rating_class'] = bond_warga_data['rating_num'].apply(lambda x: '0.IG' if x <= 10.5 else '1.HY')
bond_warga_data['market_value'] = bond_warga_data['amtout'] * bond_warga_data['flat1'] * 10 # MV is not in '000s
bond_warga_data['size'] = bond_warga_data['amtout']
bond_warga_data.rename(columns={'cusip9': 'cusip', 'yymmdd': 'eom', 'yld1': 'warga_yield', 'amtout' : 'amount_outstanding',
                                'ret1':'ret_eom','issdate':'offering_date', 'flat1':'price_eom'}, inplace=True)
columns_to_keep = [
    'eom', 'cusip', 'size', 'amount_outstanding', 'coupon', 
    'rating_num','rating_class', 'warga_yield', 'price_eom', 
    'ret_eom', 'tmt', 'maturity_years', 'offering_date','market_value'
]
bond_warga_data = bond_warga_data[columns_to_keep]
bond_warga_data['ret_eom'] = bond_warga_data['ret_eom'] / 100
bond_warga_data['eom'] = pd.to_datetime(bond_warga_data['eom'])
bond_warga_data['warga_yield'] = bond_warga_data['warga_yield'] / 100
bond_warga_data = (bond_warga_data.merge(yield_curve[['1Y']],
           left_on='eom',
           right_index=True,
           how='left')
    .rename(columns={'1Y': 't_bill_1'}))
bond_warga_data['ret'] = bond_warga_data['ret_eom']
bond_warga_data['ret_exc'] = (bond_warga_data['ret']
    - ((1 + bond_warga_data['t_bill_1']) ** (1/12) - 1))
bond_warga_data['ret_texc'] = bond_warga_data['ret_exc']

# Printing min and max dates for overview
# print("min date: ", bond_warga_data['eom'].min())
# print("max date: ", bond_warga_data['eom'].max())

# ================  Adding credit spread to the dataset  ================
# Adding credit spread to the data. First, define a function that can interpolate the yield curve.
yield_curve.columns = [
    float(col[:-1]) / 12 if col.endswith("M") else float(col[:-1]) 
    for col in yield_curve.columns
]

def get_interpolated_yield(all_treasury_yields, target_date, target_maturity):
    """
    Get interpolated Treasury yield for a given maturity on a specific date.

    Parameters:
    - all_treasury_yields: DataFrame with Treasury yields (columns = maturities, index = dates).
    - target_date: Date (YYYY-MM-DD) for which to get the yield.
    - target_maturity: Desired maturity (e.g., 1.5 years).

    Returns:
    - Interpolated yield for the given date and maturity.
    """
    target_date = pd.Timestamp(target_date)
    if target_date not in all_treasury_yields.index:
        raise ValueError(f"Date {target_date} not found in dataset")
    yield_curve = all_treasury_yields.loc[target_date]

    maturities = np.array(yield_curve.index.astype(float)) 
    yields = yield_curve.values  # Corresponding yields

    if target_maturity < 1:
        return yield_curve[1.0]

    # Perform linear interpolation
    interpolated_yield = np.interp(target_maturity, maturities, yields)
    return interpolated_yield

# Calculating credit spread for each bond
print("working on credit spread")
bond_warga_data["interp_yield"] = bond_warga_data.apply(
    lambda row: get_interpolated_yield(yield_curve, row["eom"], row["tmt"]),
    axis=1
)
bond_warga_data["credit_spread"] = bond_warga_data['warga_yield'] - bond_warga_data['interp_yield']

# ================  Fixing small adjustments and cleaning the dataset ================
# Adding rating, group and price data from last period
rating_data = bond_warga_data[['eom', 'cusip', 'rating_num', 'rating_class', 'price_eom']]
rating_data = rating_data.sort_values(by=['cusip', 'eom'])
rating_data['rating_num_start'] = rating_data.groupby('cusip')['rating_num'].shift(1)
rating_data['rating_class_start'] = rating_data.groupby('cusip')['rating_class'].shift(1)
rating_data['price_eom_start'] = rating_data.groupby('cusip')['price_eom'].shift(1)

# Merge the rating_data onto bond_warga_data
bond_warga_data = pd.merge(bond_warga_data, rating_data[['cusip', 'eom', 'rating_num_start', 'rating_class_start', 'price_eom_start']], on=['cusip', 'eom'], how='left')

#Adding credit spread from last period
bond_warga_data["eom"] = pd.to_datetime(bond_warga_data["eom"])
bond_warga_data = bond_warga_data.sort_values(["cusip", "eom"])
bond_warga_data["prior_eom"] = (bond_warga_data["eom"] - pd.DateOffset(months=1)).dt.to_period("M").dt.to_timestamp(how="end")
bond_warga_data["prior_date_cusip"] = bond_warga_data["prior_eom"].dt.strftime("%Y-%m-%d") + "_" + bond_warga_data["cusip"]
bond_warga_data["current_date_cusip"] = bond_warga_data["eom"].dt.strftime("%Y-%m-%d") + "_" + bond_warga_data["cusip"]
bond_warga_data["prior_date_cusip_shift"] = bond_warga_data["current_date_cusip"].shift(1)
bond_warga_data["dummy_prior_match"] = (bond_warga_data["prior_date_cusip_shift"] == bond_warga_data["prior_date_cusip"]).astype(int)
bond_warga_data['credit_spread_start'] = bond_warga_data['credit_spread'].shift(1)
bond_warga_data.loc[bond_warga_data['dummy_prior_match'] == 0, 'credit_spread_start'] = np.nan
bond_warga_data = bond_warga_data.drop(columns=['dummy_prior_match', 'prior_date_cusip_shift', 'prior_date_cusip', 'current_date_cusip'])

# Adding market value of last period
bond_warga_data = bond_warga_data.sort_values(by=['cusip', 'eom'])
bond_warga_data['market_value_start'] = bond_warga_data.groupby('cusip')['market_value'].shift(1)
bond_warga_data['market_value_start'] = bond_warga_data['market_value_start'].fillna(bond_warga_data['market_value'] / (1 + bond_warga_data['ret']))

# Remove observations with no rating / return / yield / amount outstanding / duration
bond_warga_data = bond_warga_data.dropna(subset=['rating_num'])
bond_warga_data = bond_warga_data.dropna(subset=['ret_exc'])
bond_warga_data = bond_warga_data.dropna(subset=['warga_yield'])
bond_warga_data = bond_warga_data.dropna(subset=['amount_outstanding'])

#The first time a rating is observed, we assume it is the same as the rating of the next month
bond_warga_data['rating_num_start'] = bond_warga_data['rating_num_start'].fillna(bond_warga_data['rating_num'])
bond_warga_data['rating_class_start'] = bond_warga_data['rating_class_start'].fillna(bond_warga_data['rating_class'])

# Adding definitions of distress
bond_warga_data['distressed_rating'] = bond_warga_data['rating_num'] > 16.5
bond_warga_data['distressed_rating_start'] = bond_warga_data['rating_num_start'] > 16.5
bond_warga_data['distressed_spread'] = bond_warga_data['credit_spread'] > 0.1
bond_warga_data['distressed_spread_start'] = bond_warga_data['credit_spread_start'] > 0.1

# Make it same order as other datasets
new_order = ['eom', 'cusip', 'market_value', 'ret_exc','ret_texc',
                                   't_bill_1','ret','size','amount_outstanding',
                                   'coupon','rating_num','rating_class','warga_yield',
                                   'price_eom','ret_eom','tmt','maturity_years','offering_date',
                                   'interp_yield','credit_spread','rating_num_start',
                                   'rating_class_start','price_eom_start','prior_eom',
                                   'credit_spread_start','market_value_start','distressed_rating',
                                   'distressed_rating_start','distressed_spread','distressed_spread_start']
bond_warga_data = bond_warga_data[new_order]

# ================  Save the data  ================
bond_warga_data.to_csv("data/preprocessed/bond_warga_data.csv")

print("done")