# =============================================================================
#
#                     Part 1B: Creating the datasets
#
#        (Creates the datasets for the clean dataset (2002-2021), large 
#            dataset using WRDS data (2002-2024), and larger dataset 
#                   using WARGA and WRDS data (1973-2024)*)
#
#           *Note: WARGA data is only available from 1973-1997, so 
#        the larger dataset excudes data on bonds between 1997-2002.
#
# =============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fredapi import Fred
fred = Fred(api_key='insert_key') # Insert FRED API key here
from datetime import datetime

#Set the working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"

# ================  Loading the necessary data  ================
# Load yield curve data (monthly)
series_ids = {
    '1M': 'DGS1MO',   # 1 Month
    '3M': 'DGS3MO',   # 3 Month
    '6M': 'DGS6MO',   # 6 Month
    '1Y': 'DGS1',     # 1 Year
    '2Y': 'DGS2',     # 2 Year
    '3Y': 'DGS3',     # 3 Year
    '5Y': 'DGS5',     # 5 Year
    '7Y': 'DGS7',     # 7 Year
    '10Y': 'DGS10',   # 10 Year
    '20Y': 'DGS20',   # 20 Year
    '30Y': 'DGS30'    # 30 Year
}
yield_data = {name: fred.get_series(series) for name, series in series_ids.items()}
yield_curve = pd.DataFrame(yield_data)
yield_curve = yield_curve[yield_curve.index >= "2002-01-01"]
# Some observations are missing due to holidays/weekends. We forward fill to get a complete dataset.
full_date_range = pd.date_range(start="2002-01-01", end="2024-12-31", freq='D')
yield_curve = yield_curve.reindex(full_date_range)
yield_curve.ffill(inplace=True)
# Converting to end of month
yield_curve = yield_curve[yield_curve.index.is_month_end]
yield_curve = yield_curve / 100 # convert to decimal

# Load return data
bond_return_data = pd.read_csv(data_folder + "/raw/bond_returns.csv")
bond_return_data['eom'] = pd.to_datetime(bond_return_data['eom'])
bond_return_data = bond_return_data.merge(
    yield_curve[['1M']], left_on='eom', right_index=True, how='left'
) # adds yield data
bond_return_data.rename(columns={'1M': 't_bill_1'}, inplace=True) # rename for clarity
bond_return_data['ret'] = bond_return_data['ret_exc'] + (1 + bond_return_data['t_bill_1']) ** (1/12) - 1 # calculate total return

#printing min and max dates
# print("min date: ", bond_return_data['eom'].min())
# print("max date: ", bond_return_data['eom'].max())

# Load WRDS data
wrds_data = pd.read_csv(data_folder + "/raw/all_wrds_data.csv")
wrds_data.columns = wrds_data.columns.str.lower()
wrds_data.rename(columns={'offering_amt': 'size', 'date': 'eom', 'yield': 'wrds_yield'}, inplace=True)
wrds_data['eom'] = pd.to_datetime(wrds_data['eom'])
wrds_data['wrds_yield'] = wrds_data['wrds_yield'].str.rstrip('%').astype(float) / 100
wrds_data['ret_eom'] = wrds_data['ret_eom'].str.rstrip('%').astype(float) / 100
wrds_data['maturity_years'] = (pd.to_datetime(wrds_data['maturity']) - pd.to_datetime(wrds_data['offering_date'])).dt.days / 365

# Preparing merge of bond_return_data and descriptive data
bond_return_data_subset = bond_return_data.drop(columns=['permno', 'rating_group', 'duration'])
bond_descriptive_data = wrds_data[wrds_data['eom'] <= "2021-11-30"]
columns_to_keep_descriptive = [
    'eom', 'cusip', 'security_level', 'size', 'amount_outstanding', 'coupon', 
    'rating_num', 'rating_cat', 'rating_class', 'wrds_yield', 'price_eom', 
    'ret_eom', 'tmt', 'maturity_years', 'duration', 'offering_date'
]
bond_descriptive_data_subset = bond_descriptive_data[columns_to_keep_descriptive]

# Merge the two first datasets
print("Merging data")
bond_data = pd.merge(bond_return_data_subset, bond_descriptive_data_subset, on=["cusip", "eom"])

# Prepare the additional data for merge
bond_additional_data = wrds_data[wrds_data['eom'] > "2021-11-30"]
bond_additional_data_subset = bond_additional_data[columns_to_keep_descriptive]
bond_additional_data_subset = bond_additional_data_subset.merge(
    yield_curve[['1M']], left_on='eom', right_index=True, how='left'
)
bond_additional_data_subset.rename(columns={'1M': 't_bill_1'}, inplace=True) # rename for clarity
bond_additional_data_subset['yield'] = bond_additional_data_subset['wrds_yield']
bond_additional_data_subset['ret'] = bond_additional_data_subset['ret_eom']
bond_additional_data_subset['ret_exc'] = bond_additional_data_subset['ret'] - ((1 + bond_additional_data_subset['t_bill_1']) ** (1/12) - 1) # excess return
bond_additional_data_subset['ret_texc'] = bond_additional_data_subset['ret_exc'] # REQUIRED TO MERGE - IS NOT ACTUALLY CALCULATED CORRECTLY
bond_additional_data_subset['market_value'] = bond_additional_data_subset['amount_outstanding'] * bond_additional_data_subset['price_eom'] * 10 # MV is NOT in '000s
bond_additional_data_subset = bond_additional_data_subset[bond_data.columns]

# Merge the additional data
print("adding data")
bond_data_large = pd.concat([bond_data, bond_additional_data_subset], ignore_index=True)

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

    # Perform linear interpolation
    interpolated_yield = np.interp(target_maturity, maturities, yields)

    return interpolated_yield

# Calculate credit spread for each bond (doing this on both datasets)
print("working on credit spread")
bond_data_large["interp_yield"] = bond_data_large.apply(
    lambda row: get_interpolated_yield(yield_curve, row["eom"], row["tmt"]),
    axis=1
)
bond_data_large["credit_spread"] = bond_data_large['yield'] - bond_data_large['interp_yield']

# ================  Fixing small adjustments and cleaning the dataset ================
# Adding rating, group and price data from last period
rating_data = wrds_data[['eom', 'cusip', 'rating_num', 'rating_class', 'price_eom']]
rating_data = rating_data.sort_values(by=['cusip', 'eom'])
rating_data['rating_num_start'] = rating_data.groupby('cusip')['rating_num'].shift(1)
rating_data['rating_class_start'] = rating_data.groupby('cusip')['rating_class'].shift(1)
rating_data['price_eom_start'] = rating_data.groupby('cusip')['price_eom'].shift(1)

# Merge the rating_data onto bond_data and bond_data_large
bond_data_large = pd.merge(bond_data_large, rating_data[['cusip', 'eom', 'rating_num_start', 'rating_class_start', 'price_eom_start']], on=['cusip', 'eom'], how='left')

#Adding credit spread from last period (large dataset)
bond_data_large["eom"] = pd.to_datetime(bond_data_large["eom"])
bond_data_large = bond_data_large.sort_values(["cusip", "eom"])
bond_data_large["prior_eom"] = (bond_data_large["eom"] - pd.DateOffset(months=1)).dt.to_period("M").dt.to_timestamp(how="end")
bond_data_large["prior_date_cusip"] = bond_data_large["prior_eom"].dt.strftime("%Y-%m-%d") + "_" + bond_data_large["cusip"]
bond_data_large["current_date_cusip"] = bond_data_large["eom"].dt.strftime("%Y-%m-%d") + "_" + bond_data_large["cusip"]
bond_data_large["prior_date_cusip_shift"] = bond_data_large["current_date_cusip"].shift(1)
bond_data_large["dummy_prior_match"] = (bond_data_large["prior_date_cusip_shift"] == bond_data_large["prior_date_cusip"]).astype(int)
bond_data_large['credit_spread_start'] = bond_data_large['credit_spread'].shift(1)
bond_data_large.loc[bond_data_large['dummy_prior_match'] == 0, 'credit_spread_start'] = np.nan
bond_data_large = bond_data_large.drop(columns=['dummy_prior_match', 'prior_date_cusip_shift', 'prior_date_cusip', 'current_date_cusip'])

# Adding market value of last period
bond_data_large = bond_data_large.sort_values(by=['cusip', 'eom'])
bond_data_large['market_value_start'] = bond_data_large.groupby('cusip')['market_value'].shift(1)
bond_data_large['market_value_start'] = bond_data_large['market_value_start'].fillna(bond_data_large['market_value'] / (1 + bond_data_large['ret']))

# Remove observations with no rating / return / yield / amount outstanding / duration
bond_data_large = bond_data_large.dropna(subset=['rating_num'])
bond_data_large = bond_data_large.dropna(subset=['ret_exc'])
bond_data_large = bond_data_large.dropna(subset=['yield'])
bond_data_large = bond_data_large.dropna(subset=['amount_outstanding'])
bond_data_large = bond_data_large.dropna(subset=['duration'])

#The first time a rating is observed, we assume it is the same as the rating of the next month
bond_data_large['rating_num_start'] = bond_data_large['rating_num_start'].fillna(bond_data_large['rating_num'])
bond_data_large['rating_class_start'] = bond_data_large['rating_class_start'].fillna(bond_data_large['rating_class'])

# Adding definitions of distress
bond_data_large['distressed_rating'] = bond_data_large['rating_num'] > 16.5
bond_data_large['distressed_rating_start'] = bond_data_large['rating_num_start'] > 16.5
bond_data_large['distressed_spread'] = bond_data_large['credit_spread'] > 0.1
bond_data_large['distressed_spread_start'] = bond_data_large['credit_spread_start'] > 0.1

# ================ Finalizing the datasets ================
# Finalizing the bond_data dataset
bond_data = bond_data_large[bond_data_large['eom'] <= '2021-11-30']

# Load warga data
bond_warga_data = pd.read_csv(data_folder + "/preprocessed/bond_warga_data.csv")
bond_warga_data['eom'] = pd.to_datetime(bond_warga_data['eom'])
all_cols = sorted(set(bond_warga_data.columns) | set(bond_data.columns))
avramov_dataset = pd.concat([
    bond_warga_data.reindex(columns=all_cols),
    bond_data.reindex(columns=all_cols)],
    ignore_index=True,
    sort=False)
avramov_dataset = avramov_dataset.sort_values(['cusip','eom']).reset_index(drop=True)
new_order = ['eom', 'cusip', 'market_value', 'ret_exc','ret_texc','yield',
                                   't_bill_1','ret','security_level','size','amount_outstanding',
                                   'coupon','rating_num','rating_class','warga_yield','wrds_yield',
                                   'price_eom','ret_eom','tmt','maturity_years','duration','offering_date',
                                   'interp_yield','credit_spread','rating_num_start',
                                   'rating_class_start','price_eom_start','prior_eom',
                                   'credit_spread_start','market_value_start','distressed_rating',
                                   'distressed_rating_start','distressed_spread','distressed_spread_start']
avramov_dataset = avramov_dataset[new_order]
avramov_dataset = avramov_dataset[
    (avramov_dataset['eom'] >= '1986-01-31') &
    (avramov_dataset['eom'] <= '2016-12-31')
] # Capping dataset to match Avramov's dataset

# ================  Calculating CGO in order to split the dataset on CGO  ================
#Calculating capital gain overhang for bond_data
cap_gain_data = wrds_data[['eom', 'cusip', 'price_eom', 'offering_date']]
print("Calculating capital gain overhang (CGO)...")
cap_gain_data['offering_date'] = pd.to_datetime(cap_gain_data['offering_date'])
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
        
        # Create an array of length n for the weights
        # k=0 => earliest date
        # k=1..n-1 => subsequent dates
        weights = np.zeros(n)
        
        # earliest date gets (1 - p)^(n-1)
        weights[0] = (1 - p)**(n - 1)
        
        # for k in [1..n-1], w_k = p * (1 - p)^(n-1 - k)
        for k in range(1, n):
            weights[k] = p * (1 - p)**(n - 1 - k)
        
        # The i-th row in group corresponds to the 'current' date
        # The 'past_prices' are the prices from index 0..(i-1)
        past_prices = group.iloc[:i]['price_eom'].values

         # Apply the condition to adjust price_eom for k=0
        if group.iloc[0]['offering_date'] < datetime(2002, 7, 31):
            past_prices[0] = 100
        
        # Weighted sum of the past prices
        effective_price = np.sum(weights * past_prices)
        effective_prices.append(effective_price)
        
        # Current price is the price at row i
        current_price = group.iloc[i]['price_eom']
        cgo = (current_price / effective_price - 1) * 100
        cgo_values.append(cgo)
    
    group['effective_price'] = effective_prices
    group['cap_gain_overhang'] = cgo_values
    return group

print('Applying the function group-wise by bond (cusip) ...')
cap_gain_data = cap_gain_data.sort_values(['cusip', 'eom'])
cap_gain_data = cap_gain_data.groupby('cusip').apply(compute_effective_purchase_price_exponential).reset_index(drop=True)
cap_gain_data['cap_gain_overhang_start'] = cap_gain_data['cap_gain_overhang'].shift(1)

bond_data = pd.merge(
    bond_data,
    cap_gain_data[['cusip', 'eom', 'cap_gain_overhang_start']],
    on=['cusip', 'eom'],
    how='left'
)
# ================  Split bond data in two samples  ================
bond_data['eom'] = pd.to_datetime(bond_data['eom'])
bond_data = bond_data.sort_values('eom').reset_index(drop=True)
mid_row   = len(bond_data) // 2
first_half  = bond_data.iloc[:mid_row]
second_half = bond_data.iloc[mid_row:]

# ================  Save the data  ================
first_half.to_csv("data/preprocessed/bond_data_first_half.csv")
second_half.to_csv("data/preprocessed/bond_data_second_half.csv")
bond_data.to_csv("data/preprocessed/bond_data.csv")
bond_data_large.to_csv("data/preprocessed/bond_data_large.csv")
avramov_dataset.to_csv("data/preprocessed/avramov_dataset.csv")

print("done")