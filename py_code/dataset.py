import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fredapi import Fred
fred = Fred(api_key='01b2825140d39531d70a035eaf06853d')

#Set the working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"

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
bond_return_data.rename(columns={'1M': 't_bill_1m'}, inplace=True) # rename for clarity
bond_return_data['ret'] = bond_return_data['ret_exc'] + (1 + bond_return_data['t_bill_1m']) ** (1/12) - 1 # calculate total return

# Load WRDS data
wrds_data = pd.read_csv(data_folder + "/raw/all_wrds_data.csv")
wrds_data.columns = wrds_data.columns.str.lower()
wrds_data.rename(columns={'offering_amt': 'size', 'date': 'eom', 'yield': 'wrds_yield'}, inplace=True)
wrds_data['eom'] = pd.to_datetime(wrds_data['eom'])
wrds_data['wrds_yield'] = wrds_data['wrds_yield'].str.rstrip('%').astype(float) / 100
wrds_data['ret_eom'] = wrds_data['ret_eom'].str.rstrip('%').astype(float) / 100

# Preparing merge of bond_return_data and descriptive data
bond_return_data_subset = bond_return_data.drop(columns=['permno', 'rating_group', 'duration'])
bond_descriptive_data = wrds_data[wrds_data['eom'] <= "2021-11-30"]
columns_to_keep_descriptive = [
    'eom', 'cusip', 'bond_type', 'size', 'amount_outstanding', 'coupon', 
    'rating_num', 'rating_cat', 'rating_class', 'wrds_yield', 'price_eom', 
    'ret_eom', 'tmt', 'duration'
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
) # adds yield data
bond_additional_data_subset.rename(columns={'1M': 't_bill_1m'}, inplace=True) # rename for clarity
bond_additional_data_subset['yield'] = bond_additional_data_subset['wrds_yield']
bond_additional_data_subset['ret'] = bond_additional_data_subset['ret_eom']
bond_additional_data_subset['ret_exc'] = bond_additional_data_subset['ret'] - (1 + bond_additional_data_subset['t_bill_1m']) ** (1/12) - 1 # excess return
bond_additional_data_subset['ret_texc'] = bond_additional_data_subset['ret_exc'] # REQUIRED TO MERGE - IS NOT ACTUALLY CALCULATED CORRECTLY
bond_additional_data_subset['market_value'] = bond_additional_data_subset['amount_outstanding'] * bond_additional_data_subset['price_eom'] * 10 # MV is NOT in '000s
bond_additional_data_subset = bond_additional_data_subset[bond_data.columns]

# Merge the additional data
print("adding data")
bond_data_large = pd.concat([bond_data, bond_additional_data_subset], ignore_index=True)

# Adding credit spread to the data. First, define a function that can interpolate the yield curve.
yield_curve.columns = [
    float(col[:-1]) / 12 if col.endswith("M") else float(col[:-1]) 
    for col in yield_curve.columns
] # convert column names to years
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
    # Ensure target_date is a Timestamp
    target_date = pd.Timestamp(target_date)

    # Check if the date exists
    if target_date not in all_treasury_yields.index:
        raise ValueError(f"Date {target_date} not found in dataset")

    # Extract the row for the given date
    yield_curve = all_treasury_yields.loc[target_date]

    # Get available maturities from column names
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

# Adding rating, group and price data from last period
rating_data = wrds_data[['eom', 'cusip', 'rating_num', 'rating_class', 'price_eom']]
rating_data = rating_data.sort_values(by=['cusip', 'eom'])
rating_data['rating_num_past'] = rating_data.groupby('cusip')['rating_num'].shift(1)
rating_data['rating_class_past'] = rating_data.groupby('cusip')['rating_class'].shift(1)
rating_data['price_eom_past'] = rating_data.groupby('cusip')['price_eom'].shift(1)
# Merge the rating_data onto bond_data and bond_data_large
bond_data_large = pd.merge(bond_data_large, rating_data[['cusip', 'eom', 'rating_num_past', 'rating_class_past', 'price_eom_past']], on=['cusip', 'eom'], how='left')

#Adding credit spread from last period (large dataset)
# Ensure data is sorted
bond_data_large["eom"] = pd.to_datetime(bond_data_large["eom"])
bond_data_large = bond_data_large.sort_values(["cusip", "eom"])
# Compute last day of the prior month
bond_data_large["prior_eom"] = (bond_data_large["eom"] - pd.DateOffset(months=1)).dt.to_period("M").dt.to_timestamp(how="end")
# Create the new column as a string combining prior date and current cusip
bond_data_large["prior_date_cusip"] = bond_data_large["prior_eom"].dt.strftime("%Y-%m-%d") + "_" + bond_data_large["cusip"]
bond_data_large["current_date_cusip"] = bond_data_large["eom"].dt.strftime("%Y-%m-%d") + "_" + bond_data_large["cusip"]
bond_data_large["prior_date_cusip_shift"] = bond_data_large["current_date_cusip"].shift(1)
# Create the dummy variable equal to 1 if the prior observations is from last month and on the same cusip
bond_data_large["dummy_prior_match"] = (bond_data_large["prior_date_cusip_shift"] == bond_data_large["prior_date_cusip"]).astype(int)
# Shift the 'credit_spread' column by 1 and store it in 'credit_spread_prior'
bond_data_large['credit_spread_past'] = bond_data_large['credit_spread'].shift(1)
# Remove observations where the prior observations was not the same cusip one month before
bond_data_large.loc[bond_data_large['dummy_prior_match'] == 0, 'credit_spread_past'] = np.nan
# Drop help columns
bond_data_large = bond_data_large.drop(columns=['dummy_prior_match', 'prior_date_cusip_shift', 'prior_date_cusip', 'current_date_cusip'])

# Adding market value of last period
bond_data_large = bond_data_large.sort_values(by=['cusip', 'eom'])
bond_data_large['market_value_past'] = bond_data_large.groupby('cusip')['market_value'].shift(1)
bond_data_large['market_value_past'] = bond_data_large['market_value_past'].fillna(bond_data_large['market_value'] / (1 + bond_data_large['ret']))

# Remove observations with no rating / return / yield / amount outstanding
bond_data_large = bond_data_large.dropna(subset=['rating_num'])
bond_data_large = bond_data_large.dropna(subset=['ret_exc'])
bond_data_large = bond_data_large.dropna(subset=['yield'])
bond_data_large = bond_data_large.dropna(subset=['amount_outstanding'])
bond_data_large = bond_data_large.dropna(subset=['duration'])

#The first time a rating is observed, we assume it is the same as the rating of the next month
bond_data_large['rating_num_past'] = bond_data_large['rating_num_past'].fillna(bond_data_large['rating_num'])
bond_data_large['rating_class_past'] = bond_data_large['rating_class_past'].fillna(bond_data_large['rating_class'])

# Adding definitions of distress
bond_data_large['distressed_rating'] = bond_data_large['rating_num'] > 18.5
bond_data_large['distressed_rating_past'] = bond_data_large['rating_num_past'] > 18.5
bond_data_large['distressed_spread'] = bond_data_large['credit_spread'] > 0.1
bond_data_large['distressed_spread_past'] = bond_data_large['credit_spread_past'] > 0.1

# Finalizing the bond_data dataset
bond_data = bond_data_large[bond_data_large['eom'] <= '2021-11-30']

# Save the data
bond_data.to_csv("data/preprocessed/bond_data.csv")
bond_data_large.to_csv("data/preprocessed/bond_data_large.csv")
print("done")

