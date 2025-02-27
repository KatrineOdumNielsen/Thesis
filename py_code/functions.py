import pandas as pd
import numpy as np

#Get interpolated Treasury yield for a given maturity on a specific date.
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

    # Get available maturities from column names (assuming format like SVENPY01, SVENPY02, ..., SVENPY30)
    maturities = np.array([int(col[-2:]) for col in yield_curve.index])  # Extract numeric part (1,2,3,...,30)
    yields = yield_curve.values  # Corresponding yields

    # Perform linear interpolation
    interpolated_yield = np.interp(target_maturity, maturities, yields)

    return interpolated_yield
