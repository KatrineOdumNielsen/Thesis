# Importing necessary libraries
import pandas as pd
import numpy as np
import os
from scipy.stats import skew
from scipy.optimize import fsolve
from math import sqrt
import timeit
import matplotlib.pyplot as plt

#Set up working directory
project_dir = os.getcwd()   # Change to your project directory
data_folder = project_dir + "/data"
figures_folder = project_dir + "/figures"

# Import data
market_returns = pd.read_csv(data_folder + "/preprocessed/market_returns.csv")
portfolio_betas = pd.read_csv(data_folder + "/preprocessed/rolling_betas.csv")

# Pivot portfolio_betas to get separate beta columns for DI, HY, and IG
portfolio_betas_wide = portfolio_betas.pivot(index="eom", columns="portfolio", values="beta").reset_index()

# Rename beta columns
portfolio_betas_wide.columns = ["eom", "beta_DI", "beta_HY", "beta_IG"]

# Merge with market_returns on 'eom'
exp_ret_texc_capm = pd.merge(portfolio_betas_wide, market_returns, on="eom", how="left")

# Compute expected returns for each portfolio
exp_ret_texc_capm["exp_ret_texc_DI"] = exp_ret_texc_capm["market_return"] * exp_ret_texc_capm["beta_DI"]
exp_ret_texc_capm["exp_ret_texc_HY"] = exp_ret_texc_capm["market_return"] * exp_ret_texc_capm["beta_HY"]
exp_ret_texc_capm["exp_ret_texc_IG"] = exp_ret_texc_capm["market_return"] * exp_ret_texc_capm["beta_IG"]

# Compute cumulative returns for each portfolio
exp_ret_texc_capm["cum_exp_ret_texc_DI"] = (1 + exp_ret_texc_capm["exp_ret_texc_DI"]).cumprod() - 1
exp_ret_texc_capm["cum_exp_ret_texc_HY"] = (1 + exp_ret_texc_capm["exp_ret_texc_HY"]).cumprod() - 1
exp_ret_texc_capm["cum_exp_ret_texc_IG"] = (1 + exp_ret_texc_capm["exp_ret_texc_IG"]).cumprod() - 1

# Total number of months in the dataset
T = exp_ret_texc_capm.shape[0]

# Compute final cumulative return for each portfolio
final_cum_ret_exp_texc_DI = exp_ret_texc_capm["cum_exp_ret_texc_DI"].iloc[-1]
final_cum_ret_exp_texc_HY = exp_ret_texc_capm["cum_exp_ret_texc_HY"].iloc[-1]
final_cum_ret_exp_texc_IG = exp_ret_texc_capm["cum_exp_ret_texc_IG"].iloc[-1]

# Compute the annualized return over the full period
ann_ret_exp_texc_DI = (1 + final_cum_ret_exp_texc_DI) ** (12 / T) - 1
ann_ret_exp_texc_HY = (1 + final_cum_ret_exp_texc_HY) ** (12 / T) - 1
ann_ret_exp_texc_IG = (1 + final_cum_ret_exp_texc_IG) ** (12 / T) - 1

# Store in a DataFrame
annualized_returns = pd.DataFrame({
    "Portfolio": ["DI", "HY", "IG"],
    "Annualized Return": [ann_ret_exp_texc_DI, ann_ret_exp_texc_HY, ann_ret_exp_texc_IG]
})

# Display results
print(annualized_returns)
