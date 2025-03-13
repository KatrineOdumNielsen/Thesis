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

#Trying to get capital overhang to work in this one
wrds_data = pd.read_csv(data_folder + "/raw/all_wrds_data.csv")
print(wrds_data.head())
