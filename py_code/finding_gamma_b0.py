import numpy as np
from scipy.optimize import fsolve

# ========================================================
# Model and Calibration Parameters (fill these in)
# ========================================================
Rf = 1.0                  # Risk-free rate (gross return), e.g., 1 means 100% of initial wealth is returned.
target_ep = 0.015         # Target equity premium (gross excess return): 1.5%
target_ud = 0.6           # Target under-diversification measure (choose based on empirical estimates)

# Other model parameters (adjust as needed)
nu = 8                    # Example: degrees of freedom for the t distribution
sigma_m = 0.08            # Example: market volatility
Ri_initial = 0.01         # Initial value for Ri (if used in your equilibrium calculation)
mu_initial = 0.005        # Initial guess for μ (if needed)
# You may have additional parameters from your model calibration, such as:
theta_i_minus1 = 0.5      # Placeholder: investor's allocation to asset i at time -1
theta_mi = 0.5            # Placeholder: market weight of asset i

# ========================================================
# Function to Compute the Equilibrium Gross Return μ̂
# ========================================================
def compute_mu_hat(gamma, b0):
    """
    Compute the equilibrium gross return μ̂ based on the model's equilibrium equation.
    Replace the dummy equation below with your actual model's calculation,
    for example by solving an equation such as Equation 35 in your paper.
    
    For demonstration, we use a placeholder linear function:
    μ̂ = Rf + 0.05 * (gamma + b0 - 1.2)
    """
    mu_hat = Rf + 0.015+0.5 * (gamma + b0 - 1.2)
    return mu_hat

# ========================================================
# Function to Compute the Under-Diversification Measure
# ========================================================
def compute_under_diversification(gamma, b0):
    """
    Compute the model-implied under-diversification measure.
    This function should reflect how concentrated investors' portfolios are.
    Replace the dummy function below with your actual calculation.
    
    For demonstration, assume under-diversification increases with b0
    and decreases with gamma. For example:
    under_div = b0 - 0.5 * gamma
    """
    under_div = b0 - 0.5 * gamma
    return under_div

# ========================================================
# Grid Search Over γ and b₀
# ========================================================
# Define plausible ranges for gamma and b0 (adjust based on your model's context)
gamma_vals = np.linspace(0, 1, 1000)  # For example, values from 0.4 to 0.8
b0_vals    = np.linspace(0, 1, 1000)   # For example, values from 0.4 to 0.8

# List to store results for each (gamma, b0) pair
results = []

# Loop over candidate pairs
for gamma in gamma_vals:
    for b0 in b0_vals:
        # Compute the equilibrium gross return μ̂
        mu_hat = compute_mu_hat(gamma, b0)
        # Compute the implied equity premium (gross excess return)
        equity_premium = mu_hat - Rf
        
        # Compute the under-diversification measure
        under_div = compute_under_diversification(gamma, b0)
        
        # Compute errors relative to target values.
        # You may weight these errors differently if one target is more important.
        error_ep = abs(equity_premium - target_ep)
        error_ud = abs(under_div - target_ud)
        total_error = error_ep + error_ud
        
        # Save the candidate pair and corresponding outputs
        results.append((gamma, b0, mu_hat, equity_premium, under_div, total_error))

# Convert the results list to a structured numpy array for easier handling
dtype = [('gamma', float), ('b0', float), ('mu_hat', float), ('ep', float), ('ud', float), ('error', float)]
results_arr = np.array(results, dtype=dtype)

# Find the pair with the smallest total error
best_index = np.argmin(results_arr['error'])
best_pair = results_arr[best_index]

# ========================================================
# Output the Best Pair
# ========================================================
print("Best (gamma, b0) pair based on calibration:")
print("gamma = {:.3f}, b0 = {:.3f}".format(best_pair['gamma'], best_pair['b0']))
print("Equilibrium gross return μ̂ = {:.3f}".format(best_pair['mu_hat']))
print("Implied equity premium (μ̂ - Rf) = {:.3f}".format(best_pair['ep']))
print("Under-diversification measure = {:.3f}".format(best_pair['ud']))
print("Total error = {:.3f}".format(best_pair['error']))

# ========================================================
# End of Calibration Script
# ========================================================