# =============================================================================
#
#                     Part X: Model Setup A
#                   (Expected Returns Estimation)
#
#         (Considers only subset including the cleaned data)
#
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import kv, gamma
from scipy.integrate import quad
from scipy.optimize import root, minimize



# Set working directory (if needed)
project_folder = os.getcwd()

# ===================================================================    
#                     a. Set parameters        
# ===================================================================
# Read CSV files using pandas
theta_all = pd.read_csv(os.path.join(project_folder, "data", "preprocessed", "thetas_df.csv"))
average_metrics_updated = pd.read_csv(os.path.join(project_folder, "data", "preprocessed", "average_metrics_updated.csv"))

# Set parameters (adjusted as in your code)
nu   = 17         # was 7.5
sigma_m = 0.05    # was 0.25
Rf   = 1

gamma_hat, b0 = 0.6, 0.6
alpha, delta, lamb = 0.7, 0.65, 1.5

# Extract columns from DataFrames
sigma_i = average_metrics_updated['volatility'].values  # array
beta_i  = average_metrics_updated['beta'].values
g_i     = average_metrics_updated['cap_gain_overhang'].values / 100
S_i     = average_metrics_updated['Si'].values
zeta_i  = average_metrics_updated['zeta'].values

Ri = 0.001  # was 0.01
mu_global = 0.0005  # was 0.005

# ===================================================================    
#                     b. Calculate μ̂ and θ̂        
# ===================================================================
# For each portfolio we’ll store the results in arrays
mu_hat = np.zeros(3)
theta_hat = np.zeros(3)

# -------------------------
# Define model functions
# -------------------------

def p_Ri(Ri_val, mu, Si, zetai, nu):
    """Probability density function p_Ri."""
    N = 1
    # Compute the argument for the Bessel function
    arg = (nu + ((Ri_val - mu)**2) / Si) * ((zetai**2)/Si)
    Kl = kv((nu + N)/2, np.sqrt(arg))
    result = (2**(1 - (nu+N)/2)) / (gamma(nu/2) * ((np.pi*nu)**(N/2)) * (np.abs(Si)**0.5))
    result *= Kl * np.exp(((Ri_val - mu)/Si)*zetai)
    result /= ((np.sqrt(arg))**(-(nu+N)/2)) * ((1 + (Ri_val - mu)**2/(Si*nu))**((nu+N)/2))
    return result

def P_Ri(x, mu, Si, zetai, nu):
    """Cumulative distribution function P_Ri, integrated from -inf to x."""
    integral, err = quad(lambda Ri_val: p_Ri(Ri_val, mu, Si, zetai, nu), -np.inf, x, epsrel=1e-8)
    return integral

def dwP_Ri(x, mu, Si, zetai, delta, nu):
    """Derivative term for the negative side."""
    P = P_Ri(x, mu, Si, zetai, nu)
    ## Clamp P to [0,1] if needed ##
    # epsilon = 1e-10
    # if P < epsilon:
    #     # Debug message:
    #     print(f"Clamping P at x={x}: original P={P}")
    #     P = epsilon
    # if P > 1:
    #     print(f"Clamping P at x={x}: original P={P}")
    #     P = 1
    p_val = p_Ri(x, mu, Si, zetai, nu)
    numerator = (delta * P**(delta-1) * (P**delta + (1-P)**delta)) - P**delta * (P**(delta-1) - (1-P)**(delta-1))
    denominator = (P**delta + (1-P)**delta)**(1+1/delta)
    return numerator/denominator * p_val

def dwP_1_Ri(Ri_val, mu, Si, zetai, delta, nu):
    """Derivative term for the positive side."""
    P = P_Ri(Ri_val, mu, Si, zetai, nu)
    # epsilon = 1e-10
    # if P < epsilon:
    #     P = epsilon
    # if P > 1:
    #     P = 1
    p_val = p_Ri(Ri_val, mu, Si, zetai, nu)
    result = -((delta * (1-P)**(delta-1) * (P**delta + (1-P)**delta)) - (1-P)**delta * ((1-P)**(delta-1) - P**(delta-1)))
    result /= (P**delta + (1-P)**delta)**(1+1/delta)
    return result * p_val

def neg_integral(mu, Si, zetai, gi, theta_mi, theta_i_minus1, Rf, alpha, delta, nu):
    """Integral for the negative part."""
    # integration limits: from -100 to (Rf - theta_i_minus1*gi/theta_mi)
    upper_limit = Rf - theta_i_minus1*gi/theta_mi
    integrand = lambda x: ((theta_mi*(Rf-x) - theta_i_minus1*gi)**(alpha-1)) * (Rf-x) * dwP_Ri(x, mu, Si, zetai, delta, nu)
    integral, err = quad(integrand, -100, upper_limit, epsrel=1e-8)
    return integral

def pos_integral(mu, Si, zetai, gi, theta_mi, theta_i_minus1, Rf, alpha, delta, nu):
    """Integral for the positive part."""
    lower_limit = Rf - theta_i_minus1*gi/theta_mi
    integrand = lambda x: ((theta_mi*(x-Rf) + theta_i_minus1*gi)**(alpha-1)) * (x-Rf) * dwP_1_Ri(x, mu, Si, zetai, delta, nu)
    integral, err = quad(integrand, lower_limit, 100, epsrel=1e-8)
    return integral

# Maybe change to version without safeguards

def neg_integral20(theta_val, mu, Si, zetai, gi, theta_i_minus1, lamb, b0, alpha, Rf, delta, nu):
    """Safeguarded version for the negative side of Equation20.
       theta_val is a scalar."""
    if abs(theta_val) < 1e-8 or abs(Si) < 1e-8:
        return 0.0
    limit = Rf - theta_i_minus1 * gi / theta_val
    if np.isnan(limit) or np.isinf(limit):
        return 0.0
    def integrand(x):
        expr = theta_val * (Rf - x) - theta_i_minus1 * gi
        if expr < 0:
            return 0.0
        else:
            return (-lamb * b0 * (expr**alpha)) * dwP_Ri(x, mu, Si, zetai, delta, nu)
    try:
        integral, _ = quad(integrand, -100, limit, epsrel=1e-8)
        return integral if np.isfinite(integral) else 0.0
    except Exception as e:
        return 0.0

def pos_integral20(theta_val, mu, Si, zetai, gi, theta_i_minus1, b0, Rf, alpha, delta, nu):
    """Safeguarded version for the positive side of Equation20.
       theta_val is a scalar."""
    if abs(theta_val) < 1e-8 or abs(Si) < 1e-8:
        return 0.0
    limit = Rf - theta_i_minus1 * gi / theta_val
    if np.isnan(limit) or np.isinf(limit):
        return 0.0
    def integrand(x):
        expr = theta_val * (x - Rf) + theta_i_minus1 * gi
        if expr < 0:
            return 0.0
        else:
            return (-b0 * (expr**alpha)) * dwP_1_Ri(x, mu, Si, zetai, delta, nu)
    try:
        integral, _ = quad(integrand, limit, 100, epsrel=1e-8)
        return integral if np.isfinite(integral) else 0.0
    except Exception as e:
        return 0.0

def Equation35(mu, zetai, Si, gi, theta_mi, theta_i_minus1, βi, sigma_m, nu, Rf, gamma_hat, alpha, lamb, b0, delta):
    """Equation35 to solve for mu (returned as a 1-element vector)."""
    term1 = (mu[0] + (nu*zetai/(nu-2) - Rf)) - gamma_hat * βi * sigma_m**2
    term2 = -alpha * lamb * b0 * neg_integral(mu[0], Si, zetai, gi, theta_mi, theta_i_minus1, Rf, alpha, delta, nu)
    term3 = -alpha * b0 * pos_integral(mu[0], Si, zetai, gi, theta_mi, theta_i_minus1, Rf, alpha, delta, nu)
    return term1 + term2 + term3

def Equation20(theta_vec, zetai, mu_val, nu, σi, βi, sigma_m, theta_mi, theta_i_minus1, Rf, gamma_hat, lamb, b0, Si, gi, alpha, delta, mu_param):
    """Entire objective function to optimize for theta."""
    theta = theta_vec[0]
    term1 = theta * (mu_val + (nu*zetai/(nu-2) - Rf)) - gamma_hat/2*(theta**2*σi**2 + 2*theta*(βi*sigma_m**2 - theta_mi*σi**2))
    term2 = neg_integral20(theta, mu_param, Si, zetai, gi, theta_i_minus1, lamb, b0, alpha, Rf, delta, nu)
    term3 = pos_integral20(theta, mu_param, Si, zetai, gi, theta_i_minus1, b0, Rf, alpha, delta, nu)
    return -(term1 + term2 + term3)

# -------------------------
# Solve for μ̂ and θ̂ for each portfolio
# -------------------------
# For each portfolio index j (0-indexed in Python)
for j in range(1): #change back to 3 when model is ready
    sigmai_val = sigma_i[j]
    betai_val = beta_i[j]
    gi_val = g_i[j]
    Si_val = S_i[j]
    zetai_val = zeta_i[j]

    # Adjusting theta_mi and theta_i_minus depending on portfolio:
    if j == 0:
        theta_mi = theta_all['theta_mi'].iloc[j] / 30  #or 150
        theta_i_minus1 = theta_all['theta_i_minus1'].iloc[j] / 30
    elif j == 1:
        theta_mi = theta_all['theta_mi'].iloc[j] / 190 #or 950
        theta_i_minus1 = theta_all['theta_i_minus1'].iloc[j] / 190
    elif j == 2:
        theta_mi = theta_all['theta_mi'].iloc[j] / 780 #or 3900
        theta_i_minus1 = theta_all['theta_i_minus1'].iloc[j] / 780

    print("Finding mu for portfolio {portfolios[j]}")
    solving = root(lambda mu: np.array([Equation35(mu, zetai_val, Si_val, gi_val, theta_mi, theta_i_minus1,
                                               betai_val, sigma_m, nu, Rf, gamma_hat, alpha, lamb, b0, delta)]),
               [0.5])
    mu_solution = solving.x[0]
    mu_hat[j] = mu_solution

    # Now optimize Equation20 for theta. We treat theta as a scalar.
    res = minimize(lambda th: Equation20([th], zetai_val, mu_hat[j], nu, sigmai_val, betai_val, sigma_m,
                                          theta_mi, theta_i_minus1, Rf, gamma_hat, lamb, b0, Si_val, gi_val, alpha, delta, mu_solution),
                   x0=[theta_mi],
                   bounds=[(-theta_mi, theta_mi*2)])
    theta_hat[j] = res.x[0]

# -------------------------
# Save results to CSV
# -------------------------
results_df = pd.DataFrame({
    "portfolio": ["DI", "HY", "IG"],
    "mu_hat": mu_hat,
    "theta_hat": theta_hat
})
results_df.to_csv(os.path.join(project_folder, "data", "results", "mu_theta_results.csv"), index=False)

# -------------------------
# Plotting the results
# -------------------------
def Equation20_plot(theta, zetai, mu_val, nu, sigma_i, beta_i, sigma_m, theta_mi, theta_i_minus1,
                      Rf, gamma_hat, lamb, b0, Si, gi, alpha, delta, mu_param):
    """
    Full objective function. 'theta' is a scalar.
    """
    term1 = theta * (mu_val + (nu * zetai)/(nu - 2) - Rf) - gamma_hat/2 * (theta**2 * sigma_i**2 +
             2 * theta * (beta_i * sigma_m**2 - theta_mi * sigma_i**2))
    term2 = neg_integral20(theta, mu_param, Si, zetai, gi, theta_i_minus1, lamb, b0, alpha, Rf, delta, nu)
    term3 = pos_integral20(theta, mu_param, Si, zetai, gi, theta_i_minus1, b0, Rf, alpha, delta, nu)
    return -(term1 + term2 + term3)

def Equation20_MV(theta, mu_val, nu, zetai, Rf, gamma_hat, beta_i, sigma_m, theta_mi, sigma_i):
    """
    MV part of the objective function (only the first term).
    """
    term1 = theta * (mu_val + (nu * zetai)/(nu - 2) - Rf) - gamma_hat/2 * (theta**2 * sigma_i**2 +
             2 * theta * (beta_i * sigma_m**2 - theta_mi * sigma_i**2))
    return -(term1)

def Equation20_PT(theta, Si, zetai, gi, theta_i_minus1, lamb, b0, mu_param, alpha, delta, Rf, nu):
    """
    PT part of the objective function (only the second and third terms).
    """
    term2 = neg_integral20(theta, mu_param, Si, zetai, gi, theta_i_minus1, lamb, b0, alpha, Rf, delta, nu)
    term3 = pos_integral20(theta, mu_param, Si, zetai, gi, theta_i_minus1, b0, Rf, alpha, delta, nu)
    return -(term2 + term3)


""" Assume the following variables are defined and loaded from your CSV files:
    sigma_i, beta_i, g_i, S_i, zeta_i are numpy arrays (or similar)
    theta_all is a pandas DataFrame containing columns 'theta_mi' and 'theta_i_minus1'
    Other parameters: nu, sigma_m, Rf, gamma_hat, b0, alpha, delta, lamb, Ri, mu_global, etc."""

portfolios = ["DI", "HY", "IG"]
mu_hat = np.array([0.5, 0.95788, 0.88524])    # These should come from your optimization step
theta_hat = np.array([1e-7, 1e-7, 1e-7])      # Likewise

# Loop over portfolios
for j in range(1): #Change to 3 when model is ready
    # Extract portfolio-specific parameters:
    sigma_i_val = sigma_i[j]
    beta_i_val  = beta_i[j]
    gi_val      = g_i[j]
    Si_val      = S_i[j]
    zetai_val   = zeta_i[j]
    
    if j == 0:
        theta_mi_val = theta_all['theta_mi'].iloc[j] / 30  #or 150
        theta_i_minus1_val = theta_all['theta_i_minus1'].iloc[j] / 30  #or 150
    elif j == 1:
        theta_mi_val = theta_all['theta_mi'].iloc[j] / 190 #or 950
        theta_i_minus1_val = theta_all['theta_i_minus1'].iloc[j] / 190 #or 950
    elif j == 2:
        theta_mi_val = theta_all['theta_mi'].iloc[j] / 780 #or 3900
        theta_i_minus1_val = theta_all['theta_i_minus1'].iloc[j] / 780 #or 3900

    # Figure 3: Full objective function over a narrow theta range.
    theta_range_narrow = np.linspace(1e-6, 0.002, 100)
    u_values = [Equation20_plot(theta, zetai_val, mu_hat[j], nu, sigma_i_val, beta_i_val, sigma_m,
                                theta_mi_val, theta_i_minus1_val, Rf, gamma_hat, lamb, b0, Si_val, gi_val,
                                alpha, delta, mu_hat[j])
                for theta in theta_range_narrow]

    theta_range_neg = np.linspace(-0.001, -1e-6, 100)
    u_values_neg = [Equation20_plot(theta, zetai_val, mu_hat[j], nu, sigma_i_val, beta_i_val, sigma_m,
                                    theta_mi_val, theta_i_minus1_val, Rf, gamma_hat, lamb, b0, Si_val, gi_val,
                                    alpha, delta, mu_hat[j])
                    for theta in theta_range_neg]
    
    # Concatenate the negative and positive ranges:
    theta_all_range = np.concatenate((theta_range_neg, theta_range_narrow))
    u_all_values = np.concatenate((u_values_neg, u_values))
    
    plt.figure()
    plt.plot(theta_all_range, -u_all_values, lw=3, color='blue')
    plt.xlabel("θ₁")
    plt.ylabel("Utility")
    plt.title(f"Objective function of Equation 20 for portfolio {portfolios[j]}")
    plt.savefig(os.path.join(project_folder, f"Figure3_portfolio{portfolios[j]}.png"))
    plt.show()
    
    # Figure 4: Overlay MV and PT parts over a wider theta range.
    theta_range_wide = np.linspace(1e-6, 0.25, 100)
    u_values_full = [Equation20_plot(theta, zetai_val, mu_hat[j], nu, sigma_i_val, beta_i_val, sigma_m,
                                     theta_mi_val, theta_i_minus1_val, Rf, gamma_hat, lamb, b0, Si_val, gi_val,
                                     alpha, delta, mu_hat[j])
                     for theta in theta_range_wide]
    MV_values = [Equation20_MV(theta, mu_hat[j], nu, zetai_val, Rf, gamma_hat, beta_i_val, sigma_m,
                               theta_mi_val, sigma_i_val)
                 for theta in theta_range_wide]
    PT_values = [Equation20_PT(theta, Si_val, zetai_val, gi_val, theta_i_minus1_val, lamb, b0, mu_hat[j], alpha, delta, Rf, nu)
                 for theta in theta_range_wide]
    
    theta_range_wide_neg = np.linspace(-0.01, -1e-5, 100)
    u_values_full_neg = [Equation20_plot(theta, zetai_val, mu_hat[j], nu, sigma_i_val, beta_i_val, sigma_m,
                                         theta_mi_val, theta_i_minus1_val, Rf, gamma_hat, lamb, b0, Si_val, gi_val,
                                         alpha, delta, mu_hat[j])
                          for theta in theta_range_wide_neg]
    MV_values_neg = [Equation20_MV(theta, mu_hat[j], nu, zetai_val, Rf, gamma_hat, beta_i_val, sigma_m,
                                   theta_mi_val, sigma_i_val)
                     for theta in theta_range_wide_neg]
    PT_values_neg = [Equation20_PT(theta, Si_val, zetai_val, gi_val, theta_i_minus1_val, lamb, b0, mu_hat[j], alpha, delta, Rf, nu)
                     for theta in theta_range_wide_neg]
    
    theta_all_wide = np.concatenate((theta_range_wide_neg, theta_range_wide))
    u_all_wide = np.concatenate((u_values_full_neg, u_values_full))
    MV_all_wide = np.concatenate((MV_values_neg, MV_values))
    PT_all_wide = np.concatenate((PT_values_neg, PT_values))
    
    plt.figure()
    plt.plot(theta_all_wide, -u_all_wide, lw=2, color='red', label="Full")
    plt.plot(theta_all_wide, -MV_all_wide, linestyle='dashed', lw=1, color='blue', label="MV")
    plt.plot(theta_all_wide, -PT_all_wide, linestyle='dashdot', lw=1, color='green', label="PT")
    plt.xlim(-0.01, 0.25)
    plt.ylim(-0.004, 0.004)
    plt.xlabel("Theta")
    plt.ylabel("Utility")
    plt.title(f"Objective function for portfolio {portfolios[j]}")
    plt.legend()
    plt.savefig(os.path.join(project_folder, f"Figure4_portfolio{portfolios[j]}.png"))
    plt.show()