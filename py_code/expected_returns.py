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
nu   = 7.5         # was 7.5
sigma_m = 0.02    # was 0.25
Rf   = 1

gamma_hat, b0 = 0.6, 0.6
alpha, delta, lamb = 0.7, 0.65, 1.5

# Extract columns from DataFrames
sigma_i = average_metrics_updated['volatility'].values  # array
beta_i  = average_metrics_updated['beta'].values
g_i     = average_metrics_updated['cap_gain_overhang'].values / 100
S_i     = average_metrics_updated['Si'].values
zeta_i  = average_metrics_updated['zeta'].values
theta_mi = theta_all['theta_mi']
theta_i_minus1 = theta_all['theta_i_minus1']

Ri = 0.001  # was 0.01
mu = 0.0005  # was 0.005

# ===================================================================    
#                     b. Calculate μ̂ and θ̂        
# ===================================================================
# For each portfolio we’ll store the results in arrays
mu_hat = np.zeros(3)
theta_hat = np.zeros(3)

# -------------------------
# Define model functions
# -------------------------
def p_Ri(Ri, mu, Si, zetai, nu):
    """Probability density function p_Ri."""
    N = 1
    arg = (nu + ((Ri - mu)**2) / Si) * ((zetai**2)/Si)     # Compute the argument for the Bessel function
    Kl = kv((nu + N)/2, np.sqrt(arg))    # Compute Bessel function
    result = (2**(1 - (nu+N)/2)) / (gamma(nu/2) * ((np.pi*nu*Si)**(N/2)))  # Calculate first part of equation
    result *= Kl * np.exp(((Ri - mu)/Si)*zetai) #Multiply by second part of equation
    result /= ((np.sqrt(arg))**(-(nu+N)/2)) * ((1 + (Ri - mu)**2/(Si*nu))**((nu+N)/2)) # Divide by the last part of the equation
    return result

def P_Ri(mu, Si, zetai, nu, x):
    """Cumulative distribution function P_Ri, integrated from -inf to x."""
    integral, err = quad(lambda Ri: p_Ri(Ri, mu, Si, zetai, nu), -np.inf, x, epsrel=1e-8)
    return integral

def dwP_Ri(mu, Si, zetai, delta, nu, Ri, x):
    """Derivative term for the negative side."""
    P = P_Ri(mu, Si, zetai, nu, x)
    p_val = p_Ri(Ri, mu, Si, zetai, nu)
    numerator = (delta * P**(delta-1) * (P**delta + (1-P)**delta)) - P**delta * (P**(delta-1) - (1-P)**(delta-1))
    denominator = (P**delta + (1-P)**delta)**(1+1/delta)
    return numerator/denominator * p_val

def dwP_1_Ri(Ri, mu, Si, zetai, delta, nu,x):
    """Derivative term for the positive side."""
    P = P_Ri(mu, Si, zetai, nu, x)
    p_val = p_Ri(Ri, mu, Si, zetai, nu)
    result = -((delta * (1-P)**(delta-1) * (P**delta + (1-P)**delta)) - (1-P)**delta * ((1-P)**(delta-1) - P**(delta-1)))
    result /= (P**delta + (1-P)**delta)**(1+1/delta)
    return result * p_val

def neg_integral(mu, Si, zetai, gi, theta_mi, theta_i_minus1, Rf, alpha, delta, nu, x):
    """Integral for the negative part."""
    # integration limits: from -100 to (Rf - theta_i_minus1*gi/theta_mi)
    upper_limit = Rf - theta_i_minus1*gi/theta_mi
    integrand = lambda t: ((theta_mi*(Rf-t) - theta_i_minus1*gi)**(alpha-1)) * (Rf-t) * dwP_Ri(mu, Si, zetai, delta, nu, Ri, x)
    integral, err = quad(integrand, -100, upper_limit, epsrel=1e-8)
    return integral

def pos_integral(mu, Si, zetai, gi, theta_mi, theta_i_minus1, Rf, alpha, delta, nu, x):
    """Integral for the positive part."""
    ##lower_limit = Rf - theta_i_minus1*gi/theta_mi
    lower_limit = 0
    integrand = lambda t: ((theta_mi*(t-Rf) + theta_i_minus1*gi)**(alpha-1)) * (t-Rf) * dwP_1_Ri(Ri, mu, Si, zetai, delta, nu, x)
    integral, err = quad(integrand, lower_limit, 100, epsrel=1e-8)
    return integral

def neg_integral20(theta, mu, Si, zetai, gi, theta_i_minus1, lamb, b0, alpha, Rf, delta, nu, x):
    if theta >= 0:
        integral, err = quad(
            lambda t: (-lamb * b0 * (theta * (Rf - t) - theta_i_minus1 * gi)**alpha)
                      * dwP_Ri(mu, Si, zetai, delta, nu, Ri, x),
            -100, Rf - theta_i_minus1 * gi / theta,
            epsrel=1e-8
        )
    else:  # theta < 0
        integral, err = quad(
            lambda t: (b0 * (theta * (t - Rf) + theta_i_minus1 * gi)**alpha)
                      * dwP_Ri(mu, Si, zetai, delta, nu, Ri, x),
            -100, Rf - theta_i_minus1 * gi / theta,
            epsrel=1e-8
        )
    return integral

def pos_integral20(theta, mu, Si, zetai, gi, theta_i_minus1, lamb, b0, alpha, Rf, delta, nu, x):
    if theta >= 0:
        integral, err = quad(
            lambda t: (-b0 * (theta * (t - Rf) + theta_i_minus1 * gi)**alpha)
                      * dwP_1_Ri(mu, Si, zetai, delta, nu, x),
            Rf - theta_i_minus1 * gi / theta, 100,
            epsrel=1e-8
        )
    else:  # theta < 0
        integral, err = quad(
            lambda t: (lamb * b0 * (theta * (Rf - t) - theta_i_minus1 * gi)**alpha)
                      * dwP_1_Ri(mu, Si, zetai, delta, nu, x),
            Rf - theta_i_minus1 * gi / theta, 100,
            epsrel=1e-8
        )
    return integral

def Equation35(mu, zetai, Si, gi, theta_mi, theta_i_minus1, betai, sigma_m, nu, Rf, gamma_hat, alpha, lamb, b0, delta,x):
    """Equation35 to solve for mu (returned as a 1-element vector)."""
    term1 = (mu[0] + (nu*zetai/(nu-2) - Rf)) - gamma_hat * betai * sigma_m**2
    term2 = -alpha * lamb * b0 * neg_integral(mu[0], Si, zetai, gi, theta_mi, theta_i_minus1, Rf, alpha, delta, nu,x)
    term3 = -alpha * b0 * pos_integral(mu[0], Si, zetai, gi, theta_mi, theta_i_minus1, Rf, alpha, delta, nu,x)
    return term1 + term2 + term3

def Equation20(theta_vec, zetai, mu_hat, nu, σi, βi, sigma_m, theta_mi, theta_i_minus1, Rf, gamma_hat, lamb, b0, Si, gi, alpha, delta,x):
    """Entire objective function to optimize for theta."""
    theta = theta_vec[0]
    term1 = theta * (mu_hat + (nu*zetai/(nu-2) - Rf)) - gamma_hat/2*(theta**2*σi**2 + 2*theta*(βi*sigma_m**2 - theta_mi*σi**2))
    term2 = neg_integral20(theta, mu_hat, Si, zetai, gi, theta_i_minus1, lamb, b0, alpha, Rf, delta, nu,x)
    term3 = pos_integral20(theta, mu_hat, Si, zetai, gi, theta_i_minus1, b0, Rf, alpha, delta, nu,x)
    return -(term1 + term2 + term3)

# -------------------------
# Solve for μ̂ and θ̂ for each portfolio
# -------------------------
# For each portfolio index j (0-indexed in Python)
for j in range(3): #change back to 3 when model is ready
    sigmai = sigma_i[j]
    betai = beta_i[j]
    gi = g_i[j]
    Si = S_i[j]
    zetai = zeta_i[j]
    thetami = theta_all['theta_mi'].iloc[j]
    thetai_minus1 = theta_all['theta_i_minus1'].iloc[j]

    # # Adjusting theta_mi and theta_i_minus depending on portfolio:
    # if j == 0:
    #     theta_mi = theta_all['theta_mi'].iloc[j] / 30  #or 150
    #     theta_i_minus1 = theta_all['theta_i_minus1'].iloc[j] / 30
    # elif j == 1:
    #     theta_mi = theta_all['theta_mi'].iloc[j] / 190 #or 950
    #     theta_i_minus1 = theta_all['theta_i_minus1'].iloc[j] / 190
    # elif j == 2:
    #     theta_mi = theta_all['theta_mi'].iloc[j] / 780 #or 3900
    #     theta_i_minus1 = theta_all['theta_i_minus1'].iloc[j] / 780

    print("Finding mu_hat for portfolio {portfolios[j]}")
    solving = root(lambda mu: np.array([Equation35(mu, zetai, Si, gi, theta_mi, theta_i_minus1,
                                               betai, sigma_m, nu, Rf, gamma_hat, alpha, lamb, b0, delta)]),
               [0.5])
    mu_solution = solving.x[0]
    mu_hat[j] = mu_solution

    # Now optimize Equation20 for theta. We treat theta as a scalar.
    res = minimize(lambda theta: Equation20([theta], zetai, mu_hat[j], nu, sigmai, betai, sigma_m,
                                          theta_mi, theta_i_minus1, Rf, gamma_hat, lamb, b0, Si, gi, alpha, delta),
                   x0=[theta_mi],
                   bounds=[(-theta_mi, theta_mi*2)])
    theta_hat[j] = res.x[0]

# -------------------------
# Save results to CSV
# -------------------------
results_df = pd.DataFrame({
    "portfolio": ["DI", "HY", "IG"],
    "mu": mu_hat,
    "theta": theta_hat
})
results_df.to_csv(os.path.join(project_folder, "data", "results", "mu_theta_results.csv"), index=False)

# -------------------------
# Plotting the results
# -------------------------
def Equation20_plot(theta, zetai, mu_hat, nu, sigma_i, beta_i, sigma_m, theta_mi, theta_i_minus1,
                      Rf, gamma_hat, lamb, b0, Si, gi, alpha, delta):
    """
    Full objective function. 'theta' is a scalar.
    """
    term1 = theta * (mu_hat + (nu * zetai)/(nu - 2) - Rf) - gamma_hat/2 * (theta**2 * sigma_i**2 +
             2 * theta * (beta_i * sigma_m**2 - theta_mi * sigma_i**2))
    term2 = neg_integral20(theta, mu_hat, Si, zetai, gi, theta_i_minus1, lamb, b0, alpha, Rf, delta, nu)
    term3 = pos_integral20(theta, mu_hat, Si, zetai, gi, theta_i_minus1, b0, Rf, alpha, delta, nu)
    return -(term1 + term2 + term3)

def Equation20_MV(theta, mu_hat, nu, zetai, Rf, gamma_hat, beta_i, sigma_m, theta_mi, sigma_i):
    """
    MV part of the objective function (only the first term).
    """
    term1 = theta * (mu_hat + (nu * zetai)/(nu - 2) - Rf) - gamma_hat/2 * (theta**2 * sigma_i**2 +
             2 * theta * (beta_i * sigma_m**2 - theta_mi * sigma_i**2))
    return -(term1)

def Equation20_PT(theta, mu_hat, Si, zetai, gi, theta_i_minus1, lamb, b0, alpha, delta, Rf, nu):
    """
    PT part of the objective function (only the second and third terms).
    """
    term2 = neg_integral20(theta, mu_hat, Si, zetai, gi, theta_i_minus1, lamb, b0, alpha, Rf, delta, nu)
    term3 = pos_integral20(theta, mu_hat, Si, zetai, gi, theta_i_minus1, b0, Rf, alpha, delta, nu)
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
    theta_mi_val = theta_all['theta_mi'].iloc[j]
    theta_i_minus1_val = theta_all['theta_i_minus1'].iloc[j]

    # if j == 0:
    #     theta_mi_val = theta_all['theta_mi'].iloc[j] / 30  #or 150
    #     theta_i_minus1_val = theta_all['theta_i_minus1'].iloc[j] / 30  #or 150
    # elif j == 1:
    #     theta_mi_val = theta_all['theta_mi'].iloc[j] / 190 #or 950
    #     theta_i_minus1_val = theta_all['theta_i_minus1'].iloc[j] / 190 #or 950
    # elif j == 2:
    #     theta_mi_val = theta_all['theta_mi'].iloc[j] / 780 #or 3900
    #     theta_i_minus1_val = theta_all['theta_i_minus1'].iloc[j] / 780 #or 3900

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