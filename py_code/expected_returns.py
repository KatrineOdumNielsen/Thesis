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
from scipy.special import besselk, gamma
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

# Extract columns from DataFrames (assuming column names are as in Julia)
sigma_i = average_metrics_updated['volatility'].values  # array
beta_i  = average_metrics_updated['beta'].values
g_i     = average_metrics_updated['cap_gain_overhang'].values
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
    Kl = besselk((nu + N)/2, np.sqrt(arg))
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
    epsilon = 1e-10
    if P < epsilon:
        # Debug message:
        print(f"Clamping P at x={x}: original P={P}")
        P = epsilon
    if P > 1:
        print(f"Clamping P at x={x}: original P={P}")
        P = 1
    p_val = p_Ri(x, mu, Si, zetai, nu)
    numerator = (delta * P**(delta-1) * (P**delta + (1-P)**delta)) - P**delta * (P**(delta-1) - (1-P)**(delta-1))
    denominator = (P**delta + (1-P)**delta)**(1+1/delta)
    return numerator/denominator * p_val

def dwP_1_Ri(Ri_val, mu, Si, zetai, delta, nu):
    """Derivative term for the positive side."""
    P = P_Ri(Ri_val, mu, Si, zetai, nu)
    epsilon = 1e-10
    if P < epsilon:
        P = epsilon
    if P > 1:
        P = 1
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

def Equation35(mu, zetai, Si, gi, theta_mi, theta_i_minus1, βi, σm, nu, Rf, γ̂, alpha, lamb, b0, delta):
    """Equation35 to solve for mu (returned as a 1-element vector)."""
    term1 = (mu[0] + (nu*zetai/(nu-2) - Rf)) - γ̂ * βi * σm**2
    term2 = -alpha * lamb * b0 * neg_integral(mu[0], Si, zetai, gi, theta_mi, theta_i_minus1, Rf, alpha, delta, nu)
    term3 = -alpha * b0 * pos_integral(mu[0], Si, zetai, gi, theta_mi, theta_i_minus1, Rf, alpha, delta, nu)
    return term1 + term2 + term3

def Equation20(theta_vec, zetai, mu_val, nu, σi, βi, σm, theta_mi, theta_i_minus1, Rf, γ̂, lamb, b0, Si, gi, alpha, delta, mu_param):
    """Entire objective function to optimize for theta."""
    theta = theta_vec[0]
    term1 = theta * (mu_val + (nu*zetai/(nu-2) - Rf)) - γ̂/2*(theta**2*σi**2 + 2*theta*(βi*σm**2 - theta_mi*σi**2))
    term2 = neg_integral20(theta, mu_param, Si, zetai, gi, theta_i_minus1, lamb, b0, alpha, Rf, delta, nu)
    term3 = pos_integral20(theta, mu_param, Si, zetai, gi, theta_i_minus1, b0, Rf, alpha, delta, nu)
    return -(term1 + term2 + term3)

# For the safeguarded versions of neg_integral20 and pos_integral20 (see .jl version)

# -------------------------
# Solve for μ̂ and θ̂ for each portfolio
# -------------------------
# For each portfolio index j (0-indexed in Python)
for j in range(1): #change back to 3 when model is ready
    σi_val = σ_i[j]
    βi_val = β_i[j]
    gi_val = g_i[j]
    Si_val = S_i[j]
    zetai_val = zeta_i[j]

    # Adjusting theta_mi and theta_i_minus depending on portfolio:
    if j == 0:
        theta_mi = theta_all['theta_mi'].iloc[j] / 150  #or 30
        theta_i_minus1 = theta_all['theta_i_minus1'].iloc[j] / 150
    elif j == 1:
        theta_mi = theta_all['theta_mi'].iloc[j] / 950 #or 190
        theta_i_minus1 = theta_all['theta_i_minus1'].iloc[j] / 950
    elif j == 2:
        theta_mi = theta_all['theta_mi'].iloc[j] / 3900 #or 780
        theta_i_minus1 = theta_all['theta_i_minus1'].iloc[j] / 3900

    print("Finding mu for portfolio $j")
    solving = root(lambda mu: np.array([Equation35(mu, zetai_val, Si_val, gi_val, theta_mi, theta_i_minus1,
                                               βi_val, σm, nu, Rf, γ̂, alpha, lamb, b0, delta)]),
               [0.5])
    mu_solution = solving.x[0]
    mu_hat[j] = mu_solution

    # Now optimize Equation20 for theta. We treat theta as a scalar.
    res = minimize(lambda th: Equation20([th], zetai_val, mu_hat[j], nu, σi_val, βi_val, σm,
                                          theta_mi, theta_i_minus1, Rf, γ̂, lamb, b0, Si_val, gi_val, alpha, delta, mu_solution),
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
# Define a function for plotting the objective (Equation20_plot).
def Equation20_plot(theta_vec, zetai, mu_val, nu, σi, βi, σm, theta_mi, theta_i_minus1, Rf, γ̂, lamb, b0, Si, gi, alpha, delta, mu_param):
    theta = theta_vec[0]
    term1 = theta * (mu_val + (nu*zetai/(nu-2) - Rf)) - γ̂/2*(theta**2*σi**2 + 2*theta*(βi*σm**2 - theta_mi*σi**2))
    term2 = neg_integral20(theta, mu_param, Si, zetai, gi, theta_i_minus1, lamb, b0, alpha, Rf, delta, nu)
    term3 = pos_integral20(theta, mu_param, Si, zetai, gi, theta_i_minus1, b0, Rf, alpha, delta, nu)
    return -(term1 + term2 + term3)

# Prepare a plot for portfolio 1 (for example)
j = 0
σi_val = σ_i[j]
βi_val = β_i[j]
gi_val = g_i[j]
Si_val = S_i[j]
zetai_val = zeta_i[j]
if j == 0:
    theta_mi = theta_all['theta_mi'].iloc[j] / 150
    theta_i_minus1 = theta_all['theta_i_minus1'].iloc[j] / 150
elif j == 1:
    theta_mi = theta_all['theta_mi'].iloc[j] / 950
    theta_i_minus1 = theta_all['theta_i_minus1'].iloc[j] / 950
elif j == 2:
    theta_mi = theta_all['theta_mi'].iloc[j] / 3900
    theta_i_minus1 = theta_all['theta_i_minus1'].iloc[j] / 3900

# Define a range for theta values
theta_range = np.linspace(0.000001, 0.002, 100)
u_values = [Equation20_plot([theta], zetai_val, mu_hat[j], nu, σi_val, βi_val, σm,
                            theta_mi, theta_i_minus1, Rf, γ̂, lamb, b0, Si_val, gi_val, alpha, delta, mu_hat[j])
            for theta in theta_range]

plt.figure()
plt.plot(theta_range, -np.array(u_values), lw=3, color='blue')
plt.xlabel("θ₁")
plt.ylabel("utility")
plt.title("Objective function of Equation 20 for portfolio 1")
plt.savefig(os.path.join(project_folder, "Figure3_portfolio1.png"))
plt.show()