# ====================================================================================
#
#         Part 5A: Estimating the expected returns and model-predicted alphas
#
#                (Considers only subset including the cleaned data)
#
# =====================================================================================

run_title = "title_of_run" 

# ===================================================================    
#                 a. set up Julia and download packages   
# ===================================================================

ENV["JULIA_SSL_CA_ROOTS_PATH"] = ""
ENV["SSL_CERT_FILE"] = ""

# Get the current working directory in Julia
project_folder = pwd()
cd(joinpath(project_folder))

using Pkg, Base.Filesystem
#Pkg.activate(joinpath(pwd(),""))
#Pkg.instantiate()

#Pkg.add(url="https://github.com/JuliaMPC/NLOptControl.jl")
#Pkg.add(PackageSpec(name="KNITRO", version="0.5.0"))
#Pkg.pin("KNITRO")  # This pins the currently resolved version
#Pkg.add("Statistics")
#Pkg.add("Distributions")
#Pkg.add("LinearAlgebra")
#Pkg.add("Parameters")
#Pkg.add("PrettyTables")
#Pkg.add("StatsPlots")
#Pkg.add("SpecialFunctions")
#Pkg.add("Optim")
#Pkg.add("QuadGK")
#Pkg.add("NLsolve")
#Pkg.add("ForwardDiff")
#Pkg.add("CSV")
#Pkg.add("DataFrames")
#Pkg.add("BlackBoxOptim")
#Pkg.add("JuMP")
#Pkg.add("Ipopt")
#Pkg.add("GLPK")
#Pkg.add("GR")
#Pkg.add("PGFPlotsX")
#Pkg.add("PlotlyJS")
#Pkg.add("ORCA")
#Pkg.add("PyPlot")
#Pkg.add("PlotThemes")
#Pkg.add("DocStringExtensions")

using LinearAlgebra, Random, Distributions, Plots, Parameters, PrettyTables, Printf
using Optim
using DocStringExtensions
using Plots, StatsPlots
using SpecialFunctions
using QuadGK
using NLsolve
using NLsolve
using ForwardDiff
using Optim: converged, maximum, maximizer, minimizer, iterations #some extra functions
using CSV
using DataFrames
using BlackBoxOptim
using JuMP, Ipopt
using GLPK
using Dates
using FileIO 
Plots.showtheme(:vibrant)
theme(:vibrant)

# ===================================================================    
#                 b. download data and set up parameters   
# ===================================================================

start_time = now()  # Record the start time

nu = 17
sigma_m = 0.07
Rf = 1 

gamma_hat, b0 = (2, 0.3)
alpha, delta, lambda = (0.7, 0.65, 1.5)

Ri = 0.01
mu = 0.005

theta_all = DataFrame(CSV.File(joinpath(project_folder, "data", "preprocessed", "thetas_df.csv")))
average_metrics_updated = DataFrame(CSV.File(joinpath(project_folder, "data", "preprocessed", "average_metrics_updated.csv")))

sigma_all = average_metrics_updated.volatility
beta_all = average_metrics_updated.beta
g_i_all = average_metrics_updated.cap_gain_overhang ./ 100
Si_all = average_metrics_updated.Si
zetai_all = average_metrics_updated.zeta
theta_mi_all = theta_all.theta_mi
theta_i_minus1_all = theta_all.theta_i_minus1

mu_hat = zeros(3,1)
theta_i_hat = zeros(3,1)
exp_exc_ret = zeros(3,1)
capm_alpha = zeros(3,1)
theta_high = zeros(3,1)
theta_low = zeros(3,1)
x = ones(3,1)              # fraction of investors with low holding (only relevant for hetro equilibrium)
y = zeros(3,1)             # fraction of investors with high holding (only relevant for hetro equilibrium)
bound = [20,20,2.5]        # list for bounds of integrals

# ===================================================================    
#         c. define functions, solve for mu_hat and optimize theta 
# ===================================================================
for j = 1:3
    println("I am calculating mu_hat and theta_i_hat for portfolio ",j)

    L_bound = -bound[j]
    U_bound = bound[j]

    # c1. Store results
    sigma_i = sigma_all[j]
    beta_i = beta_all[j]
    g_i = g_i_all[j]
    Si = Si_all[j]
    zetai = zetai_all[j]
    theta_mi = theta_mi_all[j]
    theta_i_minus1 = theta_i_minus1_all[j]

    # c2. Define functions
    function p_Ri(Ri, mu, Si, zetai)
        threshold = 0.01

        if abs(zetai) < threshold
            # Formula for the case ξ = 0
            return gamma((nu + 1) / 2) / ( gamma(nu / 2) * sqrt(pi * nu * Si) ) *
                   ((1 + ((Ri - mu)^2)*nu^(-1)) / Si)^(-(nu + 1) / 2)
        else
            # Formula for the case ξ ≠ 0
            N = 1
            Kl = besselk((nu + N) / 2, sqrt((nu + ((Ri - mu)^2) / Si) * (zetai^2) / Si))
            result = (2^(1 - (nu + N) / 2)) / ( gamma(nu / 2) * ((pi * nu)^(N / 2)) * sqrt(abs(Si)) ) *
            ( Kl * exp( (Ri - mu) / Si * zetai ) ) /
            ( (sqrt((nu + ((Ri - mu)^2) / Si) * (zetai^2) / Si))^(-(nu + N) / 2) *
              (1 + (Ri - mu)^2 / (Si * nu))^((nu + N) / 2) )
            
            #println("p_Ri: For Ri = $Ri, result = $result")
            
            return result
            
        end
    end

    # Define P_Ri
    function P_Ri(x, mu, Si, zetai)
        #println("P_Ri: Computing integral for x = $x, mu = $mu, Si = $Si, zetai = $zetai")
        integral, err = quadgk(Ri -> p_Ri(Ri, mu, Si, zetai), -Inf, x, rtol=1e-8)
        #println("P_Ri: Integral = $integral, error estimate = $err")
        return integral
    end

    # Define dwP_Ri
    function dwP_Ri(x, mu, Si, zetai)
        P = P_Ri(x, mu, Si, zetai)    
        P = min(P,1)
        if abs(P) < 1e-10 #P == 0
            return P = 1e-10 #so we don't get NaN
        end 
        #println("dwP_Ri: For x = $x, computed P = $P")
        # dwP_Ri = ((delta * P**(delta-1) * (P**delta + (1-P)**delta))
        #           - P**delta * (P**(delta-1) - (1-P)**(delta-1))) / \
        #          ((P**delta + (1-P)**delta)**(1+1/delta)) * p_Ri(Ri, mu, Si, zetai)
        numerator = ((delta * P^(delta-1) * (P^delta + (1-P)^delta)) - P^delta * (P^(delta-1) - (1-P)^(delta-1)))
        denominator = ((P^delta + (1-P)^delta)^(1+1/delta))
        p_val = p_Ri(x, mu, Si, zetai)

        #println("dwP_Ri: For x = $x, numerator = $numerator, denominator = $denominator, p_Ri(x) = $p_val")
    
        result = numerator / denominator * p_val
        return result
        
    end

    function dwP_1_Ri(Ri, mu, Si, zetai)
        # Compute P using P_Ri
        P = P_Ri(Ri, mu, Si, zetai)
        P = min(P,1) # capping at one due to round off errors, whereby P = 1.000000001 is set to P = 1

        if abs(P) < 1e-10 #P == 0
            return P = 1e-10 #so we don't get NaN
        end

        if P == 1
            return result = 0 #so we don't get NaN
        else
            numerator = -((delta * (1 - P)^(delta - 1) * (P^delta + (1 - P)^delta)) - (1 - P)^delta * ((1 - P)^(delta - 1) - P^(delta - 1)))
            denominator = (P^delta + (1 - P)^delta)^(1 + 1/delta)
            
            # Compute p_Ri for the given inputs
            p_val = p_Ri(Ri, mu, Si, zetai)
            #println("dwP_1_Ri: For Ri = $Ri, numerator = $numerator, denominator = $denominator, p_Ri(x) = $p_val")

            result = numerator / denominator * p_val
            #println("dwP_1_Ri: For Ri = $Ri, final result = $result")
            return result 
        end
    end

    function neg_integral(mu, Si, zetai, g_i, theta_mi, theta_i_minus1)
        lower_bound = L_bound
        upper_bound = Rf - theta_i_minus1 * g_i / theta_mi
        #println("neg_integral: Integrating from $lower_bound to $upper_bound")
        integral, err = quadgk(x -> ((theta_mi * (Rf - x) - theta_i_minus1 * g_i)^(alpha - 1)) *
                                 (Rf - x) * dwP_Ri(x, mu, Si, zetai),
                                 lower_bound, upper_bound, rtol=1e-4)
        #println("neg_integral: Result = $integral, error estimate = $err")
        return integral
    end

    # Define pos_integral
    function pos_integral(mu, Si, zetai, g_i, theta_mi,theta_i_minus1)
        lower_bound = Rf - theta_i_minus1 * g_i / theta_mi
        upper_bound = U_bound
        #println("pos_integral: Integrating from $lower_bound to $upper_bound")
        integral, err = quadgk(x -> ((theta_mi * (x-Rf) + theta_i_minus1 * g_i) ^(alpha-1)) * (x-Rf) * dwP_1_Ri(x, mu, Si, zetai), 
        lower_bound, upper_bound, rtol=1e-4)
        #println("pos_integral: Result = $integral, error estimate = $err")
        return integral
    end

    # Define neg_integral in Equation 23
    function neg_integral20(theta_i, mu, Si, zetai, g_i,theta_i_minus1,lambda, b0)
        lower_bound = L_bound
        upper_bound = Rf-theta_i_minus1*g_i/theta_i
        if theta_i >= 0
            integral, err = quadgk(x -> (-lambda * b0 *(theta_i * (Rf-x) - theta_i_minus1 * g_i ) ^(alpha)) * dwP_Ri(x, mu, Si, zetai), 
            lower_bound, upper_bound, rtol=1e-4)
        elseif theta_i < 0
            integral, err = quadgk(x -> (b0 *(theta_i * (x-Rf) + theta_i_minus1 * g_i) ^(alpha)) * dwP_Ri(x, mu, Si, zetai), 
            lower_bound, upper_bound, rtol=1e-4)
        end
        #println("neg_integral20: Result = $integral, error estimate = $err")
        return integral
    end

    # Define pos_integral in Equation 23
    function pos_integral20(theta_i, mu, Si, zetai, g_i,theta_i_minus1,lambda, b0)
        lower_bound = Rf-theta_i_minus1*g_i/theta_i
        upper_bound = U_bound
        if theta_i >= 0
            integral, err = quadgk(x -> (-b0 * (theta_i * (x-Rf) + theta_i_minus1 * g_i) ^(alpha)) * dwP_1_Ri(x, mu, Si, zetai), 
            lower_bound, upper_bound, rtol=1e-4)
        elseif theta_i < 0
            integral, err = quadgk(x -> (lambda * b0 * (theta_i * (Rf-x) - theta_i_minus1 * g_i ) ^(alpha)) * dwP_1_Ri(x, mu, Si, zetai), 
            lower_bound, upper_bound, rtol=1e-4)
        end
        #println("pos_integral20: Result = $integral, error estimate = $err")
        return integral
    end

    function Equation35(mu)
        term1 = (mu[1] + (nu * zetai / (nu-2) - Rf)) - gamma_hat * beta_i * sigma_m ^ 2
        term2 = -alpha * lambda * b0 * neg_integral(mu[1], Si, zetai, g_i,theta_mi,theta_i_minus1)
        term3 = - alpha * b0 * pos_integral(mu[1], Si, zetai, g_i,theta_mi,theta_i_minus1)
        return term1 + term2 + term3
    end

    function Equation20(theta_i,mu_hat)

        term1 = theta_i[1] * (mu_hat + (nu * zetai)/(nu-2) - Rf) - gamma_hat / 2 *(theta_i[1]^2 * sigma_i^2 + 2*theta_i[1]*(beta_i*sigma_m^2 - theta_mi * sigma_i^2))
        term2 =  neg_integral20(theta_i[1], mu_hat, Si, zetai, g_i,theta_i_minus1,lambda, b0)
        term3 =  pos_integral20(theta_i[1], mu_hat, Si, zetai, g_i,theta_i_minus1,lambda, b0)

        return -(term1 + term2 + term3)
    end

    # c.3 Solve Equation 26 and get mu_hat
    results = nlsolve(Equation35, [0.9])
    mu_hat[j] = results.zero[1]
    
    # c.4 Optimize equation 23 and get theta_i_hat
    result2 = optimize(theta_i  -> Equation20(theta_i,mu_hat[j]), -theta_mi, theta_mi*2)
    theta_i_hat[j] = Optim.minimizer(result2)[1]

    println("$j theta is ", theta_i_hat[j])
    println("$j mu is ", mu_hat[j])

    # c.5 Check if theta_i_hat[j] is a homogeneous or heterogeneous equilibrium and draw figures
    if abs(theta_i_hat[j] - theta_mi) < 0.00001
        println("$j is a homogeneous equilibrium")
        println("Drawing homogenous equilibrium for portfolio $j")

        # Draw homogeneous equilibrium structure for portfolio j
        function Equation20(theta_i,mu_hat)

            term1 = theta_i[1] * (mu_hat + (nu * zetai)/(nu-2) - Rf) - gamma_hat / 2 *(theta_i[1]^2 * sigma_i^2 + 2*theta_i[1]*(beta_i*sigma_m^2 - theta_mi * sigma_i^2))
            term2 =  neg_integral20(theta_i[1], mu_hat, Si, zetai, g_i,theta_i_minus1,lambda, b0)
            term3 =  pos_integral20(theta_i[1], mu_hat, Si, zetai, g_i,theta_i_minus1,lambda, b0)

            return -(term1 + term2 + term3)
        end
        
        #theta_i_rand = LinRange(0.00001,0.002,50)
        theta_i_rand = LinRange(0.00001,0.005,100)
        u_rand = Equation20.(theta_i_rand,mu_hat[j])

        #theta_i_rand_neg = LinRange(-0.001,-0.00001,50)
        theta_i_rand_neg = LinRange(-0.0025,-0.00001,100)
        u_rand_neg = Equation20.(theta_i_rand_neg,mu_hat[j])

        theta_i_rand_all = [theta_i_rand_neg; theta_i_rand]
        u_rand_all = [u_rand_neg; u_rand]

        # Store utility values
        theta_low[j] = theta_i_hat[j]

        # Plotting graphs
        pyplot()
        Plots.PyPlotBackend()
        plot(theta_i_rand_all, -u_rand_all, w=3, leg = false, color=:darkblue, dpi=300, xlims = (-0.0015, 0.003), ylims = (-0.0007, -0.0003))
        xlabel!("θ", xguidefontsize=10)
        ylabel!("utility", yguidefontsize=10)
        title!("Objective function of Equation 23 for portfolio $(j)", titlefontsize=10)
        savefig(joinpath("figures","homogeneous_equi_portfolio_$(j).png"))

        println("done with plot for portfolio $j")

        function Equation20_MV_homogeneous(theta_i,mu_hat)

            term1 = theta_i[1] * (mu_hat + (nu * zetai)/(nu-2) - Rf) - gamma_hat / 2 *(theta_i[1]^2 * sigma_i^2 + 2*theta_i[1]*(beta_i*sigma_m^2 - theta_mi * sigma_i^2))
            # term2 =  neg_integral20(theta_i[1], mu_hat, Si, zetai, g_i,theta_i_minus1,lambda, b0)
            # term3 =  pos_integral20(theta_i[1], mu_hat, Si, zetai, g_i,theta_i_minus1,lambda, b0)

            return -(term1)
        end

        function Equation20_PT_homogeneous(theta_i,mu_hat)

            # term1 = theta_i[1] * (mu_hat + (nu * zetai)/(nu-2) - Rf) - gamma_hat / 2 *(theta_i[1]^2 * sigma_i^2 + 2*theta_i[1]*(beta_i*sigma_m^2 - theta_mi * sigma_i^2))
            term2 =  neg_integral20(theta_i[1], mu_hat, Si, zetai, g_i,theta_i_minus1,lambda, b0)
            term3 =  pos_integral20(theta_i[1], mu_hat, Si, zetai, g_i,theta_i_minus1,lambda, b0)

            return -(term2 + term3)
        end
        
    elseif abs(theta_i_hat[j] - theta_mi) >= 0.00001
        println("$j is a heterogeneous equilibrium")

        μ_pot = LinRange(mu_hat[j]-0.025,mu_hat[j],75)
        using DataFrames, Optim

        # Create a DataFrame to store the results
        results_df = DataFrame(μ_pot = Float64[], opt_theta_low = Float64[], opt_theta_high = Float64[], utility_low = Float64[], utility_high = Float64[], utility_diff = Float64[])
        
        # Iterate over all μ_pot values
        for (i, μ_pot_i) in enumerate(μ_pot)
            try
                println("Processing iteration $i out of $(length(μ_pot)) with μ_pot_i = $μ_pot_i")
            
                # Optimize for the range [0, theta_mi]
                result_low = optimize(theta_i -> Equation20(theta_i, μ_pot_i), 0, theta_mi - 0.0001)
                opt_theta_low = Optim.minimizer(result_low)[1]  # Extract the optimal theta for the low range
            
                # Optimize for the range [theta_mi, 1]
                result_high = optimize(theta_i -> Equation20(theta_i, μ_pot_i), theta_mi + 0.0001, 1.0)
                opt_theta_high = Optim.minimizer(result_high)[1]  # Extract the optimal theta for the high range
            
                # Calculate utilities
                utility_low = Equation20(opt_theta_low, μ_pot_i) * -1  # Utility for opt_theta_low
                utility_high = Equation20(opt_theta_high, μ_pot_i) * -1  # Utility for opt_theta_high
            
                # Calculate the difference between the two utilities
                utility_diff = abs(utility_low - utility_high)
            
                # Add the results to the DataFrame
                push!(results_df, (μ_pot_i, opt_theta_low, opt_theta_high, utility_low, utility_high, utility_diff))
            catch e
                println("Error in iteration $i: ", e)
                continue  # Skip to the next iteration if an error occurs
            end
        end

        println(results_df)
        # Find the index of the row with the lowest u_diff
        index_of_min_u_diff = argmin(results_df.utility_diff)

        # Extract the corresponding μ_pot value
        optimal_mu = results_df[index_of_min_u_diff, :μ_pot]

        # Overwrite mu_hat[j] with the optimal μ_pot
        mu_hat[j] = optimal_mu

        # Print the updated mu_hat[j]
        println("Updated mu_hat[$j] with the optimal μ_pot: ", mu_hat[j])

        # Extract the corresponding utility_low value
        optimal_utility_low = results_df[index_of_min_u_diff, :utility_low]

        #Save theta's and holdings
        optimal_theta_low = results_df[index_of_min_u_diff, :opt_theta_low]
        theta_low[j] = optimal_theta_low
        optimal_theta_high = results_df[index_of_min_u_diff, :opt_theta_high]
        theta_high[j] = optimal_theta_high
        x[j] = 1 - (theta_mi - theta_low[j]) / (theta_high[j] - theta_low[j])
        y[j] = (theta_mi - theta_low[j]) / (theta_high[j] - theta_low[j])

        # Print the row with the lowest u_diff
        println("Row with the lowest u_diff:")
        println(results_df[index_of_min_u_diff, :])

        println("Drawing heterogeneous equilibrium for portfolio $j")

        # Draw heterogeneous equilibrium structure for portfolio j
        function Equation20(theta_i,mu_hat)

            term1 = theta_i[1] * (mu_hat + (nu * zetai)/(nu-2) - Rf) - gamma_hat / 2 *(theta_i[1]^2 * sigma_i^2 + 2*theta_i[1]*(beta_i*sigma_m^2 - theta_mi * sigma_i^2))
            term2 =  neg_integral20(theta_i[1], mu_hat, Si, zetai, g_i,theta_i_minus1,lambda, b0)
            term3 =  pos_integral20(theta_i[1], mu_hat, Si, zetai, g_i,theta_i_minus1,lambda, b0)

            return -(term1 + term2 + term3)
        end

        function Equation20_MV(theta_i,mu_hat)

            term1 = theta_i[1] * (mu_hat + (nu * zetai)/(nu-2) - Rf) - gamma_hat / 2 *(theta_i[1]^2 * sigma_i^2 + 2*theta_i[1]*(beta_i*sigma_m^2 - theta_mi * sigma_i^2))
            # term2 =  neg_integral20(theta_i[1], mu_hat, Si, zetai, g_i,theta_i_minus1,lambda, b0)
            # term3 =  pos_integral20(theta_i[1], mu_hat, Si, zetai, g_i,theta_i_minus1,lambda, b0)

            return -(term1)
        end

        function Equation20_PT(theta_i,mu_hat)

            # term1 = theta_i[1] * (mu_hat + (nu * zetai)/(nu-2) - Rf) - gamma_hat / 2 *(theta_i[1]^2 * sigma_i^2 + 2*theta_i[1]*(beta_i*sigma_m^2 - theta_mi * sigma_i^2))
            term2 =  neg_integral20(theta_i[1], mu_hat, Si, zetai, g_i,theta_i_minus1,lambda, b0)
            term3 =  pos_integral20(theta_i[1], mu_hat, Si, zetai, g_i,theta_i_minus1,lambda, b0)

            return -(term2 + term3)
        end
        
        hetro_mu = mu_hat[j]

        theta_i_rand = LinRange(0.0005,0.4,100)
        u_rand = Equation20.(theta_i_rand,hetro_mu)
        MV_rand = Equation20_MV.(theta_i_rand,hetro_mu)
        PT_rand = Equation20_PT.(theta_i_rand,hetro_mu)

        theta_i_rand_neg = LinRange(-0.1,-0.001,50)
        u_rand_neg = Equation20.(theta_i_rand_neg,hetro_mu)
        MV_rand_neg = Equation20_MV.(theta_i_rand_neg,hetro_mu)
        PT_rand_neg = Equation20_PT.(theta_i_rand_neg,hetro_mu)

        theta_i_rand_all = [theta_i_rand_neg; theta_i_rand]
        u_rand_all = [u_rand_neg; u_rand]
        MV_rand_all = [MV_rand_neg; MV_rand]
        PT_rand_all = [PT_rand_neg; PT_rand]

        # Plotting graphs
        pyplot()
        Plots.PyPlotBackend()
        plot(theta_i_rand_all, -u_rand_all, w=2,xlims=(-0.04,0.35), ylims=(-0.002,0.0015) ,color=:darkblue, leg = false, dpi=300)
        plot!(theta_i_rand_all, -MV_rand_all, linestyle=:dash, w=1,xlims=(-0.04,0.35), ylims=(-0.002,0.0015) , color=:lightgreen ,leg = false, dpi=300)
        plot!(theta_i_rand_all, -PT_rand_all, linestyle=:dashdot, w=1,xlims=(-0.04,0.35), ylims=(-0.002,0.0015) , color=:lightseagreen, leg = false, dpi=300)
        xlabel!("θ", xguidefontsize=10)
        ylabel!("utility", yguidefontsize=10)
        title!("Objective function for portfolio $(j)", titlefontsize=10)
        savefig(joinpath("figures", "heterogeneous_equi_portfolio_$(j).png"))

    end
    exp_exc_ret[j] = mu_hat[j] + (nu * zetai)/(nu-2) - Rf
    println("Done with portfolio $j")
end

# ===================================================================    
#         d. calculate and show results 
# ===================================================================
market_return = theta_mi_all[1] * 30 * exp_exc_ret[1] + theta_mi_all[2] * 190 * exp_exc_ret[2] + theta_mi_all[3] * 780 * exp_exc_ret[3]

capm_alpha = exp_exc_ret - beta_all * market_return

println("Low holding of each asset: $theta_low")
println("High holding of each asset: $theta_high")
println("Fraction of investors with low holding: $x")
println("Fraction of investors with high holding: $y")
println("Mu: $mu_hat")
println("Expected excess return: $exp_exc_ret")
println("Market return: $market_return")
println("alpha: $capm_alpha")
println("Done with code")

end_time = now()  # Record the end time
elapsed_time = end_time - start_time  # Calculate the elapsed time
println("Total execution time: $elapsed_time")

# ===================================================================    
#                e. save each run (optional)
# ===================================================================
timestamp = Dates.now()

# flatten vectors into scalars
mu1, mu2, mu3             = mu_hat[1], mu_hat[2], mu_hat[3]
theta1, theta2, theta3    = theta_i_hat[1], theta_i_hat[2], theta_i_hat[3]
exp1, exp2, exp3          = exp_exc_ret[1], exp_exc_ret[2], exp_exc_ret[3]
mkt_ret                   = market_return
alpha1, alpha2, alpha3    = capm_alpha[1], capm_alpha[2], capm_alpha[3]
bound_di, bound_hy, bound_ig = bound[1], bound[2], bound[3]

# collect all the parameters
log_row = DataFrame(
  timestamp      = timestamp,
  run_title      = run_title,
  nu             = nu,
  sigma_m        = sigma_m,
  Rf             = Rf,
  gamma_hat      = gamma_hat,
  b0             = b0,
  alpha_param    = alpha,
  delta          = delta,
  lambda         = lambda,
  Ri             = Ri,
  mu             = mu,
  bound_di       = bound_di,
  bound_hy       = bound_hy,
  bound_ig       = bound_ig,
  mu1            = mu1,
  mu2            = mu2,
  mu3            = mu3,
  theta1         = theta1,
  theta2         = theta2,
  theta3         = theta3,
  exp_exc1       = exp1,
  exp_exc2       = exp2,
  exp_exc3       = exp3,
  market_return  = mkt_ret,
  alpha1         = alpha1,
  alpha2         = alpha2,
  alpha3         = alpha3
)

first_avg = average_metrics_updated[1:1, :] #extract the first row of average_metrics_updated
log_row_full = hcat(log_row, first_avg)

results_dir = joinpath(project_folder, "data", "results")
log_file = joinpath(results_dir, "run_log.csv")
CSV.write(
  log_file,
  log_row_full;
  append = isfile(log_file),
)

println("Logged this run to $log_file")