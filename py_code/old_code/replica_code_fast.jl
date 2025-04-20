# ======================================================================================================================
#                           Replicate Barberis, Jin, and Wang (2021)
#                              Part 3a: computing expected returns
#
#                                       Author: Gen Li
#                                         03/14/2021
#
#   Note: I run Part 1 on Google Datalab from Google Cloud Platform (GCP). It takes around 1.5h to finish part 1 with
#         32-core CPU and parallel computation.
#
#
# ======================================================================================================================
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
Plots.showtheme(:vibrant)
theme(:vibrant)

#%% Import parameters generated from python part 3a
momr_avg_theta_all = DataFrame(CSV.File(joinpath(project_folder, "data", "raw", "momr_avg_theta_all.csv")))
momr_beta = DataFrame(CSV.File(joinpath(project_folder, "data", "raw", "momr_avg_beta_all.csv")))
momr_gi = DataFrame(CSV.File(joinpath(project_folder, "data", "raw", "momr_avg_g_i_all.csv")))
momr_std_skew = DataFrame(CSV.File(joinpath(project_folder, "data", "raw", "momr_avg_std_skew_Si_xi_all.csv")))

#%% Set parameters (their parameters)
#nu = 7.5
#σm = 0.25
#Rf = 1

# γ̂, b0 = (0.6, 0.6)
# α, δ, lamb = (0.7, 0.65, 1.5)

# σᵢ_all = momr_std_skew.avg_std
# βᵢ_all = momr_beta.avg_beta
# g_i_all = momr_gi.avg_gi
# Si_all = momr_std_skew.Si
# xi_all = momr_std_skew.xi
# theta_mi_all = momr_avg_theta_all.avg_theta_mi ./100
# theta_i_minus1_all = momr_avg_theta_all.avg_theta_mi ./100

#Ri = 0.01
#mu = 0.005

## =========== Our parameters ============= ##
nu = 17 #changed
σm = 0.055 #changed
Rf = 1 #unchanged

γ̂, b0 = (2, 0.4) #unchanged
α, δ, lamb = (0.7, 0.65, 1.5) #unchanged

Ri = 0.01 #changed
mu = 0.005 #changed

theta_all = DataFrame(CSV.File(joinpath(project_folder, "data", "preprocessed", "thetas_df.csv")))
average_metrics_updated = DataFrame(CSV.File(joinpath(project_folder, "data", "preprocessed", "average_metrics_updated.csv")))

σᵢ_all = average_metrics_updated.volatility
βᵢ_all = average_metrics_updated.beta
g_i_all = average_metrics_updated.cap_gain_overhang ./ 100
Si_all = average_metrics_updated.Si
zetai_all = average_metrics_updated.zeta
theta_mi_all = theta_all.theta_mi
theta_i_minus1_all = theta_all.theta_i_minus1

## 

#%% Calculate μ̂ and θ̂ᵢ
# μ̂ = zeros(10,1)
# θ̂ᵢ = zeros(10,1)
μ̂ = zeros(3,1)
θ̂ᵢ = zeros(3,1)
exp_exc_ret = zeros(3,1)
alpha = zeros(3,1)
utility = zeros(3,1)
utility_pt_high = zeros(3,1)
utility_mv_high = zeros(3,1)
utility_pt_low = zeros(3,1)
utility_mv_low = zeros(3,1)
utility_no_investment = zeros(3,1)
theta_high = zeros(3,1)
theta_low = zeros(3,1)
x = ones(3,1) #fraction of investors with low holding (only relevant for hetro equilibrium)
y = zeros(3,1) #fraction of investors with high holding (only relevant for hetro equilibrium)
test = zeros(3,1)

## list for bounds of integrals
bound = [20,20,15]

for j = 1:3
    println("I am calculating μ̂ and θ̂ᵢ for portfolio ",j)

    L_bound = -bound[j]
    U_bound = bound[j]

    σᵢ = σᵢ_all[j]
    βᵢ = βᵢ_all[j]
    g_i = g_i_all[j]
    Si = Si_all[j]
    zetai = zetai_all[j]
    theta_mi = theta_mi_all[j]
    theta_i_minus1 = theta_i_minus1_all[j]

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
        # dwP_Ri = ((δ * P**(δ-1) * (P**δ + (1-P)**δ))
        #           - P**δ * (P**(δ-1) - (1-P)**(δ-1))) / \
        #          ((P**δ + (1-P)**δ)**(1+1/δ)) * p_Ri(Ri, mu, Si, zetai)
        numerator = ((δ * P^(δ-1) * (P^δ + (1-P)^δ)) - P^δ * (P^(δ-1) - (1-P)^(δ-1)))
        denominator = ((P^δ + (1-P)^δ)^(1+1/δ))
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
            numerator = -((δ * (1 - P)^(δ - 1) * (P^δ + (1 - P)^δ)) - (1 - P)^δ * ((1 - P)^(δ - 1) - P^(δ - 1)))
            denominator = (P^δ + (1 - P)^δ)^(1 + 1/δ)
            
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
        integral, err = quadgk(x -> ((theta_mi * (Rf - x) - theta_i_minus1 * g_i)^(α - 1)) *
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
        integral, err = quadgk(x -> ((theta_mi * (x-Rf) + theta_i_minus1 * g_i) ^(α-1)) * (x-Rf) * dwP_1_Ri(x, mu, Si, zetai), 
        lower_bound, upper_bound, rtol=1e-4)
        #println("pos_integral: Result = $integral, error estimate = $err")
        return integral
    end

    # Define neg_integral in Equation 20
    function neg_integral20(θᵢ, mu, Si, zetai, g_i,theta_i_minus1,lamb, b0)
        lower_bound = L_bound
        upper_bound = Rf-theta_i_minus1*g_i/θᵢ
        if θᵢ >= 0
            integral, err = quadgk(x -> (-lamb * b0 *(θᵢ * (Rf-x) - theta_i_minus1 * g_i ) ^(α)) * dwP_Ri(x, mu, Si, zetai), 
            lower_bound, upper_bound, rtol=1e-4)
        elseif θᵢ < 0
            integral, err = quadgk(x -> (b0 *(θᵢ * (x-Rf) + theta_i_minus1 * g_i) ^(α)) * dwP_Ri(x, mu, Si, zetai), 
            lower_bound, upper_bound, rtol=1e-4)
        end
        #println("neg_integral20: Result = $integral, error estimate = $err")
        return integral
    end

    # Define pos_integral in Equation 20
    function pos_integral20(θᵢ, mu, Si, zetai, g_i,theta_i_minus1,lamb, b0)
        lower_bound = Rf-theta_i_minus1*g_i/θᵢ
        upper_bound = U_bound
        if θᵢ >= 0
            integral, err = quadgk(x -> (-b0 * (θᵢ * (x-Rf) + theta_i_minus1 * g_i) ^(α)) * dwP_1_Ri(x, mu, Si, zetai), 
            lower_bound, upper_bound, rtol=1e-4)
        elseif θᵢ < 0
            integral, err = quadgk(x -> (lamb * b0 * (θᵢ * (Rf-x) - theta_i_minus1 * g_i ) ^(α)) * dwP_1_Ri(x, mu, Si, zetai), 
            lower_bound, upper_bound, rtol=1e-4)
        end
        #println("pos_integral20: Result = $integral, error estimate = $err")
        return integral
    end

    # Solve Equation 35 and get μ̂
    function Equation35(mu)
        term1 = (mu[1] + (nu * zetai / (nu-2) - Rf)) - γ̂ * βᵢ * σm ^ 2
        term2 = -α * lamb * b0 * neg_integral(mu[1], Si, zetai, g_i,theta_mi,theta_i_minus1)
        term3 = - α * b0 * pos_integral(mu[1], Si, zetai, g_i,theta_mi,theta_i_minus1)
        return term1 + term2 + term3
    end


    # Equation 20
    function Equation20(θᵢ,μ̂)

        term1 = θᵢ[1] * (μ̂ + (nu * zetai)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σᵢ^2 + 2*θᵢ[1]*(βᵢ*σm^2 - theta_mi * σᵢ^2))
        term2 =  neg_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)
        term3 =  pos_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)

        return -(term1 + term2 + term3)
    end

    results = nlsolve(Equation35, [0.1])
    μ̂[j] = results.zero[1]
    # Equation35(μ̂)

    result2 = optimize(θᵢ  -> Equation20(θᵢ,μ̂[j]), -theta_mi, theta_mi*2)
    θ̂ᵢ[j] = Optim.minimizer(result2)[1]

    println("$j theta is ", θ̂ᵢ[j])
    println("$j mu is ", μ̂[j])


    if abs(θ̂ᵢ[j] - theta_mi) < 0.00001
        println("$j is a homogeneous equilibrium")
        println("Drawing figure 3 for portfolio $j")

        ### Draw Figure 3 for portfolio j ###
        function Equation20(θᵢ,μ̂)

            term1 = θᵢ[1] * (μ̂ + (nu * zetai)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σᵢ^2 + 2*θᵢ[1]*(βᵢ*σm^2 - theta_mi * σᵢ^2))
            term2 =  neg_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)
            term3 =  pos_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)

            return -(term1 + term2 + term3)
        end
        
        #θᵢ_rand = LinRange(0.00001,0.002,50)
        θᵢ_rand = LinRange(0.00001,0.005,100)
        u_rand = Equation20.(θᵢ_rand,μ̂[j])

        #θᵢ_rand_neg = LinRange(-0.001,-0.00001,50)
        θᵢ_rand_neg = LinRange(-0.0025,-0.00001,100)
        u_rand_neg = Equation20.(θᵢ_rand_neg,μ̂[j])

        θᵢ_rand_all = [θᵢ_rand_neg; θᵢ_rand]
        u_rand_all = [u_rand_neg; u_rand]
        test[j] = (u_rand[1] + u_rand_neg[end]) / 2  * -1

        # Store utility values
        utility[j] = Equation20(θ̂ᵢ[j],μ̂[j])  * -1
        utility_no_investment[j] = 0.5 * Equation20.(0.00001,μ̂[j])  * -1 + 0.5 * Equation20.(-0.00001,μ̂[j])  * -1
        theta_low[j] = θ̂ᵢ[j]

        #   Plot graphs
        # gr()
        # Plots.GRBackend()
        pyplot()
        Plots.PyPlotBackend()
        plot(θᵢ_rand_all, -u_rand_all, w=3, leg = false, color=:blues, dpi=300)
        xlabel!("θ₁", xguidefontsize=10)
        ylabel!("utility", yguidefontsize=10)
        title!("Objective function of Equation 20 for portfolio $(j)", titlefontsize=10)
        savefig(joinpath("figures","Figure3_fast_portfolio_$(j).png"))

        println("done with fig 3")


        function Equation20_MV_homogeneous(θᵢ,μ̂)

            term1 = θᵢ[1] * (μ̂ + (nu * zetai)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σᵢ^2 + 2*θᵢ[1]*(βᵢ*σm^2 - theta_mi * σᵢ^2))
            # term2 =  neg_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)
            # term3 =  pos_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)

            return -(term1)
        end

        function Equation20_PT_homogeneous(θᵢ,μ̂)

            # term1 = θᵢ[1] * (μ̂ + (nu * zetai)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σᵢ^2 + 2*θᵢ[1]*(βᵢ*σm^2 - theta_mi * σᵢ^2))
            term2 =  neg_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)
            term3 =  pos_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)

            return -(term2 + term3)
        end
        utility_pt_low[j] = Equation20_PT_homogeneous(θ̂ᵢ[j],μ̂[j]) * -1
        utility_mv_low[j] = Equation20_MV_homogeneous(θ̂ᵢ[j],μ̂[j]) * -1

    elseif abs(θ̂ᵢ[j] - theta_mi) >= 0.00001
        println("$j is a heterogeneous equilibrium")

        using DataFrames, Optim
        
        # Define a function to calculate the utility difference for a given μ_pot
        function utility_difference(μ_pot, theta_mi)
            try
                # Optimize for the range [0, theta_mi]
                result_low = optimize(θᵢ -> Equation20(θᵢ, μ_pot), 0, theta_mi)
                opt_theta_low = Optim.minimizer(result_low)[1]

                # Optimize for the range [theta_mi, 1]
                result_high = optimize(θᵢ -> Equation20(θᵢ, μ_pot), theta_mi + 0.01, 1.0)
                opt_theta_high = Optim.minimizer(result_high)[1]

                # Calculate utilities
                utility_low = Equation20(opt_theta_low, μ_pot) * -1
                utility_high = Equation20(opt_theta_high, μ_pot) * -1

                # Return the absolute difference between the two utilities
                return abs(utility_low - utility_high)
            catch e
                # Handle errors by returning a large value
                return Inf  # Return a large value to indicate failure
            end
        end

        # Initial bounds for μ_pot
        μ_pot_lower = μ̂[j] - 0.02
        μ_pot_upper = μ̂[j]
        adjustment_step = 0.001  # Increment to adjust the lower bound

        # Define a flag to track whether optimization succeeded
        optimization_success = false
        optimal_mu = nothing  # Initialize optimal_mu to ensure it is defined

        # Loop to adjust the lower bound dynamically
        while μ_pot_lower < μ_pot_upper
            try
                # Minimize the utility difference with respect to μ_pot
                result = optimize(μ_pot -> utility_difference(μ_pot, theta_mi), μ_pot_lower, μ_pot_upper)
                optimal_mu = Optim.minimizer(result)
                optimization_success = true
                break  # Exit the loop if optimization succeeds
            catch e
                μ_pot_lower += adjustment_step  # Increase the lower bound
                println("Adjusting μ_pot_lower to $μ_pot_lower")
            end
        end

        # Check if optimization succeeded
        if optimization_success && optimal_mu !== nothing
            # Calculate the corresponding optimal thetas and utilities
            result_low = optimize(θᵢ -> Equation20(θᵢ, optimal_mu), 0, theta_mi - 0.0001)
            opt_theta_low = Optim.minimizer(result_low)[1]

            result_high = optimize(θᵢ -> Equation20(θᵢ, optimal_mu), theta_mi + 0.0001, 1.0)
            opt_theta_high = Optim.minimizer(result_high)[1]

            utility_low = Equation20(opt_theta_low, optimal_mu) * -1
            utility_high = Equation20(opt_theta_high, optimal_mu) * -1
            utility_diff = abs(utility_low - utility_high)

            # Print results
            println("Utility difference in portfolio $j: ", utility_diff)
            println("Optimal theta_low for portfolio $j: ", opt_theta_low)
            println("Optimal theta_high for portfolio $j: ", opt_theta_high)

            # Update utility and μ̂[j]
            utility[j] = utility_low
            μ̂[j] = optimal_mu
            println("Found optimization for portfolio $j")
        else
            println("Optimization failed for portfolio $j after adjusting bounds.")
            # Provide a fallback value for optimal_mu if needed
            println("Did not find a valid μ_pot for portfolio $j")
        end

        utility_no_investment[j] = 0.5 * Equation20(0.00005,μ̂[j]) * -1 + 0.5 * Equation20(-0.00005,μ̂[j]) * -1

        #Save theta's and holdings
        theta_low[j] = opt_theta_low
        theta_high[j] = opt_theta_high
        x[j] = 1 - (theta_mi - theta_low[j]) / (theta_high[j] - theta_low[j])
        y[j] = (theta_mi - theta_low[j]) / (theta_high[j] - theta_low[j])

        println("Drawing figure 4 for portfolio $j")
        ### Draw Figure 4 for portfolio j ###
        function Equation20(θᵢ,μ̂)

            term1 = θᵢ[1] * (μ̂ + (nu * zetai)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σᵢ^2 + 2*θᵢ[1]*(βᵢ*σm^2 - theta_mi * σᵢ^2))
            term2 =  neg_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)
            term3 =  pos_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)

            return -(term1 + term2 + term3)
        end

        function Equation20_MV(θᵢ,μ̂)

            term1 = θᵢ[1] * (μ̂ + (nu * zetai)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σᵢ^2 + 2*θᵢ[1]*(βᵢ*σm^2 - theta_mi * σᵢ^2))
            # term2 =  neg_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)
            # term3 =  pos_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)

            return -(term1)
        end

        function Equation20_PT(θᵢ,μ̂)

            # term1 = θᵢ[1] * (μ̂ + (nu * zetai)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σᵢ^2 + 2*θᵢ[1]*(βᵢ*σm^2 - theta_mi * σᵢ^2))
            term2 =  neg_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)
            term3 =  pos_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)

            return -(term2 + term3)
        end
        
        hetro_mu = μ̂[j]

        θᵢ_rand = LinRange(0.0005,0.4,100)
        u_rand = Equation20.(θᵢ_rand,hetro_mu)
        MV_rand = Equation20_MV.(θᵢ_rand,hetro_mu)
        PT_rand = Equation20_PT.(θᵢ_rand,hetro_mu)

        θᵢ_rand_neg = LinRange(-0.1,-0.001,50)
        u_rand_neg = Equation20.(θᵢ_rand_neg,hetro_mu)
        MV_rand_neg = Equation20_MV.(θᵢ_rand_neg,hetro_mu)
        PT_rand_neg = Equation20_PT.(θᵢ_rand_neg,hetro_mu)

        θᵢ_rand_all = [θᵢ_rand_neg; θᵢ_rand]
        u_rand_all = [u_rand_neg; u_rand]
        MV_rand_all = [MV_rand_neg; MV_rand]
        PT_rand_all = [PT_rand_neg; PT_rand]

        #   Plot graphs
        # gr()
        # Plots.GRBackend()
        pyplot()
        Plots.PyPlotBackend()
        plot(θᵢ_rand_all, -u_rand_all, w=2,xlims=(-0.1,0.4), ylims=(-0.004,0.002) ,color=:red, leg = false, dpi=300)
        plot!(θᵢ_rand_all, -MV_rand_all, linestyle=:dash, w=1,xlims=(-0.1,0.4), ylims=(-0.004,0.002) ,leg = false, dpi=300)
        plot!(θᵢ_rand_all, -PT_rand_all, linestyle=:dashdot, w=1,xlims=(-0.1,0.4), ylims=(-0.004,0.002) ,leg = false, dpi=300)
        xlabel!("θ₁", xguidefontsize=10)
        ylabel!("utility", yguidefontsize=10)
        title!("Objective function for portfolio $(j)", titlefontsize=10)
        savefig(joinpath("figures", "Figure4_fast_portfolio_$(j).png"))

        utility_pt_low[j] = Equation20_PT(theta_low[j],μ̂[j]) * -1
        utility_mv_low[j] = Equation20_MV(theta_low[j],μ̂[j]) * -1
        utility_pt_high[j] = Equation20_PT(theta_high[j],μ̂[j]) * -1
        utility_mv_high[j] = Equation20_MV(theta_high[j],μ̂[j]) * -1

    end
    exp_exc_ret[j] = μ̂[j] + (nu * zetai)/(nu-2) - Rf
    println("Done with portfolio $j")
end

utility_total = utility[1] * 100 + utility[2] * 100 + utility[3] * 100

utility_equal = utility[1] + utility[2] + utility[3]

utility_no_investment_total = utility_no_investment[1] * 30 + utility_no_investment[2] * 190 + utility_no_investment[3] * 780

utility_pt = utility_pt_low[1] * 100 * x[1] + utility_pt_high[1] * 100 * y[1] +
              utility_pt_low[2] * 100 * x[2] + utility_pt_high[2] * 100 * y[2] +
              utility_pt_low[3] * 100 * x[3] + utility_pt_high[3] * 100 * y[3]

utility_pt_equal = utility_pt_low[1] * x[1] + utility_pt_high[1] * y[1] +
              utility_pt_low[2] * x[2] + utility_pt_high[2] * y[2] +
              utility_pt_low[3] * x[3] + utility_pt_high[3] * y[3]

utility_mv = utility_mv_low[1] * 100 * x[1] + utility_mv_high[1] * 190 * y[1] +
              utility_mv_low[2] * 100 * x[2] + utility_mv_high[2] * 100 * y[2] +
              utility_mv_low[3] * 100 * x[3] + utility_mv_high[3] * 100 * y[3]

utility_mv_equal = utility_mv_low[1] * x[1] + utility_mv_high[1] * y[1] +
              utility_mv_low[2] * x[2] + utility_mv_high[2] * y[2] +
              utility_mv_low[3] * x[3] + utility_mv_high[3] * y[3]
market_return = theta_mi_all[1] * 100 * exp_exc_ret[1] + theta_mi_all[2] * 100 * exp_exc_ret[2] + theta_mi_all[3] * 100 * exp_exc_ret[3]

alpha = exp_exc_ret - βᵢ_all * market_return

pt_equal_share = utility_pt_equal / utility_equal

pt_total_share = utility_pt / utility_total

mv_incremental_share = utility_mv / (utility_total - utility_no_investment_total)

println("Utility from each asset: $utility")
println("Utility from each asset with no investment: $utility_no_investment")
println("test: $test")
println("Low holding of each asset: $theta_low")
println("High holding of each asset: $theta_high")
println("Fraction of investors with low holding: $x")
println("Fraction of investors with high holding: $y")
println("Utilty from low holding: $utility_pt_low")
println("Utilty from high holding: $utility_pt_high")
println("Utility total: $utility_total")
println("Utility from prospect theory: $utility_pt")
println("Utility from mean-variance: $utility_mv")
println("Utility from no investment: $utility_no_investment_total")
println("Utility from pt low: $utility_pt_low")
println("Utility from pt high: $utility_pt_high")
println("Utility from mv low: $utility_mv_low")
println("Utility from mv high: $utility_mv_high")
println("Expected excess return: $exp_exc_ret")
println("Market return: $market_return")
println("alpha: $alpha")
print("utility from equal investment: $utility_equal")
println("utility from pt equal investment: $utility_pt_equal")
println("utility from mv equal investment: $utility_mv_equal")
println("pt equal share: $pt_equal_share")
println("pt total share: $pt_total_share")
println("mv incremental share: $mv_incremental_share")
println("Done with code")