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
#Pkg.activate(joinpath(pwd(),"code"))

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

#%% Set parameters
nu = 7.5
σm = 0.25
Rf = 1

γ̂, b0 = (0.6, 0.6)
α, δ, lamb = (0.7, 0.65, 1.5)

σᵢ_all = momr_std_skew.avg_std
βᵢ_all = momr_beta.avg_beta
g_i_all = momr_gi.avg_gi
Si_all = momr_std_skew.Si
xi_all = momr_std_skew.xi
theta_mi_all = momr_avg_theta_all.avg_theta_mi ./100
theta_i_minus1_all = momr_avg_theta_all.avg_theta_mi ./100


Ri = 0.01
mu = 0.005

#%% Calculate μ̂ and θ̂ᵢ
μ̂ = zeros(10,1)
θ̂ᵢ = zeros(10,1)

for j = 10:10
    println("I am calculating μ̂ and θ̂ᵢ for momentum decile ",j)

    σᵢ = σᵢ_all[j]
    βᵢ = βᵢ_all[j]
    g_i = g_i_all[j]
    Si = Si_all[j]
    xi = xi_all[j]
    theta_mi = theta_mi_all[j]
    theta_i_minus1 = theta_i_minus1_all[j]

    # Define function p_Ri
    function p_Ri(Ri, mu, Si, xi)
        N = 1
        Kl = besselk((nu + N) / 2, sqrt((nu + ((Ri - mu) ^ 2)/Si) * (xi^2) /Si))

        result = (2^(1-(nu+N)/2)) / ( gamma(nu/2) * ((pi * nu)^(N/2)) * (abs(Si)^(1/2))) * (Kl * exp( (Ri - mu) / Si * xi )) / ( (sqrt((nu+((Ri - mu)^2) /Si) * (xi^2) /Si) )^(-(nu+N)/2) * (1+(Ri - mu)^2 / (Si * nu)) ^((nu+N)/2) )

        return result
    end

    # Define P_Ri
    function P_Ri(x, mu, Si, xi)
        integral, err = quadgk(Ri -> p_Ri(Ri, mu, Si, xi), -Inf, x, rtol=1e-8)
        return integral
    end


    # Define dwP_Ri
    function dwP_Ri(x, mu, Si, xi)
        P = P_Ri(x, mu, Si, xi)
        # dwP_Ri = ((δ * P**(δ-1) * (P**δ + (1-P)**δ))
        #           - P**δ * (P**(δ-1) - (1-P)**(δ-1))) / \
        #          ((P**δ + (1-P)**δ)**(1+1/δ)) * p_Ri(Ri, mu, Si, xi)

        return ((δ * P^(δ-1) * (P^δ + (1-P)^δ)) - P^δ * (P^(δ-1) - (1-P)^(δ-1))) /((P^δ + (1-P)^δ)^(1+1/δ)) * p_Ri(x, mu, Si, xi)
    end

    # Define dwP_1_Ri
    function dwP_1_Ri(Ri, mu, Si, xi)
        P = P_Ri(Ri, mu, Si, xi)
        result = -((δ * (1-P)^(δ-1) * (P^δ + (1-P)^δ)) - (1-P)^δ * ((1-P)^(δ-1) - P^(δ-1))) / ((P^δ + (1-P)^δ)^(1+1/δ)) * p_Ri(Ri, mu, Si, xi)

        return result
    end


    # Define neg_integral
    function neg_integral(mu, Si, xi, g_i, theta_mi,theta_i_minus1)
        integral, err = quadgk(x -> ((theta_mi * (Rf-x) - theta_i_minus1 * g_i) ^(α-1))* (Rf-x) * dwP_Ri(x, mu, Si, xi), -100, Rf-theta_i_minus1*g_i/theta_mi, rtol=1e-8)

        return integral
    end

    # Define pos_integral
    function pos_integral(mu, Si, xi, g_i, theta_mi,theta_i_minus1)
        integral, err = quadgk(x -> ((theta_mi * (x-Rf) + theta_i_minus1 * g_i) ^(α-1)) * (x-Rf) * dwP_1_Ri(x, mu, Si, xi), Rf-theta_i_minus1*g_i/theta_mi, 100, rtol=1e-8)

        return integral
    end


    # Define neg_integral in Equation 20
    function neg_integral20(θᵢ, mu, Si, xi, g_i,theta_i_minus1,lamb, b0)
        if θᵢ >= 0
            integral, err = quadgk(x -> (-lamb * b0 *(θᵢ * (Rf-x) - theta_i_minus1 * g_i ) ^(α)) * dwP_Ri(x, mu, Si, xi), -100, Rf-theta_i_minus1*g_i/θᵢ, rtol=1e-8)
        elseif θᵢ < 0
            integral, err = quadgk(x -> (b0 *(θᵢ * (x-Rf) + theta_i_minus1 * g_i) ^(α)) * dwP_Ri(x, mu, Si, xi), -100, Rf-theta_i_minus1*g_i/θᵢ, rtol=1e-8)
        end

        return integral
    end

    # Define pos_integral in Equation 20
    function pos_integral20(θᵢ, mu, Si, xi, g_i,theta_i_minus1,lamb, b0)
        if θᵢ >= 0
            integral, err = quadgk(x -> (-b0 * (θᵢ * (x-Rf) + theta_i_minus1 * g_i) ^(α)) * dwP_1_Ri(x, mu, Si, xi), Rf-theta_i_minus1*g_i/θᵢ, 100, rtol=1e-8)
        elseif θᵢ < 0
            integral, err = quadgk(x -> (lamb * b0 * (θᵢ * (Rf-x) - theta_i_minus1 * g_i ) ^(α)) * dwP_1_Ri(x, mu, Si, xi), Rf-theta_i_minus1*g_i/θᵢ, 100, rtol=1e-8)
        end

        return integral
    end


    # Solve Equation 35 and get μ̂
    function Equation35(mu)
        term1 = (mu[1] + (nu * xi / (nu-2) - Rf)) - γ̂ * βᵢ * σm ^ 2
        term2 = -α * lamb * b0 * neg_integral(mu[1], Si, xi, g_i,theta_mi,theta_i_minus1)
        term3 = - α * b0 * pos_integral(mu[1], Si, xi, g_i,theta_mi,theta_i_minus1)

        return term1 + term2 + term3
    end


    # Equation 20
    function Equation20(θᵢ,μ̂)

        term1 = θᵢ[1] * (μ̂ + (nu * xi)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σᵢ^2 + 2*θᵢ[1]*(βᵢ*σm^2 - theta_mi * σᵢ^2))
        term2 =  neg_integral20(θᵢ[1], μ̂, Si, xi, g_i,theta_i_minus1,lamb, b0)
        term3 =  pos_integral20(θᵢ[1], μ̂, Si, xi, g_i,theta_i_minus1,lamb, b0)

        return -(term1 + term2 + term3)
    end

    results = nlsolve(Equation35, [0.1])
    μ̂[j] = results.zero[1]
    # Equation35(μ̂)

    result2 = optimize(θᵢ  -> Equation20(θᵢ,μ̂[j]), -theta_mi, theta_mi*2)
    θ̂ᵢ[j] = Optim.minimizer(result2)[1]

    println("$j theta is ", θ̂ᵢ[j])
    println("$j mu is ", μ̂[j])
    if abs(θ̂ᵢ[j] - theta_mi) < 0.0000001
        println("$j is a homogeneous equilibrium")
    elseif abs(θ̂ᵢ[j] - theta_mi) >= 0.0000001
        println("$j is a heterogeneous equilibrium")

        μ_pot = LinRange(μ̂[j]-0.1,μ̂[j]+0.1,100)
        using DataFrames, Optim

        # Create a DataFrame to store the results
        results_df = DataFrame(μ_pot = Float64[], opt_theta_low = Float64[], opt_theta_high = Float64[], utility_low = Float64[], utility_high = Float64[], utility_diff = Float64[])
        
        # Iterate over all μ_pot values
        for (i, μ_pot_i) in enumerate(μ_pot)
            println("Processing iteration $i out of $(length(μ_pot)) with μ_pot_i = $μ_pot_i")
        
            # Optimize for the range [0, theta_mi]
            result_low = optimize(θᵢ -> Equation20(θᵢ, μ_pot_i), 0, theta_mi - 0.0001)
            opt_theta_low = Optim.minimizer(result_low)[1]  # Extract the optimal theta for the low range
        
            # Optimize for the range [theta_mi, 1]
            result_high = optimize(θᵢ -> Equation20(θᵢ, μ_pot_i), theta_mi + 0.0001, 1.0)
            opt_theta_high = Optim.minimizer(result_high)[1]  # Extract the optimal theta for the high range
        
            # Calculate utilities
            utility_low = Equation20(opt_theta_low, μ_pot_i)  # Utility for opt_theta_low
            utility_high = Equation20(opt_theta_high, μ_pot_i)  # Utility for opt_theta_high
        
            # Calculate the difference between the two utilities
            utility_diff = abs(utility_low - utility_high)
        
            # Add the results to the DataFrame
            push!(results_df, (μ_pot_i, opt_theta_low, opt_theta_high, utility_low, utility_high, utility_diff))
        end
        println(results_df)
        # Find the index of the row with the lowest u_diff
        index_of_min_u_diff = argmin(results_df.utility_diff)

        # Print the row with the lowest u_diff
        println("Row with the lowest u_diff:")
        println(results_df[index_of_min_u_diff, :])
    end
end

