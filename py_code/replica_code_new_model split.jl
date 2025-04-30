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
σm = 0.08 #changed
Rf = 1 #unchanged

γ̂, b0 = (2, 0.6) #unchanged
a = 5 #new parameter
α, δ, lamb = (0.7, 0.65, 1.5) #unchanged

Ri = 0.01 #changed
mu = 0.005 #changed

theta_all = DataFrame(CSV.File(joinpath(project_folder, "data", "preprocessed", "thetas_df_split.csv")))
average_metrics_updated = DataFrame(CSV.File(joinpath(project_folder, "data", "preprocessed", "average_metrics_split_updated.csv")))

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
μ̂ = zeros(4,1)
θ̂ᵢ = zeros(4,1)
exp_exc_ret = zeros(4,1)
alpha = zeros(4,1)
utility = zeros(4,1)
utility_pt_high = zeros(4,1)
utility_mv_high = zeros(4,1)
utility_pt_low = zeros(4,1)
utility_mv_low = zeros(4,1)
theta_high = zeros(4,1)
theta_low = zeros(4,1)
l = ones(4,1) #fraction of investors with low holding (only relevant for hetro equilibrium)
h = zeros(4,1) #fraction of investors with high holding (only relevant for hetro equilibrium)
const P_eps = 1e-8

## list for bounds of integrals
bound = [5,20,20,15]

for j = 1:4
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
        threshold = 0.02
    
        if abs(zetai) < threshold
            # ξ = 0 → Student-t
            result = gamma((nu + 1) / 2) /
                     (gamma(nu / 2) * sqrt(pi * nu * Si)) *
                     ((1 + ((Ri - mu)^2) / (nu * Si)))^(-(nu + 1) / 2)
    
        else
            # ξ ≠ 0 → skew-t
            N = 1
            z = sqrt((nu + ((Ri - mu)^2) / Si) * (zetai^2) / Si)
            z_cut = 50.0
    
            Kl = z > z_cut ? sqrt(pi / (2*z)) * exp(-z) :
                             besselk((nu + N) / 2, z)
    
            result = (2^(1 - (nu + N) / 2)) /
                     (gamma(nu / 2) * (pi * nu)^(N / 2) * sqrt(abs(Si))) *
                     (Kl * exp((Ri - mu) / Si * zetai)) /
                     ((z)^(-(nu + N) / 2) * (1 + (Ri - mu)^2 / (Si * nu))^((nu + N) / 2))
        end
    
        # —— NEW CLAMPING LOGIC ——
        # never return Inf/NaN or absurdly large values
        if !isfinite(result) || result > 1e6
            return 0.0
        end
    
        return result
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
        # 1) raw density
        p_val = p_Ri(x, mu, Si, zetai)
        if !isfinite(p_val)
            return 0.0
        end
    
        # 2) raw CDF
        P = P_Ri(x, mu, Si, zetai)
        if !isfinite(P)
            return 0.0
        end
    
        # 3) clamp into [P_eps, 1 - P_eps]
        P = clamp(P, P_eps, 1 - P_eps)
    
        # 4) if in extreme tails, weight ≈ 0
        if P ≤ P_eps || P ≥ 1 - P_eps
            return 0.0
        end
    
        # 5) safe to compute numerator/denominator
        num = (δ * P^(δ-1) * (P^δ + (1-P)^δ)) -
              (    P^δ  * (P^(δ-1) - (1-P)^(δ-1)))
    
        den = (P^δ + (1-P)^δ)^(1 + 1/δ)
    
        return (num/den) * p_val
    end
    
    function dwP_1_Ri(x, mu, Si, zetai)
        # 1) raw density
        p_val = p_Ri(x, mu, Si, zetai)
        if !isfinite(p_val)
            return 0.0
        end
    
        # 2) raw CDF
        P = P_Ri(x, mu, Si, zetai)
        if !isfinite(P)
            return 0.0
        end
    
        # 3) clamp into [P_eps, 1 - P_eps]
        P = clamp(P, P_eps, 1 - P_eps)
    
        # 4) if in extreme tails, weight ≈ 0
        if P ≤ P_eps || P ≥ 1 - P_eps
            return 0.0
        end
    
        # 5) safe to compute numerator/denominator
        num = -(
            (δ * (1 - P)^(δ - 1) * (P^δ + (1 - P)^δ))
          - ((1 - P)^δ * ((1 - P)^(δ - 1) - P^(δ - 1)))
        )
    
        den = (P^δ + (1 - P)^δ)^(1 + 1/δ)
    
        return (num/den) * p_val
    end

    function neg_integral(mu, Si, zetai, g_i, theta_mi, theta_i_minus1)
        lower = L_bound
        upper = Rf - theta_i_minus1*g_i/theta_mi
        quadgk(x ->
          begin
            # 1) compute the “tilt argument”
            t = theta_mi*(Rf - x) - theta_i_minus1*g_i
    
            # 2) if it's nonpositive, the integrand is zero
            if t <= 0
              return 0.0
            end
    
            # 3) otherwise safe to take the power
            w = dwP_Ri(x, mu, Si, zetai)
            return t^(α - 1) * (Rf - x) * w
          end,
        lower, upper, rtol=1e-4)[1]
    end

    # Define pos_integral
    function pos_integral(mu, Si, zetai, g_i, theta_mi, theta_i_minus1)
        lower = Rf - theta_i_minus1*g_i/theta_mi
        upper = U_bound
        quadgk(x ->
          begin
            # here the term is θ*(x-Rf)+θ_{i-1}*g
            t = theta_mi*(x - Rf) + theta_i_minus1*g_i
    
            if t <= 0
              return 0.0
            end
    
            w = dwP_1_Ri(x, mu, Si, zetai)
            return t^(α - 1) * (x - Rf) * w
          end,
        lower, upper, rtol=1e-4)[1]
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
        term1 = a * ((mu[1] + (nu * zetai / (nu-2) - Rf)) - γ̂ * βᵢ * σm ^ 2)
        term2 = -α * lamb * b0 * neg_integral(mu[1], Si, zetai, g_i,theta_mi,theta_i_minus1)
        term3 = - α * b0 * pos_integral(mu[1], Si, zetai, g_i,theta_mi,theta_i_minus1)
        return term1 + term2 + term3
    end


    # Equation 20
    function Equation20(θᵢ,μ̂)

        term1 = a * (θᵢ[1] * (μ̂ + (nu * zetai)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σᵢ^2 + 2*θᵢ[1]*(βᵢ*σm^2 - theta_mi * σᵢ^2)))
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

            term1 = a * (θᵢ[1] * (μ̂ + (nu * zetai)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σᵢ^2 + 2*θᵢ[1]*(βᵢ*σm^2 - theta_mi * σᵢ^2)))
            term2 =  neg_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)
            term3 =  pos_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)

            return -(term1 + term2 + term3)
        end
        
        # #θᵢ_rand = LinRange(0.00001,0.002,50)
        # θᵢ_rand = LinRange(0.00005,0.01,100)
        # u_rand = Equation20.(θᵢ_rand,μ̂[j])

        # #θᵢ_rand_neg = LinRange(-0.001,-0.00001,50)
        # θᵢ_rand_neg = LinRange(-0.005,-0.00005,100)
        # u_rand_neg = Equation20.(θᵢ_rand_neg,μ̂[j])

        # θᵢ_rand_all = [θᵢ_rand_neg; θᵢ_rand]
        # u_rand_all = [u_rand_neg; u_rand]

        # Store utility values
        utility[j] = Equation20(θ̂ᵢ[j],μ̂[j])  * -1
        theta_low[j] = θ̂ᵢ[j]

        # #   Plot graphs
        # # gr()
        # # Plots.GRBackend()
        # pyplot()
        # Plots.PyPlotBackend()
        # plot(θᵢ_rand_all, -u_rand_all, w=3, leg = false, color=:blues, dpi=300)
        # xlabel!("θ₁", xguidefontsize=10)
        # ylabel!("utility", yguidefontsize=10)
        # title!("Objective function of Equation 20 for portfolio $(j)", titlefontsize=10)
        # savefig(joinpath("figures","Figure3_with_a,_portfolio$(j).png"))

        # println("done with fig 3")


        function Equation20_MV_homogeneous(θᵢ,μ̂)

            term1 = a * (θᵢ[1] * (μ̂ + (nu * zetai)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σᵢ^2 + 2*θᵢ[1]*(βᵢ*σm^2 - theta_mi * σᵢ^2)))
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
        utility_pt_low[j] = Equation20_PT_homogeneous(θ̂ᵢ[j],μ̂[j]) * (-1)
        utility_mv_low[j] = Equation20_MV_homogeneous(θ̂ᵢ[j],μ̂[j]) * (-1)

    elseif abs(θ̂ᵢ[j] - theta_mi) >= 0.00001
        println("$j is a heterogeneous equilibrium")

        μ_pot = LinRange(μ̂[j]-0.0025,μ̂[j]+0.0025,100)
        using DataFrames, Optim

        # Create a DataFrame to store the results
        results_df = DataFrame(μ_pot = Float64[], opt_theta_low = Float64[], opt_theta_high = Float64[], utility_low = Float64[], utility_high = Float64[], utility_diff = Float64[])
        
        # Iterate over all μ_pot values
        for (i, μ_pot_i) in enumerate(μ_pot)
            try
                println("Processing iteration $i out of $(length(μ_pot)) with μ_pot_i = $μ_pot_i")
            
                # Optimize for the range [0, theta_mi]
                result_low = optimize(θᵢ -> Equation20(θᵢ, μ_pot_i), 0, theta_mi - 0.0001)
                opt_theta_low = Optim.minimizer(result_low)[1]  # Extract the optimal theta for the low range
            
                # Optimize for the range [theta_mi, 1]
                result_high = optimize(θᵢ -> Equation20(θᵢ, μ_pot_i), theta_mi + 0.0001, 1.0)
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

        # Overwrite μ̂[j] with the optimal μ_pot
        μ̂[j] = optimal_mu

        # Print the updated μ̂[j]
        println("Updated μ̂[$j] with the optimal μ_pot: ", μ̂[j])

                
        # Calculate utility
        # Extract the corresponding utility_low value
        optimal_utility_low = results_df[index_of_min_u_diff, :utility_low]

        # Set utility[j] equal to the optimal utility_low
        utility[j] = optimal_utility_low

        #Save theta's and holdings
        optimal_theta_low = results_df[index_of_min_u_diff, :opt_theta_low]
        theta_low[j] = optimal_theta_low
        optimal_theta_high = results_df[index_of_min_u_diff, :opt_theta_high]
        theta_high[j] = optimal_theta_high
        l[j] = 1 - (theta_mi - theta_low[j]) / (theta_high[j] - theta_low[j])
        h[j] = (theta_mi - theta_low[j]) / (theta_high[j] - theta_low[j])

        # Print the row with the lowest u_diff
        println("Row with the lowest u_diff:")
        println(results_df[index_of_min_u_diff, :])

        println("Drawing figure 4 for portfolio $j")
        ### Draw Figure 4 for portfolio j ###
        function Equation20(θᵢ,μ̂)

            term1 = a * (θᵢ[1] * (μ̂ + (nu * zetai)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σᵢ^2 + 2*θᵢ[1]*(βᵢ*σm^2 - theta_mi * σᵢ^2)))
            term2 =  neg_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)
            term3 =  pos_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)

            return -(term1 + term2 + term3)
        end

        function Equation20_MV(θᵢ,μ̂)

            term1 = a * (θᵢ[1] * (μ̂ + (nu * zetai)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σᵢ^2 + 2*θᵢ[1]*(βᵢ*σm^2 - theta_mi * σᵢ^2)))
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

        # θᵢ_rand = LinRange(0.00005,0.3,100)
        # u_rand = Equation20.(θᵢ_rand,hetro_mu)
        # MV_rand = Equation20_MV.(θᵢ_rand,hetro_mu)
        # PT_rand = Equation20_PT.(θᵢ_rand,hetro_mu)

        # θᵢ_rand_neg = LinRange(-0.03,-0.0001,50)
        # u_rand_neg = Equation20.(θᵢ_rand_neg,hetro_mu)
        # MV_rand_neg = Equation20_MV.(θᵢ_rand_neg,hetro_mu)
        # PT_rand_neg = Equation20_PT.(θᵢ_rand_neg,hetro_mu)

        # θᵢ_rand_all = [θᵢ_rand_neg; θᵢ_rand]
        # u_rand_all = [u_rand_neg; u_rand]
        # MV_rand_all = [MV_rand_neg; MV_rand]
        # PT_rand_all = [PT_rand_neg; PT_rand]

        # #   Plot graphs
        # # gr()
        # # Plots.GRBackend()
        # pyplot()
        # Plots.PyPlotBackend()
        # plot(θᵢ_rand_all, -u_rand_all, w=2,xlims=(-0.03,0.25), ylims=(-0.003,0.003) ,color=:red, leg = false, dpi=300)
        # plot!(θᵢ_rand_all, -MV_rand_all, linestyle=:dash, w=1,xlims=(-0.03,0.25), ylims=(-0.003,0.003) ,leg = false, dpi=300)
        # plot!(θᵢ_rand_all, -PT_rand_all, linestyle=:dashdot, w=1,xlims=(-0.03,0.25), ylims=(-0.003,0.003) ,leg = false, dpi=300)
        # xlabel!("θ₁", xguidefontsize=10)
        # ylabel!("utility", yguidefontsize=10)
        # title!("Objective function for portfolio $(j)", titlefontsize=10)
        # savefig(joinpath("figures", "Figure4_with_a,_portfolio_$(j).png"))

        utility_pt_low[j] = Equation20_PT(theta_low[j],μ̂[j]) * -1
        utility_mv_low[j] = Equation20_MV(theta_low[j],μ̂[j]) * -1
        utility_pt_high[j] = Equation20_PT(theta_high[j],μ̂[j]) * -1
        utility_mv_high[j] = Equation20_MV(theta_high[j],μ̂[j]) * -1

    end
    exp_exc_ret[j] = μ̂[j] + (nu * zetai)/(nu-2) - Rf
    println("Done with portfolio $j")
end

#Utilities and alphas:
utility_total = utility[1] * 20 + utility[2] * 20 + utility[3] * 180 + utility[4] * 780

utility_pt = utility_pt_low[1] * 20 * l[1] + utility_pt_high[1] * 20 * h[1] +
              utility_pt_low[2] * 20 * l[2] + utility_pt_high[2] * 20 * h[2] +
              utility_pt_low[3] * 180 * l[3] + utility_pt_high[3] * 180 * h[3]
              utility_pt_low[4] * 780 * l[4] + utility_pt_high[4] * 780 * h[4]

utility_mv = utility_mv_low[1] * 20 * l[1] + utility_mv_high[1] * 20 * h[1] +
              utility_mv_low[2] * 20 * l[2] + utility_mv_high[2] * 20 * h[2] +
              utility_mv_low[3] * 180 * l[3] + utility_mv_high[3] * 180 * h[3]
                utility_mv_low[4] * 780 * l[4] + utility_mv_high[4] * 780 * h[4]

market_return = theta_mi_all[1] * 20 * exp_exc_ret[1] + theta_mi_all[2] * 20 * exp_exc_ret[2] + theta_mi_all[3] * 180 * exp_exc_ret[3] + theta_mi_all[4] * 780 * exp_exc_ret[4]

alpha = exp_exc_ret - βᵢ_all * market_return

pt_total_share = utility_pt / utility_total

println("μ̂: $μ̂")
println("Utility from each asset: $utility")
println("Low holding of each asset: $theta_low")
println("High holding of each asset: $theta_high")
println("Fraction of investors with low holding: $l")
println("Fraction of investors with high holding: $h")
println("Utilty from low holding: $utility_pt_low")
println("Utilty from high holding: $utility_pt_high")
println("Utility total: $utility_total")
println("Utility from prospect theory: $utility_pt")
println("Utility from mean-variance: $utility_mv")
println("Expected excess return: $exp_exc_ret")
println("Market return: $market_return")
println("alpha: $alpha")
println("Share of utility from PT: $pt_total_share")
println("Done with code")