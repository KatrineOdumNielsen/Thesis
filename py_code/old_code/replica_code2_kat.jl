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
σm = 0.07 #changed
Rf = 1 #unchanged

γ̂, b0 = (4, 0.5) #unchanged
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

## list for bounds of integrals
bound = [18,8,8]

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
        # println("Drawing figure 3 for portfolio $j")

        # ### Draw Figure 3 for portfolio j ###
        # function Equation20(θᵢ,μ̂)

        #     term1 = θᵢ[1] * (μ̂ + (nu * zetai)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σᵢ^2 + 2*θᵢ[1]*(βᵢ*σm^2 - theta_mi * σᵢ^2))
        #     term2 =  neg_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)
        #     term3 =  pos_integral20(θᵢ[1], μ̂, Si, zetai, g_i,theta_i_minus1,lamb, b0)

        #     return -(term1 + term2 + term3)
        # end
        
        # #θᵢ_rand = LinRange(0.00001,0.002,50)
        # θᵢ_rand = LinRange(0.0001,0.02,100)
        # u_rand = Equation20.(θᵢ_rand,μ̂[j])

        # #θᵢ_rand_neg = LinRange(-0.001,-0.00001,50)
        # θᵢ_rand_neg = LinRange(-0.01,-0.0001,50)
        # u_rand_neg = Equation20.(θᵢ_rand_neg,μ̂[j])

        # θᵢ_rand_all = [θᵢ_rand_neg; θᵢ_rand]
        # u_rand_all = [u_rand_neg; u_rand]

        #   Plot graphs
        # gr()
        # Plots.GRBackend()
        # pyplot()
        # Plots.PyPlotBackend()
        # plot(θᵢ_rand_all, -u_rand_all, w=3, leg = false, color=:blues, dpi=300)
        # xlabel!("θ₁", xguidefontsize=10)
        # ylabel!("utility", yguidefontsize=10)
        # title!("Objective function of Equation 20 for portfolio $(j)", titlefontsize=10)
        # savefig(joinpath("figures","Figure3_portfolio_$(j).png"))

        # println("done with fig 3")


    elseif abs(θ̂ᵢ[j] - theta_mi) >= 0.00001
        println("$j is a heterogeneous equilibrium")

        μ_pot = LinRange(μ̂[j]-0.01,μ̂[j],30)
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
                utility_low = Equation20(opt_theta_low, μ_pot_i)  # Utility for opt_theta_low
                utility_high = Equation20(opt_theta_high, μ_pot_i)  # Utility for opt_theta_high
            
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

        # Print the row with the lowest u_diff
        println("Row with the lowest u_diff:")
        println(results_df[index_of_min_u_diff, :])

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
        
        hetro_mu = μ̂[j] #CHANGE - between 0.539 (second utility is too low) and 0.515 (too high)

        θᵢ_rand = LinRange(0.0005,0.75,100)
        u_rand = Equation20.(θᵢ_rand,hetro_mu)
        MV_rand = Equation20_MV.(θᵢ_rand,hetro_mu)
        PT_rand = Equation20_PT.(θᵢ_rand,hetro_mu)

        θᵢ_rand_neg = LinRange(-0.01,-0.75,100)
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
        # pyplot()
        # Plots.PyPlotBackend()
        # plot(θᵢ_rand_all, -u_rand_all, w=2,xlims=(-0.01,0.75), ylims=(-0.004,0.004) ,color=:red, leg = false, dpi=300)
        # plot!(θᵢ_rand_all, -MV_rand_all, linestyle=:dash, w=1,xlims=(-0.01,0.75), ylims=(-0.004,0.004) ,leg = false, dpi=300)
        # plot!(θᵢ_rand_all, -PT_rand_all, linestyle=:dashdot, w=1,xlims=(-0.01,0.75), ylims=(-0.004,0.004) ,leg = false, dpi=300)
        # xlabel!("θ₁", xguidefontsize=10)
        # ylabel!("utility", yguidefontsize=10)
        # title!("Objective function for portfolio $(j)", titlefontsize=10)
        # savefig(joinpath("figures", "Figure4_portfolio_$(j).png"))

    end
    exp_exc_ret[j] = μ̂[j] + (nu * zetai)/(nu-2) - Rf
    println("Done with portfolio $j")
end
println(exp_exc_ret)

market_return = exp_exc_ret[1]*30*0.00027080114067964175  + exp_exc_ret[2]*190*0.0006408868164037302 + exp_exc_ret[3]*780*0.001115522398285772
println("Market return is ", market_return)

println("Done with code")