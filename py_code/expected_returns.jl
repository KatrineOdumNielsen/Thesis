# =============================================================================
#
#                     Part X: Model Setup A
#                   (Expected Returns Estimation)
#
#         (Considers only subset including the cleaned data)
#
# =============================================================================

using Pkg, Base.Filesystem
Pkg.activate(joinpath(pwd(), "."))

# Get the current working directory in Julia
project_folder = pwd()
cd(joinpath(project_folder))

#Loading packages
# Pkg.add("Statistics")
# Pkg.add("Distributions")
# Pkg.add("LinearAlgebra")
# Pkg.add("Plots")
# Pkg.add("Parameters")
# Pkg.add("PrettyTables")
# Pkg.add("StatsPlots")
# Pkg.add("SpecialFunctions")
# Pkg.add("Optim")
# Pkg.add("QuadGK")
# Pkg.add("NLsolve")
# Pkg.add("ForwardDiff")
# Pkg.add("CSV")
# Pkg.add("DataFrames")
# # Pkg.add("BlackBoxOptim")
# Pkg.add("JuMP")
# Pkg.add("Ipopt")
# Pkg.add("GLPK")
# # Pkg.add(url="https://github.com/JuliaMPC/NLOptControl.jl")
# Pkg.add("GR")
# # Pkg.add("PGFPlotsX")
# # Pkg.add("PlotlyJS")
# # Pkg.add("ORCA")
# Pkg.add("PyPlot")
# # Pkg.add("PlotThemes")



using LinearAlgebra, Random, Distributions, Plots, Parameters, PrettyTables, Printf
using Optim
# using DocStringExtensions
using Plots, StatsPlots
using SpecialFunctions
using QuadGK
using NLsolve
using ForwardDiff
using Optim: converged, maximum, maximizer, minimizer, iterations
using CSV
using DataFrames
using JuMP, Ipopt
using GLPK
Plots.showtheme(:vibrant)
theme(:vibrant)


# ===================================================================    
#                     a. Set parameters        
# ===================================================================
theta_all = DataFrame(CSV.File(joinpath(project_folder, "data", "preprocessed", "thetas_df.csv")))
average_metrics_updated = DataFrame(CSV.File(joinpath(project_folder, "data", "preprocessed", "average_metrics_updated.csv")))

nu = 7.5
σm = 0.25
Rf = 1

γ̂, b0 = (0.6, 0.6)
α, δ, lamb = (0.7, 0.65, 1.5)

σ_i = average_metrics_updated.volatility
β_i = average_metrics_updated.beta
g_i = average_metrics_updated.cap_gain_overhang
S_i = average_metrics_updated.Si
zeta_i = average_metrics_updated.zeta

Ri = 0.01
mu = 0.005 ### Er defineret her, men er ikke brugt i modellen??? ###

# ===================================================================    
#                     b. Calculate μ̂ and θ̂        
# ===================================================================
μ̂ = zeros(3,1)
θ̂ᵢ = zeros(3,1)

# ------ Define model functions ------
# Define function p_Ri (the density function)
function p_Ri(Ri, mu, Si, zetai)
    N = 1
    Kl = besselk((nu + N) / 2, sqrt((nu + ((Ri - mu) ^ 2)/Si) * (zetai^2) /Si))

    result = (2^(1-(nu+N)/2)) / ( gamma(nu/2) * ((pi * nu)^(N/2)) * (abs(Si)^(1/2))) * (Kl * exp( (Ri - mu) / Si * zetai )) / ( (sqrt((nu+((Ri - mu)^2) /Si) * (zetai^2) /Si) )^(-(nu+N)/2) * (1+(Ri - mu)^2 / (Si * nu)) ^((nu+N)/2) )

    return result
end

# Define P_Ri (the cumulative distribution function)
function P_Ri(x, mu, Si, zetai)
    integral, err = quadgk(Ri -> p_Ri(Ri, mu, Si, zetai), -Inf, x, rtol=1e-8)
    return integral
end


# Define dwP_Ri (derivative term for negative side)
function dwP_Ri(x, mu, Si, zetai)
    P = P_Ri(x, mu, Si, zetai)
    # dwP_Ri = ((δ * P**(δ-1) * (P**δ + (1-P)**δ))
    #           - P**δ * (P**(δ-1) - (1-P)**(δ-1))) / \
    #          ((P**δ + (1-P)**δ)**(1+1/δ)) * p_Ri(Ri, mu, Si, zetai)

    return ((δ * P^(δ-1) * (P^δ + (1-P)^δ)) - P^δ * (P^(δ-1) - (1-P)^(δ-1))) /((P^δ + (1-P)^δ)^(1+1/δ)) * p_Ri(x, mu, Si, zetai)
end

# Define dwP_1_Ri (derivative term for positive side)
function dwP_1_Ri(Ri, mu, Si, zetai)
    P = P_Ri(Ri, mu, Si, zetai)
    result = -((δ * (1-P)^(δ-1) * (P^δ + (1-P)^δ)) - (1-P)^δ * ((1-P)^(δ-1) - P^(δ-1))) / ((P^δ + (1-P)^δ)^(1+1/δ)) * p_Ri(Ri, mu, Si, zetai)

    return result
end


# Define neg_integral (integral for the negative part of the function)
function neg_integral(mu, Si, zetai, gi, theta_mi,theta_i_minus1)
    integral, err = quadgk(x -> ((theta_mi * (Rf-x) - theta_i_minus1 * gi) ^(α-1))* (Rf-x) * dwP_Ri(x, mu, Si, zetai), -100, Rf-theta_i_minus1*gi/theta_mi, rtol=1e-8)

    return integral
end

# Define pos_integral (integral for the positive part of the function)
function pos_integral(mu, Si, zetai, gi, theta_mi,theta_i_minus1)
    integral, err = quadgk(x -> ((theta_mi * (x-Rf) + theta_i_minus1 * gi) ^(α-1)) * (x-Rf) * dwP_1_Ri(x, mu, Si, zetai), Rf-theta_i_minus1*gi/theta_mi, 100, rtol=1e-8)

    return integral
end


# Define neg_integral in Equation 20 (negative part of equation 20)
function neg_integral20(θᵢ, mu, Si, zetai, gi,theta_i_minus1,lamb, b0)
    if θᵢ >= 0
        integral, err = quadgk(x -> (-lamb * b0 *(θᵢ * (Rf-x) - theta_i_minus1 * gi ) ^(α)) * dwP_Ri(x, mu, Si, zetai), -100, Rf-theta_i_minus1*gi/θᵢ, rtol=1e-8)
    elseif θᵢ < 0
        integral, err = quadgk(x -> (b0 *(θᵢ * (x-Rf) + theta_i_minus1 * gi) ^(α)) * dwP_Ri(x, mu, Si, zetai), -100, Rf-theta_i_minus1*gi/θᵢ, rtol=1e-8)
    end

    return integral
end

# Define pos_integral in Equation 20 (positive part of equation 20)
function pos_integral20(θᵢ, mu, Si, zetai, gi,theta_i_minus1,lamb, b0)
    if θᵢ >= 0
        integral, err = quadgk(x -> (-b0 * (θᵢ * (x-Rf) + theta_i_minus1 * gi) ^(α)) * dwP_1_Ri(x, mu, Si, zetai), Rf-theta_i_minus1*gi/θᵢ, 100, rtol=1e-8)
    elseif θᵢ < 0
        integral, err = quadgk(x -> (lamb * b0 * (θᵢ * (Rf-x) - theta_i_minus1 * gi ) ^(α)) * dwP_1_Ri(x, mu, Si, zetai), Rf-theta_i_minus1*gi/θᵢ, 100, rtol=1e-8)
    end

    return integral
end

############## TRYING WITH SAFEGUARDS ################
# # Simplified safeguarded version for the negative side of Equation20
# function neg_integral20(θ, mu, Si, zetai, gi, theta_i_minus1, lamb, b0)
#     # If θ or Si is near zero, return 0.0 to avoid division problems.
#     if abs(θ) < 1e-8 || abs(Si) < 1e-8
#         return 0.0
#     end
#     # Compute integration limit
#     limit = Rf - theta_i_minus1 * gi / θ
#     if isnan(limit) || isinf(limit)
#         return 0.0
#     end
#     # Define the integrand; if the inner expression is negative, return 0.0.
#     integrand(x) = begin
#         expr = θ * (Rf - x) - theta_i_minus1 * gi
#         if expr < 0
#             0.0
#         else
#             (-lamb * b0 * expr^α) * dwP_Ri(x, mu, Si, zetai)
#         end
#     end
#     # Try to integrate; if any error occurs or the result is not finite, return 0.0.
#     try
#         integral, _ = quadgk(integrand, -100, limit, rtol=1e-8)
#         return isfinite(integral) ? integral : 0.0
#     catch
#         return 0.0
#     end
# end

# # Simplified safeguarded version for the positive side of Equation20
# function pos_integral20(θ, mu, Si, zetai, gi, theta_i_minus1, lamb, b0)
#     if abs(θ) < 1e-8 || abs(Si) < 1e-8
#         return 0.0
#     end
#     limit = Rf - theta_i_minus1 * gi / θ
#     if isnan(limit) || isinf(limit)
#         return 0.0
#     end
#     integrand(x) = begin
#         expr = θ * (x - Rf) + theta_i_minus1 * gi
#         if expr < 0
#             0.0
#         else
#             (-b0 * expr^α) * dwP_1_Ri(x, mu, Si, zetai)
#         end
#     end
#     try
#         integral, _ = quadgk(integrand, limit, 100, rtol=1e-8)
#         return isfinite(integral) ? integral : 0.0
#     catch
#         return 0.0
#     end
# end

###########################################################

# Solve Equation 35 and get μ̂
function Equation35(mu, zetai, Si, gi, theta_mi, theta_i_minus1, βi, σm, nu,Rf,γ̂, α, lamb, b0)
    term1 = (mu[1] + (nu * zetai / (nu-2) - Rf)) - γ̂ * βi * σm ^ 2
    term2 = -α * lamb * b0 * neg_integral(mu[1], Si, zetai, gi,theta_mi,theta_i_minus1)
    term3 = - α * b0 * pos_integral(mu[1], Si, zetai, gi,theta_mi,theta_i_minus1)

    return term1 + term2 + term3
end


# Equation 20 (entire objective function used for optimization)
function Equation20(θᵢ,zetai,μ̂,nu,σi,βi,σm,theta_mi,theta_i_minus1,Rf,γ̂, lamb, b0,Si,gi)
    term1 = θᵢ[1] * (μ̂ + (nu * zetai)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σi^2 + 2*θᵢ[1]*(βi*σm^2 - theta_mi * σi^2))
    term2 =  neg_integral20(θᵢ[1], μ̂, Si, zetai, gi,theta_i_minus1,lamb, b0)
    term3 =  pos_integral20(θᵢ[1], μ̂, Si, zetai, gi,theta_i_minus1,lamb, b0)

    return -(term1 + term2 + term3)
end

# ------ Run model and solve for μ̂ and θ̂ᵢ for each portfolio ------
for j = 1:1#CHANGE TO 3
    println("Calculating μ̂ and θ̂ᵢ for portfolio ",j)

    σi = σ_i[j]
    βi = β_i[j]
    gi = g_i[j]
    Si = S_i[j]
    zetai = zeta_i[j]
    # theta_mi = theta_m_i[j]
    # theta_i_minus1 = theta_i_minus_1[j]

    if j == 1
        theta_mi = theta_all.theta_mi[j] / 30        # For DI, CHANGE BACK TO 150
        theta_i_minus1 = theta_all.theta_i_minus1[j] / 30 #CHANGE BACK TO 150
    elseif j == 2
        theta_mi = theta_all.theta_mi[j] / 190        # For HY, CHANGE BACK TO 950
        theta_i_minus1 = theta_all.theta_i_minus1[j] / 190       #CHANGE BACK TO 950
    elseif j == 3
        theta_mi = theta_all.theta_mi[j] / 780       # For IG, CHANGE BACK TO 3900
        theta_i_minus1 = theta_all.theta_i_minus1[j] / 780      #CHANGE BACK TO 3900
    end


    println("Finding mu...")
    results = nlsolve(mu -> Equation35(mu, zetai, Si, gi, theta_mi, theta_i_minus1, βi, σm, nu,Rf,γ̂, α, lamb, b0), [0.5])
    μ̂[j] = results.zero[1]
    # Equation35(μ̂)
    println("Finding theta...")
    result2 = optimize(θᵢ  -> Equation20(θᵢ,zetai,μ̂[j],nu,σi,βi,σm,theta_mi,theta_i_minus1,Rf,γ̂, lamb, b0,Si,gi), -theta_mi, theta_mi*2)
    θ̂ᵢ[j] = Optim.minimizer(result2)[1]


end

# ===================================================================    
#                     c. Save results        
# ===================================================================
println("Saving the results...")

portfolio_labels = ["DI", "HY", "IG"]
μ̂_vector = vec(μ̂)
θ̂ᵢ_vector = vec(θ̂ᵢ)

# Create a new DataFrame with the results
results_df = DataFrame(
    portfolio = portfolio_labels,
    mu_hat = μ̂_vector,
    theta_hat = θ̂ᵢ_vector)

CSV.write(joinpath(project_folder, "data","results", "mu_theta_results.csv"), results_df)

# ===================================================================    
#                     d. Show results       
# ===================================================================
println("Creating plots")

# ------ Defining functions for plotting ------

# Entire objective function
function Equation20_plot(θᵢ, μ̂, nu, zetai, Rf, γ̂, βi, σm, theta_mi, theta_i_minus1, σi, lamb, b0, Si, gi)
    term1 = θᵢ[1] * (μ̂ + (nu * zetai)/(nu-2) - Rf) - γ̂ / 2 * (θᵢ[1]^2 * σi^2 + 2*θᵢ[1]*(βi*σm^2 - theta_mi * σi^2))
    term2 = neg_integral20(θᵢ[1], μ̂, Si, zetai, gi, theta_i_minus1, lamb, b0)
    term3 = pos_integral20(θᵢ[1], μ̂, Si, zetai, gi, theta_i_minus1, lamb, b0)
    return -(term1 + term2 + term3)
end

# MV part of the objective function (only first term)
function Equation20_MV(θᵢ,μ̂, nu, zetai, Rf, γ̂, βi, σm, theta_mi, σi)
    term1 = θᵢ[1] * (μ̂ + (nu * zetai)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σi^2 + 2*θᵢ[1]*(βi*σm^2 - theta_mi * σi^2))
    # term2 =  neg_integral20(θᵢ[1], μ̂, Si, zetai, gi,theta_i_minus1,lamb, b0)
    # term3 =  pos_integral20(θᵢ[1], μ̂, Si, zetai, gi,theta_i_minus1,lamb, b0)

    return -(term1)
end

# PT part of the objective function (only second and third terms)
function Equation20_PT(θᵢ,μ̂, Si, zetai, gi,theta_i_minus1,lamb, b0)

    # term1 = θᵢ[1] * (μ̂ + (nu * zetai)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σi^2 + 2*θᵢ[1]*(β*σm^2 - theta_mi * σi^2))
    term2 =  neg_integral20(θᵢ[1], μ̂, Si, zetai, gi,theta_i_minus1,lamb, b0)
    term3 =  pos_integral20(θᵢ[1], μ̂, Si, zetai, gi,theta_i_minus1,lamb, b0)

    return -(term2 + term3)
end

# ------ Running functions and plotting ------

for j in 1:1#CHANGE TO 3
    σi = σ_i[j]
    βi = β_i[j]
    gi = g_i[j]
    Si = S_i[j]
    zetai = zeta_i[j]

    if j == 1
        theta_mi = theta_all.theta_mi[j] / 30        # For DI, CHANGE BACK TO 150
        theta_i_minus1 = theta_all.theta_i_minus1[j] / 30 #CHANGE BACK TO 150
    elseif j == 2
        theta_mi = theta_all.theta_mi[j] / 190        # For HY, CHANGE BACK TO 950
        theta_i_minus1 = theta_all.theta_i_minus1[j] / 190       #CHANGE BACK TO 950
    elseif j == 3
        theta_mi = theta_all.theta_mi[j] / 780       # For IG, CHANGE BACK TO 3900
        theta_i_minus1 = theta_all.theta_i_minus1[j] / 780      #CHANGE BACK TO 3900
    end

    # -------------------------------------
    # Figure 3: Full Objective Function Plot
    # -------------------------------------
    θᵢ_rand = LinRange(0.000001,0.002,100)
    u_rand = [Equation20_plot([θ],μ̂[j],nu,zetai,Rf, γ̂, βi, σm, theta_mi, theta_i_minus1, σi, lamb, b0, Si, gi) for θ in θᵢ_rand]
    θᵢ_rand_neg = LinRange(-0.001,-0.000001,100)
    u_rand_neg = [Equation20_plot([θ],μ̂[j],nu,zetai,Rf, γ̂, βi, σm, theta_mi, theta_i_minus1, σi, lamb, b0, Si, gi) for θ in θᵢ_rand_neg]

    θᵢ_rand_all = [θᵢ_rand_neg; θᵢ_rand]
    u_rand_all = [u_rand_neg; u_rand]

    pyplot()
    Plots.PyPlotBackend()
    plot(θᵢ_rand_all, -u_rand_all, w=3, leg = false, color=:blue, dpi=300)
    xlabel!("θ₁", xguidefontsize=10)
    ylabel!("utility", yguidefontsize=10)
    title!("Objective function of Equation 20", titlefontsize=10)
    savefig("Figure3_portfolio$(j).png")

    # -------------------------------------
    # Figure 4: Overlay MV and PT Parts
    # -------------------------------------
    # Evaluate the functions over a wider range
    θᵢ_rand = LinRange(0.000001,0.25,100)
    u_rand = [Equation20_plot([θ], μ̂[j], nu, zetai, Rf, γ̂, βi, σm, theta_mi, theta_i_minus1, σi, lamb, b0, Si, gi) for θ in θᵢ_rand]    
    MV_rand = [Equation20_MV([θ],μ̂[j], nu, zetai, Rf, γ̂, βi, σm, theta_mi, σi) for θ in θᵢ_rand]
    PT_rand = [Equation20_PT([θ],μ̂[j], Si, zetai, gi,theta_i_minus1,lamb, b0) for θ in θᵢ_rand]

    θᵢ_rand_neg = LinRange(-0.01,-0.00001,100)
    u_rand_neg = [Equation20_plot([θ],μ̂[j],nu, zetai, Rf, γ̂, βi, σm, theta_mi, theta_i_minus1, σi, lamb, b0, Si, gi) for θ in θᵢ_rand_neg]
    MV_rand_neg = [Equation20_MV([θ],μ̂[j], nu, zetai, Rf, γ̂, βi, σm, theta_mi, σi) for θ in θᵢ_rand_neg]
    PT_rand_neg = [Equation20_PT([θ],μ̂[j], Si, zetai, gi,theta_i_minus1,lamb, b0) for θ in θᵢ_rand_neg]

    θᵢ_rand_all = [θᵢ_rand_neg; θᵢ_rand]
    u_rand_all = [u_rand_neg; u_rand]
    MV_rand_all = [MV_rand_neg; MV_rand]
    PT_rand_all = [PT_rand_neg; PT_rand]


    # Create plot showing MV and PT parts
    pyplot()
    Plots.PyPlotBackend()
    plot(θᵢ_rand_all, -u_rand_all, w=2,xlims=(-0.01,0.25), ylims=(-0.004,0.004) ,color=:red, leg = false, dpi=300)
    plot!(θᵢ_rand_all, -MV_rand_all, linestyle=:dash, w=1,xlims=(-0.01,0.25), ylims=(-0.004,0.004) ,leg = false, dpi=300)
    plot!(θᵢ_rand_all, -PT_rand_all, linestyle=:dashdot, w=1,xlims=(-0.01,0.25), ylims=(-0.004,0.004) ,leg = false, dpi=300)
    xlabel!("θ₁₀", xguidefontsize=10)
    ylabel!("utility", yguidefontsize=10)
    title!("Objective function for portfolio$(j)", titlefontsize=10)
    savefig("Figure4_portfolio$(j).png")
end

print("Done")