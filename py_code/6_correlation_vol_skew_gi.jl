# ====================================================================================
#
#    Part 6: Estimating the correlation between volatility, skewness and gain overhang
#
#              (Considers only subset including the cleaned data)
#
# =====================================================================================

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
Plots.showtheme(:vibrant)
theme(:bright)

# ===================================================================    
#                      b. Load data
# ===================================================================
theta_all = DataFrame(CSV.File(joinpath(project_folder, "data", "preprocessed", "thetas_df.csv")))
average_metrics_updated = DataFrame(CSV.File(joinpath(project_folder, "data", "preprocessed", "average_metrics_updated.csv")))

# ===================================================================    
#            c. Plotting correlation figures
# ===================================================================
# Correlation figure between volatility and skewness
pyplot()
Plots.PyPlotBackend()
labels = ["DI", "HY", "IG"]
plot(average_metrics_updated.volatility, average_metrics_updated.skewness, marker_z   = 1:3, 
    colormap=:blues, framestyle=:box, xlims=(0,0.4), ylims=(0,0.9),linetype=:scatter ,
    markershape=:octagon, markersize=11,leg = false, dpi=300)
xlabel!("standard deviation", xguidefontsize=12)
ylabel!("skewness", yguidefontsize=12)
for (xi, yi, lab) in zip(average_metrics_updated.volatility, average_metrics_updated.skewness, labels)
    annotate!(
      xi, yi + 0.05,
      text(lab, 10, halign = :center)
    )
end
savefig(joinpath("figures", "correlation_vol_skew.png"))

# Correlation figure between volatility and gain overhang
pyplot()
Plots.PyPlotBackend()
labels = ["DI", "HY", "IG"]
plot(average_metrics_updated.volatility, average_metrics_updated.cap_gain_overhang, marker_z   = 1:3, 
    colormap=:blues, framestyle=:box, xlims=(0,0.4), ylims=(-20,5), linetype=:scatter ,
    markershape=:octagon, markersize=12,leg = false, dpi=300)
xlabel!("standard deviation", xguidefontsize=12)
ylabel!("gain overhang", yguidefontsize=12)
for (xi, yi, lab) in zip(
    average_metrics_updated.volatility,
    average_metrics_updated.cap_gain_overhang,
    labels,)
  annotate!(
    xi, yi + 1,
    text(lab, 10, halign = :center))
end
savefig(joinpath("figures", "correlation_vol_CGO.png"))

# Correlation figure between skewness and gain overhang
pyplot()
Plots.PyPlotBackend()
labels = ["DI", "HY", "IG"]
plot(average_metrics_updated.skewness, average_metrics_updated.cap_gain_overhang, marker_z   = 1:3, 
    colormap=:blues, framestyle=:box, xlims=(0,0.9), ylims=(-20,5),linetype=:scatter ,
    markershape=:octagon, markersize=12,leg = false, dpi=300)
xlabel!("skewness", xguidefontsize=12)
ylabel!("gain overhang", yguidefontsize=12)
for (xi, yi, lab) in zip(
    average_metrics_updated.skewness,
    average_metrics_updated.cap_gain_overhang,
    labels,
  )
  annotate!(
    xi, yi + 1,
    text(lab, 10, halign = :center)
  )
end
savefig(joinpath("figures", "correlation_skew_CGO.png"))

print("Done")