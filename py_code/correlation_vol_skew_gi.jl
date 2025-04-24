# ======================================================================================================================
#                           Replicate Barberis, Jin, and Wang (2021)
#                              Part 3d: computing expected returns
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
theme(:bright)

### Our parameters ###
theta_all = DataFrame(CSV.File(joinpath(project_folder, "data", "preprocessed", "thetas_df.csv")))
average_metrics_updated = DataFrame(CSV.File(joinpath(project_folder, "data", "preprocessed", "average_metrics_updated.csv")))

#%% Draw figure 2
# pyplot()
# Plots.PyPlotBackend()
# l = @layout [a  b; c]


# p2 = plot!(momr_param_all.avg_std, momr_param_all.avg_gi, linetype=:scatter ,markershape=:star5, markersize=10,leg = false, dpi=300)
# xlabel!("standard deviation", xguidefontsize=10)
# ylabel!("gain overhang", yguidefontsize=10)
# p3 = plot(momr_param_all.avg_skew, momr_param_all.avg_gi, linetype=:scatter ,markershape=:star5, markersize=10,leg = false, dpi=300)
# xlabel!("skewness", xguidefontsize=10)
# ylabel!("gain overhang", yguidefontsize=10)
# plot(p1, p2, p3, layout = l)

# title!("Objective function of Equation 20", titlefontsize=10)
# gr()
# Plots.GRBackend()
pyplot()
Plots.PyPlotBackend()
labels = ["DI", "HY", "IG"]
plot(average_metrics_updated.volatility, average_metrics_updated.skewness, marker_z   = 1:3, 
    colormap=:blues, framestyle=:box, xlims=(0,0.4), ylims=(0,0.9),linetype=:scatter ,
    markershape=:octagon, markersize=11,leg = false, dpi=300)
xlabel!("standard deviation", xguidefontsize=12)
ylabel!("skewness", yguidefontsize=12)
for (xi, yi, lab) in zip(x, y, labels)
    annotate!(
      xi, yi + 0.05,                    # shift label 0.02 up
      text(lab, 10, halign = :center)    # font size 8, centered
    )
end
savefig(joinpath("figures", "Figure2a.png"))


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
savefig(joinpath("figures", "Figure2b.png"))

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
savefig(joinpath("figures", "Figure2c.png"))

print("Done")