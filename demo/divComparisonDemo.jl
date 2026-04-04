#!/usr/bin/env julia
#=
CosmoNEXUS divergence comparison demo.

Compares the density-based `NEXUSPlus` pipeline with `NEXUSDiv` on the bundled
density and velocity-divergence cubes.
=#

using Pkg
Pkg.activate(@__DIR__)
pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..")))

using CosmoNEXUS
using JLD2
using NeoNEXUS
using Plots

densityPath = joinpath(@__DIR__, "exampleDensity64.jld2")
divergencePath = joinpath(@__DIR__, "exampleDivergence64.jld2")

density = Float32.(load(densityPath)["dens"])
divField = Float32.(load(divergencePath)["divs"])

gridSize = size(density, 1)
H0 = 70.4f0
thetaField = divField ./ H0
scales = [3.0, 6.0, 9.0]

println("Running NEXUS+...")
nexusPlus = NEXUSPlus(gridSize, scales)
timePlus = @elapsed thresholdsPlus = nexusPlus(density)

println("Running NEXUS_div...")
nexusDiv = NEXUSDiv(gridSize, scales)
timeDiv = @elapsed thresholdsDiv = nexusDiv(thetaField)

println("NEXUS+ thresholds: ", thresholdsPlus)
println("NEXUS_div thresholds: ", thresholdsDiv)
println("Timing: NEXUS+ = $(round(timePlus, digits=2)) s, NEXUS_div = $(round(timeDiv, digits=2)) s")

sliceIndex = div(gridSize, 2)
wallSigPlus = nexusPlus.wall.significanceMap[:, :, sliceIndex]
wallSigDiv = nexusDiv.wall.significanceMap[:, :, sliceIndex]

logSigPlus = log10.(max.(wallSigPlus, 1f-10))
logSigDiv = log10.(max.(wallSigDiv, 1f-10))

sigMin = min(
    minimum(logSigPlus[logSigPlus .> -10]),
    minimum(logSigDiv[logSigDiv .> -10]),
)
sigMax = max(maximum(logSigPlus), maximum(logSigDiv))

gr(size=(1200, 1000))

p1 = heatmap(
    logSigPlus';
    title="NEXUS+ Wall Signature",
    clims=(sigMin, sigMax),
    color=:inferno,
    aspect_ratio=1,
    xlabel="x",
    ylabel="y",
    colorbar_title="log10(S)",
)

p2 = heatmap(
    logSigDiv';
    title="NEXUS_div Wall Signature",
    clims=(sigMin, sigMax),
    color=:inferno,
    aspect_ratio=1,
    xlabel="x",
    ylabel="y",
    colorbar_title="log10(S)",
)

p3 = contour(
    nexusPlus.wall.thresholdMap[:, :, sliceIndex]';
    title="NEXUS+ Thresholds",
    aspect_ratio=1,
    xlabel="x",
    ylabel="y",
    levels=[0.5],
    color=:blue,
    linewidth=1.5,
    label="Wall",
)
contour!(nexusPlus.filament.thresholdMap[:, :, sliceIndex]'; levels=[0.5], color=:green, linewidth=1.5, label="Filament")
contour!(nexusPlus.node.thresholdMap[:, :, sliceIndex]'; levels=[0.5], color=:red, linewidth=1.5, label="Node")

p4 = contour(
    nexusDiv.wall.thresholdMap[:, :, sliceIndex]';
    title="NEXUS_div Thresholds",
    aspect_ratio=1,
    xlabel="x",
    ylabel="y",
    levels=[0.5],
    color=:blue,
    linewidth=1.5,
    label="Wall",
)
contour!(nexusDiv.filament.thresholdMap[:, :, sliceIndex]'; levels=[0.5], color=:green, linewidth=1.5, label="Filament")
contour!(nexusDiv.node.thresholdMap[:, :, sliceIndex]'; levels=[0.5], color=:red, linewidth=1.5, label="Node")

figure = plot(
    p1,
    p2,
    p3,
    p4;
    layout=(2, 2),
    plot_title="NEXUS+ vs NEXUS_div Comparison (z=$sliceIndex)",
    margin=5Plots.mm,
    dpi=200,
)

outputPath = joinpath(@__DIR__, "nexusPlusVsDiv.png")
savefig(figure, outputPath)
println("Saved comparison plot to: $outputPath")
