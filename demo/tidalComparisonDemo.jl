#!/usr/bin/env julia
#=
CosmoNEXUS tidal comparison demo.

Compares the density-based `NEXUSPlus` pipeline with `NEXUSTidal` on the
bundled density cube.
=#

using Pkg
Pkg.activate(@__DIR__)
pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..")))

using CosmoNEXUS
using JLD2
using NeoNEXUS
using Plots

densityPath = joinpath(@__DIR__, "exampleDensity64.jld2")
density = Float32.(load(densityPath)["dens"])

gridSize = size(density, 1)
scales = [3.0, 6.0, 9.0]

println("Running NEXUS+...")
nexusPlus = NEXUSPlus(gridSize, scales)
timePlus = @elapsed thresholdsPlus = nexusPlus(density)

println("Running NEXUS_tidal...")
nexusTidal = NEXUSTidal(gridSize, scales)
timeTidal = @elapsed thresholdsTidal = nexusTidal(density)

println("NEXUS+ thresholds: ", thresholdsPlus)
println("NEXUS_tidal thresholds: ", thresholdsTidal)
println("Timing: NEXUS+ = $(round(timePlus, digits=2)) s, NEXUS_tidal = $(round(timeTidal, digits=2)) s")

sliceIndex = div(gridSize, 2)
wallSigPlus = nexusPlus.wall.significanceMap[:, :, sliceIndex]
wallSigTidal = nexusTidal.wall.significanceMap[:, :, sliceIndex]

logSigPlus = log10.(max.(wallSigPlus, 1f-10))
logSigTidal = log10.(max.(wallSigTidal, 1f-10))

sigMin = min(
    minimum(logSigPlus[logSigPlus .> -10]),
    minimum(logSigTidal[logSigTidal .> -10]),
)
sigMax = max(maximum(logSigPlus), maximum(logSigTidal))

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
    logSigTidal';
    title="NEXUS_tidal Wall Signature",
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
)
contour!(nexusPlus.filament.thresholdMap[:, :, sliceIndex]'; levels=[0.5], color=:green)
contour!(nexusPlus.node.thresholdMap[:, :, sliceIndex]'; levels=[0.5], color=:red)

p4 = contour(
    nexusTidal.wall.thresholdMap[:, :, sliceIndex]';
    title="NEXUS_tidal Thresholds",
    aspect_ratio=1,
    xlabel="x",
    ylabel="y",
    levels=[0.5],
    color=:blue,
)
contour!(nexusTidal.filament.thresholdMap[:, :, sliceIndex]'; levels=[0.5], color=:green)
contour!(nexusTidal.node.thresholdMap[:, :, sliceIndex]'; levels=[0.5], color=:red)

figure = plot(
    p1,
    p2,
    p3,
    p4;
    layout=(2, 2),
    plot_title="NEXUS+ vs NEXUS_tidal Comparison (z=$sliceIndex)",
    margin=5Plots.mm,
    dpi=200,
)

outputPath = joinpath(@__DIR__, "nexusPlusVsTidal.png")
savefig(figure, outputPath)
println("Saved comparison plot to: $outputPath")
