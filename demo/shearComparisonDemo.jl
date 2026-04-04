#!/usr/bin/env julia
#=
CosmoNEXUS shear comparison demo.

Compares `NEXUSTidal` on the bundled density cube with `NEXUSShear` on the
paired traceless shear and velocity-divergence fields. The bundled
`exampleShear64.jld2` field is traceless, so the demo reconstructs the full
velocity-shear tensor internally by passing the separate `thetaField`.
=#

using Pkg
Pkg.activate(@__DIR__)
pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..", "..")))

using CosmoNEXUS
using JLD2
using NeoNEXUS
using Plots

densityPath = joinpath(@__DIR__, "exampleDensity64.jld2")
shearPath = joinpath(@__DIR__, "exampleShear64.jld2")
divergencePath = joinpath(@__DIR__, "exampleDivergence64.jld2")

density = Float32.(load(densityPath)["dens"])
tracelessShearRaw = load(shearPath)["shears"]
thetaField = Float32.(load(divergencePath)["divs"])

gridSize = size(density, 1)
tracelessShear = Float32[
    tracelessShearRaw[i, j, k][u, v]
    for i in 1:gridSize, j in 1:gridSize, k in 1:gridSize, u in 1:3, v in 1:3
]

scales = [3.0, 6.0, 9.0]

println("Running NEXUS_tidal...")
nexusTidal = NEXUSTidal(gridSize, scales)
timeTidal = @elapsed thresholdsTidal = nexusTidal(density)

println("Running NEXUS_shear with traceless shear + theta...")
nexusShear = NEXUSShear(gridSize, scales)
timeShear = @elapsed thresholdsShear = nexusShear(tracelessShear, thetaField)

println("NEXUS_tidal thresholds: ", thresholdsTidal)
println("NEXUS_shear thresholds: ", thresholdsShear)
println("Timing: NEXUS_tidal = $(round(timeTidal, digits=2)) s, NEXUS_shear = $(round(timeShear, digits=2)) s")

sliceIndex = div(gridSize, 2)
wallSigTidal = nexusTidal.wall.significanceMap[:, :, sliceIndex]
wallSigShear = nexusShear.wall.significanceMap[:, :, sliceIndex]

logSigTidal = log10.(max.(wallSigTidal, 1f-10))
logSigShear = log10.(max.(wallSigShear, 1f-10))

sigMin = min(
    minimum(logSigTidal[logSigTidal .> -10]),
    minimum(logSigShear[logSigShear .> -10]),
)
sigMax = max(maximum(logSigTidal), maximum(logSigShear))

gr(size=(1200, 1000))

p1 = heatmap(
    logSigTidal';
    title="NEXUS_tidal Wall Signature",
    clims=(sigMin, sigMax),
    color=:inferno,
    aspect_ratio=1,
    xlabel="x",
    ylabel="y",
    colorbar_title="log10(S)",
)

p2 = heatmap(
    logSigShear';
    title="NEXUS_shear Wall Signature",
    clims=(sigMin, sigMax),
    color=:inferno,
    aspect_ratio=1,
    xlabel="x",
    ylabel="y",
    colorbar_title="log10(S)",
)

p3 = contour(
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

p4 = contour(
    nexusShear.wall.thresholdMap[:, :, sliceIndex]';
    title="NEXUS_shear Thresholds",
    aspect_ratio=1,
    xlabel="x",
    ylabel="y",
    levels=[0.5],
    color=:blue,
)
contour!(nexusShear.filament.thresholdMap[:, :, sliceIndex]'; levels=[0.5], color=:green)
contour!(nexusShear.node.thresholdMap[:, :, sliceIndex]'; levels=[0.5], color=:red)

figure = plot(
    p1,
    p2,
    p3,
    p4;
    layout=(2, 2),
    plot_title="NEXUS_tidal vs NEXUS_shear Comparison (z=$sliceIndex)",
    margin=5Plots.mm,
    dpi=200,
)

outputPath = joinpath(@__DIR__, "nexusTidalVsShear.png")
savefig(figure, outputPath)
println("Saved comparison plot to: $outputPath")
