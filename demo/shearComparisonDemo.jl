"""
Visual comparison of NEXUS_tidal vs NEXUS_shear on the example field.

Generates a 2×2 heatmap:
- Top-left:  NEXUS_tidal wall signature
- Top-right: NEXUS_shear wall signature
- Bottom-left:  NEXUS_tidal threshold map (wall+filament+node)
- Bottom-right: NEXUS_shear threshold map (wall+filament+node)
"""

using NeoNEXUS
using CosmoNEXUS
using JLD2
using Plots
using Statistics

# ─── Load Example Data ────────────────────────────────────
dataDir = joinpath(@__DIR__)

println("Loading density...")
density = Float32.(load(joinpath(dataDir, "exampleDensity64.jld2"))["dens"])

N = size(density, 1)
println("Loaded density field: $(size(density))")

println("Loading shear...")
shear = load(joinpath(dataDir, "exampleShear64.jld2"))["shears"]


scales = [3.0, 6.0, 9.0]

# ─── Run NEXUS_tidal ─────────────────────────────────────────
println("\n=== Running NEXUS_tidal ===")
nexusTidal = NEXUSTidal(N, scales)
t1 = @elapsed thresTidal = nexusTidal(density)
println("  Time: $(round(t1, digits=2))s")
println("  Node threshold:     $(thresTidal.nodeThres)")
println("  Filament threshold: $(thresTidal.filamentThres)")
println("  Wall threshold:     $(thresTidal.wallThres)")

# ─── Run NEXUS_shear ─────────────────────────────────────────
println("\n=== Running NEXUS_shear ===")
nexusShear = NEXUSShear(N, scales)
t2 = @elapsed thresShear = nexusShear(density)
println("  Time: $(round(t2, digits=2))s")
println("  Node threshold:     $(thresShear.nodeThres)")
println("  Filament threshold: $(thresShear.filamentThres)")
println("  Wall threshold:     $(thresShear.wallThres)")

# ─── Extract mid-z slice ──────────────────────────────────────
midZ = N ÷ 2

wallSigTidal = nexusTidal.wall.significanceMap[:, :, midZ]
wallSigShear = nexusShear.wall.significanceMap[:, :, midZ]

# ─── Plot 2×2 ─────────────────────────────────────────────────
gr(size=(1200, 1000))

# Use log scale for wall signatures (avoid log(0))
logSigTidal = log10.(max.(wallSigTidal, 1f-10))
logSigShear = log10.(max.(wallSigShear, 1f-10))

# Shared color range for signatures
sigMin = min(minimum(logSigTidal[logSigTidal.>-10]), minimum(logSigShear[logSigShear.>-10]))
sigMax = max(maximum(logSigTidal), maximum(logSigShear))

p1 = heatmap(logSigTidal', title="NEXUS_tidal Wall Signature",
    clims=(sigMin, sigMax), color=:inferno, aspect_ratio=1,
    xlabel="x", ylabel="y", colorbar_title="log₁₀(S)")

p2 = heatmap(logSigShear', title="NEXUS_shear Wall Signature",
    clims=(sigMin, sigMax), color=:inferno, aspect_ratio=1,
    xlabel="x", ylabel="y", colorbar_title="log₁₀(S)")

p3 = contour(nexusTidal.wall.thresholdMap[:, :, midZ]', title="NEXUS_tidal Threshold (Node/Fil/Wall)", aspect_ratio=1,
    xlabel="x", ylabel="y", levels=[0.5], color=:blue)
contour!(nexusTidal.filament.thresholdMap[:, :, midZ]', title="NEXUS_tidal Threshold (Node/Fil/Wall)", aspect_ratio=1,
    xlabel="x", ylabel="y", levels=[0.5], color=:green)
contour!(nexusTidal.node.thresholdMap[:, :, midZ]', title="NEXUS_tidal Threshold (Node/Fil/Wall)", aspect_ratio=1,
    xlabel="x", ylabel="y", levels=[0.5], color=:red)

p4 = contour(nexusShear.wall.thresholdMap[:, :, midZ]', title="NEXUS_shear Threshold (Node/Fil/Wall)", aspect_ratio=1,
    xlabel="x", ylabel="y", levels=[0.5], color=:blue)
contour!(nexusShear.filament.thresholdMap[:, :, midZ]', title="NEXUS_shear Threshold (Node/Fil/Wall)", aspect_ratio=1,
    xlabel="x", ylabel="y", levels=[0.5], color=:green)
contour!(nexusShear.node.thresholdMap[:, :, midZ]', title="NEXUS_shear Threshold (Node/Fil/Wall)", aspect_ratio=1,
    xlabel="x", ylabel="y", levels=[0.5], color=:red)

fig = plot(p1, p2, p3, p4, layout=(2, 2),
    plot_title="NEXUS_tidal vs NEXUS_shear Comparison (z=$(midZ))",
    margin=5Plots.mm, dpi=200)

outPath = joinpath(@__DIR__, "nexusTidalVsShear.png")
savefig(fig, outPath)
println("\nSaved comparison plot to: $(outPath)")