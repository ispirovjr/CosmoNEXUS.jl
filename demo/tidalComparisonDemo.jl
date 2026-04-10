"""
Visual comparison of NEXUS+ vs NEXUS_tidal on the example density field.

Generates a 2×2 heatmap:
- Top-left:  NEXUS+ wall signature
- Top-right: NEXUS_tidal wall signature
- Bottom-left:  NEXUS+ threshold map (wall+filament+node)
- Bottom-right: NEXUS_tidal threshold map (wall+filament+node)
"""

using NeoNEXUS
using CosmoNEXUS
using JLD2
using Plots
using Statistics

# ─── Load Example Density ────────────────────────────────────
dataDir = joinpath(@__DIR__)

println("Loading density...")
density = Float32.(load(joinpath(dataDir, "exampleDensity64.jld2"))["dens"])

N = size(density, 1)
println("Loaded density field: $(size(density))")

scales = [3.0, 6.0, 9.0]

# ─── Run NEXUS+ ──────────────────────────────────────────────
println("\n=== Running NEXUS+ ===")
nexusPlus = NEXUSPlus(N, scales)
t1 = @elapsed thresPlus = nexusPlus(density)
println("  Time: $(round(t1, digits=2))s")
println("  Node threshold:     $(thresPlus.nodeThres)")
println("  Filament threshold: $(thresPlus.filamentThres)")
println("  Wall threshold:     $(thresPlus.wallThres)")

# ─── Run NEXUS_tidal ─────────────────────────────────────────
println("\n=== Running NEXUS_tidal ===")
nexusTidal = NEXUSTidal(N, scales)
t2 = @elapsed thresTidal = nexusTidal(density)
println("  Time: $(round(t2, digits=2))s")
println("  Node threshold:     $(thresTidal.nodeThres)")
println("  Filament threshold: $(thresTidal.filamentThres)")
println("  Wall threshold:     $(thresTidal.wallThres)")

# ─── Extract mid-z slice ──────────────────────────────────────
midZ = N ÷ 2

wallSigPlus = nexusPlus.wall.significanceMap[:, :, midZ]
wallSigTidal = nexusTidal.wall.significanceMap[:, :, midZ]

# ─── Plot 2×2 ─────────────────────────────────────────────────
gr(size=(1200, 1000))

# Use log scale for wall signatures (avoid log(0))
logSigPlus = log10.(max.(wallSigPlus, 1f-10))
logSigTidal = log10.(max.(wallSigTidal, 1f-10))

# Shared color range for signatures
sigMin = min(minimum(logSigPlus[logSigPlus.>-10]), minimum(logSigTidal[logSigTidal.>-10]))
sigMax = max(maximum(logSigPlus), maximum(logSigTidal))

p1 = heatmap(logSigPlus', title="NEXUS+ Wall Signature",
    clims=(sigMin, sigMax), color=:inferno, aspect_ratio=1,
    xlabel="x", ylabel="y", colorbar_title="log₁₀(S)")

p2 = heatmap(logSigTidal', title="NEXUS_tidal Wall Signature",
    clims=(sigMin, sigMax), color=:inferno, aspect_ratio=1,
    xlabel="x", ylabel="y", colorbar_title="log₁₀(S)")

p3 = contour(nexusPlus.wall.thresholdMap[:, :, midZ]', title="NEXUS+ Threshold (Node/Fil/Wall)", aspect_ratio=1,
    xlabel="x", ylabel="y", levels=[0.5], color=:blue)
contour!(nexusPlus.filament.thresholdMap[:, :, midZ]', title="NEXUS+ Threshold (Node/Fil/Wall)", aspect_ratio=1,
    xlabel="x", ylabel="y", levels=[0.5], color=:green)
contour!(nexusPlus.node.thresholdMap[:, :, midZ]', title="NEXUS+ Threshold (Node/Fil/Wall)", aspect_ratio=1,
    xlabel="x", ylabel="y", levels=[0.5], color=:red)

p4 = contour(nexusTidal.wall.thresholdMap[:, :, midZ]', title="NEXUS_tidal Threshold (Node/Fil/Wall)", aspect_ratio=1,
    xlabel="x", ylabel="y", levels=[0.5], color=:blue)
contour!(nexusTidal.filament.thresholdMap[:, :, midZ]', title="NEXUS_tidal Threshold (Node/Fil/Wall)", aspect_ratio=1,
    xlabel="x", ylabel="y", levels=[0.5], color=:green)
contour!(nexusTidal.node.thresholdMap[:, :, midZ]', title="NEXUS_tidal Threshold (Node/Fil/Wall)", aspect_ratio=1,
    xlabel="x", ylabel="y", levels=[0.5], color=:red)

fig = plot(p1, p2, p3, p4, layout=(2, 2),
    plot_title="NEXUS+ vs NEXUS_tidal Comparison (z=$(midZ))",
    margin=5Plots.mm, dpi=200)

outPath = joinpath(@__DIR__, "nexusPlusVsTidal.png")
savefig(fig, outPath)
println("\nSaved comparison plot to: $(outPath)")
