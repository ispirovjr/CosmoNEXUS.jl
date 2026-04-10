"""
Visual comparison of NEXUS+ (density) vs NEXUS_div (velocity divergence).

Generates a 2×2 plot:
- Top-left:  NEXUS+ wall signature (heatmap)
- Top-right: NEXUS_div wall signature (heatmap)
- Bottom-left:  NEXUS+ threshold contours (wall/filament/node)
- Bottom-right: NEXUS_div threshold contours (wall/filament/node)
"""

using NeoNEXUS
using CosmoNEXUS
using JLD2
using Plots
using Statistics

# ─── Load Data ────────────────────────────────────────────────
dataDir = joinpath(@__DIR__)

println("Loading density...")
density = Float32.(load(joinpath(dataDir, "exampleDensity64.jld2"))["dens"])

N = size(density, 1)
println("  Density: $(size(density)), range [$(minimum(density)), $(maximum(density))]")

println("Loading velocity divergence...")
divField = Float32.(load(joinpath(dataDir, "exampleDivergence64.jld2"))["divs"])
println("  Divergence: $(size(divField)), range [$(minimum(divField)), $(maximum(divField))]")

H0 = 70.4f0
θField = divField ./ H0
println("  θ = ∇·v/H0: range [$(minimum(θField)), $(maximum(θField))]")

scales = [3.0, 6.0, 9.0]

# ─── Run NEXUS+ ──────────────────────────────────────────────
println("\n=== Running NEXUS+ (density) ===")
nexusPlus = NEXUSPlus(N, scales)
t1 = @elapsed thresPlus = nexusPlus(density)
println("  Time: $(round(t1, digits=2))s")
println("  Thresholds: node=$(thresPlus.nodeThres), fil=$(thresPlus.filamentThres), wall=$(thresPlus.wallThres)")

# ─── Run NEXUS_div ────────────────────────────────────────────
println("\n=== Running NEXUS_div (velocity divergence) ===")
nexusDiv = NEXUSDiv(N, scales)
t2 = @elapsed thresDiv = nexusDiv(θField)
println("  Time: $(round(t2, digits=2))s")
println("  Thresholds: node=$(thresDiv.nodeThres), fil=$(thresDiv.filamentThres), wall=$(thresDiv.wallThres)")

# ─── Extract mid-z slice ──────────────────────────────────────
midZ = N ÷ 2

wallSigPlus = nexusPlus.wall.significanceMap[:, :, midZ]
wallSigDiv = nexusDiv.wall.significanceMap[:, :, midZ]

# ─── Plot 2×2 ─────────────────────────────────────────────────
gr(size=(1200, 1000))

# Log-scale signatures
logSigPlus = log10.(max.(wallSigPlus, 1f-10))
logSigDiv = log10.(max.(wallSigDiv, 1f-10))

sigMin = min(
    minimum(logSigPlus[logSigPlus.>-10]),
    minimum(logSigDiv[logSigDiv.>-10])
)
sigMax = max(maximum(logSigPlus), maximum(logSigDiv))

p1 = heatmap(logSigPlus', title="NEXUS+ Wall Signature",
    clims=(sigMin, sigMax), color=:inferno, aspect_ratio=1,
    xlabel="x", ylabel="y", colorbar_title="log₁₀(S)")

p2 = heatmap(logSigDiv', title="NEXUS_div Wall Signature",
    clims=(sigMin, sigMax), color=:inferno, aspect_ratio=1,
    xlabel="x", ylabel="y", colorbar_title="log₁₀(S)")

# Contour threshold maps
p3 = contour(nexusPlus.wall.thresholdMap[:, :, midZ]', title="NEXUS+ Thresholds",
    aspect_ratio=1, xlabel="x", ylabel="y", levels=[0.5], color=:blue, linewidth=1.5, label="Wall")
contour!(nexusPlus.filament.thresholdMap[:, :, midZ]',
    levels=[0.5], color=:green, linewidth=1.5, label="Filament")
contour!(nexusPlus.node.thresholdMap[:, :, midZ]',
    levels=[0.5], color=:red, linewidth=1.5, label="Node")

p4 = contour(nexusDiv.wall.thresholdMap[:, :, midZ]', title="NEXUS_div Thresholds",
    aspect_ratio=1, xlabel="x", ylabel="y", levels=[0.5], color=:blue, linewidth=1.5, label="Wall")
contour!(nexusDiv.filament.thresholdMap[:, :, midZ]',
    levels=[0.5], color=:green, linewidth=1.5, label="Filament")
contour!(nexusDiv.node.thresholdMap[:, :, midZ]',
    levels=[0.5], color=:red, linewidth=1.5, label="Node")

fig = plot(p1, p2, p3, p4, layout=(2, 2),
    plot_title="NEXUS+ vs NEXUS_div Comparison (z=$(midZ))",
    margin=5Plots.mm, dpi=200)

outPath = joinpath(@__DIR__, "nexusPlusVsDiv.png")
savefig(fig, outPath)
println("\nSaved comparison plot to: $(outPath)")
