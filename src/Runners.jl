"""
    NEXUSTidal

NEXUS_tidal pipeline (Cautun et al. 2013).

Uses the tidal field tensor (Hessian of gravitational potential) instead of the
density Hessian. The tidal tensor is computed via Fourier-space Poisson inversion:
`T_ij(k) = -(k_i k_j / k²) ρ̂(k)`.
Same filtering and thresholding pipeline as NEXUS+.

# Fields
- `filter::AbstractScaleFilter` — scale-space filter
- `node::NodeFeature` — node detector
- `filament::LineFeature` — filament detector
- `wall::SheetFeature` — wall detector
- `scales::Vector{Float64}` — smoothing scales
"""
struct NEXUSTidal
    filter::AbstractScaleFilter
    node::NodeFeature
    filament::LineFeature
    wall::SheetFeature
    scales::Vector{Float64}
end

"""
    NEXUSTidal(gridSize::Int, scales)

Convenience constructor for a cubic grid of side `gridSize`.
"""
function NEXUSTidal(gridSize::Int, scales::Vector{Float64})
    kx = FFTW.rfftfreq(gridSize) .* gridSize .* 2π
    ky = kz = FFTW.fftfreq(gridSize) .* gridSize .* 2π
    sheet = SheetFeature((gridSize, gridSize, gridSize), kx, ky, kz)
    line = LineFeature((gridSize, gridSize, gridSize), kx, ky, kz)
    node = NodeFeature((gridSize, gridSize, gridSize), kx, ky, kz)

    return NEXUSTidal(GaussianFourierFilter((gridSize, gridSize, gridSize)), node, line, sheet, scales)
end

"""
    run(runner::NEXUSTidal, densityField) -> NamedTuple

Execute the NEXUS_tidal pipeline. Identical to NEXUS+ but computes eigenvalues
of the tidal tensor rather than the density Hessian.

Returns `(nodeThres, filamentThres, wallThres)`.
"""
function NeoNEXUS.run(runner::NEXUSTidal, densityField::AbstractArray{<:Real,3})
    meanρ = Statistics.mean(densityField)
    normDensity = densityField ./ meanρ

    gridSize = size(runner.node.significanceMap)
    cache = HessianEigenCache(gridSize...)

    # === Scale Loop ===
    for scale in runner.scales
        R² = scale^2

        linFiltered = runner.filter(normDensity, scale, runner.node) .* R²
        computeTidalEigenvalues!(linFiltered, runner.node.kx, runner.node.ky, runner.node.kz, cache)
        sigMap = NeoNEXUS.computeSignature(runner.node, cache)
        @. runner.node.significanceMap = max(runner.node.significanceMap, sigMap)

        logFiltered = runner.filter(normDensity, scale, runner.filament) .* R²
        computeTidalEigenvalues!(logFiltered, runner.filament.kx, runner.filament.ky, runner.filament.kz, cache)
        sigMap = NeoNEXUS.computeSignature(runner.filament, cache)
        @. runner.filament.significanceMap = max(runner.filament.significanceMap, sigMap)

        sigMap = NeoNEXUS.computeSignature(runner.wall, cache)
        @. runner.wall.significanceMap = max(runner.wall.significanceMap, sigMap)
    end

    # === Thresholding (hierarchical with signature masking) ===
    nodeThres = findComponentPercentageThreshold!(
        runner.node,
        normDensity,
        370.0,
        0.50;
        excludeEmpty=false
    )

    maskSignatureMap!(runner.filament, runner.node)
    filamentThres = deltaMSquaredThreshold!(runner.filament, normDensity)

    maskSignatureMap!(runner.wall, runner.node)
    maskSignatureMap!(runner.wall, runner.filament)
    wallThres = deltaMSquaredThreshold!(runner.wall, normDensity)

    return (nodeThres=nodeThres, filamentThres=filamentThres, wallThres=wallThres)
end


"""
    runMultithreaded(runner::NEXUSTidal, densityField) -> NamedTuple

Multithreaded variant of [`run`](@ref) for [`NEXUSTidal`](@ref).

Parallelises the scale loop using `Threads.@threads`; each scale gets its own
cache and signature arrays. Thresholding is identical to the sequential version.

!!! note
    Requires `julia --threads=N`. FFTW internal threading is disabled
    automatically for thread safety.
"""
function NeoNEXUS.runMultithreaded(runner::NEXUSTidal, densityField::AbstractArray{<:Real,3})
    # Disable FFTW internal threading for thread safety
    FFTW.set_num_threads(1)

    meanρ = Statistics.mean(densityField)
    normDensity = densityField ./ meanρ

    gridSize = size(runner.node.significanceMap)
    nScales = length(runner.scales)

    nodeSigs = [zeros(Float32, gridSize) for _ in 1:nScales]
    filaSigs = [zeros(Float32, gridSize) for _ in 1:nScales]
    wallSigs = [zeros(Float32, gridSize) for _ in 1:nScales]

    # === Parallel scale loop ===
    Threads.@threads for idx in 1:nScales
        scale = runner.scales[idx]
        R² = scale^2

        linFiltered = runner.filter(normDensity, scale, runner.node) .* R²
        localCache = computeTidalEigenvalues(linFiltered, runner.node.kx, runner.node.ky, runner.node.kz)
        nodeSigs[idx] .= NeoNEXUS.computeSignature(runner.node, localCache)

        logFiltered = runner.filter(normDensity, scale, runner.filament) .* R²
        computeTidalEigenvalues!(logFiltered, runner.filament.kx, runner.filament.ky, runner.filament.kz, localCache)
        filaSigs[idx] .= NeoNEXUS.computeSignature(runner.filament, localCache)
        wallSigs[idx] .= NeoNEXUS.computeSignature(runner.wall, localCache)
    end

    for idx in 1:nScales
        @. runner.node.significanceMap = max(runner.node.significanceMap, nodeSigs[idx])
        @. runner.filament.significanceMap = max(runner.filament.significanceMap, filaSigs[idx])
        @. runner.wall.significanceMap = max(runner.wall.significanceMap, wallSigs[idx])
    end

    # 1. Nodes: find threshold where 50% of components have >= 370x density
    nodeThres = findComponentPercentageThreshold!(
        runner.node,
        normDensity,
        370.0,
        0.50;
        excludeEmpty=false
    )

    # 2. Filaments: mask by nodes, then ΔM² threshold
    maskSignatureMap!(runner.filament, runner.node)
    filamentThres = deltaMSquaredThreshold!(runner.filament, normDensity)

    # 3. Walls: mask by nodes and filaments, then ΔM² threshold
    maskSignatureMap!(runner.wall, runner.node)
    maskSignatureMap!(runner.wall, runner.filament)
    wallThres = deltaMSquaredThreshold!(runner.wall, normDensity)

    return (nodeThres=nodeThres, filamentThres=filamentThres, wallThres=wallThres)
end


"""
    NEXUSDiv

NEXUS_div pipeline — velocity divergence classification.

Uses the velocity divergence field `θ(x) = ∇·v(x)/H` as input instead of density.
Implemented as the standard density Hessian on `-θ` (negation restores sign
conventions). Uses uniform Gaussian filtering for all features (no log₁₀
transform, since θ can be negative).

# Fields
- `filter::AbstractScaleFilter` — scale-space filter
- `node::NodeFeature` — node detector
- `filament::LineFeature` — filament detector
- `wall::SheetFeature` — wall detector
- `scales::Vector{Float64}` — smoothing scales
"""
struct NEXUSDiv
    filter::AbstractScaleFilter
    node::NodeFeature
    filament::LineFeature
    wall::SheetFeature
    scales::Vector{Float64}
end

"""
    NEXUSDiv(gridSize::Int, scales)

Convenience constructor for a cubic grid of side `gridSize`.
"""
function NEXUSDiv(gridSize::Int, scales::Vector{Float64})
    kx = FFTW.rfftfreq(gridSize) .* gridSize .* 2π
    ky = kz = FFTW.fftfreq(gridSize) .* gridSize .* 2π
    sheet = SheetFeature((gridSize, gridSize, gridSize), kx, ky, kz)
    line = LineFeature((gridSize, gridSize, gridSize), kx, ky, kz)
    node = NodeFeature((gridSize, gridSize, gridSize), kx, ky, kz)

    return NEXUSDiv(GaussianFourierFilter((gridSize, gridSize, gridSize)), node, line, sheet, scales)
end


"""
    run(runner::NEXUSDiv, θField) -> NamedTuple

Execute the NEXUS_div pipeline on a velocity divergence field `θ = ∇·v/H`.

Negates θ so that the standard density Hessian produces the correct positive-sign
tidal tensor. Uses plain Gaussian filtering for all features (no log₁₀ transform
since θ can be negative).

Returns `(nodeThres, filamentThres, wallThres)`.
"""
function NeoNEXUS.run(runner::NEXUSDiv, θField::AbstractArray{<:Real,3})
    # Negate θ: -(-k_i k_j) · (-θ̂) = +k_i k_j · θ̂  (matches NEXUS_div formula)
    negθ = .-θField

    gridSize = size(runner.node.significanceMap)
    cache = HessianEigenCache(gridSize...)

    # === Scale Loop ===
    for scale in runner.scales
        R² = scale^2

        # Plain Gaussian for ALL features (no log₁₀) — filter(field, scale) bypasses feature dispatch
        filtered = runner.filter(negθ, scale) .* R²

        computeHessianEigenvalues!(filtered, runner.node.kx, runner.node.ky, runner.node.kz, cache)

        sigMap = NeoNEXUS.computeSignature(runner.node, cache)
        @. runner.node.significanceMap = max(runner.node.significanceMap, sigMap)

        sigMap = NeoNEXUS.computeSignature(runner.filament, cache)
        @. runner.filament.significanceMap = max(runner.filament.significanceMap, sigMap)

        sigMap = NeoNEXUS.computeSignature(runner.wall, cache)
        @. runner.wall.significanceMap = max(runner.wall.significanceMap, sigMap)
    end

    # === Thresholding (hierarchical with signature masking) ===
    # Note: using negθ for density proxy in thresholding
    nodeThres = deltaMSquaredThreshold!(runner.node, negθ)

    maskSignatureMap!(runner.filament, runner.node)
    filamentThres = deltaMSquaredThreshold!(runner.filament, negθ)

    maskSignatureMap!(runner.wall, runner.node)
    maskSignatureMap!(runner.wall, runner.filament)
    wallThres = deltaMSquaredThreshold!(runner.wall, negθ)

    return (nodeThres=nodeThres, filamentThres=filamentThres, wallThres=wallThres)
end


"""
    runMultithreaded(runner::NEXUSDiv, θField) -> NamedTuple

Multithreaded variant of [`run`](@ref) for [`NEXUSDiv`](@ref).

!!! note
    Requires `julia --threads=N`. FFTW internal threading is disabled
    automatically for thread safety.
"""
function NeoNEXUS.runMultithreaded(runner::NEXUSDiv, θField::AbstractArray{<:Real,3})
    FFTW.set_num_threads(1)

    negθ = .-θField

    gridSize = size(runner.node.significanceMap)
    nScales = length(runner.scales)

    nodeSigs = [zeros(Float32, gridSize) for _ in 1:nScales]
    filaSigs = [zeros(Float32, gridSize) for _ in 1:nScales]
    wallSigs = [zeros(Float32, gridSize) for _ in 1:nScales]

    # === Parallel scale loop ===
    Threads.@threads for idx in 1:nScales
        scale = runner.scales[idx]
        R² = scale^2

        filtered = runner.filter(negθ, scale) .* R²
        localCache = computeHessianEigenvalues(filtered, runner.node.kx, runner.node.ky, runner.node.kz)

        nodeSigs[idx] .= NeoNEXUS.computeSignature(runner.node, localCache)
        filaSigs[idx] .= NeoNEXUS.computeSignature(runner.filament, localCache)
        wallSigs[idx] .= NeoNEXUS.computeSignature(runner.wall, localCache)
    end

    for idx in 1:nScales
        @. runner.node.significanceMap = max(runner.node.significanceMap, nodeSigs[idx])
        @. runner.filament.significanceMap = max(runner.filament.significanceMap, filaSigs[idx])
        @. runner.wall.significanceMap = max(runner.wall.significanceMap, wallSigs[idx])
    end

    nodeThres = findComponentPercentageThreshold!(
        runner.node,
        negθ,
        370.0,
        0.50;
        excludeEmpty=false
    )

    maskSignatureMap!(runner.filament, runner.node)
    filamentThres = deltaMSquaredThreshold!(runner.filament, negθ)

    maskSignatureMap!(runner.wall, runner.node)
    maskSignatureMap!(runner.wall, runner.filament)
    wallThres = deltaMSquaredThreshold!(runner.wall, negθ)

    return (nodeThres=nodeThres, filamentThres=filamentThres, wallThres=wallThres)
end
