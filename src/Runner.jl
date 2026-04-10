"""
    NEXUSTidal

NEXUS_tidal pipeline (Cautun et al. 2013).

Uses the tidal field tensor (Hessian of gravitational potential) instead of the
density Hessian. The tidal tensor is computed via Fourier-space Poisson inversion:
`T_ij(k) = -(k_i k_j / k²) ρ̂(k)`.
Same filtering and thresholding pipeline as NEXUS+.

# Fields
- `filter::AbstractScaleFilter` - scale-space filter
- `node::NodeFeature` - node detector
- `filament::LineFeature` - filament detector
- `wall::SheetFeature` - wall detector
- `scales::Vector{Float64}` - smoothing scales
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
    (runner::NEXUSTidal)(densityField; multithread = false)

Execute the NEXUS_tidal pipeline.
Depending on `multithread`, dispatches to [`run`](@ref) or
[`runMultithreaded`](@ref).
"""
function (runner::NEXUSTidal)(densityField::AbstractArray{<:Real,3}; multithread::Bool=false)
    if multithread
        return runMultithreaded(runner, densityField)
    else
        return run(runner, densityField)
    end
end

function (runner::NEXUSTidal)(densityField::AbstractArray{<:Real,3}, multithread::Bool)
    return runner(densityField; multithread=multithread)
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

Parallelizes the scale loop using `Threads.@threads`; each scale gets its own
cache and signature arrays. Thresholding is identical to the sequential version.

!!! note
    Requires `julia --threads=N`. FFTW internal threading is disabled
    automatically for thread safety.
"""
function NeoNEXUS.runMultithreaded(runner::NEXUSTidal, densityField::AbstractArray{<:Real,3})
    FFTW.set_num_threads(1)

    meanρ = Statistics.mean(densityField)
    normDensity = densityField ./ meanρ

    gridSize = size(runner.node.significanceMap)
    nScales = length(runner.scales)

    nodeSigs = [zeros(Float32, gridSize) for _ in 1:nScales]
    filaSigs = [zeros(Float32, gridSize) for _ in 1:nScales]
    wallSigs = [zeros(Float32, gridSize) for _ in 1:nScales]

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
    NEXUSDiv

NEXUS_div pipeline - velocity divergence classification.

Uses the velocity divergence field `theta = div v / H` as input instead of
density. Implemented as the standard density Hessian on `-theta`, which
restores the collapsing-region sign convention. Uses plain Gaussian filtering
for all features, since `theta` can be negative.

# Fields
- `filter::AbstractScaleFilter` - scale-space filter
- `node::NodeFeature` - node detector
- `filament::LineFeature` - filament detector
- `wall::SheetFeature` - wall detector
- `scales::Vector{Float64}` - smoothing scales
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
    (runner::NEXUSDiv)(thetaField; multithread = false)

Execute the NEXUS_div pipeline.
Depending on `multithread`, dispatches to [`run`](@ref) or
[`runMultithreaded`](@ref).
"""
function (runner::NEXUSDiv)(thetaField::AbstractArray{<:Real,3}; multithread::Bool=false)
    if multithread
        return runMultithreaded(runner, thetaField)
    else
        return run(runner, thetaField)
    end
end

function (runner::NEXUSDiv)(thetaField::AbstractArray{<:Real,3}, multithread::Bool)
    return runner(thetaField; multithread=multithread)
end


function _thresholdCollapseProxy!(runner, collapseProxyField::AbstractArray{<:Real,3})
    nodeThres = deltaMSquaredThreshold!(runner.node, collapseProxyField)

    maskSignatureMap!(runner.filament, runner.node)
    filamentThres = deltaMSquaredThreshold!(runner.filament, collapseProxyField)

    maskSignatureMap!(runner.wall, runner.node)
    maskSignatureMap!(runner.wall, runner.filament)
    wallThres = deltaMSquaredThreshold!(runner.wall, collapseProxyField)

    return (nodeThres=nodeThres, filamentThres=filamentThres, wallThres=wallThres)
end


"""
    run(runner::NEXUSDiv, thetaField) -> NamedTuple

Execute the NEXUS_div pipeline on a velocity divergence field
`theta = div v / H`.

Thresholding uses the collapse proxy `-theta` for nodes, filaments, and walls.

Returns `(nodeThres, filamentThres, wallThres)`.
"""
function NeoNEXUS.run(runner::NEXUSDiv, thetaField::AbstractArray{<:Real,3})
    negTheta = .-thetaField

    gridSize = size(runner.node.significanceMap)
    cache = HessianEigenCache(gridSize...)

    for scale in runner.scales
        R2 = scale^2
        filtered = runner.filter(negTheta, scale) .* R2
        computeHessianEigenvalues!(filtered, runner.node.kx, runner.node.ky, runner.node.kz, cache)

        sigMap = NeoNEXUS.computeSignature(runner.node, cache)
        @. runner.node.significanceMap = max(runner.node.significanceMap, sigMap)

        sigMap = NeoNEXUS.computeSignature(runner.filament, cache)
        @. runner.filament.significanceMap = max(runner.filament.significanceMap, sigMap)

        sigMap = NeoNEXUS.computeSignature(runner.wall, cache)
        @. runner.wall.significanceMap = max(runner.wall.significanceMap, sigMap)
    end

    return _thresholdCollapseProxy!(runner, negTheta)
end


"""
    runMultithreaded(runner::NEXUSDiv, thetaField) -> NamedTuple

Multithreaded variant of [`run`](@ref) for [`NEXUSDiv`](@ref).

Parallelizes the scale loop using `Threads.@threads`; each scale gets its own
cache and signature arrays. Thresholding is identical to the sequential version.

!!! note
    Requires `julia --threads=N`. FFTW internal threading is disabled
    automatically for thread safety.
"""
function NeoNEXUS.runMultithreaded(runner::NEXUSDiv, thetaField::AbstractArray{<:Real,3})
    FFTW.set_num_threads(1)

    negTheta = .-thetaField

    gridSize = size(runner.node.significanceMap)
    nScales = length(runner.scales)

    nodeSigs = [zeros(Float32, gridSize) for _ in 1:nScales]
    filaSigs = [zeros(Float32, gridSize) for _ in 1:nScales]
    wallSigs = [zeros(Float32, gridSize) for _ in 1:nScales]

    Threads.@threads for idx in 1:nScales
        scale = runner.scales[idx]
        R2 = scale^2

        filtered = runner.filter(negTheta, scale) .* R2
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

    return _thresholdCollapseProxy!(runner, negTheta)
end


"""
    NEXUSShear

NEXUS_shear pipeline for symmetric velocity-shear classification.

Uses a symmetric velocity-shear tensor field as input. At each smoothing scale,
the six unique tensor components are filtered independently, rescaled by `R^2`,
and diagonalized voxel-by-voxel before evaluating the standard NEXUS
morphological signatures.

The full-tensor [`run`](@ref) method expects a field of size
`(Nx, Ny, Nz, 3, 3)` whose trace is the velocity divergence field `theta`.
A convenience overload also accepts a traceless shear field together with
`thetaField` and reconstructs the full tensor internally. The bundled demo data
use this traceless representation.

# Fields
- `filter::AbstractScaleFilter` - scale-space filter
- `node::NodeFeature` - node detector
- `filament::LineFeature` - filament detector
- `wall::SheetFeature` - wall detector
- `scales::Vector{Float64}` - smoothing scales
"""
struct NEXUSShear
    filter::AbstractScaleFilter
    node::NodeFeature
    filament::LineFeature
    wall::SheetFeature
    scales::Vector{Float64}
end


"""
    NEXUSShear(gridSize::Int, scales)

Convenience constructor for a cubic grid of side `gridSize`.
"""
function NEXUSShear(gridSize::Int, scales::Vector{Float64})
    kx = FFTW.rfftfreq(gridSize) .* gridSize .* 2π
    ky = kz = FFTW.fftfreq(gridSize) .* gridSize .* 2π
    sheet = SheetFeature((gridSize, gridSize, gridSize), kx, ky, kz)
    line = LineFeature((gridSize, gridSize, gridSize), kx, ky, kz)
    node = NodeFeature((gridSize, gridSize, gridSize), kx, ky, kz)

    return NEXUSShear(GaussianFourierFilter((gridSize, gridSize, gridSize)), node, line, sheet, scales)
end


function _requireShearRunnerGridSize(runner::NEXUSShear, shearField::AbstractArray{<:Real,5})
    gridSize = _shearGridSize(shearField)
    size(runner.node.significanceMap) == gridSize || throw(ArgumentError("runner grid size must match shearField"))
    return gridSize
end


function _runNEXUSShear(
    runner::NEXUSShear,
    shearField::AbstractArray{<:Real,5},
    negTraceField::AbstractArray{<:Real,3}
)
    gridSize = _requireShearRunnerGridSize(runner, shearField)
    size(negTraceField) == gridSize || throw(ArgumentError("negTraceField must match the spatial size of shearField"))

    cache = HessianEigenCache(gridSize...)

    for scale in runner.scales
        Sxx, Syy, Szz, Sxy, Sxz, Syz = _smoothShearComponents(runner.filter, shearField, scale)
        _computeShearEigenvaluesFromComponents!(Sxx, Syy, Szz, Sxy, Sxz, Syz, cache)

        sigMap = NeoNEXUS.computeSignature(runner.node, cache)
        @. runner.node.significanceMap = max(runner.node.significanceMap, sigMap)

        sigMap = NeoNEXUS.computeSignature(runner.filament, cache)
        @. runner.filament.significanceMap = max(runner.filament.significanceMap, sigMap)

        sigMap = NeoNEXUS.computeSignature(runner.wall, cache)
        @. runner.wall.significanceMap = max(runner.wall.significanceMap, sigMap)
    end

    return _thresholdCollapseProxy!(runner, negTraceField)
end


function _runNEXUSShearMultithreaded(
    runner::NEXUSShear,
    shearField::AbstractArray{<:Real,5},
    negTraceField::AbstractArray{<:Real,3}
)
    FFTW.set_num_threads(1)

    gridSize = _requireShearRunnerGridSize(runner, shearField)
    size(negTraceField) == gridSize || throw(ArgumentError("negTraceField must match the spatial size of shearField"))
    nScales = length(runner.scales)

    nodeSigs = [zeros(Float32, gridSize) for _ in 1:nScales]
    filaSigs = [zeros(Float32, gridSize) for _ in 1:nScales]
    wallSigs = [zeros(Float32, gridSize) for _ in 1:nScales]

    Threads.@threads for idx in 1:nScales
        scale = runner.scales[idx]
        Sxx, Syy, Szz, Sxy, Sxz, Syz = _smoothShearComponents(runner.filter, shearField, scale)

        localCache = HessianEigenCache(gridSize...)
        _computeShearEigenvaluesFromComponents!(Sxx, Syy, Szz, Sxy, Sxz, Syz, localCache)

        nodeSigs[idx] .= NeoNEXUS.computeSignature(runner.node, localCache)
        filaSigs[idx] .= NeoNEXUS.computeSignature(runner.filament, localCache)
        wallSigs[idx] .= NeoNEXUS.computeSignature(runner.wall, localCache)
    end

    for idx in 1:nScales
        @. runner.node.significanceMap = max(runner.node.significanceMap, nodeSigs[idx])
        @. runner.filament.significanceMap = max(runner.filament.significanceMap, filaSigs[idx])
        @. runner.wall.significanceMap = max(runner.wall.significanceMap, wallSigs[idx])
    end

    return _thresholdCollapseProxy!(runner, negTraceField)
end


"""
    (runner::NEXUSShear)(shearField; multithread = false)

Execute the NEXUS_shear pipeline on a full symmetric tensor field.
Depending on `multithread`, dispatches to [`run`](@ref) or
[`runMultithreaded`](@ref).
"""
function (runner::NEXUSShear)(shearField::AbstractArray{<:Real,5}; multithread::Bool=false)
    if multithread
        return runMultithreaded(runner, shearField)
    else
        return run(runner, shearField)
    end
end

function (runner::NEXUSShear)(shearField::AbstractArray{<:Real,5}, multithread::Bool)
    return runner(shearField; multithread=multithread)
end


"""
    (runner::NEXUSShear)(tracelessShearField, divField; multithread = false)

Execute the NEXUS_shear pipeline on a traceless shear field together with a
separate divergence field `divField`.
"""
function (runner::NEXUSShear)(
    tracelessShearField::AbstractArray{<:Real,5},
    divField::AbstractArray{<:Real,3},
    ;
    multithread::Bool=false
)
    if multithread
        return runMultithreaded(runner, tracelessShearField, divField)
    else
        return run(runner, tracelessShearField, divField)
    end
end

function (runner::NEXUSShear)(
    tracelessShearField::AbstractArray{<:Real,5},
    divField::AbstractArray{<:Real,3},
    multithread::Bool
)
    return runner(tracelessShearField, divField; multithread=multithread)
end


"""
    run(runner::NEXUSShear, shearField) -> NamedTuple

Execute the NEXUS_shear pipeline on a full symmetric velocity-shear tensor field
stored as `(Nx, Ny, Nz, 3, 3)`.

Thresholding uses `-tr(shearField)` as the collapse proxy, matching the `-divField`
proxy used by [`NEXUSDiv`](@ref).

Use [`run(runner::NEXUSShear, tracelessShearField, divField)`](@ref) when the
available shear field is traceless.

Returns `(nodeThres, filamentThres, wallThres)`.
"""
function NeoNEXUS.run(runner::NEXUSShear, shearField::AbstractArray{<:Real,5})
    return _runNEXUSShear(runner, shearField, _negativeTraceField(shearField))
end


"""
    run(runner::NEXUSShear, tracelessShearField, divField) -> NamedTuple

Convenience overload that reconstructs the full symmetric velocity-shear tensor
from a traceless shear field and a divergence field `divField`.
"""
function NeoNEXUS.run(
    runner::NEXUSShear,
    tracelessShearField::AbstractArray{<:Real,5},
    divField::AbstractArray{<:Real,3}
)
    fullShear = _reconstructVelocityShearField(tracelessShearField, divField)
    return _runNEXUSShear(runner, fullShear, .-divField)
end


"""
    runMultithreaded(runner::NEXUSShear, shearField) -> NamedTuple

Multithreaded variant of [`run`](@ref) for [`NEXUSShear`](@ref).

Parallelizes the scale loop using `Threads.@threads`; each scale gets its own
temporary component arrays and eigenvalue cache. Thresholding is identical to
the sequential version.

!!! note
    Requires `julia --threads=N`. FFTW internal threading is disabled
    automatically for thread safety.
"""
function NeoNEXUS.runMultithreaded(runner::NEXUSShear, shearField::AbstractArray{<:Real,5})
    return _runNEXUSShearMultithreaded(runner, shearField, _negativeTraceField(shearField))
end


"""
    runMultithreaded(runner::NEXUSShear, tracelessShearField, divField) -> NamedTuple

Multithreaded convenience overload for traceless shear plus `divField`.
"""
function NeoNEXUS.runMultithreaded(
    runner::NEXUSShear,
    tracelessShearField::AbstractArray{<:Real,5},
    divField::AbstractArray{<:Real,3}
)
    fullShear = _reconstructVelocityShearField(tracelessShearField, divField)
    return _runNEXUSShearMultithreaded(runner, fullShear, .-divField)
end
