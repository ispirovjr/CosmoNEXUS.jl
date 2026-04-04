"""
    computeShearEigenvalues(shearField) -> HessianEigenCache

Compute voxel-wise eigenvalues of a symmetric velocity-shear tensor field stored
as a `(Nx, Ny, Nz, 3, 3)` array. Allocates and returns a new
[`HessianEigenCache`](@ref).

The input is assumed to contain the full velocity-shear tensor. If the
available field is traceless, reconstruct the trace contribution first or use
[`run`](@ref) with [`NEXUSShear`](@ref) and a separate `thetaField`.
"""
function computeShearEigenvalues(
    shearField::AbstractArray{<:Real,5}
)::HessianEigenCache
    gridSize = _shearGridSize(shearField)
    cache = HessianEigenCache(gridSize...)

    computeShearEigenvalues!(shearField, cache)

    return cache
end


"""
    computeShearEigenvalues!(shearField, cache)

Compute eigenvalues of a symmetric velocity-shear tensor field in-place,
storing results in `cache`.
"""
function computeShearEigenvalues!(
    shearField::AbstractArray{<:Real,5},
    cache::HessianEigenCache
)
    gridSize = _shearGridSize(shearField)
    size(cache.λ1) == gridSize || throw(ArgumentError("cache must match the spatial size of shearField"))

    _computeShearEigenvaluesFromComponents!(
        @view(shearField[:, :, :, 1, 1]),
        @view(shearField[:, :, :, 2, 2]),
        @view(shearField[:, :, :, 3, 3]),
        @view(shearField[:, :, :, 1, 2]),
        @view(shearField[:, :, :, 1, 3]),
        @view(shearField[:, :, :, 2, 3]),
        cache
    )

    return cache
end


@inline function _shearGridSize(shearField::AbstractArray{<:Real,5})
    size(shearField, 4) == 3 || throw(ArgumentError("shearField must have size (Nx, Ny, Nz, 3, 3)"))
    size(shearField, 5) == 3 || throw(ArgumentError("shearField must have size (Nx, Ny, Nz, 3, 3)"))
    return (size(shearField, 1), size(shearField, 2), size(shearField, 3))
end


@inline function _computeShearEigenvaluesFromComponents!(
    Sxx::AbstractArray{<:Real,3},
    Syy::AbstractArray{<:Real,3},
    Szz::AbstractArray{<:Real,3},
    Sxy::AbstractArray{<:Real,3},
    Sxz::AbstractArray{<:Real,3},
    Syz::AbstractArray{<:Real,3},
    cache::HessianEigenCache
)
    NeoNEXUS.computeEigenvalues!(Sxx, Syy, Szz, Sxy, Sxz, Syz, cache)
    return cache
end


function _smoothShearComponents(
    filter::AbstractScaleFilter,
    shearField::AbstractArray{<:Real,5},
    scale::Real
)
    _shearGridSize(shearField)
    R2 = scale^2

    return (
        filter(@view(shearField[:, :, :, 1, 1]), scale) .* R2,
        filter(@view(shearField[:, :, :, 2, 2]), scale) .* R2,
        filter(@view(shearField[:, :, :, 3, 3]), scale) .* R2,
        filter(@view(shearField[:, :, :, 1, 2]), scale) .* R2,
        filter(@view(shearField[:, :, :, 1, 3]), scale) .* R2,
        filter(@view(shearField[:, :, :, 2, 3]), scale) .* R2,
    )
end


function _negativeTraceField(shearField::AbstractArray{<:Real,5})
    gridSize = _shearGridSize(shearField)
    negTrace = Array{Float32}(undef, gridSize...)

    @inbounds for k in axes(negTrace, 3), j in axes(negTrace, 2), i in axes(negTrace, 1)
        negTrace[i, j, k] = -Float32(
            shearField[i, j, k, 1, 1] +
            shearField[i, j, k, 2, 2] +
            shearField[i, j, k, 3, 3]
        )
    end

    return negTrace
end


function _reconstructVelocityShearField(
    tracelessShearField::AbstractArray{<:Real,5},
    thetaField::AbstractArray{<:Real,3}
)
    gridSize = _shearGridSize(tracelessShearField)
    size(thetaField) == gridSize || throw(ArgumentError("thetaField must match the spatial size of tracelessShearField"))

    fullShear = Array{Float32}(undef, gridSize..., 3, 3)

    @inbounds for k in 1:gridSize[3], j in 1:gridSize[2], i in 1:gridSize[1]
        traceTerm = Float32(thetaField[i, j, k]) / 3f0

        s11 = Float32(tracelessShearField[i, j, k, 1, 1])
        s22 = Float32(tracelessShearField[i, j, k, 2, 2])
        s33 = Float32(tracelessShearField[i, j, k, 3, 3])
        s12 = Float32(tracelessShearField[i, j, k, 1, 2])
        s13 = Float32(tracelessShearField[i, j, k, 1, 3])
        s23 = Float32(tracelessShearField[i, j, k, 2, 3])

        fullShear[i, j, k, 1, 1] = s11 + traceTerm
        fullShear[i, j, k, 2, 2] = s22 + traceTerm
        fullShear[i, j, k, 3, 3] = s33 + traceTerm

        fullShear[i, j, k, 1, 2] = s12
        fullShear[i, j, k, 2, 1] = s12
        fullShear[i, j, k, 1, 3] = s13
        fullShear[i, j, k, 3, 1] = s13
        fullShear[i, j, k, 2, 3] = s23
        fullShear[i, j, k, 3, 2] = s23
    end

    return fullShear
end
