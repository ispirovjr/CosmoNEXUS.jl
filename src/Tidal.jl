# Tidal tensor computation via Fourier-space Poisson inversion
# T_ij(k) = -(k_i k_j / k²) ρ̂(k)

# Local helper: extract k-vector component based on dimension (1=x, 2=y, 3=z)
@inline _selectK(kVec, i, j, k, dim) = dim == 1 ? kVec[i] : (dim == 2 ? kVec[j] : kVec[k])


# Compute one tidal tensor component in Fourier space: T_ij → -(kα·kβ / k²)·f̂(k).
# The 1/k² converts density to gravitational potential; the negative sign
# preserves eigenvalue sign conventions.
@inline function tidalHessianComp!(tmp, fftField, kα, kβ, dimα::Int, dimβ::Int, kx, ky, kz)
    Nx, Ny, Nz = size(fftField)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        kαVal = _selectK(kα, i, j, k, dimα)
        kβVal = _selectK(kβ, i, j, k, dimβ)
        k² = kx[i]^2 + ky[j]^2 + kz[k]^2
        invK² = k² > 0.0 ? 1.0 / k² : 0.0
        tmp[i, j, k] = fftField[i, j, k] * (-kαVal * kβVal * invK²)
    end

    return nothing
end


# Compute all 6 unique tidal tensor components from the FFT of a scalar field.
function computeTidalHessianComponents!(
    fftField, tmp,
    kx, ky, kz,
    kx_odd, ky_odd, kz_odd,
    Hxx::AbstractArray{<:Real,3}, Hyy::AbstractArray{<:Real,3}, Hzz::AbstractArray{<:Real,3},
    Hxy::AbstractArray{<:Real,3}, Hxz::AbstractArray{<:Real,3}, Hyz::AbstractArray{<:Real,3},
    Nx::Int
)
    # Diagonal components (tidal T_xx, T_yy, T_zz)
    tidalHessianComp!(tmp, fftField, kx, kx, 1, 1, kx, ky, kz)
    Hxx .= FFTW.irfft(tmp, Nx)
    tidalHessianComp!(tmp, fftField, ky, ky, 2, 2, kx, ky, kz)
    Hyy .= FFTW.irfft(tmp, Nx)
    tidalHessianComp!(tmp, fftField, kz, kz, 3, 3, kx, ky, kz)
    Hzz .= FFTW.irfft(tmp, Nx)

    # Off-diagonal components (tidal T_xy, T_xz, T_yz)
    # MUST use odd k-vectors to preserve Hermitian symmetry across Nyquist
    tidalHessianComp!(tmp, fftField, kx_odd, ky_odd, 1, 2, kx, ky, kz)
    Hxy .= FFTW.irfft(tmp, Nx)
    tidalHessianComp!(tmp, fftField, kx_odd, kz_odd, 1, 3, kx, ky, kz)
    Hxz .= FFTW.irfft(tmp, Nx)
    tidalHessianComp!(tmp, fftField, ky_odd, kz_odd, 2, 3, kx, ky, kz)
    Hyz .= FFTW.irfft(tmp, Nx)

    return nothing
end


"""
    computeTidalEigenvalues(field, kx, ky, kz) -> HessianEigenCache

Compute tidal tensor eigenvalues of a scalar field via Fourier-space Poisson
inversion. Allocates and returns a new [`HessianEigenCache`](@ref).
"""
function computeTidalEigenvalues(
    field::AbstractArray{<:Real,3},
    kx, ky, kz
)::HessianEigenCache

    Nx, Ny, Nz = size(field)
    cache = HessianEigenCache(Nx, Ny, Nz)

    computeTidalEigenvalues!(field, collect(Float64, kx), collect(Float64, ky), collect(Float64, kz), cache)

    return cache
end


"""
    computeTidalEigenvalues!(field, kx, ky, kz, cache)

Compute tidal tensor eigenvalues of `field` in-place, storing results in `cache`.
"""
function computeTidalEigenvalues!(
    field::AbstractArray{<:Real,3},
    kx::Vector{Float64}, ky::Vector{Float64}, kz::Vector{Float64},
    cache::HessianEigenCache
)
    # Use real-to-Complex FFT for halving memory and compute
    fftField = FFTW.rfft(field)
    Nx = size(field, 1)
    tmp = similar(fftField)

    Hxx = similar(cache.λ1)
    Hyy = similar(cache.λ1)
    Hzz = similar(cache.λ1)
    Hxy = similar(cache.λ1)
    Hxz = similar(cache.λ1)
    Hyz = similar(cache.λ1)

    # Allocate zeroed-Nyquist wavevectors to preserve Hermitian symmetry in cross-derivatives
    Ny, Nz = size(field, 2), size(field, 3)
    kx_odd = copy(kx)
    kx_odd[end] = 0.0  # rfftfreq ends at Nyquist
    ky_odd = copy(ky)
    ky_odd[Ny÷2+1] = 0.0  # fftfreq has Nyquist at N/2 + 1
    kz_odd = copy(kz)
    kz_odd[Nz÷2+1] = 0.0

    # Compute all 6 tidal tensor components in Real Space
    computeTidalHessianComponents!(fftField, tmp, kx, ky, kz, kx_odd, ky_odd, kz_odd, Hxx, Hyy, Hzz, Hxy, Hxz, Hyz, Nx)

    # Compute eigenvalues (λ1, λ2, λ3) for every voxel directly into cache
    NeoNEXUS.computeEigenvalues!(Hxx, Hyy, Hzz, Hxy, Hxz, Hyz, cache)

    return cache
end
