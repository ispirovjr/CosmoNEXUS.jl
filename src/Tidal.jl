# Tidal tensor computation via Fourier-space Poisson inversion
# T_ij(k) = -(k_i k_j / k²) ρ̂(k)

# Extract a wave-number component based on dimension (1 = x, 2 = y, 3 = z).
@inline selectK(kVec, i, j, k, dim) = dim == 1 ? kVec[i] : (dim == 2 ? kVec[j] : kVec[k])


# Compute one tidal tensor component in Fourier space: T_ij → -(kα·kβ / k²)·f̂(k).
# The 1/k² converts density to gravitational potential; the negative sign
# preserves eigenvalue sign conventions.
@inline function tidalHessianComp!(tmp, fftField, kα, kβ, dimα::Int, dimβ::Int, kx, ky, kz)
    Nx, Ny, Nz = size(fftField)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        kαVal = selectK(kα, i, j, k, dimα)
        kβVal = selectK(kβ, i, j, k, dimβ)
        k² = kx[i]^2 + ky[j]^2 + kz[k]^2
        invK² = k² > 0.0 ? 1.0 / k² : 0.0
        tmp[i, j, k] = fftField[i, j, k] * (-kαVal * kβVal * invK²)
    end

    return nothing
end


# Compute all six unique tidal tensor components from the FFT of a scalar field.
function computeTidalHessianComponents!(
    fftField, tmp,
    kx, ky, kz,
    kxOdd, kyOdd, kzOdd,
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

    # Off-diagonal components must use odd wavevectors to preserve Hermitian
    # symmetry across the Nyquist frequencies.
    tidalHessianComp!(tmp, fftField, kxOdd, kyOdd, 1, 2, kx, ky, kz)
    Hxy .= FFTW.irfft(tmp, Nx)
    tidalHessianComp!(tmp, fftField, kxOdd, kzOdd, 1, 3, kx, ky, kz)
    Hxz .= FFTW.irfft(tmp, Nx)
    tidalHessianComp!(tmp, fftField, kyOdd, kzOdd, 2, 3, kx, ky, kz)
    Hyz .= FFTW.irfft(tmp, Nx)

    return nothing
end


"""
    computeTidalEigenvalues(field, kx, ky, kz) -> HessianEigenCache

Compute tidal tensor eigenvalues of a scalar field via Fourier-space Poisson
inversion. Allocates and returns a new `HessianEigenCache`.
"""
function computeTidalEigenvalues(
    field::AbstractArray{<:Real,3},
    kx, ky, kz
)::HessianEigenCache
    Nx, Ny, Nz = size(field)
    cache = HessianEigenCache(Nx, Ny, Nz)

    computeTidalEigenvalues!(
        field,
        collect(Float64, kx),
        collect(Float64, ky),
        collect(Float64, kz),
        cache
    )

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
    # Use a real-to-complex FFT to reduce the temporary footprint.
    fftField = FFTW.rfft(field)
    Nx = size(field, 1)
    tmp = similar(fftField)

    Hxx = similar(cache.λ1)
    Hyy = similar(cache.λ1)
    Hzz = similar(cache.λ1)
    Hxy = similar(cache.λ1)
    Hxz = similar(cache.λ1)
    Hyz = similar(cache.λ1)

    # Zero Nyquist modes for cross-derivatives so the inverse FFT stays real.
    Ny, Nz = size(field, 2), size(field, 3)
    kxOdd = copy(kx)
    kyOdd = copy(ky)
    kzOdd = copy(kz)

    if iseven(Nx)
        kxOdd[end] = 0.0
    end
    if iseven(Ny)
        kyOdd[div(Ny, 2)+1] = 0.0
    end
    if iseven(Nz)
        kzOdd[div(Nz, 2)+1] = 0.0
    end

    computeTidalHessianComponents!(
        fftField,
        tmp,
        kx,
        ky,
        kz,
        kxOdd,
        kyOdd,
        kzOdd,
        Hxx,
        Hyy,
        Hzz,
        Hxy,
        Hxz,
        Hyz,
        Nx
    )

    NeoNEXUS.computeEigenvalues!(Hxx, Hyy, Hzz, Hxy, Hxz, Hyz, cache)

    return cache
end
