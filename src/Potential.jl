"""
    computePotentialField(field, kx, ky, kz) -> Array{Float32,3}

Compute the Poisson potential associated with a scalar source field using
Fourier-space inversion. The zero mode is set to zero, fixing the arbitrary
additive constant of the potential.
"""
function computePotentialField(
    field::AbstractArray{<:Real,3},
    kx, ky, kz
)::Array{Float32,3}
    potentialField = Array{Float32}(undef, size(field))

    computePotentialField!(
        potentialField,
        field,
        collect(Float64, kx),
        collect(Float64, ky),
        collect(Float64, kz)
    )

    return potentialField
end


"""
    computePotentialField!(potentialField, field, kx, ky, kz)

Compute the Poisson potential of `field` in-place, storing the result in
`potentialField`.
"""
function computePotentialField!(
    potentialField::AbstractArray{<:Real,3},
    field::AbstractArray{<:Real,3},
    kx::Vector{Float64}, ky::Vector{Float64}, kz::Vector{Float64}
)
    fftField = FFTW.rfft(field)
    Nx = size(field, 1)
    Nkx, Ny, Nz = size(fftField)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nkx
        k2 = kx[i]^2 + ky[j]^2 + kz[k]^2
        fftField[i, j, k] *= k2 > 0.0 ? 1.0 / k2 : 0.0
    end

    potentialField .= FFTW.irfft(fftField, Nx)

    return potentialField
end
