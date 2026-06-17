"""
    CosmoNEXUS

Cosmology-specific extensions for `NeoNEXUS`.

Provides tidal-tensor ([`NEXUSTidal`](@ref)), velocity-divergence
([`NEXUSDiv`](@ref)), potential-field ([`NEXUSPotential`](@ref)),
potential-void ([`NEXUSVoid`](@ref)), and velocity-shear
([`NEXUSShear`](@ref)) pipelines while reusing the feature, filter, and
thresholding machinery from `NeoNEXUS`.
"""
module CosmoNEXUS

using FFTW
using NeoNEXUS
using Statistics

import NeoNEXUS: run, runMultithreaded

using NeoNEXUS:
    AbstractScaleFilter,
    AbstractMorphologicalFeature,
    GaussianFourierFilter,
    SheetFeature,
    LineFeature,
    NodeFeature,
    Write,
    Read,
    None,
    HessianEigenCache,
    computeHessianEigenvalues,
    computeHessianEigenvalues!,
    findComponentPercentageThreshold!,
    maskSignatureMap!,
    deltaMSquaredThreshold!,
    componentErosionPlateauThreshold!

include("Tidal.jl")
include("Potential.jl")
include("Shear.jl")
include("Runner.jl")

export
    # tidal
    computeTidalEigenvalues,
    computeTidalEigenvalues!,

    # potential
    computePotentialField,
    computePotentialField!,

    # shear
    computeShearEigenvalues,
    computeShearEigenvalues!,

    # runners
    NEXUSTidal,
    NEXUSPotential,
    NEXUSVoid,
    NEXUSDiv,
    NEXUSShear,
    run,
    runMultithreaded

end
