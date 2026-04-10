"""
    CosmoNEXUS

Cosmology-specific extensions for `NeoNEXUS`.

Provides tidal-tensor ([`NEXUSTidal`](@ref)), velocity-divergence
([`NEXUSDiv`](@ref)), and velocity-shear ([`NEXUSShear`](@ref)) pipelines
while reusing the feature, filter, and thresholding machinery from
`NeoNEXUS`.
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
    HessianEigenCache,
    computeHessianEigenvalues,
    computeHessianEigenvalues!,
    findComponentPercentageThreshold!,
    maskSignatureMap!,
    deltaMSquaredThreshold!

include("Tidal.jl")
include("Shear.jl")
include("Runner.jl")

export
    # tidal
    computeTidalEigenvalues,
    computeTidalEigenvalues!,
    computeShearEigenvalues,
    computeShearEigenvalues!,

    # runners
    NEXUSTidal,
    NEXUSDiv,
    NEXUSShear,
    run,
    runMultithreaded

end
