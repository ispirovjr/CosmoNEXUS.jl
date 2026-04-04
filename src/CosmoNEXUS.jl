"""
    CosmoNEXUS

Cosmology-specific extensions for [`NeoNEXUS`](@ref).

Provides tidal tensor classification ([`NEXUSTidal`](@ref)) and velocity
divergence classification ([`NEXUSDiv`](@ref)) pipelines for cosmic web analysis.
"""
module CosmoNEXUS

using NeoNEXUS
using FFTW
using Statistics

# Import functions we extend with new methods
import NeoNEXUS: run, runMultithreaded

# Re-import types used in our signatures (for user convenience)
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
include("Runners.jl")

export
    # tidal
    computeTidalEigenvalues,
    computeTidalEigenvalues!,

    # runners
    NEXUSTidal,
    NEXUSDiv,
    run,
    runMultithreaded

end
