# CosmoNEXUS.jl

**CosmoNEXUS** extends **NeoNEXUS** with cosmology-specific pipelines for tidal
tensors, velocity divergence, and velocity shear while keeping the same
multi-scale morphology workflow.

## Installation

```julia
using Pkg
Pkg.add("CosmoNEXUS")
```

## Quick Start

```julia
using CosmoNEXUS

N = 64
scales = [3.0, 6.0, 9.0]

density = abs.(randn(Float32, N, N, N)) .+ 1f0
runner = NEXUSTidal(N, scales)
thresholds = runner(density)

println(thresholds)
println(sum(runner.wall.thresholdMap))
```

## What the Package Provides

- **Tidal tensor tools**: [`computeTidalEigenvalues`](@ref) and [`computeTidalEigenvalues!`](@ref) evaluate the tidal-tensor eigenvalues of a scalar field.
- **Shear tensor tools**: [`computeShearEigenvalues`](@ref) and [`computeShearEigenvalues!`](@ref) evaluate the eigenvalues of a symmetric velocity-shear tensor field.
- **Runners**: [`NEXUSTidal`](@ref), [`NEXUSDiv`](@ref), and [`NEXUSShear`](@ref) reuse the `NeoNEXUS` feature, filter, and thresholding stack for cosmological observables.

## Usage Notes

- `NEXUSTidal(gridSize, scales)`, `NEXUSDiv(gridSize, scales)`, and `NEXUSShear(gridSize, scales)` are convenience constructors for cubic grids.
- `NEXUSTidal` normalizes the input density field to mean density 1 internally, matching `NEXUSPlus`.
- `NEXUSDiv` expects `thetaField = div(v) / H`.
- `NEXUSShear` accepts either a full `(Nx, Ny, Nz, 3, 3)` velocity-shear tensor or a traceless shear field together with `thetaField`.
- The bundled `exampleShear64.jld2` demo field is traceless, so it should be paired with `exampleDivergence64.jld2` and passed through the two-field `NEXUSShear` overload.
- Feature objects and runners are stateful: their `significanceMap` and `thresholdMap` arrays live on the structs and are reused across calls.

## Next Steps

- See [API Reference](api.md) for the exported types and functions.
- See the repository demos for end-to-end examples using the bundled density, divergence, and shear cubes.
