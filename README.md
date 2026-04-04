# CosmoNEXUS

**CosmoNEXUS** is a Julia package that extends **NeoNEXUS** with
cosmology-specific pipelines for tidal tensors, velocity divergence, and
velocity shear. It reuses the same multi-scale morphology framework, feature
definitions, filters, and thresholding utilities while adapting the field
construction to cosmological observables.

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

## Main Components

- `NEXUSTidal` runs the NEXUS_tidal workflow on a density field by replacing the density Hessian with the tidal tensor.
- `NEXUSDiv` runs the NEXUS_div workflow on `thetaField = div(v) / H`.
- `NEXUSShear` runs the NEXUS_shear workflow on either a full velocity-shear tensor or a traceless shear field together with `thetaField`.
- `computeTidalEigenvalues` and `computeTidalEigenvalues!` provide low-level tidal-tensor eigenvalue access.
- `computeShearEigenvalues` and `computeShearEigenvalues!` provide low-level symmetric shear-tensor eigenvalue access.

## Important Notes

- The convenience constructors are for cubic grids.
- `NEXUSTidal` normalizes the input density field to mean density 1 internally.
- `NEXUSDiv` and `NEXUSShear` use a collapse proxy for thresholding, so the sign and normalization convention of `thetaField` matter.
- `NEXUSShear` expects the full tensor when called as `runner(shearField)`. If the available field is traceless, call `runner(tracelessShearField, thetaField)` instead.
- The bundled `demo/exampleShear64.jld2` file is traceless and is paired with `demo/exampleDivergence64.jld2` in `demo/shearComparisonDemo.jl`.
- Feature objects and runners are stateful. Recreate them, or clear their maps manually, before processing a new dataset.

## Repository Demos

- `demo/tidalComparisonDemo.jl` compares `NEXUSPlus` and `NEXUSTidal` on the bundled density cube.
- `demo/divComparisonDemo.jl` compares `NEXUSPlus` and `NEXUSDiv` on the bundled density and divergence cubes.
- `demo/shearComparisonDemo.jl` compares `NEXUSTidal` and `NEXUSShear` using the bundled density cube together with the traceless shear and divergence cubes.

Full API docs are available in `docs/`.
