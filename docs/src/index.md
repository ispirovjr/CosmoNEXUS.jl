# CosmoNEXUS.jl

**CosmoNEXUS** is a cosmology-specific extension package for [NeoNEXUS.jl](https://github.com/ispirovjr/NeoNEXUS.jl). It provides tidal tensor and velocity divergence classification pipelines for cosmic web analysis.

## Installation

```julia
using Pkg
Pkg.add("CosmoNEXUS")
```

## Overview

CosmoNEXUS extends the NeoNEXUS morphological analysis framework with two cosmological pipelines:

- **[`NEXUSTidal`](@ref)**: Classifies structures using the tidal field tensor (Hessian of gravitational potential) via Fourier-space Poisson inversion.
- **[`NEXUSDiv`](@ref)**: Classifies structures using the velocity divergence field `θ = ∇·v/H`.

Both pipelines follow the same signature computation and hierarchical thresholding approach as NEXUS+, but operate on different physical fields.

## Quick Start

```julia
using NeoNEXUS, CosmoNEXUS

N = 64
scales = [1.0, 2.0, 4.0]

runner = NEXUSTidal(N, scales)
density = abs.(randn(Float32, N, N, N))

thresholds = run(runner, density)
```

See [NeoNEXUS.jl documentation](https://ispirovjr.github.io/NeoNEXUS.jl) for the core API.
