# API Reference

```@meta
CurrentModule = CosmoNEXUS
```

## Package

```@docs
CosmoNEXUS
```

## Tidal Tensor

```@docs
computeTidalEigenvalues
computeTidalEigenvalues!
```

## Potential Field

```@docs
computePotentialField
computePotentialField!
```

## Shear Tensor

```@docs
computeShearEigenvalues
computeShearEigenvalues!
```

## Pipelines

```@docs
NEXUSTidal
NEXUSPotential
NEXUSVoid
NEXUSDiv
NEXUSShear
```

```@docs
run(::NEXUSTidal, ::AbstractArray{<:Real,3})
run(::NEXUSPotential, ::AbstractArray{<:Real,3})
run(::NEXUSVoid, ::AbstractArray{<:Real,3})
run(::NEXUSDiv, ::AbstractArray{<:Real,3})
run(::NEXUSShear, ::AbstractArray{<:Real,5})
run(::NEXUSShear, ::AbstractArray{<:Real,5}, ::AbstractArray{<:Real,3})
runMultithreaded(::NEXUSTidal, ::AbstractArray{<:Real,3})
runMultithreaded(::NEXUSPotential, ::AbstractArray{<:Real,3})
runMultithreaded(::NEXUSVoid, ::AbstractArray{<:Real,3})
runMultithreaded(::NEXUSDiv, ::AbstractArray{<:Real,3})
runMultithreaded(::NEXUSShear, ::AbstractArray{<:Real,5})
runMultithreaded(::NEXUSShear, ::AbstractArray{<:Real,5}, ::AbstractArray{<:Real,3})
```
