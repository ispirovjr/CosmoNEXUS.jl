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

## Shear Tensor

```@docs
computeShearEigenvalues
computeShearEigenvalues!
```

## Pipelines

```@docs
NEXUSTidal
NEXUSDiv
NEXUSShear
```

```@docs
run(::NEXUSTidal, ::AbstractArray{<:Real,3})
run(::NEXUSDiv, ::AbstractArray{<:Real,3})
run(::NEXUSShear, ::AbstractArray{<:Real,5})
run(::NEXUSShear, ::AbstractArray{<:Real,5}, ::AbstractArray{<:Real,3})
runMultithreaded(::NEXUSTidal, ::AbstractArray{<:Real,3})
runMultithreaded(::NEXUSDiv, ::AbstractArray{<:Real,3})
runMultithreaded(::NEXUSShear, ::AbstractArray{<:Real,5})
runMultithreaded(::NEXUSShear, ::AbstractArray{<:Real,5}, ::AbstractArray{<:Real,3})
```
