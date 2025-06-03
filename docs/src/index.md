# LogDensityTestSuite

```@docs
LogDensityTestSuite.LogDensityTestSuite
```

## Overview

```@setup all
using ..LogDensityTestSuite
using Miter, Contour, Statistics, LinearAlgebra
using LogDensityProblems: logdensity
using Unitful.DefaultSymbols

const Q = range(0.1, 0.9, length = 9)
const II = range(-2, 2; length = 100)

# a contour plot of HPD quantiles (default Q)
function contour_plot(d, x_range = II, y_range = x_range;
                      q = Q, N_samples = 5000, max_width = 0.5mm)
    s = samples(d, N_samples)
    l = map(x -> logdensity(d, x), eachcol(s))
    c = quantile(l, q)          # levels
    z = [logdensity(d, [x, y]) for x in x_range, y in y_range]
    C = contours(x_range, y_range, z, c)
    L = mapreduce(vcat, zip(q, levels(C))) do (q, l)
        w = max_width * q
        [Lines(vertices(l); line_width = w) for l in lines(l)]
    end
    Plot(vcat(L, Invisible(Miter.bounds_xy(collect(zip(x_range, y_range)))...)))
end
```

This package is helpful in defining a family of distributions defined on $\mathbb{R}^n$, each with the following properties:

1. *samples* can be drawn from the distribution using quasi-random methods (low-discrepancy [Sobol sequences](https://en.wikipedia.org/wiki/Sobol_sequence)),

2. the log density and its gradient can be calculated for each coordinate efficiently and accurately, without relying on automatic differentiation (using closed forms), implemented using the [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) interface.

This is achieved by constructing distributions using simple transformations that preserve these properties.

```@example all
d = (funnel() ∘ shift([-1.0, 2.0]))(StandardMultivariateNormal(2))
samples(d, 10)
```

```@docs
samples
```

## Constructing distributions

### Primitives

All distributions are constructed from *primitives*. For practical purposes, these are isotropic extensions of univariate distributions with known density (& its derivative) and inverse quantile functions. Currently the package contains only one primitive, from the standard normal distribution, but similar distributions would be easy to define. However, it is preferred to construct more complex distributions using [transformations](@ref).

```@docs
StandardMultivariateNormal
```

We introduce the convention used for illustrating two-dimensional distributions below: lines increasing thickness are contour plots of the density, containing the 90%, ..., 10% highest posterior density regions:

```@example all
const N2 = StandardMultivariateNormal(2)
contour_plot(N2) #hide
```

## [Transformations](@id transformations)

```@docs
linear
```

```@example all
A = [0.5 0.8;
     0.0 0.7]
D =  linear(A)(N2)
contour_plot(D) # hide
```

```@docs
shift
```

```@example all
b = [1.0, 0.0]
D = (shift(b) ∘ linear(A))(N2)
contour_plot(D) # hide
```

```@docs
elongate
```

```@example all
D = (shift(-2.2*b) ∘ elongate(1.2) ∘ shift(b) ∘ linear(I(2) ./ 4))(N2)
contour_plot(D) # hide
```

```@docs
funnel
```

```@example all
D = funnel()(N2)
contour_plot(D) # hide
```

## Mixtures

```@docs
mix
LogDensityTestSuite.weight_and_gradient
directional_weight
```

Example with constant weight:

```@example all
D = mix(0.5, 
        (shift([-0.7, 0.3]) ∘ linear(I(2) ./ 1.5))(N2),
        (shift([0.7, 0.7]) ∘ linear(I(2) ./ 3))(N2))
contour_plot(D) # hide        
```


```@example all
D = mix(directional_weight([1,1]), 
        (shift([-0.7, 0.3]) ∘ linear(I(2) ./ 1.5))(N2),
        (shift([0.7, 0.7]) ∘ linear(I(2) ./ 3))(N2))
contour_plot(D) # hide        
```

New kinds of mixtures that depend on coordinates can be created by implementing the methods below.

```@doc
LogDensityTestSuite.weight
LogDensityTestSuite.weight_and_gradient
```
