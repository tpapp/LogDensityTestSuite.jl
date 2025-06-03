# LogDensityTestSuite.jl

![lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
[![build](https://github.com/tpapp/LogDensityTestSuite.jl/workflows/CI/badge.svg)](https://github.com/tpapp/LogDensityTestSuite.jl/actions?query=workflow%3ACI)
[![codecov.io](http://codecov.io/github/tpapp/LogDensityTestSuite.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/LogDensityTestSuite.jl?branch=master)
[![DOI](https://zenodo.org/badge/199613252.svg)](https://zenodo.org/badge/latestdoi/199613252)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://tpapp.github.io/LogDensityTestSuite.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://tpapp.github.io/LogDensityTestSuite.jl/dev)

Construct distributions that support

1. the [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) interface (with gradients),

2. generating “samples” deterministically using low-discrepancy sequences.

Gradient calculations use optimized closed formss instead automatic differentiation.

This package was developed mainly for testing [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl/), but other projects may also find it useful.

# Acknowledgements

The development of this package was partially supported by the Austrian National Bank Jubileumsfonds Projekt 18847.
