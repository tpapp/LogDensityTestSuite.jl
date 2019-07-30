# LogDensityTestSuite.jl

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.com/tpapp/LogDensityTestSuite.jl.svg?branch=master)](https://travis-ci.com/tpapp/LogDensityTestSuite.jl)
[![codecov.io](http://codecov.io/github/tpapp/LogDensityTestSuite.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/LogDensityTestSuite.jl?branch=master)

Implementation of log densities as callable object that support

1. the [LogDensityProblems.jl](https://github.com/tpapp/LogDensityProblems.jl) interface with gradients,

2. generating “samples” deterministically using low-discrepancy sequences.

This package was developed mainly for testing [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl/), but other projects may also find it useful.
