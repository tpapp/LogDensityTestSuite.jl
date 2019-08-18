module LogDensityTestSuite

export
    # generic
    samples,
    # primitives
    StandardMultivariateNormal,
    # transformations
    linear, shift,
    # mixtures
    mix,
    # diagnostics
    quantile_boundaries, bin_counts, two_sided_pvalues, print_ascii_plot

using ArgCheck: @argcheck
using DocStringExtensions: FIELDS, FUNCTIONNAME, SIGNATURES, TYPEDEF
using LinearAlgebra: checksquare, lu, AbstractTriangular, Diagonal
using LogDensityProblems: LogDensityOrder
import LogDensityProblems: capabilities, dimension, logdensity, logdensity_and_gradient
using MCMCDiagnostics: ess_factor_estimate
using Parameters: @unpack
using Sobol: next!, SobolSeq
using Statistics: quantile
using StatsFuns: norminvcdf, normcdf

include("generic.jl")
include("primitives.jl")
include("transformations.jl")
include("mixtures.jl")
include("diagnostics.jl")

end # module
