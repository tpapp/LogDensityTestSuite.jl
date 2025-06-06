"""
$(README)
"""
module LogDensityTestSuite

export
    # generic
    samples,
    # primitives
    StandardMultivariateNormal,
    # transformations
    linear, shift, elongate, funnel,
    # mixtures
    mix, directional_weight,
    # diagnostics
    quantile_boundaries, bin_counts, two_sided_pvalues, print_ascii_plot

using ArgCheck: @argcheck
using Compat: @compat
using DocStringExtensions: FIELDS, FUNCTIONNAME, SIGNATURES, TYPEDEF, README
using LinearAlgebra: AbstractTriangular, checksquare, diag, Diagonal, I, dot, logabsdet, lu,
    norm, Symmetric
using LogDensityProblems: LogDensityOrder
import LogDensityProblems: capabilities, dimension, logdensity, logdensity_and_gradient
import Random                   # don't import anything since we only use it in one place
using Sobol: next!, SobolSeq
using StatsFuns: norminvcdf, normcdf
using LogExpFunctions: logaddexp, logistic

include("generic.jl")
include("primitives.jl")
include("transformations.jl")
include("mixtures.jl")

end # module
