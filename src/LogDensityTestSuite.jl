module LogDensityTestSuite

export
    # generic
    samples,
    # primitives
    StandardMultivariateNormal,
    # transformations
    linear, shift,
    # mixtures
    mix

using ArgCheck: @argcheck
using DocStringExtensions: FUNCTIONNAME, SIGNATURES
using LinearAlgebra: checksquare, lu, AbstractTriangular, Diagonal
using LogDensityProblems: LogDensityOrder
import LogDensityProblems: capabilities, dimension, logdensity, logdensity_and_gradient
using Parameters: @unpack
using Sobol: next!, SobolSeq
using StatsFuns: norminvcdf

include("generic.jl")
include("primitives.jl")
include("transformations.jl")
include("mixtures.jl")

end # module
