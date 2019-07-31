module LogDensityTestSuite

export samples, StandardMultivariateNormal

using DocStringExtensions: FUNCTIONNAME, SIGNATURES
using LogDensityProblems: LogDensityOrder
import LogDensityProblems: capabilities, dimension, logdensity, logdensity_and_gradient
using Parameters: @unpack
using Sobol: next!, SobolSeq
using StatsFuns: norminvcdf

####
#### generic
####

"Abstract type for convenient dispatch. Not part of the API."
abstract type SamplingLogDensity end

"""
$(FUNCTIONNAME)(ℓ)

Dimension for the second argument argument [`hypercube_transform`].
"""
function hypercube_dimension end

"""
$(FUNCTIONNAME)(ℓ, x)

Transform ``x ∈ [0,1]ⁿ`` into a random variable that has the given distribution when `x` is
uniform on the given hypercube, where `n` is `hypercube_dimension(ℓ)`.

The result needs to be distinct from `x`, as that is a buffer that may be reused.
"""
function hypercube_transform end

"""
$(SIGNATURES)

`N` samples from `ℓ`, as columns of a matrix.
"""
function samples(ℓ::SamplingLogDensity, N::Integer)
    K = hypercube_dimension(ℓ)
    Z = Matrix{Float64}(undef, K, N)
    s = SobolSeq(K)
    x = Vector{Float64}(undef, K)
    for i in 1:N
        next!(s, x)
        Z[:, i] = hypercube_transform(ℓ, x)
    end
    Z
end

capabilities(::Type{<:SamplingLogDensity}) = LogDensityOrder{1}()

####
#### distributions
####

"Standard multivariate normal with the given dimension `K`."
struct StandardMultivariateNormal <: SamplingLogDensity
    K::Int
end

dimension(ℓ::StandardMultivariateNormal) = ℓ.K

logdensity(ℓ::StandardMultivariateNormal, x) = -0.5 * sum(abs2, x)

logdensity_and_gradient(ℓ::StandardMultivariateNormal, x) = logdensity(ℓ, x), -x

hypercube_dimension(ℓ::StandardMultivariateNormal) = ℓ.K

hypercube_transform(ℓ::StandardMultivariateNormal, x) = norminvcdf.(x)

end # module
