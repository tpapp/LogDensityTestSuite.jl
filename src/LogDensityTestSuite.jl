module LogDensityTestSuite

export samples, StandardMultivariateNormal

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
    samples(ℓ::SamplingLogDensity, N::Integer)

`N` samples from `ℓ`, as columns of a matrix.
"""
function samples end

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

function samples(ℓ::StandardMultivariateNormal, N::Integer)
    @unpack K = ℓ
    Z = Matrix{Float64}(undef, K, N)
    s = SobolSeq(K)
    x = Vector{Float64}(undef, K)
    for i in 1:N
        next!(s, x)
        Z[:, i] .= norminvcdf.(x)
    end
    Z
end

end # module
