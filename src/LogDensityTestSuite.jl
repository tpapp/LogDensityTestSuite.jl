module LogDensityTestSuite

export samples, StandardMultivariateNormal, linear, shift

using ArgCheck: @argcheck
using DocStringExtensions: FUNCTIONNAME, SIGNATURES
using LinearAlgebra: checksquare, lu, AbstractTriangular, Diagonal
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
$(SIGNATURES)

Dimension for the second argument argument [`hypercube_transform`].

Falls back to `dimension` for `SamplingLogDensity`, since only mixtures need to override it.

Mostly for internal use, see [`samples`](@ref).
"""
hypercube_dimension(ℓ::SamplingLogDensity) = dimension(ℓ)

"""
$(FUNCTIONNAME)(ℓ, x)

Transform ``x ∈ [0,1]ⁿ`` into a random variable that has the given distribution when `x` is
uniform on the given hypercube, where `n` is `hypercube_dimension(ℓ)`.

The result needs to be distinct from `x`, as that is a buffer that may be reused.

Mostly for internal use, see [`samples`](@ref).
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
#### primitive distributions
####

"Standard multivariate normal with the given dimension `K`."
struct StandardMultivariateNormal <: SamplingLogDensity
    K::Int
end

dimension(ℓ::StandardMultivariateNormal) = ℓ.K

logdensity(ℓ::StandardMultivariateNormal, x) = -0.5 * sum(abs2, x)

logdensity_and_gradient(ℓ::StandardMultivariateNormal, x) = logdensity(ℓ, x), -x

hypercube_transform(ℓ::StandardMultivariateNormal, x) = norminvcdf.(x)

####
#### transformations
####

###
### linear transformation
###

struct Linear{L,T,S} <: SamplingLogDensity
    ℓ::L
    A::T
    divA::S
end

"""
$(SIGNATURES)

Internal method for returning an object that makes `A \\ x` fast and stable.
"""
_fastdiv(A::AbstractMatrix) = lu(A)
_fastdiv(A::AbstractTriangular) = A
_fastdiv(A::Diagonal) = A

"""
$(SIGNATURES)

Transform a distribution on `x` to `y = Ax`, where `A` is a conformable square matrix.

Since the log Jacobian is constant, it is dropped in the log density.
"""
function linear(ℓ, A::AbstractMatrix)
    K = dimension(ℓ)
    @argcheck checksquare(A) == K
    Linear(ℓ, A, _fastdiv(A))
end

dimension(ℓ::Linear) = dimension(ℓ.ℓ)

logdensity(ℓ::Linear, x) = logdensity(ℓ.ℓ, ℓ.divA \ x)

function logdensity_and_gradient(ℓ::Linear, x)
    f, ∇f = logdensity_and_gradient(ℓ.ℓ, ℓ.divA \ x)
    f, (ℓ.divA') \ ∇f
end

hypercube_transform(ℓ::Linear, x) = ℓ.A * hypercube_transform(ℓ.ℓ, x)

###
### shift (translation)
###

struct Shift{L, T <: AbstractVector} <: SamplingLogDensity
    ℓ::L
    b::T
end

"""
$(SIGNATURES)

Transform a distribution on `x` to `y = x + b`, where `b` is a conformable vector.
"""
function shift(ℓ, b::AbstractVector)
    @argcheck length(b) == dimension(ℓ)
    Shift(ℓ, b)
end

dimension(ℓ::Shift) = dimension(ℓ.ℓ)

logdensity(ℓ::Shift, x) = logdensity(ℓ.ℓ, x - ℓ.b)

# The log Jacobian adjustment is zero
logdensity_and_gradient(ℓ::Shift, x) = logdensity_and_gradient(ℓ.ℓ, x - ℓ.b)

hypercube_transform(ℓ::Shift, x) = ℓ.b .+ hypercube_transform(ℓ.ℓ, x)

end # module
