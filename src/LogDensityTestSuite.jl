module LogDensityTestSuite

export samples, StandardMultivariateNormal, linear, shift, mix

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

"""
Abstract type for convenient dispatch.

**Not part of the API**, takes care of the following defaults:

1. `hypercube_dimension` falls back to `dimension`,

2. `logdensity` works through `logdensity_and_gradient` (inefficient, but should not matter)
"""
abstract type SamplingLogDensity end

"""
$(SIGNATURES)

Dimension for the second argument argument [`hypercube_transform`].

Falls back to `dimension` for `SamplingLogDensity`, since only mixtures need to override it.

Mostly for internal use, see [`samples`](@ref).
"""
hypercube_dimension(ℓ::SamplingLogDensity) = dimension(ℓ)

logdensity(ℓ::SamplingLogDensity, x) = first(logdensity_and_gradient(ℓ, x))

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
    Z = Matrix{Float64}(undef, dimension(ℓ), N)
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
function linear(A::AbstractMatrix, ℓ)
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
function shift(b::AbstractVector, ℓ)
    @argcheck length(b) == dimension(ℓ)
    Shift(ℓ, b)
end

dimension(ℓ::Shift) = dimension(ℓ.ℓ)

logdensity(ℓ::Shift, x) = logdensity(ℓ.ℓ, x - ℓ.b)

# The log Jacobian adjustment is zero
logdensity_and_gradient(ℓ::Shift, x) = logdensity_and_gradient(ℓ.ℓ, x - ℓ.b)

hypercube_transform(ℓ::Shift, x) = ℓ.b .+ hypercube_transform(ℓ.ℓ, x)

###
### mixtures of distributions
###

struct Mix{T <: Real,L1,L2} <: SamplingLogDensity
    α::T
    ℓ1::L1
    ℓ2::L2
end

"""
$(SIGNATURES)

A *mixture* of two densities: `ℓ1(x)` with probability `α`, and `ℓ2(x)` with probability
`1-α`, where `α` is a real number between `0` and `1`.
"""
function mix(α, ℓ1, ℓ2)
    @argcheck dimension(ℓ1) == dimension(ℓ2)
    @argcheck 0 ≤ α ≤ 1
    Mix(α, ℓ1, ℓ2)
end

hypercube_dimension(ℓ::Mix) = max(hypercube_dimension(ℓ.ℓ1), hypercube_dimension(ℓ.ℓ2)) + 1

function hypercube_transform(ℓ::Mix, x)
    @unpack α, ℓ1, ℓ2 = ℓ
    if x[end] < α
        hypercube_transform(ℓ1, @view x[1:hypercube_dimension(ℓ1)])
    else
        hypercube_transform(ℓ2, @view x[1:hypercube_dimension(ℓ2)])
    end
end

dimension(ℓ::Mix) = dimension(ℓ.ℓ1)

function logdensity_and_gradient(ℓ::Mix, x)
    @unpack α, ℓ1, ℓ2 = ℓ
    ℓ1x, ∇ℓ1x = logdensity_and_gradient(ℓ1, x)
    ℓ2x, ∇ℓ2x = logdensity_and_gradient(ℓ2, x)
    (isfinite(ℓ1x) && isfinite(ℓ2x)) || return -Inf, nothing
    Δ = ℓ1x - ℓ2x
    if Δ ≥ 0
        αeΔ = α * exp(Δ)
        Omα = 1 - α
        D = αeΔ + Omα
        ℓx = log(D) + ℓ2x
        ∇ℓx =  (αeΔ .* ∇ℓ1x .+ Omα .* ∇ℓ2x) ./ D
    else
        OmαeΔ = (1 - α) * exp(-Δ)
        D = α + OmαeΔ
        ℓx = log(D) + ℓ1x
        ∇ℓx = (α .* ∇ℓ1x .+ OmαeΔ .* ∇ℓ2x) ./ D
    end
    ℓx, ∇ℓx
end

end # module
