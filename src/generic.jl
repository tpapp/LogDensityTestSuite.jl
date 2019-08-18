#####
##### generic code
#####

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
