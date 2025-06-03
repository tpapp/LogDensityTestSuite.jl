#####
##### mixtures of distributions
#####

@compat public weight, weight_and_gradient

"""
$(TYPEDEF)

Mixture of two distributions.

# Fields

$(FIELDS)
"""
struct Mix{T,L1,L2} <: SamplingLogDensity
    "Mixture weight on `ℓ1` (an object that supports [`weight_and_gradient`](@ref)."
    α::T
    "The log density with weight `α`."
    ℓ1::L1
    "The log density with weight `1-α`."
    ℓ2::L2
end

function Base.show(io::IO, mix::Mix)
    (; α, ℓ1, ℓ2) = mix
    print(io, "mix(", α, ", ", ℓ1, ", ", ℓ2, ")")
end

"""
$(SIGNATURES)

Return `α(x)` and `∇α(x)`, where `α` is the mixture weight on the first distribution.

Public, but not exported.
"""
weight_and_gradient(α::Real, x) = α, zero(x)

"""
$(SIGNATURES)

Return `α(x)`, where `α` is the mixture weight on the first distribution. When not defined,
falls back to [`weight_and_gradient`](@ref).

Public, but not exported.
"""
weight(α, x) = first(weight_and_gradient(α, x))

"""
$(SIGNATURES)

A *mixture* of two densities: `ℓ1(x)` with probability `α(x)`, and `ℓ2(x)` with probability
`1-α(x)`. `α` needs to implement [`weight_and_gradient`](@ref).
"""
function mix(α, ℓ1, ℓ2)
    @argcheck(dimension(ℓ1) == dimension(ℓ2),
              DimensionMismatch("can only mix distributions with the same dimension"))
    Mix(α, ℓ1, ℓ2)
end

hypercube_dimension(ℓ::Mix) = max(hypercube_dimension(ℓ.ℓ1), hypercube_dimension(ℓ.ℓ2)) + 1

function hypercube_transform(ℓ::Mix, x)
    (; α, ℓ1, ℓ2) = ℓ
    if x[end] < weight(α, x[1:dimension(ℓ)])
        hypercube_transform(ℓ1, @view x[1:hypercube_dimension(ℓ1)])
    else
        hypercube_transform(ℓ2, @view x[1:hypercube_dimension(ℓ2)])
    end
end

dimension(ℓ::Mix) = dimension(ℓ.ℓ1)

function logdensity_and_gradient(ℓ::Mix, x)
    (; α, ℓ1, ℓ2) = ℓ
    ℓ1x, ∇ℓ1x = logdensity_and_gradient(ℓ1, x)
    ℓ2x, ∇ℓ2x = logdensity_and_gradient(ℓ2, x)
    αx, ∇αx = weight_and_gradient(α, x)
    ℓx = logaddexp(log(αx) + ℓ1x, log1p(-αx) + ℓ2x)
    f(x, y) = iszero(x) ? zero(x + y) : x * y # avoiding NaN in corner cases
    ∇ℓx = @. f((∇αx + αx * ∇ℓ1x), exp(ℓ1x - ℓx)) + f((-∇αx + (1 - αx) * ∇ℓ2x), exp(ℓ2x - ℓx))
    ℓx, ∇ℓx
end

struct DirectionalWeight{V}
    "The direction vector."
    y::V
end

Base.show(io::IO, α::DirectionalWeight) = print(io, "directional_weight(", α.y, ")")

"""
$(SIGNATURES)

Represents the weight `α(x) = logistic(dot(x, y))`. The Euclidean norm of `y` is related to
the slope of the gradient of `α` in the given direction.
"""
function directional_weight(y)
    @argcheck norm(y, 2) > 0
    DirectionalWeight(y)
end

function weight_and_gradient(α::DirectionalWeight, x)
    (; y) = α
    αx = logistic(dot(x, y))
    ∇αx = (αx * (1 - αx)) .* y
    αx, ∇αx
end
