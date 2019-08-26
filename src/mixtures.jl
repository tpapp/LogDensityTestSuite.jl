#####
##### mixtures of distributions
#####

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

"""
$(SIGNATURES)

Return `α(x)` and `∇α(x)`, where `α` is the mixture weight on the first distribution.
"""
weight_and_gradient(α::Real, x) = α, zero(x)

"""
$(SIGNATURES)

Return `α(x)`, where `α` is the mixture weight on the first distribution. When not defined,
falls back to [`weight_and_gradient`](@ref).
"""
weight(α, x) = first(weight_and_gradient(α, x))

"""
$(SIGNATURES)

A *mixture* of two densities: `ℓ1(x)` with probability `α(x)`, and `ℓ2(x)` with probability
`1-α(x)`. `α` needs to implement [`weight_and_gradient`](@ref).
"""
function mix(α, ℓ1, ℓ2)
    @argcheck dimension(ℓ1) == dimension(ℓ2)
    Mix(α, ℓ1, ℓ2)
end

hypercube_dimension(ℓ::Mix) = max(hypercube_dimension(ℓ.ℓ1), hypercube_dimension(ℓ.ℓ2)) + 1

function hypercube_transform(ℓ::Mix, x)
    @unpack α, ℓ1, ℓ2 = ℓ
    if x[end] < weight(α, x)
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
    αx, ∇αx = weight_and_gradient(α, x)
    ℓx = logaddexp(log(αx) + ℓ1x, log1p(-αx) + ℓ2x)
    f(x, y) = iszero(x) ? zero(x + y) : x * y # avoiding NaN in corner cases
    ∇ℓx = @. f((∇αx + αx * ∇ℓ1x), exp(ℓ1x - ℓx)) + f((-∇αx + (1 - αx) * ∇ℓ2x), exp(ℓ2x - ℓx))
    ℓx, ∇ℓx
end
