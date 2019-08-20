#####
##### mixtures of distributions
#####

"""
$(TYPEDEF)

Mixture of two distributions.

# Fields

$(FIELDS)
"""
struct Mix{T <: Real,L1,L2} <: SamplingLogDensity
    "Mixture weight on `ℓ1`"
    α::T
    "The log density with weight `α`."
    ℓ1::L1
    "The log density with weight `1-α`."
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
    logα = log(α)
    log1mα = log1p(-α)
    y1 = logα + ℓ1x
    y2 = log1mα + ℓ2x
    if y1 > y2
        Δ = y2 - y1
        ∇Δ = ∇ℓ2x - ∇ℓ1x        # NOTE: ∂Δ/∂α term will be added here
        ℓx = y1 + log1p(exp(Δ))
        ∇ℓx = ∇ℓ1x .+ ∇Δ ./ (1 + exp(-Δ))
    else
        Δ = y1 - y2
        ∇Δ = ∇ℓ1x - ∇ℓ2x        # NOTE: see above
        ℓx = y2 + log1p(exp(Δ))
        ∇ℓx = ∇ℓ2x .+ ∇Δ ./ (1 + exp(-Δ))
    end
    ℓx, ∇ℓx
end
