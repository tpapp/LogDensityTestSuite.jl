#####
##### mixtures of distributions
#####

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
