#####
##### transformations
#####

"""
$(TYPEDEF)

Abstract type used internally for simplifying forwarding. Assumes a slot `ℓ`, and forwards
`dimension` to it.
"""
abstract type LogDensityTransformation <: SamplingLogDensity end

dimension(ℓ::LogDensityTransformation) = dimension(ℓ.ℓ)

####
#### linear transformation
####

struct Linear{L,M,S,T} <: LogDensityTransformation
    "the parent log density"
    ℓ::L
    "the matrix that multiplies coordinates"
    A::M
    "a form so that `divA \\ x ≈ A \\ x`, yet it is fast"
    divA::S
    "``log(abs(det(A)))``"
    logabsdetA::T
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

Transform a distribution on `x` to `y = Ax`, where `A` is a conformable square matrix or a
`UniformScaling` (eg `I * scalar`).

Since the log Jacobian is constant, it is dropped in the log density.
"""
function linear(A::AbstractMatrix, ℓ)
    K = dimension(ℓ)
    @argcheck checksquare(A) == K
    Linear(ℓ, A, _fastdiv(A), first(logabsdet(A)))
end

# special-cased, not an AbstractMatrix
function linear(A::UniformScaling, ℓ)
    Linear(ℓ, A, A, log(abs(A.λ)) * dimension(ℓ))
end

logdensity(ℓ::Linear, x) = logdensity(ℓ.ℓ, ℓ.divA \ x) - ℓ.logabsdetA

function logdensity_and_gradient(ℓ::Linear, x)
    f, ∇f = logdensity_and_gradient(ℓ.ℓ, ℓ.divA \ x)
    f - ℓ.logabsdetA, (ℓ.divA') \ ∇f # note here \ is needed because / is not defined for LU
end

hypercube_transform(ℓ::Linear, x) = ℓ.A * hypercube_transform(ℓ.ℓ, x)

####
#### shift (translation)
####

struct Shift{L, T <: AbstractVector} <: LogDensityTransformation
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

logdensity(ℓ::Shift, x) = logdensity(ℓ.ℓ, x - ℓ.b)

# The log Jacobian adjustment is zero
logdensity_and_gradient(ℓ::Shift, x) = logdensity_and_gradient(ℓ.ℓ, x - ℓ.b)

hypercube_transform(ℓ::Shift, x) = ℓ.b .+ hypercube_transform(ℓ.ℓ, x)

####
#### elongate
####

struct Elongate{L, T <: Real} <: LogDensityTransformation
    ℓ::L
    k::T
end

"""
$(SIGNATURES)

Transform a distribution on `x` to ``y = (1 + ‖x‖²)ᵏ⋅x`` where ``‖ ‖`` is the Euclidean norm
and `k` is a real number. `k > 0` values make the tails heavier.
"""
function elongate(k::Real, ℓ)
    Elongate(ℓ, k)
end

"""
$(SIGNATURES)

Helper function that solves
```math
y = x ⋅ (1 + x²)ᵏ
```
for `x`, given `y ≥ 0` and `k`, using Newton's method.

Absolute tolerance `atol` should not be set *too* low to avoid occasional numerical cycles
near the root.
"""
function _find_x_norm(y, k; x = zero(y), atol = 16 * eps(y))
    for _ in 1:100
        x2 = abs2(x)
        A = (1 + x2)^k
        f = A * x - y
        abs(f) ≤ atol && return x
        B = 2 * k * x2 * (1 + x2)^(k - 1)
        x = (B * x + y) / (A + B)
    end
    error("internal error: reached maximum number of iterations, y = $(y), k = $(k)")
end

function logdensity_and_gradient(ℓ::Elongate, y)
    @unpack ℓ, k = ℓ
    d = dimension(ℓ)
    ynorm = norm(y, 2)
    xnorm = _find_x_norm(ynorm, k)
    xnorm2 = abs2(xnorm)
    D = (1 + xnorm2)^(-k)       # x = D ⋅ y
    x = D .* y
    ℓx, ∇ℓx = logdensity_and_gradient(ℓ, x)
    ℓy = ℓx - (k*d - 1) * log1p(xnorm2) - log1p((1 + 2*k) * xnorm2)
    A = 1 + xnorm2
    B = 1 + (1 + 2*k) * xnorm2
    L1 = (I - (2 * k / B) .* (x * x')) * ∇ℓx .* D
    L2 = (((k * d - 1)/A + (1 + 2 * k)/B) * 2 * A^(1 - 2*k) / B) .* y
    ℓy, L1 .- L2
end

function hypercube_transform(ℓ::Elongate, z)
    @unpack ℓ, k = ℓ
    x = hypercube_transform(ℓ, z)
    (1 + sum(abs2, x))^k .* x
end
