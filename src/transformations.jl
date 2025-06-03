#####
##### transformations
#####

####
#### generic implementation
####

"""
$(TYPEDEF)

Abstract type used internally for simpler code.
"""
struct TransformedLogDensity{L,T} <: SamplingLogDensity
    source::L
    transformation::T
end

function Base.show(io::IO, t::TransformedLogDensity)
    _t = t
    chain = []
    while _t isa TransformedLogDensity
        push!(chain, _t.transformation)
        _t = _t.source
    end
    if length(chain) == 1
        show(io, chain[1])
    else
        print(io, "(")
        for (i, c) in enumerate(chain)
            if i > 1
                print(io, " ∘ ")
            end
            show(io, c)
        end
        print(io, ")")
    end
    print(io, "(", _t, ")")
end

dimension(ℓ::TransformedLogDensity) = dimension(ℓ.source)

"""
$(FUNCTIONNAME)(transformation, x)

Return the transformed `x`. Internal, for implementing transformations.
"""
function source_to_destination end

"""
[`destination_to_source`](@ref) returns `(; x, c)`. Internal, for implementing
transformations.
"""
struct DensityMode end

"""
[`destination_to_source`](@ref) returns `(; x, c, ∂x∂y, ∂c∂y)`. Internal, for implementing
transformations.
"""
struct DensityGradientMode end

"See [`destination_to_source`](@ref)."
const ValidModes = Union{DensityMode,DensityGradientMode}

"""
$(FUNCTIONNAME)(mode::ValidModes, transformation, y)

The inverse of [`source_to_destination`](@ref). `mode` determines what is calculated,
see [`DensityMode`](@ref) and [`DensityGradientMode`](@ref). Implementations can assume
that `mode` is either of these.
"""
function destination_to_source end

function hypercube_transform(ℓ::TransformedLogDensity, u)
    (; source, transformation) = ℓ
    x = hypercube_transform(source, u)
    source_to_destination(transformation, x)
end

function logdensity(ℓ::TransformedLogDensity, y)
    (; source, transformation) = ℓ
    (; x, c) = destination_to_source(DensityMode(), transformation, y)
    logdensity(source, x) - c
end

function logdensity_and_gradient(ℓ::TransformedLogDensity, y)
    (; source, transformation) = ℓ
    (; x, c, ∂x∂y⊤, ∂c∂y) = destination_to_source(DensityGradientMode(), transformation, y)
    ℓx, ∇ℓx = logdensity_and_gradient(source, x)
    ℓy = ℓx - c
    ∇ℓy = ∂x∂y⊤ * ∇ℓx - ∂c∂y
    ℓy, ∇ℓy
end

"Log density transformation. Internal, for organizing code."
abstract type LogDensityTransformation end

function (transformation::LogDensityTransformation)(source)
    TransformedLogDensity(source, transformation)
end

####
#### linear transformation
####

struct Linear{M,S,T} <: LogDensityTransformation
    "the matrix-like object that multiplies coordinates"
    A::M
    "a faster representation with the property `divA \\ x ≈ A \\ x`"
    divA::S
    "``log(abs(det(A)))``"
    logabsdetA::T
end

Base.show(io::IO, transform::Linear) = print(io, "linear(", transform.A, ")")

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
"""
function linear(A::AbstractMatrix)
    checksquare(A)
    Linear(A, _fastdiv(A), first(logabsdet(A)))
end

source_to_destination(transform::Linear, x) = transform.A * x

"""
A lazy inverse wrapper, similar in concept to `LinearAlgebra.Adjoint`.

Only supports `matrix` * vector, the sole purpose is to make it easier for
implementations to return in inverse (ie `∂y∂x` instead of `∂x∂y`) for applying the
chain rule for gradients

Internal, not exported.
"""
struct Inv{M}
    A::M
end

(Base.:*)(B::Inv, v::AbstractVector) = B.A \ v

"""
A placeholder for a vector of zeros, in particular `∂c∂y`. Only supports being subtracted.
"""
struct Zeros end

(Base.:-)(a, ::Zeros) = a

function destination_to_source(mode::ValidModes, transform::Linear, y)
    (; A, divA, logabsdetA) = transform
    x = divA \ y
    c = logabsdetA
    if mode ≡ DensityMode()
        (; x, c)
    else
        (; x, c, ∂x∂y⊤ = Inv(divA'), ∂c∂y = Zeros())
    end
end

####
#### shift (translation)
####

struct Shift{T <: AbstractVector} <: LogDensityTransformation
    b::T
end

Base.show(io::IO, transform::Shift) = print(io, "shift(", transform.b, ")")

"""
$(SIGNATURES)

Transform a distribution on `x` to `y = x + b`, where `b` is a conformable vector.
"""
shift(b::AbstractVector) = Shift(b)

source_to_destination(transform::Shift, x) = x .+ transform.b

function destination_to_source(mode::ValidModes, transform::Shift, y)
    (; b) = transform
    x = y .- b
    c = zero(eltype(x))
    if mode ≡ DensityMode()
        (; x, c)
    else
        (; x, c, ∂x∂y⊤ = I, ∂c∂y = Zeros())
    end
end

####
#### elongate
####

struct Elongate{T <: Real} <: LogDensityTransformation
    k::T
end

Base.show(io::IO, transform::Elongate) = print(io, "elongate(", transform.k, ")")

"""
$(SIGNATURES)

Transform a distribution on `x` to ``y = (1 + ‖x‖²)ᵏ⋅x`` where ``‖ ‖`` is the Euclidean norm
and `k` is a real number. `k > 0` values make the tails heavier.
"""
elongate(k::Real,) = Elongate(k)

function source_to_destination(transformation::Elongate, x)
    (1 + sum(abs2, x))^transformation.k .* x
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

function destination_to_source(mode::ValidModes, transformation::Elongate, y)
    (; k) = transformation
    n = length(y)
    ynorm = norm(y, 2)
    xnorm = _find_x_norm(ynorm, k)
    X = abs2(xnorm)
    D = (1 + X)^(-k)       # x = D ⋅ y
    x = D .* y
    # NOTE derivation for Jacobian:
    # ∂y / ∂x = I(d) * (1 + xnorm2)^k + 2*(x*x')*(k*(1+xnorm2)^(k-1)) =
    #          (1 + xnorm2)^k * (I(d) + (2*k)*(x*x')/(1+xnorm2))
    c = k * n * log1p(X) + log1p(2 * k * X / (1 + X))
    if mode ≡ DensityMode()
        return (; x, c)
    end
    A = 1 + X
    B = 1 + (1 + 2*k) * X
    ∂x∂y⊤ = (I - (2 * k / B) .* Symmetric(x * x')) .* D
    ∂c∂y = (k * (2 + B * n) / (A^(2*k) * B^2) * 2) .* y
    (; x, c, ∂x∂y⊤, ∂c∂y)
end

####
#### funnel
####

struct Funnel <: LogDensityTransformation end

Base.show(io::IO, transform::Funnel) = print(io, "funnel()")

"""
$(SIGNATURES)

Transform the distribution with the mapping `x ↦ y`, such that `y[begin] = x[begin]` and
`y[i] = x[i] exp(x[begin])` for all other indices.
"""
funnel() = Funnel()

"""
A type that helps multiply by `[w, v, …, v]`. Internal.

When not provided, `w = 1` is the default.
"""
struct AfterFirst{T,A<:AbstractUnitRange} <: AbstractVector{T}
    w::T
    v::T
    axis::A
end
AfterFirst(v::T, axis) where T = AfterFirst(one(T), v, axis)
Base.axes(a::AfterFirst) = (a.axis,)
Base.size(a::AfterFirst) = (length(a.axis),)
Base.getindex(a::AfterFirst{T}, i) where T = i == firstindex(a.axis) ? a.w : a.v

function source_to_destination(transformation::Funnel, x)
    map(*, x, AfterFirst(exp(x[begin]), axes(x, 1)))
end

"""
An implementation of the `∂x∂y⊤` partial derivative for `Funnel`. The only method needed
is right multiplication by vector. Internal.
"""
struct Funnel∂X∂Y⊤{T,V<:AbstractVector{T},A} <: AbstractMatrix{T}
    invw::T
    y::V
    axis::A
end

# the AbstractArray API is implemented just for debugging/display purposes, we only need `*`
Base.size(B::Funnel∂X∂Y⊤) = (n = length(B.axis); (n, n))
Base.axes(B::Funnel∂X∂Y⊤) = (B.axis, B.axis)
function Base.getindex(B::Funnel∂X∂Y⊤{T}, i, j) where T
    (; invw, y, axis) = B
    i1 = firstindex(axis)
    @argcheck i ∈ axis && j ∈ axis BoundsError(B, (i, j))
    if i == i1
        if j == i1
            one(T)
        else
            -invw * y[i]
        end
    elseif i == j
        invw
    else
        zero(T)
    end
end

function (Base.:*)(B::Funnel∂X∂Y⊤, v::AbstractVector)
    (; invw, y, axis) = B
    @argcheck axis == axes(v, 1)
    z0 = -dot(@view(y[(begin+1):end]), @view(v[(begin+1):end])) * invw
    map((a, b, c) -> a * b + c, v, AfterFirst(invw, axis), AfterFirst(z0, zero(z0), axis))
end

function destination_to_source(mode::ValidModes, transformation::Funnel, y)
    n = length(y)
    axis = axes(y, 1)
    y1 = y[begin]
    invw = exp(-y1)
    x = map(*, y, AfterFirst(invw, axis))
    c = (n - 1) * y1
    if mode ≡ DensityMode()
        return (; x, c)
    end
    ∂x∂y⊤ = Funnel∂X∂Y⊤(invw, y, axis)
    ∂c∂y = AfterFirst(n - 1, 0, axis)
    (; x, c, ∂x∂y⊤, ∂c∂y)
end
