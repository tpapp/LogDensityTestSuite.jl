#####
##### diagnostics
#####

"""
$(TYPEDEF)

# Fields

$(FIELDS)
"""
struct UnivariateQuantileBoundaries{T <: AbstractVector}
    "Univariate (interior, equispaced) quantile boundaries."
    boundaries::T
end

"""
$(SIGNATURES)

Quantiles ``1/K, …, (K-1)/K`` of real numbers `xs`, as a `UnivariateQuantileBoundaries`
object.
"""
function quantile_boundaries(xs, K)
    UnivariateQuantileBoundaries(quantile(xs, (1:(K - 1)) ./ K))
end

"""
$(TYPEDEF)

Summary statistics from univariate bin counts.

# Fields

$(FIELDS)
"""
struct UnivariateBinCounts
    "Number of samples in chain."
    N::Int
    "Effective sample size correction factor."
    τ::Float64
    "Bin counts."
    bin_counts::Vector{Int}
end

"""
$(SIGNATURES)

Calculate univariate bin counts of `xs` (a vector of real numbers), using the given quantile
boundaries. Effective sample size is saved.
"""
function bin_counts(uqb::UnivariateQuantileBoundaries, xs)
    @unpack boundaries = uqb
    bin_counts = zeros(1 + length(boundaries))
    for x in xs
        bin_counts[searchsortedfirst(boundaries, x)] += 1
    end
    τ = first(ess_factor_estimate(xs))
    UnivariateBinCounts(length(xs), τ, bin_counts)
end

const ESS_CORRECTION_DOC = """
When `ess_correction = true` (the default), the standard deviation is calculated using the
the effective sample size.
"""

"""
$(SIGNATURES)

Normal approximation for bin count distributions. Return a `NamedTuple` of `μ` (the mean)
and `σ` (the standard deviation).

$(ESS_CORRECTION_DOC)
"""
function _normal_approximation(ubc::UnivariateBinCounts; ess_correction::Bool = true)
    @unpack N, τ, bin_counts = ubc
    π = 1 / length(bin_counts)
    μ = N * π
    σ = √(N / (ess_correction ? τ : one(τ)) * π * (1 - π))
    (μ = μ, σ = σ)
end

"""
$(SIGNATURES)

Calculate two-sided p-values for bin counts.

Uses a normal approxiation with continuity correction.

$(ESS_CORRECTION_DOC)
"""
function two_sided_pvalues(ubc::UnivariateBinCounts; ess_correction::Bool = true)
    @unpack μ, σ = _normal_approximation(ubc; ess_correction = ess_correction)
    @unpack bin_counts = ubc
    # use a normal approximation for p-values
    qs = normcdf.((bin_counts .+ 0.5 .- μ) ./ σ)
    map(q -> (q > 0.5 ? 1 - q : q) * 2, qs)
end

####
#### ASCII printing
####

"""
$(SIGNATURES)

Helper function for making sure everything fits into the plot area nicely.
"""
function _inflate_extrema(left, right, padding)
    @argcheck left ≤ right
    @argcheck padding ≥ 0
    Δ = right - left
    (left - Δ * padding, right + Δ * padding)
end

"""
$(SIGNATURES)

Print an ASCII plot to `io` with statistics from the given univariate bin counts. When the
first argument is a `String`, return the output as a string instead of printing.

# Keyword arguments

- `canvas_width`: width of the canvas for the plot
- `p_colname`: column name for p-values (the actual value printed is `-log10(p)`)
- `bin_colname`: column name for bin indices
- `count_colname`: column name for counts
- `α`: two markers (with `(` and `)` are placed at the quantiles `α` and `1-α`
- `padding`: enlargement factor for canvas

$(ESS_CORRECTION_DOC)
"""
function print_ascii_plot(io::IO, ubc::UnivariateBinCounts; canvas_width = 80,
                          p_colname = "-log10(p)", bin_colname = "bin", α = 0.05,
                          count_colname = "count", padding = 0.05, ess_correction = true)
    @unpack μ, σ = _normal_approximation(ubc)
    @unpack N, bin_counts = ubc
    @argcheck 0 < α < 0.5
    @argcheck canvas_width ≥ 10
    z = norminvcdf(α / 2)
    N1, N2 = μ .+ σ .* (z, -z)  # boundary lines
    Nl, Nr = _inflate_extrema(extrema(vcat(bin_counts, [N1, N2]))..., padding)
    # linear map from coordinates to canvas
    b = (canvas_width - 1) / (Nr - Nl)
    a = 0.5 - b * Nl
    f(n) = clamp(round(Int, a + n * b), 1, canvas_width)
    # draw
    K = length(bin_counts)
    canvas = fill(' ', K, canvas_width) # empty canvas
    canvas[:, f(N1)] .= '('
    canvas[:, f(N2)] .= ')'
    canvas[:, f(μ)] .= '|'
    for (i, n) in enumerate(bin_counts)
        canvas[i, f(n)] = '#'
    end
    ps = two_sided_pvalues(ubc; ess_correction = ess_correction)
    # print everything
    p_pad = max(5, length(p_colname))
    b_pad = max(length(string(K)), length(bin_colname))
    c_pad = max(length(string(maximum(bin_counts))), length(count_colname))
    canvas_label = " counts with boundaries at p-value = $(α) "
    canvas_label = lpad(canvas_label * '-'^((canvas_width - length(canvas_label) - 2) ÷ 2),
                        canvas_width - 2, '-')
    println(io, rpad(p_colname, p_pad, ' '), ' ', rpad(bin_colname, b_pad, ' '), ' ',
            rpad(count_colname, c_pad, ' '), ' ', " (", canvas_label, ")")
    for i in 1:K
        p_string = @sprintf "%.1f" -log10(ps[i])
        if length(p_string) > p_pad # should never happen
            p_string = "LARGE"
        end
        print(io, lpad(p_string, p_pad, ' '), ' ', lpad(string(i), b_pad, ' '), ' ',
              lpad(string(bin_counts[i]), c_pad), ' ')
        for c in canvas[i, :]
            print(io, c)
        end
        println(io)
    end
    nothing
end

function print_ascii_plot(::Type{String}, ubc::UnivariateBinCounts; kwargs...)
    io = IOBuffer()
    print_ascii_plot(io, ubc; kwargs...)
    String(take!(io))
end

Base.show(io::IO, ubc::UnivariateBinCounts) = print_ascii_plot(io, ubc)
