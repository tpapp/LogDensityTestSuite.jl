using LogDensityTestSuite, Test, Statistics, LinearAlgebra, Distributions, StatsFuns, FiniteDifferences
using LogDensityProblems: capabilities, dimension, logdensity, logdensity_and_gradient,
    LogDensityOrder
using LogDensityTestSuite: hypercube_dimension, _find_x_norm, _elongate_x_xnorm2_Δℓ_D,
    weight, weight_and_gradient

"""
Test gradient with automatic differentiation.

NOTE: default tolerances are generous because we are using finite differences.
"""
function test_gradient(ℓ, x; atol = √eps(), rtol = 0.01)
    l, g = logdensity_and_gradient(ℓ, x)
    l2 = logdensity(ℓ, x)
    g2 = grad(central_fdm(5, 1), x -> logdensity(ℓ, x), x)[1]
    @test l ≈ l2 atol = atol rtol = rtol
    @test g ≈ g2 atol = atol rtol = rtol
end

####
#### primitives
####

@testset "standard multivariate normal" begin
    K, N = 5, 1000
    ℓ = StandardMultivariateNormal(K)
    @test dimension(ℓ) == hypercube_dimension(ℓ) == K
    @test capabilities(ℓ) == LogDensityOrder(1)
    Z = samples(ℓ, N)
    @test size(Z) == (K, N)
    d = MvNormal(zeros(K), Diagonal(ones(K)))
    for x in eachcol(Z)
        test_gradient(ℓ, x)
        @test logdensity(ℓ, x) ≈ logpdf(d, x)
    end
    @test mean(Z; dims = 2) ≈ zeros(K) atol = 0.01
    @test std(Z; dims = 2) ≈ ones(K) atol = 0.02
end

@testset "random samples" begin
    # NOTE this is a very crude test, basically checking that the `rand` interface hooks
    # into the right calls. everything else is tested using samples.
    μ = [0.5, 0.3]
    L = [0.1 0.3; 0.9 0.7]
    ℓ = shift(μ, linear(L, StandardMultivariateNormal(2)))
    @test mean(rand(ℓ) for _ in 1:10000) ≈ μ atol = 0.05
end

####
#### transformations
####

@testset "multivariate normal using transform" begin
    K = 4
    Q = qr(reshape(range(0.1; step = 0.05, length = K * K), K, K)).Q
    D = Diagonal(range(1; step = .1, length = K))
    μ = collect(range(0.04; step = 0.2, length = K))

    function test_mvnormal(μ, A, Σ)
        K = size(Σ, 1)
        ℓ = shift(μ, linear(A, StandardMultivariateNormal(K)))
        d = MvNormal(μ, Σ)
        @test dimension(ℓ) == hypercube_dimension(ℓ) == K
        @test capabilities(ℓ) == LogDensityOrder(1)
        Z = samples(ℓ, 1000)
        for x in eachcol(Z)
            f, ∇f = logdensity_and_gradient(ℓ, x)
            @test f ≈ logpdf(d, x)
            @test ∇f ≈ gradlogpdf(d, x)
        end
        @test mean(Z; dims = 2) ≈ μ atol = 0.02
        @test norm(cov(Z; dims = 2) .- Σ, Inf) ≤ 0.07
    end

    @testset "MvNormal Diagonal" begin
        A = Diagonal(2 .* ones(K))
        Σ = A * A'
        test_mvnormal(μ, A, Σ)
    end

    @testset "MvNormal Diagonal w/ UniformScaling" begin
        A = I * 0.4
        Σ = (A * A')(K)
        test_mvnormal(μ, A, Σ)
    end

    @testset "MvNormal triangular" begin
        Σ = Symmetric(Q * D * Q')
        A = cholesky(Σ, NoPivot()).L
        test_mvnormal(μ, A, Σ)
    end

    @testset "MvNormal full" begin
        Σ = Symmetric(Q * D * Q')
        A = Q * Diagonal(.√diag(D))
        test_mvnormal(μ, A, Σ)
    end
end

@testset "elongate building blocks" begin
    for _ in 1:1000
        y = abs(randn())
        k = abs(randn() * 3)
        x = _find_x_norm(y, k)
        @test y ≈ x * (1 + abs2(x))^k
    end

    for d in 1:5
        for k in range(0.1, 2, length = 10)
            for _ in 1:10
                y = randn(d)
                x, xnorm2, Δℓ, D = _elongate_x_xnorm2_Δℓ_D(y, k, d)
                @test x .* (1 + dot(x, x))^k ≈ y
                @test xnorm2 ≈ dot(x, x)
                ∂y∂x = jacobian(central_fdm(5, 1), x -> x .* (1 + dot(x, x))^k, x)[1]
                @test logabsdet(∂y∂x)[1] ≈ Δℓ
            end
        end
    end
end

@testset "elongate" begin
    K, N = 5, 1000
    ℓ = elongate(0.5, StandardMultivariateNormal(K))
    @test dimension(ℓ) == hypercube_dimension(ℓ) == K
    @test capabilities(ℓ) == LogDensityOrder(1)
    Z = samples(ℓ, N)
    @test vec(mean(Z; dims = 2)) ≈ zeros(K) norm = x -> norm(x, Inf) atol = 0.02
    for x in eachcol(Z)
        test_gradient(ℓ, x)
    end
end

####
#### mixtures
####

@testset "scalar mixture" begin
    K, N = 5, 1000
    ℓ1 = StandardMultivariateNormal(K)
    μ2 = fill(1.7, K)
    ℓ2 = shift(μ2, linear(Diagonal(fill(0.01, K)), StandardMultivariateNormal(K)))
    for α in [0, √eps(), 0.7, 1-√eps(), 1] # extreme, near-extreme, interior α
        ℓ = mix(α, ℓ1, ℓ2)
        @test dimension(ℓ) == K
        @test hypercube_dimension(ℓ) == K + 1
        @test capabilities(ℓ) == LogDensityOrder(1)
        Z = samples(ℓ, N)
        @test size(Z) == (K, N)

        # test at sample values
        for x in eachcol(Z)
            @test logdensity(ℓ, x) ≈
                logaddexp(log(α) + logdensity(ℓ1, x), log1p(-α) + logdensity(ℓ2, x))
            test_gradient(ℓ, x)
        end

        # test at random extreme values (numerical stability)
        for _ in 1:1000
            x = normalize!(randn(K), 2) .* 100
            @test logdensity(ℓ, x) ≈
                logaddexp(log(α) + logdensity(ℓ1, x), log1p(-α) + logdensity(ℓ2, x))
            test_gradient(ℓ, x)
        end

        # compare means
        @test mean(Z; dims = 2) ≈ (1 - α) .* μ2 atol = 0.02
    end

    @test_throws ArgumentError mix(0.5, ℓ1, StandardMultivariateNormal(K + 1))
end

@testset "directional mixture" begin
    K, N = 5, 1000
    ℓ1 = StandardMultivariateNormal(K)
    μ2 = fill(1.7, K)
    ℓ2 = shift(μ2, linear(Diagonal(fill(0.01, K)), StandardMultivariateNormal(K)))
    α = directional_weight(ones(K))
    ℓ = mix(α, ℓ1, ℓ2)
    @test dimension(ℓ) == K
    @test hypercube_dimension(ℓ) == K + 1
    @test capabilities(ℓ) == LogDensityOrder(1)
    Z = samples(ℓ, N)
    @test size(Z) == (K, N)
    # test at sample values
    for x in eachcol(Z)
        αx, ∇αx = weight_and_gradient(α, x)
        @test ∇αx ≈ grad(central_fdm(5, 1), x -> weight(α, x), x)[1]
        test_gradient(ℓ, x)
    end
    @test_throws ArgumentError directional_weight(zeros(5))
end

####
#### diagnostics
####

@testset "diagnostics" begin
    x = range(0, 1; length = 100001)
    q = quantile_boundaries(x, 10)

    # test p-values and printing
    bc = bin_counts(q, rand(1000))
    bc.bin_counts[1] = 1        # to test printing extremes
    ps = two_sided_pvalues(bc)
    @test all(0 .≤ ps .≤ 1)
    @test print_ascii_plot(String, bc) isa String # very rudimentary
    @info "this is what a printed ascii plot looks like"
    show(stdout, bc)

    # validate p-values
    q̄ = (1:3)./4
    ps = reduce(hcat, [two_sided_pvalues(bin_counts(q, rand(1000))) for _ in 1:1000])
    for i in axes(ps, 1)
        @test quantile(ps[i, :], q̄) ≈ q̄ atol = 0.1 norm = x -> norm(x, Inf)
    end
end
