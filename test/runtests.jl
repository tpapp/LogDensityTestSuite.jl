using LogDensityTestSuite, Test, Statistics, LinearAlgebra, Distributions
import ForwardDiff
using LogDensityProblems: capabilities, dimension, logdensity, logdensity_and_gradient,
    LogDensityOrder
using LogDensityTestSuite: hypercube_dimension

"Test gradient with automatic differentiation."
function test_gradient(ℓ, x; atol = √eps())
    l, g = logdensity_and_gradient(ℓ, x)
    l2 = logdensity(ℓ, x)
    g2 = ForwardDiff.gradient(x -> logdensity(ℓ, x), x)
    @test l ≈ l2 atol = atol
    @test g ≈ g2 atol = atol
end

@testset "standard multivariate normal" begin
    K, N = 5, 1000
    ℓ = StandardMultivariateNormal(K)
    @test dimension(ℓ) == hypercube_dimension(ℓ) == K
    @test capabilities(ℓ) == LogDensityOrder(1)
    Z = samples(ℓ, N)
    @test size(Z) == (K, N)
    for i in axes(Z, 1)
        test_gradient(ℓ, Z[:, i])
    end
    @test mean(Z; dims = 2) ≈ zeros(K) atol = 0.01
    @test std(Z; dims = 2) ≈ ones(K) atol = 0.02
end

@testset "multivariate normal using transform" begin
    K = 4
    Q = qr(reshape(range(0.1; step = 0.05, length = K * K), K, K)).Q
    D = Diagonal(range(1; step = .1, length = K))
    μ = collect(range(0.04; step = 0.2, length = K))

    function test_mvnormal(μ, A, Σ)
        K = size(A, 1)
        ℓ = shift(linear(StandardMultivariateNormal(K), A), μ)
        d = MvNormal(μ, Σ)
        C = logpdf(d, μ) - logdensity(ℓ, μ) # get the constant
        @test dimension(ℓ) == hypercube_dimension(ℓ) == K
        @test capabilities(ℓ) == LogDensityOrder(1)
        Z = samples(ℓ, 1000)
        for i in axes(Z, 1)
            x = Z[:, i]
            f, ∇f = logdensity_and_gradient(ℓ, x)
            @test f + C ≈ logpdf(d, x)
            @test ∇f ≈ gradlogpdf(d, x)
        end
        @test mean(Z; dims = 2) ≈ μ atol = 0.02
        @test norm(cov(Z; dims = 2) .- Σ, Inf) ≤ 0.07
    end

    @testset "MvNormal Diagonal" begin
        A = Diagonal(2 .* ones(K))
        Σ = A * A
        test_mvnormal(μ, A, Σ)
    end

    @testset "MvNormal triangular" begin
        Σ = Symmetric(Q * D * Q')
        A = cholesky(Σ, Val(false)).L
        test_mvnormal(μ, A, Σ)
    end

    @testset "MvNormal full" begin
        Σ = Symmetric(Q * D * Q')
        A = Q * Diagonal(.√diag(D))
        test_mvnormal(μ, A, Σ)
    end
end
