using LogDensityTestSuite, Test, Statistics
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
