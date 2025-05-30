using LogDensityTestSuite, Test, Statistics, LinearAlgebra, Distributions, FiniteDifferences
using LogDensityProblems: capabilities, dimension, logdensity, logdensity_and_gradient,
    LogDensityOrder
using LogDensityTestSuite: hypercube_dimension, source_to_destination, destination_to_source,
    DensityMode, DensityGradientMode, _find_x_norm, weight, weight_and_gradient
using LogExpFunctions: logaddexp, logistic

"Finite difference method used in tests."
const FDM = central_fdm(5, 1)

# for comparing ∂c∂y
Base.isapprox(::LogDensityTestSuite.Zeros, a; atol) = all(a -> abs(a) ≤ atol, a)

"""
Test consistency of `transformation` by comparing source <-> destination calculations,
including Jacobians, determinants, etc.

`D` is the dimension of inputs. `N` is the number of samples to calculate.
"""
function test_transformation_consistency(transformation, D; N = 100, atol = √eps())
    for _ in 1:N
        x = randn(D)
        y = source_to_destination(transformation, x)
        ∂y∂x = jacobian(FDM, x -> source_to_destination(transformation, x), x)[1]
        c = logabsdet(∂y∂x)[1]
        x1 = destination_to_source(DensityMode(), transformation, y)
        @test x1.x ≈ x atol = atol
        @test x1.c ≈ c atol = atol
        x2 = destination_to_source(DensityGradientMode(), transformation, y)
        @test x2.x ≈ x atol = atol
        @test x2.c ≈ c atol = atol
        # NOTE: line below implemented this way because Inv only supports v'*M
        @test mapreduce(r -> x2.∂x∂y⊤ * r, hcat, eachrow(∂y∂x)) ≈ I(D) atol = atol
        ∂c∂y = grad(FDM, y -> destination_to_source(DensityMode(), transformation, y).c, y)[1]
        @test x2.∂c∂y ≈ ∂c∂y atol = atol
    end
end

"""
Test gradient with automatic differentiation.

NOTE: default tolerances are generous because we are using finite differences.
"""
function test_gradient(ℓ, x; atol = √eps(), rtol = 0.01)
    l, g = logdensity_and_gradient(ℓ, x)
    l2 = logdensity(ℓ, x)
    g2 = grad(FDM, x -> logdensity(ℓ, x), x)[1]
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
    ℓ = (shift(μ) ∘ linear(L))(StandardMultivariateNormal(2))
    @test mean(rand(ℓ) for _ in 1:10000) ≈ μ atol = 0.05
end

####
#### transformations
####

@testset "transformation mappings" begin
    # a “random” matrix
    A = [1.2735208104831561 -1.457297079176796 1.319353974452655;
         0.4689489441596014 0.16213981899823227 0.8220450119958059;
         -0.05684164000044907 0.533081876167392 0.3732266922991697]
    test_transformation_consistency(linear(A), size(A, 1))
    test_transformation_consistency(shift(A[:, 1]), size(A, 1))
    test_transformation_consistency(elongate(1.2), 3)
    test_transformation_consistency(elongate(0.4), 3)
    test_transformation_consistency(funnel(), 3)
end

@testset "multivariate normal using transform" begin

    function test_mvnormal(μ, A, Σ)
        ℓ = (shift(μ) ∘ linear(A))(StandardMultivariateNormal(length(μ)))
        d = MvNormal(μ, Σ)
        @test dimension(ℓ) == hypercube_dimension(ℓ) == length(μ)
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

    K = 4
    μ = collect(range(0.04; step = 0.2, length = K))

    @testset "MvNormal standard" begin
        N = 3
        D = Diagonal(ones(3))
        test_mvnormal(zeros(N), D, D)
    end

    @testset "MvNormal Diagonal" begin
        A = Diagonal(2 .* ones(K))
        Σ = A * A'
        test_mvnormal(μ, A, Σ)
    end

    @testset "MvNormal Diagonal w/ UniformScaling" begin
        A = I(K) * 0.4
        Σ = (A * A')
        test_mvnormal(μ, A, Σ)
    end

    @testset "MvNormal triangular and full" begin
        Q = qr(reshape(range(0.1; step = 0.05, length = K * K), K, K)).Q
        D = Diagonal(range(1; step = .1, length = K))
        @testset "MvNormal triangular" begin
            Σ = Symmetric(Q * D * Q')
            A = cholesky(Σ).L
            test_mvnormal(μ, A, Σ)
        end
        @testset "MvNormal full" begin
            Σ = Symmetric(Q * D * Q')
            A = Q * Diagonal(.√diag(D))
            test_mvnormal(μ, A, Σ)
        end
    end

    @testset "MvNormal ill conditioned" begin
        μ = [-1.729922440774685, -0.011762500688978205, 0.11423091067230899, 0.05085717388622323, 0.09102774773399233, -0.3769237300508154, -1.1645971596831883, -1.4196407006756644, 0.07406060991401947]
        D = Diagonal([0.31285715405356296, 1.6321047397137334, 1.9304214045496948, 0.9408515651923572, 0.632832415315841, 0.3994529605030148, 0.9479547802750243, 0.000686699019868418, 0.14074551354895906])
        C = [1.0 -0.625893845478092 -0.8607538232958145 0.4906036948283603 -0.045129301268019346 -0.9798256449980116 -0.09448716779625055 0.1972478332046149 -0.38125524332165456; 0.0 0.7799082601131022 0.22963314745353192 -0.8390321758549951 -0.2940681265758735 0.05788305453491861 -0.30348581879657555 -0.3395815944065493 0.40817023926937634; 0.0 0.0 0.45428127109998945 0.07704183020878513 0.5013749270904165 0.09940288184055725 -0.4898077520422466 -0.04390387380845317 -0.39358273046921877; 0.0 0.0 0.0 0.22225566111771966 -0.5034002085122711 0.1540822287067389 -0.52831870161212 -0.20197326086456527 -0.4230725997740589; 0.0 0.0 0.0 0.0 0.6377293278924043 0.002108173376346147 -0.563819920556515 0.07024142256309863 0.20409522211102057; 0.0 0.0 0.0 0.0 0.0 0.05444765270890811 0.21770654511030652 0.4167989822452558 0.4096707796964533; 0.0 0.0 0.0 0.0 0.0 0.0 0.12102564140379203 0.6237333486866049 -0.1142510107612157; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.4851374500990013 -0.2027266958462243; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.30084429646746724]
        A = D * C'
        Σ = Symmetric(D * (C'*C) * D)
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
end

@testset "elongate" begin
    K, N = 5, 1000
    ℓ = elongate(0.5)(StandardMultivariateNormal(K))
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
    ℓ2 = (shift(μ2) ∘ linear(Diagonal(fill(0.01, K))))(StandardMultivariateNormal(K))
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

    @test_throws DimensionMismatch mix(0.5, ℓ1, StandardMultivariateNormal(K + 1))
end

@testset "directional mixture" begin
    K, N = 5, 1000
    ℓ1 = StandardMultivariateNormal(K)
    μ2 = fill(1.7, K)
    ℓ2 = (shift(μ2) ∘ linear(Diagonal(fill(0.01, K))))(StandardMultivariateNormal(K))
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
