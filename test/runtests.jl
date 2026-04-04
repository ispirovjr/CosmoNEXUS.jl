using Test
using FFTW
using Statistics
using NeoNEXUS
using CosmoNEXUS

@testset "CosmoNEXUS" begin

    # Shared grid setup
    function centeredGrid(N, L=1.0)
        dx = L / N
        range(-L / 2 + dx / 2, L / 2 - dx / 2; length=N)
    end

    N = 16
    x = centeredGrid(N)
    X = reshape(x, :, 1, 1)
    Y = reshape(x, 1, :, 1)
    Z = reshape(x, 1, 1, :)
    kr = collect(Float64, fftfreq(N) .* N .* 2π)
    kx = kr
    ky = kr
    kz = kr


    # === Tidal Eigenvalue Tests ===

    @testset "Tidal - Constant Field" begin
        # Tidal eigenvalues of constant field should be zero
        field = ones(Float32, N, N, N)
        cache = computeTidalEigenvalues(field, kx, ky, kz)
        @test maximum(abs.(cache.λ1)) < 1e-5
        @test maximum(abs.(cache.λ2)) < 1e-5
        @test maximum(abs.(cache.λ3)) < 1e-5
    end

    @testset "Tidal - Isotropic Quadratic" begin
        # For ρ = x²+y²+z², compare tidal eigenvalues to density eigenvalues.
        # The 1/k² factor redistributes power, so tidal eigenvalues
        # should be smaller in magnitude than density eigenvalues.
        field = X .^ 2 .+ Y .^ 2 .+ Z .^ 2
        tidalCache = computeTidalEigenvalues(field, kx, ky, kz)
        densityCache = computeHessianEigenvalues(field, kx, ky, kz)

        # Tidal eigenvalues should be non-zero
        @test maximum(abs.(tidalCache.λ3)) > 0.01

        # Tidal eigenvalues should be smaller than density eigenvalues (1/k² suppresses)
        @test median(abs.(tidalCache.λ3)) < median(abs.(densityCache.λ3))
    end

    @testset "Tidal - Planar Wall" begin
        # x² only → density Hessian: (0, 0, 2)
        # Tidal tensor for anisotropic source should produce non-zero eigenvalues
        field = X .^ 2 .+ Y .* 0 .+ Z .* 0
        tidalCache = computeTidalEigenvalues(field, kx, ky, kz)

        # Tidal eigenvalues should be non-zero (potential spreads force)
        @test maximum(abs.(tidalCache.λ3)) > 0.01
    end


    # === Orchestration Tests ===

    @testset "NEXUSTidal Pipeline Orchestration" begin
        filter = GaussianFourierFilter((N, N, N))
        node = NodeFeature((N, N, N), kx, ky, kz)
        filament = LineFeature((N, N, N), kx, ky, kz)
        wall = SheetFeature((N, N, N), kx, ky, kz)
        scales = [1.0, 2.0]
        runner = NEXUSTidal(filter, node, filament, wall, scales)

        # Use positive density (required for log filtering)
        field = abs.(randn(Float32, N, N, N)) .+ 1f0

        # Run the pipeline
        thresholds = CosmoNEXUS.run(runner, field)

        # Check significance maps populated
        @test any(node.significanceMap .> 0)
        @test any(filament.significanceMap .> 0)
        @test any(wall.significanceMap .> 0)

        # Check threshold maps have correct size
        @test size(node.thresholdMap) == (N, N, N)
        @test size(filament.thresholdMap) == (N, N, N)
        @test size(wall.thresholdMap) == (N, N, N)
    end

    @testset "NEXUSDiv Pipeline Orchestration" begin
        filter = GaussianFourierFilter((N, N, N))
        node = NodeFeature((N, N, N), kx, ky, kz)
        filament = LineFeature((N, N, N), kx, ky, kz)
        wall = SheetFeature((N, N, N), kx, ky, kz)
        scales = [1.0, 2.0]
        runner = NEXUSDiv(filter, node, filament, wall, scales)

        # Velocity divergence can be positive or negative
        θField = randn(Float32, N, N, N)

        # Run the pipeline
        thresholds = CosmoNEXUS.run(runner, θField)

        # Check significance maps populated
        @test any(node.significanceMap .> 0)
        @test any(filament.significanceMap .> 0)
        @test any(wall.significanceMap .> 0)

        # Check threshold maps have correct size
        @test size(node.thresholdMap) == (N, N, N)
        @test size(filament.thresholdMap) == (N, N, N)
        @test size(wall.thresholdMap) == (N, N, N)
    end

end
