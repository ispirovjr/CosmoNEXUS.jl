using Test
using FFTW
using Statistics
using NeoNEXUS
using CosmoNEXUS

function centeredGrid(N, L=1.0)
    dx = L / N
    return range(-L / 2 + dx / 2, L / 2 - dx / 2; length=N)
end


function makeFullShearField(X, Y, Z)
    amp = exp.(-10f0 .* Float32.(X .^ 2 .+ Y .^ 2 .+ Z .^ 2))

    fullShear = zeros(Float32, size(amp)..., 3, 3)
    fullShear[:, :, :, 1, 1] .= -3f0 .* amp
    fullShear[:, :, :, 2, 2] .= -2f0 .* amp
    fullShear[:, :, :, 3, 3] .= -1f0 .* amp

    s12 = 0.15f0 .* amp .* Float32.(X .- Y)
    s13 = 0.10f0 .* amp .* Float32.(X .+ Z)
    s23 = -0.12f0 .* amp .* Float32.(Y .- Z)

    fullShear[:, :, :, 1, 2] .= s12
    fullShear[:, :, :, 2, 1] .= s12
    fullShear[:, :, :, 1, 3] .= s13
    fullShear[:, :, :, 3, 1] .= s13
    fullShear[:, :, :, 2, 3] .= s23
    fullShear[:, :, :, 3, 2] .= s23

    return fullShear
end


function splitTracelessShear(fullShear)
    thetaField =
        fullShear[:, :, :, 1, 1] .+
        fullShear[:, :, :, 2, 2] .+
        fullShear[:, :, :, 3, 3]

    tracelessShear = copy(fullShear)
    traceThird = thetaField ./ 3f0

    tracelessShear[:, :, :, 1, 1] .-= traceThird
    tracelessShear[:, :, :, 2, 2] .-= traceThird
    tracelessShear[:, :, :, 3, 3] .-= traceThird

    return tracelessShear, thetaField
end


@testset "CosmoNEXUS" begin
    N = 16
    x = centeredGrid(N)
    X = reshape(x, :, 1, 1)
    Y = reshape(x, 1, :, 1)
    Z = reshape(x, 1, 1, :)
    kr = collect(Float64, fftfreq(N) .* N .* 2pi)
    kx = kr
    ky = kr
    kz = kr

    @testset "Tidal - Constant Field" begin
        field = ones(Float32, N, N, N)
        cache = computeTidalEigenvalues(field, kx, ky, kz)

        @test maximum(abs.(cache.λ1)) < 1e-5
        @test maximum(abs.(cache.λ2)) < 1e-5
        @test maximum(abs.(cache.λ3)) < 1e-5
    end

    @testset "Tidal - Isotropic Quadratic" begin
        field = X .^ 2 .+ Y .^ 2 .+ Z .^ 2
        tidalCache = computeTidalEigenvalues(field, kx, ky, kz)
        densityCache = computeHessianEigenvalues(field, kx, ky, kz)

        @test maximum(abs.(tidalCache.λ3)) > 0.01
        @test median(abs.(tidalCache.λ3)) < median(abs.(densityCache.λ3))
    end

    @testset "Tidal - Planar Wall" begin
        field = X .^ 2 .+ Y .* 0 .+ Z .* 0
        tidalCache = computeTidalEigenvalues(field, kx, ky, kz)

        @test maximum(abs.(tidalCache.λ3)) > 0.01
    end

    @testset "Shear - Constant Tensor Field" begin
        shearField = zeros(Float32, N, N, N, 3, 3)
        shearField[:, :, :, 1, 1] .= -3f0
        shearField[:, :, :, 2, 2] .= -2f0
        shearField[:, :, :, 3, 3] .= -1f0

        cache = computeShearEigenvalues(shearField)

        @test maximum(abs.(cache.λ1 .+ 3f0)) < 1e-5
        @test maximum(abs.(cache.λ2 .+ 2f0)) < 1e-5
        @test maximum(abs.(cache.λ3 .+ 1f0)) < 1e-5
    end

    @testset "NEXUSTidal Pipeline Orchestration" begin
        filter = GaussianFourierFilter((N, N, N))
        node = NodeFeature((N, N, N), kx, ky, kz)
        filament = LineFeature((N, N, N), kx, ky, kz)
        wall = SheetFeature((N, N, N), kx, ky, kz)
        scales = [1.0, 2.0]
        runner = NEXUSTidal(filter, node, filament, wall, scales)

        field = abs.(randn(Float32, N, N, N)) .+ 1f0
        thresholds = CosmoNEXUS.run(runner, field)

        @test thresholds isa NamedTuple
        @test any(node.significanceMap .> 0)
        @test any(filament.significanceMap .> 0)
        @test any(wall.significanceMap .> 0)
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

        thetaField = randn(Float32, N, N, N)
        thresholds = CosmoNEXUS.run(runner, thetaField)

        @test thresholds isa NamedTuple
        @test any(node.significanceMap .> 0)
        @test any(filament.significanceMap .> 0)
        @test any(wall.significanceMap .> 0)
        @test size(node.thresholdMap) == (N, N, N)
        @test size(filament.thresholdMap) == (N, N, N)
        @test size(wall.thresholdMap) == (N, N, N)
    end

    @testset "NEXUSShear Pipeline Orchestration" begin
        filter = GaussianFourierFilter((N, N, N))
        node = NodeFeature((N, N, N), kx, ky, kz)
        filament = LineFeature((N, N, N), kx, ky, kz)
        wall = SheetFeature((N, N, N), kx, ky, kz)
        scales = [1.0, 2.0]
        runner = NEXUSShear(filter, node, filament, wall, scales)

        fullShear = makeFullShearField(X, Y, Z)
        thresholds = CosmoNEXUS.run(runner, fullShear)

        @test thresholds isa NamedTuple
        @test any(node.significanceMap .> 0)
        @test any(filament.significanceMap .> 0)
        @test any(wall.significanceMap .> 0)
        @test size(node.thresholdMap) == (N, N, N)
        @test size(filament.thresholdMap) == (N, N, N)
        @test size(wall.thresholdMap) == (N, N, N)
    end

    @testset "NEXUSShear Traceless Convenience Overload" begin
        scales = [1.0, 2.0]
        fullShear = makeFullShearField(X, Y, Z)
        tracelessShear, thetaField = splitTracelessShear(fullShear)

        fullRunner = NEXUSShear(N, scales)
        splitRunner = NEXUSShear(N, scales)

        fullThresholds = CosmoNEXUS.run(fullRunner, fullShear)
        splitThresholds = CosmoNEXUS.run(splitRunner, tracelessShear, thetaField)

        @test fullThresholds.nodeThres ≈ splitThresholds.nodeThres atol=1e-5 rtol=1e-5
        @test fullThresholds.filamentThres ≈ splitThresholds.filamentThres atol=1e-5 rtol=1e-5
        @test fullThresholds.wallThres ≈ splitThresholds.wallThres atol=1e-5 rtol=1e-5

        @test fullRunner.node.significanceMap ≈ splitRunner.node.significanceMap atol=1e-5 rtol=1e-5
        @test fullRunner.filament.significanceMap ≈ splitRunner.filament.significanceMap atol=1e-5 rtol=1e-5
        @test fullRunner.wall.significanceMap ≈ splitRunner.wall.significanceMap atol=1e-5 rtol=1e-5
    end

    @testset "NEXUSShear Multithreaded Matches Sequential" begin
        scales = [1.0, 2.0]
        fullShear = makeFullShearField(X, Y, Z)

        seqRunner = NEXUSShear(N, scales)
        mtRunner = NEXUSShear(N, scales)

        seqThresholds = CosmoNEXUS.run(seqRunner, fullShear)
        mtThresholds = CosmoNEXUS.runMultithreaded(mtRunner, fullShear)

        @test seqThresholds.nodeThres ≈ mtThresholds.nodeThres atol=1e-5 rtol=1e-5
        @test seqThresholds.filamentThres ≈ mtThresholds.filamentThres atol=1e-5 rtol=1e-5
        @test seqThresholds.wallThres ≈ mtThresholds.wallThres atol=1e-5 rtol=1e-5

        @test seqRunner.node.significanceMap ≈ mtRunner.node.significanceMap atol=1e-5 rtol=1e-5
        @test seqRunner.filament.significanceMap ≈ mtRunner.filament.significanceMap atol=1e-5 rtol=1e-5
        @test seqRunner.wall.significanceMap ≈ mtRunner.wall.significanceMap atol=1e-5 rtol=1e-5
    end
end
