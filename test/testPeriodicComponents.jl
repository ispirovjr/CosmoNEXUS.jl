@testset "Periodic component thresholding in CosmoNEXUS runners" begin
    density = fill(0.001f0, 8, 8, 8)
    density[1, 3, 4] = 600f0

    for multithread in (false, true), periodicBounds in (false, true)
        # Empty scales leave this controlled significance field untouched.
        tidal = NEXUSTidal(8, Float64[])
        tidal.node.significanceMap[1, 3, 4] = 3f0
        tidal.node.significanceMap[8, 3, 4] = 1f0
        tidal.node.significanceMap[4, 4, 4] = 2f0
        result = tidal(density; multithread=multithread, periodicBounds=periodicBounds)
        @test periodicBounds ? 1f0 < result.nodeThres < 2f0 : result.nodeThres == 1f0

        potential = NEXUSPotential(8, Float64[])
        potential.node.significanceMap .= tidal.node.significanceMap
        expectedNode = deepcopy(potential.node)
        expected = componentErosionPlateauThreshold!(expectedNode; periodicBounds=periodicBounds)
        # Exercise the existing positional multithread overload as well.
        result = potential(zeros(Float32, 8, 8, 8), multithread; periodicBounds=periodicBounds)
        @test result.nodeThres == expected
        @test potential.node.thresholdMap == expectedNode.thresholdMap
        @test periodicBounds ? 2f0 < result.nodeThres < 3f0 : 1f0 < result.nodeThres < 2f0
    end
end
