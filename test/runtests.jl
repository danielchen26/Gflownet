# test/runtests.jl
# Main test runner for GFlowNet.jl

using Test

println("🧪 Running GFlowNet.jl Test Suite")
println("=" ^ 40)

@testset "GFlowNet.jl Test Suite" begin
    
    @testset "Test Utilities" begin
        include("test_utilities.jl")
    end
    
    @testset "Neural Network Integration" begin
        include("test_neural_networks.jl")
    end
    
    @testset "Core Interface" begin
        include("test_core_interface.jl")
    end
    
    @testset "Grid World Application" begin
        include("test_grid_world.jl")
    end
    
    @testset "Training Infrastructure" begin
        include("test_training.jl")
    end
    
    # @testset "Supply Chain Application" begin  # Removed in core-fixes branch
    #     include("test_supply_chain.jl")
    # end
    
    @testset "Core Functions (Comprehensive)" begin
        include("test_core_functions.jl")
    end
    
    @testset "Working vs Broken Features Documentation" begin
        include("test_working_vs_broken_features.jl")
    end
    
end

println("\n🎉 All tests completed!")
println("\n📝 Note: Some tests have been archived in test/archive/old_tests/")
println("   These tests reference outdated APIs and need updating to work with")
println("   the current on-demand computation architecture.")