# test/runtests.jl
# Main test runner for GFlowNet.jl

using Test

println("🧪 Running GFlowNet.jl Test Suite")
println("=" ^ 40)

@testset "GFlowNet.jl Test Suite" begin

    @testset "Core Functions Validation" begin
        include("test_core_functions.jl")
    end

    @testset "ComponentArrays and Lux.jl Integration" begin
        include("test_componentarrays_integration.jl")
    end

    @testset "Minimal GFlowNet Training" begin
        include("test_minimal_gfn_training.jl")
    end

    @testset "End-to-End Training" begin
        include("test_end_to_end_training.jl")
    end

end

println("\n🎉 All tests completed!")
