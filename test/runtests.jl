# test/runtests.jl
# Main test runner for GFlowNet.jl

using Test

println("🧪 Running GFlowNet.jl Test Suite")
println("=" ^ 40)

# Define test groups with their paths
test_groups = [
    ("Core Functions", [
        "core/test_core_functions.jl",
        "core/test_core_interface.jl",
        "core/test_utilities.jl"
    ]),
    ("Flow Computation", [
        "core/flow_computation/test_flow_functions.jl"
    ]),
    ("Policies", [
        "core/policies/test_backward_policy.jl"
    ]),
    ("Neural Networks", [
        "core/neural_networks/test_neural_networks.jl"
    ]),
    ("Detailed Balance Core", [
        "core/detailed_balance/test_detailed_balance.jl",
        "core/detailed_balance/test_detailed_balance_comprehensive.jl",
        "core/detailed_balance/test_detailed_balance_summary.jl"
    ]),
    ("Training Objectives", [
        "objectives/detailed_balance/test_training.jl",
        "objectives/learnable_z/test_learnable_z.jl",
        "objectives/learnable_z/test_perfect_z_learning.jl"
    ]),
    ("Applications", [
        "applications/grid_world/test_grid_world.jl",
        "applications/supply_chain/test_supply_chain.jl"
    ]),
    ("Integration", [
        "integration/test_training.jl"
    ])
]

@testset "GFlowNet.jl Test Suite" begin
    for (group_name, test_files) in test_groups
        @testset "$group_name" begin
            for test_file in test_files
                test_path = joinpath(@__DIR__, test_file)
                if isfile(test_path)
                    println("\n  Running $test_file...")
                    include(test_path)
                else
                    @warn "Test file not found: $test_file"
                end
            end
        end
    end
end

println("\n🎉 All tests completed!")
println("\n📝 Note: Debugging tests are in test/debugging/ and are not run by default.")
println("   Run them individually when debugging specific issues.")