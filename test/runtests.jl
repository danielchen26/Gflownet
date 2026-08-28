# test/runtests.jl
# Main test runner for GFlowNet.jl
# Updated after training reorganization (January 2025)

using Test
using Dates

println("🧪 Running GFlowNet.jl Test Suite")
println("Started at: $(now())")
println("=" ^ 60)

# Define test groups with their paths
# Order matters: core functionality first, then higher-level features
test_groups = [
    # MATHEMATICAL CORRECTNESS, run FIRST and deliberately so. These do not
    # test plumbing: they compare the implementation against exact enumeration
    # of the 3x3 grid DAG, where Z = 19 and the target distribution is known in
    # closed form. test_reward_proportionality.jl asserts the defining theorem
    # p(x) proportional to R(x); test_objective_health.jl asserts that every
    # objective produces a non-zero gradient for each component it claims to
    # train, that P_B is a distribution over parents, and that every objective
    # responds to a change in reward. Each of these failed before the repair,
    # and each failure was a silent one -- training ran, losses fell, and the
    # sampled distribution was wrong.
    ("Mathematical Correctness", [
        "theory/test_reward_proportionality.jl",
        "theory/test_objective_health.jl"
    ]),
    ("Neural Networks", [
        "core/neural_networks/test_neural_networks.jl"
    ]),
    ("Core Interface", [
        "core/test_core_interface.jl",
        "core/test_exports.jl"
    ]),
    ("Core Functions", [
        "core/test_core_functions.jl"
    ]),
    ("Flow Computation", [
        "core/flow_computation/test_flow_functions.jl"
    ]),
    ("Policies", [
        "core/policies/test_backward_policy.jl"
    ]),
    ("Training Infrastructure", [
        "integration/test_training.jl"
    ]),
    ("Training Reorganization", [
        "reorganization/test_training_reorganization.jl"
    ]),
    ("Detailed Balance", [
        "core/detailed_balance/test_detailed_balance.jl",
        "core/detailed_balance/test_detailed_balance_comprehensive.jl",
        "core/detailed_balance/test_detailed_balance_summary.jl",
        "objectives/detailed_balance/test_training.jl"
    ]),
    ("Flow Matching", [
        "objectives/flow_matching/test_flow_matching.jl",
        "objectives/flow_matching/test_flow_matching_comprehensive.jl"
    ]),
    ("Learnable Z", [
        "objectives/learnable_z/test_learnable_z.jl",
        "objectives/learnable_z/test_perfect_z_learning.jl"
    ]),
    # Previously orphaned: present on disk but never included. Both verified
    # green standalone before wiring (24 and 20 assertions respectively).
    # NOT wired, because they fail standalone and need a real fix first:
    #   objectives/direct_flow/test_direct_flow.jl            17 pass, 2 fail
    #   objectives/sub_trajectory_balance/...                 ArgumentError:
    #     "Invalid trajectory: length(states) must equal length(actions) + 1"
    ("Exploration", [
        "exploration/test_exploration_improvements.jl",
        "exploration/test_z_learning_rate_multiplier.jl"
    ]),
    ("Multi-Start GFlowNets", [
        "core/multi_start/test_multi_start.jl"
    ]),
    ("Grid World Application", [
        "applications/grid_world/test_grid_world.jl",
        "applications/grid_world/test_grid_world_versions.jl"
    ]),
    ("Molecular Generation", [
        # FIRST in this group, deliberately. It pins the tree structure of the
        # fragment DAG, which is what makes the constant P_B = 1 returned for
        # MolState exact rather than an approximation. If the join site ever
        # becomes a choice, the DAG turns into a lattice and Trajectory Balance
        # silently converges to n(x)R(x) instead of R(x). See the file header.
        "applications/molecular/test_fragment_dag_is_tree.jl",
        "applications/molecular/test_state_features.jl",
        "applications/molecular/test_fragment_joining.jl",
        "applications/molecular/test_reward_function.jl",
        "applications/molecular/test_action_masking.jl",
        "applications/molecular/test_fragment_library.jl",
        "applications/molecular/test_diversity.jl",
        "applications/molecular/test_docking.jl",
        "applications/molecular/test_mogfn.jl",
        "applications/molecular/test_reaction_constraints.jl",
        "applications/molecular/test_integration.jl",
        "applications/molecular/test_state_dim_consistency.jl",
    ]),
    ("Visualization", [
        # Previously orphaned: on disk since the visualization work but never
        # referenced here, so its failures went unreported for months. It needs
        # no running server; the file reproduces the minimum prefix of the
        # include chain in src/utils/visualization/api/unified_server.jl:13-17.
        # Verified standalone: 193 pass, 0 fail, ~55s.
        "visualization/test_real_training_viz.jl"
    ]),
]

# Optional debugging tests (not run by default)
debugging_tests = [
    ("Feature Status", "debugging/diagnostics/test_feature_status.jl"),
    ("Detailed Balance Debug", "debugging/diagnostics/test_detailed_balance_debug.jl"),
    ("Zygote Compatibility", "debugging/zygote_issues/test_zygote_compatibility.jl"),
    ("Mutation Trace", "debugging/zygote_issues/test_mutation_trace.jl")
]

# Track results
failed_groups = String[]
total_time = Ref(0.0)

@testset "GFlowNet.jl Test Suite" begin
    for (group_name, test_files) in test_groups
        group_start = time()
        println("\n" * "-"^60)
        println("📦 Testing: $group_name")
        println("-"^60)
        
        @testset "$group_name" begin
            for test_file in test_files
                test_path = joinpath(@__DIR__, test_file)
                if isfile(test_path)
                    println("  📄 $test_file")
                    try
                        include(test_path)
                    catch e
                        push!(failed_groups, "$group_name - $test_file")
                        @error "Test file errored" file=test_file exception=(e, catch_backtrace())
                        @test false  # surface as a failure; without this a collected error would pass
                    end
                else
                    @warn "Test file not found: $test_file"
                end
            end
        end
        
        group_time = time() - group_start
        total_time[] += group_time
        println("  ✅ Completed in $(round(group_time, digits=2))s")
    end
end

# Summary
println("\n" * "="^60)
println("TEST SUMMARY")
println("="^60)
println("Total time: $(round(total_time[], digits=2))s")

if isempty(failed_groups)
    println("\n🎉 All tests passed!")
else
    println("\n❌ Failed test groups:")
    for group in failed_groups
        println("  - $group")
    end
    println("\n💡 Common issues after reorganization:")
    println("  - Missing imports (functions moved to different modules)")
    println("  - Changed function signatures")
    println("  - Module loading order dependencies")
end

println("\n📝 Note: Debugging tests are available but not run by default:")
for (name, path) in debugging_tests
    println("  - $name: test/$path")
end
println("\nRun them individually when debugging specific issues.")

# Collected errors must still fail the run. Without this, replacing the old
# rethrow with collect-and-continue would let an erroring file pass silently.
if !isempty(failed_groups)
    error("$(length(failed_groups)) test file(s) errored — see the list above")
end
println("\nCompleted at: $(now())")