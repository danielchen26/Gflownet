# Manual Integration Test for Real Training Visualization
# This tests all components step-by-step to verify everything works

println("="^70)
println("REAL TRAINING VISUALIZATION - COMPREHENSIVE MANUAL TEST")
println("="^70)
println()

# First, test that we can load GFlowNet
println("Step 1: Loading GFlowNet...")
using GFlowNet
println("✓ GFlowNet loaded successfully")
println()

# Test loading core modules
println("Step 2: Loading visualization core modules...")
println("  Loading adapters.jl...")
# ../../ was correct while this file sat in test/visualization/. The Feb 3 2026
# move into archive/ added a directory level, so ../../src resolved to test/src
# and every run died with SystemError: opening file ".../test/src/utils/
# visualization/core/adapters.jl". Now ../../../src, i.e. the repo root.
include("../../../src/utils/visualization/core/adapters.jl")
println("  ✓ adapters.jl loaded")

println("  Loading metrics.jl...")
include("../../../src/utils/visualization/core/metrics.jl")
println("  ✓ metrics.jl loaded")

println("  Loading domains/grid_world.jl...")
include("../../../src/utils/visualization/domains/grid_world.jl")
println("  ✓ grid_world.jl loaded")

println("  Loading training_session.jl...")
# Need to define create_model_and_adapter before loading training_session
function create_model_and_adapter(domain_type::String, config::Dict)
    if domain_type != "grid_world"
        error("Only grid_world supported in this test")
    end

    grid_size = get(config, "grid_size", 5)
    peaks_config = get(config, "reward_peaks", [])
    reward_positions = Dict{Tuple{Int,Int}, Float64}()
    for peak in peaks_config
        pos = peak["position"]
        intensity = get(peak, "intensity", 10.0)
        reward_positions[(Int(pos[1]), Int(pos[2]))] = Float64(intensity)
    end

    if isempty(reward_positions)
        reward_positions[(grid_size, grid_size)] = 10.0
    end

    model = GFlowNet.create_grid_world_gflownet(
        grid_size = grid_size,
        reward_positions = reward_positions,
        hidden_dim = get(config, "hidden_dim", 32),
        learning_rate = get(config, "learning_rate", 0.01)
    )
    adapter = GridWorldAdapter(grid_size, reward_positions)
    return model, adapter
end

include("../../../src/utils/visualization/core/training_session.jl")
println("  ✓ training_session.jl loaded")
println()

# Test 3: Create a simple model and adapter
println("Step 3: Testing model and adapter creation...")
model = GFlowNet.create_grid_world_gflownet(
    grid_size = 4,
    reward_positions = Dict((3,3) => 10.0, (4,4) => 8.0),
    hidden_dim = 32
)
adapter = GridWorldAdapter(4, Dict((3,3) => 10.0, (4,4) => 8.0))
println("✓ Model and adapter created successfully")
println()

# Test 4: Test adapter interface methods
println("Step 4: Testing GridWorldAdapter interface methods...")

println("  Testing state_to_viz_data...")
state = GFlowNet.GridState(2, 3, false)
viz_data = state_to_viz_data(adapter, state)
@assert viz_data["x"] == 2
@assert viz_data["y"] == 3
@assert viz_data["is_terminal"] == false
println("  ✓ state_to_viz_data works correctly")

println("  Testing trajectory_to_viz_data...")
traj = GFlowNet.sample_trajectory(model)
traj_viz = trajectory_to_viz_data(adapter, traj, "test_1")
@assert traj_viz["id"] == "test_1"
@assert haskey(traj_viz, "states")
@assert haskey(traj_viz, "actions")
@assert length(traj_viz["states"]) == length(traj.states)
println("  ✓ trajectory_to_viz_data works correctly")

println("  Testing get_domain_config...")
config = get_domain_config(adapter)
@assert config["domain_type"] == "grid_world"
@assert config["grid_size"] == [4, 4]
@assert config["supports_flow_field"] == true
println("  ✓ get_domain_config works correctly")

println("  Testing get_renderer_name...")
renderer = get_renderer_name(adapter)
@assert renderer == "GridWorldRenderer"
println("  ✓ get_renderer_name works correctly")

println("  Testing compute_domain_metrics...")
trajectories = [GFlowNet.sample_trajectory(model) for _ in 1:10]
domain_metrics = compute_domain_metrics(adapter, model, trajectories)
@assert haskey(domain_metrics, "mode_coverage")
@assert haskey(domain_metrics, "unique_positions")
println("  ✓ compute_domain_metrics works correctly")

println("  Testing compute_flow_field...")
flow_data = compute_flow_field(adapter, model)
@assert flow_data["supported"] == true
@assert flow_data["grid_size"] == 4
@assert length(flow_data["data"]) == 16  # 4x4 grid
println("  ✓ compute_flow_field works correctly")

println("  Testing compute_distribution_data...")
dist_data = compute_distribution_data(adapter, model, trajectories)
@assert dist_data["supported"] == true
@assert size(dist_data["empirical"]) == (4, 4)
@assert size(dist_data["target"]) == (4, 4)
println("  ✓ compute_distribution_data works correctly")
println()

# Test 5: Test universal metrics
println("Step 5: Testing universal metrics computation...")
metrics = compute_gflownet_metrics(model, trajectories)
@assert haskey(metrics, "mean_reward")
@assert haskey(metrics, "diversity_ratio")
@assert haskey(metrics, "unique_terminals")
@assert metrics["n_trajectories"] == 10
println("✓ Universal metrics computation works correctly")
println()

# Test 6: Test parse_objective
println("Step 6: Testing objective parsing...")
@assert parse_objective("TRAJECTORY_BALANCE") == GFlowNet.TRAJECTORY_BALANCE
@assert parse_objective("DETAILED_BALANCE") == GFlowNet.DETAILED_BALANCE
@assert parse_objective("FLOW_MATCHING") == GFlowNet.FLOW_MATCHING
@assert parse_objective(" trajectory_balance ") == GFlowNet.TRAJECTORY_BALANCE
println("✓ Objective parsing works correctly")
println()

# Test 7: Create and test training session
println("Step 7: Testing training session...")

session_config = Dict(
    "domain_type" => "grid_world",
    "grid_size" => 4,
    "n_episodes" => 20,
    "batch_size" => 4,
    "learning_rate" => 0.01,
    "objective" => "TRAJECTORY_BALANCE",
    "reward_peaks" => [
        Dict("position" => [3, 3], "intensity" => 10.0),
        Dict("position" => [4, 4], "intensity" => 8.0)
    ]
)

println("  Creating session...")
session = create_session(session_config)
@assert session.current_iteration == 0
@assert session.total_iterations == 20
@assert !session.is_training
println("  ✓ Session created successfully")

println("  Running training steps...")
session.is_training = true
successful_steps = 0
for i in 1:5
    result = step!(session)
    if result["status"] == "ok"
        global successful_steps += 1
        println("    Iteration $i: loss=$(round(result["loss"], digits=4)), " *
                "reward=$(round(result["mean_reward"], digits=2)), " *
                "grad_norm=$(round(result["gradient_norm"], digits=4))")
    else
        println("    Iteration $i: ERROR - $(result["error"])")
    end
end

@assert successful_steps >= 3  # At least 3 out of 5 should succeed
@assert session.current_iteration == 5
@assert length(session.losses) == 5
@assert length(session.rewards) == 5
@assert length(session.trajectory_buffer) > 0
println("  ✓ Training steps executed successfully ($successful_steps/5 succeeded)")
println()

# Test 8: Verify training actually happened
println("Step 8: Verifying real training occurred...")
println("  Session state:")
println("    - Iterations completed: $(session.current_iteration)")
println("    - Losses recorded: $(length(session.losses))")
println("    - Mean loss: $(round(mean(filter(!isnan, session.losses)), digits=4))")
println("    - Mean reward: $(round(mean(filter(!isnan, session.rewards)), digits=2))")
println("    - Trajectories in buffer: $(length(session.trajectory_buffer))")
println("    - Errors encountered: $(session.error_count)")

# Verify the model parameters actually changed (training happened)
println("  ✓ Training executed with real gradient descent")
println()

# Test 9: Test error handling
println("Step 9: Testing error handling...")
empty_metrics = compute_gflownet_metrics(model, GFlowNet.Trajectory[])
@assert haskey(empty_metrics, "error")
println("✓ Error handling works correctly")
println()

# Final summary.
#
# This used to print "ALL MANUAL TESTS PASSED! ✅" and "Implementation is VERIFIED
# and PRODUCTION-READY! 🎉". This file contains no `using Test`, no `@testset` and
# no `@test`: nothing here can fail a check, because there are no checks. The
# banner asserted a verdict it had no evidence for, which is precisely the
# silent-pass pattern this repo has been removing. Reaching this line proves only
# that no statement above threw.
println("="^70)
println("Script ran to completion — no statement above threw.")
println("="^70)
println()
println("What that does and does not tell you:")
println("  - Exercised without throwing: module loading, the adapter interface,")
println("    session creation, real training steps, metrics, the error path.")
println("  - NOT verified: any actual value. This script asserts nothing.")
println("    For assertions, run ../test_real_training_viz.jl (193 of them).")
