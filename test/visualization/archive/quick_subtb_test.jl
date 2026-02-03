# Quick TB vs SubTB comparison test
using GFlowNet
using Statistics

println("=" ^ 60)
println("Quick TB vs SubTB Comparison (500 iterations)")
println("=" ^ 60)

function train_and_evaluate(objective, name, iterations; include_flow_est=false, epsilon=0.1)
    model = GFlowNet.create_grid_world_gflownet(
        grid_size = 5,
        reward_positions = Dict((5, 5) => 10.0, (1, 5) => 8.0),
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,
        hidden_dim = 64,
        include_flow_estimator = include_flow_est
    )

    config = GFlowNet.TrainingConfig(
        objective = objective,
        batch_size = 64,
        n_iterations = iterations,
        learning_rate = 0.005,
        epsilon = epsilon,
        epsilon_decay = true
    )

    t0 = time()
    history = GFlowNet.train_gflownet(model, config; verbose=false)
    train_time = time() - t0

    # Sample with NO exploration for evaluation
    eval_config = GFlowNet.SamplingConfig(epsilon=0.0)
    trajectories = [GFlowNet.sample_trajectory(model; config=eval_config) for _ in 1:500]

    position_counts = Dict{Tuple{Int,Int}, Int}()
    for traj in trajectories
        if length(traj.states) > 0 && GFlowNet.is_terminal_state(traj.states[end])
            pos = (traj.states[end].x, traj.states[end].y)
            position_counts[pos] = get(position_counts, pos, 0) + 1
        end
    end

    peak_a = get(position_counts, (5, 5), 0)
    peak_b = get(position_counts, (1, 5), 0)
    ratio = peak_b > 0 ? peak_a / peak_b : Inf
    modes_found = (peak_a > 10 ? 1 : 0) + (peak_b > 10 ? 1 : 0)

    return (
        name = name,
        peak_a = peak_a,
        peak_b = peak_b,
        ratio = ratio,
        modes_found = modes_found,
        final_loss = history[:losses][end],
        train_time = train_time
    )
end

println("\n1. Training TB + eps=0.1...")
tb_result = train_and_evaluate(
    GFlowNet.TRAJECTORY_BALANCE,
    "TB",
    500,
    include_flow_est = false,
    epsilon = 0.1
)
println("   Time: $(round(tb_result.train_time, digits=1))s")
println("   Peak A: $(tb_result.peak_a), Peak B: $(tb_result.peak_b)")
println("   Ratio: $(round(tb_result.ratio, digits=2)), Modes: $(tb_result.modes_found)/2")

println("\n2. Training SubTB + eps=0.1...")
subtb_result = train_and_evaluate(
    GFlowNet.SUB_TRAJECTORY_BALANCE,
    "SubTB",
    500,
    include_flow_est = true,
    epsilon = 0.1
)
println("   Time: $(round(subtb_result.train_time, digits=1))s")
println("   Peak A: $(subtb_result.peak_a), Peak B: $(subtb_result.peak_b)")
println("   Ratio: $(round(subtb_result.ratio, digits=2)), Modes: $(subtb_result.modes_found)/2")

println("\n" * "=" ^ 60)
println("COMPARISON (Expected ratio: 1.25)")
println("=" ^ 60)
println("| Method | Peak A | Peak B | Ratio  | Modes |")
println("|--------|--------|--------|--------|-------|")
println("| TB     | $(lpad(tb_result.peak_a, 6)) | $(lpad(tb_result.peak_b, 6)) | $(lpad(round(tb_result.ratio, digits=2), 6)) | $(tb_result.modes_found)/2   |")
println("| SubTB  | $(lpad(subtb_result.peak_a, 6)) | $(lpad(subtb_result.peak_b, 6)) | $(lpad(round(subtb_result.ratio, digits=2), 6)) | $(subtb_result.modes_found)/2   |")
println("=" ^ 60)

# Analysis
println("\nANALYSIS:")
if subtb_result.peak_b > tb_result.peak_b
    println("- SubTB discovered more Peak B samples (local credit helps)")
end
if subtb_result.modes_found >= tb_result.modes_found
    println("- SubTB found at least as many modes as TB")
end
if subtb_result.ratio < tb_result.ratio && subtb_result.ratio != Inf
    println("- SubTB achieved better ratio (closer to expected 1.25)")
end

println("\nSUMMARY:")
println("Both TB and SubTB struggle with this hard problem (70:1 path asymmetry).")
println("SubTB provides O(T^2) local credit signals, which should help with longer training.")
println("The key fix was making SubTB differentiable (flow estimator gradients flow properly).")
