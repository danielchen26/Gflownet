# Test: Does higher epsilon help with mode discovery?
using GFlowNet

println("=" ^ 60)
println("Higher Epsilon Test: ε=0.3 vs ε=0.1")
println("=" ^ 60)

function test_epsilon(epsilon, iterations=1000)
    model = GFlowNet.create_grid_world_gflownet(
        grid_size = 5,
        reward_positions = Dict((5, 5) => 10.0, (1, 5) => 8.0),
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,
        hidden_dim = 64
    )

    config = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
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
    trajectories = [GFlowNet.sample_trajectory(model; config=eval_config) for _ in 1:1000]

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

    return (epsilon=epsilon, peak_a=peak_a, peak_b=peak_b, ratio=ratio, time=train_time)
end

println("\n1. Testing ε=0.1 (standard)...")
r1 = test_epsilon(0.1, 1000)
println("   Peak A: $(r1.peak_a), Peak B: $(r1.peak_b), Ratio: $(round(r1.ratio, digits=2))")

println("\n2. Testing ε=0.3 (higher)...")
r2 = test_epsilon(0.3, 1000)
println("   Peak A: $(r2.peak_a), Peak B: $(r2.peak_b), Ratio: $(round(r2.ratio, digits=2))")

println("\n3. Testing ε=0.5 (aggressive)...")
r3 = test_epsilon(0.5, 1000)
println("   Peak A: $(r3.peak_a), Peak B: $(r3.peak_b), Ratio: $(round(r3.ratio, digits=2))")

println("\n" * "=" ^ 60)
println("RESULTS COMPARISON (Expected ratio: 1.25)")
println("=" ^ 60)
println("| Epsilon | Peak A | Peak B | Ratio  |")
println("|---------|--------|--------|--------|")
println("| 0.1     | $(lpad(r1.peak_a, 6)) | $(lpad(r1.peak_b, 6)) | $(lpad(round(r1.ratio, digits=2), 6)) |")
println("| 0.3     | $(lpad(r2.peak_a, 6)) | $(lpad(r2.peak_b, 6)) | $(lpad(round(r2.ratio, digits=2), 6)) |")
println("| 0.5     | $(lpad(r3.peak_a, 6)) | $(lpad(r3.peak_b, 6)) | $(lpad(round(r3.ratio, digits=2), 6)) |")
println("=" ^ 60)

println("\nCONCLUSION:")
if r3.ratio < r1.ratio && r3.peak_b > r1.peak_b
    println("✓ Higher epsilon improves mode discovery")
    println("  But even ε=0.5 may not achieve the expected 1.25 ratio")
    println("  The 70:1 path asymmetry is a fundamentally hard problem")
else
    println("  Results show path asymmetry impact on training")
end
