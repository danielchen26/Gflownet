# Reward Shaping Test - Compensating for 70:1 Path Asymmetry
# Run with: julia --project=. examples/core_features/reward_shaping_test.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using GFlowNet

function test_reward_shaping()
    println("=" ^ 70)
    println("REWARD SHAPING TEST - Compensating for 70:1 Path Asymmetry")
    println("=" ^ 70)
    println()

    # 5×5 grid - compensate for path asymmetry with reward
    # Peak (5,5): 70 paths × R=10 = 700 weighted
    # Peak (1,5): 1 path × R=700 = 700 weighted (balanced!)
    grid_size = 5

    println("Theory: P(x) ∝ R(x) × #paths(x)")
    println("  Peak (5,5): 70 paths × R=10 = 700")
    println("  Peak (1,5): 1 path × R=700 = 700")
    println("  → Both modes should be discovered with ~1:1 ratio")
    println()

    # Test with compensated rewards
    reward_positions = Dict(
        (5, 5) => 10.0,   # 70 paths
        (1, 5) => 700.0   # 1 path, but 70x higher reward
    )

    model = GFlowNet.create_grid_world_gflownet(
        grid_size = grid_size,
        reward_positions = reward_positions,
        hidden_dim = 64,
        learning_rate = 0.005,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
    )

    config = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
        n_iterations = 1000,
        batch_size = 32,
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = 0.1,
        epsilon_decay = true,
        entropy_weight = 0.01,
        use_replay_buffer = true,
        replay_buffer_size = 5000,
        replay_ratio = 0.5,
        verbose = false
    )

    print("Training with reward shaping (R(1,5)=700)... ")
    GFlowNet.train_gflownet(model, config; verbose=false)
    println("done!")

    eval_config = GFlowNet.SamplingConfig(
        strategy = GFlowNet.STOCHASTIC_SAMPLING,
        epsilon = 0.0,
        max_trajectory_length = 100
    )

    samples = [GFlowNet.sample_trajectory(model; config=eval_config) for _ in 1:1000]

    p1 = 0
    p2 = 0
    for traj in samples
        terminal = traj.states[end]
        pos = (terminal.x, terminal.y)
        if pos == (5, 5)
            p1 += 1
        elseif pos == (1, 5)
            p2 += 1
        end
    end

    println()
    println("Results WITH reward shaping:")
    println("  Peak (5,5) R=10:  $p1 / 1000 ($(round(p1/10, digits=1))%)")
    println("  Peak (1,5) R=700: $p2 / 1000 ($(round(p2/10, digits=1))%)")
    modes = (p1 > 10 ? 1 : 0) + (p2 > 10 ? 1 : 0)
    println("  Modes discovered: $modes / 2")

    if p2 > 10
        actual_ratio = p1 / max(p2, 1)
        println("  Actual sampling ratio: $(round(actual_ratio, digits=2)):1")
        println()
        println("✅ SUCCESS: Reward shaping SOLVES the extreme mode collapse!")
        println("   By compensating the 70:1 path asymmetry with reward,")
        println("   both modes are now discovered.")
    else
        println()
        println("⚠️  Still collapsed even with reward shaping")
    end
    println()

    return (p1, p2)
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_reward_shaping()
end
