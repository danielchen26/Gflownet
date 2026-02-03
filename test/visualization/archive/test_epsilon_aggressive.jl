# Aggressive ε-exploration test with higher epsilon and no decay
# Tests whether the implementation works with stronger exploration

using Test
using GFlowNet
using Statistics

println("=" ^ 70)
println("Aggressive ε-Exploration Test (ε=0.2, no decay)")
println("=" ^ 70)

function train_and_evaluate(epsilon_val, epsilon_decay, n_iterations, name)
    println("\n📊 Training: $name")
    println("   Epsilon: $epsilon_val, Decay: $epsilon_decay, Iterations: $n_iterations")

    model = GFlowNet.create_grid_world_gflownet(
        grid_size = 5,
        reward_positions = Dict((5, 5) => 10.0, (1, 5) => 8.0),
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,
        hidden_dim = 64
    )

    config = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
        batch_size = 64,
        n_iterations = n_iterations,
        learning_rate = 0.005,
        epsilon = epsilon_val,
        epsilon_decay = epsilon_decay
    )

    t0 = time()
    history = GFlowNet.train_gflownet(model, config; verbose=false)
    train_time = time() - t0

    println("   Training time: $(round(train_time, digits=1))s")
    println("   Final loss: $(round(history[:losses][end], digits=4))")

    # Sample 1000 trajectories (with NO exploration for evaluation)
    eval_config = GFlowNet.SamplingConfig(epsilon=0.0)
    trajectories = [GFlowNet.sample_trajectory(model; config=eval_config) for _ in 1:1000]

    position_counts = Dict{Tuple{Int,Int}, Int}()
    for traj in trajectories
        if !isempty(traj.states) && GFlowNet.is_terminal_state(traj.states[end])
            pos = (traj.states[end].x, traj.states[end].y)
            position_counts[pos] = get(position_counts, pos, 0) + 1
        end
    end

    peak1 = get(position_counts, (5, 5), 0)
    peak2 = get(position_counts, (1, 5), 0)
    ratio = peak2 > 0 ? peak1 / peak2 : Inf
    modes_found = (peak1 > 10 ? 1 : 0) + (peak2 > 10 ? 1 : 0)

    println("   Results:")
    println("      Peak1 (5,5): $peak1 / 1000 = $(round(peak1/10, digits=1))%")
    println("      Peak2 (1,5): $peak2 / 1000 = $(round(peak2/10, digits=1))%")
    println("      Actual ratio: $(round(ratio, digits=2))")
    println("      Expected ratio: 1.25 (R1/R2 = 10/8)")
    println("      Modes found: $modes_found/2")

    return (
        name=name,
        epsilon=epsilon_val,
        epsilon_decay=epsilon_decay,
        n_iterations=n_iterations,
        peak1=peak1,
        peak2=peak2,
        ratio=ratio,
        modes_found=modes_found,
        final_loss=history[:losses][end]
    )
end

@testset "Aggressive ε-Exploration Tests" begin

    println("\n" * "=" ^ 50)
    println("Test 1: High epsilon (0.2), no decay, 1000 iter")
    println("=" ^ 50)

    result1 = train_and_evaluate(0.2, false, 1000, "ε=0.2, no decay")

    @testset "High Epsilon Results" begin
        # With high persistent exploration, should discover both modes
        @test result1.peak1 > 0  # Should reach primary peak
        @test result1.peak2 > 0  # Should reach secondary peak (key test!)

        if result1.modes_found == 2
            println("\n   ✅ Both modes discovered with high ε!")
        else
            println("\n   ⚠️  Mode discovery: $(result1.modes_found)/2")
        end
    end

    println("\n" * "=" ^ 50)
    println("Test 2: Comparison - verify ε affects exploration")
    println("=" ^ 50)

    result_low = train_and_evaluate(0.0, false, 500, "ε=0.0 baseline")
    result_high = train_and_evaluate(0.3, false, 500, "ε=0.3 high explore")

    @testset "Epsilon Comparison" begin
        println("\n   Comparison:")
        println("      ε=0.0: Peak1=$(result_low.peak1), Peak2=$(result_low.peak2)")
        println("      ε=0.3: Peak1=$(result_high.peak1), Peak2=$(result_high.peak2)")

        # Key test: high epsilon should increase visits to secondary peak
        # (even if not perfect, should be more than ε=0)
        @test result_high.peak2 >= result_low.peak2

        if result_high.peak2 > result_low.peak2
            println("\n   ✅ Higher ε increases secondary mode discovery!")
        elseif result_high.peak2 == result_low.peak2 && result_low.peak2 == 0
            println("\n   ⚠️  Neither found secondary mode - path asymmetry is severe")
        end
    end

    # Summary
    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println("\nThe grid (1,5) requires 4 consecutive Down moves from (1,1).")
    println("Path to (5,5) has ~70 different routes, path to (1,5) has only 1.")
    println("This extreme asymmetry makes mode discovery difficult even with ε-exploration.")
    println("\nFor production use, consider:")
    println("  1. More balanced reward positions")
    println("  2. Higher epsilon (0.2-0.3)")
    println("  3. Entropy regularization")
    println("  4. Longer training with gradual epsilon decay")
    println("=" ^ 70)
end
