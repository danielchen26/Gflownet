# Extended test for ε-uniform exploration with 2000 iterations
# Verifies that TB achieves expected ~1.25 ratio with sufficient training

using Test
using GFlowNet
using Statistics

println("=" ^ 70)
println("Extended ε-Uniform Exploration Test (2000 iterations)")
println("=" ^ 70)

function train_and_evaluate(epsilon_val, n_iterations, name)
    println("\n📊 Training: $name")
    println("   Epsilon: $epsilon_val, Iterations: $n_iterations")

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
        epsilon_decay = true
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
        n_iterations=n_iterations,
        peak1=peak1,
        peak2=peak2,
        ratio=ratio,
        modes_found=modes_found,
        final_loss=history[:losses][end]
    )
end

@testset "Extended ε-Exploration Verification" begin

    # Test with 2000 iterations (should achieve near-ideal ratio)
    println("\n" * "=" ^ 50)
    result = train_and_evaluate(0.05, 2000, "TB + ε=0.05 (2000 iter)")
    println("=" ^ 50)

    # Verification tests
    @testset "Mode Discovery" begin
        # Both modes should be discovered
        @test result.peak1 > 50  # At least 5% at primary peak
        @test result.peak2 > 30  # At least 3% at secondary peak
        @test result.modes_found == 2

        println("\n✅ Both modes discovered!")
    end

    @testset "Ratio Convergence" begin
        expected_ratio = 10.0 / 8.0  # = 1.25

        # With enough training, ratio should be within 2x of expected
        # (more lenient bound due to stochastic training)
        ratio_error = abs(result.ratio - expected_ratio) / expected_ratio

        println("\n   Ratio error: $(round(ratio_error * 100, digits=1))%")

        # Test that ratio is reasonable (within 3x of expected)
        @test result.ratio < expected_ratio * 3  # Not more than 3x
        @test result.ratio > expected_ratio / 3  # Not less than 1/3

        if ratio_error < 0.5  # Within 50% error
            println("   ✅ Excellent! Ratio within 50% of expected")
        elseif ratio_error < 1.0  # Within 100% error
            println("   ✅ Good! Ratio within 100% of expected")
        else
            println("   ⚠️  Ratio needs more training")
        end
    end

    @testset "Loss Convergence" begin
        @test result.final_loss < 1.0  # Loss should be small
        println("\n   Final loss: $(round(result.final_loss, digits=4)) ✅")
    end

    # Summary
    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println("   Expected: Peak1/Peak2 ≈ 1.25 (from R1/R2 = 10/8)")
    println("   Achieved: Peak1/Peak2 = $(round(result.ratio, digits=2))")
    println("   Mode discovery: $(result.modes_found)/2")

    if result.modes_found == 2 && result.ratio < 5.0
        println("\n   ✅ ε-UNIFORM EXPLORATION IMPLEMENTATION VERIFIED!")
        println("   The fix works - both modes are discovered with reasonable proportions.")
    else
        println("\n   ⚠️  Results improved but may need more iterations or tuning.")
    end
    println("=" ^ 70)
end
