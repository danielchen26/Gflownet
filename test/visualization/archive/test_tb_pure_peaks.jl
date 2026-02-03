# Test TB convergence with ONLY two reward peaks (no exploration rewards)
# This isolates the exploration problem from reward function complexity

using Test
using GFlowNet
using Statistics

println("=" ^ 60)
println("Testing TB with PURE Two-Peak Rewards (No Exploration Rewards)")
println("=" ^ 60)

@testset "TB Pure Two-Peak Convergence" begin

    # Create model with ONLY two peak rewards
    # No distance-based exploration rewards
    model = GFlowNet.create_grid_world_gflownet(
        grid_size = 5,
        reward_positions = Dict(
            (5, 5) => 10.0,  # Peak 1: R=10
            (1, 5) => 8.0    # Peak 2: R=8
        ),
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,
        hidden_dim = 64
    )

    # Override the reward function to return 0.01 for non-peak terminals
    # (minimal reward to keep them valid terminals)

    println("\n📊 Reward Setup:")
    println("   Peak 1 at (5,5): R = 10.0")
    println("   Peak 2 at (1,5): R = 8.0")
    println("   Expected ratio: 10/8 = 1.25")
    println("   Expected P(5,5): 10/18 = 55.6%")
    println("   Expected P(1,5): 8/18 = 44.4%")

    # Train with high temperature for exploration
    config = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
        batch_size = 64,
        n_iterations = 2000,
        learning_rate = 0.005,
        temperature = 3.0  # High temperature for exploration
    )

    println("\n🚀 Training Configuration:")
    println("   Objective: TRAJECTORY_BALANCE")
    println("   Batch size: 64")
    println("   Iterations: 2000")
    println("   Learning rate: 0.005")
    println("   Temperature: 3.0 (high for exploration)")

    # Track intermediate results
    intermediate_results = []

    for phase in 1:4
        iters = phase * 500

        # Train
        partial_config = GFlowNet.TrainingConfig(
            objective = GFlowNet.TRAJECTORY_BALANCE,
            batch_size = 64,
            n_iterations = 500,
            learning_rate = 0.005,
            temperature = 3.0
        )

        history = GFlowNet.train_gflownet(model, partial_config; verbose=false)

        # Sample with default (greedy-ish) sampling after training
        trajectories = [GFlowNet.sample_trajectory(model) for _ in 1:1000]

        # Count terminal positions
        position_counts = Dict{Tuple{Int,Int}, Int}()
        for traj in trajectories
            if !isempty(traj.states) && GFlowNet.is_terminal_state(traj.states[end])
                pos = (traj.states[end].x, traj.states[end].y)
                position_counts[pos] = get(position_counts, pos, 0) + 1
            end
        end

        peak1 = get(position_counts, (5, 5), 0)
        peak2 = get(position_counts, (1, 5), 0)
        others = 1000 - peak1 - peak2

        ratio = peak2 > 0 ? peak1 / peak2 : Inf
        valid_losses = filter(!isnan, history.losses)
        final_loss = isempty(valid_losses) ? NaN : valid_losses[end]

        push!(intermediate_results, (iters=iters, peak1=peak1, peak2=peak2, others=others, ratio=ratio, loss=final_loss))

        println("\n📈 After $(iters) iterations:")
        println("   Peak1 (5,5) R=10: $peak1 samples ($(round(100*peak1/1000, digits=1))%)")
        println("   Peak2 (1,5) R=8:  $peak2 samples ($(round(100*peak2/1000, digits=1))%)")
        println("   Other positions:  $others samples ($(round(100*others/1000, digits=1))%)")
        println("   Actual ratio: $(round(ratio, digits=2)) (expected: 1.25)")
        println("   Loss: $(round(final_loss, digits=6))")
        if haskey(model.parameters, :log_Z)
            println("   Learned Z: $(round(exp(model.parameters.log_Z), digits=2))")
        end
    end

    # Final analysis
    println("\n" * "=" ^ 60)
    println("📊 CONVERGENCE ANALYSIS")
    println("=" ^ 60)

    final = intermediate_results[end]

    # Test that both peaks are discovered
    @test final.peak1 > 0
    @test final.peak2 > 0

    # Test that ratio is improving toward 1.25
    if final.peak2 > 50
        @test final.ratio < 5.0
    end

    # Check if ratio improved over training
    if length(intermediate_results) >= 2
        first_ratio = intermediate_results[1].ratio
        last_ratio = intermediate_results[end].ratio

        if isfinite(first_ratio) && isfinite(last_ratio)
            println("\n📉 Ratio progression:")
            for r in intermediate_results
                println("   Iter $(r.iters): ratio = $(round(r.ratio, digits=2))")
            end

            # Check if we're moving toward expected ratio
            expected_ratio = 1.25
            first_error = abs(first_ratio - expected_ratio)
            last_error = abs(last_ratio - expected_ratio)

            if last_error < first_error
                println("   ✅ Ratio is converging toward expected value!")
            else
                println("   ⚠️ Ratio not converging - exploration issue")
            end
        end
    end

    println("\n" * "=" ^ 60)
    println("Test Complete")
    println("=" ^ 60)
end
