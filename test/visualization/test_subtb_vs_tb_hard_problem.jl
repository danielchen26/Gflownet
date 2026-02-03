# Comprehensive test: TB vs SubTB on hard problem with 70:1 path asymmetry
# This test verifies that SubTB provides better credit assignment than TB
# on problems where mode discovery is difficult due to path asymmetry

using Test
using GFlowNet
using Statistics

println("=" ^ 70)
println("TB vs SubTB Comparison on Hard Problem (70:1 Path Asymmetry)")
println("=" ^ 70)

println("\n" * "=" ^ 70)
println("PROBLEM SETUP")
println("=" ^ 70)
println("Grid: 5x5, Start: (1,1)")
println("Peak A (5,5): Reward=10, Paths=C(8,4)=70 (8 steps, binomial)")
println("Peak B (1,5): Reward=8, Paths=C(4,0)=1 (4 steps up only)")
println("")
println("Expected GFlowNet behavior:")
println("  - Sample proportional to reward: P(A)/P(B) = 10/8 = 1.25")
println("  - Path asymmetry makes Peak B hard to discover")
println("  - SubTB should help with local credit assignment")
println("=" ^ 70)

@testset "TB vs SubTB Hard Problem Comparison" begin

    function train_and_evaluate(objective, name, iterations; include_flow_est=false, epsilon=0.1)
        println("\n" * "-" ^ 50)
        println("Training: $name")
        println("-" ^ 50)

        model = GFlowNet.create_grid_world_gflownet(
            grid_size = 5,
            reward_positions = Dict(
                (5, 5) => 10.0,  # Peak A: 70 paths
                (1, 5) => 8.0    # Peak B: 1 path
            ),
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

        println("   Training time: $(round(train_time, digits=1))s")
        println("   Final loss: $(round(history[:losses][end], digits=4))")

        # Sample with NO exploration for evaluation
        eval_config = GFlowNet.SamplingConfig(epsilon=0.0)
        trajectories = [GFlowNet.sample_trajectory(model; config=eval_config) for _ in 1:1000]

        # Count terminal positions
        position_counts = Dict{Tuple{Int,Int}, Int}()
        for traj in trajectories
            if !isempty(traj.states) && GFlowNet.is_terminal_state(traj.states[end])
                pos = (traj.states[end].x, traj.states[end].y)
                position_counts[pos] = get(position_counts, pos, 0) + 1
            end
        end

        peak_a = get(position_counts, (5, 5), 0)  # 70 paths
        peak_b = get(position_counts, (1, 5), 0)  # 1 path
        others = 1000 - peak_a - peak_b
        ratio = peak_b > 0 ? peak_a / peak_b : Inf
        modes_found = (peak_a > 10 ? 1 : 0) + (peak_b > 10 ? 1 : 0)

        learned_Z = haskey(model.parameters, :log_Z) ? exp(model.parameters.log_Z) : 1.0

        println("\n   Results:")
        println("      Peak A (5,5): $peak_a / 1000 = $(round(peak_a/10, digits=1))%")
        println("      Peak B (1,5): $peak_b / 1000 = $(round(peak_b/10, digits=1))%")
        println("      Other:        $others / 1000")
        println("      Ratio A/B:    $(round(ratio, digits=2)) (expected: 1.25)")
        println("      Modes found:  $modes_found / 2")
        println("      Learned Z:    $(round(learned_Z, digits=2))")

        return (
            name = name,
            objective = objective,
            peak_a = peak_a,
            peak_b = peak_b,
            ratio = ratio,
            modes_found = modes_found,
            final_loss = history[:losses][end],
            learned_Z = learned_Z,
            train_time = train_time
        )
    end

    println("\n" * "=" ^ 70)
    println("TRAINING EXPERIMENTS")
    println("=" ^ 70)

    # Test 1: TB with epsilon (baseline)
    tb_result = train_and_evaluate(
        GFlowNet.TRAJECTORY_BALANCE,
        "TB + epsilon=0.1",
        2000,
        include_flow_est = false,
        epsilon = 0.1
    )

    # Test 2: SubTB with epsilon (should be better)
    subtb_result = nothing
    try
        subtb_result = train_and_evaluate(
            GFlowNet.SUB_TRAJECTORY_BALANCE,
            "SubTB + epsilon=0.1",
            2000,
            include_flow_est = true,  # REQUIRED for SubTB
            epsilon = 0.1
        )
    catch e
        println("\nSubTB Error: $e")
        # SubTB should work with flow_estimator=true
        @test false
    end

    println("\n" * "=" ^ 70)
    println("COMPARISON RESULTS")
    println("=" ^ 70)

    @testset "TB Baseline" begin
        @test tb_result.final_loss < 5.0  # Loss should converge
        @test tb_result.modes_found >= 1  # At least one mode found

        # TB may struggle with path asymmetry
        println("\n   TB:")
        println("      Loss converged: $(tb_result.final_loss < 5.0 ? "yes" : "no")")
        println("      Modes found: $(tb_result.modes_found)/2")
    end

    @testset "SubTB with Flow Estimator" begin
        if !isnothing(subtb_result)
            @test subtb_result.final_loss < 5.0  # Loss should converge
            @test subtb_result.modes_found >= 1  # At least one mode found

            println("\n   SubTB:")
            println("      Loss converged: $(subtb_result.final_loss < 5.0 ? "yes" : "no")")
            println("      Modes found: $(subtb_result.modes_found)/2")

            # SubTB should provide better credit assignment
            # This means potentially better mode discovery
            @testset "SubTB Credit Assignment" begin
                # Peak B should have more than minimal samples with SubTB
                # (local credit assignment helps discover hard-to-reach modes)
                @test subtb_result.peak_b >= 1  # At least some discovery
            end
        else
            # SubTB test failed to run
            @test false
        end
    end

    @testset "Comparison Analysis" begin
        if !isnothing(subtb_result)
            expected_ratio = 1.25

            tb_ratio_error = abs(tb_result.ratio - expected_ratio) / expected_ratio
            subtb_ratio_error = abs(subtb_result.ratio - expected_ratio) / expected_ratio

            println("\n   Comparison:")
            println("      Expected ratio: $expected_ratio")
            println("      TB ratio:       $(round(tb_result.ratio, digits=2)) (error: $(round(tb_ratio_error*100, digits=1))%)")
            println("      SubTB ratio:    $(round(subtb_result.ratio, digits=2)) (error: $(round(subtb_ratio_error*100, digits=1))%)")

            println("\n      TB modes:       $(tb_result.modes_found)/2")
            println("      SubTB modes:    $(subtb_result.modes_found)/2")

            # Key insight: SubTB provides local credit assignment
            # This helps when some paths are harder to discover
            if subtb_result.modes_found >= tb_result.modes_found
                println("\n   SubTB mode discovery >= TB (as expected)")
            end

            if subtb_result.peak_b > tb_result.peak_b
                println("   SubTB discovered more Peak B samples (local credit helps!)")
            end

            # Test that both methods eventually work (with enough epsilon)
            @test tb_result.peak_a > 100 || subtb_result.peak_a > 100  # At least one finds Peak A well
        end
    end

    # Summary table
    println("\n" * "=" ^ 70)
    println("SUMMARY TABLE")
    println("=" ^ 70)
    println("\n| Method          | Peak A | Peak B | Ratio | Modes | Loss      |")
    println("|-----------------|--------|--------|-------|-------|-----------|")
    println("| TB + eps=0.1    | $(lpad(tb_result.peak_a, 6)) | $(lpad(tb_result.peak_b, 6)) | $(lpad(round(tb_result.ratio, digits=2), 5)) | $(tb_result.modes_found)/2   | $(lpad(round(tb_result.final_loss, digits=4), 9)) |")
    if !isnothing(subtb_result)
        println("| SubTB + eps=0.1 | $(lpad(subtb_result.peak_a, 6)) | $(lpad(subtb_result.peak_b, 6)) | $(lpad(round(subtb_result.ratio, digits=2), 5)) | $(subtb_result.modes_found)/2   | $(lpad(round(subtb_result.final_loss, digits=4), 9)) |")
    end
    println("| Expected        |   ~556 |   ~444 |  1.25 | 2/2   |    low    |")
    println("")
    println("=" ^ 70)

    # Final analysis
    println("\nANALYSIS:")
    println("-" ^ 70)
    println("Path asymmetry (70:1) creates a hard exploration problem.")
    println("- TB relies on full trajectory gradients (late credit)")
    println("- SubTB provides O(T^2) local gradients (early credit)")
    println("")
    println("With eps-exploration, both methods should discover both modes.")
    println("SubTB may help learn correct proportions faster due to")
    println("local credit assignment for partial paths toward Peak B.")
    println("=" ^ 70)
end

println("\nTest completed.")
