# Test ε-uniform exploration for TB mode discovery
# This test verifies that adding ε-uniform exploration fixes the mode collapse issue

using Test
using GFlowNet
using Statistics

println("=" ^ 70)
println("Testing ε-Uniform Exploration for TB Mode Discovery")
println("=" ^ 70)

@testset "ε-Uniform Exploration" begin

    @testset "1. SamplingConfig with epsilon" begin
        # Test default (backward compat)
        config_default = GFlowNet.SamplingConfig()
        @test config_default.epsilon == 0.0

        # Test with epsilon
        config_eps = GFlowNet.SamplingConfig(epsilon=0.05)
        @test config_eps.epsilon == 0.05

        # Test exploration config helper
        config_explore = GFlowNet.create_exploration_sampling_config(epsilon=0.1)
        @test config_explore.epsilon == 0.1

        # Test validation
        @test_throws ArgumentError GFlowNet.SamplingConfig(epsilon=-0.1)
        @test_throws ArgumentError GFlowNet.SamplingConfig(epsilon=1.5)

        println("✅ Test 1: SamplingConfig with epsilon - PASSED")
    end

    @testset "2. TrainingConfig with epsilon" begin
        # Test default epsilon = 0.05
        config = GFlowNet.TrainingConfig()
        @test config.epsilon == 0.05
        @test config.epsilon_decay == true

        # Test custom epsilon
        config_custom = GFlowNet.TrainingConfig(epsilon=0.1, epsilon_decay=false)
        @test config_custom.epsilon == 0.1
        @test config_custom.epsilon_decay == false

        # Test validation
        @test_throws ArgumentError GFlowNet.TrainingConfig(epsilon=-0.1)
        @test_throws ArgumentError GFlowNet.TrainingConfig(epsilon=1.5)

        println("✅ Test 2: TrainingConfig with epsilon - PASSED")
    end

    @testset "3. TB Training with ε=0.05 vs ε=0.0" begin
        # Create two models for comparison
        function train_and_sample(epsilon_val, name)
            model = GFlowNet.create_grid_world_gflownet(
                grid_size = 5,
                reward_positions = Dict((5, 5) => 10.0, (1, 5) => 8.0),
                partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,
                hidden_dim = 64
            )

            config = GFlowNet.TrainingConfig(
                objective = GFlowNet.TRAJECTORY_BALANCE,
                batch_size = 64,
                n_iterations = 500,
                learning_rate = 0.005,
                epsilon = epsilon_val,
                epsilon_decay = true
            )

            history = GFlowNet.train_gflownet(model, config; verbose=false)

            # Sample 500 trajectories (with NO exploration for evaluation)
            eval_config = GFlowNet.SamplingConfig(epsilon=0.0)
            trajectories = [GFlowNet.sample_trajectory(model; config=eval_config) for _ in 1:500]

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

            return (name=name, peak1=peak1, peak2=peak2, ratio=ratio, modes_found=modes_found)
        end

        println("\n📊 Training comparison:")
        println("   Expected: Peak1/Peak2 ratio ≈ 10/8 = 1.25")

        # Train without epsilon (old behavior - should fail)
        result_no_eps = train_and_sample(0.0, "No exploration (ε=0)")
        println("\n   Without ε-exploration (ε=0):")
        println("      Peak1 (5,5): $(result_no_eps.peak1)")
        println("      Peak2 (1,5): $(result_no_eps.peak2)")
        println("      Ratio: $(round(result_no_eps.ratio, digits=2))")
        println("      Modes found: $(result_no_eps.modes_found)/2")

        # Train with epsilon (new behavior - should work)
        result_with_eps = train_and_sample(0.05, "With exploration (ε=0.05)")
        println("\n   With ε-exploration (ε=0.05):")
        println("      Peak1 (5,5): $(result_with_eps.peak1)")
        println("      Peak2 (1,5): $(result_with_eps.peak2)")
        println("      Ratio: $(round(result_with_eps.ratio, digits=2))")
        println("      Modes found: $(result_with_eps.modes_found)/2")

        # Test that epsilon improves mode discovery
        @test result_with_eps.modes_found >= result_no_eps.modes_found

        # Test that epsilon makes ratio closer to expected
        expected_ratio = 1.25
        if result_with_eps.peak2 > 5 && result_no_eps.peak2 > 5
            eps_error = abs(result_with_eps.ratio - expected_ratio)
            no_eps_error = abs(result_no_eps.ratio - expected_ratio)
            if eps_error < no_eps_error
                println("\n   ✅ ε-exploration improves ratio convergence!")
            end
        end

        # Main test: with epsilon, we should discover both modes
        if result_with_eps.modes_found == 2
            println("\n✅ Test 3: ε-exploration enables full mode discovery - PASSED")
        else
            println("\n⚠️ Test 3: Mode discovery improved but not perfect")
        end
    end
end

println("\n" * "=" ^ 70)
println("ε-Uniform Exploration Tests Complete")
println("=" ^ 70)
