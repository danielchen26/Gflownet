# Test suite for GFlowNet exploration improvements
# Tests: Entropy regularization, Adaptive Z learning, Experience replay, Importance sampling
#
# Run with: julia --project=. test/exploration/test_exploration_improvements.jl

using Test
using Statistics

# Add parent path for GFlowNet
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using GFlowNet

@testset "GFlowNet Exploration Improvements" begin

    @testset "Phase 1: Entropy Regularization Integration" begin
        @testset "1.1 entropy_weight affects TrainingConfig" begin
            # Default entropy_weight should be 0.01 (AISTATS 2024)
            config = TrainingConfig(n_iterations=10)
            @test config.entropy_weight == 0.01

            # Custom entropy_weight should work
            config_high = TrainingConfig(n_iterations=10, entropy_weight=0.1)
            @test config_high.entropy_weight == 0.1

            config_zero = TrainingConfig(n_iterations=10, entropy_weight=0.0)
            @test config_zero.entropy_weight == 0.0
        end

        @testset "1.2 entropy_weight validation" begin
            # Negative entropy_weight should throw
            @test_throws ArgumentError TrainingConfig(n_iterations=10, entropy_weight=-0.1)
        end
    end

    @testset "Phase 2: Adaptive Z Learning Rate" begin
        @testset "2.1 z_learning_rate_multiplier in config" begin
            # Default should be 10.0 (per peptide paper)
            config = TrainingConfig(n_iterations=10)
            @test config.z_learning_rate_multiplier == 10.0

            # Custom multiplier should work
            config_custom = TrainingConfig(n_iterations=10, z_learning_rate_multiplier=5.0)
            @test config_custom.z_learning_rate_multiplier == 5.0
        end

        @testset "2.2 z_learning_rate_multiplier validation" begin
            # Non-positive should throw
            @test_throws ArgumentError TrainingConfig(n_iterations=10, z_learning_rate_multiplier=0.0)
            @test_throws ArgumentError TrainingConfig(n_iterations=10, z_learning_rate_multiplier=-1.0)
        end
    end

    @testset "Phase 3: Experience Replay Buffer" begin
        @testset "3.1 Replay buffer configuration" begin
            config = TrainingConfig(
                n_iterations=10,
                use_replay_buffer=true,
                replay_buffer_size=1000,
                replay_ratio=0.5,
                replay_priority_alpha=0.6
            )
            @test config.use_replay_buffer == true
            @test config.replay_buffer_size == 1000
            @test config.replay_ratio == 0.5
            @test config.replay_priority_alpha == 0.6
        end

        @testset "3.2 ReplayBuffer basic operations" begin
            buffer = GFlowNet.ReplayBuffer(100; alpha=0.6)

            # Initially empty
            @test length(buffer) == 0
            @test isempty(buffer)

            # Cannot sample from empty buffer
            trajs, weights, indices = GFlowNet.sample_with_weights(buffer, 10)
            @test isempty(trajs)
        end

        @testset "3.3 Replay buffer validation" begin
            # Invalid replay_ratio
            @test_throws ArgumentError TrainingConfig(n_iterations=10, replay_ratio=-0.1)
            @test_throws ArgumentError TrainingConfig(n_iterations=10, replay_ratio=1.5)

            # Invalid priority_alpha
            @test_throws ArgumentError TrainingConfig(n_iterations=10, replay_priority_alpha=-0.1)
            @test_throws ArgumentError TrainingConfig(n_iterations=10, replay_priority_alpha=1.5)
        end
    end

    @testset "Phase 4: Importance Sampling" begin
        @testset "4.1 compute_trajectory_priority" begin
            # Priority computation should work
            priority = GFlowNet.compute_trajectory_priority(1.0)
            @test priority > 0

            # Higher loss = higher priority
            low_loss_priority = GFlowNet.compute_trajectory_priority(0.1)
            high_loss_priority = GFlowNet.compute_trajectory_priority(10.0)
            @test high_loss_priority > low_loss_priority
        end
    end

    @testset "Integration: Full Training with Exploration Features" begin
        @testset "Training with entropy regularization" begin
            # Create a simple grid world model
            model = create_grid_world_gflownet(
                grid_size=5,
                hidden_dim=32,
                learning_rate=0.01
            )

            # Train with entropy regularization enabled
            config = TrainingConfig(
                objective=TRAJECTORY_BALANCE,
                n_iterations=10,
                batch_size=8,
                entropy_weight=0.01,  # Entropy enabled
                epsilon=0.1,
                verbose=false
            )

            history = train_gflownet(model, config; verbose=false)

            # Should complete without errors
            @test length(history.losses) == 10
            @test any(!isnan, history.losses)
        end

        @testset "Training with replay buffer (disabled by default)" begin
            model = create_grid_world_gflownet(
                grid_size=5,
                hidden_dim=32,
                learning_rate=0.01
            )

            # Train without replay buffer (default)
            config = TrainingConfig(
                objective=TRAJECTORY_BALANCE,
                n_iterations=10,
                batch_size=8,
                use_replay_buffer=false,
                verbose=false
            )

            history = train_gflownet(model, config; verbose=false)

            @test length(history.losses) == 10
        end
    end

end

println("\n✅ All exploration improvement tests passed!")
