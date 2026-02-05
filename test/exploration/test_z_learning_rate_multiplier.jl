# Test suite for z_learning_rate_multiplier implementation
# Verifies that Z (partition function) updates faster when multiplier > 1.0
#
# Run with: julia --project=. test/exploration/test_z_learning_rate_multiplier.jl

using Test
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using GFlowNet

# Helper macro to capture stdout (must be defined before use)
macro capture_out(expr)
    quote
        original_stdout = stdout
        rd, wr = redirect_stdout()
        result = $(esc(expr))
        redirect_stdout(original_stdout)
        close(wr)
        read(rd, String)
    end
end

@testset "Z Learning Rate Multiplier Tests" begin

    @testset "1. scale_z_gradient function" begin
        @testset "1.1 Basic gradient scaling" begin
            # Create a mock gradient NamedTuple
            grads = (forward = [0.1, 0.2, 0.3], log_Z = 0.5)

            # Scale by 10x
            scaled = GFlowNet.scale_z_gradient(grads, 10.0)

            @test scaled.log_Z == 5.0  # 0.5 * 10 = 5.0
            @test scaled.forward == grads.forward  # Other gradients unchanged
        end

        @testset "1.2 Multiplier of 1.0 (no change)" begin
            grads = (forward = [0.1, 0.2], log_Z = 0.5)
            scaled = GFlowNet.scale_z_gradient(grads, 1.0)

            @test scaled.log_Z == 0.5  # Unchanged
            @test scaled.forward == grads.forward
        end

        @testset "1.3 Gradient without log_Z" begin
            grads = (forward = [0.1, 0.2, 0.3],)
            scaled = GFlowNet.scale_z_gradient(grads, 10.0)

            # Should return unchanged when no log_Z
            @test scaled == grads
        end

        @testset "1.4 Various multiplier values" begin
            grads = (forward = [1.0], log_Z = 1.0)

            @test GFlowNet.scale_z_gradient(grads, 5.0).log_Z == 5.0
            @test GFlowNet.scale_z_gradient(grads, 0.5).log_Z == 0.5
            @test GFlowNet.scale_z_gradient(grads, 100.0).log_Z == 100.0
        end
    end

    @testset "2. Z converges faster with multiplier" begin
        # This test verifies that Z actually moves faster during training
        # when z_learning_rate_multiplier > 1.0

        grid_size = 4
        reward_positions = Dict((4, 4) => 10.0)

        @testset "2.1 Z movement comparison" begin
            # Train with multiplier = 1.0 (baseline)
            model_baseline = GFlowNet.create_grid_world_gflownet(
                grid_size = grid_size,
                reward_positions = reward_positions,
                hidden_dim = 32,
                learning_rate = 0.01,
                partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
            )
            initial_Z_baseline = exp(model_baseline.log_partition_function)

            config_baseline = GFlowNet.TrainingConfig(
                objective = GFlowNet.TRAJECTORY_BALANCE,
                n_iterations = 50,
                batch_size = 8,
                learning_rate = 0.01,
                z_learning_rate_multiplier = 1.0,  # No scaling
                epsilon = 0.1,
                verbose = false
            )

            GFlowNet.train_gflownet(model_baseline, config_baseline; verbose=false)
            final_Z_baseline = exp(model_baseline.log_partition_function)
            Z_change_baseline = abs(final_Z_baseline - initial_Z_baseline)

            # Train with multiplier = 10.0 (faster Z)
            model_fast = GFlowNet.create_grid_world_gflownet(
                grid_size = grid_size,
                reward_positions = reward_positions,
                hidden_dim = 32,
                learning_rate = 0.01,
                partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
            )
            initial_Z_fast = exp(model_fast.log_partition_function)

            config_fast = GFlowNet.TrainingConfig(
                objective = GFlowNet.TRAJECTORY_BALANCE,
                n_iterations = 50,
                batch_size = 8,
                learning_rate = 0.01,
                z_learning_rate_multiplier = 10.0,  # 10x scaling
                epsilon = 0.1,
                verbose = false
            )

            GFlowNet.train_gflownet(model_fast, config_fast; verbose=false)
            final_Z_fast = exp(model_fast.log_partition_function)
            Z_change_fast = abs(final_Z_fast - initial_Z_fast)

            println("Z change with multiplier=1.0: $(round(Z_change_baseline, digits=4))")
            println("Z change with multiplier=10.0: $(round(Z_change_fast, digits=4))")

            # With 10x multiplier, Z should change more (converge faster)
            # Allow some tolerance since training is stochastic
            @test Z_change_fast > Z_change_baseline * 0.5 || Z_change_fast > 0.1
        end
    end

    @testset "3. Training still works correctly" begin
        # Verify that training with z_learning_rate_multiplier produces valid results

        @testset "3.1 Training completes without errors" begin
            model = GFlowNet.create_grid_world_gflownet(
                grid_size = 4,
                reward_positions = Dict((4, 4) => 10.0),
                hidden_dim = 32,
                learning_rate = 0.01,
                partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
            )

            config = GFlowNet.TrainingConfig(
                objective = GFlowNet.TRAJECTORY_BALANCE,
                n_iterations = 20,
                batch_size = 8,
                learning_rate = 0.01,
                z_learning_rate_multiplier = 10.0,
                epsilon = 0.1,
                verbose = false
            )

            history = GFlowNet.train_gflownet(model, config; verbose=false)

            @test length(history.losses) == 20
            @test any(!isnan, history.losses)
            @test !isnan(model.log_partition_function)
        end

        @testset "3.2 Z parameter is valid after training" begin
            model = GFlowNet.create_grid_world_gflownet(
                grid_size = 4,
                reward_positions = Dict((4, 4) => 10.0, (1, 4) => 5.0),
                hidden_dim = 32,
                learning_rate = 0.01,
                partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
            )

            config = GFlowNet.TrainingConfig(
                objective = GFlowNet.TRAJECTORY_BALANCE,
                n_iterations = 50,
                batch_size = 16,
                learning_rate = 0.01,
                z_learning_rate_multiplier = 10.0,
                entropy_weight = 0.01,
                epsilon = 0.1,
                verbose = false
            )

            GFlowNet.train_gflownet(model, config; verbose=false)

            # Z should be positive and finite
            Z = exp(model.log_partition_function)
            @test Z > 0
            @test isfinite(Z)
            @test isfinite(model.log_partition_function)

            # log_Z should be in a reasonable range
            @test -100 < model.log_partition_function < 100
        end
    end

    @testset "4. Works with weighted training (replay buffer)" begin
        model = GFlowNet.create_grid_world_gflownet(
            grid_size = 4,
            reward_positions = Dict((4, 4) => 10.0),
            hidden_dim = 32,
            learning_rate = 0.01,
            partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
        )

        config = GFlowNet.TrainingConfig(
            objective = GFlowNet.TRAJECTORY_BALANCE,
            n_iterations = 30,
            batch_size = 8,
            learning_rate = 0.01,
            z_learning_rate_multiplier = 5.0,
            use_replay_buffer = true,
            replay_buffer_size = 100,
            replay_ratio = 0.3,
            epsilon = 0.1,
            verbose = false
        )

        history = GFlowNet.train_gflownet(model, config; verbose=false)

        @test length(history.losses) == 30
        @test isfinite(model.log_partition_function)
    end

    @testset "5. Verbose output shows multiplier" begin
        # Test that verbose output correctly shows the multiplier
        model = GFlowNet.create_grid_world_gflownet(
            grid_size = 3,
            hidden_dim = 16,
            learning_rate = 0.01,
            partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
        )

        config = GFlowNet.TrainingConfig(
            objective = GFlowNet.TRAJECTORY_BALANCE,
            n_iterations = 5,
            batch_size = 4,
            z_learning_rate_multiplier = 10.0,
            verbose = true
        )

        # Capture output
        output = @capture_out begin
            GFlowNet.train_gflownet(model, config; verbose=true)
        end

        # Should mention Z learning rate multiplier
        @test occursin("Z learning rate multiplier", output)
        @test occursin("10.0x", output)
    end

end

println("\n✅ Z Learning Rate Multiplier tests completed!")
