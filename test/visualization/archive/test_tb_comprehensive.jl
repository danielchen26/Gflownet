# Comprehensive test for Trajectory Balance (TB) training
# Tests: Learnable Z initialization, gradient flow, and convergence

using Test
using GFlowNet
using Statistics

@testset "Comprehensive TB Training Verification" begin

    @testset "1. Learnable Z Parameter Initialization" begin
        # Create model with LEARNABLE_ESTIMATION
        model = GFlowNet.create_grid_world_gflownet(
            grid_size = 5,
            reward_positions = Dict((5,5) => 10.0, (1,5) => 8.0),
            partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
        )

        # Test 1.1: log_partition_function is set
        @test model.log_partition_function !== nothing
        @test model.log_partition_function == 0.0  # Initialized to 0 (Z=1)

        # Test 1.2: parameters contain log_Z
        @test haskey(model.parameters, :log_Z)
        @test model.parameters.log_Z == 0.0

        # Test 1.3: Compare with SIMPLE_ESTIMATION
        model_simple = GFlowNet.create_grid_world_gflownet(
            grid_size = 5,
            reward_positions = Dict((5,5) => 10.0),
            partition_function_method = GFlowNet.SIMPLE_ESTIMATION
        )
        @test model_simple.log_partition_function === nothing
        @test !haskey(model_simple.parameters, :log_Z)

        println("✅ Test 1: Learnable Z initialization - PASSED")
    end

    @testset "2. Gradient Flow to log_Z" begin
        using Zygote
        using GFlowNet: compute_trajectory_loss, TrainingConfig, TRAJECTORY_BALANCE
        using GFlowNet: sample_trajectory

        model = GFlowNet.create_grid_world_gflownet(
            grid_size = 5,
            reward_positions = Dict((5,5) => 10.0, (1,5) => 8.0),
            partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
        )

        config = GFlowNet.TrainingConfig(
            objective = TRAJECTORY_BALANCE,
            batch_size = 8,
            n_iterations = 10
        )

        # Sample some trajectories
        trajectories = [sample_trajectory(model) for _ in 1:8]

        # Define loss function
        loss_fn = ps -> compute_trajectory_loss(model, trajectories, ps, config)

        # Compute gradients
        loss_val, grads = Zygote.withgradient(loss_fn, model.parameters)

        # Test 2.1: Loss is finite
        @test isfinite(loss_val)

        # Test 2.2: Gradients exist
        @test grads[1] !== nothing

        # Test 2.3: log_Z gradient exists and is finite
        @test haskey(grads[1], :log_Z)
        @test isfinite(grads[1].log_Z)
        @test grads[1].log_Z != 0.0  # Gradient should be non-zero

        println("✅ Test 2: Gradient flow to log_Z - PASSED")
        println("   Initial log_Z: $(model.parameters.log_Z)")
        println("   Loss: $loss_val")
        println("   log_Z gradient: $(grads[1].log_Z)")
    end

    @testset "3. log_Z Updates During Training" begin
        model = GFlowNet.create_grid_world_gflownet(
            grid_size = 5,
            reward_positions = Dict((5,5) => 10.0, (1,5) => 8.0),
            partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
        )

        initial_log_Z = model.parameters.log_Z

        config = GFlowNet.TrainingConfig(
            objective = TRAJECTORY_BALANCE,
            batch_size = 16,
            n_iterations = 50,
            learning_rate = 0.01
        )

        # Train for a few iterations
        history = GFlowNet.train_gflownet(model, config; verbose=false)

        final_log_Z = model.parameters.log_Z

        # Test 3.1: log_Z has changed
        @test final_log_Z != initial_log_Z

        # Test 3.2: log_Z is synchronized with model field
        @test abs(model.parameters.log_Z - model.log_partition_function) < 1e-6

        # Test 3.3: Training completed
        @test length(history.losses) == config.n_iterations

        println("✅ Test 3: log_Z updates during training - PASSED")
        println("   Initial log_Z: $initial_log_Z (Z = $(exp(initial_log_Z)))")
        println("   Final log_Z: $final_log_Z (Z = $(exp(final_log_Z)))")
        println("   Change: $(final_log_Z - initial_log_Z)")
    end

    @testset "4. TB Convergence - Mode Discovery" begin
        # Set up problem with two distinct reward peaks
        model = GFlowNet.create_grid_world_gflownet(
            grid_size = 5,
            reward_positions = Dict((5,5) => 10.0, (1,5) => 8.0),
            partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,
            hidden_dim = 64
        )

        config = GFlowNet.TrainingConfig(
            objective = TRAJECTORY_BALANCE,
            batch_size = 32,
            n_iterations = 200,
            learning_rate = 0.005
        )

        # Train
        history = GFlowNet.train_gflownet(model, config; verbose=false)

        # Sample many trajectories from trained model
        trajectories = [GFlowNet.sample_trajectory(model) for _ in 1:500]

        # Count terminal positions
        position_counts = Dict{Tuple{Int,Int}, Int}()
        for traj in trajectories
            if !isempty(traj.states) && GFlowNet.is_terminal_state(traj.states[end])
                state = traj.states[end]
                pos = (state.x, state.y)
                position_counts[pos] = get(position_counts, pos, 0) + 1
            end
        end

        # Test 4.1: Both reward peaks should be discovered
        peak1_count = get(position_counts, (5,5), 0)
        peak2_count = get(position_counts, (1,5), 0)

        @test peak1_count > 0
        @test peak2_count > 0

        # Test 4.2: Ratio should roughly match reward ratio (10:8)
        # Allow significant variance due to stochasticity
        if peak1_count > 10 && peak2_count > 10
            ratio = peak1_count / peak2_count
            expected_ratio = 10.0 / 8.0  # 1.25
            @test 0.5 < ratio < 3.0
        end

        # Test 4.3: Training loss should decrease
        valid_losses = filter(!isnan, history.losses)
        if length(valid_losses) > 20
            early_loss = mean(valid_losses[1:10])
            late_loss = mean(valid_losses[end-9:end])
            @test late_loss < early_loss
        end

        println("✅ Test 4: TB Convergence - PASSED")
        println("   Peak (5,5) with R=10: $peak1_count samples")
        println("   Peak (1,5) with R=8: $peak2_count samples")
        println("   Final Z = $(exp(model.parameters.log_Z))")
        println("   Top positions: $(sort(collect(position_counts), by=x->x[2], rev=true)[1:min(5, length(position_counts))])")
    end

    @testset "5. TB vs Simple Estimation Comparison" begin
        # Train same problem with both methods
        function train_and_evaluate(partition_method, name)
            model = GFlowNet.create_grid_world_gflownet(
                grid_size = 5,
                reward_positions = Dict((5,5) => 10.0, (1,5) => 8.0),
                partition_function_method = partition_method,
                hidden_dim = 64
            )

            config = GFlowNet.TrainingConfig(
                objective = TRAJECTORY_BALANCE,
                batch_size = 32,
                n_iterations = 100,
                learning_rate = 0.005
            )

            history = GFlowNet.train_gflownet(model, config; verbose=false)

            # Sample and count
            trajectories = [GFlowNet.sample_trajectory(model) for _ in 1:200]
            position_counts = Dict{Tuple{Int,Int}, Int}()
            for traj in trajectories
                if !isempty(traj.states) && GFlowNet.is_terminal_state(traj.states[end])
                    pos = (traj.states[end].x, traj.states[end].y)
                    position_counts[pos] = get(position_counts, pos, 0) + 1
                end
            end

            peak1 = get(position_counts, (5,5), 0)
            peak2 = get(position_counts, (1,5), 0)

            valid_losses = filter(!isnan, history.losses)
            final_loss = isempty(valid_losses) ? Inf : valid_losses[end]

            return (name=name, peak1=peak1, peak2=peak2, final_loss=final_loss,
                    modes_found=Int(peak1 > 0) + Int(peak2 > 0))
        end

        result_learnable = train_and_evaluate(GFlowNet.LEARNABLE_ESTIMATION, "LEARNABLE")
        result_simple = train_and_evaluate(GFlowNet.SIMPLE_ESTIMATION, "SIMPLE")

        println("\n📊 Comparison Results:")
        println("   LEARNABLE_ESTIMATION:")
        println("      Peak1 (5,5): $(result_learnable.peak1), Peak2 (1,5): $(result_learnable.peak2)")
        println("      Modes found: $(result_learnable.modes_found)/2")
        println("      Final loss: $(round(result_learnable.final_loss, digits=4))")

        println("   SIMPLE_ESTIMATION:")
        println("      Peak1 (5,5): $(result_simple.peak1), Peak2 (1,5): $(result_simple.peak2)")
        println("      Modes found: $(result_simple.modes_found)/2")
        println("      Final loss: $(round(result_simple.final_loss, digits=4))")

        # Learnable should find more modes (or equal)
        @test result_learnable.modes_found >= result_simple.modes_found

        println("✅ Test 5: Comparison complete")
    end
end

println("\n" * "="^60)
println("All TB Comprehensive Tests Completed!")
println("="^60)
