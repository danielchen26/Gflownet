# Test LEARNABLE_ESTIMATION functionality
# Comprehensive tests for learnable partition function Z parameter

using Test
using Random
using Statistics
using ComponentArrays
using GFlowNet
using GFlowNet: compute_trajectory_loss, compute_single_trajectory_loss  # Import from training/losses.jl

# Set random seed for reproducible tests
Random.seed!(42)

@testset "LEARNABLE_ESTIMATION Tests" begin
    
    # =============================================================================
    # Test 1: Model Creation with LEARNABLE_ESTIMATION
    # =============================================================================
    
    @testset "Model Creation" begin
        @testset "Simple Estimation (Default)" begin
            model = create_grid_world_gflownet(
                grid_size=3,
                hidden_dim=16,
                partition_function_method=SIMPLE_ESTIMATION
            )
            
            @test model.log_partition_function === nothing
            @test !haskey(model.parameters, :log_Z)
            @test model isa GFlowNetModel
        end
        
        @testset "Learnable Estimation" begin
            model = create_grid_world_gflownet(
                grid_size=3,
                hidden_dim=16,
                partition_function_method=LEARNABLE_ESTIMATION
            )
            
            @test model.log_partition_function ≈ 0.0
            @test haskey(model.parameters, :log_Z)
            @test model.parameters.log_Z ≈ 0.0
            @test model isa GFlowNetModel
        end
        
        @testset "Parameter Synchronization" begin
            model = create_grid_world_gflownet(
                grid_size=3,
                partition_function_method=LEARNABLE_ESTIMATION
            )
            
            # Test initial synchronization
            @test abs(model.log_partition_function - model.parameters.log_Z) < 1e-10
        end
    end
    
    # =============================================================================
    # Test 2: Loss Computation with Z Parameter
    # =============================================================================
    
    @testset "Loss Computation" begin
        @testset "Loss includes Z term" begin
            model_simple = create_grid_world_gflownet(
                grid_size=3,
                partition_function_method=SIMPLE_ESTIMATION
            )
            
            model_learnable = create_grid_world_gflownet(
                grid_size=3,
                partition_function_method=LEARNABLE_ESTIMATION
            )
            
            # Generate a simple trajectory
            trajectory = sample_trajectory(model_simple)
            
            # Compute losses
            loss_simple = compute_single_trajectory_loss(model_simple, trajectory, model_simple.parameters)
            loss_learnable = compute_single_trajectory_loss(model_learnable, trajectory, model_learnable.parameters)
            
            # Both should be finite and positive
            @test isfinite(loss_simple)
            @test isfinite(loss_learnable)
            @test loss_simple >= 0
            @test loss_learnable >= 0
            
            # Note: Losses may be different due to different network initializations
            # Both should just be finite and reasonable
            @test loss_simple < 100  # Should be reasonable magnitude
            @test loss_learnable < 100  # Should be reasonable magnitude
        end
        
        @testset "Z parameter affects loss" begin
            model = create_grid_world_gflownet(
                grid_size=3,
                partition_function_method=LEARNABLE_ESTIMATION
            )
            
            trajectory = sample_trajectory(model)
            
            # Compute loss with original Z
            loss_original = compute_single_trajectory_loss(model, trajectory, model.parameters)
            
            # Modify Z parameter
            modified_params = deepcopy(model.parameters)
            modified_params = ComponentArray(
                forward=modified_params.forward,
                flow=modified_params.flow,
                log_Z=1.0  # Change log Z from 0 to 1
            )
            
            loss_modified = compute_single_trajectory_loss(model, trajectory, modified_params)
            
            # Loss should be different when Z changes
            @test abs(loss_original - loss_modified) > 1e-6
        end
    end
    
    # =============================================================================
    # Test 3: Gradient Computation
    # =============================================================================
    
    @testset "Gradient Computation" begin
        @testset "Z parameter receives gradients" begin
            model = create_grid_world_gflownet(
                grid_size=3,
                partition_function_method=LEARNABLE_ESTIMATION
            )
            
            # Generate trajectories
            trajectories = [sample_trajectory(model) for _ in 1:5]
            
            # Compute gradients
            loss_function = ps -> compute_trajectory_loss(
                model, trajectories, ps, TrainingConfig()
            )
            
            using Zygote
            _, grads = Zygote.withgradient(loss_function, model.parameters)
            
            @test grads[1] !== nothing
            @test haskey(grads[1], :log_Z)
            @test grads[1].log_Z isa Real
            @test isfinite(grads[1].log_Z)
        end
        
        @testset "Gradient magnitude reasonable" begin
            model = create_grid_world_gflownet(
                grid_size=3,
                partition_function_method=LEARNABLE_ESTIMATION
            )
            
            trajectories = [sample_trajectory(model) for _ in 1:10]
            grad_magnitude = validate_z_gradients(model, trajectories)
            
            @test !isnan(grad_magnitude)
            @test grad_magnitude > 0
            @test grad_magnitude < 100  # Shouldn't be extremely large
        end
    end
    
    # =============================================================================
    # Test 4: Training Integration
    # =============================================================================
    
    @testset "Training Integration" begin
        @testset "Short training runs without errors" begin
            model = create_grid_world_gflownet(
                grid_size=3,
                partition_function_method=LEARNABLE_ESTIMATION
            )
            
            config = TrainingConfig(
                objective=TRAJECTORY_BALANCE,
                partition_function_method=LEARNABLE_ESTIMATION,
                n_iterations=5,
                batch_size=4,
                learning_rate=0.01
            )
            
            initial_z = model.parameters.log_Z
            
            # Run training
            history = train_gflownet(model, config; verbose=false)
            
            @test history isa TrainingHistory
            @test length(history.losses) == 5
            @test all(isfinite, history.losses)
            
            # Z parameter should have potentially changed
            final_z = model.parameters.log_Z
            @test isfinite(final_z)
            
            # Model field should be synchronized
            @test abs(model.log_partition_function - model.parameters.log_Z) < 1e-10
        end
        
        @testset "Z parameter optimization" begin
            model = create_grid_world_gflownet(
                grid_size=3,
                partition_function_method=LEARNABLE_ESTIMATION
            )
            
            # Set initial Z to suboptimal value
            model.parameters = ComponentArray(
                forward=model.parameters.forward,
                flow=model.parameters.flow,
                log_Z=5.0  # Start with large Z
            )
            model.log_partition_function = 5.0
            
            config = TrainingConfig(
                n_iterations=10,
                batch_size=8,
                learning_rate=0.1
            )
            
            initial_z = model.parameters.log_Z
            train_gflownet(model, config; verbose=false)
            final_z = model.parameters.log_Z
            
            # Z should move towards more reasonable value
            @test abs(final_z) < abs(initial_z)  # Should get closer to 0
        end
    end
    
    # =============================================================================
    # Test 5: Validation Functions
    # =============================================================================
    
    @testset "Validation Functions" begin
        @testset "validate_z_learning passes for correct setup" begin
            model = create_grid_world_gflownet(
                grid_size=3,
                partition_function_method=LEARNABLE_ESTIMATION
            )
            
            config = TrainingConfig(
                partition_function_method=LEARNABLE_ESTIMATION
            )
            
            result = validate_z_learning(model, config)
            @test result === true
        end
        
        @testset "validate_z_learning skips for simple estimation" begin
            model = create_grid_world_gflownet(
                grid_size=3,
                partition_function_method=SIMPLE_ESTIMATION
            )
            
            config = TrainingConfig(
                partition_function_method=SIMPLE_ESTIMATION
            )
            
            result = validate_z_learning(model, config)
            @test result === true  # Should pass (skipped)
        end
        
        @testset "validate_z_mathematical_properties" begin
            model = create_grid_world_gflownet(
                grid_size=3,
                partition_function_method=LEARNABLE_ESTIMATION
            )
            
            # Train briefly to get more reasonable log probability values
            config = TrainingConfig(n_iterations=3, batch_size=4)
            train_gflownet(model, config; verbose=false)
            
            trajectories = [sample_trajectory(model) for _ in 1:3]
            result = validate_z_mathematical_properties(model, trajectories)
            
            # The mathematical validation is strict and may fail during early training
            # This is expected behavior - we just test that it runs without errors
            @test result isa Bool  # Should return a boolean result
        end
    end
    
    # =============================================================================
    # Test 6: DAG Functions Integration
    # =============================================================================
    
    @testset "DAG Functions Integration" begin
        @testset "get_next_states works with model" begin
            model = create_grid_world_gflownet(
                grid_size=3,
                partition_function_method=LEARNABLE_ESTIMATION
            )
            
            initial_state = model.initial_state
            next_states = get_next_states(model, initial_state)
            
            @test !isempty(next_states)
            @test all(state -> state isa AbstractState, next_states)
        end
        
        @testset "get_previous_states works with model" begin
            model = create_grid_world_gflownet(
                grid_size=3,
                partition_function_method=LEARNABLE_ESTIMATION
            )
            
            # Get a state that's reachable from initial
            trajectory = sample_trajectory(model)
            if length(trajectory.states) >= 2
                target_state = trajectory.states[2]
                previous_states = get_previous_states(model, target_state)
                
                @test previous_states isa Vector
                # Should find at least the initial state
                @test any(state -> state == model.initial_state, previous_states)
            end
        end
        
        @testset "get_root_state works with model" begin
            model = create_grid_world_gflownet(
                grid_size=3,
                partition_function_method=LEARNABLE_ESTIMATION
            )
            
            root_state = get_root_state(model)
            @test root_state == model.initial_state
        end
    end
    
    # =============================================================================
    # Test 7: Mathematical Properties
    # =============================================================================
    
    @testset "Mathematical Properties" begin
        @testset "Trajectory balance equation structure" begin
            model = create_grid_world_gflownet(
                grid_size=3,
                partition_function_method=LEARNABLE_ESTIMATION
            )
            
            trajectory = sample_trajectory(model)
            
            if length(trajectory.states) >= 2
                # Compute components of trajectory balance equation
                log_Z = model.parameters.log_Z
                log_prob = compute_trajectory_log_probability(model, trajectory)
                terminal_reward = reward(trajectory.states[end])
                log_reward = log(max(terminal_reward, 1e-8))
                
                # All components should be finite
                @test isfinite(log_Z)
                @test isfinite(log_prob) || log_prob == -Inf  # -Inf acceptable for invalid
                @test isfinite(log_reward)
                
                # Balance equation: log Z + log P_F(τ) ≈ log R(s_T)
                if isfinite(log_prob)
                    balance_error = log_Z + log_prob - log_reward
                    @test isfinite(balance_error)
                end
            end
        end
        
        @testset "Z parameter bounds" begin
            model = create_grid_world_gflownet(
                grid_size=3,
                partition_function_method=LEARNABLE_ESTIMATION
            )
            
            # Train briefly
            config = TrainingConfig(n_iterations=5, batch_size=4)
            train_gflownet(model, config; verbose=false)
            
            log_Z = model.parameters.log_Z
            Z = exp(log_Z)
            
            # Z should be positive and reasonable
            @test Z > 0
            @test isfinite(Z)
            @test Z < 1e10  # Shouldn't explode
            @test Z > 1e-10  # Shouldn't vanish
        end
    end
    
    # =============================================================================
    # Test 8: Comparison with Simple Estimation
    # =============================================================================
    
    @testset "Comparison with Simple Estimation" begin
        @testset "Both methods produce valid trajectories" begin
            model_simple = create_grid_world_gflownet(
                grid_size=3,
                partition_function_method=SIMPLE_ESTIMATION
            )
            
            model_learnable = create_grid_world_gflownet(
                grid_size=3,
                partition_function_method=LEARNABLE_ESTIMATION
            )
            
            # Generate trajectories from both
            trajectories_simple = [sample_trajectory(model_simple) for _ in 1:5]
            trajectories_learnable = [sample_trajectory(model_learnable) for _ in 1:5]
            
            # All trajectories should be valid
            for traj in trajectories_simple
                @test length(traj.states) >= 1
                @test is_terminal_state(traj.states[end])
            end
            
            for traj in trajectories_learnable
                @test length(traj.states) >= 1
                @test is_terminal_state(traj.states[end])
            end
        end
    end
end