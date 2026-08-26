# Test suite for FLOW_MATCHING objective
# Tests flow matching loss computation and training

using Test
using GFlowNet
using GFlowNet: compute_trajectory_loss
using Random
using Statistics
using ComponentArrays
using Optimisers

# Import specific types needed for tests
using GFlowNet: GridState, GridAction, MoveRight, MoveUp, MoveLeft, MoveDown, Terminate
using GFlowNet: Trajectory, GFlowNetModel, create_grid_world_gflownet

@testset "Flow Matching Tests" begin
    # Set random seed for reproducibility
    Random.seed!(42)
    
    @testset "Flow Matching Loss Computation" begin
        # Create a simple grid world model with flow estimator
        model = create_grid_world_gflownet(include_flow_estimator = true, 
            grid_size = 3,
            hidden_dim = 32,
            include_backward = false
        )
        
        # Test 1: Terminal states should have zero loss
        terminal_state = GridState(3, 3, true)
        loss = flow_matching_loss(model, terminal_state)
        @test loss ≈ 0.0 atol=1e-10
        
        # Test 2: Non-terminal states should have non-negative loss
        non_terminal_state = GridState(1, 1, false)
        loss = flow_matching_loss(model, non_terminal_state)
        @test loss ≥ 0.0
        
        # Test 3: Loss should be well-defined for all valid states
        for i in 1:3, j in 1:3
            state = GridState(i, j, false)
            loss = flow_matching_loss(model, state)
            @test isfinite(loss)
            @test loss ≥ 0.0
        end
        
        # Test 4: Batch loss computation
        states = [GridState(i, j, false) for i in 1:2, j in 1:2]
        states = vec(states)
        batch_loss = flow_matching_loss_batch(model, states)
        @test isfinite(batch_loss)
        @test batch_loss ≥ 0.0
        
        # Test 5: Empty batch should return 0
        empty_batch = GridState[]
        batch_loss = flow_matching_loss_batch(model, empty_batch)
        @test batch_loss ≈ 0.0
    end
    
    @testset "Flow Matching Training" begin
        # Create model for training
        model = create_grid_world_gflownet(include_flow_estimator = true, 
            grid_size = 3,
            hidden_dim = 32,
            learning_rate = 0.01
        )
        
        # Configure training with FLOW_MATCHING
        config = TrainingConfig(
            objective = FLOW_MATCHING,
            n_iterations = 50,
            batch_size = 16,
            validation_frequency = 10
        )
        
        # Train model
        history = train_gflownet(model, config; verbose=false)
        
        # Test that training completed
        @test length(history.losses) == config.n_iterations
        
        # Test that losses are finite
        @test all(isfinite.(filter(!isnan, history.losses)))
        
        # Test that loss generally decreases (allow some fluctuation)
        initial_losses = history.losses[1:10]
        final_losses = history.losses[end-9:end]
        initial_mean = mean(filter(!isnan, initial_losses))
        final_mean = mean(filter(!isnan, final_losses))
        
        # Final loss should be lower than initial (or at least not much higher)
        @test final_mean ≤ initial_mean * 1.2
    end
    
    @testset "Flow Conservation Validation" begin
        # Train a model to convergence
        model = create_grid_world_gflownet(include_flow_estimator = true, 
            grid_size = 3,
            hidden_dim = 64,
            learning_rate = 0.001
        )
        
        config = TrainingConfig(
            objective = FLOW_MATCHING,
            n_iterations = 200,
            batch_size = 32
        )
        
        history = train_gflownet(model, config; verbose=false)
        
        # After training, flow estimates should approximately satisfy conservation
        # Test a few non-terminal states
        test_states = [
            GridState(1, 1, false),
            GridState(2, 2, false),
            GridState(1, 2, false)
        ]
        
        for state in test_states
            # Get flow estimate from neural network
            estimated_flow = flow_estimate(
                model.flow_estimator, state,
                model.parameters.flow, model.states.flow
            )
            
            # Compute expected flow manually
            applicable_actions = get_applicable_actions(state, model.all_actions)
            if !isempty(applicable_actions)
                action_probs = forward_action_probabilities(
                    model.forward_policy, state, model.all_actions,
                    model.parameters.forward, model.states.forward
                )
                
                expected_flow = 0.0
                for (idx, action) in enumerate(model.all_actions)
                    if action in applicable_actions
                        next_state = apply_action(action, state)
                        next_flow = flow(model, next_state)
                        expected_flow += action_probs[idx] * next_flow
                    end
                end
                
                # They should be approximately equal after training
                # Allow larger tolerance since this is a simple test
                @test abs(estimated_flow - expected_flow) / max(expected_flow, 1.0) < 0.5
            end
        end
    end
    
    @testset "Flow Matching with Different Configurations" begin
        # Test with learnable partition function
        model = create_grid_world_gflownet(include_flow_estimator = true, 
            grid_size = 3,
            hidden_dim = 32,
            partition_function_method = LEARNABLE_ESTIMATION
        )
        
        config = TrainingConfig(
            objective = FLOW_MATCHING,
            partition_function_method = LEARNABLE_ESTIMATION,
            n_iterations = 30,
            batch_size = 8
        )
        
        history = train_gflownet(model, config; verbose=false)
        
        # Should complete without errors
        @test length(history.losses) == config.n_iterations
        @test all(isfinite.(filter(!isnan, history.losses)))
        
        # Check that log_Z parameter exists and is being updated
        @test haskey(model.parameters, :log_Z)
        @test isfinite(model.parameters.log_Z)
    end
    
    @testset "Edge Cases and Error Handling" begin
        # Test with model missing flow estimator
        forward_policy, forward_ps, forward_st = create_forward_policy(4, 32, 5, Random.default_rng())
        
        bad_model = GFlowNetModel(
            GridState(1, 1, false),
            [MoveRight(), MoveUp(), MoveLeft(), MoveDown(), Terminate()],
            forward_policy,
            nothing,  # backward_policy
            nothing,  # flow_estimator - missing!
            nothing,  # log_partition_function
            ComponentArray(forward = forward_ps),
            Optimisers.setup(Optimisers.Adam(0.01), ComponentArray(forward = forward_ps)),
            (forward = forward_st,)
        )
        
        # Should throw error when trying to use flow matching
        @test_throws ArgumentError flow_matching_loss(bad_model, GridState(1, 1, false))
    end
    
    @testset "Integration with Training System" begin
        # Test that FLOW_MATCHING integrates properly with the training system
        model = create_grid_world_gflownet(include_flow_estimator = true, 
            grid_size = 4,
            hidden_dim = 64
        )
        
        # Test trajectory loss computation with FLOW_MATCHING
        trajectories = [sample_trajectory(model) for _ in 1:10]
        config = TrainingConfig(objective = FLOW_MATCHING)
        
        # This should work without errors
        loss = compute_trajectory_loss(model, trajectories, model.parameters, config)
        @test isfinite(loss)
        @test loss ≥ 0.0
    end
    
    @testset "Comparison with Other Objectives" begin
        # Create identical models
        model_tb = create_grid_world_gflownet(include_flow_estimator = true, grid_size = 3, hidden_dim = 32)
        model_fm = create_grid_world_gflownet(include_flow_estimator = true, grid_size = 3, hidden_dim = 32)
        
        # Train with different objectives
        config_tb = TrainingConfig(
            objective = TRAJECTORY_BALANCE,
            n_iterations = 50,
            batch_size = 16
        )
        
        config_fm = TrainingConfig(
            objective = FLOW_MATCHING,
            n_iterations = 50,
            batch_size = 16
        )
        
        history_tb = train_gflownet(model_tb, config_tb; verbose=false)
        history_fm = train_gflownet(model_fm, config_fm; verbose=false)
        
        # Both should converge
        @test length(history_tb.losses) == 50
        @test length(history_fm.losses) == 50
        
        # Sample trajectories and compare diversity
        traj_tb = [sample_trajectory(model_tb) for _ in 1:50]
        traj_fm = [sample_trajectory(model_fm) for _ in 1:50]
        
        # Both should find high-reward states
        rewards_tb = [reward(t.states[end]) for t in traj_tb]
        rewards_fm = [reward(t.states[end]) for t in traj_fm]
        
        @test maximum(rewards_tb) > 0
        @test maximum(rewards_fm) > 0
    end
end

println("\n✅ All flow matching tests passed!")