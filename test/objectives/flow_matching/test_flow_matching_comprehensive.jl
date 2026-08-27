# Comprehensive tests for FLOW_MATCHING objective
# Including integration with training system and comparison with other objectives

using Test
using GFlowNet
using Random
using Statistics
using Plots

# Import specific types needed
using GFlowNet: GridState, create_grid_world_gflownet
using GFlowNet: flow, flow_estimate, flow_matching_loss

@testset "Flow Matching Comprehensive Tests" begin
    Random.seed!(42)
    
    @testset "Mathematical Properties" begin
        # Test that flow matching loss has correct mathematical properties
        model = create_grid_world_gflownet(include_flow_estimator = true, 
            grid_size = 4,
            hidden_dim = 32
        )
        
        # Property 1: Loss is 0 when Z(s) = F(s)
        state = GridState(2, 2, false)
        
        # Manually set flow estimator to return true flow
        true_flow = flow(model, state)
        
        # This would be 0 if we could set the neural network output
        # In practice, we test that loss is non-negative
        loss = flow_matching_loss(model, state)
        @test loss ≥ 0
        
        # Property 2: Loss is quadratic in the difference
        # Test by training and checking convergence
    end
    
    @testset "Convergence Analysis" begin
        # Test that FLOW_MATCHING converges to correct flow values
        model = create_grid_world_gflownet(include_flow_estimator = true, 
            grid_size = 3,
            hidden_dim = 64,
            learning_rate = 0.001
        )
        
        config = TrainingConfig(
            objective = FLOW_MATCHING,
            n_iterations = 500,
            batch_size = 32,
            validation_frequency = 50
        )
        
        # Track flow estimates during training
        test_states = [
            GridState(1, 1, false),
            GridState(2, 2, false),
            GridState(1, 3, false)
        ]
        
        flow_history = Dict(state => Float64[] for state in test_states)
        
        # Custom callback to track flow estimates
        function track_flows(model, history, iteration)
            if iteration % 50 == 0
                for state in test_states
                    est_flow = flow_estimate(
                        model.flow_estimator, state,
                        model.parameters.flow, model.states.flow
                    )
                    push!(flow_history[state], est_flow)
                end
            end
        end
        
        # Train with callback
        history = train_gflownet(model, config; verbose=false)
        
        # Manually track final flows
        for state in test_states
            final_flow = flow_estimate(
                model.flow_estimator, state,
                model.parameters.flow, model.states.flow
            )
            true_flow = flow(model, state)
            
            # Should converge close to true flow
            @test abs(final_flow - true_flow) / max(true_flow, 1.0) < 0.3
        end
    end
    
    @testset "Comparison with Other Objectives" begin
        # Compare FLOW_MATCHING with TB and DB
        models = Dict()
        histories = Dict()
        
        for (name, objective, include_backward) in [
            ("TB", TRAJECTORY_BALANCE, false),
            ("FM", FLOW_MATCHING, false),
            ("DB", DETAILED_BALANCE, true)
        ]
            # TB REQUIRES a learnable Z. Under the default SIMPLE_ESTIMATION,
            # Z is fixed at 1 so log Z = 0, and the TB residual
            #   (log Z + sum log P_F - log R - sum log P_B)^2
            # cannot reach zero unless Z_true = 1. The objective is then
            # unsatisfiable by construction -- as src/utils/visualization/api/
            # unified_server.jl:154 already documents. Measured on this exact
            # 4x4 setup: TB sat at 21.17 after 200 iterations and got WORSE at
            # 1000 (25.46), because sharpening the policy only grows the
            # residual, while FM and DB both reached about 0.0. That was a
            # broken model, not a broken objective.
            model = create_grid_world_gflownet(include_flow_estimator = true,
                grid_size = 4,
                hidden_dim = 64,
                include_backward = include_backward,
                partition_function_method = objective == TRAJECTORY_BALANCE ?
                    LEARNABLE_ESTIMATION : SIMPLE_ESTIMATION
            )
            
            config = TrainingConfig(
                objective = objective,
                n_iterations = 200,
                batch_size = 32
            )
            
            history = train_gflownet(model, config; verbose=false)
            
            models[name] = model
            histories[name] = history
        end
        
        # All should achieve reasonable performance
        for name in ["TB", "FM", "DB"]
            final_losses = histories[name].losses[end-19:end]
            avg_final_loss = mean(filter(!isnan, final_losses))
            @test avg_final_loss < 1.0  # Reasonable convergence
            
            # Sample trajectories
            trajectories = [sample_trajectory(models[name]) for _ in 1:50]
            rewards = [reward(t.states[end]) for t in trajectories]
            
            # Should find high-reward states
            @test maximum(rewards) > 0
            @test mean(rewards) > 0.1
        end
    end
    
    @testset "Flow Conservation Verification" begin
        # After training, verify flow conservation is satisfied
        model = create_grid_world_gflownet(include_flow_estimator = true, 
            grid_size = 3,
            hidden_dim = 128,
            learning_rate = 0.0005
        )
        
        config = TrainingConfig(
            objective = FLOW_MATCHING,
            n_iterations = 1000,
            batch_size = 64
        )
        
        history = train_gflownet(model, config; verbose=false)
        
        # Check flow conservation for all non-terminal states
        conservation_errors = Float64[]
        
        for i in 1:3, j in 1:3
            state = GridState(i, j, false)
            
            # Skip if no applicable actions
            applicable_actions = get_applicable_actions(state, model.all_actions)
            if isempty(applicable_actions)
                continue
            end
            
            # Compute LHS: Z(s) (neural network estimate)
            lhs = flow_estimate(
                model.flow_estimator, state,
                model.parameters.flow, model.states.flow
            )
            
            # Compute RHS under the CORRECTED conservation law: the children are
            # weighted by P_B(s|s'), not by P_F(s'|s). The P_F-weighted form is a
            # convex combination, which pins F(s0) between min and max R instead
            # of equalling sum_x R(x) -- the defect that made partition_function
            # return 2.2184 rather than 19.0 on the 3x3 grid. This testset was
            # asserting that defect.
            rhs = 0.0
            for action in applicable_actions
                child = apply_action(action, state)
                back_prob = if isnothing(model.backward_policy)
                    parents = GFlowNet.backward_parent_states(child, model.all_actions)
                    isempty(parents) ? 0.0 : 1.0 / length(parents)
                else
                    GFlowNet.compute_backward_probability(
                        model.backward_policy, child, state,
                        model.parameters.backward, model.states.backward,
                        model.all_actions)
                end
                rhs += back_prob * flow(model, child)
            end

            error = abs(lhs - rhs) / max(rhs, 1.0)
            push!(conservation_errors, error)
        end
        
        # Average conservation error should be small
        @test mean(conservation_errors) < 0.2
        @test maximum(conservation_errors) < 0.5
    end
    
    @testset "Edge Cases and Robustness" begin
        # Test with very small grid
        model_small = create_grid_world_gflownet(include_flow_estimator = true, 
            grid_size = 2,
            hidden_dim = 16
        )
        
        config = TrainingConfig(
            objective = FLOW_MATCHING,
            n_iterations = 50,
            batch_size = 8
        )
        
        history = train_gflownet(model_small, config; verbose=false)
        @test length(history.losses) == 50
        @test all(isfinite.(filter(!isnan, history.losses)))
        
        # Test with very large hidden dimension
        model_large = create_grid_world_gflownet(include_flow_estimator = true, 
            grid_size = 3,
            hidden_dim = 256
        )
        
        config_large = TrainingConfig(
            objective = FLOW_MATCHING,
            n_iterations = 30,
            batch_size = 16
        )
        
        history_large = train_gflownet(model_large, config_large; verbose=false)
        @test length(history_large.losses) == 30
    end
    
    @testset "Integration with Learnable Z" begin
        # Test FLOW_MATCHING with learnable partition function
        model = create_grid_world_gflownet(include_flow_estimator = true, 
            grid_size = 3,
            hidden_dim = 64,
            partition_function_method = LEARNABLE_ESTIMATION
        )
        
        config = TrainingConfig(
            objective = FLOW_MATCHING,
            partition_function_method = LEARNABLE_ESTIMATION,
            n_iterations = 100,
            batch_size = 32
        )
        
        initial_log_Z = model.parameters.log_Z
        history = train_gflownet(model, config; verbose=false)
        final_log_Z = model.parameters.log_Z
        
        # log Z MUST NOT change under FLOW_MATCHING. The symbol log_Z appears
        # nowhere in the flow-matching loss -- FM's unknowns are the state flows
        # and the policy, and F(s0) already plays the role of Z -- so its
        # gradient is exactly zero and no correct optimiser can move it.
        # Asserting that Z "should change during training" asserted a fiction:
        # it was evaluated as `0.0 != 0.0` and had been failing continuously.
        # Measured gradient norm for the :log_Z component under FM: exactly 0.
        @test initial_log_Z == final_log_Z
        @test isfinite(final_log_Z)
        
        # Should still achieve good performance
        final_losses = history.losses[end-9:end]
        @test mean(filter(!isnan, final_losses)) < 1.0
    end
end

println("\n✅ All comprehensive flow matching tests passed!")