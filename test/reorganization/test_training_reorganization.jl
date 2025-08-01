# Test for Training Code Reorganization
# Ensures all training functionality works correctly after reorganization

using Test
using GFlowNet
using Statistics
using Zygote

# Import grid world types for testing
using GFlowNet: GridState, GridAction, MoveRight, MoveUp, MoveLeft, MoveDown, Terminate
using GFlowNet: create_grid_world_gflownet, create_multi_start_gflownet

println("\n🔧 Testing Training Code Reorganization...")

@testset "Training Code Reorganization" begin
    
    @testset "Module Loading and Exports" begin
        println("  Testing module structure...")
        
        # Test that all training-related exports are available
        @test isdefined(GFlowNet, :train_gflownet)
        @test isdefined(GFlowNet, :TrainingConfig)
        @test isdefined(GFlowNet, :TrainingHistory)
        @test isdefined(GFlowNet, :TRAJECTORY_BALANCE)
        @test isdefined(GFlowNet, :DETAILED_BALANCE)
        @test isdefined(GFlowNet, :FLOW_MATCHING)
        
        # Test that interface.jl only has model creation and sampling
        @test isdefined(GFlowNet, :create_gflownet)
        @test isdefined(GFlowNet, :sample_trajectory)
        
        # Test backward policy validation exports
        @test isdefined(GFlowNet, :validate_backward_policy_normalization)
        @test isdefined(GFlowNet, :validate_backward_policy_consistency)
        @test isdefined(GFlowNet, :monitor_backward_policy_learning)
    end
    
    @testset "Training Function Location" begin
        println("  Verifying function locations...")
        
        # These should be in training/ module, not core/
        # We can't directly test file locations, but we can verify they work
        
        # Create a simple test model
        model = create_grid_world_gflownet(
            grid_size = 3,
            hidden_dim = 16,
            learning_rate = 0.01
        )
        
        # Test that training config works
        config = TrainingConfig(
            objective = TRAJECTORY_BALANCE,
            n_iterations = 2,
            batch_size = 4
        )
        
        # Should be able to train
        history = train_gflownet(model, config; verbose=false)
        @test isa(history, TrainingHistory)
        @test length(history.losses) == 2
    end
    
    @testset "Backward Policy Validation Functions" begin
        println("  Testing backward policy validation...")
        
        # Create model with backward policy
        model = create_grid_world_gflownet(
            grid_size = 3,
            hidden_dim = 16,
            include_backward = true
        )
        
        # Test state
        state = GridState(2, 2, false)
        
        # Test normalization validation
        is_valid, total_prob, parents = validate_backward_policy_normalization(
            model, state, model.all_actions
        )
        @test isa(is_valid, Bool)
        @test isa(total_prob, Float64)
        @test isa(parents, Vector)
        
        # Test consistency validation
        trajectories = [sample_trajectory(model) for _ in 1:5]
        result = validate_backward_policy_consistency(model, trajectories)
        @test haskey(result, :is_valid)
        @test haskey(result, :message)
        @test haskey(result, :stats)
        
        # Test monitoring
        validation_states = [
            GridState(1, 1, false),
            GridState(2, 1, false),
            GridState(2, 2, false)
        ]
        metrics = monitor_backward_policy_learning(model, validation_states; verbose=false)
        @test isa(metrics, Dict)
        @test haskey(metrics, "mean_normalization_error")
    end
    
    @testset "Training with All Objectives" begin
        println("  Testing all training objectives...")
        
        # Test each objective works after reorganization
        for (obj_name, objective) in [
            ("TRAJECTORY_BALANCE", TRAJECTORY_BALANCE),
            ("DETAILED_BALANCE", DETAILED_BALANCE),
            ("FLOW_MATCHING", FLOW_MATCHING)
        ]
            println("    Testing $obj_name...")
            
            # Create appropriate model
            include_backward = (objective == DETAILED_BALANCE)
            model = create_grid_world_gflownet(
                grid_size = 3,
                hidden_dim = 16,
                include_backward = include_backward
            )
            
            config = TrainingConfig(
                objective = objective,
                n_iterations = 5,
                batch_size = 4
            )
            
            # Should train without errors
            history = train_gflownet(model, config; verbose=false)
            @test length(history.losses) == 5
            @test !all(isnan, history.losses)
        end
    end
    
    @testset "Multi-Start Training Integration" begin
        println("  Testing multi-start training...")
        
        # Create multi-start model
        initial_states = [
            GridState(1, 1, false),
            GridState(2, 2, false),
            GridState(3, 1, false)
        ]
        
        # Create actions manually
        actions = [
            MoveRight(), MoveUp(), Terminate()
        ]
        
        model = create_multi_start_gflownet(
            initial_states,
            actions;
            state_dim = 2,
            hidden_dim = 16
        )
        
        # Test training works
        config = TrainingConfig(
            objective = TRAJECTORY_BALANCE,
            n_iterations = 5,
            batch_size = 6
        )
        
        history = train_gflownet(model, config; verbose=false)
        @test length(history.losses) == 5
        
        # Test initial state distribution
        probs = get_initial_state_distribution(model)
        @test length(probs) == 3
        @test sum(probs) ≈ 1.0
    end
    
    @testset "Gradient Computation After Reorganization" begin
        println("  Testing gradient computation...")
        
        model = create_grid_world_gflownet(
            grid_size = 3,
            hidden_dim = 16,
            include_backward = true,
            partition_function_method = LEARNABLE_ESTIMATION
        )
        
        # Sample trajectories
        trajectories = [sample_trajectory(model) for _ in 1:4]
        
        # Test gradient computation for each objective
        for objective in [TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING]
            config = TrainingConfig(objective = objective)
            
            # Define loss function
            loss_fn = ps -> begin
                Zygote.@ignore GFlowNet.clear_flow_cache!()
                GFlowNet.compute_trajectory_loss(model, trajectories, ps, config)
            end
            
            # Compute gradients
            loss_val, grads = Zygote.withgradient(loss_fn, model.parameters)
            
            @test !isnan(loss_val)
            @test !isnothing(grads[1])
            
            # Check specific gradient components
            if haskey(model.parameters, :log_Z)
                @test haskey(grads[1], :log_Z)
                @test !isnan(grads[1].log_Z)
            end
        end
    end
    
    @testset "Import Dependencies" begin
        println("  Checking import dependencies...")
        
        # Verify no circular dependencies by checking module can reload
        # This is implicit - if tests pass, imports are correct
        
        # Test that commonly used functions are accessible
        @test isdefined(GFlowNet, :state_to_features)
        @test isdefined(GFlowNet, :reward)
        @test isdefined(GFlowNet, :is_terminal_state)
        @test isdefined(GFlowNet, :get_applicable_actions)
        @test isdefined(GFlowNet, :apply_action)
    end
end

println("✅ Training reorganization tests completed!")