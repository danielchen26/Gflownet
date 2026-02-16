# Test DIRECT_FLOW_OBJECTIVE training objective
# Domain-agnostic test using generic state/action types

using Test
using GFlowNet
using Statistics

# Define generic test domain
struct TestState <: AbstractState
    value::Int
    is_terminal::Bool
end

struct TestAction <: AbstractAction
    delta::Int
end

# Implement required GFlowNet interface
function GFlowNet.state_to_features(state::TestState)::Vector{Float32}
    return Float32[
        state.value / 10.0,  # Normalize value
        state.is_terminal ? 1.0 : 0.0
    ]
end

function GFlowNet.is_terminal_state(state::TestState)::Bool
    return state.is_terminal
end

function GFlowNet.reward(state::TestState)::Float64
    if !state.is_terminal
        return 0.0
    end
    # Reward function: prefer values around 5
    return exp(-0.5 * (state.value - 5)^2)
end

function GFlowNet.is_applicable(action::TestAction, state::TestState)::Bool
    return !state.is_terminal && state.value + action.delta >= 0 && state.value + action.delta <= 10
end

function GFlowNet.apply_action(action::TestAction, state::TestState)::TestState
    new_value = state.value + action.delta
    # Terminate when value reaches certain thresholds
    is_terminal = new_value >= 4 && new_value <= 6
    return TestState(new_value, is_terminal)
end

# Define equality and hashing
Base.:(==)(a::TestState, b::TestState) = a.value == b.value && a.is_terminal == b.is_terminal
Base.hash(s::TestState, h::UInt) = hash((s.value, s.is_terminal), h)
Base.:(==)(a::TestAction, b::TestAction) = a.delta == b.delta
Base.hash(a::TestAction, h::UInt) = hash(a.delta, h)

@testset "DIRECT_FLOW_OBJECTIVE Training Tests" begin
    
    @testset "Basic DIRECT_FLOW functionality" begin
        # Create domain
        initial_state = TestState(0, false)
        all_actions = [TestAction(i) for i in [1, 2, 3]]
        
        # Create model with flow estimator
        model = create_gflownet(
            initial_state,
            all_actions;
            state_dim = 2,
            hidden_dim = 32,
            learning_rate = 0.01,
            include_flow_estimator = true  # Required for DIRECT_FLOW
        )
        
        # Sample a trajectory
        trajectory = sample_trajectory(model)
        
        # Test direct flow loss computation
        loss = GFlowNet.direct_flow_loss(model, trajectory)
        @test loss >= 0.0
        @test !isnan(loss)
        @test !isinf(loss)
        
        # Test batch computation
        trajectories = [sample_trajectory(model) for _ in 1:5]
        batch_loss = GFlowNet.direct_flow_loss_batch(model, trajectories)
        @test batch_loss >= 0.0
        @test !isnan(batch_loss)
    end
    
    @testset "DIRECT_FLOW training integration" begin
        # Create domain
        initial_state = TestState(0, false)
        all_actions = [TestAction(i) for i in [1, 2]]
        
        # Create model with flow estimator
        model = create_gflownet(
            initial_state,
            all_actions;
            state_dim = 2,
            hidden_dim = 32,
            learning_rate = 0.01,
            include_flow_estimator = true
        )
        
        # Configure training for DIRECT_FLOW
        config = TrainingConfig(
            objective = DIRECT_FLOW_OBJECTIVE,
            n_iterations = 50,
            batch_size = 16,
            verbose = false
        )
        
        # Train model
        history = train_gflownet(model, config)
        
        # Check training completed
        @test length(history.losses) == config.n_iterations
        
        # Check losses are reasonable
        finite_losses = filter(!isnan, history.losses)
        @test !isempty(finite_losses)
        
        # Check loss decreased (on average)
        if length(finite_losses) > 10
            early_loss = mean(finite_losses[1:5])
            late_loss = mean(finite_losses[end-4:end])
            @test late_loss <= early_loss * 1.5  # Allow some tolerance
        end
    end
    
    @testset "DIRECT_FLOW vs TRAJECTORY_BALANCE comparison" begin
        # Create domain
        initial_state = TestState(0, false)
        all_actions = [TestAction(i) for i in [1, 2]]
        
        # Train with TRAJECTORY_BALANCE
        model_tb = create_gflownet(
            initial_state,
            all_actions;
            state_dim = 2,
            hidden_dim = 32,
            learning_rate = 0.01,
            include_flow_estimator = false  # Not needed for TB
        )
        
        config_tb = TrainingConfig(
            objective = TRAJECTORY_BALANCE,
            n_iterations = 50,
            batch_size = 16,
            verbose = false
        )
        
        history_tb = train_gflownet(model_tb, config_tb)
        
        # Train with DIRECT_FLOW
        model_df = create_gflownet(
            initial_state,
            all_actions;
            state_dim = 2,
            hidden_dim = 32,
            learning_rate = 0.01,
            include_flow_estimator = true  # Required for DIRECT_FLOW
        )
        
        config_df = TrainingConfig(
            objective = DIRECT_FLOW_OBJECTIVE,
            n_iterations = 50,
            batch_size = 16,
            verbose = false
        )
        
        history_df = train_gflownet(model_df, config_df)
        
        # Both should achieve reasonable performance
        @test !isempty(filter(!isnan, history_tb.losses))
        @test !isempty(filter(!isnan, history_df.losses))
        
        # Sample and analyze terminal state distributions
        function analyze_terminal_distribution(model, n_samples=100)
            trajectories = [sample_trajectory(model) for _ in 1:n_samples]
            terminal_values = [t.states[end].value for t in trajectories]
            return terminal_values
        end
        
        tb_values = analyze_terminal_distribution(model_tb)
        df_values = analyze_terminal_distribution(model_df)
        
        # Both should find the high-reward region (values around 5)
        @test 3 <= mean(tb_values) <= 7
        @test 3 <= mean(df_values) <= 7
    end
    
    @testset "Flow estimator consistency" begin
        # Create model with flow estimator
        initial_state = TestState(0, false)
        all_actions = [TestAction(i) for i in [1, 2]]
        
        model = create_gflownet(
            initial_state,
            all_actions;
            state_dim = 2,
            hidden_dim = 32,
            include_flow_estimator = true
        )
        
        # Check flow estimates are positive
        test_states = [
            TestState(0, false),
            TestState(2, false),
            TestState(5, true)
        ]
        
        for state in test_states
            if !is_terminal_state(state)
                flow_est = GFlowNet.compute_flow_estimate(model, state)
                @test flow_est > 0
                @test !isnan(flow_est)
                @test !isinf(flow_est)
            end
        end
    end
    
    @testset "Error handling" begin
        # Create model without flow estimator
        initial_state = TestState(0, false)
        all_actions = [TestAction(1)]
        
        model_no_estimator = create_gflownet(
            initial_state,
            all_actions;
            state_dim = 2,
            hidden_dim = 32,
            include_flow_estimator = false
        )
        
        # Should throw error when trying to use DIRECT_FLOW without estimator
        trajectory = sample_trajectory(model_no_estimator)
        @test_throws ArgumentError GFlowNet.direct_flow_loss(model_no_estimator, trajectory)
        
        # Training with DIRECT_FLOW should fail gracefully
        config = TrainingConfig(
            objective = DIRECT_FLOW_OBJECTIVE,
            n_iterations = 1,
            batch_size = 1,
            verbose = false
        )
        
        # This should handle the error internally and return reasonable values
        history = train_gflownet(model_no_estimator, config)
        @test all(isnan.(history.losses) .| isinf.(history.losses))
    end
end

println("All DIRECT_FLOW_OBJECTIVE tests passed! ✅")