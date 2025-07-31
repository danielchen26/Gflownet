#!/usr/bin/env julia

# Comprehensive validation suite for GFlowNet core functions
# This test suite validates mathematical consistency and proper implementation

using Pkg
Pkg.activate(".")
using GFlowNet
using Test
using ComponentArrays
using Lux
using Random

println("🧪 GFlowNet Core Functions Validation Suite")
println("=" ^ 50)

# =============================================================================
# Test 1: State-Action Interface
# =============================================================================

@testset "State-Action Interface" begin
    println("\n1️⃣ Testing State-Action Interface...")
    
    # Create test states and actions
    state1 = GFlowNet.SimpleState([0])
    state2 = GFlowNet.SimpleState([5])
    state3 = GFlowNet.SimpleState([10])
    
    action_increment = GFlowNet.SimpleAction(1)
    action_decrement = GFlowNet.SimpleAction(2)
    action_terminate = GFlowNet.SimpleAction(-1)
    
    # Test is_applicable
    @test GFlowNet.is_applicable(action_increment, state1) == true
    @test GFlowNet.is_applicable(action_increment, state3) == false  # sum >= 10
    @test GFlowNet.is_applicable(action_decrement, state1) == false  # sum <= 0
    @test GFlowNet.is_applicable(action_decrement, state2) == true
    @test GFlowNet.is_applicable(action_terminate, state1) == true
    @test GFlowNet.is_applicable(action_terminate, state2) == true
    
    # Test apply_action
    new_state = GFlowNet.apply_action(action_increment, state1)
    @test new_state.data == [1]
    
    new_state = GFlowNet.apply_action(action_decrement, state2)
    @test new_state.data == [4]
    
    terminal_state = GFlowNet.apply_action(action_terminate, state1)
    @test terminal_state.data == [-1]
    
    println("✅ State-Action interface tests passed")
end

# =============================================================================
# Test 2: Reward Function
# =============================================================================

@testset "Reward Function" begin
    println("\n2️⃣ Testing Reward Function...")
    
    # Test reward computation for different states
    state1 = GFlowNet.SimpleState([0])
    state2 = GFlowNet.SimpleState([1, 1])
    terminal_state = GFlowNet.SimpleState([-1])
    
    reward1 = GFlowNet.reward(state1)
    reward2 = GFlowNet.reward(state2)
    reward_terminal = GFlowNet.reward(terminal_state)
    
    # All rewards should be positive
    @test reward1 > 0
    @test reward2 > 0
    @test reward_terminal > 0
    
    # Terminal state should have reward = 1.0
    @test reward_terminal ≈ 1.0
    
    # Rewards should be different for different states
    @test reward1 != reward2
    
    println("✅ Reward function tests passed")
end

# =============================================================================
# Test 3: Model Creation and Basic Functions
# =============================================================================

@testset "Model Creation" begin
    println("\n3️⃣ Testing Model Creation...")
    
    # Create a complete working model
    dag = GFlowNet.DirectedAcyclicGraph()
    
    # Add states
    initial_state = GFlowNet.SimpleState([0])
    intermediate_state = GFlowNet.SimpleState([1])
    terminal_state = GFlowNet.SimpleState([-1])
    
    GFlowNet.add_state!(dag, initial_state)
    GFlowNet.add_state!(dag, intermediate_state)
    GFlowNet.add_state!(dag, terminal_state)
    
    # Add actions
    GFlowNet.add_action!(dag, GFlowNet.SimpleAction(1))  # increment
    GFlowNet.add_action!(dag, GFlowNet.SimpleAction(2))  # decrement
    GFlowNet.add_action!(dag, GFlowNet.SimpleAction(-1)) # terminate
    
    # Create real Lux.jl neural networks
    state_dim = 1  # SimpleState has 1-dimensional data
    n_actions = 3
    hidden_dim = 8

    # Forward policy network
    forward_model = Chain(
        Dense(state_dim, hidden_dim, tanh),
        Dense(hidden_dim, n_actions)
    )

    # Initialize parameters
    rng = Random.default_rng()
    forward_params, forward_states = Lux.setup(rng, forward_model)

    # Convert to ComponentArrays
    all_params = ComponentArray(
        forward = ComponentArray(forward_params)
    )

    # Create policies
    forward_policy = GFlowNet.ForwardPolicy(forward_model)

    # Create model
    model = GFlowNet.GFlowNetModel(
        dag = dag,
        forward_policy = forward_policy,
        backward_policy = nothing,
        flow_estimator = nothing,
        partition_function = nothing,
        objectives = GFlowNet.AbstractGFlowNetObjective[],
        optimizer = nothing,
        parameters = all_params,
        states = (forward = forward_states, backward = nothing, flow = nothing)
    )
    
    @test model isa GFlowNet.GFlowNetModel
    @test !isnothing(model.forward_policy)
    
    println("✅ Model creation tests passed")
    
    # Store model for later tests
    global test_model = model
end

# =============================================================================
# Test 4: Probability Functions
# =============================================================================

@testset "Probability Functions" begin
    println("\n4️⃣ Testing Probability Functions...")
    
    state1 = GFlowNet.SimpleState([1])
    state2 = GFlowNet.SimpleState([2])
    
    # Test forward transition probability
    prob = GFlowNet.forward_transition_prob(test_model, state1, state2)
    @test prob >= 0.0
    @test prob <= 1.0
    
    # Test that probabilities from a state sum to approximately 1
    next_states = GFlowNet.get_next_states(test_model.dag, state1)
    if !isempty(next_states)
        total_prob = sum(GFlowNet.forward_transition_prob(test_model, state1, ns) for ns in next_states)
        @test total_prob ≈ 1.0 atol=0.1  # Allow some tolerance for numerical issues
    end
    
    println("✅ Probability function tests passed")
end

# =============================================================================
# Test 5: Flow Functions
# =============================================================================

@testset "Flow Functions" begin
    println("\n5️⃣ Testing Flow Functions...")
    
    state1 = GFlowNet.SimpleState([1])
    terminal_state = GFlowNet.SimpleState([-1])
    
    # Test flow computation
    flow1 = GFlowNet.flow(test_model, state1)
    flow_terminal = GFlowNet.flow(test_model, terminal_state)
    
    # All flows should be positive
    @test flow1 > 0
    @test flow_terminal > 0
    
    # Terminal state flow should equal its reward
    @test flow_terminal ≈ GFlowNet.reward(terminal_state) atol=1e-6
    
    # Test edge flow
    state2 = GFlowNet.SimpleState([2])
    edge_flow_val = GFlowNet.edge_flow(test_model, state1, state2)
    @test edge_flow_val >= 0
    
    println("✅ Flow function tests passed")
end

# =============================================================================
# Test 6: Loss Computation
# =============================================================================

@testset "Loss Computation" begin
    println("\n6️⃣ Testing Loss Computation...")
    
    # Create trajectories
    traj1 = GFlowNet.Trajectory([
        GFlowNet.SimpleState([0]),
        GFlowNet.SimpleState([1]),
        GFlowNet.SimpleState([-1])
    ])
    
    traj2 = GFlowNet.Trajectory([
        GFlowNet.SimpleState([0]),
        GFlowNet.SimpleState([-1])
    ])
    
    trajectories = [traj1, traj2]
    
    # Test loss computation
    loss = GFlowNet.trajectory_balance_loss(test_model, trajectories)
    
    # Loss should be finite and non-negative
    @test isfinite(loss)
    @test loss >= 0
    
    # Loss should be different for different trajectory sets
    single_traj = [traj1]
    loss_single = GFlowNet.trajectory_balance_loss(test_model, single_traj)
    @test loss != loss_single
    
    println("✅ Loss computation tests passed")
end

# =============================================================================
# Test 7: Training Configuration
# =============================================================================

@testset "Training Configuration" begin
    println("\n7️⃣ Testing Training Configuration...")
    
    config = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
        batch_size = 8,
        learning_rate = 0.01,
        n_iterations = 50
    )
    
    @test config.objective == GFlowNet.TRAJECTORY_BALANCE
    @test config.batch_size == 8
    @test config.learning_rate == 0.01
    @test config.n_iterations == 50
    
    # Test validation
    @test GFlowNet.validate_training_config(config) == true
    
    println("✅ Training configuration tests passed")
end

println("\n🎉 All core function tests passed!")
println("✅ GFlowNet core framework is mathematically consistent and functional!")
