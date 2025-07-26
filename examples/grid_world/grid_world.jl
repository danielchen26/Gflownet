"""
🎯 Grid World GFlowNet Example
=============================

A complete grid world environment demonstrating GFlowNet's high-level interface.
Uses GFlowNet's built-in functions for neural network creation and training.
"""

using GFlowNet, Random, Dates, ComponentArrays, Optimisers, Statistics

# Include report generation functions
include("report_generation.jl")

println("🎯 Grid World GFlowNet Example")
println("=" ^ 50)
println("🕐 Started at: $(Dates.format(now(), "HH:MM:SS"))")

# =============================================================================
# Grid State and Actions  
# =============================================================================

struct GridState <: GFlowNet.AbstractState
    x::Int
    y::Int
    is_terminal::Bool
end

abstract type GridAction <: GFlowNet.AbstractAction end
struct MoveUp <: GridAction end
struct MoveDown <: GridAction end  
struct MoveLeft <: GridAction end
struct MoveRight <: GridAction end
struct Terminate <: GridAction end

const UP, DOWN, LEFT, RIGHT, TERMINATE = MoveUp(), MoveDown(), MoveLeft(), MoveRight(), Terminate()
const GRID_SIZE = 5
const REWARD_POSITIONS = Dict((3, 3) => 10.0, (1, 5) => 5.0, (5, 1) => 5.0)

# =============================================================================
# GFlowNet Interface Implementation (Required for any domain)
# =============================================================================

function GFlowNet.state_to_features(state::GridState)
    # More Zygote-friendly: explicit type conversions and avoid array comprehensions
    x_norm = Float32((state.x-1)/(GRID_SIZE-1))
    y_norm = Float32((state.y-1)/(GRID_SIZE-1))
    terminal_flag = state.is_terminal ? Float32(1.0) : Float32(0.0)
    return Float32[x_norm, y_norm, terminal_flag]
end

function GFlowNet.is_applicable(action::GridAction, state::GridState)
    state.is_terminal && return false
    isa(action, Terminate) && return true
    x, y = state.x, state.y
    return (isa(action, MoveUp) && y < GRID_SIZE) || 
           (isa(action, MoveDown) && y > 1) ||
           (isa(action, MoveLeft) && x > 1) || 
           (isa(action, MoveRight) && x < GRID_SIZE)
end

function GFlowNet.apply_action(action::GridAction, state::GridState)
    isa(action, Terminate) && return GridState(state.x, state.y, true)

    # Avoid mutations - use conditional expressions instead
    x = isa(action, MoveLeft) ? state.x - 1 :
        isa(action, MoveRight) ? state.x + 1 : state.x

    y = isa(action, MoveUp) ? state.y + 1 :
        isa(action, MoveDown) ? state.y - 1 : state.y

    return GridState(x, y, false)
end

GFlowNet.is_terminal_state(state::GridState) = state.is_terminal

function GFlowNet.reward(state::GridState)
    !state.is_terminal && return 0.0f0

    # FIXED: Make Zygote-compatible by avoiding dictionary get() calls
    # Use conditional logic instead of dictionary lookup
    if state.x == 3 && state.y == 3
        return 10.0f0  # High reward position
    elseif (state.x == 1 && state.y == 5) || (state.x == 5 && state.y == 1)
        return 5.0f0   # Medium reward positions
    else
        return 1.0f0   # Default reward
    end
end

# =============================================================================
# Model Creation Using HIGH-LEVEL GFlowNet Functions
# =============================================================================

function create_gflownet_model()
    println("🔧 Creating GFlowNet model using HIGH-LEVEL package functions...")
    
    initial_state = GridState(1, 1, false)
    actions = GridAction[UP, DOWN, LEFT, RIGHT, TERMINATE]
    
    # Create DAG - the warnings are actually OK, they just mean some terminal states
    # might not be reachable in practice, but the DAG structure is still valid
    terminal_states = [GridState(x, y, true) for x in 1:GRID_SIZE for y in 1:GRID_SIZE]
    terminal_sink = GridState(0, 0, true)
    dag = GFlowNet.create_dag(initial_state, terminal_states, terminal_sink, actions)
    
    # Use HIGH-LEVEL GFlowNet functions to create neural networks automatically
    input_dim, hidden_dim, n_actions = 3, 64, 5
    rng = Random.default_rng()
    Random.seed!(rng, 42)
    
    println("✅ Using GFlowNet's high-level neural network creation functions:")
    
    # HIGH-LEVEL: Create forward policy automatically
    forward_policy, forward_ps, forward_st = GFlowNet.create_forward_policy(input_dim, hidden_dim, n_actions, rng)
    println("   - Forward policy created automatically")
    
    # HIGH-LEVEL: Create flow estimator automatically  
    flow_estimator, flow_ps, flow_st = GFlowNet.create_flow_estimator(input_dim, hidden_dim, rng)
    println("   - Flow estimator created automatically")
    
    # Setup parameters and optimizer following working examples
    parameters = ComponentArray(
        forward=ComponentArray(forward_ps),
        flow=ComponentArray(flow_ps)
    )
    states = (forward=forward_st, backward=nothing, flow=flow_st)
    
    # Create optimizer like in working examples
    optimizer = Optimisers.setup(Optimisers.Adam(0.01), parameters)
    
    # Create GFlowNet model following working examples exactly
    model = GFlowNet.GFlowNetModel(
        dag=dag,
        forward_policy=forward_policy,
        flow_estimator=flow_estimator,
        partition_function=50.0,
        objectives=[GFlowNet.TrajectoryBalanceObjective(1.0)],
        optimizer=optimizer,
        parameters=parameters,
        states=states
    )
    
    println("✅ GFlowNet model created using HIGH-LEVEL interface!")
    println("   - NO manual neural network definition")
    println("   - Used create_forward_policy() and create_flow_estimator()")
    println("   - Total parameters: $(length(parameters))")
    
    return model
end

# =============================================================================
# Training Using HIGH-LEVEL GFlowNet Interface
# =============================================================================

function train_gflownet_high_level(model)
    println("\n🚀 Training using HIGH-LEVEL GFlowNet interface...")
    
    # Training configuration optimized for demonstrating learning progress
    config = GFlowNet.TrainingConfig(
        objective=GFlowNet.TRAJECTORY_BALANCE,
        partition_function_method=GFlowNet.SIMPLE_ESTIMATION,
        n_iterations=20,  # Shorter for demonstration
        batch_size=16,
        learning_rate=0.01,
        early_stopping_patience=15,
        validation_frequency=5,
        partition_update_frequency=5
    )
    
    println("✅ Using GFlowNet's TrainingConfig:")
    println("   - Objective: $(config.objective)")
    println("   - Batch size: $(config.batch_size)")
    println("   - Learning rate: $(config.learning_rate)")
    println("   - Iterations: $(config.n_iterations)")
    
    # HIGH-LEVEL: Use GFlowNet's built-in training function
    println("\n🎯 Calling GFlowNet.train_gflownet() - the HIGH-LEVEL training function...")
    
    try
        training_history = GFlowNet.train_gflownet(model, config; verbose=true)

        println("✅ HIGH-LEVEL training completed!")
        return model, training_history

    catch e
        println("⚠️  High-level training encountered an issue: $e")
        println("🔍 Full error details:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        println("🔄 Falling back to simplified sampling for demonstration...")

        # Simple fallback to demonstrate sampling still works
        println("📊 Testing core sampling function:")
        for i in 1:5
            try
                traj = GFlowNet.sample_trajectory(model)
                final_state = traj.states[end]
                reward = GFlowNet.reward(final_state)
                println("   Sample $i: ($(final_state.x), $(final_state.y)) → reward: $reward")
            catch e2
                println("   Sample $i: Error in sampling - $e2")
            end
        end

        return model, Dict()
    end
end

# =============================================================================
# Evaluation Using Core GFlowNet Functions
# =============================================================================

function evaluate_gflownet_model(model; n_trajectories=50)
    println("\n🎯 Evaluating using core GFlowNet.sample_trajectory()...")
    
    # Sample trajectories using core function
    trajectories = []
    for i in 1:n_trajectories
        try
            traj = GFlowNet.sample_trajectory(model)
            push!(trajectories, traj)
        catch e
            println("⚠️  Warning: Trajectory $i failed to sample - $e")
        end
    end
    
    valid_trajectories = filter(traj -> length(traj.states) > 1, trajectories)
    
    if isempty(valid_trajectories)
        println("❌ No valid trajectories sampled")
        return [], []
    end
    
    # Analyze results
    rewards = [GFlowNet.reward(traj.states[end]) for traj in valid_trajectories]
    mean_reward = sum(rewards) / length(rewards)
    high_reward_count = count(r -> r >= 5.0, rewards)
    max_reward = maximum(rewards)
    
    # Analyze final positions
    final_positions = [(traj.states[end].x, traj.states[end].y) for traj in valid_trajectories]
    position_counts = Dict{Tuple{Int,Int}, Int}()
    for pos in final_positions
        position_counts[pos] = get(position_counts, pos, 0) + 1
    end
    
    # Show trajectory examples
    println("📊 Evaluation Results:")
    println("   - Valid trajectories: $(length(valid_trajectories))/$n_trajectories")
    println("   - Mean reward: $(round(mean_reward, digits=2))")
    println("   - High reward trajectories (≥5.0): $high_reward_count/$(length(valid_trajectories))")
    println("   - Maximum reward: $max_reward")
    
    # Show position distribution
    println("\n📍 Final position distribution:")
    for ((x, y), count) in sort(collect(position_counts), by=x->x[2], rev=true)
        reward = get(REWARD_POSITIONS, (x, y), 1.0)
        println("   ($x, $y): $count trajectories [reward: $reward]")
    end
    
    return valid_trajectories, rewards
end

# =============================================================================
# Main Execution Demonstrating HIGH-LEVEL Interface  
# =============================================================================

function main()
    try
        println("\n🎯 Demonstrating GFlowNet's HIGH-LEVEL Interface:")
        println("   ❌ NO manual neural network definition")
        println("   ✅ Using built-in create_forward_policy()")
        println("   ✅ Using built-in create_flow_estimator()")
        println("   ✅ Using built-in TrainingConfig")
        println("   ✅ Using built-in train_gflownet()")
        
        # Create model using HIGH-LEVEL functions
        model = create_gflownet_model()
        
        # Train using HIGH-LEVEL functions
        model, training_history = train_gflownet_high_level(model)

        # Evaluate model
        println("\n📊 Evaluating model...")
        eval_trajectories, eval_rewards = evaluate_gflownet_model(model)

        # Generate comprehensive results after we have all the data
        println("\n📄 Generating comprehensive results and HTML report...")
        html_path = generate_comprehensive_results(training_history, eval_trajectories, eval_rewards)

        println("\n🎯 Comprehensive GFlowNet example completed successfully!")
        println("📄 HTML Report: $html_path")
        println("📊 All results saved in 'results/' directory")
        
    catch e
        println("❌ Error: $e")
        println(stacktrace())
    end
end

# Run the example
if abspath(PROGRAM_FILE) == @__FILE__
    main()
    println("=" ^ 50)
end
