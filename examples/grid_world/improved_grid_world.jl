"""
🎯 Improved Grid World GFlowNet Example
=====================================

This example demonstrates the NEW generic DAG construction approach.
Key improvements:
- Uses ExplorationDAGBuilder for automatic state space discovery
- No manual state enumeration required
- Robust cycle prevention by construction
- Configurable exploration strategies
- Better scalability to larger state spaces

Compares to the original grid_world.jl to show the benefits of generic design.
"""

using GFlowNet, Random, Dates, ComponentArrays, Optimisers, Statistics

# Include the new generic DAG builder
include("../../src/core/dag_builder.jl")

# Include report generation functions
include("report_generation.jl")

println("🎯 Improved Grid World GFlowNet Example (Generic DAG Construction)")
println("="^70)
println("🕐 Started at: $(Dates.format(now(), "HH:MM:SS"))")

# =============================================================================
# Domain Definition (Same as before, but cleaner)
# =============================================================================

struct GridState <: GFlowNet.AbstractState
    x::Int
    y::Int
    is_terminal::Bool
end

# Essential for Set operations in DAG construction
Base.:(==)(a::GridState, b::GridState) = a.x == b.x && a.y == b.y && a.is_terminal == b.is_terminal
Base.hash(state::GridState, h::UInt) = hash((state.x, state.y, state.is_terminal), h)

abstract type GridAction <: GFlowNet.AbstractAction end
struct MoveRight <: GridAction end
struct MoveUp <: GridAction end
struct Terminate <: GridAction end

const RIGHT, UP, TERMINATE = MoveRight(), MoveUp(), Terminate()
const GRID_SIZE = 5
const REWARD_POSITIONS = Dict((3, 3) => 10.0, (1, 5) => 5.0, (5, 1) => 5.0)

# =============================================================================
# GFlowNet Interface Implementation (Same as before)
# =============================================================================

function GFlowNet.state_to_features(state::GridState)
    x_norm = Float32((state.x - 1) / (GRID_SIZE - 1))
    y_norm = Float32((state.y - 1) / (GRID_SIZE - 1))
    terminal_flag = state.is_terminal ? Float32(1.0) : Float32(0.0)
    return Float32[x_norm, y_norm, terminal_flag]
end

function GFlowNet.is_applicable(action::GridAction, state::GridState)
    state.is_terminal && return false
    isa(action, Terminate) && return true
    x, y = state.x, state.y
    return (isa(action, MoveUp) && y < GRID_SIZE) ||
           (isa(action, MoveRight) && x < GRID_SIZE)
end

function GFlowNet.apply_action(action::GridAction, state::GridState)
    isa(action, Terminate) && return GridState(state.x, state.y, true)

    x = isa(action, MoveRight) ? state.x + 1 : state.x
    y = isa(action, MoveUp) ? state.y + 1 : state.y

    return GridState(x, y, false)
end

GFlowNet.is_terminal_state(state::GridState) = state.is_terminal

function GFlowNet.base_reward(state::GridState)
    !state.is_terminal && return 0.0

    if state.x == 3 && state.y == 3
        return 10.0
    elseif (state.x == 1 && state.y == 5) || (state.x == 5 && state.y == 1)
        return 5.0
    else
        return 1.0
    end
end

function GFlowNet.reward(state::GridState)
    return GFlowNet.base_reward(state)
end

# =============================================================================
# NEW: Generic DAG Construction with Configuration
# =============================================================================

function create_gflownet_model_improved()
    println("🔧 Creating GFlowNet model using IMPROVED GENERIC DAG construction...")

    # Domain setup
    initial_state = GridState(1, 1, false)
    actions = GridAction[RIGHT, UP, TERMINATE]

    # =================================================================
    # NEW APPROACH: Use ExplorationDAGBuilder with configuration
    # =================================================================

    println("   📋 Configuring DAG construction...")

    # Create DAG builder configuration with sensible defaults
    dag_config = DAGBuilderConfig(
        max_states=1000,                    # Limit state space size
        max_depth=20,                       # Prevent infinite exploration
        enable_pruning=true,                # Enable optimization
        validate_construction=true,         # Comprehensive validation
        exploration_strategy=:bfs,          # Breadth-first for systematic exploration
        cycle_detection_method=:strict      # Prevent any cycles
    )

    println("   ✅ Configuration: $(dag_config.max_states) max states, $(dag_config.exploration_strategy) exploration")

    # =================================================================
    # KEY IMPROVEMENT: One-line DAG construction!
    # =================================================================

    println("   🔍 Building DAG with automatic state space exploration...")
    dag = create_dag_with_exploration(initial_state, actions, dag_config)

    # =================================================================
    # Analysis of constructed DAG
    # =================================================================

    println("   📊 Analyzing constructed DAG...")
    dag_metrics = analyze_dag(dag)

    println("   ✅ DAG Analysis:")
    println("     - States: $(dag_metrics.n_states)")
    println("     - Edges: $(dag_metrics.n_edges)")
    println("     - Average degree: $(round(dag_metrics.avg_degree, digits=2))")
    println("     - Longest path: $(dag_metrics.longest_path)")
    println("     - Is acyclic: $(dag_metrics.is_acyclic)")

    # =================================================================
    # Rest of model creation (same as before)
    # =================================================================

    input_dim, hidden_dim, n_actions = 3, 64, 5
    rng = Random.default_rng()
    Random.seed!(rng, 42)

    println("   🧠 Creating neural networks...")
    forward_policy, forward_ps, forward_st = GFlowNet.create_forward_policy(input_dim, hidden_dim, n_actions, rng)
    flow_estimator, flow_ps, flow_st = GFlowNet.create_flow_estimator(input_dim, hidden_dim, rng)

    # Setup parameters and optimizer
    parameters = ComponentArray(
        forward=ComponentArray(forward_ps),
        flow=ComponentArray(flow_ps)
    )
    states = (forward=forward_st, backward=nothing, flow=flow_st)
    optimizer = Optimisers.setup(Optimisers.Adam(0.01), parameters)

    # Create GFlowNet model
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

    println("   ✅ Model created with $(length(parameters)) parameters")
    println("   🎯 KEY IMPROVEMENT: No manual state enumeration required!")

    return model, dag_metrics
end

# =============================================================================
# Comparison Function: Show benefits of generic approach
# =============================================================================

function demonstrate_configuration_flexibility()
    println("\n🔬 Demonstrating configuration flexibility...")

    initial_state = GridState(1, 1, false)
    actions = GridAction[RIGHT, UP, TERMINATE]

    # Configuration 1: Small, fast exploration
    config_small = DAGBuilderConfig(
        max_states=50,
        exploration_strategy=:bfs,
        cycle_detection_method=:strict
    )

    # Configuration 2: Large, comprehensive exploration
    config_large = DAGBuilderConfig(
        max_states=500,
        exploration_strategy=:dfs,
        cycle_detection_method=:optimistic
    )

    println("   🔍 Building DAG with small configuration...")
    dag_small = create_dag_with_exploration(initial_state, actions, config_small)
    metrics_small = analyze_dag(dag_small)

    println("   🔍 Building DAG with large configuration...")
    dag_large = create_dag_with_exploration(initial_state, actions, config_large)
    metrics_large = analyze_dag(dag_large)

    println("   📊 Comparison:")
    println("     Small config: $(metrics_small.n_states) states, $(metrics_small.n_edges) edges")
    println("     Large config: $(metrics_large.n_states) states, $(metrics_large.n_edges) edges")
    println("   ✅ Same interface, different behavior - TRUE GENERICITY!")

    return dag_small, dag_large
end

# =============================================================================
# Training (Same as before, but with better DAG)
# =============================================================================

function train_gflownet_improved(model)
    println("\n🚀 Training with improved DAG construction...")

    config = GFlowNet.TrainingConfig(
        objective=GFlowNet.TRAJECTORY_BALANCE,
        partition_function_method=GFlowNet.SIMPLE_ESTIMATION,
        n_iterations=20,
        batch_size=16,
        learning_rate=0.01,
        early_stopping_patience=15,
        validation_frequency=5,
        partition_update_frequency=5
    )

    println("   ✅ Training configuration ready")
    println("   🎯 Expected improvement: Better state space coverage due to systematic exploration")

    try
        training_history = GFlowNet.train_gflownet(model, config; verbose=true)
        println("   ✅ Training completed successfully!")
        return model, training_history
    catch e
        println("   ⚠️  Training issue: $e")
        return model, Dict()
    end
end

# =============================================================================
# Evaluation (Enhanced with DAG analysis)
# =============================================================================

function evaluate_improved_model(model, dag_metrics; n_trajectories=50)
    println("\n🎯 Evaluating improved model...")

    # Sample trajectories
    trajectories = []
    for i in 1:n_trajectories
        try
            traj = GFlowNet.sample_trajectory(model)
            push!(trajectories, traj)
        catch e
            println("   ⚠️  Trajectory $i failed: $e")
        end
    end

    valid_trajectories = filter(traj -> length(traj.states) > 1, trajectories)

    if isempty(valid_trajectories)
        println("   ❌ No valid trajectories")
        return [], []
    end

    # Analyze results
    rewards = [GFlowNet.reward(traj.states[end]) for traj in valid_trajectories]
    final_positions = [(traj.states[end].x, traj.states[end].y) for traj in valid_trajectories]

    # Enhanced analysis
    unique_positions = length(unique(final_positions))
    state_space_coverage = unique_positions / dag_metrics.n_states * 100

    mean_reward = mean(rewards)
    high_reward_count = count(r -> r >= 5.0, rewards)
    max_reward = maximum(rewards)

    println("   📊 Enhanced Evaluation Results:")
    println("     - Valid trajectories: $(length(valid_trajectories))/$n_trajectories")
    println("     - Mean reward: $(round(mean_reward, digits=2))")
    println("     - High reward trajectories (≥5.0): $high_reward_count")
    println("     - Maximum reward: $max_reward")
    println("     - Unique end positions: $unique_positions")
    println("     - State space coverage: $(round(state_space_coverage, digits=1))%")

    # Position analysis
    position_counts = Dict{Tuple{Int,Int},Int}()
    for pos in final_positions
        position_counts[pos] = get(position_counts, pos, 0) + 1
    end

    println("\n   📍 Position distribution:")
    for ((x, y), count) in sort(collect(position_counts), by=x -> x[2], rev=true)
        reward = get(REWARD_POSITIONS, (x, y), 1.0)
        percentage = round(count / length(valid_trajectories) * 100, digits=1)
        println("     ($x, $y): $count trajectories ($percentage%) [reward: $reward]")
    end

    return valid_trajectories, rewards
end

# =============================================================================
# Main Function: Demonstrate improved approach
# =============================================================================

function main()
    try
        println("\n🎯 IMPROVED Grid World Example with Generic DAG Construction")
        println("   ✅ Key benefits demonstrated:")
        println("     - Automatic state space discovery")
        println("     - Configuration-driven exploration")
        println("     - Robust cycle prevention")
        println("     - Better scalability")
        println("     - Generic, reusable design")

        # Demonstrate configuration flexibility
        demonstrate_configuration_flexibility()

        # Create improved model
        model, dag_metrics = create_gflownet_model_improved()

        # Train the model
        model, training_history = train_gflownet_improved(model)

        # Evaluate with enhanced analysis
        eval_trajectories, eval_rewards = evaluate_improved_model(model, dag_metrics)

        # Generate results
        println("\n📄 Generating results...")
        html_path = generate_comprehensive_results(training_history, eval_trajectories, eval_rewards)

        println("\n🎯 IMPROVED Grid World Example completed successfully!")
        println("📊 Key improvements demonstrated:")
        println("   ✅ No manual state enumeration required")
        println("   ✅ Configurable exploration strategies")
        println("   ✅ Automatic cycle prevention")
        println("   ✅ Better DAG analysis and metrics")
        println("   ✅ More robust and scalable design")
        println("📄 Results: $html_path")

    catch e
        println("❌ Error in improved example: $e")
        println(stacktrace())
    end
end

# =============================================================================
# Comparison with Original Approach
# =============================================================================

function compare_approaches()
    println("\n🔄 Comparing Original vs Improved Approach")
    println("="^50)

    println("📊 Original Approach:")
    println("   - Manual state enumeration with generate_reachable_states()")
    println("   - Custom BFS implementation")
    println("   - Domain-specific cycle detection")
    println("   - Fixed exploration strategy")
    println("   - Hard to configure or extend")

    println("\n📊 Improved Approach:")
    println("   - Generic ExplorationDAGBuilder")
    println("   - Configurable exploration (BFS/DFS)")
    println("   - Multiple cycle detection strategies")
    println("   - Flexible configuration system")
    println("   - Reusable across domains")
    println("   - Built-in analysis and optimization")

    println("\n✅ Result: Much more robust, flexible, and maintainable!")
end

# Run comparison
if abspath(PROGRAM_FILE) == @__FILE__
    compare_approaches()
    main()
    println("="^70)
end
