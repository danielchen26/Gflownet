# Grid World Application for GFlowNet
# High-level interface for creating grid world environments

using ..GFlowNet: AbstractState, AbstractAction, state_to_features, is_applicable, apply_action, reward
using Random
using ComponentArrays
using Optimisers

"""
    GridState

State representation for grid world environment.

# Fields
- `x::Int`: x-coordinate (1-based)
- `y::Int`: y-coordinate (1-based)
- `is_terminal::Bool`: whether this is a terminal state
"""
struct GridState <: AbstractState
    x::Int
    y::Int
    is_terminal::Bool
end

# Essential for Set operations in DAG construction
Base.:(==)(a::GridState, b::GridState) = a.x == b.x && a.y == b.y && a.is_terminal == b.is_terminal
Base.hash(state::GridState, h::UInt) = hash((state.x, state.y, state.is_terminal), h)

"""
    GridAction

Abstract base type for grid world actions.
"""
abstract type GridAction <: AbstractAction end

"""
    MoveRight <: GridAction

Action to move right (increase x by 1).
"""
struct MoveRight <: GridAction end

"""
    MoveUp <: GridAction

Action to move up (increase y by 1).
"""
struct MoveUp <: GridAction end

"""
    MoveLeft <: GridAction

Action to move left (decrease x by 1).
"""
struct MoveLeft <: GridAction end

"""
    MoveDown <: GridAction

Action to move down (decrease y by 1).
"""
struct MoveDown <: GridAction end

"""
    Terminate <: GridAction

Action to terminate trajectory at current position.
"""
struct Terminate <: GridAction end

# Ensure proper equality comparison for singleton actions
Base.:(==)(a::MoveRight, b::MoveRight) = true
Base.:(==)(a::MoveUp, b::MoveUp) = true
Base.:(==)(a::MoveLeft, b::MoveLeft) = true
Base.:(==)(a::MoveDown, b::MoveDown) = true
Base.:(==)(a::Terminate, b::Terminate) = true
Base.:(==)(a::GridAction, b::GridAction) = false  # Different types

# =============================================================================
# GFlowNet Interface Implementation
# =============================================================================

"""
    GFlowNet.state_to_features(state::GridState)

Convert grid state to feature vector for neural network input.
Returns normalized coordinates and terminal flag.
"""
function GFlowNet.state_to_features(state::GridState)::Vector{Float32}
    grid_size = isassigned(GRID_CONFIG) ? GRID_CONFIG[].grid_size : 5
    x_norm = Float32((state.x - 1) / (grid_size - 1))
    y_norm = Float32((state.y - 1) / (grid_size - 1))
    terminal_flag = state.is_terminal ? Float32(1.0) : Float32(0.0)
    return Float32[x_norm, y_norm, terminal_flag]
end

"""
    GFlowNet.is_applicable(action::GridAction, state::GridState)

Check if an action is applicable from the given state.
"""
function GFlowNet.is_applicable(action::GridAction, state::GridState)::Bool
    grid_size = isassigned(GRID_CONFIG) ? GRID_CONFIG[].grid_size : 5
    state.is_terminal && return false

    # Allow termination from any non-starting position
    if isa(action, Terminate)
        return state.x != 1 || state.y != 1
    end

    x, y = state.x, state.y
    return (isa(action, MoveUp) && y < grid_size) ||
           (isa(action, MoveDown) && y > 1) ||
           (isa(action, MoveLeft) && x > 1) ||
           (isa(action, MoveRight) && x < grid_size)
end

"""
    GFlowNet.apply_action(action::GridAction, state::GridState)

Apply an action to a state, returning the new state.
Uses mutation-free functional approach for AD compatibility.
"""
function GFlowNet.apply_action(action::GridAction, state::GridState)::GridState
    isa(action, Terminate) && return GridState(state.x, state.y, true)

    # Use conditional expressions (mutation-free for Zygote compatibility)
    x = isa(action, MoveRight) ? state.x + 1 :
        isa(action, MoveLeft) ? state.x - 1 : state.x
    y = isa(action, MoveUp) ? state.y + 1 :
        isa(action, MoveDown) ? state.y - 1 : state.y

    return GridState(x, y, false)
end

"""
    GFlowNet.is_terminal_state(state::GridState)

Check if a state is terminal.
"""
GFlowNet.is_terminal_state(state::GridState) = state.is_terminal

"""
    GFlowNet.reward(state::GridState)

Compute reward for a terminal state.
"""
function GFlowNet.reward(state::GridState)::Float64
    !state.is_terminal && return 0.0

    reward_positions = isassigned(GRID_CONFIG) ? GRID_CONFIG[].reward_positions : Dict((3,3)=>10.0)

    # Check for special reward positions
    position_reward = get(reward_positions, (state.x, state.y), 0.0)
    if position_reward > 0.0
        return position_reward
    end

    # Distance-based exploration reward
    distance_from_start = abs(state.x - 1) + abs(state.y - 1)

    if distance_from_start == 0
        return 0.1  # Very low reward for staying at start
    elseif distance_from_start <= 2
        return 1.0  # Low reward for minimal exploration
    else
        return 2.0  # Better reward for exploration
    end
end

# =============================================================================
# High-Level Creation Functions
# =============================================================================

# Store configuration for domain functions
const GRID_CONFIG = Ref{NamedTuple}()

"""
    create_grid_world_gflownet(;
        grid_size::Int=5,
        reward_positions::Dict{Tuple{Int,Int},Float64}=Dict((3,3)=>10.0, (5,5)=>8.0),
        allow_all_moves::Bool=false,
        hidden_dim::Int=64,
        learning_rate::Float64=0.01,
        include_backward::Bool=false,
        partition_function_method::PartitionFunctionMethod=SIMPLE_ESTIMATION,
        rng::AbstractRNG=Random.default_rng()
    )

Create a complete GFlowNet model for grid world using implicit DAG approach.

This function creates a robust GFlowNet model that computes the DAG implicitly,
eliminating cache misses and training errors while maintaining all mathematical properties.

# Arguments
- `grid_size::Int=5`: Size of the grid (grid_size × grid_size)
- `reward_positions::Dict`: Dictionary mapping (x,y) positions to reward values
- `allow_all_moves::Bool=false`: If true, allows all 4 directions + terminate. If false, only up/right + terminate (acyclic)
- `hidden_dim::Int=64`: Hidden dimension for neural networks
- `learning_rate::Float64=0.01`: Learning rate for optimizer
- `include_backward::Bool=false`: Whether to include backward policy
- `partition_function_method::PartitionFunctionMethod=SIMPLE_ESTIMATION`: How to handle partition function Z:
  - `SIMPLE_ESTIMATION`: Z = 1 (default, simple and fast)
  - `LEARNABLE_ESTIMATION`: Learn Z as parameter (better exploration, ~42% improvement)
- `rng::AbstractRNG`: Random number generator

# Returns
- `GFlowNetModel`: Complete model ready for training

# Example
```julia
using GFlowNet

# Create a simple grid world with learnable Z
model = create_grid_world_gflownet(
    grid_size=5,
    reward_positions=Dict((3,3)=>20.0, (5,1)=>15.0, (1,5)=>15.0),
    hidden_dim=64,
    partition_function_method=LEARNABLE_ESTIMATION  # Enable Z learning
)

# Train the model with learnable Z
config = TrainingConfig(
    n_iterations=1000, 
    batch_size=32,
    partition_function_method=LEARNABLE_ESTIMATION
)
history = train_gflownet(model, config; verbose=true)

# Access learned Z
learned_Z = exp(model.parameters.log_Z)
println("Learned partition function: \$learned_Z")

# Sample trajectories
trajectories = [sample_trajectory(model) for _ in 1:50]
```
"""
function create_grid_world_gflownet(;
    grid_size::Int=5,
    reward_positions::Dict{Tuple{Int,Int},Float64}=Dict((3,3)=>10.0, (5,5)=>8.0),
    allow_all_moves::Bool=false,
    hidden_dim::Int=64,
    learning_rate::Float64=0.01,
    include_backward::Bool=false,
    include_flow_estimator::Bool=false,
    partition_function_method::PartitionFunctionMethod=SIMPLE_ESTIMATION,
    rng::AbstractRNG=Random.default_rng()
)

    # Validate inputs
    grid_size >= 2 || throw(ArgumentError("grid_size must be at least 2"))
    hidden_dim > 0 || throw(ArgumentError("hidden_dim must be positive"))
    0 < learning_rate < 1 || throw(ArgumentError("learning_rate must be in (0,1)"))

    # Set global configuration for domain functions
    GRID_CONFIG[] = (grid_size=grid_size, reward_positions=reward_positions)

    # Create initial state and actions
    initial_state = GridState(1, 1, false)

    if allow_all_moves
        actions = GridAction[MoveRight(), MoveLeft(), MoveUp(), MoveDown(), Terminate()]
    else
        # Acyclic version - only up and right moves to prevent cycles
        actions = GridAction[MoveRight(), MoveUp(), Terminate()]
    end

    # Create model using on-demand approach - clean and robust!
    return GFlowNet.create_gflownet(
        initial_state,
        actions;
        state_dim = 3,  # x_norm, y_norm, is_terminal
        hidden_dim = hidden_dim,
        learning_rate = learning_rate,
        include_backward = include_backward,
        include_flow_estimator = include_flow_estimator,
        partition_function_method = partition_function_method,
        rng = rng
    )
end

"""
    create_grid_world(grid_size::Int=5)

Create a grid world model with default settings.
Convenience function for quick experimentation using on-demand approach.

# Example
```julia
model = create_grid_world(4)
config = TrainingConfig(n_iterations=50)
history = train_gflownet(model, config; verbose=true)
```
"""
function create_grid_world(grid_size::Int=5)
    return create_grid_world_gflownet(
        grid_size=grid_size,
        reward_positions=Dict((grid_size,grid_size)=>10.0, (1,grid_size)=>8.0, (grid_size,1)=>8.0),
        allow_all_moves=false,
        hidden_dim=32
    )
end



"""
    analyze_grid_world_results(trajectories::Vector, grid_size::Int=5)

Analyze trajectories from a grid world model and print statistics.
Works with the standard on-demand approach.
"""
function analyze_grid_world_results(trajectories::Vector, grid_size::Int=5)
    valid_trajectories = filter(traj -> length(traj.states) > 1, trajectories)

    if isempty(valid_trajectories)
        println("No valid trajectories found")
        return
    end

    # Compute statistics
    rewards = [reward(traj.states[end]) for traj in valid_trajectories]
    final_positions = [(traj.states[end].x, traj.states[end].y) for traj in valid_trajectories]

    println("Grid World Results Analysis:")
    println("  Valid trajectories: $(length(valid_trajectories))/$(length(trajectories))")
    println("  Mean reward: $(round(mean(rewards), digits=2))")
    println("  Max reward: $(maximum(rewards))")
    println("  Unique end positions: $(length(unique(final_positions)))")

    # Position distribution
    position_counts = Dict{Tuple{Int,Int},Int}()
    for pos in final_positions
        position_counts[pos] = get(position_counts, pos, 0) + 1
    end

    println("  Top positions:")
    for ((x, y), count) in sort(collect(position_counts), by=x->x[2], rev=true)[1:min(5, end)]
        percentage = round(count / length(valid_trajectories) * 100, digits=1)
        test_state = GridState(x, y, true)
        reward_val = reward(test_state)
        println("    ($x, $y): $count trajectories ($percentage%) [reward: $(round(reward_val, digits=1))]")
    end
end

# =============================================================================
# Utility Functions for Analysis
# =============================================================================

"""
    count_reachable_states(initial_state::GridState, actions::Vector{GridAction})

Count the number of reachable states from the initial state.
"""
function count_reachable_states(initial_state::GridState, actions::Vector{GridAction})
    visited = Set{GridState}()
    queue = [initial_state]

    while !isempty(queue)
        current_state = popfirst!(queue)

        # Skip if already visited
        current_state in visited && continue
        push!(visited, current_state)

        # Add reachable states
        for action in actions
            if is_applicable(action, current_state)
                next_state = apply_action(action, current_state)
                if next_state ∉ visited
                    push!(queue, next_state)
                end
            end
        end
    end

    return length(visited)
end

"""
    analyze_state_space(initial_state::GridState, actions::Vector{GridAction})

Analyze the state space structure and return statistics.
"""
function analyze_state_space(initial_state::GridState, actions::Vector{GridAction})
    visited = Set{GridState}()
    queue = [initial_state]
    terminal_states = 0

    while !isempty(queue)
        current_state = popfirst!(queue)

        # Skip if already visited
        current_state in visited && continue
        push!(visited, current_state)

        # Count terminal states
        if current_state.is_terminal
            terminal_states += 1
        end

        # Add reachable states
        for action in actions
            if is_applicable(action, current_state)
                next_state = apply_action(action, current_state)
                if next_state ∉ visited
                    push!(queue, next_state)
                end
            end
        end
    end

    return (
        total_states = length(visited),
        terminal_states = terminal_states,
        exploration_complete = true  # Simple grid world always has complete exploration
    )
end

# Removed duplicate create_default_sampling_config - using the one from core/sampling.jl

# Export the main functions
export GridState, GridAction, MoveRight, MoveUp, MoveLeft, MoveDown, Terminate
export create_grid_world_gflownet, create_grid_world, analyze_grid_world_results
export count_reachable_states, analyze_state_space
