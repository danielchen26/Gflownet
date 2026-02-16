# Grid World domain adapter for visualization
# Reference implementation of AbstractDomainAdapter interface

using GFlowNet: AbstractState, GFlowNetModel, Trajectory
using GFlowNet: GridState, GridAction, MoveRight, MoveUp, MoveLeft, MoveDown, Terminate
using GFlowNet: forward_action_probabilities, flow, reward
using LinearAlgebra: norm

"""
    GridWorldAdapter <: AbstractDomainAdapter

Visualization adapter for Grid World domain.
This is the reference implementation for other domain adapters.

# Fields
- `grid_size::Int`: Size of the grid (N×N)
- `reward_positions::Dict{Tuple{Int,Int}, Float64}`: Reward peaks and their values
"""
struct GridWorldAdapter <: AbstractDomainAdapter
    grid_size::Int
    reward_positions::Dict{Tuple{Int,Int}, Float64}
end

# ============================================
# Interface Implementation
# ============================================

"""
    state_to_viz_data(adapter, state)

Convert a GridState to JSON-serializable visualization data.
Returns a Dict with x, y, position, is_terminal, and reward.
"""
function state_to_viz_data(adapter::GridWorldAdapter, state::GridState)::Dict
    return Dict(
        "x" => state.x,
        "y" => state.y,
        "position" => [state.x, state.y],
        "is_terminal" => state.is_terminal,
        "reward" => state.is_terminal ? reward(state) : 0.0
    )
end

"""
    trajectory_to_viz_data(adapter, traj, id)

Convert a Trajectory to JSON-serializable visualization data.
Returns a Dict with id, states, actions, rewards, total_reward, and length.
"""
function trajectory_to_viz_data(adapter::GridWorldAdapter, traj::Trajectory, id::String)::Dict
    states_data  = [[s.x, s.y] for s in traj.states]
    actions_data = [action_to_string(a) for a in traj.actions]
    rewards_data = [reward(s) for s in traj.states]

    return Dict(
        "id"           => id,
        "states"       => states_data,
        "actions"      => actions_data,
        "rewards"      => rewards_data,
        "total_reward" => reward(traj.states[end]),
        "length"       => length(traj.actions)
    )
end

"""
    get_domain_config(adapter)

Get domain configuration for frontend.
Returns grid size, reward peaks, and capability flags.
"""
function get_domain_config(adapter::GridWorldAdapter)::Dict
    peaks = [
        Dict(
            "position"  => [x, y],
            "intensity" => r,
            "name"      => "Peak at ($x,$y)"
        )
        for ((x, y), r) in adapter.reward_positions
    ]
    return Dict(
        "domain_type"          => "grid_world",
        "grid_size"            => [adapter.grid_size, adapter.grid_size],
        "reward_peaks"         => peaks,
        "supports_flow_field"  => true,
        "supports_distribution"=> true,
        "coordinate_system"    => "cartesian_2d"
    )
end

"""
    get_renderer_name(adapter)

Get the frontend renderer component name.
For Grid World, this is "GridWorldRenderer".
"""
function get_renderer_name(adapter::GridWorldAdapter)::String
    return "GridWorldRenderer"
end

"""
    compute_domain_metrics(adapter, model, trajectories)

Compute Grid World-specific metrics: mode coverage, unique positions, top positions.
"""
function compute_domain_metrics(adapter::GridWorldAdapter, model::GFlowNetModel, trajectories::Vector{Trajectory})::Dict
    if isempty(trajectories)
        return Dict()
    end

    terminals = [t.states[end] for t in trajectories]

    # Mode coverage
    peaks_visited = Set{Tuple{Int,Int}}()
    for terminal in terminals
        pos = (terminal.x, terminal.y)
        if haskey(adapter.reward_positions, pos)
            push!(peaks_visited, pos)
        end
    end
    mode_coverage = length(peaks_visited) / max(length(adapter.reward_positions), 1)

    # Position distribution
    position_counts = Dict{Tuple{Int,Int}, Int}()
    for terminal in terminals
        pos = (terminal.x, terminal.y)
        position_counts[pos] = get(position_counts, pos, 0) + 1
    end
    sorted_positions = sort(collect(position_counts), by=x->x[2], rev=true)
    top_5 = first(sorted_positions, min(5, length(sorted_positions)))

    return Dict(
        "mode_coverage"    => mode_coverage,
        "modes_discovered" => length(peaks_visited),
        "total_modes"      => length(adapter.reward_positions),
        "unique_positions" => length(position_counts),
        "top_positions"    => [
            Dict("position" => [p[1], p[2]], "count" => c,
                 "percentage" => c / length(terminals) * 100)
            for (p, c) in top_5
        ]
    )
end

"""
    compute_flow_field(adapter, model)

Compute the flow field for visualization.
Returns flow data at each grid position with velocity vectors.

# Returns
- `Dict`: Contains "supported", "grid_size", and "data" (array of position/velocity/flow)
"""
function compute_flow_field(adapter::GridWorldAdapter, model::GFlowNetModel)::Dict
    grid_size = adapter.grid_size
    flow_data = []

    for x in 1:grid_size, y in 1:grid_size
        state = GridState(x, y, false)

        # ---- CORRECTED SIGNATURE ----
        # Actual: forward_action_probabilities(policy, state, actions, parameters, states)
        probs = forward_action_probabilities(
            model.forward_policy,
            state,
            model.all_actions,
            model.parameters.forward,
            model.states.forward
        )

        # Convert policy probabilities to velocity vector
        velocity = compute_velocity_from_policy(probs, model.all_actions)

        # Get flow value (flow(model, state) is the correct unified interface)
        flow_val = try
            flow(model, state)
        catch
            1.0
        end

        # Get reward at this position
        reward_val = get(adapter.reward_positions, (x, y), 0.0)

        push!(flow_data, Dict(
            "position"   => [x, y],
            "velocity"   => velocity,
            "magnitude"  => norm(velocity),
            "flow_value" => flow_val,
            "reward"     => reward_val
        ))
    end

    return Dict(
        "supported" => true,
        "grid_size" => grid_size,
        "data"      => flow_data
    )
end

"""
    compute_distribution_data(adapter, model, trajectories)

Compute empirical vs target distribution data for visualization.

# Returns
- `Dict`: Contains empirical distribution, target distribution, and counts
"""
function compute_distribution_data(adapter::GridWorldAdapter, model::GFlowNetModel, trajectories::Vector{Trajectory})::Dict
    grid_size = adapter.grid_size

    counts = zeros(Int, grid_size, grid_size)
    for traj in trajectories
        terminal = traj.states[end]
        if terminal.is_terminal
            counts[terminal.x, terminal.y] += 1
        end
    end

    total = sum(counts)
    probs = total > 0 ? counts ./ total : Float64.(counts)

    # Target distribution P(x) proportional to R(x)
    target = zeros(grid_size, grid_size)
    for x in 1:grid_size, y in 1:grid_size
        state = GridState(x, y, true)
        target[x, y] = reward(state)
    end
    target_sum = sum(target)
    target = target_sum > 0 ? target ./ target_sum : target

    # Build terminal_distribution as dictionary with string keys "x,y" => count
    # This is the format expected by the frontend for the Endpoint Distribution panel
    terminal_distribution = Dict{String, Int}()
    for x in 1:grid_size, y in 1:grid_size
        if counts[x, y] > 0
            terminal_distribution["$x,$y"] = counts[x, y]
        end
    end

    # Count unique endpoints
    unique_endpoints = sum(counts .> 0)

    # Build reward_peaks array in the format expected by frontend
    reward_peaks = [
        Dict(
            "position"  => [x, y],
            "intensity" => r,
            "name"      => "Peak $(Char('A' + i - 1))"
        )
        for (i, ((x, y), r)) in enumerate(adapter.reward_positions)
    ]

    return Dict(
        "supported"            => true,
        "grid_size"            => grid_size,
        "empirical"            => probs,
        "target"               => target,
        "counts"               => counts,
        "total_samples"        => total,
        # Additional fields for frontend Endpoint Distribution panel
        "terminal_distribution"=> terminal_distribution,
        "total_trajectories"   => total,
        "unique_endpoints"     => unique_endpoints,
        "reward_peaks"         => reward_peaks
    )
end

# ============================================
# Reward Shaping (Domain-Agnostic Interface)
# ============================================

"""
    apply_reward_shaping(adapter::GridWorldAdapter, reward_positions::Dict{Tuple{Int,Int}, Float64})

Grid World-specific reward shaping to compensate for structural path asymmetry.

In a grid world with only Right/Up moves from (1,1), the number of unique paths
to position (x,y) is given by `binomial(x+y-2, x-1)`. This creates extreme
asymmetry: e.g., 70:1 for 5×5 grid, 3432:1 for 8×8 grid.

Scales each reward inversely to its path count so that all modes
receive equal total reward flow, making minority modes discoverable.

Formula: `adjusted_R = R × (max_paths / path_count_for_position)`

See: `docs/src/internals/implementation_notes/mode_collapse_test_results.md`
"""
function apply_reward_shaping(adapter::GridWorldAdapter, reward_positions::Dict{Tuple{Int,Int}, Float64})
    if length(reward_positions) <= 1
        return reward_positions
    end

    # Compute path counts using binomial coefficient
    # paths to (x,y) from (1,1) with Right/Up moves = binomial(x+y-2, x-1)
    path_counts = Dict{Tuple{Int,Int}, Int}()
    for (pos, _) in reward_positions
        x, y = pos
        path_counts[pos] = binomial(x + y - 2, x - 1)
    end
    max_paths = maximum(values(path_counts))

    # Scale rewards inversely to path counts
    shaped = Dict{Tuple{Int,Int}, Float64}()
    for (pos, r) in reward_positions
        pc = path_counts[pos]
        shaped[pos] = pc > 0 ? r * (max_paths / pc) : r
    end

    @info "Grid world reward shaping applied" shaped path_counts max_paths
    return shaped
end

# ============================================
# Helper Functions
# ============================================

"""
    action_to_string(action)

Convert a GridAction to a string representation for visualization.
"""
function action_to_string(action)::String
    action isa MoveRight  && return "right"
    action isa MoveUp     && return "up"
    action isa MoveLeft   && return "left"
    action isa MoveDown   && return "down"
    action isa Terminate   && return "terminate"
    return "unknown"
end

"""
    compute_velocity_from_policy(probs, actions)

Convert policy probabilities to a 2D velocity vector for flow field visualization.

# Arguments
- `probs`: Action probabilities from the policy
- `actions`: List of actions corresponding to probabilities

# Returns
- `Vector{Float64}`: [vx, vy] velocity vector
"""
function compute_velocity_from_policy(probs, actions)
    vx, vy = 0.0, 0.0
    for (i, action) in enumerate(actions)
        p = i <= length(probs) ? Float64(probs[i]) : 0.0
        if action isa MoveRight
            vx += p
        elseif action isa MoveLeft
            vx -= p
        elseif action isa MoveUp
            vy += p
        elseif action isa MoveDown
            vy -= p
        end
    end
    return [vx, vy]
end

# ============================================
# Domain Registry Interface (Phase 1)
# ============================================

"""Get unique domain identifier"""
function get_domain_id(adapter::GridWorldAdapter)::String
    return "grid_world"
end

"""Get human-readable domain description"""
function get_domain_description(adapter::GridWorldAdapter)::String
    return "2D grid navigation with configurable rewards and obstacles. Perfect for learning GFlowNet fundamentals."
end

"""Get JSON Schema for domain configuration"""
function get_config_schema(adapter::GridWorldAdapter)::Dict
    return Dict(
        "type" => "object",
        "properties" => Dict(
            "grid_size" => Dict(
                "type" => "integer",
                "description" => "Size of the grid (NxN)",
                "default" => 8,
                "minimum" => 4,
                "maximum" => 32
            ),
            "reward_peaks" => Dict(
                "type" => "array",
                "description" => "Positions and values of reward peaks",
                "items" => Dict(
                    "type" => "object",
                    "properties" => Dict(
                        "position" => Dict("type" => "array", "items" => Dict("type" => "integer")),
                        "intensity" => Dict("type" => "number")
                    )
                )
            ),
            "obstacles" => Dict(
                "type" => "array",
                "description" => "Positions of obstacles",
                "items" => Dict(
                    "type" => "array",
                    "items" => Dict("type" => "integer")
                )
            )
        ),
        "required" => ["grid_size"]
    )
end

"""Validate domain configuration against schema"""
function validate_config(adapter::GridWorldAdapter, config::Dict)::Tuple{Bool, Union{String, Nothing}}
    # Check grid_size
    if haskey(config, "grid_size")
        gs = config["grid_size"]
        if !isa(gs, Integer) || gs < 4 || gs > 32
            return (false, "grid_size must be an integer between 4 and 32")
        end
    end

    # Check reward_peaks if present
    if haskey(config, "reward_peaks")
        rp = config["reward_peaks"]
        if !isa(rp, Vector)
            return (false, "reward_peaks must be an array")
        end
        for (i, peak) in enumerate(rp)
            if !haskey(peak, "position") || !haskey(peak, "intensity")
                return (false, "reward_peaks[$i] must have position and intensity")
            end
        end
    end

    return (true, nothing)
end

"""Create GridWorldAdapter from configuration"""
function create_from_config(::Type{GridWorldAdapter}, config::Dict)::GridWorldAdapter
    grid_size = get(config, "grid_size", 8)

    # Parse reward peaks
    reward_positions = Dict{Tuple{Int,Int}, Float64}()
    if haskey(config, "reward_peaks")
        for peak in config["reward_peaks"]
            pos = peak["position"]
            intensity = peak["intensity"]
            reward_positions[(pos[1], pos[2])] = Float64(intensity)
        end
    else
        # Default: corner reward
        reward_positions[(grid_size, grid_size)] = 10.0
    end

    return GridWorldAdapter(grid_size, reward_positions)
end

# Default constructor for metadata queries
function GridWorldAdapter()
    return GridWorldAdapter(8, Dict((8, 8) => 10.0))
end

"""Get domain tags for categorization"""
function get_domain_tags(adapter::GridWorldAdapter)::Vector{String}
    return ["navigation", "2d", "beginner"]
end

"""Check if domain is a built-in domain"""
function is_builtin_domain(adapter::GridWorldAdapter)::Bool
    return true
end

"""Check if domain is marked as popular"""
function is_popular_domain(adapter::GridWorldAdapter)::Bool
    return true
end
