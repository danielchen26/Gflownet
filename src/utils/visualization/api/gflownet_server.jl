# GFlowNet-Specific Visualization API Server
# Provides realistic GFlowNet data for visualization

using Oxygen
using HTTP
using JSON3
using Dates
using Statistics
using Random
using UUIDs

# Set random seed for reproducibility
Random.seed!(42)

# GFlowNet Domain: Grid World Example
# States are positions on a 10x10 grid
# Actions: move up, down, left, right
# Goal: reach high-reward regions

struct GridState
    x::Int
    y::Int
    is_terminal::Bool
end

# Define reward function (peaks at certain locations)
function compute_reward(state::GridState)
    # Base reward is very small to encourage exploration
    reward = 0.1
    
    # Add contribution from each peak
    for peak in PROBLEM_CONFIG[].reward_peaks
        px, py = peak["position"]
        intensity = peak["intensity"]
        dist = sqrt((state.x - px)^2 + (state.y - py)^2)
        
        # Gaussian-like reward with configurable spread
        if dist < 0.5  # At the exact peak location
            reward += intensity
        else
            # Decay with distance, but with a cutoff to avoid too wide spread
            reward += intensity * exp(-dist^2 / 4.0) * (dist < 4.0)
        end
    end
    
    return min(reward, 10.0)  # Cap maximum reward at 10
end

# Generate a trajectory from start to a high-reward region
function generate_gflownet_trajectory(trajectory_id::Int)
    # Random starting position
    start_x = rand(1:3)
    start_y = rand(1:3)
    
    states = []
    actions = []
    rewards = []
    flows = []
    
    # Current position
    x, y = start_x, start_y
    
    # Generate trajectory towards high-reward region
    for step in 1:20
        state = GridState(x, y, false)
        push!(states, Dict(
            "position" => [x, y, 0.1 * step],  # 3D with time as z
            "grid_position" => [x, y],
            "features" => [x/10, y/10, step/20],
            "id" => "s_$(trajectory_id)_$step",
            "is_terminal" => false
        ))
        
        reward = compute_reward(state)
        push!(rewards, reward)
        
        # Estimate flow (increases towards high-reward regions)
        flow = reward * (1 + step / 10)
        push!(flows, flow)
        
        # Choose action (biased towards high-reward regions)
        if step < 20
            # Find best direction
            best_reward = -Inf
            best_action = "stay"
            
            for (action, dx, dy) in [("up", 0, 1), ("down", 0, -1), ("left", -1, 0), ("right", 1, 0)]
                new_x, new_y = x + dx, y + dy
                if 1 <= new_x <= PROBLEM_CONFIG[].grid_size && 1 <= new_y <= PROBLEM_CONFIG[].grid_size
                    new_reward = compute_reward(GridState(new_x, new_y, false))
                    if new_reward > best_reward
                        best_reward = new_reward
                        best_action = action
                    end
                end
            end
            
            # Add some randomness
            if rand() < 0.3
                actions_list = ["up", "down", "left", "right"]
                best_action = rand(actions_list)
            end
            
            push!(actions, Dict(
                "type" => best_action,
                "id" => "a_$(trajectory_id)_$step"
            ))
            
            # Move
            if best_action == "up" && y < PROBLEM_CONFIG[].grid_size
                y += 1
            elseif best_action == "down" && y > 1
                y -= 1
            elseif best_action == "left" && x > 1
                x -= 1
            elseif best_action == "right" && x < PROBLEM_CONFIG[].grid_size
                x += 1
            end
        end
    end
    
    # Terminal state
    push!(states, Dict(
        "position" => [x, y, 2.1],
        "grid_position" => [x, y],
        "features" => [x/10, y/10, 1.0],
        "id" => "s_$(trajectory_id)_terminal",
        "is_terminal" => true
    ))
    push!(rewards, compute_reward(GridState(x, y, true)))
    push!(flows, rewards[end] * 2)
    
    return Dict(
        "id" => string(uuid1()),
        "trajectory_id" => trajectory_id,
        "states" => states,
        "actions" => actions,
        "rewards" => rewards,
        "flows" => flows,
        "total_reward" => sum(rewards),
        "start_position" => [start_x, start_y],
        "end_position" => [x, y],
        "length" => length(states),
        "timestamp" => now()
    )
end

# Generate training history with realistic GFlowNet metrics
function generate_training_data()
    n_episodes = 200
    episodes = Float64[]
    losses = Float64[]
    rewards = Float64[]
    tb_losses = Float64[]
    flow_losses = Float64[]
    state_coverages = Float64[]  # Fraction of state space visited
    mode_coverages = Float64[]   # How many reward peaks are being sampled
    kl_divergences = Float64[]   # KL divergence from target distribution
    
    for i in 1:n_episodes
        push!(episodes, Float64(i))
        
        # Losses decrease over time
        base_loss = 5.0 / (1 + i * 0.05)
        push!(losses, base_loss + 0.2 * randn())
        push!(tb_losses, base_loss * 0.8 + 0.15 * randn())
        push!(flow_losses, base_loss * 1.2 + 0.25 * randn())
        
        # Rewards increase over time
        base_reward = 15 * (1 - exp(-i/50))
        push!(rewards, base_reward + 2 * randn())
        
        # State coverage increases as model learns
        coverage = min(0.95, 0.1 + 0.85 * (1 - exp(-i/30)))
        push!(state_coverages, coverage + 0.05 * randn())
        
        # Mode coverage (how many peaks discovered)
        modes = min(1.0, 0.25 + 0.75 * (1 - exp(-i/40)))
        push!(mode_coverages, modes)
        
        # KL divergence decreases as model improves
        kl = max(0.01, 2.0 * exp(-i/50) + 0.1 * randn())
        push!(kl_divergences, kl)
    end
    
    return Dict(
        "episodes" => episodes,
        "losses" => losses,
        "rewards" => rewards,
        "tb_losses" => tb_losses,
        "flow_losses" => flow_losses,
        "state_coverages" => state_coverages,
        "mode_coverages" => mode_coverages,
        "kl_divergences" => kl_divergences,
        "metrics" => Dict(
            "mean_loss" => mean(losses[end-10:end]),
            "mean_reward" => mean(rewards[end-10:end]),
            "mean_tb_loss" => mean(tb_losses[end-10:end]),
            "mean_flow_loss" => mean(flow_losses[end-10:end]),
            "state_coverage" => state_coverages[end],
            "mode_coverage" => mode_coverages[end],
            "kl_divergence" => kl_divergences[end],
            "total_episodes" => n_episodes,
            "convergence_estimate" => 1 - losses[end] / losses[1]
        )
    )
end

# Generate flow field for the grid world
function generate_flow_field()
    resolution = PROBLEM_CONFIG[].grid_size
    flow_data = []
    
    for x in 1:resolution
        for y in 1:resolution
            state = GridState(x, y, false)
            reward = compute_reward(state)
            
            # Flow direction points towards higher rewards
            grad_x = 0.0
            grad_y = 0.0
            
            # Compute gradient
            if x > 1
                grad_x += compute_reward(GridState(x-1, y, false)) - reward
            end
            if x < resolution
                grad_x += compute_reward(GridState(x+1, y, false)) - reward
            end
            if y > 1
                grad_y += compute_reward(GridState(x, y-1, false)) - reward
            end
            if y < resolution
                grad_y += compute_reward(GridState(x, y+1, false)) - reward
            end
            
            # Normalize
            mag = sqrt(grad_x^2 + grad_y^2) + 0.01
            
            push!(flow_data, Dict(
                "position" => [x, y, 0],
                "velocity" => [grad_x/mag, grad_y/mag, 0],
                "magnitude" => reward,
                "reward" => reward,
                "flow_value" => reward * 10  # Estimated flow
            ))
        end
    end
    
    return Dict(
        "resolution" => [resolution, resolution, 1],
        "bounds" => Dict(
            "x" => [1, resolution],
            "y" => [1, resolution],
            "z" => [0, 1]
        ),
        "data" => flow_data,
        "reward_peaks" => PROBLEM_CONFIG[].reward_peaks
    )
end

# Generate state visitation statistics
function generate_state_statistics(trajectories)
    visitation_counts = Dict{Tuple{Int,Int}, Int}()
    terminal_counts = Dict{Tuple{Int,Int}, Int}()
    value_estimates = Dict{Tuple{Int,Int}, Float64}()
    
    # Track terminal state distribution
    for traj in trajectories
        for (i, state) in enumerate(traj["states"])
            pos = tuple(state["grid_position"]...)
            visitation_counts[pos] = get(visitation_counts, pos, 0) + 1
            
            # Track terminal states separately
            if state["is_terminal"]
                terminal_counts[pos] = get(terminal_counts, pos, 0) + 1
            end
            
            # Value estimate based on future rewards
            future_reward = sum(traj["rewards"][i:end])
            current_value = get(value_estimates, pos, 0.0)
            value_estimates[pos] = 0.9 * current_value + 0.1 * future_reward
        end
    end
    
    # Calculate how well we're covering reward peaks
    peaks_covered = 0
    for peak in PROBLEM_CONFIG[].reward_peaks
        px, py = peak["position"]
        # Check if we've visited near this peak
        for dx in -1:1, dy in -1:1
            if haskey(terminal_counts, (px + dx, py + dy))
                peaks_covered += 1
                break
            end
        end
    end
    mode_coverage = length(PROBLEM_CONFIG[].reward_peaks) > 0 ? peaks_covered / length(PROBLEM_CONFIG[].reward_peaks) : 0.0
    
    # Calculate sampling distribution entropy (higher = more diverse)
    total_terminals = sum(values(terminal_counts))
    entropy = 0.0
    if total_terminals > 0
        for count in values(terminal_counts)
            p = count / total_terminals
            if p > 0
                entropy -= p * log(p)
            end
        end
    end
    
    return Dict(
        "visitation_counts" => visitation_counts,
        "terminal_counts" => terminal_counts,
        "value_estimates" => value_estimates,
        "total_states_visited" => length(visitation_counts),
        "unique_terminals" => length(terminal_counts),
        "max_visits" => length(visitation_counts) > 0 ? maximum(values(visitation_counts)) : 0,
        "coverage" => length(visitation_counts) / (PROBLEM_CONFIG[].grid_size^2),
        "mode_coverage" => mode_coverage,
        "sampling_entropy" => entropy,
        "total_trajectories" => length(trajectories)
    )
end

# Dynamic problem configuration
mutable struct ProblemConfig
    grid_size::Int
    reward_peaks::Vector{Dict{String, Any}}
    training_objective::String
    n_episodes::Int
end

# Default configuration
const DEFAULT_CONFIG = ProblemConfig(
    10,
    [
        Dict("position" => [8, 8], "intensity" => 10.0, "name" => "Primary Goal"),
        Dict("position" => [2, 8], "intensity" => 8.0, "name" => "Secondary Goal"),
        Dict("position" => [5, 5], "intensity" => 6.0, "name" => "Center Reward"),
        Dict("position" => [8, 2], "intensity" => 7.0, "name" => "Corner Reward")
    ],
    "TB",
    1000
)

# Global problem configuration
const PROBLEM_CONFIG = Ref(deepcopy(DEFAULT_CONFIG))

# Store generated data - start with empty for dynamic generation
const TRAJECTORIES = []
const TRAINING_STATE = Dict(
    "is_training" => false,
    "episode" => 0,
    "config" => nothing,
    "start_time" => now(),
    "session_id" => ""
)
const TRAINING_HISTORY = Dict(
    "episodes" => Float64[],
    "losses" => Float64[],
    "rewards" => Float64[],
    "tb_losses" => Float64[],
    "flow_losses" => Float64[],
    "state_coverages" => Float64[],
    "mode_coverages" => Float64[],
    "kl_divergences" => Float64[]
)

# Initialize with a few trajectories
for i in 1:3
    push!(TRAJECTORIES, generate_gflownet_trajectory(i))
end

# Initial flow field will be regenerated on demand
const STATE_STATS = generate_state_statistics(TRAJECTORIES)

# API Routes
@get "/health" function()
    json(Dict(
        "status" => "healthy",
        "timestamp" => now(),
        "version" => "2.0.0",
        "mode" => "GFlowNet Grid World"
    ))
end

@get "/api/trajectories" function(req::HTTP.Request)
    # Check if it's for 3D visualization from query param
    params = HTTP.queryparams(HTTP.URI(req.target))
    is_3d = get(params, "view", "") == "3d"
    
    # Use current problem configuration
    current_grid_size = PROBLEM_CONFIG[].grid_size
    current_peaks = PROBLEM_CONFIG[].reward_peaks
    
    if is_3d
        # Return full format for 3D visualization
        json(Dict(
            "trajectories" => TRAJECTORIES,
            "grid_size" => [current_grid_size, current_grid_size],
            "reward_peaks" => current_peaks
        ))
    else
        # Format trajectories for 2D visualization
        formatted_trajectories = map(TRAJECTORIES) do traj
            states_2d = map(s -> [s["grid_position"][1], s["grid_position"][2]], traj["states"])
            Dict(
                "id" => traj["id"],
                "states" => states_2d,
                "actions" => map(a -> a["type"], traj["actions"]),
                "rewards" => traj["rewards"],
                "total_reward" => traj["total_reward"],
                "length" => length(traj["states"])
            )
        end
        
        json(Dict(
            "trajectories" => formatted_trajectories,
            "grid_size" => [current_grid_size, current_grid_size],
            "reward_peaks" => current_peaks,
            "count" => length(TRAJECTORIES),
            "domain" => "Grid World",
            "state_space" => "$(current_grid_size)x$(current_grid_size) grid",
            "action_space" => ["up", "down", "left", "right"]
        ))
    end
end

@get "/api/trajectories/{id}" function(req::HTTP.Request, id::String)
    trajectory = findfirst(t -> t["id"] == id, TRAJECTORIES)
    
    if isnothing(trajectory)
        return json(Dict("error" => "Trajectory not found"), status=404)
    end
    
    json(TRAJECTORIES[trajectory])
end

@get "/api/training/history" function()
    # Return current training history
    metrics = Dict()
    if length(TRAINING_HISTORY["episodes"]) > 0
        last_n = min(10, length(TRAINING_HISTORY["episodes"]))
        metrics = Dict(
            "mean_loss" => mean(TRAINING_HISTORY["losses"][end-last_n+1:end]),
            "mean_reward" => mean(TRAINING_HISTORY["rewards"][end-last_n+1:end]),
            "mean_tb_loss" => mean(TRAINING_HISTORY["tb_losses"][end-last_n+1:end]),
            "mean_flow_loss" => mean(TRAINING_HISTORY["flow_losses"][end-last_n+1:end]),
            "state_coverage" => length(TRAINING_HISTORY["state_coverages"]) > 0 ? TRAINING_HISTORY["state_coverages"][end] : 0.1,
            "mode_coverage" => length(TRAINING_HISTORY["mode_coverages"]) > 0 ? TRAINING_HISTORY["mode_coverages"][end] : 0.0,
            "kl_divergence" => length(TRAINING_HISTORY["kl_divergences"]) > 0 ? TRAINING_HISTORY["kl_divergences"][end] : 2.0,
            "total_episodes" => TRAINING_STATE["episode"],
            "convergence_estimate" => min(0.95, TRAINING_STATE["episode"] / 1000.0)
        )
    else
        metrics = Dict(
            "mean_loss" => 0.0,
            "mean_reward" => 0.0,
            "mean_tb_loss" => 0.0,
            "mean_flow_loss" => 0.0,
            "state_coverage" => 0.1,
            "mode_coverage" => 0.0,
            "kl_divergence" => 2.0,
            "total_episodes" => 0,
            "convergence_estimate" => 0.0
        )
    end
    
    # For backward compatibility, keep exploration_rates as state_coverages
    json(Dict(
        "episodes" => TRAINING_HISTORY["episodes"],
        "losses" => TRAINING_HISTORY["losses"],
        "rewards" => TRAINING_HISTORY["rewards"],
        "tb_losses" => TRAINING_HISTORY["tb_losses"],
        "flow_losses" => TRAINING_HISTORY["flow_losses"],
        "exploration_rates" => TRAINING_HISTORY["state_coverages"],  # Map to state coverage
        "state_coverages" => TRAINING_HISTORY["state_coverages"],
        "mode_coverages" => TRAINING_HISTORY["mode_coverages"],
        "kl_divergences" => TRAINING_HISTORY["kl_divergences"],
        "metrics" => metrics
    ))
end

@get "/api/training/metrics" function()
    if length(TRAINING_HISTORY["episodes"]) > 0
        latest_idx = length(TRAINING_HISTORY["episodes"])
        # Get current state statistics
        current_stats = generate_state_statistics(TRAJECTORIES)
        
        json(Dict(
            "latest_loss" => TRAINING_HISTORY["losses"][latest_idx],
            "latest_reward" => TRAINING_HISTORY["rewards"][latest_idx],
            "latest_tb_loss" => TRAINING_HISTORY["tb_losses"][latest_idx],
            "latest_flow_loss" => TRAINING_HISTORY["flow_losses"][latest_idx],
            "exploration_rate" => TRAINING_HISTORY["state_coverages"][latest_idx],  # For backward compatibility
            "state_coverage" => current_stats["coverage"],
            "mode_coverage" => current_stats["mode_coverage"],
            "unique_terminals" => current_stats["unique_terminals"],
            "sampling_entropy" => current_stats["sampling_entropy"],
            "total_episodes" => TRAINING_STATE["episode"],
            "convergence" => min(0.95, TRAINING_STATE["episode"] / 1000.0),
            "is_training" => TRAINING_STATE["is_training"]
        ))
    else
        json(Dict(
            "latest_loss" => 0.0,
            "latest_reward" => 0.0,
            "latest_tb_loss" => 0.0,
            "latest_flow_loss" => 0.0,
            "exploration_rate" => 0.1,  # For backward compatibility
            "state_coverage" => 0.1,
            "mode_coverage" => 0.0,
            "unique_terminals" => 0,
            "sampling_entropy" => 0.0,
            "total_episodes" => 0,
            "convergence" => 0.0,
            "is_training" => false
        ))
    end
end

@get "/api/analysis/flow-field" function()
    # Regenerate flow field with current configuration
    updated_flow_field = generate_flow_field()
    json(updated_flow_field)
end

@get "/api/analysis/state-statistics" function()
    json(STATE_STATS)
end

@get "/api/domain/info" function()
    current_size = PROBLEM_CONFIG[].grid_size
    json(Dict(
        "name" => "Grid World",
        "description" => "$(current_size)x$(current_size) grid with multiple reward peaks. Agent learns to navigate from start to high-reward regions.",
        "state_space" => Dict(
            "type" => "discrete",
            "size" => [current_size, current_size],
            "total_states" => current_size * current_size
        ),
        "action_space" => Dict(
            "type" => "discrete", 
            "actions" => ["up", "down", "left", "right"],
            "size" => 4
        ),
        "reward_info" => Dict(
            "type" => "continuous",
            "range" => [0, 10],
            "peaks" => PROBLEM_CONFIG[].reward_peaks
        ),
        "objectives" => ["TRAJECTORY_BALANCE", "DETAILED_BALANCE", "FLOW_MATCHING"]
    ))
end

# Simulate new trajectory generation
@post "/api/trajectories/sample" function()
    new_id = length(TRAJECTORIES) + 1
    new_trajectory = generate_gflownet_trajectory(new_id)
    push!(TRAJECTORIES, new_trajectory)
    
    # Update statistics
    global STATE_STATS = generate_state_statistics(TRAJECTORIES)
    
    json(new_trajectory)
end

# Note: WebSocket endpoint removed - using HTTP polling for real-time updates

# Get all trajectories for distribution visualization
@get "/api/trajectories/all" function()
    json(Dict(
        "trajectories" => map(TRAJECTORIES) do traj
            states_2d = map(s -> [s["grid_position"][1], s["grid_position"][2]], traj["states"])
            Dict(
                "id" => traj["id"],
                "states" => states_2d,
                "rewards" => traj["rewards"],
                "total_reward" => traj["total_reward"]
            )
        end,
        "reward_peaks" => PROBLEM_CONFIG[].reward_peaks,
        "grid_size" => PROBLEM_CONFIG[].grid_size
    ))
end

# Distribution analysis
@get "/api/analysis/distribution" function()
    # Calculate empirical distribution from trajectories
    endpoints = Dict{String, Int}()
    terminal_rewards = Float64[]
    
    for traj in TRAJECTORIES
        last_state = traj["states"][end]
        pos = last_state["grid_position"]
        key = "$(pos[1]),$(pos[2])"
        endpoints[key] = get(endpoints, key, 0) + 1
        push!(terminal_rewards, traj["total_reward"])
    end
    
    # Normalize to get probabilities
    total_samples = length(TRAJECTORIES)
    empirical_dist = Dict(k => v / total_samples for (k, v) in endpoints)
    
    # Calculate target distribution P(x) ∝ R(x) over the entire grid
    target_dist = Dict{String, Float64}()
    total_target_mass = 0.0
    
    # For a grid world, compute rewards at all possible states
    grid_size = PROBLEM_CONFIG[].grid_size
    for x in 1:grid_size
        for y in 1:grid_size
            state = GridState(x, y, false)
            reward = compute_reward(state)
            if reward > 0.1  # Only include states with meaningful reward
                key = "$x,$y"
                target_dist[key] = reward
                total_target_mass += reward
            end
        end
    end
    
    # Normalize target distribution
    if total_target_mass > 0
        for k in keys(target_dist)
            target_dist[k] /= total_target_mass
        end
    end
    
    # Calculate KL divergence: KL(empirical || target)
    kl_divergence = 0.0
    for (pos, emp_prob) in empirical_dist
        target_prob = get(target_dist, pos, 1e-10)  # Small epsilon to avoid log(0)
        if emp_prob > 0
            kl_divergence += emp_prob * log(emp_prob / target_prob)
        end
    end
    
    # Add penalty for unvisited high-reward states
    for (pos, target_prob) in target_dist
        if !haskey(empirical_dist, pos) && target_prob > 0.01
            # Add a penalty for missing important states
            kl_divergence += target_prob  # Simplified penalty
        end
    end
    
    # Check mode coverage - how many reward peaks are visited
    visited_peaks = 0
    for peak in PROBLEM_CONFIG[].reward_peaks
        key = "$(peak["position"][1]),$(peak["position"][2])"
        if haskey(endpoints, key) && endpoints[key] > 0
            visited_peaks += 1
        end
    end
    
    json(Dict(
        "total_trajectories" => total_samples,
        "unique_endpoints" => length(keys(endpoints)),
        "diversity_score" => 1.0 - (isempty(endpoints) ? 0 : maximum(values(endpoints)) / total_samples),
        "empirical_distribution" => empirical_dist,
        "target_distribution" => target_dist,
        "terminal_distribution" => endpoints,
        "kl_divergence" => isnan(kl_divergence) || isinf(kl_divergence) ? 10.0 : kl_divergence,
        "reward_coverage" => visited_peaks / length(PROBLEM_CONFIG[].reward_peaks),
        "mean_terminal_reward" => total_samples > 0 ? mean(terminal_rewards) : 0.0,
        "grid_size" => grid_size,
        "reward_peaks" => PROBLEM_CONFIG[].reward_peaks
    ))
end

# Start training endpoint
@post "/api/training/start" function(req::HTTP.Request)
    body = JSON3.read(IOBuffer(HTTP.payload(req)))
    
    # Update problem configuration
    PROBLEM_CONFIG[].grid_size = get(body, :grid_size, 10)
    PROBLEM_CONFIG[].training_objective = get(body, :training_objective, "TB")
    PROBLEM_CONFIG[].n_episodes = get(body, :n_episodes, 1000)
    
    # Update reward peaks with names
    peaks = get(body, :reward_peaks, [])
    PROBLEM_CONFIG[].reward_peaks = [
        Dict(
            "position" => [peak[:position][1], peak[:position][2]], 
            "intensity" => peak[:intensity],
            "name" => "Peak $(Char(65 + i - 1))"  # A, B, C, etc.
        ) for (i, peak) in enumerate(peaks)
    ]
    
    # Note: Flow field and state statistics will be regenerated on demand
    # when the respective API endpoints are called with the new reward configuration
    
    # Update training state
    TRAINING_STATE["is_training"] = true
    TRAINING_STATE["episode"] = 0
    TRAINING_STATE["config"] = body
    TRAINING_STATE["start_time"] = now()
    TRAINING_STATE["session_id"] = string(UUIDs.uuid4())
    
    # Clear previous data
    empty!(TRAINING_HISTORY["episodes"])
    empty!(TRAINING_HISTORY["losses"])
    empty!(TRAINING_HISTORY["rewards"])
    empty!(TRAINING_HISTORY["tb_losses"])
    empty!(TRAINING_HISTORY["flow_losses"])
    empty!(TRAINING_HISTORY["state_coverages"])
    empty!(TRAINING_HISTORY["mode_coverages"])
    empty!(TRAINING_HISTORY["kl_divergences"])
    empty!(TRAJECTORIES)
    
    # Start background training simulation
    @async begin
        n_episodes = PROBLEM_CONFIG[].n_episodes
        
        for episode in 1:n_episodes
            if !TRAINING_STATE["is_training"]
                break
            end
            
            # Update episode count
            TRAINING_STATE["episode"] = episode
            
            # Generate new trajectory
            new_traj = generate_gflownet_trajectory(length(TRAJECTORIES) + 1)
            push!(TRAJECTORIES, new_traj)
            
            # Simulate training metrics based on selected objective
            base_loss = 5.0 / (1 + episode * 0.05)
            push!(TRAINING_HISTORY["episodes"], Float64(episode))
            
            # Generate losses based on training objective
            objective = PROBLEM_CONFIG[].training_objective
            if objective == "TB"
                # Trajectory Balance only
                tb_loss = base_loss + 0.2 * randn()
                push!(TRAINING_HISTORY["losses"], tb_loss)
                push!(TRAINING_HISTORY["tb_losses"], tb_loss)
                push!(TRAINING_HISTORY["flow_losses"], 0.0)  # Not used
            elseif objective == "DB"
                # Detailed Balance only
                db_loss = base_loss * 0.9 + 0.15 * randn()
                push!(TRAINING_HISTORY["losses"], db_loss)
                push!(TRAINING_HISTORY["tb_losses"], 0.0)  # Not used
                push!(TRAINING_HISTORY["flow_losses"], db_loss)
            elseif objective == "SUB_TB"
                # Sub-trajectory Balance
                sub_tb_loss = base_loss * 0.85 + 0.18 * randn()
                push!(TRAINING_HISTORY["losses"], sub_tb_loss)
                push!(TRAINING_HISTORY["tb_losses"], sub_tb_loss)
                push!(TRAINING_HISTORY["flow_losses"], 0.0)  # Not used
            else
                # Default to TB
                tb_loss = base_loss + 0.2 * randn()
                push!(TRAINING_HISTORY["losses"], tb_loss)
                push!(TRAINING_HISTORY["tb_losses"], tb_loss)
                push!(TRAINING_HISTORY["flow_losses"], 0.0)
            end
            
            # Rewards increase over time but stay realistic
            # Sample actual trajectory reward
            actual_reward = new_traj["total_reward"]
            # Add some noise but keep it realistic
            noisy_reward = actual_reward + 0.5 * randn()
            push!(TRAINING_HISTORY["rewards"], max(0.1, noisy_reward))
            
            # Calculate proper GFlowNet metrics
            # State coverage increases as model learns
            coverage = min(0.95, 0.1 + 0.85 * (1 - exp(-episode/30)))
            push!(TRAINING_HISTORY["state_coverages"], coverage + 0.05 * randn())
            
            # Mode coverage (how many peaks discovered)
            current_stats = generate_state_statistics(TRAJECTORIES)
            push!(TRAINING_HISTORY["mode_coverages"], current_stats["mode_coverage"])
            
            # KL divergence from target distribution (decreases over time)
            kl = max(0.01, 2.0 * exp(-episode/50) + 0.1 * randn())
            push!(TRAINING_HISTORY["kl_divergences"], kl)
            
            # Keep only recent trajectories (last 50)
            if length(TRAJECTORIES) > 50
                popfirst!(TRAJECTORIES)
            end
            
            # Simulate training delay
            sleep(0.1)  # 10 episodes per second
        end
        
        TRAINING_STATE["is_training"] = false
    end
    
    json(Dict(
        "status" => "started",
        "config" => TRAINING_STATE["config"],
        "session_id" => TRAINING_STATE["session_id"]
    ))
end

# Stop training endpoint
@post "/api/training/stop" function()
    TRAINING_STATE["is_training"] = false
    json(Dict("status" => "stopped"))
end

# Reset training endpoint
@post "/api/training/reset" function()
    TRAINING_STATE["is_training"] = false
    TRAINING_STATE["episode"] = 0
    TRAINING_STATE["start_time"] = now()
    
    # Clear data
    empty!(TRAINING_HISTORY["episodes"])
    empty!(TRAINING_HISTORY["losses"])
    empty!(TRAINING_HISTORY["rewards"])
    empty!(TRAINING_HISTORY["tb_losses"])
    empty!(TRAINING_HISTORY["flow_losses"])
    empty!(TRAINING_HISTORY["state_coverages"])
    empty!(TRAINING_HISTORY["mode_coverages"])
    empty!(TRAINING_HISTORY["kl_divergences"])
    empty!(TRAJECTORIES)
    
    # Add initial trajectories
    for i in 1:3
        push!(TRAJECTORIES, generate_gflownet_trajectory(i))
    end
    
    json(Dict(
        "status" => "reset",
        "message" => "Training reset successfully",
        "timestamp" => now()
    ))
end

# Get current training state
@get "/api/training/state" function()
    elapsed = Dates.value(now() - TRAINING_STATE["start_time"]) / 1000  # seconds
    
    if length(TRAINING_HISTORY["episodes"]) > 0
        idx = length(TRAINING_HISTORY["episodes"])
        current_stats = generate_state_statistics(TRAJECTORIES)
        
        json(Dict(
            "is_training" => TRAINING_STATE["is_training"],
            "current_episode" => TRAINING_STATE["episode"],
            "elapsed_time" => elapsed,
            "current_loss" => TRAINING_HISTORY["losses"][idx],
            "current_reward" => TRAINING_HISTORY["rewards"][idx],
            "current_tb_loss" => TRAINING_HISTORY["tb_losses"][idx],
            "current_flow_loss" => TRAINING_HISTORY["flow_losses"][idx],
            "current_exploration" => TRAINING_HISTORY["state_coverages"][idx],  # For backward compatibility
            "state_coverage" => current_stats["coverage"],
            "mode_coverage" => current_stats["mode_coverage"],
            "sampling_entropy" => current_stats["sampling_entropy"],
            "training_objective" => PROBLEM_CONFIG[].training_objective
        ))
    else
        json(Dict(
            "is_training" => TRAINING_STATE["is_training"],
            "current_episode" => 0,
            "elapsed_time" => elapsed,
            "current_loss" => 0.0,
            "current_reward" => 0.0,
            "current_tb_loss" => 0.0,
            "current_flow_loss" => 0.0,
            "current_exploration" => 0.1,  # For backward compatibility
            "state_coverage" => 0.1,
            "mode_coverage" => 0.0,
            "sampling_entropy" => 0.0,
            "training_objective" => "TB"
        ))
    end
end

# Start server with CORS enabled
println("🚀 Starting GFlowNet Visualization API Server...")
println("📍 Server running at http://localhost:8080")
println("🎮 Domain: Grid World (10x10)")
println("🎯 Multiple reward peaks for exploration")
println("✨ Serving GFlowNet-specific data")

# Configure CORS
serveparams = Dict(
    "cors" => true,
    "cors_headers" => Dict(
        "Access-Control-Allow-Origin" => "*",
        "Access-Control-Allow-Methods" => "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers" => "Content-Type"
    )
)

serve(host="127.0.0.1", port=8080; cors=true)