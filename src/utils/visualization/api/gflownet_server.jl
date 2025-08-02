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
    # Multiple reward peaks
    peaks = [(8, 8, 10.0), (2, 8, 8.0), (5, 5, 6.0), (8, 2, 7.0)]
    
    reward = 0.0
    for (px, py, intensity) in peaks
        dist = sqrt((state.x - px)^2 + (state.y - py)^2)
        reward += intensity * exp(-dist / 2.0)
    end
    
    return reward
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
                if 1 <= new_x <= 10 && 1 <= new_y <= 10
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
            if best_action == "up" && y < 10
                y += 1
            elseif best_action == "down" && y > 1
                y -= 1
            elseif best_action == "left" && x > 1
                x -= 1
            elseif best_action == "right" && x < 10
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
    exploration_rates = Float64[]
    
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
        
        # Exploration rate decreases
        push!(exploration_rates, 0.9 * exp(-i/100) + 0.1)
    end
    
    return Dict(
        "episodes" => episodes,
        "losses" => losses,
        "rewards" => rewards,
        "tb_losses" => tb_losses,
        "flow_losses" => flow_losses,
        "exploration_rates" => exploration_rates,
        "metrics" => Dict(
            "mean_loss" => mean(losses[end-10:end]),
            "mean_reward" => mean(rewards[end-10:end]),
            "mean_tb_loss" => mean(tb_losses[end-10:end]),
            "mean_flow_loss" => mean(flow_losses[end-10:end]),
            "current_exploration" => exploration_rates[end],
            "total_episodes" => n_episodes,
            "convergence_estimate" => 1 - losses[end] / losses[1]
        )
    )
end

# Generate flow field for the grid world
function generate_flow_field()
    resolution = 10
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
        "reward_peaks" => [
            Dict("position" => [8, 8], "intensity" => 10.0, "name" => "Primary Goal"),
            Dict("position" => [2, 8], "intensity" => 8.0, "name" => "Secondary Goal"),
            Dict("position" => [5, 5], "intensity" => 6.0, "name" => "Tertiary Goal"),
            Dict("position" => [8, 2], "intensity" => 7.0, "name" => "Alternative Goal")
        ]
    )
end

# Generate state visitation statistics
function generate_state_statistics(trajectories)
    visitation_counts = Dict{Tuple{Int,Int}, Int}()
    value_estimates = Dict{Tuple{Int,Int}, Float64}()
    
    for traj in trajectories
        for (i, state) in enumerate(traj["states"])
            pos = tuple(state["grid_position"]...)
            visitation_counts[pos] = get(visitation_counts, pos, 0) + 1
            
            # Value estimate based on future rewards
            future_reward = sum(traj["rewards"][i:end])
            current_value = get(value_estimates, pos, 0.0)
            value_estimates[pos] = 0.9 * current_value + 0.1 * future_reward
        end
    end
    
    return Dict(
        "visitation_counts" => visitation_counts,
        "value_estimates" => value_estimates,
        "total_states_visited" => length(visitation_counts),
        "max_visits" => maximum(values(visitation_counts)),
        "coverage" => length(visitation_counts) / 100.0  # 10x10 grid
    )
end

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
    "exploration_rates" => Float64[]
)

# Initialize with a few trajectories
for i in 1:3
    push!(TRAJECTORIES, generate_gflownet_trajectory(i))
end

const FLOW_FIELD = generate_flow_field()
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
    
    if is_3d
        # Return full format for 3D visualization
        json(Dict(
            "trajectories" => TRAJECTORIES,
            "grid_size" => [10, 10],
            "reward_peaks" => [
                Dict("position" => [8, 8], "intensity" => 10.0, "name" => "Primary Goal"),
                Dict("position" => [2, 8], "intensity" => 8.0, "name" => "Secondary Goal"),
                Dict("position" => [5, 5], "intensity" => 6.0, "name" => "Center Reward"),
                Dict("position" => [8, 2], "intensity" => 7.0, "name" => "Corner Reward")
            ]
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
            "grid_size" => [10, 10],
            "reward_peaks" => [
                Dict("position" => [8, 8], "intensity" => 10.0, "name" => "Primary Goal"),
                Dict("position" => [2, 8], "intensity" => 8.0, "name" => "Secondary Goal"),
                Dict("position" => [5, 5], "intensity" => 6.0, "name" => "Center Reward"),
                Dict("position" => [8, 2], "intensity" => 7.0, "name" => "Corner Reward")
            ],
            "count" => length(TRAJECTORIES),
            "domain" => "Grid World",
            "state_space" => "10x10 grid",
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
            "current_exploration" => length(TRAINING_HISTORY["exploration_rates"]) > 0 ? TRAINING_HISTORY["exploration_rates"][end] : 0.5,
            "total_episodes" => TRAINING_STATE["episode"],
            "convergence_estimate" => min(0.95, TRAINING_STATE["episode"] / 1000.0)
        )
    else
        metrics = Dict(
            "mean_loss" => 0.0,
            "mean_reward" => 0.0,
            "mean_tb_loss" => 0.0,
            "mean_flow_loss" => 0.0,
            "current_exploration" => 0.5,
            "total_episodes" => 0,
            "convergence_estimate" => 0.0
        )
    end
    
    json(Dict(
        "episodes" => TRAINING_HISTORY["episodes"],
        "losses" => TRAINING_HISTORY["losses"],
        "rewards" => TRAINING_HISTORY["rewards"],
        "tb_losses" => TRAINING_HISTORY["tb_losses"],
        "flow_losses" => TRAINING_HISTORY["flow_losses"],
        "exploration_rates" => TRAINING_HISTORY["exploration_rates"],
        "metrics" => metrics
    ))
end

@get "/api/training/metrics" function()
    if length(TRAINING_HISTORY["episodes"]) > 0
        latest_idx = length(TRAINING_HISTORY["episodes"])
        json(Dict(
            "latest_loss" => TRAINING_HISTORY["losses"][latest_idx],
            "latest_reward" => TRAINING_HISTORY["rewards"][latest_idx],
            "latest_tb_loss" => TRAINING_HISTORY["tb_losses"][latest_idx],
            "latest_flow_loss" => TRAINING_HISTORY["flow_losses"][latest_idx],
            "exploration_rate" => TRAINING_HISTORY["exploration_rates"][latest_idx],
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
            "exploration_rate" => 0.5,
            "total_episodes" => 0,
            "convergence" => 0.0,
            "is_training" => false
        ))
    end
end

@get "/api/analysis/flow-field" function()
    json(FLOW_FIELD)
end

@get "/api/analysis/state-statistics" function()
    json(STATE_STATS)
end

@get "/api/domain/info" function()
    json(Dict(
        "name" => "Grid World",
        "description" => "10x10 grid with multiple reward peaks. Agent learns to navigate from start to high-reward regions.",
        "state_space" => Dict(
            "type" => "discrete",
            "size" => [10, 10],
            "total_states" => 100
        ),
        "action_space" => Dict(
            "type" => "discrete", 
            "actions" => ["up", "down", "left", "right"],
            "size" => 4
        ),
        "reward_info" => Dict(
            "type" => "continuous",
            "range" => [0, 10],
            "peaks" => FLOW_FIELD["reward_peaks"]
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

# WebSocket endpoint stub
@get "/ws" function()
    return HTTP.Response(501, "WebSocket not implemented in demo server")
end

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
        "reward_peaks" => [
            Dict("position" => [8, 8], "intensity" => 10.0, "name" => "Primary Goal"),
            Dict("position" => [2, 8], "intensity" => 8.0, "name" => "Secondary Goal"),
            Dict("position" => [5, 5], "intensity" => 6.0, "name" => "Center Reward"),
            Dict("position" => [8, 2], "intensity" => 7.0, "name" => "Corner Reward")
        ]
    ))
end

# Distribution analysis
@get "/api/analysis/distribution" function()
    # Calculate distribution statistics
    endpoints = Dict()
    for traj in TRAJECTORIES
        last_state = traj["states"][end]
        key = "$(last_state["grid_position"][1]),$(last_state["grid_position"][2])"
        endpoints[key] = get(endpoints, key, 0) + 1
    end
    
    json(Dict(
        "total_trajectories" => length(TRAJECTORIES),
        "unique_endpoints" => length(keys(endpoints)),
        "diversity_score" => 1.0 - maximum(values(endpoints)) / length(TRAJECTORIES)
    ))
end

# Start training endpoint
@post "/api/training/start" function(req::HTTP.Request)
    config = JSON3.read(req.body)
    
    # Update training state
    TRAINING_STATE["is_training"] = true
    TRAINING_STATE["episode"] = 0
    TRAINING_STATE["config"] = config
    TRAINING_STATE["start_time"] = now()
    TRAINING_STATE["session_id"] = string(UUIDs.uuid4())
    
    # Clear previous data
    empty!(TRAINING_HISTORY["episodes"])
    empty!(TRAINING_HISTORY["losses"])
    empty!(TRAINING_HISTORY["rewards"])
    empty!(TRAINING_HISTORY["tb_losses"])
    empty!(TRAINING_HISTORY["flow_losses"])
    empty!(TRAINING_HISTORY["exploration_rates"])
    empty!(TRAJECTORIES)
    
    # Start background training simulation
    @async begin
        n_episodes = get(config, "n_episodes", 1000)
        exploration_rate = get(config, "exploration_rate", 0.3)
        
        for episode in 1:n_episodes
            if !TRAINING_STATE["is_training"]
                break
            end
            
            # Update episode count
            TRAINING_STATE["episode"] = episode
            
            # Generate new trajectory
            new_traj = generate_gflownet_trajectory(length(TRAJECTORIES) + 1)
            push!(TRAJECTORIES, new_traj)
            
            # Simulate training metrics
            base_loss = 5.0 / (1 + episode * 0.05)
            push!(TRAINING_HISTORY["episodes"], Float64(episode))
            push!(TRAINING_HISTORY["losses"], base_loss + 0.2 * randn())
            push!(TRAINING_HISTORY["tb_losses"], base_loss * 0.8 + 0.15 * randn())
            push!(TRAINING_HISTORY["flow_losses"], base_loss * 1.2 + 0.25 * randn())
            
            # Rewards increase over time
            base_reward = 15 * (1 - exp(-episode/50))
            push!(TRAINING_HISTORY["rewards"], base_reward + 2 * randn())
            
            # Exploration rate decreases
            current_exploration = exploration_rate * exp(-episode/300) + 0.1
            push!(TRAINING_HISTORY["exploration_rates"], current_exploration)
            
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
        "config" => config,
        "session_id" => TRAINING_STATE["session_id"]
    ))
end

# Stop training endpoint
@post "/api/training/stop" function()
    TRAINING_STATE["is_training"] = false
    json(Dict("status" => "stopped"))
end

# Start server
println("🚀 Starting GFlowNet Visualization API Server...")
println("📍 Server running at http://localhost:8080")
println("🎮 Domain: Grid World (10x10)")
println("🎯 Multiple reward peaks for exploration")
println("✨ Serving GFlowNet-specific data")

serve(host="127.0.0.1", port=8080)