# Simple GFlowNet Visualization API Server
# Provides mock data for visualization

using Oxygen
using HTTP
using JSON3
using Dates
using Statistics
using UUIDs
using Random

# Define reward peaks for grid world
const REWARD_PEAKS = [
    Dict("position" => [3, 7], "intensity" => 10.0, "name" => "Peak A"),
    Dict("position" => [7, 3], "intensity" => 8.0, "name" => "Peak B"),
    Dict("position" => [8, 8], "intensity" => 12.0, "name" => "Peak C"),
    Dict("position" => [2, 2], "intensity" => 6.0, "name" => "Peak D")
]

# Calculate reward based on distance to peaks
function calculate_reward(x, y)
    reward = 0.0
    for peak in REWARD_PEAKS
        dist = sqrt((x - peak["position"][1])^2 + (y - peak["position"][2])^2)
        reward += peak["intensity"] * exp(-dist / 2.5)
    end
    return reward
end

# Generate GFlowNet trajectory moving toward high rewards
function generate_demo_trajectory(; start_pos=nothing, trajectory_type="exploitation")
    trajectory_id = string(uuid1())
    
    # Starting position
    if isnothing(start_pos)
        x = 1.0 + 9.0 * rand()
        y = 1.0 + 9.0 * rand()
    else
        x, y = start_pos
    end
    
    states = []
    rewards = []
    n_steps = 20 + rand(10:30)
    
    for i in 1:n_steps
        # Record state as 2D position on grid
        push!(states, [x, y])
        push!(rewards, calculate_reward(x, y))
        
        # Move based on trajectory type
        if trajectory_type == "exploitation"
            # Move toward nearest high-reward peak
            best_dx, best_dy = 0.0, 0.0
            best_future_reward = calculate_reward(x, y)
            
            for dx in [-0.5, 0, 0.5], dy in [-0.5, 0, 0.5]
                new_x = clamp(x + dx, 1, 10)
                new_y = clamp(y + dy, 1, 10)
                future_reward = calculate_reward(new_x, new_y)
                
                if future_reward > best_future_reward
                    best_future_reward = future_reward
                    best_dx, best_dy = dx, dy
                end
            end
            
            # Add some noise
            x = clamp(x + best_dx + 0.2 * randn(), 1, 10)
            y = clamp(y + best_dy + 0.2 * randn(), 1, 10)
            
        else  # exploration
            # Random walk with momentum
            x = clamp(x + 0.5 * randn(), 1, 10)
            y = clamp(y + 0.5 * randn(), 1, 10)
        end
    end
    
    return Dict(
        "id" => trajectory_id,
        "states" => states,
        "rewards" => rewards,
        "total_reward" => sum(rewards),
        "trajectory_type" => trajectory_type
    )
end

# Generate initial trajectories
function generate_initial_trajectories()
    trajectories = []
    for i in 1:5
        push!(trajectories, generate_demo_trajectory())
    end
    return trajectories
end

const DEMO_TRAJECTORIES = generate_initial_trajectories()

# Generate training history with GFlowNet-specific metrics
function generate_training_history()
    n_episodes = 200
    episodes = Float64.(1:n_episodes)
    
    # Generate realistic GFlowNet training curves
    losses = Float64[]
    tb_losses = Float64[]
    flow_losses = Float64[]
    rewards = Float64[]
    exploration_rates = Float64[]
    
    for i in 1:n_episodes
        # Trajectory balance loss with convergence
        tb_loss = 5.0 * exp(-i/40) + 0.5 + 0.3 * randn()
        push!(tb_losses, max(0.1, tb_loss))
        
        # Flow matching loss
        flow_loss = 3.0 * exp(-i/50) + 0.3 + 0.2 * randn()
        push!(flow_losses, max(0.1, flow_loss))
        
        # Combined loss
        loss = 0.7 * tb_losses[end] + 0.3 * flow_losses[end]
        push!(losses, loss)
        
        # Increasing rewards with exploration
        reward = 20 * (1 - exp(-i/30)) + 2 * randn() + 3 * sin(i/10)
        push!(rewards, max(0, reward))
        
        # Decaying exploration rate
        exploration = 0.95 * exp(-i/60) + 0.05
        push!(exploration_rates, exploration)
    end
    
    return Dict(
        "episodes" => episodes,
        "losses" => losses,
        "rewards" => rewards,
        "tb_losses" => tb_losses,
        "flow_losses" => flow_losses,
        "exploration_rates" => exploration_rates,
        "metrics" => Dict(
            "mean_loss" => mean(losses[end-20:end]),
            "mean_reward" => mean(rewards[end-20:end]),
            "mean_tb_loss" => mean(tb_losses[end-20:end]),
            "mean_flow_loss" => mean(flow_losses[end-20:end]),
            "current_exploration" => exploration_rates[end],
            "total_episodes" => n_episodes,
            "convergence_estimate" => 1 - losses[end] / losses[1]
        )
    )
end

const TRAINING_HISTORY = generate_training_history()

# Routes
@get "/health" function()
    json(Dict(
        "status" => "healthy",
        "timestamp" => now(),
        "version" => "1.0.0"
    ))
end

@get "/api/trajectories" function()
    json(Dict(
        "trajectories" => DEMO_TRAJECTORIES,
        "count" => length(DEMO_TRAJECTORIES)
    ))
end

@get "/api/trajectories/{id}" function(req::HTTP.Request, id::String)
    trajectory = findfirst(t -> t["id"] == id, DEMO_TRAJECTORIES)
    
    if isnothing(trajectory)
        return json(Dict("error" => "Trajectory not found"), status=404)
    end
    
    json(DEMO_TRAJECTORIES[trajectory])
end

@post "/api/trajectories/sample" function()
    new_trajectory = generate_demo_trajectory()
    push!(DEMO_TRAJECTORIES, new_trajectory)
    json(new_trajectory)
end

@get "/api/training/history" function()
    # Return dynamic training history based on elapsed time
    elapsed = Dates.value(now() - GLOBAL_STATE.start_time) / 1000  # seconds
    progress_episodes = min(200, round(Int, elapsed * 2))  # 2 episodes per second
    
    if progress_episodes > 0
        idx = min(progress_episodes, length(GLOBAL_STATE.training_data["episodes"]))
        
        return json(Dict(
            "episodes" => GLOBAL_STATE.training_data["episodes"][1:idx],
            "losses" => GLOBAL_STATE.training_data["losses"][1:idx],
            "rewards" => GLOBAL_STATE.training_data["rewards"][1:idx],
            "tb_losses" => GLOBAL_STATE.training_data["tb_losses"][1:idx],
            "flow_losses" => GLOBAL_STATE.training_data["flow_losses"][1:idx],
            "exploration_rates" => GLOBAL_STATE.training_data["exploration_rates"][1:idx],
            "metrics" => Dict(
                "mean_loss" => mean(GLOBAL_STATE.training_data["losses"][max(1, idx-20):idx]),
                "mean_reward" => mean(GLOBAL_STATE.training_data["rewards"][max(1, idx-20):idx]),
                "mean_tb_loss" => mean(GLOBAL_STATE.training_data["tb_losses"][max(1, idx-20):idx]),
                "mean_flow_loss" => mean(GLOBAL_STATE.training_data["flow_losses"][max(1, idx-20):idx]),
                "current_exploration" => GLOBAL_STATE.training_data["exploration_rates"][idx],
                "total_episodes" => idx,
                "convergence_estimate" => 1 - GLOBAL_STATE.training_data["losses"][idx] / GLOBAL_STATE.training_data["losses"][1]
            )
        ))
    else
        return json(GLOBAL_STATE.training_data)
    end
end

@get "/api/training/metrics" function()
    elapsed = Dates.value(now() - GLOBAL_STATE.start_time) / 1000
    progress_episodes = min(200, round(Int, elapsed * 2))
    idx = max(1, min(progress_episodes, length(GLOBAL_STATE.training_data["episodes"])))
    
    json(Dict(
        "latest_loss" => idx > 0 ? GLOBAL_STATE.training_data["losses"][idx] : 0,
        "latest_reward" => idx > 0 ? GLOBAL_STATE.training_data["rewards"][idx] : 0,
        "total_episodes" => progress_episodes,
        "is_training" => GLOBAL_STATE.is_training && progress_episodes < 200
    ))
end

@get "/api/analysis/flow-field" function()
    # Generate GFlowNet policy flow field
    resolution = 15
    x_range = range(1, 10, length=resolution)
    y_range = range(1, 10, length=resolution)
    
    flow_data = []
    
    for x in x_range
        for y in y_range
            # Calculate flow toward reward peaks
            vx = 0.0
            vy = 0.0
            total_weight = 0.0
            
            for peak in REWARD_PEAKS
                dx = peak["position"][1] - x
                dy = peak["position"][2] - y
                dist = sqrt(dx^2 + dy^2)
                
                # Weight by reward and distance
                weight = peak["intensity"] * exp(-dist / 3)
                
                if dist > 0.5
                    vx += weight * dx / dist
                    vy += weight * dy / dist
                    total_weight += weight
                end
            end
            
            # Normalize and add exploration
            if total_weight > 0
                vx /= total_weight
                vy /= total_weight
                
                # Add learned exploration
                angle_noise = 0.3 * sin(x + y)
                vx += 0.2 * cos(atan(vy, vx) + angle_noise)
                vy += 0.2 * sin(atan(vy, vx) + angle_noise)
            end
            
            magnitude = sqrt(vx^2 + vy^2)
            reward = calculate_reward(x, y)
            flow_value = reward + 2.0 * exp(-minimum([sqrt((x-p["position"][1])^2 + (y-p["position"][2])^2) for p in REWARD_PEAKS]) / 2)
            
            push!(flow_data, Dict(
                "position" => [x, y, 0],
                "velocity" => [vx, vy, 0],
                "magnitude" => magnitude,
                "reward" => reward,
                "flow_value" => flow_value
            ))
        end
    end
    
    json(Dict(
        "resolution" => [resolution, resolution, 1],
        "bounds" => Dict(
            "x" => [1, 10],
            "y" => [1, 10]
        ),
        "data" => flow_data,
        "reward_peaks" => REWARD_PEAKS
    ))
end

@get "/api/trajectories/all" function()
    # Generate diverse trajectories
    all_trajectories = []
    
    # Exploitation trajectories (following high rewards)
    for i in 1:20
        traj = generate_demo_trajectory(trajectory_type="exploitation")
        push!(all_trajectories, traj)
    end
    
    # Exploration trajectories
    for i in 1:15
        traj = generate_demo_trajectory(trajectory_type="exploration")
        push!(all_trajectories, traj)
    end
    
    json(Dict(
        "trajectories" => all_trajectories,
        "reward_peaks" => REWARD_PEAKS
    ))
end

@get "/api/analysis/state-statistics" function()
    # Generate state visitation statistics
    visitation_counts = Dict{String, Int}()
    value_estimates = Dict{String, Float64}()
    
    # Simulate GFlowNet visitation patterns
    for i in 1:10
        for j in 1:10
            key = "$i,$j"
            x, y = Float64(i), Float64(j)
            
            # Visitation based on flow
            reward = calculate_reward(x, y)
            min_dist = minimum([sqrt((x-p["position"][1])^2 + (y-p["position"][2])^2) for p in REWARD_PEAKS])
            
            base_visits = 50 * exp(-min_dist / 3)
            exploration_visits = 20 * exp(-((x-5)^2 + (y-5)^2) / 25)
            visits = round(Int, base_visits + exploration_visits + rand(0:10))
            visitation_counts[key] = visits
            
            # Value estimates V(s)
            future_reward = reward * (1 + 2 * exp(-min_dist / 2))
            value_estimates[key] = future_reward + 0.5 * randn()
        end
    end
    
    total_visits = sum(values(visitation_counts))
    max_visits = maximum(values(visitation_counts))
    visited_states = count(v -> v > 5, values(visitation_counts))
    
    # Policy entropy
    visit_probs = [v / total_visits for v in values(visitation_counts) if v > 0]
    entropy = -sum(p * log(p + 1e-10) for p in visit_probs)
    
    json(Dict(
        "visitation_counts" => visitation_counts,
        "value_estimates" => value_estimates,
        "total_states_visited" => visited_states,
        "max_visits" => max_visits,
        "coverage" => visited_states / 100,
        "flow_statistics" => Dict(
            "mean_flow" => mean(values(value_estimates)),
            "max_flow" => maximum(values(value_estimates)),
            "convergence_ratio" => 0.75 + 0.2 * rand(),
            "policy_entropy" => entropy
        )
    ))
end

@get "/api/analysis/distribution" function()
    # Distribution analysis stats
    json(Dict(
        "total_trajectories" => 35,
        "unique_endpoints" => 18,
        "diversity_score" => 0.73
    ))
end

# Global state for dynamic training data
mutable struct TrainingState
    episode::Int
    is_training::Bool
    start_time::DateTime
    training_data::Dict
end

const GLOBAL_STATE = TrainingState(0, false, now(), Dict())

# Reset training endpoint
@post "/api/training/reset" function()
    GLOBAL_STATE.episode = 0
    GLOBAL_STATE.is_training = true
    GLOBAL_STATE.start_time = now()
    GLOBAL_STATE.training_data = generate_training_history()
    
    json(Dict(
        "status" => "reset",
        "message" => "Training reset successfully",
        "timestamp" => now()
    ))
end

# Get current training state
@get "/api/training/state" function()
    elapsed = Dates.value(now() - GLOBAL_STATE.start_time) / 1000  # seconds
    progress_episodes = min(GLOBAL_STATE.training_data["episodes"][end], 
                           round(Int, elapsed * 2))  # 2 episodes per second
    
    # Find the index for current progress
    idx = min(progress_episodes, length(GLOBAL_STATE.training_data["episodes"]))
    idx = max(1, idx)
    
    json(Dict(
        "is_training" => GLOBAL_STATE.is_training && progress_episodes < 200,
        "current_episode" => progress_episodes,
        "elapsed_time" => elapsed,
        "current_loss" => idx > 0 ? GLOBAL_STATE.training_data["losses"][idx] : 0,
        "current_reward" => idx > 0 ? GLOBAL_STATE.training_data["rewards"][idx] : 0,
        "current_tb_loss" => idx > 0 ? GLOBAL_STATE.training_data["tb_losses"][idx] : 0,
        "current_flow_loss" => idx > 0 ? GLOBAL_STATE.training_data["flow_losses"][idx] : 0,
        "current_exploration" => idx > 0 ? GLOBAL_STATE.training_data["exploration_rates"][idx] : 1.0
    ))
end

# Initialize global state with reset
GLOBAL_STATE.training_data = generate_training_history()
GLOBAL_STATE.start_time = now()
GLOBAL_STATE.is_training = true

# Start server without middleware
println("🚀 Starting Simple GFlowNet API Server...")
println("📍 Server running at http://localhost:8080")
println("✨ Serving demo data for visualization")
println("🔄 Dynamic training simulation enabled")
println("📊 Training started automatically")

# Configure CORS
serveparams = Dict(
    "cors" => true,
    "cors_headers" => Dict(
        "Access-Control-Allow-Origin" => "*",
        "Access-Control-Allow-Methods" => "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers" => "Content-Type"
    )
)

serve(host="127.0.0.1", port=8080)