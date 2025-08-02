# GFlowNet Visualization API Server
# Modern REST API and WebSocket server for web visualization

using Oxygen
using HTTP
using JSON3
using Dates

# Include GFlowNet modules
# using ....GFlowNet
# using ..GFlowNetUtils

using Statistics
using UUIDs

# Helper function to generate UUID
uuid4() = string(uuid1())
mean(x) = Statistics.mean(x)

# CORS middleware for web dashboard
middleware(handle) = function cors(handle)
    function(req::HTTP.Request)
        # Handle preflight requests
        if HTTP.method(req) == "OPTIONS"
            return HTTP.Response(200, [
                "Access-Control-Allow-Origin" => "*",
                "Access-Control-Allow-Methods" => "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers" => "Content-Type, Authorization",
                "Access-Control-Max-Age" => "86400"
            ])
        end
        
        # Add CORS headers to response
        response = handle(req)
        HTTP.setheader(response, "Access-Control-Allow-Origin" => "*")
        HTTP.setheader(response, "Access-Control-Allow-Methods" => "GET, POST, PUT, DELETE, OPTIONS")
        return response
    end
end

# Global state for demo (in production, use proper state management)
mutable struct APIState
    model::Union{GFlowNetModel, Nothing}
    trajectories::Vector{Any}
    training_history::Dict{String, Vector{Float64}}
    websocket_clients::Set{HTTP.WebSocket}
end

const api_state = APIState(
    nothing,
    [],
    Dict(
        "losses" => Float64[],
        "rewards" => Float64[],
        "episodes" => Float64[]
    ),
    Set{HTTP.WebSocket}()
)

# Health check endpoint
@get "/health" function()
    json(Dict(
        "status" => "healthy",
        "timestamp" => now(),
        "version" => "1.0.0"
    ))
end

# Get trajectories endpoint
@get "/api/trajectories" function()
    trajectories_data = map(api_state.trajectories) do traj
        Dict(
            "id" => traj[:id],
            "states" => traj[:states],
            "actions" => traj[:actions],
            "rewards" => traj[:rewards],
            "total_reward" => traj[:total_reward],
            "timestamp" => traj[:timestamp]
        )
    end
    
    json(Dict(
        "trajectories" => trajectories_data,
        "count" => length(trajectories_data)
    ))
end

# Get specific trajectory
@get "/api/trajectories/{id}" function(req::HTTP.Request, id::String)
    trajectory = findfirst(t -> t[:id] == id, api_state.trajectories)
    
    if isnothing(trajectory)
        return json(Dict("error" => "Trajectory not found"), status=404)
    end
    
    json(api_state.trajectories[trajectory])
end

# Sample new trajectory (mock for demo)
@post "/api/trajectories/sample" function(req::HTTP.Request)
    # In real implementation, this would use the actual model
    # For demo, generate mock trajectory
    trajectory_id = string(uuid4())
    
    # Generate smooth trajectory for visualization
    n_steps = 50
    t = range(0, 2π, length=n_steps)
    
    states = []
    actions = []
    rewards = []
    
    for (i, ti) in enumerate(t)
        # Create interesting 3D trajectory
        x = 5 + 3 * cos(ti) + 0.5 * sin(3ti)
        y = 5 + 3 * sin(ti) + 0.5 * cos(3ti)
        z = 2 + sin(2ti)
        
        push!(states, Dict(
            "position" => [x, y, z],
            "features" => [x, y, z, i/n_steps],
            "id" => "s_$i"
        ))
        
        if i < n_steps
            push!(actions, Dict(
                "type" => "move",
                "id" => "a_$i"
            ))
        end
        
        # Increasing reward
        push!(rewards, 10 * (1 - exp(-i/15)) + randn())
    end
    
    trajectory = Dict(
        "id" => trajectory_id,
        "states" => states,
        "actions" => actions,
        "rewards" => rewards,
        "total_reward" => sum(rewards),
        "timestamp" => now()
    )
    
    push!(api_state.trajectories, trajectory)
    
    # Notify WebSocket clients
    broadcast_to_clients(Dict(
        "type" => "trajectory.new",
        "data" => trajectory
    ))
    
    json(trajectory)
end

# Get training history
@get "/api/training/history" function()
    json(Dict(
        "losses" => api_state.training_history["losses"],
        "rewards" => api_state.training_history["rewards"],
        "episodes" => api_state.training_history["episodes"],
        "metrics" => Dict(
            "mean_loss" => isempty(api_state.training_history["losses"]) ? 0.0 : mean(api_state.training_history["losses"]),
            "mean_reward" => isempty(api_state.training_history["rewards"]) ? 0.0 : mean(api_state.training_history["rewards"]),
            "total_episodes" => length(api_state.training_history["episodes"])
        )
    ))
end

# Get current training metrics
@get "/api/training/metrics" function()
    # Get latest metrics
    latest_loss = isempty(api_state.training_history["losses"]) ? nothing : last(api_state.training_history["losses"])
    latest_reward = isempty(api_state.training_history["rewards"]) ? nothing : last(api_state.training_history["rewards"])
    
    json(Dict(
        "latest_loss" => latest_loss,
        "latest_reward" => latest_reward,
        "total_episodes" => length(api_state.training_history["episodes"]),
        "is_training" => false  # Would check actual training status
    ))
end

# Simulate training update (for demo)
@post "/api/training/simulate" function()
    # Simulate training step
    episode = length(api_state.training_history["episodes"]) + 1
    loss = 1.0 / (1 + episode * 0.1) + 0.1 * randn()
    reward = 10 * (1 - exp(-episode/50)) + randn()
    
    push!(api_state.training_history["episodes"], Float64(episode))
    push!(api_state.training_history["losses"], loss)
    push!(api_state.training_history["rewards"], reward)
    
    # Broadcast to WebSocket clients
    update = Dict(
        "type" => "training.update",
        "data" => Dict(
            "episode" => episode,
            "loss" => loss,
            "reward" => reward,
            "timestamp" => now()
        )
    )
    
    broadcast_to_clients(update)
    
    json(update["data"])
end

# Get flow field data (mock for demo)
@get "/api/analysis/flow-field" function()
    # Generate mock flow field data
    resolution = 20
    x_range = range(-5, 5, length=resolution)
    y_range = range(-5, 5, length=resolution)
    z_range = range(-2, 2, length=10)
    
    flow_data = []
    
    for (i, x) in enumerate(x_range)
        for (j, y) in enumerate(y_range)
            for (k, z) in enumerate(z_range)
                # Create interesting flow pattern
                r = sqrt(x^2 + y^2)
                theta = atan(y, x)
                
                vx = -y / (r + 1)
                vy = x / (r + 1)
                vz = 0.1 * sin(r)
                
                magnitude = sqrt(vx^2 + vy^2 + vz^2)
                
                push!(flow_data, Dict(
                    "position" => [x, y, z],
                    "velocity" => [vx, vy, vz],
                    "magnitude" => magnitude
                ))
            end
        end
    end
    
    json(Dict(
        "resolution" => [resolution, resolution, 10],
        "bounds" => Dict(
            "x" => [-5, 5],
            "y" => [-5, 5],
            "z" => [-2, 2]
        ),
        "data" => flow_data
    ))
end

# WebSocket endpoint for real-time updates
@websocket "/ws" function(ws::HTTP.WebSocket)
    # Add client to set
    push!(api_state.websocket_clients, ws)
    
    # Send welcome message
    send(ws, JSON3.write(Dict(
        "type" => "connection.established",
        "timestamp" => now()
    )))
    
    try
        # Keep connection alive and handle messages
        for msg in ws
            data = JSON3.read(String(msg))
            
            if data["type"] == "subscribe"
                # Handle subscription requests
                send(ws, JSON3.write(Dict(
                    "type" => "subscription.confirmed",
                    "channels" => data["channels"]
                )))
            elseif data["type"] == "ping"
                # Respond to ping
                send(ws, JSON3.write(Dict(
                    "type" => "pong",
                    "timestamp" => now()
                )))
            end
        end
    catch e
        @error "WebSocket error" exception=e
    finally
        # Remove client on disconnect
        delete!(api_state.websocket_clients, ws)
    end
end

# Helper function to broadcast to all WebSocket clients
function broadcast_to_clients(message::Dict)
    message_json = JSON3.write(message)
    
    for client in api_state.websocket_clients
        try
            send(client, message_json)
        catch e
            # Client disconnected, remove from set
            delete!(api_state.websocket_clients, client)
        end
    end
end

# Start server
function start_server(; host="127.0.0.1", port=8080)
    println("🚀 GFlowNet Visualization API Server")
    println("=" ^ 50)
    println("Starting server at http://$host:$port")
    println("\nEndpoints:")
    println("  GET  /health                - Health check")
    println("  GET  /api/trajectories       - List trajectories")
    println("  POST /api/trajectories/sample - Sample new trajectory")
    println("  GET  /api/training/history   - Training history")
    println("  GET  /api/analysis/flow-field - Flow field data")
    println("  WS   /ws                     - WebSocket connection")
    println("\n✨ Server ready for beautiful visualizations!")
    
    serve(host=host, port=port)
end

# Auto-start if run directly
if abspath(PROGRAM_FILE) == @__FILE__
    start_server()
end