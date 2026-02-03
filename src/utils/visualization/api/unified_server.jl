# Unified server for real GFlowNet training visualization
# Provides v2 API endpoints with real training integration

using Oxygen
using JSON3
using HTTP
using UUIDs
using Dates
using Statistics
using GFlowNet

# Include core visualization modules
include("../core/adapters.jl")
include("../core/metrics.jl")
include("../core/training_session.jl")
include("../domains/grid_world.jl")

# Global session storage (single session for simplicity)
const CURRENT_SESSION = Ref{Union{TrainingSession, Nothing}}(nothing)
const TRAINING_TASK   = Ref{Union{Task, Nothing}}(nothing)

# ============================================
# Domain & Model Creation
# ============================================

"""
    create_model_and_adapter(domain_type::String, config::Dict)

Create model and adapter based on domain type.
Currently supports "grid_world" only.

# Arguments
- `domain_type::String`: Domain type (e.g., "grid_world")
- `config::Dict`: Configuration dict with domain-specific parameters

# Returns
- `(model, adapter)`: GFlowNet model and corresponding domain adapter
"""
function create_model_and_adapter(domain_type::String, config::Dict)
    if domain_type == "grid_world"
        return create_grid_world_model_and_adapter(config)
    else
        error("Unsupported domain type: $domain_type. Currently only 'grid_world' is supported.")
    end
end

"""
    create_grid_world_model_and_adapter(config::Dict)

Create a Grid World GFlowNet model and its visualization adapter.

# Arguments
- `config::Dict`: Configuration with grid_size, reward_peaks, objective, etc.

# Returns
- `(model, adapter)`: Grid World model and GridWorldAdapter
"""
function create_grid_world_model_and_adapter(config::Dict)
    grid_size = get(config, "grid_size", 8)

    # Parse reward peaks
    peaks_config = get(config, "reward_peaks", [])
    reward_positions = Dict{Tuple{Int,Int}, Float64}()
    for peak in peaks_config
        pos = peak["position"]
        intensity = get(peak, "intensity", 10.0)
        reward_positions[(Int(pos[1]), Int(pos[2]))] = Float64(intensity)
    end

    # Default peaks if none specified
    if isempty(reward_positions)
        reward_positions[(grid_size, grid_size)] = 10.0
        reward_positions[(1, grid_size)]         = 8.0
        reward_positions[(grid_size, 1)]         = 8.0
    end

    # Determine if backward policy / flow estimator is needed
    objective_str = get(config, "objective", "TRAJECTORY_BALANCE")
    needs_backward = objective_str == "DETAILED_BALANCE"
    needs_flow_est = objective_str == "FLOW_MATCHING" || objective_str == "DIRECT_FLOW_OBJECTIVE"

    # Create the real GFlowNet model using high-level API
    model = GFlowNet.create_grid_world_gflownet(
        grid_size               = grid_size,
        reward_positions        = reward_positions,
        hidden_dim              = get(config, "hidden_dim", 64),
        learning_rate           = get(config, "learning_rate", 0.001),
        include_backward        = needs_backward,
        allow_all_moves         = get(config, "allow_all_moves", false),
    )

    adapter = GridWorldAdapter(grid_size, reward_positions)
    return model, adapter
end

# ============================================
# CORS Middleware
# ============================================

"""
    add_cors_headers(handler)

CORS middleware for cross-origin requests from the frontend.
"""
function add_cors_headers(handler)
    return function(req)
        headers = [
            "Access-Control-Allow-Origin"  => "*",
            "Access-Control-Allow-Methods" => "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers" => "Content-Type"
        ]
        if req.method == "OPTIONS"
            return HTTP.Response(200, headers)
        end
        response = handler(req)
        for (k, v) in headers
            HTTP.setheader(response, k => v)
        end
        return response
    end
end

# ============================================
# API v2 Endpoints — Domain Info
# ============================================

@get "/api/v2/domain/info" function(req)
    session = CURRENT_SESSION[]
    if session === nothing
        return json(Dict(
            "has_session"       => false,
            "available_domains" => ["grid_world"],
            "message"           => "No active session. Start training to initialize."
        ))
    end
    return json(Dict(
        "has_session" => true,
        "domain"      => get_domain_config(session.adapter),
        "renderer"    => get_renderer_name(session.adapter)
    ))
end

# ============================================
# API v2 Endpoints — Training Control
# ============================================

@post "/api/v2/training/start" function(req)
    config = JSON3.read(String(req.body), Dict)

    # Stop existing training if any
    if CURRENT_SESSION[] !== nothing && CURRENT_SESSION[].is_training
        CURRENT_SESSION[].is_training = false
        sleep(0.1)  # Let the training loop exit
    end

    session = create_session(config)
    CURRENT_SESSION[] = session
    session.is_training = true
    session.start_time  = now()

    # Launch async training loop
    # NOTE: @async runs cooperatively on the same thread as Oxygen,
    # so session field reads/writes are safe without locks.
    TRAINING_TASK[] = @async begin
        try
            while session.is_training && session.current_iteration < session.total_iterations
                if !session.is_paused
                    step!(session)
                end
                sleep(0.05)  # ~20 iterations/second cap; yields to server
            end
        catch e
            @error "Training loop fatal error" exception=e
            session.is_training = false
            session.last_error  = sprint(showerror, e)
        end
    end

    return json(Dict(
        "status"           => "started",
        "session_id"       => session.id,
        "domain"           => get_domain_config(session.adapter),
        "renderer"         => get_renderer_name(session.adapter),
        "total_iterations" => session.total_iterations
    ))
end

@post "/api/v2/training/stop" function(req)
    session = CURRENT_SESSION[]
    if session !== nothing
        session.is_training = false
    end
    return json(Dict("status" => "stopped"))
end

@post "/api/v2/training/pause" function(req)
    session = CURRENT_SESSION[]
    if session !== nothing
        session.is_paused = !session.is_paused
    end
    return json(Dict("paused" => session !== nothing && session.is_paused))
end

# ============================================
# API v2 Endpoints — Training State
# ============================================

@get "/api/v2/training/state" function(req)
    session = CURRENT_SESSION[]
    if session === nothing
        return json(Dict("has_session" => false))
    end

    recent = session.trajectory_buffer
    universal_metrics = isempty(recent) ? Dict() : compute_gflownet_metrics(session.model, recent)
    domain_metrics    = isempty(recent) ? Dict() : compute_domain_metrics(session.adapter, session.model, recent)

    return json(Dict(
        "has_session"          => true,
        "is_training"          => session.is_training,
        "is_paused"            => session.is_paused,
        "is_real_training"     => true,   # distinguishes from mock server
        "current_iteration"    => session.current_iteration,
        "total_iterations"     => session.total_iterations,
        "progress"             => session.current_iteration / max(session.total_iterations, 1),
        # Latest metrics
        "latest_loss"          => isempty(session.losses) ? nothing : session.losses[end],
        "latest_reward"        => isempty(session.rewards) ? nothing : session.rewards[end],
        "latest_gradient_norm" => isempty(session.gradient_norms) ? nothing : session.gradient_norms[end],
        # Computed metrics
        "metrics"              => universal_metrics,
        "domain_metrics"       => domain_metrics,
        # Error info
        "last_error"           => session.last_error,
        "error_count"          => session.error_count
    ))
end

@get "/api/v2/training/history" function(req)
    session = CURRENT_SESSION[]
    if session === nothing
        return json(Dict("has_session" => false))
    end
    return json(Dict(
        "losses"          => session.losses,
        "rewards"         => session.rewards,
        "gradient_norms"  => session.gradient_norms,
        "iteration_times" => session.iteration_times
    ))
end

# ============================================
# API v2 Endpoints — Trajectories
# ============================================

@get "/api/v2/trajectories" function(req)
    session = CURRENT_SESSION[]
    if session === nothing
        return json(Dict("trajectories" => [], "domain" => nothing))
    end
    viz_trajectories = [
        trajectory_to_viz_data(session.adapter, traj, "traj_$i")
        for (i, traj) in enumerate(session.trajectory_buffer)
    ]
    return json(Dict(
        "trajectories" => viz_trajectories,
        "domain"       => get_domain_config(session.adapter),
        "count"        => length(viz_trajectories)
    ))
end

# ============================================
# API v2 Endpoints — Analysis
# ============================================

@get "/api/v2/analysis/flow" function(req)
    session = CURRENT_SESSION[]
    if session === nothing
        return json(Dict("supported" => false, "error" => "No session"))
    end
    return json(compute_flow_field(session.adapter, session.model))
end

@get "/api/v2/analysis/distribution" function(req)
    session = CURRENT_SESSION[]
    if session === nothing
        return json(Dict("supported" => false, "error" => "No session"))
    end
    return json(compute_distribution_data(session.adapter, session.model, session.trajectory_buffer))
end

# ============================================
# Server Launch
# ============================================

"""
    start_real_training_server(; port::Int = 8080)

Start the unified GFlowNet training visualization server.

# Arguments
- `port::Int`: Port to run the server on (default: 8080)

# Example
```julia
include("src/utils/visualization/api/unified_server.jl")
start_real_training_server(port=8080)
```
"""
function start_real_training_server(; port::Int = 8080)
    @info "Starting real GFlowNet training visualization server on port $port"
    @info "Frontend: http://localhost:3000 (Vite dev server)"
    @info "API base: http://localhost:$port/api/v2/"
    @info ""
    @info "Available endpoints:"
    @info "  POST /api/v2/training/start    - Start training session"
    @info "  GET  /api/v2/training/state    - Get current training state"
    @info "  GET  /api/v2/training/history  - Get training history"
    @info "  GET  /api/v2/trajectories      - Get recent trajectories"
    @info "  GET  /api/v2/analysis/flow     - Get flow field data"
    @info "  GET  /api/v2/analysis/distribution - Get distribution data"

    serve(; port = port, middleware = [add_cors_headers])
end
