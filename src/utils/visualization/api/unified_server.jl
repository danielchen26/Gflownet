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
include("../core/domain_registry.jl")
include("../core/metrics.jl")
include("../core/training_session.jl")
include("../domains/grid_world.jl")

# ============================================
# Domain Registration (Phase 1)
# ============================================

# Register built-in domains with the registry
function register_builtin_domains!()
    register_domain!("grid_world", GridWorldAdapter)
    # Future domains:
    # register_domain!("dag", DAGAdapter)
    # register_domain!("sequence", SequenceAdapter)
    # register_domain!("molecule", MoleculeAdapter)
end

# Register domains on module load
register_builtin_domains!()

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
    # NOTE: App.tsx already converts 0-indexed JS to 1-indexed Julia coordinates
    # So positions arrive here already 1-indexed — do NOT add 1 again
    peaks_config = get(config, "reward_peaks", [])
    reward_positions = Dict{Tuple{Int,Int}, Float64}()
    for peak in peaks_config
        pos = peak["position"]
        intensity = get(peak, "intensity", 10.0)
        julia_x = Int(pos[1])
        julia_y = Int(pos[2])
        # Validate bounds
        if 1 <= julia_x <= grid_size && 1 <= julia_y <= grid_size
            reward_positions[(julia_x, julia_y)] = Float64(intensity)
        else
            @warn "Reward position out of bounds" pos=(julia_x, julia_y) grid_size
        end
    end

    # Default peaks if none specified
    if isempty(reward_positions)
        reward_positions[(grid_size, grid_size)] = 10.0
        reward_positions[(1, grid_size)]         = 8.0
        reward_positions[(grid_size, 1)]         = 8.0
    end

    # Save original rewards for display (before shaping)
    original_reward_positions = copy(reward_positions)

    # Domain-agnostic reward shaping via adapter interface
    # Each domain implements its own apply_reward_shaping() to compensate
    # for structural path asymmetry specific to that domain's DAG structure
    # Shaped rewards are only used internally for training — the adapter stores
    # original rewards so get_domain_config() returns user-facing intensities.
    reward_shaping = Bool(get(config, "reward_shaping", true))
    training_reward_positions = if reward_shaping
        adapter_for_shaping = GridWorldAdapter(grid_size, reward_positions)
        apply_reward_shaping(adapter_for_shaping, reward_positions)
    else
        reward_positions
    end

    # Determine if backward policy / flow estimator is needed
    objective_str = get(config, "objective", "TRAJECTORY_BALANCE")
    # TLM requires backward policy to learn path counts (ICLR 2025)
    needs_backward = objective_str == "DETAILED_BALANCE" || objective_str == "TRAJECTORY_LIKELIHOOD_MAXIMIZATION"
    needs_flow_est = objective_str == "FLOW_MATCHING" || objective_str == "DIRECT_FLOW_OBJECTIVE"

    # Create the real GFlowNet model using high-level API
    # CRITICAL: Use LEARNABLE_ESTIMATION for proper TB training
    # With SIMPLE_ESTIMATION (Z=1), the TB equation log(P_F) = log(R) is unsatisfiable
    # because P_F is a probability (<1) and R can be >1.
    # LEARNABLE_ESTIMATION allows the model to learn Z such that Z*P_F ∝ R
    # NOTE: Model uses SHAPED rewards for training; adapter stores ORIGINAL for display
    model = GFlowNet.create_grid_world_gflownet(
        grid_size               = grid_size,
        reward_positions        = training_reward_positions,
        hidden_dim              = get(config, "hidden_dim", 64),
        learning_rate           = get(config, "learning_rate", 0.005),
        include_backward        = needs_backward,
        include_flow_estimator  = needs_flow_est,
        allow_all_moves         = get(config, "allow_all_moves", false),
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,
    )

    # Adapter stores ORIGINAL rewards for display (get_domain_config, etc.)
    adapter = GridWorldAdapter(grid_size, original_reward_positions)
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
# Health Check Endpoint
# ============================================

@get "/health" function(req)
    return json(Dict(
        "status" => "ok",
        "server" => "unified_server",
        "real_training" => true
    ))
end

@get "/api/v2/health" function(req)
    return json(Dict(
        "status" => "ok",
        "server" => "unified_server",
        "real_training" => true
    ))
end

# ============================================
# API v2 Endpoints — Domain Registry (Phase 1)
# ============================================

# GET /api/v2/domains - List all registered domains with their metadata
@get "/api/v2/domains" function(req)
    domains = list_domains()
    return json(Dict(
        "domains" => domains,
        "count" => length(domains),
        "builtin_count" => builtin_domain_count()
    ))
end

# GET /api/v2/domains/:id - Get detailed information about a specific domain
@get "/api/v2/domains/{id}" function(req, id::String)
    info = get_domain_info(id)
    if info === nothing
        return json(Dict(
            "error" => "Domain not found: $id",
            "available" => domain_ids()
        ))
    end
    return json(info)
end

# GET /api/v2/domains/:id/schema - Get the JSON Schema for a domain's configuration
@get "/api/v2/domains/{id}/schema" function(req, id::String)
    schema = get_domain_schema(id)
    if schema === nothing
        return json(Dict("error" => "Domain not found: $id"))
    end
    return json(schema)
end

# POST /api/v2/domains/:id/validate - Validate configuration for a specific domain
@post "/api/v2/domains/{id}/validate" function(req, id::String)
    config = JSON3.read(String(req.body), Dict)

    is_valid, error_msg = validate_domain_config(id, config)
    return json(Dict(
        "valid" => is_valid,
        "error" => error_msg
    ))
end

# ============================================
# API v2 Endpoints — Domain Info (Legacy)
# ============================================

@get "/api/v2/domain/info" function(req)
    session = CURRENT_SESSION[]
    if session === nothing
        return json(Dict(
            "has_session"       => false,
            "available_domains" => domain_ids(),
            "domains"           => list_domains(),
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
            session.last_error  = sprint(showerror, e)
        finally
            # Always mark training as stopped when loop exits (normal completion,
            # error, or manual stop). This allows the extend endpoint to detect
            # that the loop has exited and relaunch it.
            session.is_training = false
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

# Extend training from current state without restarting
@post "/api/v2/training/extend" function(req)
    body = JSON3.read(String(req.body), Dict)
    additional = Int(get(body, "additional_iterations", 0))

    session = CURRENT_SESSION[]
    if session === nothing
        return json(Dict("error" => "No active session"))
    end
    if additional <= 0
        return json(Dict("error" => "additional_iterations must be positive"))
    end

    session.total_iterations += additional

    # If training loop already exited, relaunch it
    if !session.is_training
        session.is_training = true
        session.is_paused   = false
        TRAINING_TASK[] = @async begin
            try
                while session.is_training && session.current_iteration < session.total_iterations
                    if !session.is_paused
                        step!(session)
                    end
                    sleep(0.05)
                end
            catch e
                @error "Training loop fatal error" exception=e
                session.last_error  = sprint(showerror, e)
            finally
                session.is_training = false
            end
        end
    end

    return json(Dict(
        "status"             => "extended",
        "previous_total"     => session.total_iterations - additional,
        "new_total"          => session.total_iterations,
        "current_iteration"  => session.current_iteration,
        "additional"         => additional
    ))
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

    # Compute current epsilon (annealed if epsilon_decay is true)
    # This shows users the current exploration level in real-time
    current_epsilon = if session.config.epsilon_decay
        session.config.epsilon * (1.0 - session.current_iteration / max(session.total_iterations, 1))
    else
        session.config.epsilon
    end

    # Compute state visitation statistics for flow field sidebar
    grid_size = session.adapter.grid_size
    visitation_counts = Dict{String, Int}()
    for traj in recent
        for s in traj.states
            key = "$(s.x),$(s.y)"
            visitation_counts[key] = get(visitation_counts, key, 0) + 1
        end
    end
    max_visits = isempty(visitation_counts) ? 0 : maximum(values(visitation_counts))
    total_cells = grid_size * grid_size
    total_states_visited = length(visitation_counts)

    return json(Dict(
        "has_session"          => true,
        "is_training"          => session.is_training,
        "is_paused"            => session.is_paused,
        "is_real_training"     => true,
        "current_iteration"    => session.current_iteration,
        "total_iterations"     => session.total_iterations,
        "progress"             => session.current_iteration / max(session.total_iterations, 1),
        # Latest metrics
        "latest_loss"          => isempty(session.losses) ? nothing : session.losses[end],
        "latest_reward"        => isempty(session.rewards) ? nothing : session.rewards[end],
        "latest_gradient_norm" => isempty(session.gradient_norms) ? nothing : session.gradient_norms[end],
        # Exploration parameters
        "current_epsilon"      => current_epsilon,
        "entropy_weight"       => session.config.entropy_weight,
        "epsilon_decay"        => session.config.epsilon_decay,
        "z_learning_rate_multiplier" => session.config.z_learning_rate_multiplier,
        # Computed metrics
        "metrics"              => universal_metrics,
        "domain_metrics"       => domain_metrics,
        # State visitation stats (for flow field sidebar)
        "visitation_counts"    => visitation_counts,
        "total_states_visited" => total_states_visited,
        "max_visits"           => max_visits,
        "coverage"             => total_states_visited / max(total_cells, 1),
        "flow_statistics"      => Dict(
            "mean_log_Z"             => get(universal_metrics, "mean_log_Z", 0.0),
            "mean_reward"            => get(universal_metrics, "mean_reward", 0.0),
            "progress"               => session.current_iteration / max(session.total_iterations, 1),
            "mean_trajectory_length" => get(universal_metrics, "mean_trajectory_length", 0.0)
        ),
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

# Auto-start when script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    start_real_training_server()
end
