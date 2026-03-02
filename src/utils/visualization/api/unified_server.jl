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

# Include molecular domain modules
include("../python/rdkit_bridge.jl")
include("../../../applications/molecular_generation.jl")
include("../domains/molecular.jl")

# Include database module for persistence
include("../core/database.jl")

# RDKit readiness flag — set during init, checked before molecular training
const RDKIT_AVAILABLE = Ref(false)

# Initialize RDKit bridge (must be called explicitly since include() doesn't trigger __init__)
try
    RDKitBridge.init_rdkit!()
    validate_fragment_library!()
    RDKIT_AVAILABLE[] = true
    @info "RDKit initialized successfully — molecular domain available"
catch e
    @warn "RDKitBridge initialization failed (molecular domain unavailable): $e"
end

# ============================================
# Domain Registration (Phase 1)
# ============================================

# Register built-in domains with the registry
function register_builtin_domains!()
    register_domain!("grid_world", GridWorldAdapter)
    register_domain!("molecule", MolecularAdapter)
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
    elseif domain_type == "molecule"
        return create_molecule_model_and_adapter(config)
    else
        error("Unsupported domain type: $domain_type. Available: grid_world, molecule")
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

"""
    create_molecule_model_and_adapter(config::Dict)

Create a Molecular GFlowNet model and MolecularAdapter.
"""
function create_molecule_model_and_adapter(config::Dict)
    hidden_dim = get(config, "hidden_dim", 256)
    learning_rate = Float64(get(config, "learning_rate", 0.001))
    max_fragments = Int(get(config, "max_fragments", 8))

    objective_str = get(config, "objective", "TRAJECTORY_BALANCE")
    needs_backward = objective_str == "DETAILED_BALANCE" ||
                     objective_str == "TRAJECTORY_LIKELIHOOD_MAXIMIZATION"
    needs_flow_est = objective_str == "FLOW_MATCHING" ||
                     objective_str == "DIRECT_FLOW_OBJECTIVE"

    model = create_molecular_gflownet(
        hidden_dim = hidden_dim,
        learning_rate = learning_rate,
        include_backward = needs_backward,
        include_flow_estimator = needs_flow_est,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,
    )

    adapter = MolecularAdapter(max_fragments, FRAGMENT_LIBRARY, Dict[])
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
# JSON Sanitization (NaN/Inf → null for JSON spec)
# ============================================

"""Replace NaN/Inf values with `nothing` (JSON null) recursively."""
function sanitize_for_json(x)
    if x isa AbstractFloat
        return (isnan(x) || isinf(x)) ? nothing : x
    elseif x isa AbstractDict
        return Dict(k => sanitize_for_json(v) for (k, v) in x)
    elseif x isa AbstractVector
        return [sanitize_for_json(v) for v in x]
    elseif x isa Tuple
        return [sanitize_for_json(v) for v in x]
    else
        return x
    end
end

"""Serialize to JSON with NaN/Inf sanitization."""
safe_json(content; kwargs...) = json(sanitize_for_json(content); kwargs...)

# ============================================
# Health Check Endpoint
# ============================================

@get "/health" function(req)
    return safe_json(Dict(
        "status" => "ok",
        "server" => "unified_server",
        "real_training" => true
    ))
end

@get "/api/v2/health" function(req)
    return safe_json(Dict(
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
    return safe_json(Dict(
        "domains" => domains,
        "count" => length(domains),
        "builtin_count" => builtin_domain_count()
    ))
end

# GET /api/v2/domains/:id - Get detailed information about a specific domain
@get "/api/v2/domains/{id}" function(req, id::String)
    info = get_domain_info(id)
    if info === nothing
        return safe_json(Dict(
            "error" => "Domain not found: $id",
            "available" => domain_ids()
        ))
    end
    return safe_json(info)
end

# GET /api/v2/domains/:id/schema - Get the JSON Schema for a domain's configuration
@get "/api/v2/domains/{id}/schema" function(req, id::String)
    schema = get_domain_schema(id)
    if schema === nothing
        return safe_json(Dict("error" => "Domain not found: $id"))
    end
    return safe_json(schema)
end

# POST /api/v2/domains/:id/validate - Validate configuration for a specific domain
@post "/api/v2/domains/{id}/validate" function(req, id::String)
    config = JSON3.read(String(req.body), Dict)

    is_valid, error_msg = validate_domain_config(id, config)
    return safe_json(Dict(
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
        return safe_json(Dict(
            "has_session"       => false,
            "available_domains" => domain_ids(),
            "domains"           => list_domains(),
            "message"           => "No active session. Start training to initialize."
        ))
    end
    return safe_json(Dict(
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

    # Check RDKit readiness for molecular domain
    domain_type = get(config, "domain_type", "grid_world")
    if domain_type == "molecule" && !RDKIT_AVAILABLE[]
        return safe_json(Dict(
            "error" => "Molecular domain unavailable: RDKit failed to initialize. Check backend logs.",
            "domain_type" => "molecule",
            "rdkit_available" => false
        ), status=503)
    end

    # Stop existing training if any
    if CURRENT_SESSION[] !== nothing && CURRENT_SESSION[].is_training
        CURRENT_SESSION[].is_training = false
        sleep(0.1)  # Let the training loop exit
    end

    session = try
        create_session(config)
    catch e
        @error "Failed to create training session" exception=e
        return safe_json(Dict(
            "error" => "Failed to initialize model: $(sprint(showerror, e))",
            "domain_type" => domain_type
        ), status=400)
    end
    CURRENT_SESSION[] = session
    session.is_training = true
    session.start_time  = now()

    # Record session in database (if available)
    if MOL_DB[] !== nothing
        try
            db_create_session!(session.id, domain_type, config)
        catch e
            @warn "Failed to record session in database" exception=e
        end
    end

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
            # Always mark training as stopped when loop exits
            session.is_training = false
            # Record session completion in database
            if MOL_DB[] !== nothing
                try
                    valid_losses = filter(!isnan, session.losses)
                    final_loss = isempty(valid_losses) ? 0.0 : valid_losses[end]
                    mol_count = session.adapter isa MolecularAdapter ? length(session.adapter.generated_molecules) : 0
                    db_complete_session!(session.id, final_loss, mol_count)
                catch e
                    @warn "Failed to record session completion" exception=e
                end
            end
        end
    end

    return safe_json(Dict(
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
    return safe_json(Dict("status" => "stopped"))
end

@post "/api/v2/training/pause" function(req)
    session = CURRENT_SESSION[]
    if session !== nothing
        session.is_paused = !session.is_paused
    end
    return safe_json(Dict("paused" => session !== nothing && session.is_paused))
end

# Extend training from current state without restarting
@post "/api/v2/training/extend" function(req)
    body = JSON3.read(String(req.body), Dict)
    additional = Int(get(body, "additional_iterations", 0))

    session = CURRENT_SESSION[]
    if session === nothing
        return safe_json(Dict("error" => "No active session"))
    end
    if additional <= 0
        return safe_json(Dict("error" => "additional_iterations must be positive"))
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

    return safe_json(Dict(
        "status"             => "extended",
        "previous_total"     => session.total_iterations - additional,
        "new_total"          => session.total_iterations,
        "current_iteration"  => session.current_iteration,
        "additional"         => additional
    ))
end

# ============================================
# API v2 Endpoints — Session History (Database-backed)
# ============================================

@get "/api/v2/sessions" function(req)
    if MOL_DB[] === nothing
        return safe_json(Dict("sessions" => [], "error" => "Database not initialized"))
    end
    limit  = parse(Int, get(queryparams(req), "limit", "20"))
    offset = parse(Int, get(queryparams(req), "offset", "0"))
    sessions = db_list_sessions(; limit=limit, offset=offset)
    return safe_json(Dict("sessions" => sessions, "count" => length(sessions)))
end

@get "/api/v2/sessions/{id}/molecules" function(req, id::String)
    if MOL_DB[] === nothing
        return safe_json(Dict("molecules" => [], "total" => 0, "error" => "Database not initialized"))
    end
    limit    = parse(Int, get(queryparams(req), "limit", "20"))
    offset   = parse(Int, get(queryparams(req), "offset", "0"))
    sort_by  = get(queryparams(req), "sort_by", "reward")
    result = db_query_molecules(; session_id=id, sort_by=sort_by, limit=limit, offset=offset)
    return safe_json(Dict("molecules" => result.molecules, "total" => result.total))
end

@get "/api/v2/molecules/all" function(req)
    if MOL_DB[] === nothing
        return safe_json(Dict("molecules" => [], "total" => 0, "error" => "Database not initialized"))
    end
    limit     = parse(Int, get(queryparams(req), "limit", "50"))
    offset    = parse(Int, get(queryparams(req), "offset", "0"))
    sort_by   = get(queryparams(req), "sort_by", "reward")
    min_qed   = let v = get(queryparams(req), "min_qed", nothing); v !== nothing ? parse(Float64, v) : nothing end
    min_reward = let v = get(queryparams(req), "min_reward", nothing); v !== nothing ? parse(Float64, v) : nothing end
    result = db_query_molecules(; min_qed=min_qed, min_reward=min_reward, sort_by=sort_by, limit=limit, offset=offset)
    return safe_json(Dict("molecules" => result.molecules, "total" => result.total))
end

# ============================================
# API v2 Endpoints — Training State
# ============================================

@get "/api/v2/training/state" function(req)
    session = CURRENT_SESSION[]
    if session === nothing
        return safe_json(Dict("has_session" => false))
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

    # Domain-agnostic visitation statistics
    visitation_data = if session.adapter isa GridWorldAdapter
        # Grid world: compute cell visitation counts
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
        Dict(
            "visitation_counts"    => visitation_counts,
            "total_states_visited" => total_states_visited,
            "max_visits"           => max_visits,
            "coverage"             => total_states_visited / max(total_cells, 1),
        )
    elseif session.adapter isa MolecularAdapter
        # Molecular domain: molecule-level stats
        adapter = session.adapter
        all_smiles = [m["smiles"] for m in adapter.generated_molecules]
        Dict(
            "total_molecules" => length(adapter.generated_molecules),
            "unique_smiles"   => length(unique(all_smiles)),
        )
    else
        Dict()
    end

    return safe_json(Dict(
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
        # Domain-specific visitation/stats
        visitation_data...,
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
        return safe_json(Dict("has_session" => false))
    end
    return safe_json(Dict(
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
        return safe_json(Dict("trajectories" => [], "domain" => nothing))
    end
    viz_trajectories = [
        trajectory_to_viz_data(session.adapter, traj, "traj_$i")
        for (i, traj) in enumerate(session.trajectory_buffer)
    ]
    return safe_json(Dict(
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
        return safe_json(Dict("supported" => false, "error" => "No session"))
    end
    return safe_json(compute_flow_field(session.adapter, session.model))
end

@get "/api/v2/analysis/distribution" function(req)
    session = CURRENT_SESSION[]
    if session === nothing
        return safe_json(Dict("supported" => false, "error" => "No session"))
    end
    return safe_json(compute_distribution_data(session.adapter, session.model, session.trajectory_buffer))
end

# ============================================
# API v2 Endpoints — Molecular Domain
# ============================================

@get "/api/v2/molecular/molecules" function(req)
    limit   = parse(Int, get(queryparams(req), "limit", "20"))
    offset  = parse(Int, get(queryparams(req), "offset", "0"))
    sort_by = get(queryparams(req), "sort_by", "reward")

    # Map frontend sort names to in-memory field paths
    sort_key_map = Dict(
        "reward" => m -> -get(m, "reward", 0.0),
        "qed"    => m -> -get(get(m, "properties", Dict()), "qed", 0.0),
        "mw"     => m -> -get(get(m, "properties", Dict()), "molecular_weight", 0.0),
        "logp"   => m -> -get(get(m, "properties", Dict()), "logp", 0.0),
        "sa"     => m -> get(get(m, "properties", Dict()), "synthetic_accessibility", 10.0),
        "generation_step" => m -> -get(m, "generation_step", 0),
    )

    # Map frontend sort names to DB column names
    db_sort_map = Dict(
        "reward" => "reward", "qed" => "qed", "mw" => "molecular_weight",
        "logp" => "logp", "sa" => "sa_score", "generation_step" => "generation_step",
    )

    # Try in-memory first (active session)
    session = CURRENT_SESSION[]
    if session !== nothing && session.adapter isa MolecularAdapter
        molecules = session.adapter.generated_molecules
        if !isempty(molecules)
            # Sort in-memory
            key_fn = get(sort_key_map, sort_by, sort_key_map["reward"])
            sorted = sort(molecules; by=key_fn)
            total = length(sorted)
            slice = sorted[min(offset+1, total+1) : min(total, offset+limit)]
            return safe_json(Dict("molecules" => slice, "total" => total))
        end
    end

    # Fallback: query from database
    if MOL_DB[] !== nothing
        db_col = get(db_sort_map, sort_by, "reward")
        result = db_query_molecules(; sort_by=db_col, limit=limit, offset=offset)
        return safe_json(Dict("molecules" => result.molecules, "total" => result.total))
    end

    return safe_json(Dict("molecules" => [], "total" => 0))
end

@get "/api/v2/molecular/molecules/{id}" function(req, id::String)
    session = CURRENT_SESSION[]
    session === nothing && return safe_json(Dict("error" => "No session"), status=404)
    adapter = session.adapter
    !(adapter isa MolecularAdapter) && return safe_json(Dict("error" => "Not a molecular session"), status=400)

    idx = findfirst(m -> m["id"] == id, adapter.generated_molecules)
    idx === nothing && return safe_json(Dict("error" => "Molecule not found: $id"), status=404)
    return safe_json(adapter.generated_molecules[idx])
end

@get "/api/v2/molecular/space" function(req)
    method = get(queryparams(req), "method", "pca")

    # Try in-memory first (active session)
    mols = Dict[]
    session = CURRENT_SESSION[]
    if session !== nothing && session.adapter isa MolecularAdapter
        mols = session.adapter.generated_molecules
    end

    # Fallback: query from database
    if isempty(mols) && MOL_DB[] !== nothing
        result = db_query_molecules(; sort_by="reward", limit=200, offset=0)
        mols = result.molecules
    end

    isempty(mols) && return safe_json(Dict("points" => []))

    # Filter to molecules that have fingerprints
    mols_with_fp = filter(m -> get(m, "fingerprint", nothing) !== nothing, mols)

    if isempty(mols_with_fp)
        # No fingerprints available — return points without projection (use property-based 2D layout)
        points = [Dict(
            "id"               => m["id"],
            "x"                => get(get(m, "properties", Dict()), "logp", 0.0),
            "y"                => get(get(m, "properties", Dict()), "qed", 0.0),
            "smiles"           => m["smiles"],
            "reward"           => get(m, "reward", 0.0),
            "properties"       => get(m, "properties", Dict()),
            "cluster_id"       => 0,
            "generation_epoch" => get(m, "generation_step", 0),
        ) for m in mols]
        return safe_json(Dict("points" => points, "projection" => "property_scatter"))
    end

    fps = [Vector{Float32}(m["fingerprint"]) for m in mols_with_fp]

    try
        coords = RDKitBridge.compute_projection(fps, method)
        points = [Dict(
            "id"               => mols_with_fp[i]["id"],
            "x"                => coords[i]["x"],
            "y"                => coords[i]["y"],
            "smiles"           => mols_with_fp[i]["smiles"],
            "reward"           => mols_with_fp[i]["reward"],
            "properties"       => mols_with_fp[i]["properties"],
            "cluster_id"       => 0,
            "generation_epoch" => mols_with_fp[i]["generation_step"],
        ) for i in 1:length(mols_with_fp)]
        return safe_json(Dict("points" => points, "projection" => method))
    catch e
        @warn "Chemical space projection failed" exception=e
        return safe_json(Dict("points" => [], "error" => string(e)))
    end
end

@get "/api/v2/molecular/admet/{id}" function(req, id::String)
    session = CURRENT_SESSION[]
    session === nothing && return safe_json(Dict("error" => "No session"), status=404)
    adapter = session.adapter
    !(adapter isa MolecularAdapter) && return safe_json(Dict("error" => "Not a molecular session"), status=400)

    idx = findfirst(m -> m["id"] == id, adapter.generated_molecules)
    idx === nothing && return safe_json(Dict("error" => "Molecule not found"), status=404)

    mol = adapter.generated_molecules[idx]
    return safe_json(RDKitBridge.compute_admet(mol["smiles"]))
end

@get "/api/v2/molecular/attribution/{id}" function(req, id::String)
    session = CURRENT_SESSION[]
    session === nothing && return safe_json(Dict("error" => "No session"), status=404)
    adapter = session.adapter
    !(adapter isa MolecularAdapter) && return safe_json(Dict("error" => "Not a molecular session"), status=400)

    idx = findfirst(m -> m["id"] == id, adapter.generated_molecules)
    idx === nothing && return safe_json(Dict("error" => "Molecule not found"), status=404)

    mol = adapter.generated_molecules[idx]
    scores = try
        RDKitBridge.compute_atom_attribution(mol["smiles"])
    catch
        Float64[]
    end

    return safe_json(Dict(
        "molecule_id"      => id,
        "smiles"           => mol["smiles"],
        "atom_scores"      => scores,
        "attribution_type" => "reward",
    ))
end

@get "/api/v2/molecular/reward-decomposition/{id}" function(req, id::String)
    session = CURRENT_SESSION[]
    session === nothing && return safe_json(Dict("error" => "No session"), status=404)
    adapter = session.adapter
    !(adapter isa MolecularAdapter) && return safe_json(Dict("error" => "Not a molecular session"), status=400)

    idx = findfirst(m -> m["id"] == id, adapter.generated_molecules)
    idx === nothing && return safe_json(Dict("error" => "Molecule not found"), status=404)

    mol = adapter.generated_molecules[idx]
    props = RDKitBridge.compute_mol_properties(mol["smiles"])
    if props === nothing
        return safe_json(Dict("error" => "Could not compute properties"), status=500)
    end

    qed_score  = props.qed
    sa_norm    = clamp(1.0 - (props.sa_score - 1.0) / 9.0, 0.0, 1.0)
    logp_score = exp(-0.5 * ((props.logp - 2.5) / 2.5)^2)
    mw_score   = exp(-0.5 * ((props.mw - 350.0) / 150.0)^2)

    return safe_json(Dict(
        "molecule_id"  => id,
        "total_reward" => mol["reward"],
        "components"   => [
            Dict("name" => "QED (Drug-likeness)",    "value" => qed_score,  "weight" => 0.4, "contribution" => qed_score^0.4),
            Dict("name" => "SA (Synthetic Access.)",  "value" => sa_norm,    "weight" => 0.3, "contribution" => sa_norm^0.3),
            Dict("name" => "LogP Score",              "value" => logp_score, "weight" => 0.2, "contribution" => logp_score^0.2),
            Dict("name" => "MW Score",                "value" => mw_score,   "weight" => 0.1, "contribution" => mw_score^0.1),
        ],
    ))
end

@get "/api/v2/molecular/generation-dag/{id}" function(req, id::String)
    session = CURRENT_SESSION[]
    session === nothing && return safe_json(Dict("error" => "No session"), status=404)
    adapter = session.adapter
    !(adapter isa MolecularAdapter) && return safe_json(Dict("error" => "Not a molecular session"), status=400)

    idx = findfirst(m -> m["id"] == id, adapter.generated_molecules)
    idx === nothing && return safe_json(Dict("error" => "Molecule not found"), status=404)

    mol = adapter.generated_molecules[idx]

    # Build simplified generation DAG
    nodes = [
        Dict("id" => "step_0", "label" => "Empty", "smiles" => "", "depth" => 0, "flow" => 1.0),
        Dict("id" => "terminal", "label" => "Complete", "smiles" => mol["smiles"],
             "depth" => 1, "flow" => get(mol, "reward", 0.0) / 10.0),
    ]
    # Use "source"/"target" to match frontend DAGEdge interface
    edges = [
        Dict("source" => "step_0", "target" => "terminal", "action" => "Generate", "probability" => 1.0),
    ]

    return safe_json(Dict("nodes" => nodes, "edges" => edges))
end

@post "/api/v2/molecular/molecules/compare" function(req)
    body = JSON3.read(String(req.body), Dict)
    ids = get(body, "ids", String[])

    session = CURRENT_SESSION[]
    session === nothing && return safe_json(Dict("molecules" => []))
    adapter = session.adapter
    !(adapter isa MolecularAdapter) && return safe_json(Dict("molecules" => []))

    mols = filter(m -> m["id"] in ids, adapter.generated_molecules)
    return safe_json(Dict("molecules" => mols))
end

@post "/api/v2/molecular/export" function(req)
    body = JSON3.read(String(req.body), Dict)
    ids = get(body, "ids", String[])
    format = get(body, "format", "smiles")

    session = CURRENT_SESSION[]
    session === nothing && return safe_json(Dict("error" => "No session"))
    adapter = session.adapter
    !(adapter isa MolecularAdapter) && return safe_json(Dict("error" => "Not a molecular session"))

    mols = filter(m -> m["id"] in ids, adapter.generated_molecules)
    if isempty(mols)
        mols = adapter.generated_molecules
    end

    if format == "csv"
        header = "smiles,reward,mw,logp,qed,sa\n"
        rows = join([
            "$(m["smiles"]),$(m["reward"]),$(m["properties"]["molecular_weight"]),$(m["properties"]["logp"]),$(m["properties"]["qed"]),$(m["properties"]["synthetic_accessibility"])"
            for m in mols
        ], "\n")
        return HTTP.Response(200, ["Content-Type" => "text/csv"], body=header * rows)
    else
        smiles_list = join([m["smiles"] for m in mols], "\n")
        return HTTP.Response(200, ["Content-Type" => "text/plain"], body=smiles_list)
    end
end

@post "/api/v2/molecular/validate-smiles" function(req)
    body = JSON3.read(String(req.body), Dict)
    smiles = get(body, "smiles", "")
    valid = RDKitBridge.validate_smiles(smiles)
    canonical = valid ? RDKitBridge.canonicalize_smiles(smiles) : nothing
    return safe_json(Dict("valid" => valid, "smiles" => something(canonical, smiles)))
end

@post "/api/v2/molecular/generate" function(req)
    # Check RDKit readiness
    if !RDKIT_AVAILABLE[]
        return safe_json(Dict(
            "error" => "Molecular domain unavailable: RDKit failed to initialize.",
            "rdkit_available" => false
        ), status=503)
    end

    body = JSON3.read(String(req.body), Dict)
    body["domain_type"] = "molecule"

    # Stop existing training if any
    if CURRENT_SESSION[] !== nothing && CURRENT_SESSION[].is_training
        CURRENT_SESSION[].is_training = false
        sleep(0.1)
    end

    session = create_session(body)
    CURRENT_SESSION[] = session
    session.is_training = true
    session.start_time = now()

    TRAINING_TASK[] = @async begin
        try
            while session.is_training && session.current_iteration < session.total_iterations
                !session.is_paused && step!(session)
                sleep(0.05)
            end
        catch e
            @error "Molecular training loop error" exception=e
            session.last_error = sprint(showerror, e)
        finally
            session.is_training = false
        end
    end

    return safe_json(Dict(
        "status"     => "started",
        "session_id" => session.id,
        "domain"     => get_domain_config(session.adapter),
    ))
end

@post "/api/v2/molecular/retrain" function(req)
    return safe_json(Dict("status" => "not_implemented", "message" => "Focused retraining coming soon"))
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
function start_real_training_server(; port::Int = 8080, host::String = "127.0.0.1",
                                     db_path::String = joinpath(@__DIR__, "..", "..", "..", "..", "data", "gflownet_molecules.db"))
    # Initialize molecule database
    try
        init_database!(db_path)
        @info "Molecule database ready" path=db_path
    catch e
        @warn "Database initialization failed (persistence disabled)" exception=e
    end

    @info "Starting real GFlowNet training visualization server on $host:$port"
    @info "Frontend: http://localhost:3000 (Vite dev server)"
    @info "API base: http://$host:$port/api/v2/"
    @info ""
    @info "Available endpoints:"
    @info "  POST /api/v2/training/start    - Start training session"
    @info "  GET  /api/v2/training/state    - Get current training state"
    @info "  GET  /api/v2/training/history  - Get training history"
    @info "  GET  /api/v2/trajectories      - Get recent trajectories"
    @info "  GET  /api/v2/analysis/flow     - Get flow field data"
    @info "  GET  /api/v2/analysis/distribution - Get distribution data"

    serve(; host = host, port = port, middleware = [add_cors_headers])
end

# Auto-start when script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    start_real_training_server()
end
