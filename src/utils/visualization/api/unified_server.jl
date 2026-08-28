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
include("../python/oracle_bridge.jl")           # TDC oracle bridge (after rdkit, before oracle_mgr)
include("../core/oracle_manager.jl")           # Oracle config, cache, budget (after oracle_bridge, before mol_gen)
include("../core/pmo_benchmark.jl")            # PMO 23-task benchmark runner (after oracle_manager)
include("../../../applications/molecular_generation.jl")
include("../domains/molecular.jl")
include("../domains/reaction_molecular.jl")

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
    register_domain!("reaction_molecule", ReactionMolecularAdapter)
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
    elseif domain_type == "reaction_molecule"
        return create_reaction_molecule_model_and_adapter(config)
    else
        error("Unsupported domain type: $domain_type. Available: grid_world, molecule, reaction_molecule")
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
Supports oracle configuration for target-specific optimization and MOGFN activation.
"""
function create_molecule_model_and_adapter(config::Dict)
    hidden_dim = get(config, "hidden_dim", 256)
    learning_rate = Float64(get(config, "learning_rate", 0.001))
    max_fragments = Int(get(config, "max_fragments", 8))

    objective_str = get(config, "objective", "TRAJECTORY_BALANCE")
    needs_backward = objective_str in ("DETAILED_BALANCE", "TRAJECTORY_LIKELIHOOD_MAXIMIZATION")
    needs_flow_est = objective_str in ("FLOW_MATCHING", "DIRECT_FLOW_OBJECTIVE")

    # Oracle configuration
    oracle_config_raw = get(config, "oracles", nothing)
    benchmark_mode = Bool(get(config, "benchmark_mode", false))
    oracle_budget = Int(get(config, "oracle_budget", 10000))

    oracle_mgr = if oracle_config_raw !== nothing && !isempty(oracle_config_raw)
        oracle_names = [String(o["name"]) for o in oracle_config_raw]
        try
            OracleBridge.init_oracles!(oracle_names;
                cache_dir=joinpath(@__DIR__, "..", "..", "..", "..", "data", "tdc_cache"))
        catch e
            @warn "Oracle initialization failed" exception=e
        end

        configs = [OracleConfig(String(o["name"]), Float64(get(o, "weight", 1.0)))
                   for o in oracle_config_raw]
        OracleManager(configs, oracle_budget, 0,
                      Dict{String,Dict{String,Float64}}(), benchmark_mode)
    else
        nothing
    end

    # Compute total objectives for MOGFN
    n_objectives = if benchmark_mode && oracle_mgr !== nothing
        length(oracle_mgr.configs)  # benchmark: oracle-only
    else
        4 + (oracle_mgr !== nothing ? length(oracle_mgr.configs) : 0)  # QED+SA+logP+MW+oracles
    end

    # Activate MOGFN when oracles are configured or MULTI_OBJECTIVE_TB requested
    use_mogfn = objective_str == "MULTI_OBJECTIVE_TB" || oracle_mgr !== nothing

    model = if use_mogfn
        create_mogfn_molecular_gflownet(
            n_objectives = n_objectives,
            hidden_dim = hidden_dim,
            learning_rate = learning_rate,
            include_backward = needs_backward,
        )
    else
        create_molecular_gflownet(
            hidden_dim = hidden_dim,
            learning_rate = learning_rate,
            include_backward = needs_backward,
            include_flow_estimator = needs_flow_est,
            partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,
        )
    end

    adapter = MolecularAdapter(max_fragments, FRAGMENT_LIBRARY, Dict[], oracle_mgr)
    return model, adapter
end

"""
    create_reaction_molecule_model_and_adapter(config::Dict)

Create a Reaction-Based Molecular GFlowNet model and ReactionMolecularAdapter.
"""
function create_reaction_molecule_model_and_adapter(config::Dict)
    hidden_dim = get(config, "hidden_dim", 256)
    learning_rate = Float64(get(config, "learning_rate", 0.001))
    max_steps = Int(get(config, "max_steps", MAX_REACTION_STEPS))

    # Initialize reaction engine if not already done
    if !RDKitBridge._reaction_engine_available[]
        try
            RDKitBridge.load_reaction_templates!()
            RDKitBridge.init_reaction_engine!()
        catch e
            @warn "Reaction engine initialization failed" exception=e
        end
    end

    model = create_reaction_gflownet(
        hidden_dim = hidden_dim,
        learning_rate = learning_rate,
    )

    adapter = ReactionMolecularAdapter(max_steps, Dict[])
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

"""Check if adapter is any molecular domain type (fragment-based or reaction-based)."""
_is_molecular_adapter(adapter) = adapter isa MolecularAdapter || adapter isa ReactionMolecularAdapter

"""Get the oracle manager from the current session (if molecular with oracles)."""
function _get_oracle_manager()
    session = CURRENT_SESSION[]
    session === nothing && return nothing
    session.adapter isa MolecularAdapter || return nothing
    return session.adapter.oracle_manager
end

"""Get generated molecules from any molecular adapter."""
_get_molecules(adapter::MolecularAdapter) = adapter.generated_molecules
_get_molecules(adapter::ReactionMolecularAdapter) = adapter.generated_molecules
_get_molecules(::Any) = Dict[]

# ============================================
# JSON Sanitization (NaN/Inf → null for JSON spec)
# ============================================

"""Replace NaN/Inf values with `nothing` (JSON null) recursively."""
function sanitize_for_json(x)
    if x isa AbstractFloat
        return (isnan(x) || isinf(x)) ? nothing : x
    elseif x isa AbstractDict
        return Dict(k => sanitize_for_json(v) for (k, v) in x)
    elseif x isa AbstractMatrix
        return [sanitize_for_json(x[i, :]) for i in 1:size(x, 1)]
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
    if (domain_type == "molecule" || domain_type == "reaction_molecule") && !RDKIT_AVAILABLE[]
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

    # Run the FIRST training step on the main thread, before spawning the loop.
    #
    # Task stacks are far smaller than the main thread's. Type-inferring and
    # compiling the Zygote pullback for train_step! recurses deeply enough
    # (typeinf_edge -> typeinf -> abstract_call_method, hundreds deep) to
    # overflow a Task stack, which killed the entire server with
    #   [pid] signal 11 (2): Segmentation fault: 11
    #   ... withgradient -> pullback -> jl_type_infer -> typeinf_ext_toplevel
    #   train_step! at src/training/training.jl:322
    #   step! at core/training_session.jl:303
    #   #125 at api/unified_server.jl -> start_task
    # i.e. POST /api/v2/training/start returned 200 and then the process died.
    #
    # Doing one step here forces that compilation where there is stack headroom;
    # the async loop below then only executes already-compiled code. The step is
    # real work the loop would have done anyway, so nothing is duplicated.
    try
        if session.is_training && session.current_iteration < session.total_iterations
            step!(session)
        end
    catch e
        @error "First training step failed" exception = (e, catch_backtrace())
        session.last_error = sprint(showerror, e)
        session.is_training = false
    end

    # Launch async training loop.
    #
    # `@async` runs cooperatively on the same thread as Oxygen, so session field
    # reads and writes are safe without locks.
    #
    # DO NOT CHANGE THIS TO `Threads.@spawn`. It was investigated properly and it
    # does not work on this stack. Because Python is unavoidable inside `step!` --
    # `reward(::MolState)` calls RDKitBridge.compute_mol_properties
    # (molecular_generation.jl:362) from inside `Zygote.withgradient` -- moving the
    # loop to another thread puts PythonCall on a non-primary thread. Three
    # reproductions, all against the real server:
    #
    #   plain Threads.@spawn      -> "Fatal Python error: PyThreadState_Get: the
    #                                function must be called with the GIL held",
    #                                abort trap, dead about one second in
    #   concurrent RDKit, no GIL  -> signal 11 segfault inside rdBase.so / boost-python
    #   WITH correct GIL locking  -> whole process froze in __psynch_cvwait. A
    #                                pure-Julia watchdog touching no Python stopped
    #                                ticking after 6 s, i.e. Julia's stop-the-world GC
    #                                could not complete. Control: identical program
    #                                with GC.enable(false) survived 3/3 iterations.
    #
    # Root cause is upstream: PyGILState_Ensure is a bare ccall
    # (PythonCall src/C/pointers.jl:303) with no gc_safe transition, which Julia
    # 1.11.6 does not offer. PythonCall.jl issue #627 is still open; the maintainer
    # reports every locking strategy they tried could be made to deadlock.
    #
    # Also note Threads.nthreads() is 1 here, so the change would be inert anyway,
    # and the launcher must NOT be given `-t auto`.
    #
    # The real cost is a 1.43 s molecular step blocking every endpoint. Fix that by
    # making the step cheaper -- compute_mol_properties is called per molecule per
    # iteration and is UNCACHED -- or by moving training into a separate PROCESS,
    # which sidesteps the GIL entirely. Not by threading it.
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
                    mol_count = hasproperty(session.adapter, :generated_molecules) ? length(session.adapter.generated_molecules) : 0
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
    elseif session.adapter isa ReactionMolecularAdapter
        # Reaction molecular domain: synthesis-level stats
        adapter = session.adapter
        all_smiles = [m["smiles"] for m in adapter.generated_molecules]
        Dict(
            "total_molecules"  => length(adapter.generated_molecules),
            "unique_smiles"    => length(unique(all_smiles)),
            "domain_subtype"   => "reaction_molecule",
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
    if session !== nothing && _is_molecular_adapter(session.adapter)
        molecules = _get_molecules(session.adapter)
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
    !_is_molecular_adapter(adapter) && return safe_json(Dict("error" => "Not a molecular session"), status=400)

    molecules = _get_molecules(adapter)
    idx = findfirst(m -> m["id"] == id, molecules)
    idx === nothing && return safe_json(Dict("error" => "Molecule not found: $id"), status=404)
    return safe_json(molecules[idx])
end

# Memo for the chemical-space projection.
#
# The frontend polls /api/v2/molecular/space every 10 s
# (web/src/pages/ChemicalSpaceExplorer.tsx:34) with method=umap by default, and
# each request refit UMAP from scratch: rdkit_bridge.jl constructs a new reducer
# and calls fit_transform unconditionally, with no cache anywhere in the path.
# Since this runs on the same cooperative thread as the HTTP server and the
# training loop, every poll stalled every other endpoint -- which is why
# /api/v2/domain/info, whose handler is sub-millisecond, was measured at 4.08 s.
#
# Keyed on (method, number of molecules with fingerprints). Sound because the
# molecule list is APPEND-ONLY (domains/molecular.jl pushes, and there is no
# deleteat!/resize! anywhere), so the count changing is exactly the condition
# under which the projection must be recomputed. UMAP and t-SNE now pass
# random_state=42, so a cached embedding is the same one a refit would produce.
const _PROJECTION_CACHE = Ref{Union{Nothing,Tuple{String,Int,Vector{Dict}}}}(nothing)

@get "/api/v2/molecular/space" function(req)
    method = get(queryparams(req), "method", "pca")

    # Try in-memory first (active session)
    mols = Dict[]
    session = CURRENT_SESSION[]
    if session !== nothing && _is_molecular_adapter(session.adapter)
        mols = _get_molecules(session.adapter)
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

    # Serve the memo unless the method changed or the molecule set grew
    # MATERIALLY.
    #
    # Keying on the exact count was wrong: during training the count grows every
    # iteration, so the key never repeated and the cache missed on every single
    # request -- measured, this made the endpoint WORSE under load (0.56 s -> 4.71 s)
    # because each miss is a full UMAP refit and molecules now arrive 2.5x faster.
    # It only ever hit while training was stopped, which is precisely when nobody
    # is watching.
    #
    # A UMAP scatter of N points does not visibly change when 5% more points are
    # added, so a 20% growth threshold is used. That also bounds the total number
    # of refits over a whole run to about log(N)/log(1.2) rather than one per
    # request, while guaranteeing the view never lags the data by more than 20%.
    # An absolute floor as well as the ratio. Early in a run 20% of a few hundred
    # molecules is reached in about ten seconds, i.e. within one poll interval, so
    # the ratio alone still refit on roughly every other request. Requiring BOTH
    # a 20% increase AND at least 250 new molecules keeps refits rare at every
    # scale.
    n_now = length(mols_with_fp)
    cached = _PROJECTION_CACHE[]
    if cached !== nothing && cached[1] == method &&
       n_now < max(ceil(Int, cached[2] * 1.2), cached[2] + 250)
        return safe_json(Dict("points" => cached[3], "projection" => method,
                              "cached" => true, "projected_count" => cached[2],
                              "current_count" => n_now))
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
        _PROJECTION_CACHE[] = (method, length(mols_with_fp), points)
        return safe_json(Dict("points" => points, "projection" => method,
                              "cached" => false))
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
# Gap 1: Diversity Analysis Endpoints
# ============================================

@post "/api/v2/molecular/diversity" function(req)
    body = JSON3.read(String(req.body), Dict)
    ids = get(body, "ids", nothing)
    sample_size = get(body, "sample_size", 500)

    try
        # Get fingerprints from database
        fp_data = if ids !== nothing && !isempty(ids)
            db_get_fingerprints(ids=String.(ids))
        else
            db_get_fingerprints(limit=sample_size)
        end

        if isempty(fp_data.fingerprints)
            return safe_json(Dict("error" => "No fingerprints available"))
        end

        # Compute Tanimoto diversity stats
        stats = RDKitBridge.compute_diversity_stats(fp_data.fingerprints)

        # Compute scaffold diversity
        scaffold_stats = RDKitBridge.compute_scaffold_diversity(fp_data.smiles)

        # Compute nearest neighbors (for small sets or sampled)
        nn_results = if length(fp_data.fingerprints) <= 200
            RDKitBridge.compute_nearest_neighbors(fp_data.fingerprints, fp_data.ids; k=3)
        else
            # Sample for nearest neighbor computation
            sample_idx = sort(randperm(length(fp_data.fingerprints))[1:min(200, length(fp_data.fingerprints))])
            RDKitBridge.compute_nearest_neighbors(
                fp_data.fingerprints[sample_idx], fp_data.ids[sample_idx]; k=3
            )
        end

        # Compute similarity matrix for small sets
        sim_matrix = if length(fp_data.fingerprints) <= 100
            RDKitBridge.compute_tanimoto_matrix(fp_data.fingerprints)
        else
            nothing  # Too large to send
        end

        return safe_json(Dict(
            "stats" => merge(stats, Dict(
                "n_unique_scaffolds" => scaffold_stats["n_unique_scaffolds"],
                "scaffold_entropy" => scaffold_stats["scaffold_entropy"],
            )),
            "nearest_neighbors" => nn_results,
            "similarity_matrix" => sim_matrix,
        ))
    catch e
        @error "Diversity computation failed" exception=e
        return safe_json(Dict("error" => "Diversity computation failed: $(sprint(showerror, e))"))
    end
end

@get "/api/v2/molecular/diversity/training" function(req)
    session = CURRENT_SESSION[]
    session === nothing && return safe_json(Dict("error" => "No active session"))
    adapter = session.adapter
    !(adapter isa MolecularAdapter) && return safe_json(Dict("error" => "Not a molecular session"))

    # Compute diversity over the most recent molecules
    n = length(adapter.generated_molecules)
    if n < 2
        return safe_json(Dict("diversity_over_time" => [], "current" => Dict()))
    end

    # Get fingerprints from generated molecules
    fps = Vector{Float32}[]
    for mol in adapter.generated_molecules
        fp = get(mol, "fingerprint", nothing)
        if fp !== nothing
            push!(fps, Float32.(fp))
        end
    end

    if length(fps) < 2
        return safe_json(Dict("diversity_over_time" => [], "current" => Dict()))
    end

    # Current diversity
    current_stats = RDKitBridge.compute_diversity_stats(fps)

    return safe_json(Dict(
        "current" => current_stats,
        "n_molecules" => length(fps),
    ))
end

# ============================================
# Gap 3: Fragment Library Endpoints
# ============================================

@get "/api/v2/molecular/fragments" function(req)
    # Return info about available fragment libraries
    libraries_dir = joinpath(@__DIR__, "..", "..", "..", "..", "data", "fragment_libraries")
    available = Dict[]

    if isdir(libraries_dir)
        for f in readdir(libraries_dir)
            endswith(f, ".json") || continue
            path = joinpath(libraries_dir, f)
            try
                data = JSON3.read(read(path, String), Dict)
                push!(available, Dict(
                    "filename" => f,
                    "n_fragments" => get(data, "n_fragments", length(get(data, "fragments", []))),
                    "source" => get(data, "source", "unknown"),
                    "version" => get(data, "version", "1.0"),
                ))
            catch
                push!(available, Dict("filename" => f, "error" => "parse_failed"))
            end
        end
    end

    # Current library info
    session = CURRENT_SESSION[]
    current_count = if session !== nothing && session.adapter isa MolecularAdapter
        length(session.adapter.fragment_library)
    else
        length(FRAGMENT_LIBRARY)
    end

    return safe_json(Dict(
        "current_count" => current_count,
        "available_libraries" => available,
    ))
end

@get "/api/v2/molecular/fragments/current" function(req)
    # Return the current fragment library details
    session = CURRENT_SESSION[]
    fragments = if session !== nothing && session.adapter isa MolecularAdapter
        session.adapter.fragment_library
    else
        FRAGMENT_LIBRARY
    end

    frag_list = [Dict(
        "id" => f.fragment_id,
        "smiles" => f.fragment_smiles,
        "name" => f.fragment_name,
        "category" => f.metadata.category,
        "is_starter" => f.metadata.is_starter,
        "n_attachments" => f.metadata.n_attachments,
        "brics_labels" => f.metadata.brics_labels,
    ) for f in fragments]

    return safe_json(Dict(
        "n_fragments" => length(fragments),
        "fragments" => frag_list,
    ))
end

# ============================================
# Gap 5: MOGFN Pareto Optimization Endpoints
# ============================================

@get "/api/v2/molecular/pareto-front" function(req)
    session = CURRENT_SESSION[]

    # Return empty Pareto front if no session or not molecular
    if session === nothing || !(session.adapter isa MolecularAdapter)
        return safe_json(Dict(
            "points" => [],
            "hypervolume" => 0.0,
            "objective_names" => ["qed", "sa_norm", "logp_score", "mw_score"],
            "objective_ranges" => Dict(),
        ))
    end

    # Get all molecules from the current session
    molecules = session.adapter.generated_molecules

    if isempty(molecules)
        return safe_json(Dict(
            "points" => [],
            "hypervolume" => 0.0,
            "objective_names" => ["qed", "sa_norm", "logp_score", "mw_score"],
            "objective_ranges" => Dict(),
        ))
    end

    # Compute objectives for all molecules
    objective_names = ["qed", "sa_norm", "logp_score", "mw_score"]
    points = []

    for mol in molecules
        smiles = mol isa Dict ? get(mol, "smiles", "") : ""
        if isempty(smiles)
            continue
        end

        # Create a terminal MolState to compute objectives
        try
            fp = RDKitBridge.compute_fingerprint(smiles)
            state = MolState(smiles, Int[], 0, true, fp)
            oracle_mgr = _get_oracle_manager()
            objs = compute_all_objectives(state; oracle_mgr=oracle_mgr)

            if !isempty(objs)
                push!(points, Dict(
                    "id" => get(mol, "id", string(UUIDs.uuid4())),
                    "smiles" => smiles,
                    "objectives" => Dict(zip(objective_names, objs)),
                    "reward" => GFlowNet.reward(state),
                    "is_pareto_optimal" => false,  # Computed below
                ))
            end
        catch
            continue
        end
    end

    # Compute Pareto optimality
    if !isempty(points)
        for i in 1:length(points)
            is_dominated = false
            objs_i = [points[i]["objectives"][n] for n in objective_names]
            for j in 1:length(points)
                i == j && continue
                objs_j = [points[j]["objectives"][n] for n in objective_names]
                # j dominates i if all objectives of j >= i and at least one strictly >
                if all(objs_j .>= objs_i) && any(objs_j .> objs_i)
                    is_dominated = true
                    break
                end
            end
            points[i]["is_pareto_optimal"] = !is_dominated
        end
    end

    # Compute objective ranges
    obj_ranges = Dict()
    for name in objective_names
        vals = [p["objectives"][name] for p in points if haskey(p["objectives"], name)]
        if !isempty(vals)
            obj_ranges[name] = [minimum(vals), maximum(vals)]
        end
    end

    # Compute hypervolume indicator (2D approximation using first two objectives)
    pareto_pts = filter(p -> p["is_pareto_optimal"], points)
    hypervolume = if length(pareto_pts) >= 2 && length(objective_names) >= 2
        # Simple 2D hypervolume with reference point [0, 0]
        sorted_pts = sort(pareto_pts, by=p -> p["objectives"][objective_names[1]])
        hv = 0.0
        for k in 1:length(sorted_pts)
            x = sorted_pts[k]["objectives"][objective_names[1]]
            y = sorted_pts[k]["objectives"][objective_names[2]]
            if k < length(sorted_pts)
                next_x = sorted_pts[k+1]["objectives"][objective_names[1]]
                hv += (next_x - x) * y
            else
                hv += (1.0 - x) * y  # reference point at x=1
            end
        end
        hv
    else
        0.0
    end

    return safe_json(Dict(
        "points" => points,
        "hypervolume" => hypervolume,
        "objective_names" => objective_names,
        "objective_ranges" => obj_ranges,
    ))
end

@get "/api/v2/molecular/molecules/{id}/objectives" function(req, id::String)
    # Compute objectives for a specific molecule
    session = CURRENT_SESSION[]
    if session === nothing
        return safe_json(Dict("error" => "No active training session"); status=400)
    end

    # Find the molecule by ID
    molecules = if session.adapter isa MolecularAdapter
        session.adapter.generated_molecules
    else
        return safe_json(Dict("error" => "Not a molecular session"); status=400)
    end

    mol = nothing
    for m in molecules
        if m isa Dict && get(m, "id", "") == id
            mol = m
            break
        end
    end

    if mol === nothing
        return safe_json(Dict("error" => "Molecule not found: $id"); status=404)
    end

    smiles = get(mol, "smiles", "")
    objective_names = ["qed", "sa_norm", "logp_score", "mw_score"]

    try
        fp = RDKitBridge.compute_fingerprint(smiles)
        state = MolState(smiles, Int[], 0, true, fp)
        oracle_mgr = _get_oracle_manager()
        objs = compute_all_objectives(state; oracle_mgr=oracle_mgr)

        return safe_json(Dict(
            "molecule_id" => id,
            "objectives" => Dict(zip(objective_names, objs)),
            "objective_names" => objective_names,
        ))
    catch e
        return safe_json(Dict("error" => "Failed to compute objectives: $e"); status=500)
    end
end

@post "/api/v2/molecular/generate-pareto" function(req)
    # Generate molecules with a specific preference weighting
    body = try
        JSON3.read(String(req.body))
    catch
        return safe_json(Dict("error" => "Invalid JSON body"); status=400)
    end

    preferences = get(body, :preferences, nothing)
    n_molecules = get(body, :n_molecules, 100)

    if preferences === nothing
        return safe_json(Dict("error" => "preferences field required"); status=400)
    end

    session = CURRENT_SESSION[]
    if session === nothing || session.model === nothing
        return safe_json(Dict("error" => "No active MOGFN training session"); status=400)
    end

    # Check if model has MOGFN components
    if isnothing(session.model.preference_encoder)
        return safe_json(Dict("error" => "Model is not MOGFN-PC. Train with MULTI_OBJECTIVE_TB objective."); status=400)
    end

    # Convert preferences dict to vector
    objective_names = ["qed", "sa_norm", "logp_score", "mw_score"]
    w = Float64[get(preferences, Symbol(name), 0.25) for name in objective_names]
    w_sum = sum(w)
    if w_sum > 0
        w = w ./ w_sum  # Normalize to simplex
    else
        w = fill(1.0 / length(objective_names), length(objective_names))
    end

    # Generate molecules using MOGFN sampling
    molecules = []
    sampling_config = GFlowNet.SamplingConfig(
        strategy = GFlowNet.STOCHASTIC_SAMPLING,
        temperature = 1.0,
        epsilon = 0.0,
        max_trajectory_length = 100
    )

    for _ in 1:n_molecules
        try
            traj = GFlowNet.sample_mogfn_trajectory(session.model, w; config=sampling_config)
            terminal = traj.states[end]
            if GFlowNet.is_terminal_state(terminal) && !isempty(terminal.smiles)
                fp = terminal.fingerprint
                state = MolState(terminal.smiles, Int[], 0, true, fp)
                oracle_mgr = _get_oracle_manager()
                objs = compute_all_objectives(state; oracle_mgr=oracle_mgr)
                if !isempty(objs)
                    push!(molecules, Dict(
                        "id" => string(UUIDs.uuid4()),
                        "smiles" => terminal.smiles,
                        "objectives" => Dict(zip(objective_names, objs)),
                        "reward" => GFlowNet.reward(state, w),
                        "is_pareto_optimal" => false,
                    ))
                end
            end
        catch
            continue
        end
    end

    return safe_json(Dict(
        "molecules" => molecules,
        "pareto_front" => filter(m -> m["is_pareto_optimal"], molecules),
        "hypervolume" => 0.0,  # TODO: compute when pymoo available
        "preferences_used" => Dict(zip(objective_names, w)),
    ))
end

# ============================================
# Gap 2: Docking API Endpoints
# ============================================

@get "/api/v2/molecular/targets" function(req)
    targets = RDKitBridge.get_docking_targets()
    target_list = [Dict(
        "id" => t["id"],
        "name" => t["name"],
        "pdb_id" => t["pdb_id"],
        "description" => get(t, "description", ""),
        "has_receptor" => isfile(RDKitBridge._get_receptor_path(t["id"])),
    ) for t in values(targets)]

    return safe_json(Dict(
        "targets" => target_list,
        "active_target" => RDKitBridge.get_docking_target(),
        "docking_available" => RDKitBridge.is_docking_available(),
        "proxy_available" => RDKitBridge.is_proxy_available(),
    ))
end

@post "/api/v2/molecular/dock" function(req)
    body = JSON3.read(String(req.body), Dict)

    smiles = get(body, "smiles", "")
    isempty(smiles) && return safe_json(Dict("error" => "Missing 'smiles' field"))

    target = get(body, "target", RDKitBridge.get_docking_target())
    method = get(body, "method", "proxy")

    if method == "proxy"
        if !RDKitBridge.is_proxy_available()
            return safe_json(Dict("error" => "Proxy model not trained. Use method='vina' or train proxy first."))
        end
        score = RDKitBridge.proxy_dock(smiles, target)
        return safe_json(Dict(
            "smiles" => smiles,
            "method" => "proxy",
            "normalized_score" => score,
            "target" => target,
        ))
    elseif method == "vina"
        if !RDKitBridge.is_docking_available()
            return safe_json(Dict("error" => "AutoDock Vina not available. Install meeko and vina."))
        end
        result = RDKitBridge.dock_molecule(smiles, target)
        if !isempty(result.error)
            return safe_json(Dict("error" => result.error))
        end
        return safe_json(Dict(
            "smiles" => result.smiles,
            "method" => "vina",
            "affinity_kcal" => result.affinity_kcal,
            "normalized_score" => RDKitBridge.sigmoid_normalize(result.affinity_kcal),
            "n_poses" => result.n_poses,
            "runtime_ms" => result.runtime_ms,
            "target" => target,
        ))
    else
        return safe_json(Dict("error" => "Unknown method: $method. Use 'proxy' or 'vina'."))
    end
end

@post "/api/v2/molecular/dock-batch" function(req)
    body = JSON3.read(String(req.body), Dict)

    ids = get(body, "ids", String[])
    target = get(body, "target", RDKitBridge.get_docking_target())

    session = CURRENT_SESSION[]
    session === nothing && return safe_json(Dict("error" => "No active training session"))

    results = Dict[]
    for id in ids
        # Look up molecule SMILES from the session's molecule store
        mol = get(session.molecules, string(id), nothing)
        if mol !== nothing && haskey(mol, "smiles")
            smiles = mol["smiles"]
            if RDKitBridge.is_proxy_available()
                score = RDKitBridge.proxy_dock(smiles, target)
                push!(results, Dict("id" => id, "smiles" => smiles, "normalized_score" => score, "method" => "proxy"))
            elseif RDKitBridge.is_docking_available()
                dock_result = RDKitBridge.dock_molecule(smiles, target)
                push!(results, Dict(
                    "id" => id, "smiles" => smiles,
                    "affinity_kcal" => dock_result.affinity_kcal,
                    "normalized_score" => RDKitBridge.sigmoid_normalize(dock_result.affinity_kcal),
                    "method" => "vina",
                ))
            end
        end
    end

    return safe_json(Dict(
        "results" => results,
        "target" => target,
        "n_docked" => length(results),
    ))
end

@post "/api/v2/molecular/docking/set-target" function(req)
    body = JSON3.read(String(req.body), Dict)
    target_id = get(body, "target_id", "")
    isempty(target_id) && return safe_json(Dict("error" => "Missing 'target_id'"))

    success = RDKitBridge.set_docking_target!(target_id)
    return safe_json(Dict(
        "success" => success,
        "active_target" => RDKitBridge.get_docking_target(),
    ))
end

# ============================================
# Gap 4: Reaction Domain / Synthesis Route Endpoints
# ============================================

@get "/api/v2/molecular/reactions" function(req)
    templates = RDKitBridge.get_reaction_templates()
    return safe_json(Dict(
        "reactions" => [Dict(
            "id"             => t["id"],
            "name"           => t["name"],
            "class"          => t["class"],
            "yield_estimate" => t["yield_estimate"],
            "functional_groups" => get(t, "functional_groups", String[]),
        ) for t in templates],
        "n_reactions"        => length(templates),
        "engine_available"   => RDKitBridge._reaction_engine_available[],
    ))
end

@get "/api/v2/molecular/molecules/{id}/synthesis" function(req, id::String)
    session = CURRENT_SESSION[]
    session === nothing && return safe_json(Dict("error" => "No session"), status=404)
    adapter = session.adapter

    # Support both fragment and reaction adapters
    molecules = if adapter isa ReactionMolecularAdapter
        adapter.generated_molecules
    elseif adapter isa MolecularAdapter
        adapter.generated_molecules
    else
        return safe_json(Dict("error" => "Not a molecular session"), status=400)
    end

    idx = findfirst(m -> m["id"] == id, molecules)
    idx === nothing && return safe_json(Dict("error" => "Molecule not found: $id"), status=404)

    mol = molecules[idx]
    route = get(mol, "synthesis_route", nothing)

    if route === nothing || isempty(route)
        return safe_json(Dict(
            "molecule_id"    => id,
            "smiles"         => mol["smiles"],
            "has_synthesis"  => false,
            "steps"          => [],
            "message"        => "No synthesis route available (fragment-based generation)",
        ))
    end

    # Compute cumulative yield
    cumulative_yield = 1.0
    for step in route
        cumulative_yield *= get(step, "yield_estimate", 0.8)
    end

    return safe_json(Dict(
        "molecule_id"       => id,
        "smiles"            => mol["smiles"],
        "has_synthesis"     => true,
        "n_steps"           => length(route),
        "steps"             => route,
        "cumulative_yield"  => cumulative_yield,
        "method"            => get(mol, "method", "reaction_gflownet"),
    ))
end

@post "/api/v2/molecular/reaction/execute" function(req)
    body = JSON3.read(String(req.body), Dict)
    reaction_id = get(body, "reaction_id", 0)
    reactant1 = get(body, "reactant1", "")
    reactant2 = get(body, "reactant2", "")

    if reaction_id <= 0 || isempty(reactant1)
        return safe_json(Dict("error" => "Missing reaction_id or reactant1"))
    end

    templates = RDKitBridge.get_reaction_templates()
    template = nothing
    for t in templates
        if t["id"] == reaction_id
            template = t
            break
        end
    end
    template === nothing && return safe_json(Dict("error" => "Unknown reaction_id: $reaction_id"))

    reactants = String[reactant1]
    !isempty(reactant2) && push!(reactants, reactant2)

    result = RDKitBridge.execute_reaction(template["smarts"], reactants)
    return safe_json(Dict(
        "valid"           => result.valid,
        "product_smiles"  => result.product_smiles,
        "reaction_name"   => template["name"],
        "reaction_class"  => template["class"],
    ))
end

@post "/api/v2/molecular/reaction/check-compatibility" function(req)
    body = JSON3.read(String(req.body), Dict)
    smiles = get(body, "smiles", "")
    reaction_id = get(body, "reaction_id", 0)

    isempty(smiles) && return safe_json(Dict("error" => "Missing smiles"))

    compatible = RDKitBridge.check_reactant(smiles, reaction_id)
    return safe_json(Dict(
        "smiles"      => smiles,
        "reaction_id" => reaction_id,
        "compatible"  => compatible,
    ))
end

# ============================================
# Oracle API Endpoints (PMO Integration)
# ============================================

@get "/api/v2/oracles/available" function(req)
    available = OracleBridge.get_all_available_oracles()
    loaded = OracleBridge.is_initialized() ? OracleBridge.get_loaded_oracles() : String[]
    return safe_json(Dict(
        "oracles" => available["all"],
        "bioactivity" => available["bioactivity"],
        "pmo_tasks" => available["pmo_tasks"],
        "loaded" => loaded,
    ))
end

@get "/api/v2/oracles/status" function(req)
    oracle_mgr = _get_oracle_manager()
    if oracle_mgr === nothing
        return safe_json(Dict(
            "configured" => String[],
            "budget_used" => 0,
            "budget_total" => 0,
            "cache_size" => 0,
            "benchmark_mode" => false,
            "active" => false,
        ))
    end
    status = get_status(oracle_mgr)
    status["active"] = true
    return safe_json(status)
end

@post "/api/v2/oracles/evaluate" function(req)
    body = JSON3.read(String(req.body), Dict)
    smiles = get(body, "smiles", "")
    isempty(smiles) && return safe_json(Dict("error" => "Missing 'smiles' field"))

    oracle_names = get(body, "oracles", nothing)

    # Use session oracles if none specified
    oracle_mgr = _get_oracle_manager()
    if oracle_names === nothing && oracle_mgr !== nothing
        oracle_names = get_objective_names(oracle_mgr)
    end

    if oracle_names === nothing || isempty(oracle_names)
        return safe_json(Dict("error" => "No oracles specified and no oracle session active"))
    end

    # Initialize oracles if needed
    if !OracleBridge.is_initialized()
        try
            OracleBridge.init_oracles!(String.(oracle_names);
                cache_dir=joinpath(@__DIR__, "..", "..", "..", "..", "data", "tdc_cache"))
        catch e
            return safe_json(Dict("error" => "Failed to initialize oracles: $(sprint(showerror, e))"))
        end
    end

    # Check cache first if oracle_mgr exists
    cached = false
    scores = Dict{String,Float64}()

    if oracle_mgr !== nothing
        all_cached = true
        for name in oracle_names
            s = lookup_score(oracle_mgr, smiles, String(name))
            if s == 0.5  # neutral = not cached
                all_cached = false
                break
            end
            scores[String(name)] = s
        end
        cached = all_cached
    end

    if !cached
        for name in oracle_names
            try
                score = OracleBridge.evaluate(smiles, String(name))
                scores[String(name)] = score
                # Update cache if manager exists
                if oracle_mgr !== nothing
                    if !haskey(oracle_mgr.cache, smiles)
                        oracle_mgr.cache[smiles] = Dict{String,Float64}()
                    end
                    oracle_mgr.cache[smiles][String(name)] = score
                end
            catch e
                scores[String(name)] = -1.0  # error sentinel
                @warn "Oracle evaluation failed" oracle=name smiles=smiles exception=e
            end
        end
    end

    return safe_json(Dict(
        "smiles" => smiles,
        "scores" => scores,
        "cached" => cached,
    ))
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
    @info "Frontend: http://localhost:5173 (Vite dev server)"
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
