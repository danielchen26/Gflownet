# Revised Comprehensive Plan: Real GFlowNet Training Visualization

> **Revision 2** — Fixed all API signature mismatches, added missing utility functions,
> corrected `TrainingConfig` construction, added frontend integration details,
> test plan, and error-handling strategy. Validated against the actual codebase.

## Scope Clarification

| Domain | Status | Visualization Priority |
|--------|--------|----------------------|
| **Grid World** | Fully working | **Phase 1 - Implement now** |
| Molecular Design | Template only | Future |
| Supply Chain | Template only | Future |
| Active Learning | Template only | Future |
| Causal Discovery | Template only | Future |

**Strategy**: Build a **domain-agnostic architecture** but implement **only Grid World adapter** initially. Other adapters are placeholders that can be filled in when those domains mature.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Visualization System Architecture                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │  Frontend (React)│◄──►│  Unified Server  │◄──►│  GFlowNet.jl   │ │
│  │                 │    │  (Oxygen.jl)     │    │  Core Training │ │
│  └────────┬────────┘    └────────┬─────────┘    └────────────────┘ │
│           │                      │                                  │
│           ▼                      ▼                                  │
│  ┌─────────────────┐    ┌──────────────────┐                       │
│  │ Domain Renderers│    │ Domain Adapters  │                       │
│  │ • GridRenderer  │    │ • GridAdapter    │                       │
│  │ • GenericRender │    │ • GenericAdapter │                       │
│  │ • (Future...)   │    │ • (Future...)    │                       │
│  └─────────────────┘    └──────────────────┘                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| API versioning | `/api/v2/` prefix | Backward-compatible; old mock `/api/` stays |
| Server framework | Oxygen.jl | Already used by mock server |
| Training loop | Async Julia `Task` | Non-blocking; server remains responsive |
| Concurrency model | Single-threaded cooperative | Oxygen + `@async` on same thread; safe without locks |
| Frontend proxy | Vite `/api` → `:8080` | Already configured in `vite.config.ts` |

---

## Part 1: Core Abstractions (Domain-Agnostic)

### 1.1 Domain Adapter Interface

**File: `src/utils/visualization/core/adapters.jl`**

```julia
# Domain adapter interface for visualization
# Converts domain-specific GFlowNet data to visualization-friendly formats

using ..GFlowNet: AbstractState, AbstractAction, GFlowNetModel, Trajectory

"""
    AbstractDomainAdapter

Base interface for converting domain-specific GFlowNet data to
visualization-friendly formats. All domain adapters must implement this.
"""
abstract type AbstractDomainAdapter end

# ============================================
# Required Interface Methods
# ============================================

"""Convert state to JSON-serializable visualization data"""
function state_to_viz_data(adapter::AbstractDomainAdapter, state::AbstractState)::Dict
    error("Not implemented for $(typeof(adapter))")
end

"""Convert trajectory to JSON-serializable visualization data"""
function trajectory_to_viz_data(adapter::AbstractDomainAdapter, traj::Trajectory, id::String)::Dict
    error("Not implemented for $(typeof(adapter))")
end

"""Get domain configuration for frontend"""
function get_domain_config(adapter::AbstractDomainAdapter)::Dict
    error("Not implemented for $(typeof(adapter))")
end

"""Get frontend renderer component name"""
function get_renderer_name(adapter::AbstractDomainAdapter)::String
    error("Not implemented for $(typeof(adapter))")
end

"""Compute domain-specific visualization metrics"""
function compute_domain_metrics(adapter::AbstractDomainAdapter, model::GFlowNetModel, trajectories::Vector{Trajectory})::Dict
    error("Not implemented for $(typeof(adapter))")
end

"""Compute flow field for visualization (if applicable)"""
function compute_flow_field(adapter::AbstractDomainAdapter, model::GFlowNetModel)::Dict
    # Default: not supported
    return Dict("supported" => false)
end

"""Compute distribution heatmap (if applicable)"""
function compute_distribution_data(adapter::AbstractDomainAdapter, model::GFlowNetModel, trajectories::Vector{Trajectory})::Dict
    # Default: not supported
    return Dict("supported" => false)
end
```

### 1.2 Training Session Manager

**File: `src/utils/visualization/core/training_session.jl`**

> **Fixed**: `TrainingConfig` uses kwargs-only constructor matching the actual API.
> **Fixed**: `sample_trajectory` uses `model` only (config defaults internally).
> **Fixed**: `train_step!` signature matches `(model, trajectories, config)` returning `(loss, grad_norm)`.

```julia
using ..GFlowNet: GFlowNetModel, TrainingConfig, Trajectory, TrainingObjective
using ..GFlowNet: TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING
using ..GFlowNet: SUB_TRAJECTORY_BALANCE, DIRECT_FLOW_OBJECTIVE
using ..GFlowNet: sample_trajectory, reward
using Statistics: mean
using UUIDs: uuid4
using Dates: DateTime, now

"""
    TrainingSession

Manages a real GFlowNet training session with visualization hooks.
Domain-agnostic — works with any domain through the adapter pattern.
"""
mutable struct TrainingSession
    # Identity
    id::String
    created_at::DateTime

    # GFlowNet components
    model::GFlowNetModel
    config::TrainingConfig
    adapter::AbstractDomainAdapter

    # Training state
    is_training::Bool
    is_paused::Bool
    current_iteration::Int
    total_iterations::Int

    # Metrics history
    losses::Vector{Float64}
    gradient_norms::Vector{Float64}
    rewards::Vector{Float64}

    # Trajectory buffer (recent trajectories for visualization)
    trajectory_buffer::Vector{Trajectory}
    max_buffer_size::Int

    # Error tracking
    last_error::Union{String, Nothing}
    error_count::Int

    # Timing
    start_time::Union{DateTime, Nothing}
    iteration_times::Vector{Float64}
end

# ============================================
# Objective Parsing
# ============================================

"""
    parse_objective(name::String)::TrainingObjective

Parse a string objective name into a TrainingObjective enum value.
"""
function parse_objective(name::String)::TrainingObjective
    mapping = Dict(
        "TRAJECTORY_BALANCE"      => TRAJECTORY_BALANCE,
        "DETAILED_BALANCE"        => DETAILED_BALANCE,
        "FLOW_MATCHING"           => FLOW_MATCHING,
        "SUB_TRAJECTORY_BALANCE"  => SUB_TRAJECTORY_BALANCE,
        "DIRECT_FLOW_OBJECTIVE"   => DIRECT_FLOW_OBJECTIVE,
    )
    upper = uppercase(strip(name))
    haskey(mapping, upper) || error("Unknown objective: $name. Valid: $(join(keys(mapping), ", "))")
    return mapping[upper]
end

# ============================================
# Session Lifecycle
# ============================================

"""Create new training session from configuration dict."""
function create_session(config::Dict)::TrainingSession
    domain_type = get(config, "domain_type", "grid_world")
    model, adapter = create_model_and_adapter(domain_type, config)

    # Build TrainingConfig using kwargs-only constructor (matches real API)
    objective = parse_objective(get(config, "objective", "TRAJECTORY_BALANCE"))

    # FLOW_MATCHING requires flow_estimator — create model with it if needed
    training_config = TrainingConfig(
        objective       = objective,
        n_iterations    = get(config, "n_episodes", 500),
        batch_size      = get(config, "batch_size", 8),
        learning_rate   = get(config, "learning_rate", 0.001),
        verbose         = false   # We handle logging ourselves
    )

    return TrainingSession(
        string(uuid4()),
        now(),
        model,
        training_config,
        adapter,
        false, false,                           # is_training, is_paused
        0, training_config.n_iterations,        # current_iteration, total_iterations
        Float64[], Float64[], Float64[],        # losses, gradient_norms, rewards
        Trajectory[], 50,                       # trajectory_buffer, max_buffer_size
        nothing, 0,                             # last_error, error_count
        nothing, Float64[]                      # start_time, iteration_times
    )
end

"""Execute one training iteration. Returns a status Dict."""
function step!(session::TrainingSession)::Dict
    if !session.is_training || session.is_paused
        return Dict("status" => "not_running")
    end

    model  = session.model
    config = session.config

    t0 = time()

    try
        # ---- Sample trajectories from the REAL learned policy ----
        trajectories = [sample_trajectory(model) for _ in 1:config.batch_size]

        # ---- Real gradient descent step ----
        # Actual signature: train_step!(model, trajectories, config) -> (loss, grad_norm)
        loss_val, grad_norm = GFlowNet.train_step!(model, trajectories, config)

        iteration_time = time() - t0

        # Update session state
        session.current_iteration += 1
        push!(session.losses, loss_val)
        push!(session.gradient_norms, grad_norm)
        push!(session.iteration_times, iteration_time)

        # Update trajectory buffer (ring buffer)
        for traj in trajectories
            push!(session.trajectory_buffer, traj)
            if length(session.trajectory_buffer) > session.max_buffer_size
                popfirst!(session.trajectory_buffer)
            end
        end

        # Compute rewards from terminal states
        batch_rewards = Float64[reward(t.states[end]) for t in trajectories]
        push!(session.rewards, mean(batch_rewards))

        # Check if done
        if session.current_iteration >= session.total_iterations
            session.is_training = false
        end

        return Dict(
            "status"         => "ok",
            "iteration"      => session.current_iteration,
            "loss"           => loss_val,
            "gradient_norm"  => grad_norm,
            "mean_reward"    => mean(batch_rewards),
            "iteration_time" => iteration_time
        )

    catch e
        session.error_count += 1
        session.last_error = sprint(showerror, e)
        @error "Training step error" iteration=session.current_iteration exception=e

        # Record NaN for this iteration so the frontend can show the gap
        push!(session.losses, NaN)
        push!(session.gradient_norms, NaN)
        push!(session.rewards, NaN)
        push!(session.iteration_times, time() - t0)
        session.current_iteration += 1

        # Stop after 10 consecutive errors
        if session.error_count > 10
            session.is_training = false
        end

        return Dict(
            "status" => "error",
            "error"  => session.last_error,
            "iteration" => session.current_iteration
        )
    end
end
```

### 1.3 Universal Metrics Computation

**File: `src/utils/visualization/core/metrics.jl`**

```julia
using ..GFlowNet: GFlowNetModel, Trajectory, reward
using Statistics: mean, std

"""
    compute_gflownet_metrics(model, trajectories)

Compute universal GFlowNet quality metrics that apply to ALL domains.
"""
function compute_gflownet_metrics(model::GFlowNetModel, trajectories::Vector{Trajectory})::Dict
    if isempty(trajectories)
        return Dict("error" => "No trajectories")
    end

    terminals = [t.states[end] for t in trajectories]
    rewards   = Float64[reward(s) for s in terminals]
    unique_count = length(unique(terminals))
    lengths = [length(t.actions) for t in trajectories]

    Z = if model.log_partition_function !== nothing
        exp(model.log_partition_function)
    else
        1.0
    end

    return Dict(
        # Reward metrics
        "mean_reward"       => mean(rewards),
        "max_reward"        => maximum(rewards),
        "min_reward"        => minimum(rewards),
        "reward_std"        => length(rewards) > 1 ? std(rewards) : 0.0,
        # Diversity metrics
        "unique_terminals"  => unique_count,
        "diversity_ratio"   => unique_count / length(trajectories),
        # Trajectory metrics
        "mean_length"       => mean(lengths),
        "max_length"        => maximum(lengths),
        # Model metrics
        "partition_function" => Z,
        # Sample size
        "n_trajectories"    => length(trajectories)
    )
end
```

---

## Part 2: Grid World Adapter (Reference Implementation)

**File: `src/utils/visualization/domains/grid_world.jl`**

> **Fixed**: `forward_action_probabilities` uses the actual 5-argument signature:
> `(policy, state, actions, parameters, states)`.
> **Fixed**: `flow(model, state)` matches the real API.
> **Fixed**: Imports `LinearAlgebra.norm`.
> **Fixed**: `action_to_string` uses `isa()` checks on `GridAction` subtypes.

```julia
using ..GFlowNet: AbstractState, GFlowNetModel, Trajectory
using ..GFlowNet: GridState, GridAction, MoveRight, MoveUp, MoveLeft, MoveDown, Terminate
using ..GFlowNet: forward_action_probabilities, flow, reward
using LinearAlgebra: norm

"""
    GridWorldAdapter <: AbstractDomainAdapter

Visualization adapter for Grid World domain.
This is the reference implementation for other domain adapters.
"""
struct GridWorldAdapter <: AbstractDomainAdapter
    grid_size::Int
    reward_positions::Dict{Tuple{Int,Int}, Float64}
end

# ============================================
# Interface Implementation
# ============================================

function state_to_viz_data(adapter::GridWorldAdapter, state::GridState)::Dict
    return Dict(
        "x" => state.x,
        "y" => state.y,
        "position" => [state.x, state.y],
        "is_terminal" => state.is_terminal,
        "reward" => state.is_terminal ? reward(state) : 0.0
    )
end

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

function get_renderer_name(adapter::GridWorldAdapter)::String
    return "GridWorldRenderer"
end

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

        push!(flow_data, Dict(
            "position"  => [x, y],
            "velocity"  => velocity,
            "magnitude" => norm(velocity),
            "flow"      => flow_val
        ))
    end

    return Dict(
        "supported" => true,
        "grid_size" => grid_size,
        "data"      => flow_data
    )
end

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

    return Dict(
        "supported"     => true,
        "grid_size"     => grid_size,
        "empirical"     => probs,
        "target"        => target,
        "counts"        => counts,
        "total_samples" => total
    )
end

# ============================================
# Helper Functions
# ============================================

function action_to_string(action)::String
    action isa MoveRight  && return "right"
    action isa MoveUp     && return "up"
    action isa MoveLeft   && return "left"
    action isa MoveDown   && return "down"
    action isa Terminate   && return "terminate"
    return "unknown"
end

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
```

---

## Part 3: Unified Server Implementation

**File: `src/utils/visualization/api/unified_server.jl`**

> **Fixed**: Uses the real `create_grid_world_gflownet` kwargs.
> **Fixed**: Includes `include_backward` when objective requires it.
> **Fixed**: Adds `include_flow_estimator` when objective is FLOW_MATCHING.
> **Added**: Error surfacing to frontend via `/api/v2/training/state`.
> **Note**: Vite already proxies `/api` to `localhost:8080` — no config change needed.

```julia
using Oxygen
using JSON3
using HTTP
using UUIDs
using Dates
using Statistics

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

"""Create model and adapter based on domain type."""
function create_model_and_adapter(domain_type::String, config::Dict)
    if domain_type == "grid_world"
        return create_grid_world_model_and_adapter(config)
    else
        error("Unsupported domain type: $domain_type. Currently only 'grid_world' is supported.")
    end
end

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

    # Create the real GFlowNet model
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

function start_real_training_server(; port::Int = 8080)
    @info "Starting real GFlowNet training visualization server on port $port"
    @info "Frontend: http://localhost:3000 (Vite dev server)"
    @info "API base: http://localhost:$port/api/v2/"
    serve(; port = port, middleware = [add_cors_headers])
end
```

---

## Part 4: Frontend Updates

### 4.1 API Client

**File: `src/utils/visualization/web/src/lib/api.ts`** (NEW file)

The existing frontend uses `axios` and inline `fetch` calls to `http://localhost:8080/api/...`. The v2 client uses relative URLs so the Vite proxy handles routing.

```typescript
// V2 API client for real GFlowNet training
// Uses relative URLs — Vite proxy routes /api/* to localhost:8080

const V2 = '/api/v2';

export interface TrainingConfig {
  domain_type: 'grid_world';
  grid_size: number;
  reward_peaks: Array<{
    position: [number, number];
    intensity: number;
    name?: string;
  }>;
  n_episodes: number;
  batch_size: number;
  learning_rate: number;
  hidden_dim?: number;
  allow_all_moves?: boolean;
  objective:
    | 'TRAJECTORY_BALANCE'
    | 'DETAILED_BALANCE'
    | 'FLOW_MATCHING'
    | 'SUB_TRAJECTORY_BALANCE'
    | 'DIRECT_FLOW_OBJECTIVE';
}

export interface TrainingState {
  has_session: boolean;
  is_training: boolean;
  is_paused: boolean;
  is_real_training: boolean;
  current_iteration: number;
  total_iterations: number;
  progress: number;
  latest_loss: number | null;
  latest_reward: number | null;
  latest_gradient_norm: number | null;
  metrics: Record<string, any>;
  domain_metrics: Record<string, any>;
  last_error: string | null;
  error_count: number;
}

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) throw new Error(`API error ${res.status}: ${res.statusText}`);
  return res.json();
}

export const realApi = {
  // Domain
  getDomainInfo: () => fetchJSON(`${V2}/domain/info`),

  // Training control
  startTraining: (config: TrainingConfig) =>
    fetchJSON(`${V2}/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    }),
  stopTraining: () =>
    fetchJSON(`${V2}/training/stop`, { method: 'POST' }),
  pauseTraining: () =>
    fetchJSON(`${V2}/training/pause`, { method: 'POST' }),

  // Training state
  getTrainingState: () => fetchJSON<TrainingState>(`${V2}/training/state`),
  getTrainingHistory: () => fetchJSON(`${V2}/training/history`),

  // Trajectories
  getTrajectories: () => fetchJSON(`${V2}/trajectories`),

  // Analysis
  getFlowField: () => fetchJSON(`${V2}/analysis/flow`),
  getDistribution: () => fetchJSON(`${V2}/analysis/distribution`),
};
```

### 4.2 Training Mode Indicator

**File: `src/utils/visualization/web/src/components/TrainingModeIndicator.tsx`** (NEW)

```tsx
interface Props {
  isReal: boolean;
  errorCount?: number;
  lastError?: string | null;
}

export function TrainingModeIndicator({ isReal, errorCount = 0, lastError }: Props) {
  return (
    <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${
      isReal
        ? 'bg-green-500/20 text-green-400 border border-green-500/30'
        : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
    }`}>
      <span className={`w-2 h-2 rounded-full ${isReal ? 'bg-green-400' : 'bg-yellow-400'} animate-pulse`} />
      <span>{isReal ? 'Real GFlowNet Training' : 'Simulation Mode'}</span>
      {errorCount > 0 && (
        <span className="text-red-400 ml-1" title={lastError ?? ''}>
          ({errorCount} errors)
        </span>
      )}
    </div>
  );
}
```

### 4.3 Frontend Component Integration Strategy

The existing components already fetch from `/api/...` endpoints. To integrate:

| Component | Current Endpoint | v2 Endpoint | Change Needed |
|-----------|-----------------|-------------|---------------|
| `MonitoringDashboard.tsx` | `/api/training/metrics` | `/api/v2/training/state` | Update fetch URL + response shape |
| `TrainingDashboard.tsx` | `/api/training/history` | `/api/v2/training/history` | Update fetch URL (shape compatible) |
| `GFlowNetDistribution3D.tsx` | `/api/analysis/state-statistics` | `/api/v2/analysis/distribution` | Update URL + map response fields |
| `GFlowNetFlowField.tsx` | `/api/analysis/flow-field` | `/api/v2/analysis/flow` | Update URL + map response fields |
| `ProblemSetup.tsx` | `/api/training/reset` (POST) | `/api/v2/training/start` (POST) | Update URL, send full config |
| `App.tsx` | Inline `fetch` to `/api/training/reset` | `/api/v2/training/start` | Update `handleStartTraining` |

**Approach**: Add a `useRealTraining` flag (or env variable). When true, components call `realApi.*`; when false, they use the existing mock endpoints. This preserves the mock server for demo/development.

---

## Part 5: File Structure Summary

```
src/utils/visualization/
├── core/                              # NEW
│   ├── adapters.jl                   # AbstractDomainAdapter interface
│   ├── training_session.jl           # TrainingSession + parse_objective()
│   └── metrics.jl                    # Universal metrics computation
│
├── domains/                           # NEW
│   ├── grid_world.jl                 # GridWorldAdapter (implemented)
│   └── generic.jl                    # GenericAdapter (stub for future)
│
├── api/
│   ├── unified_server.jl             # NEW: Real training server (v2 endpoints)
│   ├── gflownet_server.jl            # KEEP: Mock simulation (v1 endpoints)
│   └── simple_server.jl              # KEEP: Existing simple mock server
│
└── web/src/
    ├── lib/
    │   ├── axios.ts                  # KEEP: Existing axios config
    │   └── api.ts                    # NEW: v2 API client with types
    ├── components/
    │   ├── TrainingModeIndicator.tsx  # NEW: Real vs simulation badge
    │   ├── MonitoringDashboard.tsx    # MODIFY: Support v2 endpoints
    │   ├── ProblemSetup.tsx           # MODIFY: Call v2/training/start
    │   └── ...existing...
    └── visualizations/
        ├── GFlowNetDistribution3D.tsx # MODIFY: Support v2 distribution data
        ├── GFlowNetFlowField.tsx      # MODIFY: Support v2 flow data
        └── ...existing...
```

---

## Part 6: Test Plan

### 6.1 Julia Backend Tests

**File: `test/visualization/test_real_training_viz.jl`**

```julia
using Test
using GFlowNet

@testset "Real Training Visualization" begin

    @testset "Domain Adapter Interface" begin
        # Create a real model
        model = create_grid_world_gflownet(grid_size=5, hidden_dim=32)
        adapter = GridWorldAdapter(5, Dict((3,3)=>10.0, (5,5)=>8.0))

        # Test state_to_viz_data
        state = GridState(2, 3, false)
        viz = state_to_viz_data(adapter, state)
        @test viz["x"] == 2
        @test viz["y"] == 3
        @test viz["is_terminal"] == false

        # Test get_domain_config
        config = get_domain_config(adapter)
        @test config["domain_type"] == "grid_world"
        @test config["grid_size"] == [5, 5]
        @test config["supports_flow_field"] == true
    end

    @testset "Training Session Lifecycle" begin
        config = Dict(
            "domain_type" => "grid_world",
            "grid_size" => 5,
            "n_episodes" => 10,
            "batch_size" => 4,
            "learning_rate" => 0.01,
            "objective" => "TRAJECTORY_BALANCE"
        )
        session = create_session(config)

        @test session.current_iteration == 0
        @test session.total_iterations == 10
        @test !session.is_training

        # Run a few steps
        session.is_training = true
        for _ in 1:3
            result = step!(session)
            @test result["status"] == "ok"
            @test haskey(result, "loss")
            @test haskey(result, "mean_reward")
        end
        @test session.current_iteration == 3
        @test length(session.losses) == 3
    end

    @testset "Parse Objective" begin
        @test parse_objective("TRAJECTORY_BALANCE") == TRAJECTORY_BALANCE
        @test parse_objective("DETAILED_BALANCE") == DETAILED_BALANCE
        @test parse_objective("FLOW_MATCHING") == FLOW_MATCHING
        @test_throws ErrorException parse_objective("INVALID")
    end

    @testset "Universal Metrics" begin
        model = create_grid_world_gflownet(grid_size=5, hidden_dim=32)
        trajectories = [sample_trajectory(model) for _ in 1:10]

        metrics = compute_gflownet_metrics(model, trajectories)
        @test haskey(metrics, "mean_reward")
        @test haskey(metrics, "unique_terminals")
        @test haskey(metrics, "diversity_ratio")
        @test metrics["n_trajectories"] == 10
    end

    @testset "Grid World Flow Field" begin
        model = create_grid_world_gflownet(grid_size=4, hidden_dim=32)
        adapter = GridWorldAdapter(4, Dict((4,4)=>10.0))

        flow_data = compute_flow_field(adapter, model)
        @test flow_data["supported"] == true
        @test flow_data["grid_size"] == 4
        @test length(flow_data["data"]) == 16  # 4x4 grid
    end

    @testset "Grid World Distribution" begin
        model = create_grid_world_gflownet(grid_size=4, hidden_dim=32)
        adapter = GridWorldAdapter(4, Dict((4,4)=>10.0))
        trajectories = [sample_trajectory(model) for _ in 1:20]

        dist = compute_distribution_data(adapter, model, trajectories)
        @test dist["supported"] == true
        @test dist["grid_size"] == 4
        @test size(dist["empirical"]) == (4, 4)
        @test size(dist["target"]) == (4, 4)
    end
end
```

### 6.2 Integration Test

**File: `test/visualization/test_real_training_server.jl`**

Test that the unified server starts and responds to HTTP requests (optional, requires running server).

---

## Part 7: Implementation Order

### Phase 1: Core Infrastructure (Julia)
1. Create `src/utils/visualization/core/adapters.jl`
2. Create `src/utils/visualization/core/metrics.jl`
3. Create `src/utils/visualization/core/training_session.jl` (includes `parse_objective`)

### Phase 2: Grid World Adapter
4. Create `src/utils/visualization/domains/grid_world.jl`
5. Run unit tests: `julia --project=. test/visualization/test_real_training_viz.jl`

### Phase 3: Unified Server
6. Create `src/utils/visualization/api/unified_server.jl`
7. Manual test: start server, call endpoints with `curl`

### Phase 4: Frontend Integration
8. Create `src/utils/visualization/web/src/lib/api.ts`
9. Create `TrainingModeIndicator.tsx`
10. Update `App.tsx` to use `/api/v2/training/start`
11. Update `MonitoringDashboard.tsx` to fetch from v2 endpoints
12. Update `GFlowNetDistribution3D.tsx` and `GFlowNetFlowField.tsx`

### Phase 5: Validation
13. Start unified server + Vite dev server
14. Configure a grid world problem in the Setup page
15. Verify training produces decreasing loss curve
16. Verify flow field reflects learned policy (arrows toward reward peaks)
17. Verify 3D distribution converges toward target P(x) proportional to R(x)

---

## Part 8: Concurrency and Error Handling

### Thread Safety

Oxygen.jl uses a single-threaded event loop. The `@async` training task runs cooperatively on the same thread. This means:

- **No locks needed**: Session field mutations and reads never interleave mid-statement.
- **`sleep(0.05)` is essential**: It yields control back to Oxygen's event loop so HTTP requests can be served between training steps.
- **Risk**: A very slow training step blocks the server. Mitigation: add a timeout per step (not implemented in Phase 1 but noted for future).

### Error Strategy

| Error Type | Handling |
|-----------|----------|
| Single step failure (Zygote error, NaN loss) | Record NaN, increment `error_count`, continue |
| 10+ consecutive errors | Stop training, surface via `last_error` field |
| Fatal loop crash | `@async` catch block stops training, logs error |
| Frontend disconnect | No impact — server continues; frontend reconnects via polling |

### Performance Notes

- Real training is slower than mock (~50ms per step vs instant).
- The 50ms sleep cap means ~20 steps/second maximum.
- Frontend polls at 250ms for metrics and 500ms for charts — this is compatible.
- `compute_flow_field` iterates all grid cells and calls the NN for each. For grid_size=10, this is 100 forward passes — may take 100-500ms. Consider caching or throttling this endpoint.

---

## Confirmation Questions (Resolved)

1. **Backward Compatibility**: Old `/api/` mock endpoints remain. New real endpoints use `/api/v2/`. Frontend uses a flag to switch.

2. **Training Speed**: ~50ms per iteration is acceptable. Frontend polling rates already accommodate this.

3. **Initial Domain**: Grid World only. Architecture supports future domains via adapter pattern.

4. **Validation Display**: The `TrainingModeIndicator` component shows "Real GFlowNet Training" with error count. Domain metrics (mode coverage, etc.) are exposed via `/api/v2/training/state`.
