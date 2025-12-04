# Revised Comprehensive Plan: Real GFlowNet Training Visualization

## Scope Clarification

| Domain | Status | Visualization Priority |
|--------|--------|----------------------|
| **Grid World** | ✅ Fully working | **Phase 1 - Implement now** |
| Molecular Design | ⚠️ Template only | Future |
| Supply Chain | ⚠️ Template only | Future |
| Active Learning | ⚠️ Template only | Future |
| Causal Discovery | ⚠️ Template only | Future |

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
│  │ • GridRenderer  │    │ • GridAdapter ✅  │                       │
│  │ • GenericRender │    │ • GenericAdapter │                       │
│  │ • (Future...)   │    │ • (Future...)    │                       │
│  └─────────────────┘    └──────────────────┘                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Core Abstractions (Domain-Agnostic)

### 1.1 Domain Adapter Interface

**File: `src/utils/visualization/core/adapters.jl`**

```julia
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

```julia
"""
    TrainingSession

Manages a real GFlowNet training session with visualization hooks.
Domain-agnostic - works with any domain through the adapter pattern.
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

    # Timing
    start_time::Union{DateTime, Nothing}
    iteration_times::Vector{Float64}
end

"""Create new training session from configuration"""
function create_session(config::Dict)::TrainingSession
    # Create model based on domain
    domain_type = get(config, "domain_type", "grid_world")
    model, adapter = create_model_and_adapter(domain_type, config)

    # Create training config
    training_config = TrainingConfig(
        objective = parse_objective(get(config, "objective", "TRAJECTORY_BALANCE")),
        n_iterations = get(config, "n_episodes", 500),
        batch_size = get(config, "batch_size", 8),
        learning_rate = get(config, "learning_rate", 0.001),
        verbose = false  # We handle logging ourselves
    )

    return TrainingSession(
        string(uuid4()),
        now(),
        model,
        training_config,
        adapter,
        false, false,
        0, training_config.n_iterations,
        Float64[], Float64[], Float64[],
        Trajectory[], 50,
        nothing, Float64[]
    )
end

"""Execute one training iteration"""
function step!(session::TrainingSession)::Dict
    if !session.is_training || session.is_paused
        return Dict("status" => "not_running")
    end

    model = session.model
    config = session.config

    start_time = time()

    # Sample trajectories from REAL learned policy
    trajectories = [sample_trajectory(model) for _ in 1:config.batch_size]

    # Real gradient descent step
    loss, grad_norm = train_step!(model, trajectories, config)

    iteration_time = time() - start_time

    # Update session state
    session.current_iteration += 1
    push!(session.losses, loss)
    push!(session.gradient_norms, grad_norm)
    push!(session.iteration_times, iteration_time)

    # Update trajectory buffer
    for traj in trajectories
        push!(session.trajectory_buffer, traj)
        if length(session.trajectory_buffer) > session.max_buffer_size
            popfirst!(session.trajectory_buffer)
        end
    end

    # Compute rewards
    rewards = [reward(t.states[end]) for t in trajectories]
    push!(session.rewards, mean(rewards))

    # Check if done
    if session.current_iteration >= session.total_iterations
        session.is_training = false
    end

    return Dict(
        "status" => "ok",
        "iteration" => session.current_iteration,
        "loss" => loss,
        "gradient_norm" => grad_norm,
        "mean_reward" => mean(rewards),
        "iteration_time" => iteration_time
    )
end
```

### 1.3 Universal Metrics Computation

**File: `src/utils/visualization/core/metrics.jl`**

```julia
"""
    compute_gflownet_metrics(model, trajectories)

Compute universal GFlowNet quality metrics that apply to ALL domains.
These validate that the core GFlowNet properties are satisfied.
"""
function compute_gflownet_metrics(model::GFlowNetModel, trajectories::Vector{Trajectory})::Dict
    if isempty(trajectories)
        return Dict("error" => "No trajectories")
    end

    # Terminal states
    terminals = [t.states[end] for t in trajectories]

    # Rewards
    rewards = [reward(s) for s in terminals]

    # Unique terminals (mode discovery)
    unique_count = length(unique(terminals))

    # Partition function
    Z = if model.log_partition_function !== nothing
        exp(model.log_partition_function)
    else
        1.0
    end

    # Trajectory lengths
    lengths = [length(t.actions) for t in trajectories]

    return Dict(
        # Reward metrics
        "mean_reward" => mean(rewards),
        "max_reward" => maximum(rewards),
        "min_reward" => minimum(rewards),
        "reward_std" => length(rewards) > 1 ? std(rewards) : 0.0,

        # Diversity metrics
        "unique_terminals" => unique_count,
        "diversity_ratio" => unique_count / length(trajectories),

        # Trajectory metrics
        "mean_length" => mean(lengths),
        "max_length" => maximum(lengths),

        # Model metrics
        "partition_function" => Z,

        # Sample size
        "n_trajectories" => length(trajectories)
    )
end
```

---

## Part 2: Grid World Adapter (Reference Implementation)

**File: `src/utils/visualization/domains/grid_world.jl`**

```julia
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
    states_data = [[s.x, s.y] for s in traj.states]

    actions_data = [action_to_string(a) for a in traj.actions]

    rewards_data = [reward(s) for s in traj.states]

    return Dict(
        "id" => id,
        "states" => states_data,
        "actions" => actions_data,
        "rewards" => rewards_data,
        "total_reward" => reward(traj.states[end]),
        "length" => length(traj.actions)
    )
end

function get_domain_config(adapter::GridWorldAdapter)::Dict
    peaks = [
        Dict(
            "position" => [x, y],
            "intensity" => r,
            "name" => "Peak at ($x,$y)"
        )
        for ((x, y), r) in adapter.reward_positions
    ]

    return Dict(
        "domain_type" => "grid_world",
        "grid_size" => [adapter.grid_size, adapter.grid_size],
        "reward_peaks" => peaks,
        "supports_flow_field" => true,
        "supports_distribution" => true,
        "coordinate_system" => "cartesian_2d"
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

    # Mode coverage: which reward peaks have been visited
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

    # Top positions
    sorted_positions = sort(collect(position_counts), by=x->x[2], rev=true)
    top_5 = first(sorted_positions, 5)

    return Dict(
        "mode_coverage" => mode_coverage,
        "modes_discovered" => length(peaks_visited),
        "total_modes" => length(adapter.reward_positions),
        "unique_positions" => length(position_counts),
        "top_positions" => [
            Dict("position" => [p[1], p[2]], "count" => c, "percentage" => c / length(terminals) * 100)
            for (p, c) in top_5
        ]
    )
end

function compute_flow_field(adapter::GridWorldAdapter, model::GFlowNetModel)::Dict
    grid_size = adapter.grid_size
    flow_data = []

    for x in 1:grid_size, y in 1:grid_size
        state = GridState(x, y, false)

        # Get policy probabilities
        probs = forward_action_probabilities(model, state)

        # Convert to velocity vector
        # Assuming action order: [Right, Up, ...] or similar
        velocity = compute_velocity_from_policy(probs, model.all_actions)

        # Get flow value if using flow estimator
        flow_val = try
            flow(model, state)
        catch
            1.0
        end

        push!(flow_data, Dict(
            "position" => [x, y],
            "velocity" => velocity,
            "magnitude" => norm(velocity),
            "flow" => flow_val
        ))
    end

    return Dict(
        "supported" => true,
        "grid_size" => grid_size,
        "data" => flow_data
    )
end

function compute_distribution_data(adapter::GridWorldAdapter, model::GFlowNetModel, trajectories::Vector{Trajectory})::Dict
    grid_size = adapter.grid_size

    # Count terminal positions
    counts = zeros(Int, grid_size, grid_size)
    for traj in trajectories
        terminal = traj.states[end]
        if terminal.is_terminal
            counts[terminal.x, terminal.y] += 1
        end
    end

    # Normalize to probabilities
    total = sum(counts)
    probs = total > 0 ? counts ./ total : counts

    # Also compute target distribution P(x) ∝ R(x)
    target = zeros(grid_size, grid_size)
    for x in 1:grid_size, y in 1:grid_size
        state = GridState(x, y, true)
        target[x, y] = reward(state)
    end
    target_sum = sum(target)
    target = target_sum > 0 ? target ./ target_sum : target

    return Dict(
        "supported" => true,
        "grid_size" => grid_size,
        "empirical" => probs,
        "target" => target,
        "counts" => counts,
        "total_samples" => total
    )
end

# ============================================
# Helper Functions
# ============================================

function action_to_string(action::GridAction)::String
    if action isa MoveRight
        return "right"
    elseif action isa MoveUp
        return "up"
    elseif action isa MoveLeft
        return "left"
    elseif action isa MoveDown
        return "down"
    elseif action isa Terminate
        return "terminate"
    else
        return "unknown"
    end
end

function compute_velocity_from_policy(probs::Vector{Float64}, actions::Vector{<:AbstractAction})
    vx, vy = 0.0, 0.0

    for (i, action) in enumerate(actions)
        p = i <= length(probs) ? probs[i] : 0.0

        if action isa MoveRight
            vx += p
        elseif action isa MoveLeft
            vx -= p
        elseif action isa MoveUp
            vy += p
        elseif action isa MoveDown
            vy -= p
        end
        # Terminate doesn't contribute to velocity
    end

    return [vx, vy]
end
```

---

## Part 3: Unified Server Implementation

**File: `src/utils/visualization/api/unified_server.jl`**

```julia
using Oxygen
using JSON3
using HTTP
using UUIDs
using Dates

# Global session storage (single session for now)
const CURRENT_SESSION = Ref{Union{TrainingSession, Nothing}}(nothing)
const TRAINING_TASK = Ref{Union{Task, Nothing}}(nothing)

# ============================================
# Domain & Model Creation
# ============================================

"""Create model and adapter based on domain type"""
function create_model_and_adapter(domain_type::String, config::Dict)
    if domain_type == "grid_world"
        return create_grid_world_model_and_adapter(config)
    else
        error("Unsupported domain type: $domain_type. Currently only 'grid_world' is implemented.")
    end
end

function create_grid_world_model_and_adapter(config::Dict)
    grid_size = get(config, "grid_size", 10)

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
        reward_positions[(1, grid_size)] = 8.0
        reward_positions[(grid_size, 1)] = 8.0
    end

    # Create model
    model = create_grid_world_gflownet(
        grid_size = grid_size,
        reward_positions = reward_positions,
        hidden_dim = get(config, "hidden_dim", 64),
        learning_rate = get(config, "learning_rate", 0.001)
    )

    # Create adapter
    adapter = GridWorldAdapter(grid_size, reward_positions)

    return model, adapter
end

# ============================================
# API Endpoints
# ============================================

# Enable CORS
function add_cors_headers(handler)
    return function(req)
        headers = [
            "Access-Control-Allow-Origin" => "*",
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

# --- Domain Info ---

@get "/api/v2/domain/info" function(req)
    session = CURRENT_SESSION[]

    if session === nothing
        return json(Dict(
            "has_session" => false,
            "available_domains" => ["grid_world"],
            "message" => "No active session. Start training to initialize."
        ))
    end

    return json(Dict(
        "has_session" => true,
        "domain" => get_domain_config(session.adapter),
        "renderer" => get_renderer_name(session.adapter)
    ))
end

# --- Training Control ---

@post "/api/v2/training/start" function(req)
    # Parse configuration
    config = JSON3.read(String(req.body), Dict)

    # Stop existing training if any
    if CURRENT_SESSION[] !== nothing && CURRENT_SESSION[].is_training
        CURRENT_SESSION[].is_training = false
        sleep(0.1)  # Let the training loop exit
    end

    # Create new session
    session = create_session(config)
    CURRENT_SESSION[] = session

    # Start training loop
    session.is_training = true
    session.start_time = now()

    TRAINING_TASK[] = @async begin
        try
            while session.is_training && session.current_iteration < session.total_iterations
                if !session.is_paused
                    step!(session)
                end
                sleep(0.05)  # ~20 iterations/second max
            end
        catch e
            @error "Training error" exception=e
            session.is_training = false
        end
    end

    return json(Dict(
        "status" => "started",
        "session_id" => session.id,
        "domain" => get_domain_config(session.adapter),
        "renderer" => get_renderer_name(session.adapter),
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

# --- Training State ---

@get "/api/v2/training/state" function(req)
    session = CURRENT_SESSION[]

    if session === nothing
        return json(Dict("has_session" => false))
    end

    # Compute metrics from recent trajectories
    recent = session.trajectory_buffer
    universal_metrics = isempty(recent) ? Dict() : compute_gflownet_metrics(session.model, recent)
    domain_metrics = isempty(recent) ? Dict() : compute_domain_metrics(session.adapter, session.model, recent)

    return json(Dict(
        "has_session" => true,
        "is_training" => session.is_training,
        "is_paused" => session.is_paused,
        "current_iteration" => session.current_iteration,
        "total_iterations" => session.total_iterations,
        "progress" => session.current_iteration / max(session.total_iterations, 1),

        # Latest metrics
        "latest_loss" => isempty(session.losses) ? nothing : session.losses[end],
        "latest_reward" => isempty(session.rewards) ? nothing : session.rewards[end],
        "latest_gradient_norm" => isempty(session.gradient_norms) ? nothing : session.gradient_norms[end],

        # Computed metrics
        "metrics" => universal_metrics,
        "domain_metrics" => domain_metrics
    ))
end

@get "/api/v2/training/history" function(req)
    session = CURRENT_SESSION[]

    if session === nothing
        return json(Dict("has_session" => false))
    end

    return json(Dict(
        "losses" => session.losses,
        "rewards" => session.rewards,
        "gradient_norms" => session.gradient_norms,
        "iteration_times" => session.iteration_times
    ))
end

# --- Trajectories ---

@get "/api/v2/trajectories" function(req)
    session = CURRENT_SESSION[]

    if session === nothing
        return json(Dict("trajectories" => [], "domain" => nothing))
    end

    # Convert trajectories to viz format
    viz_trajectories = [
        trajectory_to_viz_data(session.adapter, traj, "traj_$i")
        for (i, traj) in enumerate(session.trajectory_buffer)
    ]

    return json(Dict(
        "trajectories" => viz_trajectories,
        "domain" => get_domain_config(session.adapter),
        "count" => length(viz_trajectories)
    ))
end

# --- Analysis ---

@get "/api/v2/analysis/flow" function(req)
    session = CURRENT_SESSION[]

    if session === nothing
        return json(Dict("supported" => false, "error" => "No session"))
    end

    flow_data = compute_flow_field(session.adapter, session.model)
    return json(flow_data)
end

@get "/api/v2/analysis/distribution" function(req)
    session = CURRENT_SESSION[]

    if session === nothing
        return json(Dict("supported" => false, "error" => "No session"))
    end

    dist_data = compute_distribution_data(session.adapter, session.model, session.trajectory_buffer)
    return json(dist_data)
end

# --- Server Launch ---

function start_server(; port::Int = 8080)
    serve(; port = port, middleware = [add_cors_headers])
end
```

---

## Part 4: Frontend Updates

### 4.1 API Client Update

**File: `src/utils/visualization/web/src/lib/api.ts`**

```typescript
// New v2 API client for real training
const API_V2_BASE = '/api/v2';

export const api = {
  // Domain
  getDomainInfo: () =>
    fetch(`${API_V2_BASE}/domain/info`).then(r => r.json()),

  // Training
  startTraining: (config: TrainingConfig) =>
    fetch(`${API_V2_BASE}/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    }).then(r => r.json()),

  stopTraining: () =>
    fetch(`${API_V2_BASE}/training/stop`, { method: 'POST' }).then(r => r.json()),

  getTrainingState: () =>
    fetch(`${API_V2_BASE}/training/state`).then(r => r.json()),

  getTrainingHistory: () =>
    fetch(`${API_V2_BASE}/training/history`).then(r => r.json()),

  // Trajectories
  getTrajectories: () =>
    fetch(`${API_V2_BASE}/trajectories`).then(r => r.json()),

  // Analysis
  getFlowField: () =>
    fetch(`${API_V2_BASE}/analysis/flow`).then(r => r.json()),

  getDistribution: () =>
    fetch(`${API_V2_BASE}/analysis/distribution`).then(r => r.json()),
};

export interface TrainingConfig {
  domain_type: 'grid_world';  // Only grid_world for now
  grid_size: number;
  reward_peaks: Array<{
    position: [number, number];
    intensity: number;
    name?: string;
  }>;
  n_episodes: number;
  batch_size: number;
  learning_rate: number;
  objective: 'TRAJECTORY_BALANCE' | 'DETAILED_BALANCE' | 'FLOW_MATCHING';
}
```

### 4.2 Training Mode Indicator

Add a clear indicator showing "REAL TRAINING" vs "SIMULATION":

```tsx
// In MonitoringDashboard.tsx or similar

function TrainingModeIndicator({ isReal }: { isReal: boolean }) {
  return (
    <div className={`training-mode ${isReal ? 'real' : 'simulation'}`}>
      {isReal ? (
        <>
          <span className="indicator real">●</span>
          <span>Real GFlowNet Training</span>
        </>
      ) : (
        <>
          <span className="indicator sim">●</span>
          <span>Simulation Mode</span>
        </>
      )}
    </div>
  );
}
```

---

## Part 5: File Structure Summary

```
src/utils/visualization/
├── core/                              # NEW
│   ├── adapters.jl                   # AbstractDomainAdapter interface
│   ├── training_session.jl           # TrainingSession manager
│   └── metrics.jl                    # Universal metrics computation
│
├── domains/                           # NEW
│   ├── grid_world.jl                 # GridWorldAdapter (implemented)
│   └── generic.jl                    # GenericAdapter (fallback, stub)
│
├── api/
│   ├── unified_server.jl             # NEW: Real training server
│   └── gflownet_server.jl            # KEEP: Simulation mode (backward compat)
│
└── web/src/
    ├── lib/
    │   └── api.ts                    # MODIFY: Add v2 API client
    └── components/
        └── TrainingModeIndicator.tsx  # NEW: Show real vs simulation
```

---

## Part 6: Implementation Order

### Phase 1: Core Infrastructure
1. Create `src/utils/visualization/core/adapters.jl` with interface
2. Create `src/utils/visualization/core/training_session.jl`
3. Create `src/utils/visualization/core/metrics.jl`

### Phase 2: Grid World Adapter
4. Create `src/utils/visualization/domains/grid_world.jl`
5. Test adapter with existing GFlowNet training

### Phase 3: Unified Server
6. Create `src/utils/visualization/api/unified_server.jl`
7. Add all v2 endpoints
8. Test server with real training

### Phase 4: Frontend Integration
9. Update API client with v2 endpoints
10. Add training mode indicator
11. Update data fetching hooks to use v2

### Phase 5: Validation
12. Verify real training produces expected metrics
13. Verify flow field reflects learned policy
14. Verify distribution matches P(x) ∝ R(x) after convergence

---

## Confirmation Questions

1. **Backward Compatibility**: Should the old `/api/` simulation endpoints remain available, or fully replace with `/api/v2/`?

2. **Training Speed**: Real training will be slower (~0.05s per iteration). Is this acceptable for visualization responsiveness?

3. **Initial Domain**: Confirm Grid World is the only domain to implement now, with architecture ready for others?

4. **Validation Display**: Should the UI prominently show validation metrics (like "Flow Conservation: ✓ 98.5%") to confirm correctness?
