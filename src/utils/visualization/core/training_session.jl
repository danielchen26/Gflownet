# Training session management for real GFlowNet visualization
# Manages training state, metrics history, and trajectory buffers

using GFlowNet: GFlowNetModel, TrainingConfig, Trajectory, TrainingObjective
using GFlowNet: TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING
using GFlowNet: SUB_TRAJECTORY_BALANCE, DIRECT_FLOW_OBJECTIVE
using GFlowNet: sample_trajectory, reward
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

# Arguments
- `name::String`: Objective name (e.g., "TRAJECTORY_BALANCE")

# Returns
- `TrainingObjective`: The corresponding enum value

# Example
```julia
obj = parse_objective("TRAJECTORY_BALANCE")
@assert obj == TRAJECTORY_BALANCE
```
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

"""
    create_session(config::Dict)::TrainingSession

Create new training session from configuration dict.
This function is implemented in unified_server.jl which calls
create_model_and_adapter() to construct the model and adapter.

# Arguments
- `config::Dict`: Configuration with domain_type, grid_size, n_episodes, etc.

# Returns
- `TrainingSession`: Initialized session ready for training
"""
function create_session(config::Dict)::TrainingSession
    domain_type = get(config, "domain_type", "grid_world")
    model, adapter = create_model_and_adapter(domain_type, config)

    # Build TrainingConfig using kwargs-only constructor (matches real API)
    objective = parse_objective(get(config, "objective", "TRAJECTORY_BALANCE"))

    # FLOW_MATCHING requires flow_estimator — create model with it if needed
    # Use temperature > 1 to encourage exploration and discover multiple modes
    #
    # EXPLORATION PARAMETERS (Phase 6: Mode Collapse Fix)
    # - epsilon: ε-uniform exploration mixing (Malkin et al. 2022)
    # - epsilon_decay: Anneal epsilon to 0 over training
    # - entropy_weight: Policy entropy regularization (AISTATS 2024)
    # - z_learning_rate_multiplier: Faster Z convergence (peptide paper: 10x)
    training_config = TrainingConfig(
        objective       = objective,
        n_iterations    = get(config, "n_episodes", 500),
        batch_size      = get(config, "batch_size", 16),
        learning_rate   = get(config, "learning_rate", 0.001),
        temperature     = get(config, "temperature", 2.0),  # Higher temp for better exploration
        # Exploration parameters for mode discovery
        epsilon         = get(config, "epsilon", 0.05),           # ε-uniform exploration (default 5%)
        epsilon_decay   = get(config, "epsilon_decay", true),     # Anneal to 0 over training
        entropy_weight  = get(config, "entropy_weight", 0.01),    # Policy entropy (AISTATS 2024)
        z_learning_rate_multiplier = get(config, "z_learning_rate_multiplier", 10.0),  # Faster Z
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
        Trajectory[], 200,                      # trajectory_buffer, max_buffer_size (larger for better distribution)
        nothing, 0,                             # last_error, error_count
        nothing, Float64[]                      # start_time, iteration_times
    )
end

"""
    step!(session::TrainingSession)::Dict

Execute one training iteration. Returns a status Dict with metrics.

# Arguments
- `session::TrainingSession`: The training session to step

# Returns
- `Dict`: Status dict with "status", "iteration", "loss", "mean_reward", etc.
"""
function step!(session::TrainingSession)::Dict
    if !session.is_training || session.is_paused
        return Dict("status" => "not_running")
    end

    model  = session.model
    config = session.config

    t0 = time()

    try
        # ---- Compute current epsilon with annealing ----
        # Linearly anneal from config.epsilon to 0 over training
        current_epsilon = if config.epsilon_decay
            config.epsilon * (1.0 - session.current_iteration / session.total_iterations)
        else
            config.epsilon
        end

        # ---- Sample trajectories with ε-uniform exploration ----
        # This is the CRITICAL exploration mechanism for mode discovery (Malkin et al. 2022)
        sampling_config = GFlowNet.SamplingConfig(
            strategy = config.temperature != 1.0 ? GFlowNet.TEMPERATURE_SAMPLING : GFlowNet.STOCHASTIC_SAMPLING,
            temperature = config.temperature,
            epsilon = current_epsilon,
            max_trajectory_length = 100
        )
        trajectories = [sample_trajectory(model; config=sampling_config) for _ in 1:config.batch_size]

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
