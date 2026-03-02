# Training session management for real GFlowNet visualization
# Manages training state, metrics history, and trajectory buffers

using GFlowNet: GFlowNetModel, TrainingConfig, Trajectory, TrainingObjective
using GFlowNet: TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING
using GFlowNet: SUB_TRAJECTORY_BALANCE, DIRECT_FLOW_OBJECTIVE, TRAJECTORY_LIKELIHOOD_MAXIMIZATION
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

    # Experience replay buffer (JMLR 2023)
    replay_buffer::Union{Nothing, GFlowNet.ReplayBuffer}

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
        "TRAJECTORY_BALANCE"               => TRAJECTORY_BALANCE,
        "TB"                               => TRAJECTORY_BALANCE,
        "DETAILED_BALANCE"                 => DETAILED_BALANCE,
        "DB"                               => DETAILED_BALANCE,
        "FLOW_MATCHING"                    => FLOW_MATCHING,
        "FM"                               => FLOW_MATCHING,
        "SUB_TRAJECTORY_BALANCE"           => SUB_TRAJECTORY_BALANCE,
        "STB"                              => SUB_TRAJECTORY_BALANCE,
        "DIRECT_FLOW_OBJECTIVE"            => DIRECT_FLOW_OBJECTIVE,
        "DFO"                              => DIRECT_FLOW_OBJECTIVE,
        "TRAJECTORY_LIKELIHOOD_MAXIMIZATION" => TRAJECTORY_LIKELIHOOD_MAXIMIZATION,
        "TLM"                              => TRAJECTORY_LIKELIHOOD_MAXIMIZATION,
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
    # EXPERIENCE REPLAY (JMLR 2023)
    # - use_replay_buffer, replay_ratio, replay_priority_alpha
    # TLM (ICLR 2025)
    # - tlm_backward_weight, tlm_entropy_coeff
    # JSON doesn't distinguish Int vs Float, so explicitly convert all numeric params
    # Defaults tuned for 8×8 grid (3432:1 path asymmetry)
    # CRITICAL: temperature=1.0, not 2.0! Temperature 2.0 causes random early termination.
    training_config = TrainingConfig(
        objective       = objective,
        n_iterations    = Int(get(config, "n_episodes", 1000)),
        batch_size      = Int(get(config, "batch_size", 32)),
        learning_rate   = Float64(get(config, "learning_rate", 0.005)),
        temperature     = Float64(get(config, "temperature", 1.0)),
        # Exploration scaled for 8×8 path asymmetry (3432:1)
        epsilon         = Float64(get(config, "epsilon", 0.15)),
        epsilon_decay   = Bool(get(config, "epsilon_decay", true)),
        entropy_weight  = Float64(get(config, "entropy_weight", 0.02)),
        z_learning_rate_multiplier = Float64(get(config, "z_learning_rate_multiplier", 10.0)),
        # Experience Replay (JMLR 2023)
        use_replay_buffer     = Bool(get(config, "use_replay_buffer", false)),
        replay_buffer_size    = Int(get(config, "replay_buffer_size", 10000)),
        replay_ratio          = Float64(get(config, "replay_ratio", 0.5)),
        replay_priority_alpha = Float64(get(config, "replay_priority_alpha", 0.6)),
        # TLM parameters (ICLR 2025)
        tlm_backward_weight   = Float64(get(config, "tlm_backward_weight", 1.0)),
        tlm_entropy_coeff     = Float64(get(config, "tlm_entropy_coeff", 0.01)),
        verbose         = false
    )

    # Initialize replay buffer if configured (JMLR 2023: Off-policy learning)
    replay_buf = if training_config.use_replay_buffer
        GFlowNet.ReplayBuffer(training_config.replay_buffer_size; alpha=training_config.replay_priority_alpha)
    else
        nothing
    end

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
        replay_buf,                             # replay_buffer (JMLR 2023)
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
        # ---- 1. Compute current epsilon with annealing ----
        current_epsilon = if config.epsilon_decay
            config.epsilon * (1.0 - session.current_iteration / session.total_iterations)
        else
            config.epsilon
        end

        # ---- 2. Sample fresh trajectories with ε-uniform exploration ----
        sampling_config = GFlowNet.SamplingConfig(
            strategy = config.temperature > 1.0 ? GFlowNet.TEMPERATURE_SAMPLING : GFlowNet.STOCHASTIC_SAMPLING,
            temperature = config.temperature,
            epsilon = current_epsilon,
            max_trajectory_length = 100
        )
        fresh_trajectories = [sample_trajectory(model; config=sampling_config) for _ in 1:config.batch_size]

        # ---- 3. TLM backward sampling (ICLR 2025) ----
        # Bypasses forward path asymmetry by sampling from terminal states backward
        if config.objective == TRAJECTORY_LIKELIHOOD_MAXIMIZATION && !isnothing(model.backward_policy)
            backward_trajectories = try
                terminal_states = [traj.states[end] for traj in fresh_trajectories
                                   if !isempty(traj.states) && GFlowNet.is_terminal_state(traj.states[end])]
                if !isempty(terminal_states)
                    n_backward = min(config.batch_size ÷ 2, length(terminal_states))
                    GFlowNet.sample_backward_trajectories_from_terminals(
                        model, terminal_states, n_backward; config=sampling_config
                    )
                else
                    Trajectory[]
                end
            catch e
                if session.current_iteration == 0
                    @warn "TLM backward sampling failed (forward-only): $e"
                end
                Trajectory[]
            end
            if !isempty(backward_trajectories)
                fresh_trajectories = vcat(fresh_trajectories, backward_trajectories)
            end
        end

        # ---- 4. Replay buffer mixing (JMLR 2023) ----
        replay_buffer = session.replay_buffer
        training_data = if !isnothing(replay_buffer) && length(replay_buffer) >= config.batch_size
            # Add fresh trajectories to buffer with reward-based priority
            # Higher reward → higher priority → more likely to be replayed
            # This helps retain high-reward modes (critical for mode collapse prevention)
            for traj in fresh_trajectories
                traj_reward = reward(traj.states[end])
                priority = GFlowNet.compute_trajectory_priority(traj_reward)
                GFlowNet.add!(replay_buffer, traj, priority)
            end

            # Compute fresh vs replay split
            n_replay = round(Int, config.batch_size * config.replay_ratio)
            n_fresh = config.batch_size - n_replay

            # Sample from replay buffer with importance weights
            replay_trajs, replay_weights, _ = GFlowNet.sample_with_weights(replay_buffer, n_replay)

            # Combine fresh + replay
            fresh_subset = n_fresh > 0 ? fresh_trajectories[1:min(n_fresh, length(fresh_trajectories))] : Trajectory[]
            combined_trajs = vcat(fresh_subset, replay_trajs)
            combined_weights = vcat(ones(length(fresh_subset)), replay_weights)

            (trajectories=combined_trajs, weights=combined_weights, use_weights=true)
        else
            # No replay or buffer not full yet — just add to buffer and use fresh
            if !isnothing(replay_buffer)
                for traj in fresh_trajectories
                    traj_reward = reward(traj.states[end])
                    GFlowNet.add!(replay_buffer, traj, GFlowNet.compute_trajectory_priority(traj_reward))
                end
            end
            (trajectories=fresh_trajectories, weights=ones(length(fresh_trajectories)), use_weights=false)
        end

        # ---- 5. Gradient descent step (weighted if replay active) ----
        loss_val, grad_norm = if training_data.use_weights
            GFlowNet.train_step_weighted!(model, training_data.trajectories, training_data.weights, config)
        else
            GFlowNet.train_step!(model, training_data.trajectories, config)
        end

        iteration_time = time() - t0

        # ---- 6. Update session state and metrics ----
        session.current_iteration += 1
        push!(session.losses, loss_val)
        push!(session.gradient_norms, grad_norm)
        push!(session.iteration_times, iteration_time)

        # Update trajectory buffer (ring buffer for visualization)
        for traj in fresh_trajectories
            push!(session.trajectory_buffer, traj)
            if length(session.trajectory_buffer) > session.max_buffer_size
                popfirst!(session.trajectory_buffer)
            end
        end

        # Store molecules if this is a molecular domain
        if hasproperty(session.adapter, :generated_molecules)
            try
                store_molecules_from_trajectories!(session.adapter, fresh_trajectories, session.current_iteration; session_id=session.id)
            catch e
                @warn "Molecule storage failed" exception=e
            end
        end

        # Compute rewards from terminal states
        batch_rewards = Float64[reward(t.states[end]) for t in fresh_trajectories]
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
