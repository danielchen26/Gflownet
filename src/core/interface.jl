# High-Level Interface for GFlowNet - Zygote+ComponentArray+Lux Compatible
# Clean, robust implementation following official best practices

using Lux
using ComponentArrays
using Optimisers
using Zygote
using Random
using Statistics

# =============================================================================
# Model Creation - Following Official Lux Patterns
# =============================================================================

"""
    create_gflownet(initial_state, all_actions; state_dim, hidden_dim=64, learning_rate=0.01, rng=Random.default_rng())

Create a GFlowNet model using proper Lux+ComponentArray+Zygote patterns.

Follows official Lux documentation patterns for gradient computation compatibility.
"""
function create_gflownet(
    initial_state::AbstractState,
    all_actions::Vector{<:AbstractAction};
    state_dim::Int,
    hidden_dim::Int = 64,
    learning_rate::Float64 = 0.01,
    rng = Random.default_rng()
)
    n_actions = length(all_actions)

    # Create neural networks using official Lux patterns
    forward_policy, forward_ps, forward_st = create_forward_policy(state_dim, hidden_dim, n_actions, rng)
    flow_estimator, flow_ps, flow_st = create_flow_estimator(state_dim, hidden_dim, rng)

    # Organize parameters in ComponentArray (official Lux pattern)
    parameters = ComponentArray(
        forward = forward_ps,
        flow = flow_ps
    )

    # Setup optimizer (official Lux pattern)
    opt = Optimisers.Adam(learning_rate)
    optimizer = Optimisers.setup(opt, parameters)

    # Create model
    return GFlowNetModel(
        initial_state,
        all_actions,
        forward_policy,
        flow_estimator,
        parameters,
        optimizer,
        (forward = forward_st, flow = flow_st)
    )
end

"""
    create_forward_policy(state_dim::Int, hidden_dim::Int, n_actions::Int, rng)

Create forward policy neural network following Lux patterns.
"""
function create_forward_policy(state_dim::Int, hidden_dim::Int, n_actions::Int, rng)
    # Standard Lux Chain
    policy_net = Lux.Chain(
        Lux.Dense(state_dim => hidden_dim, tanh),
        Lux.Dense(hidden_dim => hidden_dim, tanh),
        Lux.Dense(hidden_dim => n_actions)  # Raw logits
    )

    ps, st = Lux.setup(rng, policy_net)
    return ForwardPolicy(policy_net), ps, st
end

"""
    create_flow_estimator(state_dim::Int, hidden_dim::Int, rng)

Create flow estimation neural network following Lux patterns.
"""
function create_flow_estimator(state_dim::Int, hidden_dim::Int, rng)
    # Standard Lux Chain
    flow_net = Lux.Chain(
        Lux.Dense(state_dim => hidden_dim, tanh),
        Lux.Dense(hidden_dim => hidden_dim, tanh),
        Lux.Dense(hidden_dim => 1)  # Single flow value
    )

    ps, st = Lux.setup(rng, flow_net)
    return FlowEstimator(flow_net), ps, st
end

# =============================================================================
# Trajectory Sampling - Zygote-Safe Implementation
# =============================================================================

"""
    sample_trajectory(model::GFlowNetModel; config::SamplingConfig = SamplingConfig())

Sample trajectory using Zygote-safe operations.
"""
function sample_trajectory(model::GFlowNetModel; config = SamplingConfig())
    # Handle both SamplingConfig and named tuple configs
    if isa(config, NamedTuple)
        # Convert named tuple to SamplingConfig
        acyclic_rate = get(config, :acyclic_rate, 0.0)
        strategy = get(config, :strategy, :stochastic)
        temperature = get(config, :temperature, 1.0)

        # Convert strategy symbol to enum
        strategy_enum = if strategy == :stochastic
            STOCHASTIC_SAMPLING
        elseif strategy == :greedy
            GREEDY_SAMPLING
        elseif strategy == :temperature
            TEMPERATURE_SAMPLING
        else
            STOCHASTIC_SAMPLING
        end

        config = SamplingConfig(
            strategy=strategy_enum,
            temperature=temperature,
            acyclic_rate=acyclic_rate
        )
    elseif !isa(config, SamplingConfig)
        config = SamplingConfig()
    end
    trajectory_states = [model.initial_state]
    trajectory_actions = AbstractAction[]

    current_state = model.initial_state
    steps = 0

    # Simple acyclic control: track visited states if acyclic_rate > 0
    visited_states = Set()
    if config.acyclic_rate > 0.0
        push!(visited_states, current_state)
    end

    while !is_terminal_state(current_state) && steps < config.max_trajectory_length
        steps += 1

        # Get applicable actions - computed fresh each time (on-demand approach)
        applicable_actions = Zygote.@ignore get_applicable_actions(current_state, model.all_actions)

        # Apply simple acyclic control if enabled
        if config.acyclic_rate > 0.0
            # Filter out actions that would lead to visited states
            applicable_actions = filter(applicable_actions) do action
                next_state = compute_next_state(action, current_state)
                next_state ∉ visited_states || rand() > config.acyclic_rate
            end
        end

        if isempty(applicable_actions)
            break
        end

        # Sample action using Zygote-safe operations
        action = sample_action_from_policy(model, current_state, applicable_actions; config = config)

        # Apply action
        next_state = compute_next_state(action, current_state)

        # Update visited states for acyclic control
        if config.acyclic_rate > 0.0
            push!(visited_states, next_state)
        end

        # Update trajectory (non-mutating pattern for Zygote)
        trajectory_actions = [trajectory_actions..., action]
        trajectory_states = [trajectory_states..., next_state]

        current_state = next_state
    end

    return Trajectory(trajectory_states, trajectory_actions)
end

"""
    sample_action_from_policy(model, state, applicable_actions; config)

Sample action from forward policy using Zygote-safe operations.
"""
function sample_action_from_policy(model::GFlowNetModel, state::AbstractState,
                                  applicable_actions::Vector{<:AbstractAction};
                                  config::SamplingConfig = SamplingConfig())

    # Get state features
    features = state_to_features(state)

    # Compute forward logits using proper Lux call
    logits_vec, _ = model.forward_policy.model(features, model.parameters.forward, model.states.forward)

    # Find applicable indices (discrete logic - non-differentiable)
    applicable_indices = Zygote.@ignore [i for (i, a) in enumerate(model.all_actions) if a in applicable_actions]

    if isempty(applicable_indices)
        # Fallback: return first applicable action
        return applicable_actions[1]
    end

    # Extract logits for applicable actions
    applicable_logits = logits_vec[applicable_indices]

    # Apply temperature if needed
    if config.strategy == TEMPERATURE_SAMPLING
        applicable_logits = applicable_logits ./ config.temperature
    end

    # Convert to probabilities using numerically stable softmax
    max_logit = maximum(applicable_logits)
    exp_logits = exp.(applicable_logits .- max_logit)
    probs = exp_logits ./ sum(exp_logits)

    # Sample action based on strategy
    if config.strategy == GREEDY_SAMPLING
        action_idx = argmax(probs)
    else
        # Stochastic sampling
        cumulative = cumsum(probs)
        r = rand()
        action_idx = findfirst(p -> p >= r, cumulative)
        if isnothing(action_idx)
            action_idx = length(probs)
        end
    end

    return applicable_actions[action_idx]
end

# =============================================================================
# Training - Proper Zygote+ComponentArray+Lux Pattern
# =============================================================================

"""
    train_gflownet(model::GFlowNetModel, config::TrainingConfig; verbose::Bool = false)

Train GFlowNet using proper Lux+Zygote patterns.
"""
function train_gflownet(model::GFlowNetModel, config::TrainingConfig; verbose::Bool = false)
    history = TrainingHistory()

    if verbose
        println("🚀 Starting GFlowNet training...")
        println("   Configuration:")
        println("     - Objective: $(config.objective)")
        println("     - Iterations: $(config.n_iterations)")
        println("     - Batch size: $(config.batch_size)")
        println("     - Learning rate: $(config.learning_rate)")
    end

    for iteration in 1:config.n_iterations
        start_time = time()

        try
            # Sample trajectories
            trajectories = [sample_trajectory(model) for _ in 1:config.batch_size]

            # Compute loss and gradients using official Lux pattern
            loss_val, gradient_norm = train_step!(model, trajectories, config)

            # Record metrics
            push!(history.losses, loss_val)
            push!(history.gradient_norms, gradient_norm)
            push!(history.iteration_times, time() - start_time)

            # Verbose output
            if verbose && (iteration % config.validation_frequency == 0)
                avg_loss = mean(filter(!isnan, history.losses[max(1, end-4):end]))
                println("   Iteration $iteration:")
                println("     - Loss: $(round(loss_val, digits=4))")
                println("     - Avg Loss (5): $(isnan(avg_loss) ? "NaN" : round(avg_loss, digits=4))")
                println("     - Gradient norm: $(round(gradient_norm, digits=4))")
                println("     - Time: $(round(time() - start_time, digits=3))s")
                println("     - Trajectories: $(length(trajectories))")
            end

        catch e
            # Record failed iteration
            push!(history.losses, NaN)
            push!(history.gradient_norms, NaN)
            push!(history.iteration_times, time() - start_time)

            if verbose
                println("   ⚠️  Training error at iteration $iteration: $e")
            end
        end
    end

    if verbose
        successful_iterations = count(!isnan, history.losses)
        final_loss = isempty(filter(!isnan, history.losses)) ? NaN : filter(!isnan, history.losses)[end]
        total_time = sum(history.iteration_times)

        println("   ✅ Training completed:")
        println("     - Final loss: $(isnan(final_loss) ? "NaN" : round(final_loss, digits=4))")
        println("     - Total time: $(round(total_time, digits=1))s")
        println("     - Successful iterations: $successful_iterations/$(config.n_iterations)")
    end

    return history
end

"""
    train_step!(model, trajectories, config)

Perform single training step using official Lux+Zygote pattern.
"""
function train_step!(model::GFlowNetModel, trajectories::Vector{Trajectory}, config::TrainingConfig)

    # Define loss function following official Lux pattern
    loss_function = ps -> compute_trajectory_loss(model, trajectories, ps, config)

    # Compute gradients using official Zygote pattern
    loss_val, grads = Zygote.withgradient(loss_function, model.parameters)

    # Check for valid gradients
    if grads[1] === nothing || any_invalid(grads[1])
        return Inf, 0.0
    end

    # Compute gradient norm
    gradient_norm = compute_gradient_norm(grads[1])

    # Update parameters using Optimisers.jl
    optimizer_state, parameters = Optimisers.update(model.optimizer, model.parameters, grads[1])

    # Update model state (mutation after gradient computation is safe)
    model.optimizer = optimizer_state
    model.parameters = parameters

    return loss_val, gradient_norm
end

# =============================================================================
# Loss Computation - Mathematically Correct Trajectory Balance
# =============================================================================

"""
    compute_trajectory_loss(model, trajectories, params, config)

Compute trajectory balance loss using Zygote-safe operations.

FIXED: Now implements correct trajectory balance: P_F(τ) ∝ R(s_T)
"""
function compute_trajectory_loss(model::GFlowNetModel, trajectories::Vector{Trajectory},
                                params, config::TrainingConfig)

    # Filter valid trajectories (discrete validation - non-differentiable)
    valid_trajectories = Zygote.@ignore [traj for traj in trajectories if is_valid_trajectory(traj)]

    if isempty(valid_trajectories)
        return 0.0
    end

    # Compute losses using Zygote-safe operations
    losses = [compute_single_trajectory_loss(model, traj, params) for traj in valid_trajectories]

    # Filter out infinite losses
    finite_losses = filter(!isinf, losses)

    if isempty(finite_losses)
        return Inf
    end

    return mean(finite_losses)
end

"""
    compute_single_trajectory_loss(model, trajectory, params)

Compute loss for single trajectory with CORRECTED trajectory balance.
"""
function compute_single_trajectory_loss(model::GFlowNetModel, trajectory::Trajectory, params)

    # Compute log probability of trajectory
    log_prob_sum = 0.0

    for i in 1:(length(trajectory.states)-1)
        state = trajectory.states[i]
        action = trajectory.actions[i]

        # Get state features
        features = state_to_features(state)

        # Compute forward logits using proper Lux call (Zygote-safe)
        logits_vec, _ = model.forward_policy.model(features, params.forward, model.states.forward)

        # Get applicable actions on-demand (discrete logic - non-differentiable)
        applicable_actions = Zygote.@ignore get_applicable_actions(state, model.all_actions)

        if isempty(applicable_actions)
            return Inf  # Invalid trajectory
        end

        # Find action and applicable indices (discrete logic - non-differentiable)
        action_idx = Zygote.@ignore findfirst(a -> a == action, model.all_actions)
        applicable_indices = Zygote.@ignore [i for (i, a) in enumerate(model.all_actions) if a in applicable_actions]

        if isnothing(action_idx)
            return Inf  # Invalid action
        end

        if !(action_idx in applicable_indices)
            return Inf  # Action not applicable
        end

        # Compute log probability using numerically stable operations
        applicable_logits = logits_vec[applicable_indices]
        if isempty(applicable_logits)
            return Inf
        end

        # Use logsumexp for numerical stability
        log_probs = applicable_logits .- logsumexp(applicable_logits)

        # Find action position in applicable actions (discrete logic - non-differentiable)
        action_pos = Zygote.@ignore findfirst(==(action_idx), applicable_indices)

        if isnothing(action_pos)
            return Inf
        end

        log_prob_sum += log_probs[action_pos]
    end

    # Get terminal reward (domain-specific function - non-differentiable)
    terminal_state = trajectory.states[end]
    terminal_reward = Zygote.@ignore reward(terminal_state)

    # Ensure positive reward for GFlowNet
    if terminal_reward <= 0
        terminal_reward = 1e-8
    end

    # CORRECTED Trajectory Balance Loss: (log P_F(τ) - log R(s_T))²
    # This enforces P_F(τ) ∝ R(s_T) which is the correct GFlowNet objective
    log_reward = log(terminal_reward)
    trajectory_balance_error = log_prob_sum - log_reward

    return trajectory_balance_error^2
end

# =============================================================================
# Utility Functions - Zygote-Safe
# =============================================================================

"""
    sample_trajectory_batch(model, batch_size; config)

Sample multiple trajectories efficiently.
"""
function sample_trajectory_batch(model::GFlowNetModel, batch_size::Int;
                                config::SamplingConfig = SamplingConfig())
    return [sample_trajectory(model; config = config) for _ in 1:batch_size]
end

"""
    is_valid_trajectory(trajectory)

Check if trajectory is valid.
"""
function is_valid_trajectory(trajectory::Trajectory)
    return !isempty(trajectory.states) &&
           length(trajectory.states) == length(trajectory.actions) + 1 &&
           is_terminal_state(trajectory.states[end])
end

"""
    any_invalid(gradients)

Check if gradients contain invalid values.
"""
function any_invalid(gradients)
    for grad in values(gradients)
        if grad isa AbstractArray
            if any(isnan, grad) || any(isinf, grad)
                return true
            end
        end
    end
    return false
end

"""
    compute_gradient_norm(gradients)

Compute L2 norm of gradients with proper ComponentArray support.
"""
function compute_gradient_norm(gradients)
    norm_squared = 0.0

    function add_gradient_contribution!(obj)
        if obj isa AbstractArray && !isempty(obj)
            norm_squared += sum(abs2, obj; init=0.0)
        elseif obj isa NamedTuple
            for value in values(obj)
                add_gradient_contribution!(value)
            end
        elseif hasproperty(obj, :axes) && hasmethod(values, (typeof(obj),))
            # ComponentArray or similar structure
            try
                for value in values(obj)
                    add_gradient_contribution!(value)
                end
            catch
                # Fallback: try to access as NamedTuple-like
                try
                    for key in keys(obj)
                        add_gradient_contribution!(getproperty(obj, key))
                    end
                catch
                    # Last resort: treat as array if possible
                    if obj isa AbstractArray && !isempty(obj)
                        norm_squared += sum(abs2, obj; init=0.0)
                    end
                end
            end
        end
    end

    try
        add_gradient_contribution!(gradients)
    catch e
        @warn "Error computing gradient norm: $e"
        return 0.0
    end

    return sqrt(max(norm_squared, 0.0))
end

"""
    logsumexp(x)

Numerically stable log-sum-exp operation.
"""
function logsumexp(x::AbstractVector)
    if isempty(x)
        return -Inf
    end
    max_x = maximum(x)
    if isinf(max_x)
        return max_x
    end
    return max_x + log(sum(exp.(x .- max_x)))
end

# =============================================================================
# Domain Interface Declarations - Required by Applications
# =============================================================================

"""Convert state to feature vector - must return Vector{Float32}"""
function state_to_features end

"""Check if state is terminal"""
function is_terminal_state end

"""Compute reward for state (positive for terminals)"""
function reward end

"""Check if action is applicable from state"""
function is_applicable end

"""Apply action to state, returning new state"""
function apply_action end

# =============================================================================
# Exports
# =============================================================================

export create_gflownet, create_forward_policy, create_flow_estimator
export sample_trajectory, sample_trajectory_batch, train_gflownet
export compute_trajectory_loss, compute_single_trajectory_loss
