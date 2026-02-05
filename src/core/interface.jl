# High-Level Interface for GFlowNet - Model Creation and Sampling
# Clean implementation with training functions moved to training/

using Lux
using ComponentArrays
using Optimisers
using Zygote
using Random
using Statistics

using ..GFlowNet: AbstractState, AbstractAction, GFlowNetModel, Trajectory
using ..GFlowNet: ForwardPolicy, BackwardPolicy, FlowEstimator
using ..GFlowNet: PartitionFunctionMethod, SIMPLE_ESTIMATION, LEARNABLE_ESTIMATION
using ..GFlowNet: SamplingConfig, SamplingStrategy
using ..GFlowNet: STOCHASTIC_SAMPLING, GREEDY_SAMPLING, TEMPERATURE_SAMPLING
using ..GFlowNet: get_applicable_actions, is_terminal_state

# =============================================================================
# Model Creation - Following Official Lux Patterns
# =============================================================================

"""
    create_gflownet(initial_state, all_actions; kwargs...)

Create a GFlowNet model using proper Lux+ComponentArray+Zygote patterns.

# Arguments
- `initial_state`: Starting state s₀
- `all_actions`: Complete action space
- `state_dim`: Dimension of state features
- `hidden_dim=64`: Hidden layer size for neural networks
- `learning_rate=0.01`: Learning rate for optimizer
- `include_backward=false`: Whether to include backward policy for DETAILED_BALANCE
- `include_flow_estimator=false`: Whether to include flow estimator for DIRECT_FLOW/FLOW_MATCHING
- `partition_function_method=SIMPLE_ESTIMATION`: How to handle partition function Z:
  - `SIMPLE_ESTIMATION`: Z = 1 (default, fixed)
  - `LEARNABLE_ESTIMATION`: Learn Z as trainable parameter (recommended for complex environments)
- `rng=Random.default_rng()`: Random number generator

# Returns
`GFlowNetModel` with all components initialized

# Example
```julia
# With fixed Z (default)
model = create_gflownet(
    initial_state, all_actions;
    state_dim = 10,
    hidden_dim = 64
)

# With learnable Z (recommended)
model = create_gflownet(
    initial_state, all_actions;
    state_dim = 10,
    hidden_dim = 64,
    partition_function_method = LEARNABLE_ESTIMATION
)
```

Follows official Lux documentation patterns for gradient computation compatibility.
"""
function create_gflownet(
    initial_state::AbstractState,
    all_actions::Vector{<:AbstractAction};
    state_dim::Int,
    hidden_dim::Int = 64,
    learning_rate::Float64 = 0.01,
    include_backward::Bool = false,
    include_flow_estimator::Bool = false,
    partition_function_method::PartitionFunctionMethod = SIMPLE_ESTIMATION,
    rng = Random.default_rng()
)
    n_actions = length(all_actions)

    # Create neural networks using official Lux patterns
    forward_policy, forward_ps, forward_st = create_forward_policy(state_dim, hidden_dim, n_actions, rng)
    
    # Optionally create flow estimator
    if include_flow_estimator
        flow_estimator, flow_ps, flow_st = create_flow_estimator(state_dim, hidden_dim, rng)
    else
        flow_estimator = nothing
        flow_ps = nothing
        flow_st = nothing
    end
    
    # Initialize partition function parameter based on method
    log_partition_function = if partition_function_method == LEARNABLE_ESTIMATION
        0.0  # Initialize log Z to 0 (Z = 1)
    else
        nothing  # For SIMPLE_ESTIMATION, SAMPLING_ESTIMATION, etc.
    end

    # Build parameters and states based on which components are included
    if include_backward && include_flow_estimator
        # Both backward policy and flow estimator
        backward_policy, backward_ps, backward_st = create_backward_policy(state_dim, hidden_dim, rng)
        
        parameters = if partition_function_method == LEARNABLE_ESTIMATION
            ComponentArray(
                forward = forward_ps,
                backward = backward_ps,
                flow = flow_ps,
                log_Z = log_partition_function
            )
        else
            ComponentArray(
                forward = forward_ps,
                backward = backward_ps,
                flow = flow_ps
            )
        end
        
        states = (forward = forward_st, backward = backward_st, flow = flow_st)
        
    elseif include_backward && !include_flow_estimator
        # Only backward policy
        backward_policy, backward_ps, backward_st = create_backward_policy(state_dim, hidden_dim, rng)
        
        parameters = if partition_function_method == LEARNABLE_ESTIMATION
            ComponentArray(
                forward = forward_ps,
                backward = backward_ps,
                log_Z = log_partition_function
            )
        else
            ComponentArray(
                forward = forward_ps,
                backward = backward_ps
            )
        end
        
        states = (forward = forward_st, backward = backward_st)
        
    elseif !include_backward && include_flow_estimator
        # Only flow estimator
        backward_policy = nothing
        
        parameters = if partition_function_method == LEARNABLE_ESTIMATION
            ComponentArray(
                forward = forward_ps,
                flow = flow_ps,
                log_Z = log_partition_function
            )
        else
            ComponentArray(
                forward = forward_ps,
                flow = flow_ps
            )
        end
        
        states = (forward = forward_st, flow = flow_st)
        
    else
        # Neither backward policy nor flow estimator
        backward_policy = nothing
        
        parameters = if partition_function_method == LEARNABLE_ESTIMATION
            ComponentArray(
                forward = forward_ps,
                log_Z = log_partition_function
            )
        else
            ComponentArray(
                forward = forward_ps
            )
        end
        
        states = (forward = forward_st,)
    end

    # Setup optimizer (official Lux pattern)
    opt = Optimisers.Adam(learning_rate)
    optimizer = Optimisers.setup(opt, parameters)

    # Create model
    return GFlowNetModel(
        initial_state,
        all_actions,
        forward_policy,
        backward_policy,
        flow_estimator,
        log_partition_function,
        parameters,
        optimizer,
        states
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

"""
    create_backward_policy(state_dim::Int, hidden_dim::Int, rng)

Create backward policy neural network for computing P_B(s|s').

Uses joint state representation: takes concatenated features of (source, target) states
and outputs a single probability value.
"""
function create_backward_policy(state_dim::Int, hidden_dim::Int, rng)
    # Input dimension is 2 * state_dim for joint representation
    input_dim = 2 * state_dim
    
    # Standard Lux Chain
    backward_net = Lux.Chain(
        Lux.Dense(input_dim => hidden_dim, tanh),
        Lux.Dense(hidden_dim => hidden_dim, tanh),
        Lux.Dense(hidden_dim => 1)  # Single probability logit
    )

    ps, st = Lux.setup(rng, backward_net)
    return BackwardPolicy(backward_net), ps, st
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

    # ε-Uniform Exploration Mixing (Standard GFlowNet practice)
    # P(a|s) = (1-ε) × P_F(a|s) + ε × Uniform(applicable_actions)
    # Reference: Malkin et al. (2022), Shen et al. (ICML 2023)
    if config.epsilon > 0.0
        n_actions = length(probs)
        uniform_prob = 1.0 / n_actions
        probs = (1.0 - config.epsilon) .* probs .+ config.epsilon * uniform_prob
    end

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
# Backward Trajectory Sampling - TLM Support (ICLR 2025)
# =============================================================================

"""
    sample_backward_trajectory(model::GFlowNetModel, terminal_state::AbstractState;
                               config::SamplingConfig = SamplingConfig())

Sample a trajectory backwards from a terminal state to the initial state using the backward policy.

# Mathematical Foundation (TLM - ICLR 2025)
The backward policy P_B(s|s') is trained to implicitly encode path counts:
    P_B(s|s') ≈ n(s) / n(s')
where n(s) = number of paths from initial state to s.

By sampling backwards from terminal states (which are sampled ∝ R), we:
1. Bypass the 70:1 forward path asymmetry
2. Generate diverse trajectories that would be rarely sampled forward
3. Use these for training to improve mode coverage

# Arguments
- `model::GFlowNetModel`: The GFlowNet model (must have backward_policy)
- `terminal_state::AbstractState`: Terminal state to sample backward from
- `config::SamplingConfig`: Sampling configuration

# Returns
`Trajectory` with states/actions from initial to terminal (forward direction)
"""
function sample_backward_trajectory(model::GFlowNetModel, terminal_state::AbstractState;
                                    config = SamplingConfig())

    if isnothing(model.backward_policy)
        throw(ArgumentError("Backward trajectory sampling requires backward_policy. Use include_backward=true in create_gflownet"))
    end

    # Build trajectory in reverse (from terminal to initial)
    backward_states = [terminal_state]
    backward_actions = AbstractAction[]

    current_state = terminal_state
    steps = 0
    max_steps = config.max_trajectory_length

    while current_state != model.initial_state && steps < max_steps
        steps += 1

        # Find all parent states that can transition to current_state
        parent_candidates = find_parent_states(model, current_state)

        if isempty(parent_candidates)
            # Cannot find parents - may be at initial state or disconnected
            break
        end

        # Sample parent state using backward policy
        parent_state, action = sample_parent_from_backward_policy(
            model, current_state, parent_candidates; config=config
        )

        # Add to backward trajectory
        pushfirst!(backward_states, parent_state)
        pushfirst!(backward_actions, action)

        current_state = parent_state
    end

    # The trajectory is now in forward direction (initial → terminal)
    return Trajectory(backward_states, backward_actions)
end

"""
    find_parent_states(model::GFlowNetModel, target_state::AbstractState)

Find all states that can transition to target_state.

Returns vector of (parent_state, action) tuples.
"""
function find_parent_states(model::GFlowNetModel, target_state::AbstractState)
    # This is a simple implementation that works for grid worlds
    # For more complex domains, this would need domain-specific implementation

    parents = Tuple{AbstractState, AbstractAction}[]

    # Try each action from potential parent states
    # For grid world: parent is one step behind in the direction of the action
    for action in model.all_actions
        # Try to find a parent state where applying action gives target
        parent = find_parent_for_action(target_state, action)
        if !isnothing(parent)
            # Verify the transition is valid
            if is_applicable(action, parent) && apply_action(action, parent) == target_state
                push!(parents, (parent, action))
            end
        end
    end

    return parents
end

"""
    find_parent_for_action(target_state, action)

For a given target state and action, find the parent state that would produce target via action.
Domain-specific implementation.
"""
function find_parent_for_action(target_state::AbstractState, action::AbstractAction)
    # Default implementation - requires domain override
    # For grid world, this is handled by specialized methods
    return nothing
end

"""
    sample_parent_from_backward_policy(model, target_state, parent_candidates; config)

Sample a parent state using the backward policy P_B(parent|target).
"""
function sample_parent_from_backward_policy(model::GFlowNetModel, target_state::AbstractState,
                                            parent_candidates::Vector{<:Tuple}; config = SamplingConfig())

    if isempty(parent_candidates)
        error("No parent candidates for backward sampling")
    end

    if length(parent_candidates) == 1
        return parent_candidates[1]
    end

    # Compute backward probabilities for each parent
    probs = Float64[]
    for (parent, action) in parent_candidates
        prob = compute_backward_probability(
            model.backward_policy, target_state, parent,
            model.parameters.backward, model.states.backward,
            model.all_actions
        )
        push!(probs, max(prob, 1e-8))
    end

    # Normalize
    prob_sum = sum(probs)
    if prob_sum > 1e-8
        probs ./= prob_sum
    else
        probs = fill(1.0 / length(probs), length(probs))
    end

    # Apply epsilon exploration if configured
    if config.epsilon > 0.0
        uniform_prob = 1.0 / length(probs)
        probs = (1.0 - config.epsilon) .* probs .+ config.epsilon * uniform_prob
    end

    # Sample
    cumulative = cumsum(probs)
    r = rand()
    idx = findfirst(p -> p >= r, cumulative)
    if isnothing(idx)
        idx = length(probs)
    end

    return parent_candidates[idx]
end

"""
    sample_backward_trajectories_from_terminals(model::GFlowNetModel, terminal_states::Vector,
                                                n_samples::Int; config = SamplingConfig())

Sample backward trajectories from given terminal states.
Terminal states should be sampled proportionally to their rewards.

# Arguments
- `model`: GFlowNet model with backward policy
- `terminal_states`: Vector of terminal states to sample from
- `n_samples`: Number of backward trajectories to generate
- `config`: Sampling configuration

# Returns
Vector of Trajectory objects
"""
function sample_backward_trajectories_from_terminals(model::GFlowNetModel,
                                                     terminal_states::Vector,
                                                     n_samples::Int;
                                                     config = SamplingConfig())

    trajectories = Trajectory[]

    for _ in 1:n_samples
        # Randomly select a terminal state
        terminal = terminal_states[rand(1:length(terminal_states))]
        traj = sample_backward_trajectory(model, terminal; config=config)
        push!(trajectories, traj)
    end

    return trajectories
end

# =============================================================================
# Exports
# =============================================================================

export create_gflownet, create_forward_policy, create_backward_policy, create_flow_estimator
export sample_trajectory, sample_action_from_policy
export sample_backward_trajectory, sample_backward_trajectories_from_terminals