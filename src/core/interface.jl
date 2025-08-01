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
- `include_backward=false`: Whether to include backward policy for full trajectory balance
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
    partition_function_method::PartitionFunctionMethod = SIMPLE_ESTIMATION,
    rng = Random.default_rng()
)
    n_actions = length(all_actions)

    # Create neural networks using official Lux patterns
    forward_policy, forward_ps, forward_st = create_forward_policy(state_dim, hidden_dim, n_actions, rng)
    flow_estimator, flow_ps, flow_st = create_flow_estimator(state_dim, hidden_dim, rng)
    
    # Initialize partition function parameter based on method
    log_partition_function = if partition_function_method == LEARNABLE_ESTIMATION
        0.0  # Initialize log Z to 0 (Z = 1)
    else
        nothing  # For SIMPLE_ESTIMATION, SAMPLING_ESTIMATION, etc.
    end

    # Optionally create backward policy
    if include_backward
        backward_policy, backward_ps, backward_st = create_backward_policy(state_dim, hidden_dim, rng)
        
        # Organize parameters with backward policy and optional Z parameter
        parameters = if partition_function_method == LEARNABLE_ESTIMATION
            ComponentArray(
                forward = forward_ps,
                backward = backward_ps,
                flow = flow_ps,
                log_Z = log_partition_function  # Add Z as learnable parameter
            )
        else
            ComponentArray(
                forward = forward_ps,
                backward = backward_ps,
                flow = flow_ps
            )
        end
        
        # Network states
        states = (forward = forward_st, backward = backward_st, flow = flow_st)
    else
        backward_policy = nothing
        
        # Organize parameters without backward policy and optional Z parameter  
        parameters = if partition_function_method == LEARNABLE_ESTIMATION
            ComponentArray(
                forward = forward_ps,
                flow = flow_ps,
                log_Z = log_partition_function  # Add Z as learnable parameter
            )
        else
            ComponentArray(
                forward = forward_ps,
                flow = flow_ps
            )
        end
        
        # Network states
        states = (forward = forward_st, flow = flow_st)
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
# Exports
# =============================================================================

export create_gflownet, create_forward_policy, create_backward_policy, create_flow_estimator
export sample_trajectory, sample_action_from_policy