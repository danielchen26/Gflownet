# Multi-Start GFlowNets - Support for Multiple Initial States
# Each initial state has its own partition function Z(s₀)

using ComponentArrays
using Random
using ..GFlowNet: AbstractState, AbstractAction, ForwardPolicy, BackwardPolicy, FlowEstimator
using ..GFlowNet: forward_action_probabilities, sample_action_from_policy
using ..GFlowNet: state_to_features, is_terminal_state, reward, apply_action, is_applicable
using ..GFlowNet: Trajectory, SamplingConfig, TEMPERATURE_SAMPLING, GREEDY_SAMPLING
using ..GFlowNet: create_forward_policy, create_backward_policy, create_flow_estimator
using Zygote
using Optimisers

# =============================================================================
# Multi-Start Model Type
# =============================================================================

"""
    MultiStartGFlowNetModel

GFlowNet model supporting multiple initial states with per-state partition functions.

Each initial state s₀ⁱ has its own partition function Z(s₀ⁱ), and the probability
of starting from a particular initial state is:

P(s₀ⁱ) = Z(s₀ⁱ) / Σⱼ Z(s₀ʲ)

# Fields
- `initial_states::Vector{<:AbstractState}` - Multiple starting states
- `all_actions::Vector{<:AbstractAction}` - Complete action space
- `forward_policy::ForwardPolicy` - Forward policy network
- `backward_policy::Union{Nothing,BackwardPolicy}` - Optional backward policy
- `flow_estimator::Union{Nothing,FlowEstimator}` - Optional flow estimator
- `log_partition_functions::Vector{Float64}` - log Z for each initial state
- `parameters::ComponentArray` - All trainable parameters
- `optimizer` - Optimizer state
- `states::NamedTuple` - Neural network states
"""
mutable struct MultiStartGFlowNetModel
    initial_states::Vector{<:AbstractState}
    all_actions::Vector{<:AbstractAction}
    forward_policy::ForwardPolicy
    backward_policy::Union{Nothing,BackwardPolicy}
    flow_estimator::Union{Nothing,FlowEstimator}
    log_partition_functions::Vector{Float64}
    parameters::ComponentArray
    optimizer
    states::NamedTuple
    
    function MultiStartGFlowNetModel(
        initial_states::Vector{<:AbstractState},
        all_actions::Vector{<:AbstractAction},
        forward_policy::ForwardPolicy,
        backward_policy::Union{Nothing,BackwardPolicy},
        flow_estimator::Union{Nothing,FlowEstimator},
        log_partition_functions::Vector{Float64},
        parameters::ComponentArray,
        optimizer,
        states::NamedTuple
    )
        if isempty(initial_states)
            throw(ArgumentError("initial_states cannot be empty"))
        end
        if isempty(all_actions)
            throw(ArgumentError("all_actions cannot be empty"))
        end
        if length(log_partition_functions) != length(initial_states)
            throw(ArgumentError("Must have one log Z per initial state"))
        end
        new(initial_states, all_actions, forward_policy, backward_policy, 
            flow_estimator, log_partition_functions, parameters, optimizer, states)
    end
end

# =============================================================================
# Initial State Selection
# =============================================================================

"""
    sample_initial_state(model::MultiStartGFlowNetModel)

Sample an initial state based on the learned partition functions.

The probability of selecting initial state s₀ⁱ is:
P(s₀ⁱ) = exp(log Z(s₀ⁱ)) / Σⱼ exp(log Z(s₀ʲ))
"""
function sample_initial_state(model::MultiStartGFlowNetModel)
    # Get log probabilities from log partition functions
    log_probs = model.log_partition_functions
    
    # Convert to probabilities using softmax
    max_log_prob = maximum(log_probs)
    exp_probs = exp.(log_probs .- max_log_prob)
    probs = exp_probs ./ sum(exp_probs)
    
    # Sample from categorical distribution
    cumsum_probs = cumsum(probs)
    r = rand()
    idx = findfirst(p -> p ≥ r, cumsum_probs)
    if isnothing(idx)
        idx = length(model.initial_states)
    end
    
    return model.initial_states[idx], idx
end

"""
    get_initial_state_distribution(model::MultiStartGFlowNetModel)

Get the probability distribution over initial states.

Returns a vector of probabilities corresponding to model.initial_states.
"""
function get_initial_state_distribution(model::MultiStartGFlowNetModel)
    log_probs = model.log_partition_functions
    max_log_prob = maximum(log_probs)
    exp_probs = exp.(log_probs .- max_log_prob)
    return exp_probs ./ sum(exp_probs)
end

# =============================================================================
# Trajectory Sampling
# =============================================================================

"""
    sample_trajectory(model::MultiStartGFlowNetModel; config::SamplingConfig = SamplingConfig())

Sample a trajectory starting from a sampled initial state.

The initial state is selected based on the learned partition functions.
"""
function sample_trajectory(model::MultiStartGFlowNetModel; config::SamplingConfig = SamplingConfig())
    # Sample initial state
    initial_state, initial_idx = sample_initial_state(model)
    
    # Continue with standard trajectory sampling
    trajectory_states = [initial_state]
    trajectory_actions = AbstractAction[]
    
    current_state = initial_state
    steps = 0
    
    while !is_terminal_state(current_state) && steps < config.max_trajectory_length
        steps += 1
        
        # Get applicable actions
        applicable_actions = Zygote.@ignore begin
            [a for a in model.all_actions if is_applicable(a, current_state)]
        end
        
        if isempty(applicable_actions)
            break
        end
        
        # Sample action
        action = sample_action_multi_start(model, current_state, applicable_actions; config = config)
        
        # Apply action
        next_state = apply_action(action, current_state)
        
        # Update trajectory
        push!(trajectory_actions, action)
        push!(trajectory_states, next_state)
        
        current_state = next_state
    end
    
    # Return trajectory with initial state index for loss computation
    traj = Trajectory(trajectory_states, trajectory_actions)
    return traj, initial_idx
end

"""
    sample_action_multi_start(model, state, applicable_actions; config)

Sample action using the forward policy (same as single-start).
"""
function sample_action_multi_start(model::MultiStartGFlowNetModel, state::AbstractState,
                                  applicable_actions::Vector{<:AbstractAction};
                                  config::SamplingConfig = SamplingConfig())
    # Get state features
    features = state_to_features(state)
    
    # Compute forward logits
    logits_vec, _ = model.forward_policy.model(features, model.parameters.forward, model.states.forward)
    
    # Find applicable indices
    applicable_indices = Zygote.@ignore [i for (i, a) in enumerate(model.all_actions) if a in applicable_actions]
    
    if isempty(applicable_indices)
        return applicable_actions[1]
    end
    
    # Extract logits for applicable actions
    applicable_logits = logits_vec[applicable_indices]
    
    # Apply temperature if needed
    if config.strategy == TEMPERATURE_SAMPLING
        applicable_logits = applicable_logits ./ config.temperature
    end
    
    # Convert to probabilities
    max_logit = maximum(applicable_logits)
    exp_logits = exp.(applicable_logits .- max_logit)
    probs = exp_logits ./ sum(exp_logits)

    # ε-Uniform Exploration Mixing (Standard GFlowNet practice)
    # P(a|s) = (1-ε) × P_F(a|s) + ε × Uniform(applicable_actions)
    if config.epsilon > 0.0
        n_actions = length(probs)
        uniform_prob = 1.0 / n_actions
        probs = (1.0 - config.epsilon) .* probs .+ config.epsilon * uniform_prob
    end

    # Sample action
    if config.strategy == GREEDY_SAMPLING
        action_idx = argmax(probs)
    else
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
# Model Creation
# =============================================================================

"""
    create_multi_start_gflownet(initial_states, all_actions; kwargs...)

Create a multi-start GFlowNet model with multiple initial states.

# Arguments
- `initial_states::Vector{<:AbstractState}`: Multiple starting states
- `all_actions::Vector{<:AbstractAction}`: Complete action space
- `state_dim::Int`: Dimension of state features
- `hidden_dim::Int = 64`: Hidden layer size
- `learning_rate::Float64 = 0.01`: Learning rate
- `include_backward::Bool = false`: Include backward policy
- `initialize_log_z::Float64 = 0.0`: Initial value for log Z

# Returns
`MultiStartGFlowNetModel` with per-initial-state partition functions
"""
function create_multi_start_gflownet(
    initial_states::Vector{<:AbstractState},
    all_actions::Vector{<:AbstractAction};
    state_dim::Int,
    hidden_dim::Int = 64,
    learning_rate::Float64 = 0.01,
    include_backward::Bool = false,
    initialize_log_z::Float64 = 0.0,
    rng = Random.default_rng()
)
    n_actions = length(all_actions)
    n_initial_states = length(initial_states)
    
    # Create neural networks
    forward_policy, forward_ps, forward_st = GFlowNet.create_forward_policy(state_dim, hidden_dim, n_actions, rng)
    flow_estimator, flow_ps, flow_st = GFlowNet.create_flow_estimator(state_dim, hidden_dim, rng)
    
    # Initialize log partition functions
    log_partition_functions = fill(initialize_log_z, n_initial_states)
    
    # Create backward policy if requested
    if include_backward
        backward_policy, backward_ps, backward_st = GFlowNet.create_backward_policy(state_dim, hidden_dim, rng)
        
        # Organize parameters with multiple log Z values
        parameters = ComponentArray(
            forward = forward_ps,
            backward = backward_ps,
            flow = flow_ps,
            log_Z = log_partition_functions
        )
        
        states = (forward = forward_st, backward = backward_st, flow = flow_st)
    else
        backward_policy = nothing
        
        parameters = ComponentArray(
            forward = forward_ps,
            flow = flow_ps,
            log_Z = log_partition_functions
        )
        
        states = (forward = forward_st, flow = flow_st)
    end
    
    # Setup optimizer
    opt = Optimisers.Adam(learning_rate)
    optimizer = Optimisers.setup(opt, parameters)
    
    return MultiStartGFlowNetModel(
        initial_states,
        all_actions,
        forward_policy,
        backward_policy,
        flow_estimator,
        log_partition_functions,
        parameters,
        optimizer,
        states
    )
end