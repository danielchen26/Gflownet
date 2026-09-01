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

# =============================================================================
# Reachability, flow and transition probabilities for multi-start models
# =============================================================================
#
# The flow and transition methods existed only for GFlowNetModel, which is why
# DETAILED_BALANCE and FLOW_MATCHING threw MethodError on every iteration for a multi-start
# model and the training loop recorded the result as NaN.
#
# F(s) IS NOT START-INDEPENDENT, and an earlier version of this comment claimed it was.
# The recursion F(s) = sum_children F(c) P_B(s|c) reads only the action set and the backward
# policy, so the RECURSION does not mention the initial state -- but the P_B it uses must be
# normalised over the parents of each child WITHIN THE DAG ROOTED AT THAT START, and that
# set does depend on the start.
#
# Measured consequence of getting this wrong, 4x4 grid, rewards {(4,4): 10.0, (1,4): 8.0},
# normalising over the GLOBAL parent set:
#
#     start    Z with global parents    true sum_x R(x)    error    P_B trajectory mass
#     (1,1)              39.000              39.000        +0.0%    [1.000, 1.000]
#     (2,2)              10.250              25.000       -59.0%    [0.250, 1.000]
#     (3,1)              10.375              23.000       -54.9%    [0.125, 1.000]
#     (1,3)              16.375              29.000       -43.5%    [0.125, 1.000]
#
# The source start is exact because every global parent is reachable from it. For any other
# start, cells outside its cone are counted as parents, sum_tau P_B(tau|x) leaks below 1 --
# measured as low as 0.125 -- and by Z = sum_x R(x) sum_tau P_B(tau|x) the learned Z
# collapses. End-to-end training confirmed it: 900 iterations, two seeds, start (1,1)
# landed on 38.90 and 38.88 against a true 39.0 while start (2,2) landed on 10.18 and
# 10.17 against a true 25.0.
#
# `reachable_from` below is the fix: parents are filtered to the reachable cone before
# normalising. Only F(s_0^i) = Z_i is per-start in the log_Z VECTOR, but the P_B used to
# get there is per-start too.

"""
    reachable_from(model::MultiStartGFlowNetModel, idx::Int; max_states) -> Set

Every state reachable from `model.initial_states[idx]` by applicable actions.

Depends only on the initial state, the action set and the domain's applicability rules --
never on parameters -- so it is safe to compute once per training run and reuse across
iterations. It is NOT safe to cache across runs: a domain whose configuration changes
(grid world installs a global `GRID_CONFIG` at model construction) can change what is
applicable. `clear_reachability_cache!` is called at the top of `train_gflownet`.

`max_states` refuses rather than exhausting memory on a domain whose reachable set is not
enumerable. A domain that trips it cannot use multi-start with a P_B-dependent objective.
"""
const REACHABILITY_CACHE = Dict{Tuple{UInt64,Int},Set{Any}}()

clear_reachability_cache!() = (empty!(REACHABILITY_CACHE); nothing)

function reachable_from(model::MultiStartGFlowNetModel, idx::Int;
                        max_states::Int = 200_000)
    key = (objectid(model), idx)
    cached = get(REACHABILITY_CACHE, key, nothing)
    isnothing(cached) || return cached

    seen = Set{Any}()
    stack = Any[model.initial_states[idx]]
    while !isempty(stack)
        s = pop!(stack)
        s in seen && continue
        push!(seen, s)
        if length(seen) > max_states
            throw(ArgumentError(
                "reachable set from initial state $idx exceeded $max_states states. " *
                "Multi-start objectives that need P_B must normalise over the parents " *
                "reachable from each start, which requires enumerating that cone; this " *
                "domain's cone is too large. Use a single-start model."))
        end
        for a in get_applicable_actions(s, model.all_actions)
            push!(stack, apply_action(a, s))
        end
    end

    REACHABILITY_CACHE[key] = seen
    return seen
end

"""
    reachable_parent_count(model, child, idx) -> Int

Number of parents of `child` that lie in the cone of `model.initial_states[idx]`.

This is the denominator a uniform backward policy must use. `backward_parent_states`
returns the GLOBAL parent set, which over-counts for any start that is not the source.
"""
function reachable_parent_count(model::MultiStartGFlowNetModel, child, idx::Int)
    cone = reachable_from(model, idx)
    n = 0
    for p in backward_parent_states(child, model.all_actions)
        p in cone && (n += 1)
    end
    return n
end

# CAVEAT on the two flow methods below, stated because it is load-bearing and easy to miss:
# they delegate to the policy-context recursion in flows.jl, which normalises P_B over the
# GLOBAL parent set. That is exact only for a start from which every global parent is
# reachable -- the source. For any other start these return the flow of the global network,
# not of that start's cone, and the two differ by the same factor documented above. The TB
# loss does NOT use them; it uses `reachable_parent_count`. DETAILED_BALANCE and
# FLOW_MATCHING do use them, so their multi-start optimum is the global-network reading.
# Fixing that means threading the start index into the flow recursion, which changes a
# public signature and is out of scope here; it is recorded rather than hidden.

compute_recursive_flow(model::MultiStartGFlowNetModel, state::AbstractState)::Float64 =
    compute_recursive_flow(model.all_actions, model.backward_policy,
                           model.parameters, model.states, state)

# No memoized variant: memoization in flows.jl caches per single-start model, and the
# multi-start grids in use are small. Correctness first; add a cache when a profile asks.
flow(model::MultiStartGFlowNetModel, state::AbstractState)::Float64 =
    compute_recursive_flow(model, state)

function forward_transition_probability(model::MultiStartGFlowNetModel, source_state, target_state)
    applicable = get_applicable_actions(source_state, model.all_actions)
    isempty(applicable) && return 0.0

    probs = forward_action_probabilities(model.forward_policy, source_state,
                                        model.all_actions, model.parameters.forward,
                                        model.states.forward)

    # Sum over every applicable action that actually lands on target_state. Summing rather
    # than taking the first match matters wherever two distinct actions produce the same
    # child, which is exactly the multi-parent structure this whole repair is about.
    total = 0.0
    for (i, action) in enumerate(model.all_actions)
        action in applicable || continue
        apply_action(action, source_state) == target_state || continue
        total += probs[i]
    end
    return total
end

function backward_transition_probability(model::MultiStartGFlowNetModel, target_state, source_state)
    if isnothing(model.backward_policy) || !haskey(model.parameters, :backward)
        parents = backward_parent_states(target_state, model.all_actions)
        return isempty(parents) ? 1.0 : 1.0 / length(parents)
    end
    return compute_backward_probability(
        model.backward_policy, target_state, source_state,
        model.parameters.backward, model.states.backward, model.all_actions
    )
end