# Policy Functions - Core GFlowNet Mathematical Operations
# Implementation of P_F, P_B, and Z - the three fundamental GFlowNet functions

using Random
using StatsBase: Weights, sample
using NNlib: softmax, sigmoid
using Zygote

# =============================================================================
# Forward Policy P_F(a|s) - Mathematical Foundation
# =============================================================================

"""
    forward_probability(policy::ForwardPolicy, state, action, parameters, states)

Compute P_F(a|s) - the probability of taking action a from state s.

# Mathematical Foundation
The forward policy defines the probability distribution over actions:
P_F(a|s) = exp(f_θ(s)[a]) / Σ_a' exp(f_θ(s)[a'])

where f_θ(s) are the logits from the neural network.

# Arguments
- `policy::ForwardPolicy`: The forward policy neural network
- `state::S`: Current state
- `action::A`: Action to compute probability for
- `parameters`: Neural network parameters
- `states`: Neural network internal states

# Returns
- `Float64`: Probability P_F(a|s)
"""
function forward_probability(policy::ForwardPolicy, state, action,
    parameters, states, actions)
    # Features are a deterministic function of the state with no dependence on the
    # parameters, so they are constant inputs and must stay OFF the tape. A domain
    # whose state_to_features builds its vector by mutation -- `features = zeros(...)`
    # then `features[i] = ...`, which is the idiomatic way to write it -- otherwise
    # dies with "Mutating arrays is not supported -- called setindex!". This was
    # latent until the flow-based objectives started calling these paths inside a
    # gradient; examples/core_features/direct_flow/direct_flow_demo.jl is the case
    # that found it.
    features = Zygote.@ignore state_to_features(state)

    # Compute logits for all actions
    logits, _ = compute_forward_logits(policy, features, parameters, states)

    # Get applicable actions directly with robust indexing (non-mutating for Zygote)
    applicable_actions = [a for (i, a) in enumerate(actions) if is_applicable(a, state)]
    applicable_indices = [i for (i, a) in enumerate(actions) if is_applicable(a, state)]

    # Check if the requested action is applicable
    if action ∉ applicable_actions
        return 0.0
    end

    if isempty(applicable_indices)
        return 0.0
    end

    # Extract logits for applicable actions and apply softmax
    applicable_logits = logits[applicable_indices]
    max_logit = maximum(applicable_logits)
    exp_logits = exp.(applicable_logits .- max_logit)
    probabilities = exp_logits ./ sum(exp_logits)

    # Find the probability for the requested action using direct indexing
    for (i, a) in enumerate(applicable_actions)
        if a == action
            return Float64(probabilities[i])
        end
    end

    return 0.0
end

"""
    compute_forward_logits(policy::ForwardPolicy, features::Vector{Float32}, parameters, states)

Compute raw logits from forward policy network.

# Mathematical Foundation
Computes f_θ(s) = NN_θ(φ(s)) where:
- φ(s) are the state features
- NN_θ is the neural network with parameters θ
- Output dimension matches total number of actions

# Arguments
- `policy::ForwardPolicy`: Forward policy network
- `features::Vector{Float32}`: State features φ(s)
- `parameters`: Network parameters θ
- `states`: Network internal states

# Returns
- `Tuple{Vector{Float32}, Any}`: (logits, new_states)
"""
function compute_forward_logits(policy::ForwardPolicy, features::Vector{Float32}, parameters, states)
    # Validate inputs (non-differentiable)
    Zygote.@ignore begin
        validate_neural_network_input(features, "forward policy features")
        validate_model_parameters(parameters, "forward policy parameters")
    end

    # Neural network forward pass (differentiable)
    logits, new_states = safe_model_call(policy.model, features, parameters, states)

    # Validate outputs (non-differentiable)
    Zygote.@ignore validate_neural_network_output(logits, "forward policy logits")

    return logits, new_states
end

"""
    sample_forward_action(policy::ForwardPolicy, state, action_space,
                         parameters, states; rng=nothing)

Sample an action from the forward policy P_F(a|s).

# Mathematical Foundation
Samples a ∼ P_F(·|s) where P_F is defined by the softmax over applicable actions:
P_F(a|s) = exp(f_θ(s)[a]) / Σ_{a'∈A(s)} exp(f_θ(s)[a'])

where A(s) are the applicable actions from state s.

# Arguments
- `policy::ForwardPolicy`: Forward policy
- `state::S`: Current state
- `action_space::ActionSpace{A}`: Action space with indexing
- `parameters`: Policy parameters
- `states`: Policy internal states
- `rng`: Random number generator (optional)

# Returns
- `Tuple{A, Int}`: (sampled_action, action_index)
"""
function sample_forward_action(policy::ForwardPolicy, state, actions,
    parameters, states; rng=nothing)
    if isnothing(rng)
        rng = Random.default_rng()
    end

    # Constant input, off the tape. See the note at forward_action_probabilities.
    features = Zygote.@ignore state_to_features(state)

    # Compute logits
    logits, _ = compute_forward_logits(policy, features, parameters, states)

    # Get applicable actions directly
    applicable_actions = [action for action in actions if is_applicable(action, state)]

    # Direct sampling from applicable actions
    if isempty(applicable_actions)
        throw(ArgumentError("No applicable actions available for sampling"))
    end

    # Find indices of applicable actions
    applicable_indices = [findfirst(a -> a == action, actions) for action in applicable_actions]
    applicable_indices = filter(!isnothing, applicable_indices)

    if isempty(applicable_indices)
        throw(ArgumentError("No applicable actions found in action space"))
    end

    # Extract logits for applicable actions and apply softmax
    applicable_logits = logits[applicable_indices]
    max_logit = maximum(applicable_logits)
    exp_logits = exp.(applicable_logits .- max_logit)
    probabilities = exp_logits ./ sum(exp_logits)

    # Sample from the probability distribution
    cumulative_probs = cumsum(probabilities)
    rand_val = rand(rng)
    sample_idx = findfirst(p -> p >= rand_val, cumulative_probs)
    if isnothing(sample_idx)
        sample_idx = length(cumulative_probs)
    end

    action = applicable_actions[sample_idx]
    probability = probabilities[sample_idx]

    return action, probability
end

"""
    forward_action_probabilities(policy::ForwardPolicy, state, actions,
                                parameters, states)

Compute probability distribution P_F(·|s) over all actions.

# Returns
- `Vector{Float32}`: Probability distribution over all actions
"""
function forward_action_probabilities(policy::ForwardPolicy, state, actions,
    parameters, states)
    # Constant input, off the tape. See the note at forward_action_probabilities.
    features = Zygote.@ignore state_to_features(state)

    # Compute logits
    logits, _ = compute_forward_logits(policy, features, parameters, states)

    # Get applicable actions directly
    applicable_actions = [action for action in actions if is_applicable(action, state)]

    # Compute probabilities directly using softmax
    if isempty(applicable_actions)
        return zeros(Float32, length(actions))
    end

    # Find indices of applicable actions and apply softmax
    applicable_indices = [findfirst(a -> a == action, actions) for action in applicable_actions]
    applicable_indices = filter(!isnothing, applicable_indices)

    if isempty(applicable_indices)
        return zeros(Float32, length(actions))
    end

    # Compute normalized probabilities for applicable actions
    applicable_logits = logits[applicable_indices]
    max_logit = maximum(applicable_logits)
    exp_logits = exp.(applicable_logits .- max_logit)
    normalized_probs = exp_logits ./ sum(exp_logits)

    # Create probability vector functionally without mutations (Zygote-safe)
    probs = [
        begin
            applicable_idx = findfirst(==(i), applicable_indices)
            isnothing(applicable_idx) ? Float32(0.0) : normalized_probs[applicable_idx]
        end
        for i in 1:length(actions)
    ]

    return probs
end

# =============================================================================
# Backward Policy P_B(s|s') - Mathematical Foundation
# =============================================================================

"""
    backward_probability(policy::BackwardPolicy, target_state, source_state,
                        parameters, states)

Compute P_B(s|s') - the probability of having come from source_state given target_state.

# Mathematical Foundation
The backward policy defines the probability distribution over previous states:
P_B(s|s') = exp(b_θ(s')[s]) / Σ_s'' exp(b_θ(s')[s''])

where b_θ(s') are the logits for previous states from s'.

# Arguments
- `policy::BackwardPolicy`: Backward policy network
- `target_state::S`: Current state s'
- `source_state::S`: Previous state s to compute probability for
- `parameters`: Network parameters
- `states`: Network internal states

# Returns
- `Float64`: Probability P_B(s|s')
"""
function backward_probability(policy::BackwardPolicy, target_state, source_state,
    parameters, states, dag)
    # Get target state features
    features = state_to_features(target_state)

    # Get all possible previous states
    prev_states = get_previous_states(dag, target_state)
    if isempty(prev_states) || source_state ∉ prev_states
        return 0.0
    end

    # Compute backward logits
    logits, _ = compute_backward_logits(policy, features, prev_states, parameters, states)

    # Convert to probabilities
    probs = softmax(logits)

    # Find index of source state in prev_states
    source_idx = findfirst(s -> s == source_state, prev_states)
    return isnothing(source_idx) ? 0.0 : Float64(probs[source_idx])
end

"""
    compute_backward_logits(policy::BackwardPolicy, features::Vector{Float32},
                           prev_states::Vector{S}, parameters, states) where {S}

Compute raw logits from backward policy network.

# Mathematical Foundation
Computes b_θ(s') where the output dimension matches the number of previous states.

# Arguments
- `policy::BackwardPolicy`: Backward policy network
- `features::Vector{Float32}`: Target state features
- `prev_states::Vector{S}`: All possible previous states
- `parameters`: Network parameters
- `states`: Network internal states

# Returns
- `Tuple{Vector{Float32}, Any}`: (logits, new_states)
"""
function compute_backward_logits(policy::BackwardPolicy, features::Vector{Float32},
    prev_states::Vector, parameters, states)
    # Validate inputs (non-differentiable)
    Zygote.@ignore begin
        validate_neural_network_input(features, "backward policy features")
        validate_model_parameters(parameters, "backward policy parameters")
    end

    # Neural network forward pass (differentiable)
    logits, new_states = safe_model_call(policy.model, features, parameters, states)

    # Validate dimension consistency (non-differentiable)
    Zygote.@ignore begin
        if length(logits) != length(prev_states)
            throw(DimensionMismatch(
                "Backward policy output dimension ($(length(logits))) doesn't match " *
                "number of previous states ($(length(prev_states)))"
            ))
        end
        validate_neural_network_output(logits, "backward policy logits")
    end

    return logits, new_states
end


# =============================================================================
# Flow Estimator Z(s) - Mathematical Foundation
# =============================================================================

"""
    flow_estimate(estimator::FlowEstimator, state, parameters, states)

Compute Z(s) - the estimated total flow through state s.

# Mathematical Foundation
The flow estimator approximates the total flow F(s) passing through state s:
Z(s) = exp(z_θ(s))

where z_θ(s) is the log-flow estimate from the neural network.

# Arguments
- `estimator::FlowEstimator`: Flow estimation network
- `state`: State to estimate flow for
- `parameters`: Network parameters
- `states`: Network internal states

# Returns
- `Float64`: Estimated flow Z(s)
"""
function flow_estimate(estimator::FlowEstimator, state, parameters, states)
    # Constant input, off the tape. See the note at forward_action_probabilities.
    # This one specifically broke DETAILED_BALANCE on any domain with a mutating
    # state_to_features, because DB now differentiates through flow_estimate.
    features = Zygote.@ignore state_to_features(state)

    # Compute log-flow estimate
    log_flow, _ = compute_flow_logits(estimator, features, parameters, states)

    # Return flow estimate (exp of log-flow)
    return exp(Float64(log_flow[1]))  # Assuming scalar output
end

"""
    compute_flow_logits(estimator::FlowEstimator, features::Vector{Float32}, parameters, states)

Compute raw log-flow logits from flow estimator network.

# Mathematical Foundation
Computes z_θ(s) = NN_θ(φ(s)) where the output is a scalar log-flow estimate.

# Arguments
- `estimator::FlowEstimator`: Flow estimator network
- `features::Vector{Float32}`: State features
- `parameters`: Network parameters
- `states`: Network internal states

# Returns
- `Tuple{Vector{Float32}, Any}`: (log_flow_logits, new_states)
"""
function compute_flow_logits(estimator::FlowEstimator, features::Vector{Float32}, parameters, states)
    # Validate inputs (non-differentiable)
    Zygote.@ignore begin
        validate_neural_network_input(features, "flow estimator features")
        validate_model_parameters(parameters, "flow estimator parameters")
    end

    # Neural network forward pass (differentiable)
    logits, new_states = safe_model_call(estimator.model, features, parameters, states)

    # Validate outputs (non-differentiable)
    Zygote.@ignore validate_neural_network_output(logits, "flow estimator logits")

    return logits, new_states
end

# =============================================================================
# Unified Policy Operations
# =============================================================================

"""
    forward_transition_probability(model::GFlowNetModel, source_state::S, target_state::S) where {S}

Compute P_F(s'|s) - probability of transitioning from source to target state.

# Mathematical Foundation
P_F(s'|s) = Σ_{a: apply_action(a,s)=s'} P_F(a|s)

Sums over all actions that lead from source to target state.
"""
function forward_transition_probability(model::GFlowNetModel, source_state, target_state)
    # Get applicable actions from source state
    applicable_actions = get_applicable_actions(source_state, model.all_actions)
    if isempty(applicable_actions)
        return 0.0
    end

    # Check if any action leads to target state
    # Use array comprehension instead of push! for Zygote compatibility
    valid_actions = [action for action in applicable_actions 
                     if apply_action(action, source_state) == target_state]
    
    if isempty(valid_actions)
        return 0.0  # No action leads to target state
    end

    # Compute action probabilities over all actions
    probs = forward_action_probabilities(
        model.forward_policy, source_state, model.all_actions,
        model.parameters.forward, model.states.forward
    )

    # Sum probabilities of actions that lead to target
    total_prob = 0.0
    for (i, action) in enumerate(model.all_actions)
        if action in valid_actions
            total_prob += probs[i]
        end
    end

    return total_prob
end

"""
    backward_transition_probability(model::GFlowNetModel, target_state::S, source_state::S) where {S}

Compute P_B(s|s') - probability of having transitioned from source to target state.
"""
function backward_transition_probability(model::GFlowNetModel, target_state, source_state)
    # Check if backward policy exists
    if isnothing(model.backward_policy)
        # Default: assume deterministic backward for tree-structured spaces
        # where each state has a unique parent (P_B = 1)
        return 1.0
    end

    # Check if parameters exist for backward policy
    if !haskey(model.parameters, :backward)
        @warn "Backward policy exists but no parameters found. Using deterministic backward."
        return 1.0
    end

    # Use learned backward policy
    return compute_backward_probability(
        model.backward_policy, target_state, source_state,
        model.parameters.backward, model.states.backward,
        model.all_actions
    )
end

"""
    compute_backward_probability(policy::BackwardPolicy, target_state, source_state,
                               parameters, states, all_actions)

Compute P_B(s|s') using a learned backward policy without DAG.

# Mathematical Foundation
Uses a joint representation approach where the backward policy network takes
both source and target state features and outputs P_B(source|target).

# Arguments
- `policy::BackwardPolicy`: Backward policy network
- `target_state`: Target state s'
- `source_state`: Source state s
- `parameters`: Network parameters
- `states`: Network internal states
- `all_actions`: All possible actions (for validation)

# Returns
- `Float64`: Probability P_B(s|s')
"""
function compute_backward_probability(policy::BackwardPolicy, target_state, source_state,
    parameters, states, all_actions)

    # Naming note: `source_state` is the PARENT and `target_state` is the CHILD.
    # The DB loss calls this as (policy, target, source, ...) with source earlier
    # in the trajectory, so this returns P_B(source | target).
    #
    # A GFlowNet requires sum over parents of P_B(parent | child) == 1. The old
    # implementation returned an independent per-edge sigmoid, which cannot sum
    # to 1 except by coincidence -- measured sums were 1.1967 to 1.2922 on
    # multi-parent states and 0.51 to 0.68 on single-parent states, where the
    # only correct value is exactly 1. Normalise over the parent set instead.
    # The parent SET is discrete graph structure with no dependence on the
    # parameters, so it carries no gradient and must be computed off the tape --
    # enumerating it involves push!, which Zygote refuses to differentiate
    # ("Mutating arrays is not supported"). Only the logits below are
    # differentiated, which is what actually matters.
    #
    # There is deliberately NO separate is_valid_backward_transition call here.
    # It was redundant: backward_parent_states validates every candidate parent
    # with exactly that predicate, so membership in `parents` already implies a
    # valid transition, and the findfirst below rejects a non-parent. Worse, it
    # was expensive -- it calls apply_action for every applicable action, which in
    # the molecular domain is an RDKit fragment-join round-trip per action across
    # a 51-action space. Measured at 11.8 ms per call, 935 ms per training
    # iteration at batch_size 16, i.e. a third of the whole molecular step, to
    # recompute something the parent enumeration establishes anyway.
    parents = Zygote.@ignore backward_parent_states(target_state, all_actions)

    # A domain that does not implement find_parent_for_action yields an empty
    # parent set. Return 1.0, not 0.0: log P_B then contributes 0 and the backward
    # term vanishes, which is exactly the "fixed uniform P_B" objective and is a
    # valid GFlowNet. Returning 0.0 injected a constant log(1e-8) = -18.4 per
    # transition instead. This matches compute_recursive_flow and both flow
    # validators, which treat an unknown parent set as a unique parent.
    isempty(parents) && return 1.0

    # Unique parent: P_B is exactly 1, not a learned quantity. This is the
    # definition, not a shortcut.
    length(parents) == 1 && return 1.0

    idx = Zygote.@ignore findfirst(p -> p == source_state, parents)
    idx === nothing && return 0.0

    target_features = Zygote.@ignore state_to_features(target_state)
    Zygote.@ignore begin
        validate_model_parameters(parameters, "backward policy parameters")
    end

    # One scalar logit per parent, then a softmax over them. Written without
    # mutation so it stays Zygote-differentiable.
    logits = map(parents) do p
        joint = Zygote.@ignore vcat(target_features, state_to_features(p))
        first(safe_model_call(policy.model, joint, parameters, states)[1])
    end

    mx = maximum(logits)
    ex = exp.(logits .- mx)
    return Float64(ex[idx] / sum(ex))
end

"""
    backward_parent_states(child, all_actions)

Enumerate every state that can reach `child` in one forward action, using the
`find_parent_for_action` hook each domain already provides. Duplicates are
removed, since two different actions may induce the same parent.

This is what makes `compute_backward_probability` normalisable: you cannot form a
distribution over parents without knowing the parent set.
"""
function backward_parent_states(child, all_actions)
    parents = Any[]
    for action in all_actions
        p = find_parent_for_action(child, action)
        p === nothing && continue
        is_valid_backward_transition(p, child, all_actions) || continue
        any(q -> q == p, parents) && continue
        push!(parents, p)
    end
    return parents
end

"""
    is_valid_backward_transition(source_state, target_state, all_actions)

Check if source_state can transition to target_state using any action.
"""
function is_valid_backward_transition(source_state, target_state, all_actions)
    applicable_actions = get_applicable_actions(source_state, all_actions)
    for action in applicable_actions
        if apply_action(action, source_state) == target_state
            return true
        end
    end
    return false
end

# =============================================================================
# Backward Policy Validation Functions
# =============================================================================

"""
    validate_backward_policy_normalization(model, state, all_actions; tolerance=1e-3)

Validate that backward policy probabilities sum to 1 for all parent states.

# Mathematical Property
For any state s', the backward probabilities should satisfy:
∑_{s ∈ parents(s')} P_B(s|s') = 1

# Returns
- `is_valid::Bool`: Whether normalization is satisfied
- `total_prob::Float64`: Actual sum of probabilities
- `parent_states::Vector`: List of parent states checked
"""
function validate_backward_policy_normalization(
    model::GFlowNetModel,
    state::AbstractState,
    all_actions::Vector{<:AbstractAction};
    tolerance::Float64 = 1e-3
)
    # Skip validation if no backward policy
    if isnothing(model.backward_policy)
        return true, 1.0, AbstractState[]
    end
    
    # Skip terminal states (no parents)
    if is_terminal_state(state)
        return true, 0.0, AbstractState[]
    end
    
    # Find all parent states (states that can transition to current state)
    parent_states = AbstractState[]
    for potential_parent in Zygote.@ignore get_all_states_in_dag(model, all_actions)
        if is_valid_backward_transition(potential_parent, state, all_actions)
            push!(parent_states, potential_parent)
        end
    end
    
    # If no parents (initial state), it's valid
    if isempty(parent_states)
        return true, 0.0, parent_states
    end
    
    # Compute sum of backward probabilities
    total_prob = 0.0
    for parent in parent_states
        prob = compute_backward_probability(
            model.backward_policy, parent, state,
            model.parameters.backward, model.states.backward, all_actions
        )
        total_prob += prob
    end
    
    # Check if normalized within tolerance
    is_valid = abs(total_prob - 1.0) < tolerance
    
    return is_valid, total_prob, parent_states
end

"""
    validate_backward_policy_consistency(model, trajectories; tolerance=1e-3)

Validate backward policy consistency across a batch of trajectories.

# Checks performed:
1. All backward transitions have positive probability
2. Invalid transitions have near-zero probability
3. Normalization constraint is satisfied

# Returns
NamedTuple with validation results and statistics.
"""
function validate_backward_policy_consistency(
    model::GFlowNetModel,
    trajectories::Vector{Trajectory};
    tolerance::Float64 = 1e-3
)
    # Skip if no backward policy
    if isnothing(model.backward_policy)
        return (
            is_valid = true,
            message = "No backward policy to validate",
            stats = nothing
        )
    end
    
    # Collect statistics
    valid_transition_probs = Float64[]
    invalid_transition_probs = Float64[]
    normalization_errors = Float64[]
    
    # Check each trajectory
    for traj in trajectories
        for i in 2:length(traj.states)
            prev_state = traj.states[i-1]
            curr_state = traj.states[i]
            
            # Valid transition probability
            prob = compute_backward_probability(
                model.backward_policy, prev_state, curr_state,
                model.parameters.backward, model.states.backward, model.all_actions
            )
            push!(valid_transition_probs, prob)
            
            # Check normalization for current state
            is_normalized, total_prob, _ = validate_backward_policy_normalization(
                model, curr_state, model.all_actions; tolerance=tolerance
            )
            if !is_normalized
                push!(normalization_errors, abs(total_prob - 1.0))
            end
        end
    end
    
    # Compute summary statistics
    min_valid_prob = isempty(valid_transition_probs) ? 0.0 : minimum(valid_transition_probs)
    mean_valid_prob = isempty(valid_transition_probs) ? 0.0 : mean(valid_transition_probs)
    max_norm_error = isempty(normalization_errors) ? 0.0 : maximum(normalization_errors)
    
    # Determine overall validity
    is_valid = min_valid_prob > 1e-8 && max_norm_error < tolerance
    
    message = if is_valid
        "Backward policy validation passed"
    else
        issues = String[]
        if min_valid_prob <= 1e-8
            push!(issues, "Some valid transitions have near-zero probability")
        end
        if max_norm_error >= tolerance
            push!(issues, "Normalization constraint violated (max error: $(round(max_norm_error, digits=4)))")
        end
        "Backward policy validation failed: " * join(issues, ", ")
    end
    
    return (
        is_valid = is_valid,
        message = message,
        stats = (
            min_valid_prob = min_valid_prob,
            mean_valid_prob = mean_valid_prob,
            max_norm_error = max_norm_error,
            n_transitions_checked = length(valid_transition_probs),
            n_normalization_errors = length(normalization_errors)
        )
    )
end

"""
    monitor_backward_policy_learning(model, validation_states; verbose=true)

Monitor backward policy learning progress during training.

# Returns
Dictionary with monitoring metrics.
"""
function monitor_backward_policy_learning(
    model::GFlowNetModel,
    validation_states::Vector{<:AbstractState};
    verbose::Bool = true
)
    if isnothing(model.backward_policy)
        return Dict("status" => "No backward policy to monitor")
    end
    
    metrics = Dict{String, Any}()
    
    # Check normalization for each validation state
    norm_errors = Float64[]
    for state in validation_states
        if !is_terminal_state(state)
            is_valid, total_prob, parents = validate_backward_policy_normalization(
                model, state, model.all_actions
            )
            if !isempty(parents)
                push!(norm_errors, abs(total_prob - 1.0))
            end
        end
    end
    
    # Compute metrics
    metrics["mean_normalization_error"] = isempty(norm_errors) ? 0.0 : mean(norm_errors)
    metrics["max_normalization_error"] = isempty(norm_errors) ? 0.0 : maximum(norm_errors)
    metrics["states_checked"] = length(validation_states)
    metrics["states_with_parents"] = length(norm_errors)
    
    if verbose
        println("\n🔍 Backward Policy Monitoring:")
        println("   - Mean norm error: $(round(metrics["mean_normalization_error"], digits=6))")
        println("   - Max norm error: $(round(metrics["max_normalization_error"], digits=6))")
        println("   - States checked: $(metrics["states_checked"])")
    end
    
    return metrics
end

# Helper function to get all states in DAG (for validation)
function get_all_states_in_dag(model::GFlowNetModel, all_actions::Vector{<:AbstractAction})
    # This is a simplified version - in practice, you might want to
    # explore the state space more systematically
    states = Set{AbstractState}([model.initial_state])
    to_explore = [model.initial_state]
    
    while !isempty(to_explore) && length(states) < 1000  # Limit exploration
        current = popfirst!(to_explore)
        if !is_terminal_state(current)
            applicable = get_applicable_actions(current, all_actions)
            for action in applicable
                next_state = apply_action(action, current)
                if next_state ∉ states
                    push!(states, next_state)
                    push!(to_explore, next_state)
                end
            end
        end
    end
    
    return collect(states)
end

# =============================================================================
# Safe Model Call - Neural Network Wrapper
# =============================================================================

"""
    safe_model_call(model, features, parameters, states)

Safe wrapper for neural network calls with proper validation and error handling.

# Mathematical Foundation
Provides a safe interface to neural network evaluation with:
- Input validation (non-differentiable)
- Type conversion for stability
- Output validation (non-differentiable)
- Proper error handling

This function is used by all policy computations to ensure numerical stability.
"""
function safe_model_call(model, features, parameters, states)
    # Validate inputs (non-differentiable)
    Zygote.@ignore begin
        validate_neural_network_input(features, "model input features")
        if !isnothing(parameters)
            validate_model_parameters(parameters, "model parameters")
        end
    end

    # Ensure type stability - convert to Float32
    features_f32 = convert(Vector{Float32}, features)

    try
        # Neural network forward pass (differentiable)
        outputs, new_states = model(features_f32, parameters, states)

        # Validate outputs (non-differentiable)
        Zygote.@ignore validate_neural_network_output(outputs, "model outputs")

        return outputs, new_states

    catch e
        # Enhanced error reporting (non-differentiable)
        Zygote.@ignore begin
            @error "Neural network forward pass failed" exception = e
            @error "Input shape: $(size(features_f32))"
            @error "Parameter keys: $(keys(parameters))"
        end
        rethrow(e)
    end
end

# =============================================================================
# Utility Functions for State-Action Operations
# =============================================================================

"""
    get_applicable_actions_for_state(state, actions)

Get applicable actions for a specific state from a list of all actions.

# Mathematical Foundation
Returns A(s) = {a ∈ A : is_applicable(a, s)}
"""
function get_applicable_actions_for_state(state, actions)
    return [action for action in actions if is_applicable(action, state)]
end

"""
    validate_policy_consistency(model::GFlowNetModel)

Validate that all policies in the model are mathematically consistent.

# Mathematical Requirements
1. Forward policy outputs match action space size
2. Backward policy (if present) works with DAG structure
3. Flow estimator (if present) outputs scalar values
4. All policies work with the same state/action types
"""
function validate_policy_consistency(model::GFlowNetModel)
    # Test forward policy with initial state
    test_state = model.initial_state

    try
        probs = forward_action_probabilities(
            model.forward_policy, test_state, model.all_actions,
            model.parameters.forward, model.states.forward
        )

        if length(probs) != length(model.all_actions)
            throw(ArgumentError("Forward policy output size mismatch"))
        end

    catch e
        throw(ArgumentError("Forward policy validation failed: $e"))
    end

    # Test backward policy if present
    if !isnothing(model.backward_policy) && haskey(model.parameters, :backward)
        try
            # Test with a simple self-transition (may not be valid but tests the network)
            compute_backward_probability(
                model.backward_policy, test_state, test_state,
                model.parameters.backward, model.states.backward,
                model.all_actions
            )
        catch e
            # Backward policy validation is non-critical
            @debug "Backward policy validation note: $e"
        end
    end

    # Test flow estimator if present
    if !isnothing(model.flow_estimator)
        try
            flow_val = flow_estimate(
                model.flow_estimator, test_state,
                model.parameters.flow, model.states.flow
            )

            if !isfinite(flow_val) || flow_val <= 0
                throw(ArgumentError("Flow estimator produced invalid value: $flow_val"))
            end

        catch e
            throw(ArgumentError("Flow estimator validation failed: $e"))
        end
    end
end

# =============================================================================
# Display Methods
# =============================================================================

function Base.show(io::IO, policy::ForwardPolicy{M}) where {M}
    print(io, "ForwardPolicy(model=$M)")
end

function Base.show(io::IO, policy::BackwardPolicy{M}) where {M}
    print(io, "BackwardPolicy(model=$M)")
end

function Base.show(io::IO, estimator::FlowEstimator{M}) where {M}
    print(io, "FlowEstimator(model=$M)")
end
