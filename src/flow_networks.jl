using Distributions: Categorical
using StatsBase: sample, Weights
using NNlib: softmax

"""
    state_to_features(state::AbstractState)

Convert a state to a feature vector. Should be implemented by concrete types.
"""
function state_to_features end

"""
    reward(state::AbstractState)

Calculate the reward for a state. Should be implemented by concrete types.
"""
function reward end

"""
    flow(model::GFlowNetModel, state::AbstractState)

Compute the flow value for a given state.
"""
function flow(model::GFlowNetModel, state::AbstractState)
    if !isnothing(model.flow_estimator)
        # Use direct flow estimation
        return model.flow_estimator.model(state_to_features(state))[1]
    else
        # Compute flow by summing incoming edge flows
        incoming_edges = get_incoming_edges(model.dag, state)
        if isempty(incoming_edges)
            # Initial state
            if state == model.dag.initial_state
                if isnothing(model.partition_function)
                    # Estimate partition function
                    return estimate_partition_function(model)
                else
                    return model.partition_function
                end
            else
                return 0.0
            end
        end
        
        total_flow = 0.0
        for (prev_state, _) in incoming_edges
            edge_flow = flow(model, prev_state) * 
                        forward_transition_prob(model, prev_state, state)
            total_flow += edge_flow
        end
        return total_flow
    end
end

"""
    edge_flow(model::GFlowNetModel, source::AbstractState, target::AbstractState)

Compute the flow value for an edge between two states.
"""
function edge_flow(model::GFlowNetModel, source::AbstractState, target::AbstractState)
    return flow(model, source) * forward_transition_prob(model, source, target)
end

"""
    forward_transition_prob(model::GFlowNetModel, source::AbstractState, target::AbstractState)

Compute the forward transition probability from source to target state.
"""
function forward_transition_prob(model::GFlowNetModel, source::AbstractState, target::AbstractState)
    # Use the forward policy to compute transition probability
    features = state_to_features(source)
    logits = model.forward_policy.model(features)
    
    # Get all possible next states from source
    next_states = get_next_states(model.dag, source)
    if isempty(next_states) || target ∉ next_states
        return 0.0
    end
    
    next_state_indices = [model.dag.state_to_idx[s] for s in next_states]
    
    # Extract relevant logits and compute softmax
    relevant_logits = logits[next_state_indices]
    probs = softmax(relevant_logits)
    
    # Find target state index
    target_idx = findfirst(s -> s == target, next_states)
    if isnothing(target_idx)
        return 0.0
    else
        return probs[target_idx]
    end
end

"""
    backward_transition_prob(model::GFlowNetModel, target::AbstractState, source::AbstractState)

Compute the backward transition probability from target to source state.
"""
function backward_transition_prob(model::GFlowNetModel, target::AbstractState, source::AbstractState)
    if isnothing(model.backward_policy)
        # If no backward policy is defined, compute from flow and forward policy
        source_flow = flow(model, source)
        if source_flow ≈ 0.0
            return 0.0
        end
        edge_f = edge_flow(model, source, target)
        target_flow = flow(model, target)
        if target_flow ≈ 0.0
            return 0.0
        end
        return edge_f / target_flow
    else
        # Use the backward policy to compute transition probability
        features = state_to_features(target)
        logits = model.backward_policy.model(features)
        
        # Get all possible previous states from target
        prev_states = get_previous_states(model.dag, target)
        if isempty(prev_states) || source ∉ prev_states
            return 0.0
        end
        
        prev_state_indices = [model.dag.state_to_idx[s] for s in prev_states]
        
        # Extract relevant logits and compute softmax
        relevant_logits = logits[prev_state_indices]
        probs = softmax(relevant_logits)
        
        # Find source state index
        source_idx = findfirst(s -> s == source, prev_states)
        if isnothing(source_idx)
            return 0.0
        else
            return probs[source_idx]
        end
    end
end

"""
    estimate_partition_function(model::GFlowNetModel)

Estimate the partition function (total flow) by summing rewards over terminal states.
"""
function estimate_partition_function(model::GFlowNetModel)
    total = 0.0
    for state in model.dag.terminal_states
        total += reward(state)
    end
    return total
end

"""
    sample_trajectory(model::GFlowNetModel)

Sample a complete trajectory from the GFlowNet.
"""
function sample_trajectory(model::GFlowNetModel)
    trajectory = [model.dag.initial_state]
    current_state = model.dag.initial_state
    
    while current_state ∉ model.dag.terminal_states
        # Get next state probabilities
        features = state_to_features(current_state)
        logits = model.forward_policy.model(features)
        
        # Get all possible next states
        next_states = get_next_states(model.dag, current_state)
        if isempty(next_states)
            break
        end
        
        next_state_indices = [model.dag.state_to_idx[s] for s in next_states]
        relevant_logits = logits[next_state_indices]
        probs = softmax(relevant_logits)
        
        # Sample next state
        next_state_idx = sample(1:length(next_states), Weights(probs))
        next_state = next_states[next_state_idx]
        
        push!(trajectory, next_state)
        current_state = next_state
    end
    
    return Trajectory(trajectory)
end 