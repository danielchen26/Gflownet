using Distributions: Categorical
using StatsBase: sample, Weights
using NNlib: softmax
using Random
using Lux

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
        # Use direct flow estimation with safe model call
        features = state_to_features(state)
        
        # Use safe model call helper
        flow_values, _ = safe_model_call(
            model.flow_estimator.model,
            features,
            model.parameters.flow,
            model.states.flow
        )
        
        return flow_values[1]
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
    
    # Use safe model call helper
    logits, _ = safe_model_call(
        model.forward_policy.model,
        features,
        model.parameters.forward,
        model.states.forward
    )
    
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
        
        # Use safe model call helper
        logits, _ = safe_model_call(
            model.backward_policy.model,
            features,
            model.parameters.backward,
            model.states.backward
        )
        
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
    safe_model_call(model, features, parameters, states)

Helper function to safely call a Lux model with proper feature formatting.
Ensures features are properly shaped for Lux models and handles batch dimensions.
"""
function safe_model_call(model, features, parameters, states)
    # Convert to Float32 to ensure type stability
    features = convert(Array{Float32}, features)
    
    # Reshape features to ensure they're a matrix with correct dimensions
    # Lux expects input in the format [features, batch]
    if features isa Vector
        features = reshape(features, :, 1)
    end
    
    # Call the model using Lux's explicit function call format
    # This is safer than relying on functor behavior
    if model isa Lux.Chain
        outputs, new_states = Lux.apply(model, features, parameters, states)
    else
        outputs, new_states = model(features, parameters, states)
    end
    
    # If outputs have batch dimension of 1, flatten to a vector
    if size(outputs, 2) == 1
        outputs = vec(outputs)
    end
    
    return outputs, new_states
end

"""
    sample_trajectory(model::GFlowNetModel, rng=nothing)

Sample a complete trajectory from the GFlowNet.
"""
function sample_trajectory(model::GFlowNetModel, rng=nothing)
    if isnothing(rng)
        rng = Random.default_rng()
    end
    
    states = [model.dag.initial_state]
    actions = []
    
    current_state = model.dag.initial_state
    
    while current_state ∉ model.dag.terminal_states
        # Get next state probabilities
        features = state_to_features(current_state)
        
        # Use safe model call to get logits
        logits, _ = safe_model_call(
            model.forward_policy.model,
            features,
            model.parameters.forward,
            model.states.forward
        )
        
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
        
        push!(states, next_state)
        current_state = next_state
    end
    
    return Trajectory(states)
end 