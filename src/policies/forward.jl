using ..GFlowNet: AbstractState, ForwardPolicy
using Lux
using NNlib: softmax

"""
    create_forward_policy(input_dim::Int, hidden_dim::Int, output_dim::Int, rng=nothing)

Create a forward policy neural network for GFlowNet.
The forward policy maps states to distributions over next states.
"""
function create_forward_policy(input_dim::Int, hidden_dim::Int, output_dim::Int, rng=nothing)
    if isnothing(rng)
        rng = Random.default_rng()
    end
    
    # Create a simple MLP for the forward policy
    model = Chain(
        Dense(input_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => output_dim)
    )
    
    # Initialize parameters
    ps, st = Lux.setup(rng, model)
    
    return ForwardPolicy(model), ps, st
end

"""
    forward_transition_logits(policy::ForwardPolicy, state::AbstractState, ps, st)

Compute the forward transition logits from a given state.
"""
function forward_transition_logits(policy::ForwardPolicy, state::AbstractState, ps, st)
    features = state_to_features(state)
    return policy.model(features, ps, st)
end

"""
    forward_action_probabilities(policy::ForwardPolicy, 
                                state::AbstractState, 
                                next_states::Vector{<:AbstractState},
                                next_state_indices::Vector{Int},
                                ps, st)

Compute action probabilities from a given state to its possible next states.
"""
function forward_action_probabilities(policy::ForwardPolicy, 
                                     state::AbstractState, 
                                     next_states::Vector{<:AbstractState},
                                     next_state_indices::Vector{Int},
                                     ps, st)
    logits, st = forward_transition_logits(policy, state, ps, st)
    
    # Extract relevant logits and compute softmax
    relevant_logits = logits[next_state_indices]
    probs = softmax(relevant_logits)
    
    return probs, st
end

"""
    sample_action(policy::ForwardPolicy, 
                 state::AbstractState, 
                 next_states::Vector{<:AbstractState},
                 next_state_indices::Vector{Int},
                 ps, st, rng=nothing)

Sample an action from the forward policy distribution.
"""
function sample_action(policy::ForwardPolicy, 
                      state::AbstractState, 
                      next_states::Vector{<:AbstractState},
                      next_state_indices::Vector{Int},
                      ps, st, rng=nothing)
    if isnothing(rng)
        rng = Random.default_rng()
    end
    
    if isempty(next_states)
        return nothing, 0.0, st
    end
    
    # Get action probabilities
    probs, new_st = forward_action_probabilities(
        policy, state, next_states, next_state_indices, ps, st
    )
    
    # Sample action index
    action_idx = rand(rng, Categorical(probs))
    
    # Return selected action, its probability, and the new state
    return next_states[action_idx], probs[action_idx], new_st
end 