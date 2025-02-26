using ..GFlowNet: AbstractState, BackwardPolicy
using Lux
using NNlib: softmax

"""
    create_backward_policy(input_dim::Int, hidden_dim::Int, output_dim::Int, rng=nothing)

Create a backward policy neural network for GFlowNet.
The backward policy maps states to distributions over previous states.
"""
function create_backward_policy(input_dim::Int, hidden_dim::Int, output_dim::Int, rng=nothing)
    if isnothing(rng)
        rng = Random.default_rng()
    end
    
    # Create a simple MLP for the backward policy
    model = Chain(
        Dense(input_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => output_dim)
    )
    
    # Initialize parameters
    ps, st = Lux.setup(rng, model)
    
    return BackwardPolicy(model), ps, st
end

"""
    backward_transition_logits(policy::BackwardPolicy, state::AbstractState, ps, st)

Compute the backward transition logits from a given state.
"""
function backward_transition_logits(policy::BackwardPolicy, state::AbstractState, ps, st)
    features = state_to_features(state)
    return policy.model(features, ps, st)
end

"""
    backward_action_probabilities(policy::BackwardPolicy, 
                                 state::AbstractState, 
                                 prev_states::Vector{<:AbstractState},
                                 prev_state_indices::Vector{Int},
                                 ps, st)

Compute backward action probabilities from a given state to its possible previous states.
"""
function backward_action_probabilities(policy::BackwardPolicy, 
                                      state::AbstractState, 
                                      prev_states::Vector{<:AbstractState},
                                      prev_state_indices::Vector{Int},
                                      ps, st)
    logits, st = backward_transition_logits(policy, state, ps, st)
    
    # Extract relevant logits and compute softmax
    relevant_logits = logits[prev_state_indices]
    probs = softmax(relevant_logits)
    
    return probs, st
end

"""
    sample_prev_state(policy::BackwardPolicy, 
                     state::AbstractState, 
                     prev_states::Vector{<:AbstractState},
                     prev_state_indices::Vector{Int},
                     ps, st, rng=nothing)

Sample a previous state from the backward policy distribution.
"""
function sample_prev_state(policy::BackwardPolicy, 
                          state::AbstractState, 
                          prev_states::Vector{<:AbstractState},
                          prev_state_indices::Vector{Int},
                          ps, st, rng=nothing)
    if isnothing(rng)
        rng = Random.default_rng()
    end
    
    if isempty(prev_states)
        return nothing, 0.0, st
    end
    
    # Get previous state probabilities
    probs, new_st = backward_action_probabilities(
        policy, state, prev_states, prev_state_indices, ps, st
    )
    
    # Sample previous state index
    prev_state_idx = rand(rng, Categorical(probs))
    
    # Return selected previous state, its probability, and the new state
    return prev_states[prev_state_idx], probs[prev_state_idx], new_st
end 