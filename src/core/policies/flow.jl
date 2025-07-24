using ..GFlowNet: AbstractState, FlowEstimator
using Lux
using NNlib: softplus

"""
    create_flow_estimator(input_dim::Int, hidden_dim::Int, rng=nothing)

Create a flow estimator neural network for GFlowNet.
The flow estimator directly predicts the flow value for a given state.
"""
function create_flow_estimator(input_dim::Int, hidden_dim::Int, rng=nothing)
    if isnothing(rng)
        rng = Random.default_rng()
    end
    
    # Create a simple MLP for the flow estimator with positive output
    model = Chain(
        Dense(input_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => 1),
        x -> softplus.(x)  # Ensure positive flow values
    )
    
    # Initialize parameters
    ps, st = Lux.setup(rng, model)
    
    return FlowEstimator(model), ps, st
end

"""
    estimate_flow(estimator::FlowEstimator, state::AbstractState, ps, st)

Estimate the flow value for a given state using the flow estimator neural network.
"""
function estimate_flow(estimator::FlowEstimator, state::AbstractState, ps, st)
    features = state_to_features(state)
    flow_value, new_st = estimator.model(features, ps, st)
    return flow_value[1], new_st  # Extract scalar value
end

"""
    estimate_edge_flow(estimator::FlowEstimator, source::AbstractState, target::AbstractState, 
                      forward_prob::Float64, ps, st)

Estimate the flow value for an edge between two states.
"""
function estimate_edge_flow(estimator::FlowEstimator, source::AbstractState, target::AbstractState, 
                           forward_prob::Float64, ps, st)
    source_flow, new_st = estimate_flow(estimator, source, ps, st)
    return source_flow * forward_prob, new_st
end 