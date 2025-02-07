# src/model.jl

# This file defines the model structure for the GFlowNet project.

module Models

using ..ForwardPolicies, ..BackwardPolicies
using Flux: Chain, Dense

export GFlowNet

struct GFlowNet
    forward::ForwardPolicy
    backward::BackwardPolicy
    value_head::Chain
end

function GFlowNet(input_dim::Int, hidden_dim::Int, output_dim::Int)
    forward = ForwardPolicies.ForwardPolicy(input_dim, hidden_dim, output_dim)
    backward = BackwardPolicies.BackwardPolicy(input_dim, hidden_dim, output_dim)
    value_head = Chain(Dense(hidden_dim => 1))
    GFlowNet(forward, backward, value_head)
end

end
