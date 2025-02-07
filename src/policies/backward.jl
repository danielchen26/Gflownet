module BackwardPolicies

using Flux

export BackwardPolicy

struct BackwardPolicy
    encoder::Chain
    head::Chain
end

function BackwardPolicy(input_dim::Int, hidden_dim::Int, output_dim::Int)
    encoder = Chain(
        Dense(input_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu)
    )
    head = Chain(
        Dense(hidden_dim => output_dim, softmax)
    )
    BackwardPolicy(encoder, head)
end

end
