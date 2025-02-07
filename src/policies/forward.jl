module ForwardPolicies

using Flux

export ForwardPolicy

struct ForwardPolicy
    encoder::Chain
    head::Chain
end

function ForwardPolicy(input_dim::Int, hidden_dim::Int, output_dim::Int)
    encoder = Chain(
        Dense(input_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu)
    )
    head = Chain(
        Dense(hidden_dim => output_dim, softmax)
    )
    ForwardPolicy(encoder, head)
end

end
