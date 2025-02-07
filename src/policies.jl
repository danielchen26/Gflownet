using Lux, Random, Optimisers, ComponentArrays

abstract type AbstractPolicy end

struct ForwardPolicy{P} <: AbstractPolicy
    model::Lux.Chain
    logZ::Float32
    ps::P
    st::NamedTuple
end

struct BackwardPolicy{P} <: AbstractPolicy
    model::Lux.Chain
    ps::P
    st::NamedTuple
end

function ForwardPolicy(input_dim::Int, hidden_dim::Int, output_dim::Int)
    model = Chain(
        Dense(input_dim, hidden_dim, relu),
        Dense(hidden_dim, output_dim),
        softmax
    )
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    ForwardPolicy(model, 0f0, ps, st)
end

function BackwardPolicy(input_dim::Int, hidden_dim::Int, output_dim::Int)
    model = Chain(
        Dense(input_dim, hidden_dim, relu),
        Dense(hidden_dim, output_dim),
        softmax
    )
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    BackwardPolicy(model, ps, st)
end

function action_probabilities(policy::AbstractPolicy, state)
    y, _ = policy.model(state, policy.ps, policy.st)
    return y
end 