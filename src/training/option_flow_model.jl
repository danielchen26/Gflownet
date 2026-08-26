# Option-Flow v0 finite-catalog scoring model

using Random

struct OptionFlowMLPConfig
    input_dim::Int
    hidden_dim::Int
    second_hidden_dim::Int
end

function create_option_flow_mlp(input_dim::Int; hidden_dim::Int=128, second_hidden_dim::Int=64)
    input_dim > 0 || throw(ArgumentError("input_dim must be positive"))
    hidden_dim > 0 || throw(ArgumentError("hidden_dim must be positive"))
    second_hidden_dim > 0 || throw(ArgumentError("second_hidden_dim must be positive"))
    return OptionFlowMLPConfig(input_dim, hidden_dim, second_hidden_dim)
end

function init_option_flow_params(config::OptionFlowMLPConfig; rng::AbstractRNG=MersenneTwister(1))
    scale1 = Float32(sqrt(2.0 / max(1, config.input_dim)))
    scale2 = Float32(sqrt(2.0 / max(1, config.hidden_dim)))
    scale3 = Float32(sqrt(2.0 / max(1, config.second_hidden_dim)))
    return (
        W1 = scale1 .* randn(rng, Float32, config.hidden_dim, config.input_dim),
        b1 = zeros(Float32, config.hidden_dim, 1),
        W2 = scale2 .* randn(rng, Float32, config.second_hidden_dim, config.hidden_dim),
        b2 = zeros(Float32, config.second_hidden_dim, 1),
        W3 = scale3 .* randn(rng, Float32, 1, config.second_hidden_dim),
        b3 = zeros(Float32, 1, 1),
    )
end

function option_flow_logits(params, catalog::OptionFlowCatalog)
    X = option_flow_input_matrix(catalog)
    h1 = max.(params.W1 * X .+ params.b1, 0.0f0)
    h2 = max.(params.W2 * h1 .+ params.b2, 0.0f0)
    logits = vec(params.W3 * h2 .+ params.b3)
    return logits
end

function option_flow_log_probs_from_logits(logits::AbstractVector{<:Real})
    isempty(logits) && throw(ArgumentError("logits cannot be empty"))
    m = maximum(logits)
    z = m + log(sum(exp.(logits .- m)))
    return Float32.(logits .- z)
end

function option_flow_probs_from_logits(logits::AbstractVector{<:Real})
    logp = option_flow_log_probs_from_logits(logits)
    return Float32.(exp.(logp))
end

function option_flow_log_probs(params, catalog::OptionFlowCatalog)
    return option_flow_log_probs_from_logits(option_flow_logits(params, catalog))
end

function option_flow_probs(params, catalog::OptionFlowCatalog)
    return option_flow_probs_from_logits(option_flow_logits(params, catalog))
end

function option_flow_param_count(params)
    return sum(length, (params.W1, params.b1, params.W2, params.b2, params.W3, params.b3))
end
