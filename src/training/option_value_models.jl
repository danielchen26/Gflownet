# Option Value Models
#
# Batch 1M scope:
# - preserve the stable linear option-value ranker from Batch 1K/1L
# - add a simple calibrated ordinal policy with a separate confidence head
# - keep serialization and scoring lightweight and deterministic

using Serialization
using Random

struct LearnedOptionValueModel
    weights::Vector{Float32}
    bias::Float32
    input_dim::Int
    feature_mode::Symbol
end

struct CalibratedOrdinalOptionPolicy
    ranking_model::LearnedOptionValueModel
    confidence_weights::Vector{Float32}
    confidence_bias::Float32
    confidence_input_dim::Int
    selection_rule::Symbol
    confidence_threshold::Float32
    confidence_low_threshold::Float32
    confidence_high_threshold::Float32
    ambiguity_threshold::Float32
    override_gain_threshold::Float32
end

function create_learned_option_value_model(input_dim::Int;
                                           feature_mode::Symbol=:augmented,
                                           rng::AbstractRNG=Random.MersenneTwister(0))
    weights = 0.01f0 .* randn(rng, Float32, input_dim)
    return LearnedOptionValueModel(weights, 0.0f0, input_dim, feature_mode)
end

function create_calibrated_ordinal_option_policy(ranking_model::LearnedOptionValueModel,
                                                 confidence_input_dim::Int;
                                                 selection_rule::Symbol=:confidence_threshold,
                                                 confidence_threshold::Float64=0.60,
                                                 confidence_low_threshold::Float64=0.45,
                                                 confidence_high_threshold::Float64=0.65,
                                                 ambiguity_threshold::Float64=0.05,
                                                 override_gain_threshold::Float64=0.02,
                                                 rng::AbstractRNG=Random.MersenneTwister(0))
    confidence_weights = 0.01f0 .* randn(rng, Float32, confidence_input_dim)
    return CalibratedOrdinalOptionPolicy(
        ranking_model,
        confidence_weights,
        0.0f0,
        confidence_input_dim,
        selection_rule,
        Float32(confidence_threshold),
        Float32(confidence_low_threshold),
        Float32(confidence_high_threshold),
        Float32(ambiguity_threshold),
        Float32(override_gain_threshold),
    )
end

function option_value_score(model::LearnedOptionValueModel,
                            features::AbstractVector{<:Real})
    x = Float32.(features)
    return dot(model.weights, x) + model.bias
end

function _sigmoid32(x::Real)
    xf = Float32(x)
    xf >= 0 ? 1.0f0 / (1.0f0 + exp(-xf)) : begin
        ex = exp(xf)
        ex / (1.0f0 + ex)
    end
end

function option_override_confidence(policy::CalibratedOrdinalOptionPolicy,
                                    features::AbstractVector{<:Real})
    x = Float32.(features)
    return _sigmoid32(dot(policy.confidence_weights, x) + policy.confidence_bias)
end

save_learned_option_value_model(path::AbstractString, model::LearnedOptionValueModel) = open(path, "w") do io
    serialize(io, model)
end

load_learned_option_value_model(path::AbstractString) = open(path, "r") do io
    deserialize(io)
end

save_calibrated_ordinal_option_policy(path::AbstractString, policy::CalibratedOrdinalOptionPolicy) = open(path, "w") do io
    serialize(io, policy)
end

load_calibrated_ordinal_option_policy(path::AbstractString) = open(path, "r") do io
    deserialize(io)
end
