# Hierarchical Controller Models
#
# Batch 1A.1 scope:
# - compare a simple linear basin regressor against a small MLP basin regressor
# - keep online selection helper generic over controller type
# - avoid AD/tooling fragility while investigating data/target semantics

using Serialization

struct LearnedBasinController
    weights::Vector{Float32}
    bias::Float32
    input_dim::Int
    feature_mode::Symbol
end

struct MLPBasinController
    W1::Matrix{Float32}
    b1::Vector{Float32}
    W2::Vector{Float32}
    b2::Float32
    input_dim::Int
    hidden_dim::Int
    feature_mode::Symbol
end

function create_learned_basin_controller(input_dim::Int;
                                         hidden_dim::Int=32,
                                         feature_mode::Symbol=:basic,
                                         rng::AbstractRNG=Random.MersenneTwister(0))
    weights = 0.01f0 .* randn(rng, Float32, input_dim)
    return LearnedBasinController(weights, 0.0f0, input_dim, feature_mode)
end

function create_mlp_basin_controller(input_dim::Int;
                                     hidden_dim::Int=32,
                                     feature_mode::Symbol=:basic,
                                     rng::AbstractRNG=Random.MersenneTwister(0))
    W1 = 0.05f0 .* randn(rng, Float32, hidden_dim, input_dim)
    b1 = zeros(Float32, hidden_dim)
    W2 = 0.05f0 .* randn(rng, Float32, hidden_dim)
    b2 = 0.0f0
    return MLPBasinController(W1, b1, W2, b2, input_dim, hidden_dim, feature_mode)
end

function basin_candidate_score(controller::LearnedBasinController,
                               features::AbstractVector{<:Real})
    x = Float32.(features)
    return dot(controller.weights, x) + controller.bias
end

function basin_candidate_score(controller::MLPBasinController,
                               features::AbstractVector{<:Real})
    x = Float32.(features)
    h = tanh.(controller.W1 * x .+ controller.b1)
    return dot(controller.W2, h) + controller.b2
end

function score_basin_candidates(controller,
                                snapshot::FrontierSnapshot,
                                candidates::Vector{ScoredBasinCandidate};
                                step_index::Int=0)
    isempty(candidates) && return Float32[]
    candidate_count = length(candidates)
    return Float32[
        basin_candidate_score(controller,
            basin_candidate_feature_vector(snapshot, candidate;
                step_index=step_index,
                candidate_count=candidate_count,
                all_candidates=candidates,
                candidate_index=idx,
                feature_mode=getfield(controller, :feature_mode)))
        for (idx, candidate) in enumerate(candidates)
    ]
end

function select_basin(controller,
                      snapshot::FrontierSnapshot,
                      candidates::Vector{ScoredBasinCandidate};
                      step_index::Int=0)
    isempty(candidates) && return nothing
    scores = score_basin_candidates(controller, snapshot, candidates; step_index=step_index)
    idx = argmax(scores)
    return candidates[idx]
end

function save_learned_basin_controller(path::String,
                                       controller)
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, controller)
    end
    return path
end

function load_learned_basin_controller(path::String)
    open(path, "r") do io
        return deserialize(io)
    end
end
