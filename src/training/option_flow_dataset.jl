# Option-Flow v0 finite-catalog dataset utilities
#
# This module implements the smallest clean object for validating the
# Option-Flow thesis: for a frozen search state S_t, learn a distribution over a
# finite catalog of realized bounded options ω_i with P(ω_i | S_t) ∝ U(ω_i; S_t).

using Random
using Statistics
using LinearAlgebra

struct OptionFlowCandidate
    task_name::String
    snapshot_id::UInt64
    option_id::String
    state_features::Vector{Float32}
    option_features::Vector{Float32}
    utility::Float64
    metadata::Dict{String,Any}
end

struct OptionFlowCatalog
    task_name::String
    snapshot_id::UInt64
    candidates::Vector{OptionFlowCandidate}
    target_probs::Vector{Float32}
    informative::Bool
end

function normalize_option_utilities(utilities::AbstractVector{<:Real};
                                    epsilon::Float64=1.0e-6,
                                    low_information_tol::Float64=1.0e-8)
    isempty(utilities) && throw(ArgumentError("cannot normalize an empty utility vector"))
    raw = Float64.(utilities)
    positive = map(u -> max(0.0, u), raw)
    informative = maximum(positive) > low_information_tol
    floored = positive .+ epsilon
    total = sum(floored)
    total <= 0.0 && throw(ArgumentError("utility normalization produced non-positive total"))
    return Float32.(floored ./ total), informative
end

function make_option_flow_catalog(task_name::String,
                                  snapshot_id::UInt64,
                                  state_features::AbstractVector{<:Real},
                                  option_features::Vector{<:AbstractVector{<:Real}},
                                  utilities::AbstractVector{<:Real};
                                  option_ids::Union{Nothing,Vector{String}}=nothing,
                                  metadata::Union{Nothing,Vector{Dict{String,Any}}}=nothing,
                                  epsilon::Float64=1.0e-6)
    n = length(option_features)
    n == length(utilities) || throw(ArgumentError("option feature count and utility count differ"))
    n > 0 || throw(ArgumentError("catalog must contain at least one option"))
    state_vec = Float32.(collect(state_features))
    option_dim = length(option_features[1])
    all(length(f) == option_dim for f in option_features) || throw(ArgumentError("all option features must have the same dimension"))

    ids = isnothing(option_ids) ? ["option-$(i)" for i in 1:n] : option_ids
    length(ids) == n || throw(ArgumentError("option_ids length must match option count"))
    metas = isnothing(metadata) ? [Dict{String,Any}() for _ in 1:n] : metadata
    length(metas) == n || throw(ArgumentError("metadata length must match option count"))

    target_probs, informative = normalize_option_utilities(utilities; epsilon=epsilon)
    candidates = OptionFlowCandidate[
        OptionFlowCandidate(task_name, snapshot_id, ids[i], copy(state_vec), Float32.(collect(option_features[i])), Float64(utilities[i]), metas[i])
        for i in 1:n
    ]
    return OptionFlowCatalog(task_name, snapshot_id, candidates, target_probs, informative)
end

function option_flow_state_dim(catalog::OptionFlowCatalog)
    isempty(catalog.candidates) && throw(ArgumentError("catalog has no candidates"))
    return length(catalog.candidates[1].state_features)
end

function option_flow_option_dim(catalog::OptionFlowCatalog)
    isempty(catalog.candidates) && throw(ArgumentError("catalog has no candidates"))
    return length(catalog.candidates[1].option_features)
end

function option_flow_input_dim(catalog::OptionFlowCatalog)
    return option_flow_state_dim(catalog) + option_flow_option_dim(catalog)
end

function option_flow_input_matrix(catalog::OptionFlowCatalog)
    isempty(catalog.candidates) && throw(ArgumentError("catalog has no candidates"))
    features = [vcat(c.state_features, c.option_features) for c in catalog.candidates]
    return reduce(hcat, features)
end

function option_flow_utilities(catalog::OptionFlowCatalog)
    return Float64[c.utility for c in catalog.candidates]
end

function validate_option_flow_catalog(catalog::OptionFlowCatalog)
    n = length(catalog.candidates)
    n > 0 || return false
    length(catalog.target_probs) == n || return false
    abs(sum(catalog.target_probs) - 1.0f0) < 1.0f-4 || return false
    sdim = option_flow_state_dim(catalog)
    odim = option_flow_option_dim(catalog)
    for c in catalog.candidates
        c.task_name == catalog.task_name || return false
        c.snapshot_id == catalog.snapshot_id || return false
        length(c.state_features) == sdim || return false
        length(c.option_features) == odim || return false
    end
    return true
end

function grouped_split_option_flow_catalogs(catalogs::Vector{OptionFlowCatalog};
                                            validation_fraction::Float64=0.25,
                                            seed::Int=17)
    isempty(catalogs) && return OptionFlowCatalog[], OptionFlowCatalog[]
    validation_fraction >= 0.0 && validation_fraction < 1.0 || throw(ArgumentError("validation_fraction must be in [0, 1)"))
    rng = MersenneTwister(seed)
    order = shuffle(rng, collect(eachindex(catalogs)))
    n_val = validation_fraction == 0.0 ? 0 : max(1, round(Int, length(catalogs) * validation_fraction))
    n_val = min(n_val, max(0, length(catalogs) - 1))
    val_idx = Set(order[1:n_val])
    train = OptionFlowCatalog[]
    val = OptionFlowCatalog[]
    for (i, catalog) in pairs(catalogs)
        if i in val_idx
            push!(val, catalog)
        else
            push!(train, catalog)
        end
    end
    return train, val
end

function synthetic_option_flow_catalogs(; n_catalogs::Int=48,
                                          n_candidates::Int=6,
                                          state_dim::Int=8,
                                          option_dim::Int=10,
                                          seed::Int=11,
                                          tasks::Vector{String}=["qed", "celecoxib_rediscovery", "drd2"])
    n_catalogs > 0 || throw(ArgumentError("n_catalogs must be positive"))
    n_candidates > 1 || throw(ArgumentError("n_candidates must be at least 2"))
    rng = MersenneTwister(seed)
    w_state = Float32.(range(-0.2, 0.25; length=state_dim))
    w_option = Float32.(range(0.6, -0.4; length=option_dim))
    catalogs = OptionFlowCatalog[]
    for k in 1:n_catalogs
        task = tasks[mod1(k, length(tasks))]
        state = randn(rng, Float32, state_dim)
        option_features = Vector{Vector{Float32}}()
        utilities = Float64[]
        for i in 1:n_candidates
            opt = randn(rng, Float32, option_dim)
            # Add a smooth, learnable signal plus a task/snapshot-specific offset.
            score = dot(w_state, state) + dot(w_option, opt) + 0.15f0 * Float32(i == mod1(k, n_candidates))
            utility = exp(Float64(score))
            push!(option_features, opt)
            push!(utilities, utility)
        end
        push!(catalogs, make_option_flow_catalog(task, UInt64(k), state, option_features, utilities;
            option_ids=["synthetic-$(k)-$(i)" for i in 1:n_candidates]))
    end
    return catalogs
end

function option_flow_catalog_stats(catalogs::Vector{OptionFlowCatalog})
    if isempty(catalogs)
        return Dict{String,Any}(
            "n_catalogs" => 0,
            "n_candidates" => 0,
            "n_informative" => 0,
        )
    end
    candidate_counts = [length(c.candidates) for c in catalogs]
    utilities = reduce(vcat, [option_flow_utilities(c) for c in catalogs])
    return Dict{String,Any}(
        "n_catalogs" => length(catalogs),
        "n_candidates" => sum(candidate_counts),
        "mean_candidates_per_catalog" => mean(candidate_counts),
        "n_informative" => count(c -> c.informative, catalogs),
        "mean_utility" => mean(utilities),
        "max_utility" => maximum(utilities),
        "min_utility" => minimum(utilities),
    )
end
