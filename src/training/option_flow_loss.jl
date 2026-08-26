# Option-Flow v0 losses and diagnostics

using Statistics

function catalog_cross_entropy_loss(params, catalog::OptionFlowCatalog)
    logp = option_flow_log_probs(params, catalog)
    target = catalog.target_probs
    length(logp) == length(target) || throw(ArgumentError("log prob and target length mismatch"))
    return -sum(target .* logp)
end

function mean_catalog_cross_entropy_loss(params, catalogs::Vector{OptionFlowCatalog})
    isempty(catalogs) && throw(ArgumentError("cannot compute loss on empty catalog set"))
    return mean(catalog_cross_entropy_loss(params, c) for c in catalogs)
end

function uniform_catalog_cross_entropy(catalog::OptionFlowCatalog)
    n = length(catalog.candidates)
    n > 0 || throw(ArgumentError("catalog has no candidates"))
    return log(Float32(n))
end

function catalog_kl_to_target(params, catalog::OptionFlowCatalog)
    logp = option_flow_log_probs(params, catalog)
    target = catalog.target_probs
    return sum(target .* (log.(target .+ 1.0f-12) .- logp))
end

function option_entropy(probs::AbstractVector{<:Real})
    isempty(probs) && throw(ArgumentError("probabilities cannot be empty"))
    p = Float32.(probs)
    return -sum(p .* log.(p .+ 1.0f-12))
end

function option_entropy(params, catalog::OptionFlowCatalog)
    return option_entropy(option_flow_probs(params, catalog))
end

function flow_residual_diagnostics(params, catalog::OptionFlowCatalog)
    logp = option_flow_log_probs(params, catalog)
    utilities = option_flow_utilities(catalog)
    floored = max.(0.0, utilities) .+ 1.0e-6
    log_total = log(sum(floored))
    residuals = Float64.(logp) .- log.(floored) .+ log_total
    return Dict{String,Any}(
        "residuals" => residuals,
        "mean_abs_residual" => mean(abs.(residuals)),
        "max_abs_residual" => maximum(abs.(residuals)),
    )
end

function _top_utility_indices(catalog::OptionFlowCatalog; fraction::Float64=0.25)
    fraction > 0.0 && fraction <= 1.0 || throw(ArgumentError("fraction must be in (0, 1]"))
    n = length(catalog.candidates)
    k = max(1, ceil(Int, n * fraction))
    utilities = option_flow_utilities(catalog)
    return sortperm(utilities; rev=true)[1:k]
end

function top_utility_mass(probs::AbstractVector{<:Real}, catalog::OptionFlowCatalog; fraction::Float64=0.25)
    idx = _top_utility_indices(catalog; fraction=fraction)
    return sum(Float32.(probs)[idx])
end

function top_utility_mass(params, catalog::OptionFlowCatalog; fraction::Float64=0.25)
    return top_utility_mass(option_flow_probs(params, catalog), catalog; fraction=fraction)
end

function uniform_top_utility_mass(catalog::OptionFlowCatalog; fraction::Float64=0.25)
    n = length(catalog.candidates)
    idx = _top_utility_indices(catalog; fraction=fraction)
    return length(idx) / n
end

function _rank_vector(values::AbstractVector{<:Real})
    order = sortperm(values)
    ranks = zeros(Float64, length(values))
    for (rank, idx) in enumerate(order)
        ranks[idx] = rank
    end
    return ranks
end

function rank_correlation(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    length(a) == length(b) || throw(ArgumentError("rank correlation vectors must have same length"))
    length(a) > 1 || return 0.0
    ra = _rank_vector(a)
    rb = _rank_vector(b)
    sda = std(ra)
    sdb = std(rb)
    (sda == 0.0 || sdb == 0.0) && return 0.0
    return cor(ra, rb)
end

function option_utility_rank_correlation(params, catalog::OptionFlowCatalog)
    logits = option_flow_logits(params, catalog)
    utilities = option_flow_utilities(catalog)
    return rank_correlation(Float64.(logits), utilities)
end

function evaluate_option_flow_model(params, catalogs::Vector{OptionFlowCatalog})
    isempty(catalogs) && return Dict{String,Any}(
        "n_catalogs" => 0,
        "mean_ce" => NaN,
        "mean_uniform_ce" => NaN,
        "mean_ce_vs_uniform" => NaN,
        "mean_kl" => NaN,
        "mean_entropy" => NaN,
        "mean_top_quartile_mass" => NaN,
        "mean_uniform_top_quartile_mass" => NaN,
        "mean_top_quartile_lift" => NaN,
        "mean_rank_correlation" => NaN,
    )
    ces = [catalog_cross_entropy_loss(params, c) for c in catalogs]
    uniform_ces = [uniform_catalog_cross_entropy(c) for c in catalogs]
    kls = [catalog_kl_to_target(params, c) for c in catalogs]
    entropies = [option_entropy(params, c) for c in catalogs]
    top_masses = [top_utility_mass(params, c) for c in catalogs]
    uniform_top_masses = [uniform_top_utility_mass(c) for c in catalogs]
    rank_corrs = [option_utility_rank_correlation(params, c) for c in catalogs]
    return Dict{String,Any}(
        "n_catalogs" => length(catalogs),
        "mean_ce" => mean(ces),
        "mean_uniform_ce" => mean(uniform_ces),
        "mean_ce_vs_uniform" => mean(uniform_ces) - mean(ces),
        "mean_kl" => mean(kls),
        "mean_entropy" => mean(entropies),
        "mean_top_quartile_mass" => mean(top_masses),
        "mean_uniform_top_quartile_mass" => mean(uniform_top_masses),
        "mean_top_quartile_lift" => mean(top_masses) - mean(uniform_top_masses),
        "mean_rank_correlation" => mean(rank_corrs),
    )
end
