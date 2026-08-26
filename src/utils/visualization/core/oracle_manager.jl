# Oracle Manager — Configuration, Cache, and Budget Tracking
#
# Manages oracle configurations, SMILES-level caching, and oracle call budgets.
# Used by MolecularAdapter and the PMO benchmark runner.
#
# Key design decisions:
# 1. SMILES-level cache: canonical SMILES → {oracle_name → score}
# 2. Budget counts unique graph identities only (replayed molecules don't waste budget)
# 3. Two modes: benchmark_mode (oracle-only reward) vs normal (multi-objective)

"""
    OracleConfig

Configuration for a single TDC oracle.
"""
struct OracleConfig
    name::String        # Oracle name ("DRD2", "GSK3B", "JNK3", etc.)
    weight::Float64     # Weight in multi-objective reward scalarization
end

"""
    OracleManager

Manages oracle evaluation, caching, and budget tracking.

# Fields
- `configs`: Vector of oracle configurations (name + weight)
- `budget`: Maximum oracle calls allowed (10000 for PMO benchmark)
- `calls_used`: Number of unique graph identities evaluated so far
- `cache`: canonical SMILES → {oracle_name → score} mapping
- `benchmark_mode`: If true, oracle scores are the ONLY reward (PMO-compliant)
- `top_scores`: Sorted list of top-K scores for AUC computation
"""
mutable struct OracleManager
    configs::Vector{OracleConfig}
    budget::Int
    calls_used::Int
    cache::Dict{String,Dict{String,Float64}}
    benchmark_mode::Bool
    top_scores::Vector{Float64}  # Top-K scores seen so far (for AUC top-10)
    auc_checkpoints::Vector{Float64}  # Score sampled every 100 calls
end

"""
    OracleManager(configs, budget, benchmark_mode)

Create a new OracleManager with empty cache.
"""
function OracleManager(configs::Vector{OracleConfig}, budget::Int=10000,
                       calls_used::Int=0,
                       cache::Dict{String,Dict{String,Float64}}=Dict{String,Dict{String,Float64}}(),
                       benchmark_mode::Bool=false)
    return OracleManager(configs, budget, calls_used, cache, benchmark_mode, Float64[], Float64[])
end

"""
    evaluate_molecules!(mgr, smiles_list)

Batch evaluate all configured oracles for a list of molecules.
Only evaluates uncached graph identities. Updates cache in-place.
Respects budget limits.
"""
function evaluate_molecules!(mgr::OracleManager, smiles_list::Vector{String})
    canonical_pairs = Tuple{String,String}[]
    seen = Set{String}()
    for smiles in smiles_list
        canonical = canonicalize_smiles_identity(smiles)
        isempty(canonical) && continue
        canonical in seen && continue
        push!(canonical_pairs, (canonical, smiles))
        push!(seen, canonical)
    end

    for config in mgr.configs
        uncached_pairs = Tuple{String,String}[]
        for (canonical, raw_smiles) in canonical_pairs
            oracle_cache = get(mgr.cache, canonical, nothing)
            if oracle_cache === nothing || !haskey(oracle_cache, config.name)
                push!(uncached_pairs, (canonical, raw_smiles))
            end
        end

        if isempty(uncached_pairs) || mgr.calls_used >= mgr.budget
            continue
        end

        n_affordable = min(length(uncached_pairs), mgr.budget - mgr.calls_used)
        batch_pairs = uncached_pairs[1:n_affordable]
        batch_raw = [raw for (_canonical, raw) in batch_pairs]

        scores = try
            OracleBridge.evaluate_batch(batch_raw, config.name)
        catch e
            @warn "Oracle evaluation failed for $(config.name)" exception=e
            fill(0.5, length(batch_raw))
        end

        for ((canonical, _raw_smiles), score) in zip(batch_pairs, scores)
            if !haskey(mgr.cache, canonical)
                mgr.cache[canonical] = Dict{String,Float64}()
            end
            mgr.cache[canonical][config.name] = score
        end
        mgr.calls_used += length(batch_pairs)

        if mgr.benchmark_mode
            for score in scores
                _update_top_scores!(mgr, score)
            end
            if mgr.calls_used > 0 && mgr.calls_used % 100 < length(batch_pairs)
                _record_auc_checkpoint!(mgr)
            end
        end
    end
end

"""
    lookup_score(mgr, smiles, oracle_name) → Float64

Look up a cached oracle score. Returns 0.5 (neutral) if not cached.
SMILES is canonicalized before lookup.
"""
function lookup_score(mgr::OracleManager, smiles::String, oracle_name::String)::Float64
    canonical = canonicalize_smiles_identity(smiles)
    oracle_cache = get(mgr.cache, canonical, nothing)
    oracle_cache === nothing && return 0.5
    return get(oracle_cache, oracle_name, 0.5)
end

"""
    budget_remaining(mgr) → Int

How many oracle calls remain?
"""
function budget_remaining(mgr::OracleManager)::Int
    return max(0, mgr.budget - mgr.calls_used)
end

"""
    budget_exhausted(mgr) → Bool

Whether the oracle budget is exhausted.
"""
function budget_exhausted(mgr::OracleManager)::Bool
    return mgr.calls_used >= mgr.budget
end

"""
    get_oracle_reward(mgr, smiles) → Float64

Get the combined reward for a molecule from all configured oracles.
In benchmark_mode, returns the first oracle's score directly.
Otherwise returns weighted scalarization.
"""
function get_oracle_reward(mgr::OracleManager, smiles::String)::Float64
    canonical = canonicalize_smiles_identity(smiles)
    oracle_cache = get(mgr.cache, canonical, nothing)
    if oracle_cache === nothing
        return 0.5
    end

    if mgr.benchmark_mode
        isempty(mgr.configs) && return 0.5
        return get(oracle_cache, mgr.configs[1].name, 0.5)
    end

    total_weight = sum(c.weight for c in mgr.configs)
    total_weight <= 0 && return 0.5

    reward = 0.0
    for config in mgr.configs
        reward += config.weight * get(oracle_cache, config.name, 0.5)
    end
    return reward / total_weight
end

function _update_top_scores!(mgr::OracleManager, score::Float64)
    push!(mgr.top_scores, score)
    sort!(mgr.top_scores, rev=true)
    if length(mgr.top_scores) > 10
        resize!(mgr.top_scores, 10)
    end
    return nothing
end

function _record_auc_checkpoint!(mgr::OracleManager)
    top10_mean = isempty(mgr.top_scores) ? 0.0 : sum(mgr.top_scores) / length(mgr.top_scores)
    push!(mgr.auc_checkpoints, top10_mean)
    return nothing
end

"""
    compute_auc_top10(mgr) → Float64

Compute PMO AUC top-10 score from checkpointed top-10 means.
Follows the Gao et al. NeurIPS 2022 PMO protocol.
"""
function compute_auc_top10(mgr::OracleManager)::Float64
    isempty(mgr.auc_checkpoints) && return 0.0
    return sum(mgr.auc_checkpoints) / length(mgr.auc_checkpoints)
end
