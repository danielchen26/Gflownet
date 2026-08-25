# Oracle Manager — Configuration, Cache, and Budget Tracking
#
# Manages oracle configurations, SMILES-level caching, and oracle call budgets.
# Used by MolecularAdapter and the PMO benchmark runner.
#
# Key design decisions:
# 1. SMILES-level cache: canonical SMILES → {oracle_name → score}
# 2. Budget counts unique SMILES only (replayed molecules don't waste budget)
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
- `calls_used`: Number of unique SMILES evaluated so far
- `cache`: SMILES → {oracle_name → score} mapping
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
Only evaluates uncached SMILES. Updates cache in-place.
Respects budget limits.
"""
function evaluate_molecules!(mgr::OracleManager, smiles_list::Vector{String})
    for config in mgr.configs
        # Filter to uncached SMILES for this oracle
        uncached = String[]
        for s in smiles_list
            oracle_cache = get(mgr.cache, s, nothing)
            if oracle_cache === nothing || !haskey(oracle_cache, config.name)
                push!(uncached, s)
            end
        end
        unique_uncached = unique(uncached)

        if isempty(unique_uncached) || mgr.calls_used >= mgr.budget
            continue
        end

        # Respect budget — only evaluate what we can afford
        n_affordable = min(length(unique_uncached), mgr.budget - mgr.calls_used)
        batch = unique_uncached[1:n_affordable]

        # Single PythonCall crossing for entire batch
        scores = try
            OracleBridge.evaluate_batch(batch, config.name)
        catch e
            @warn "Oracle evaluation failed for $(config.name)" exception=e
            fill(0.5, length(batch))  # Neutral fallback
        end

        # Update cache
        for (smi, score) in zip(batch, scores)
            if !haskey(mgr.cache, smi)
                mgr.cache[smi] = Dict{String,Float64}()
            end
            mgr.cache[smi][config.name] = score
        end
        mgr.calls_used += length(batch)

        # Track top scores for AUC computation (benchmark mode)
        if mgr.benchmark_mode
            for score in scores
                _update_top_scores!(mgr, score)
            end
            # Checkpoint every 100 calls
            if mgr.calls_used > 0 && mgr.calls_used % 100 < length(batch)
                _record_auc_checkpoint!(mgr)
            end
        end
    end
end

"""
    lookup_score(mgr, smiles, oracle_name) → Float64

Look up a cached oracle score. Returns 0.5 (neutral) if not cached.
"""
function lookup_score(mgr::OracleManager, smiles::String, oracle_name::String)::Float64
    oracle_cache = get(mgr.cache, smiles, nothing)
    oracle_cache === nothing && return 0.5
    return get(oracle_cache, oracle_name, 0.5)
end

"""
    budget_remaining(mgr) → Int

Return the number of oracle calls remaining in the budget.
"""
function budget_remaining(mgr::OracleManager)::Int
    return max(0, mgr.budget - mgr.calls_used)
end

"""
    budget_exhausted(mgr) → Bool

Check if the oracle budget has been exhausted.
"""
function budget_exhausted(mgr::OracleManager)::Bool
    return mgr.calls_used >= mgr.budget
end

"""
    get_objective_names(mgr) → Vector{String}

Return the names of configured oracles.
"""
function get_objective_names(mgr::OracleManager)::Vector{String}
    return [c.name for c in mgr.configs]
end

"""
    get_objective_weights(mgr) → Vector{Float64}

Return the weights of configured oracles.
"""
function get_objective_weights(mgr::OracleManager)::Vector{Float64}
    return [c.weight for c in mgr.configs]
end

"""
    get_status(mgr) → Dict

Return oracle manager status for API responses.
"""
function get_status(mgr::OracleManager)::Dict
    return Dict(
        "configured" => get_objective_names(mgr),
        "budget_used" => mgr.calls_used,
        "budget_total" => mgr.budget,
        "budget_remaining" => budget_remaining(mgr),
        "cache_size" => length(mgr.cache),
        "benchmark_mode" => mgr.benchmark_mode,
        "n_auc_checkpoints" => length(mgr.auc_checkpoints),
    )
end

# ============================================
# AUC Top-10 Tracking (PMO Protocol)
# ============================================

"""Update the sorted top-10 scores list."""
function _update_top_scores!(mgr::OracleManager, score::Float64)
    push!(mgr.top_scores, score)
    sort!(mgr.top_scores, rev=true)
    # Keep only top-10
    if length(mgr.top_scores) > 10
        resize!(mgr.top_scores, 10)
    end
end

"""Record AUC checkpoint (mean of top-10 at current budget usage)."""
function _record_auc_checkpoint!(mgr::OracleManager)
    if isempty(mgr.top_scores)
        push!(mgr.auc_checkpoints, 0.0)
    else
        n = min(10, length(mgr.top_scores))
        push!(mgr.auc_checkpoints, sum(mgr.top_scores[1:n]) / n)
    end
end

"""
    compute_auc_top10(mgr) → Float64

Compute the AUC of top-10 average score curve.
This is the primary PMO benchmark metric.
"""
function compute_auc_top10(mgr::OracleManager)::Float64
    isempty(mgr.auc_checkpoints) && return 0.0
    # Trapezoidal integration normalized by budget
    n = length(mgr.auc_checkpoints)
    if n == 1
        return mgr.auc_checkpoints[1]
    end
    auc = 0.0
    for i in 1:(n-1)
        auc += (mgr.auc_checkpoints[i] + mgr.auc_checkpoints[i+1]) / 2.0
    end
    return auc / n
end
