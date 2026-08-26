#!/usr/bin/env julia

# PTA-GFN Feasibility Validation (v2)
#
# PTA-0 tests a learned batch-flow selector on frozen PMO states with exact
# same candidate pools and exact same batch-option catalogs. Evidence oracle
# calls are reported separately from any later online PMO deployment budget.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GFlowNet
using Random
using Serialization
using Statistics
using Dates
using Printf
using SHA

include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "oracle_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "training", "option_flow_dataset.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "training", "option_flow_model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "training", "option_flow_loss.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "training", "option_flow_training.jl"))

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUTDIR = joinpath(ROOT, "checkpoints", "pta_gfn_feasibility")
const PTA_SCHEMA_VERSION = "pta_batch_flow_v2"
const PTA_FEATURE_SCHEMA_VERSION = "pta_features_v1"
const AUTHORITATIVE_DATE_NOTE = "User supplied authoritative date/time: Saturday, 2026-06-20 00:02 EDT"
mkpath(OUTDIR)

const DEFAULT_BOOTSTRAP_SEEDS = [
    "CCO", "CCN", "CCC", "CCCl", "CC(=O)O", "c1ccccc1", "c1ccncc1", "CC(C)O",
]

const TASK_BOOTSTRAP_SEEDS = Dict(
    "qed" => ["CCO", "CCN", "CC(=O)O", "c1ccccc1", "CC(C)O", "COc1ccccc1"],
    "drd2" => ["N1CCCCC1", "CN1CCNCC1", "c1ccncc1", "CCN(CC)CC", "c1ccccc1Cl"],
    "gsk3b" => ["c1ccncc1", "Nc1ncccc1", "c1ccccc1", "CCN", "CC(=O)N"],
    "jnk3" => ["c1ccncc1", "Nc1ncccc1", "c1ccc(F)cc1", "CCN", "CC(=O)N"],
    "celecoxib_rediscovery" => ["Cc1ccccc1", "NS(=O)(=O)c1ccccc1", "FC(F)(F)c1ccccc1", "c1ccn[nH]1", "c1ccccc1S(N)(=O)=O"],
)

const PTA_TASK_VOCAB = ["qed", "drd2", "celecoxib_rediscovery", "gsk3b", "jnk3", "unknown_holdout"]
const PTA0_ARMS = Set([
    "uniform_batch", "proxy_parent_reward", "proxy_diverse",
    "pta_batch_flow_sample", "pta_batch_flow_greedy",
    "candidate_ranker_sample", "candidate_ranker_greedy",
    "oracle_upper",
])

mutable struct PTAState
    task::String
    seed::Int
    state_index::Int
    split::String
    history_scores::Dict{String,Float64}
    replay::SMILESReplayBuffer
    bootstrap_calls::Int
    behavior_calls::Int
end

struct PTACandidate
    id::String
    smiles::String
    tokens::Vector{Int}
    source::String
    parent_smiles::String
    parent_reward::Float64
    scaffold::String
    length_norm::Float64
end

function logmsg(msg)
    println("[", Dates.format(now(), "HH:MM:SS"), "] ", msg)
    flush(stdout)
end

function parse_env_int(name::String, default::Int)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    return parse(Int, raw)
end

function parse_env_float(name::String, default::Float64)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    return parse(Float64, raw)
end

function parse_env_bool(name::String, default::Bool)
    raw = lowercase(strip(get(ENV, name, "")))
    isempty(raw) && return default
    return raw in ("1", "true", "yes", "y", "on")
end

function parse_csv_strings(name::String, default::Vector{String})
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    return [String(strip(x)) for x in split(raw, ',') if !isempty(strip(x))]
end

function parse_csv_ints(name::String, default::Vector{Int})
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    return parse.(Int, strip.(split(raw, ',')))
end

function bounded_hash_int(x; modulus::Int=10_000_000)
    return Int(hash(x) % UInt(modulus))
end

function stable_pool_hash(ids::Vector{String})
    return bytes2hex(sha1(join(sort(ids), "|")))[1:16]
end

function clone_replay_buffer(buf::SMILESReplayBuffer)
    clone = SMILESReplayBuffer(buf.max_size)
    clone.entries = [SMILESReplayEntry(e.smiles, copy(e.tokens), e.reward) for e in buf.entries]
    clone.seen_smiles = copy(buf.seen_smiles)
    clone.needs_sort = buf.needs_sort
    clone.tb_deltas = copy(buf.tb_deltas)
    return clone
end

function topk_mean(scores::AbstractVector{<:Real}; k::Int=10)
    isempty(scores) && return 0.0, 0
    vals = sort(Float64.(scores), rev=true)
    n = min(k, length(vals))
    return mean(vals[1:n]), n
end

function history_top_stats(history_scores::Dict{String,Float64})
    vals = collect(values(history_scores))
    top10, denom = topk_mean(vals; k=10)
    return Dict{String,Any}(
        "n" => length(vals),
        "top1" => isempty(vals) ? 0.0 : maximum(vals),
        "top10_mean" => top10,
        "top10_denominator" => denom,
        "mean" => isempty(vals) ? 0.0 : mean(vals),
        "std" => length(vals) <= 1 ? 0.0 : std(vals),
    )
end

function task_onehot(task::String, vocab::Vector{String}=PTA_TASK_VOCAB)
    out = zeros(Float32, length(vocab))
    idx = findfirst(==(task), vocab)
    if idx === nothing
        unk = findfirst(==("unknown_holdout"), vocab)
        unk === nothing || (out[unk] = 1.0f0)
    else
        out[idx] = 1.0f0
    end
    return out
end

function build_reward_functions(task_name::String; cache_dir=joinpath(ROOT, "data", "tdc_cache"))
    OracleBridge.init_oracles!([task_name]; cache_dir=cache_dir)
    cache = Dict{String,Float64}()
    calls = Ref(0)

    function reward_batch(smiles_list::Vector{String})
        uncached = String[]
        seen = Set{String}()
        for smiles in smiles_list
            canonical = canonicalize_smiles_identity(smiles)
            isempty(canonical) && continue
            if !haskey(cache, canonical) && !(canonical in seen)
                push!(uncached, canonical)
                push!(seen, canonical)
            end
        end
        if !isempty(uncached)
            scores = OracleBridge.evaluate_batch(uncached, task_name)
            for (s, score) in zip(uncached, scores)
                cache[s] = Float64(score)
            end
            calls[] += length(uncached)
        end
        return Float64[get(cache, canonicalize_smiles_identity(s), 0.0) for s in smiles_list]
    end

    function reward_one(smiles::String)
        canonical = canonicalize_smiles_identity(smiles)
        haskey(cache, canonical) && return cache[canonical]
        scores = reward_batch([canonical])
        return isempty(scores) ? 0.0 : scores[1]
    end

    return reward_one, reward_batch, calls, cache
end

function add_history_entry!(history::Dict{String,Float64}, replay::SMILESReplayBuffer, vocab, smiles::String, reward::Float64)
    canonical = canonicalize_smiles_identity(smiles)
    isempty(canonical) && return false
    reward <= 0.0 && return false
    tokens = try
        encode(vocab, canonical)
    catch
        Int[]
    end
    length(tokens) < 2 && return false
    history[canonical] = max(get(history, canonical, -Inf), reward)
    add_to_replay!(replay, canonical, tokens, reward)
    return true
end

function seed_initial_state!(task::String, history::Dict{String,Float64}, replay::SMILESReplayBuffer, vocab, reward_batch)
    seeds = unique(vcat(DEFAULT_BOOTSTRAP_SEEDS, get(TASK_BOOTSTRAP_SEEDS, task, String[])))
    scores = reward_batch(seeds)
    added = 0
    for (s, r) in zip(seeds, scores)
        added += add_history_entry!(history, replay, vocab, s, r) ? 1 : 0
    end
    return added, length(seeds)
end

function safe_scaffold(smiles::String)
    s = try
        get_scaffold(smiles)
    catch
        ""
    end
    return isempty(s) ? "unknown" : s
end

function add_candidate!(cands::Vector{PTACandidate}, seen::Set{String}, vocab, history::Dict{String,Float64}, smiles::String, source::String, parent_smiles::String, parent_reward::Float64)
    canonical = canonicalize_smiles_identity(smiles)
    isempty(canonical) && return false
    canonical in seen && return false
    haskey(history, canonical) && return false
    tokens = try
        encode(vocab, canonical)
    catch
        Int[]
    end
    length(tokens) < 2 && return false
    push!(seen, canonical)
    push!(cands, PTACandidate(
        canonical,
        canonical,
        tokens,
        source,
        parent_smiles,
        Float64(parent_reward),
        safe_scaffold(canonical),
        min(length(canonical), 150) / 150.0,
    ))
    return true
end

function generate_candidate_pool(state::PTAState, vocab; pool_size::Int, rng::AbstractRNG)
    replay = state.replay
    candidates = PTACandidate[]
    seen = Set{String}()
    isempty(replay) && return candidates
    top = get_top_molecules(replay, min(30, length(replay)))

    # Deterministic source quotas: roughly balanced mutation/crossover/token fallback.
    max_attempts = max(80, pool_size * 12)
    attempts = 0
    while length(candidates) < pool_size && attempts < max_attempts
        attempts += 1
        source_draw = rand(rng)
        if source_draw < 0.50 || length(top) < 2
            parent = top[rand(rng, 1:min(15, length(top)))]
            Random.seed!(rand(rng, 1:typemax(Int32)))
            mutants = smiles_mutate_rdkit(parent.smiles; n_mutations=3)
            for child in mutants
                add_candidate!(candidates, seen, vocab, state.history_scores, child, "mutation", parent.smiles, parent.reward)
                length(candidates) >= pool_size && break
            end
        elseif source_draw < 0.85
            i = rand(rng, 1:min(15, length(top)))
            j = rand(rng, 1:min(15, length(top)))
            i == j && continue
            p1, p2 = top[i], top[j]
            Random.seed!(rand(rng, 1:typemax(Int32)))
            children = smiles_crossover_rdkit(p1.smiles, p2.smiles)
            for child in children
                add_candidate!(candidates, seen, vocab, state.history_scores, child, "crossover", string(p1.smiles, "|", p2.smiles), max(p1.reward, p2.reward))
                length(candidates) >= pool_size && break
            end
        else
            # Token mutation fallback: can create invalid molecules, filtered by tokenization/canonicalization only.
            parent = top[rand(rng, 1:min(15, length(top)))]
            mutated_tokens = smiles_mutate_tokens(parent.tokens, vocab; n_mutations=1)
            smi = try
                decode(vocab, mutated_tokens)
            catch
                ""
            end
            add_candidate!(candidates, seen, vocab, state.history_scores, smi, "token_mutation", parent.smiles, parent.reward)
        end
    end

    # Stable ordering independent of arm.
    sort!(candidates, by = c -> c.id)
    if length(candidates) > pool_size
        candidates = candidates[1:pool_size]
    end
    return candidates
end

function state_features(state::PTAState; budget_scale::Int=300)
    stats = history_top_stats(state.history_scores)
    rewards = [e.reward for e in state.replay.entries]
    scaffolds = Set{String}()
    for e in state.replay.entries[1:min(end, 64)]
        push!(scaffolds, safe_scaffold(e.smiles))
    end
    return Float32.(vcat(task_onehot(state.task), Float32[
        Float32(state.state_index) / 32.0f0,
        Float32(length(state.history_scores)) / 512.0f0,
        Float32(stats["top1"]),
        Float32(stats["top10_mean"]),
        Float32(stats["top1"] - stats["top10_mean"]),
        Float32(stats["top10_denominator"]) / 10.0f0,
        Float32(length(scaffolds)) / 128.0f0,
        isempty(rewards) ? 0.0f0 : Float32(mean(rewards)),
        length(rewards) <= 1 ? 0.0f0 : Float32(std(rewards)),
        isempty(rewards) ? 0.0f0 : Float32(maximum(rewards)),
        Float32(max(0, budget_scale - length(state.history_scores))) / Float32(max(1, budget_scale)),
    ]))
end

function candidate_features(c::PTACandidate, idx::Int, n::Int)
    return Float32[
        c.source == "mutation" ? 1.0f0 : 0.0f0,
        c.source == "crossover" ? 1.0f0 : 0.0f0,
        c.source == "token_mutation" ? 1.0f0 : 0.0f0,
        Float32(c.parent_reward),
        Float32(c.length_norm),
        Float32(idx) / Float32(max(n, 1)),
    ]
end

function batch_features(batch_indices::Vector{Int}, candidates::Vector{PTACandidate})
    batch = candidates[batch_indices]
    parent_rewards = Float64[c.parent_reward for c in batch]
    lengths = Float64[c.length_norm for c in batch]
    sources = [c.source for c in batch]
    scaffolds = Set(c.scaffold for c in batch)
    parents = Set(c.parent_smiles for c in batch)
    n = max(1, length(batch))
    return Float32[
        Float32(mean(parent_rewards)),
        Float32(maximum(parent_rewards)),
        length(parent_rewards) <= 1 ? 0.0f0 : Float32(std(parent_rewards)),
        Float32(mean(lengths)),
        length(lengths) <= 1 ? 0.0f0 : Float32(std(lengths)),
        count(==("mutation"), sources) / Float32(n),
        count(==("crossover"), sources) / Float32(n),
        count(==("token_mutation"), sources) / Float32(n),
        length(scaffolds) / Float32(n),
        length(parents) / Float32(n),
        Float32(n) / 16.0f0,
    ]
end

function make_batch_options(candidates::Vector{PTACandidate}; batch_size::Int, n_options::Int, rng::AbstractRNG)
    n = length(candidates)
    n >= batch_size || return Vector{Int}[]
    options = Vector{Int}[]
    seen = Set{String}()

    function add_option!(idxs)
        clean = sort(unique(Int.(idxs)))
        length(clean) == batch_size || return false
        key = join(clean, ",")
        key in seen && return false
        push!(seen, key)
        push!(options, clean)
        return true
    end

    # Proxy parent reward batch.
    parent_order = sortperm([c.parent_reward for c in candidates], rev=true)
    add_option!(parent_order[1:batch_size])

    # Proxy diverse batch: greedily add new scaffolds/parents with decent parent reward.
    diverse = Int[]
    used_scaffolds = Set{String}()
    for idx in parent_order
        c = candidates[idx]
        if !(c.scaffold in used_scaffolds) || length(diverse) < batch_size ÷ 2
            push!(diverse, idx)
            push!(used_scaffolds, c.scaffold)
        end
        length(diverse) >= batch_size && break
    end
    if length(diverse) < batch_size
        for idx in parent_order
            idx in diverse && continue
            push!(diverse, idx)
            length(diverse) >= batch_size && break
        end
    end
    add_option!(diverse)

    attempts = 0
    while length(options) < n_options && attempts < n_options * 50
        attempts += 1
        add_option!(shuffle(rng, collect(1:n))[1:batch_size])
    end
    return options
end

function utility_for_batch(state::PTAState, batch::Vector{PTACandidate}, reward_batch)
    before, denom_before = topk_mean(collect(values(state.history_scores)); k=10)
    smiles = [c.smiles for c in batch]
    scores = reward_batch(smiles)
    after_scores = copy(state.history_scores)
    for (s, r) in zip(smiles, scores)
        after_scores[s] = Float64(r)
    end
    after, denom_after = topk_mean(collect(values(after_scores)); k=10)
    delta = after - before
    top1_before = isempty(state.history_scores) ? 0.0 : maximum(values(state.history_scores))
    top1_after = maximum(vcat([top1_before], scores))
    return Dict{String,Any}(
        "utility" => max(0.0, delta) / max(1, length(batch)),
        "raw_delta_top10" => delta,
        "delta_top1" => top1_after - top1_before,
        "scores" => Float64.(scores),
        "top10_before" => before,
        "top10_after" => after,
        "top10_denominator_before" => denom_before,
        "top10_denominator_after" => denom_after,
    )
end

function build_batch_and_candidate_catalog(state::PTAState, vocab, reward_batch;
                                           pool_size::Int,
                                           batch_size::Int,
                                           n_batch_options::Int,
                                           rng::AbstractRNG,
                                           budget_scale::Int=300)
    candidates = generate_candidate_pool(state, vocab; pool_size=pool_size, rng=rng)
    length(candidates) >= batch_size || return nothing
    batch_options = make_batch_options(candidates; batch_size=batch_size, n_options=n_batch_options, rng=rng)
    isempty(batch_options) && return nothing

    sfeat = state_features(state; budget_scale=budget_scale)
    batch_feats = Vector{Vector{Float32}}()
    batch_utils = Float64[]
    batch_ids = String[]
    batch_meta = Dict{String,Any}[]
    batch_utilities = Dict{String,Any}[]

    for (i, idxs) in enumerate(batch_options)
        batch = candidates[idxs]
        util = utility_for_batch(state, batch, reward_batch)
        push!(batch_feats, batch_features(idxs, candidates))
        push!(batch_utils, Float64(util["utility"]))
        bid = "batch_$(state.task)_$(state.seed)_$(state.state_index)_$(i)"
        push!(batch_ids, bid)
        push!(batch_utilities, util)
        push!(batch_meta, Dict{String,Any}(
            "schema_version" => PTA_SCHEMA_VERSION,
            "feature_schema_version" => PTA_FEATURE_SCHEMA_VERSION,
            "batch_id" => bid,
            "batch_indices" => idxs,
            "candidate_ids" => [candidates[j].id for j in idxs],
            "parent_reward_mean" => mean([candidates[j].parent_reward for j in idxs]),
            "parent_reward_max" => maximum([candidates[j].parent_reward for j in idxs]),
            "unique_scaffold_fraction" => length(Set(candidates[j].scaffold for j in idxs)) / max(1, length(idxs)),
            "unique_parent_fraction" => length(Set(candidates[j].parent_smiles for j in idxs)) / max(1, length(idxs)),
            "utility_detail" => util,
        ))
    end

    snapshot_id = UInt64(hash((state.task, state.seed, state.state_index, stable_pool_hash([c.id for c in candidates]))))
    batch_catalog = make_option_flow_catalog(state.task, snapshot_id, sfeat, batch_feats, batch_utils;
        option_ids=batch_ids,
        metadata=batch_meta,
        epsilon=1.0e-6)

    cand_feats = [candidate_features(c, i, length(candidates)) for (i, c) in enumerate(candidates)]
    cand_utils = Float64[]
    cand_ids = String[]
    cand_meta = Dict{String,Any}[]
    for (i, c) in enumerate(candidates)
        util = utility_for_batch(state, [c], reward_batch)
        push!(cand_utils, Float64(util["utility"]))
        push!(cand_ids, c.id)
        push!(cand_meta, Dict{String,Any}(
            "candidate_id" => c.id,
            "source" => c.source,
            "parent_reward" => c.parent_reward,
            "scaffold" => c.scaffold,
            "utility_detail" => util,
        ))
    end
    candidate_catalog = make_option_flow_catalog(state.task, snapshot_id + UInt64(1), sfeat, cand_feats, cand_utils;
        option_ids=cand_ids,
        metadata=cand_meta,
        epsilon=1.0e-6)

    return Dict{String,Any}(
        "state" => state,
        "pool_hash" => stable_pool_hash([c.id for c in candidates]),
        "candidate_ids" => [c.id for c in candidates],
        "candidate_pool_size" => length(candidates),
        "candidates" => candidates,
        "batch_options" => batch_options,
        "batch_catalog" => batch_catalog,
        "candidate_catalog" => candidate_catalog,
        "batch_utilities" => batch_utilities,
    )
end

function choose_batch_from_candidate_scores(scores::Vector{Float64}, candidates::Vector{PTACandidate}, batch_options::Vector{Vector{Int}}; greedy::Bool, rng::AbstractRNG)
    n = length(candidates)
    if greedy
        idxs = sortperm(scores, rev=true)[1:min(8, n)]
    else
        probs = exp.(scores .- maximum(scores))
        probs = probs ./ max(sum(probs), eps(Float64))
        selected = Int[]
        available = collect(1:n)
        while length(selected) < min(8, n) && !isempty(available)
            local_probs = probs[available]
            local_probs = local_probs ./ max(sum(local_probs), eps(Float64))
            cum = cumsum(local_probs)
            pick_local = clamp(searchsortedfirst(cum, rand(rng)), 1, length(available))
            push!(selected, available[pick_local])
            deleteat!(available, pick_local)
        end
        idxs = selected
    end
    key = Set(idxs)
    best_i = 1
    best_overlap = -1
    for (i, opt) in enumerate(batch_options)
        overlap = length(intersect(key, Set(opt)))
        if overlap > best_overlap
            best_overlap = overlap
            best_i = i
        end
    end
    return best_i
end

function evaluate_catalog(catalog_bundle::Dict{String,Any}, batch_selector, candidate_selector; rng::AbstractRNG)
    batch_catalog = catalog_bundle["batch_catalog"]::OptionFlowCatalog
    candidate_catalog = catalog_bundle["candidate_catalog"]::OptionFlowCatalog
    candidates = catalog_bundle["candidates"]::Vector{PTACandidate}
    batch_options = catalog_bundle["batch_options"]::Vector{Vector{Int}}
    utils = option_flow_utilities(batch_catalog)
    metas = [c.metadata for c in batch_catalog.candidates]

    arms = Dict{String,Any}()
    uniform_idx = rand(rng, 1:length(utils))
    arms["uniform_batch"] = Dict("selected_index"=>uniform_idx, "utility"=>utils[uniform_idx], "expected_utility"=>mean(utils))

    parent_scores = [Float64(get(m, "parent_reward_mean", 0.0)) for m in metas]
    parent_idx = argmax(parent_scores)
    arms["proxy_parent_reward"] = Dict("selected_index"=>parent_idx, "utility"=>utils[parent_idx], "proxy"=>parent_scores[parent_idx])

    diverse_scores = [Float64(get(m, "unique_scaffold_fraction", 0.0)) + 0.05 * Float64(get(m, "parent_reward_mean", 0.0)) for m in metas]
    diverse_idx = argmax(diverse_scores)
    arms["proxy_diverse"] = Dict("selected_index"=>diverse_idx, "utility"=>utils[diverse_idx], "proxy"=>diverse_scores[diverse_idx])

    if !isnothing(batch_selector)
        logits = Float64.(option_flow_logits(batch_selector["params"], batch_catalog))
        probs = Float64.(option_flow_probs_from_logits(logits))
        cum = cumsum(probs ./ max(sum(probs), eps(Float64)))
        sample_idx = clamp(searchsortedfirst(cum, rand(rng)), 1, length(utils))
        greedy_idx = argmax(logits)
        arms["pta_batch_flow_sample"] = Dict("selected_index"=>sample_idx, "utility"=>utils[sample_idx], "probs"=>probs, "logits"=>logits)
        arms["pta_batch_flow_greedy"] = Dict("selected_index"=>greedy_idx, "utility"=>utils[greedy_idx], "probs"=>probs, "logits"=>logits)
    end

    if !isnothing(candidate_selector)
        cand_logits = Float64.(option_flow_logits(candidate_selector["params"], candidate_catalog))
        sample_i = choose_batch_from_candidate_scores(cand_logits, candidates, batch_options; greedy=false, rng=rng)
        greedy_i = choose_batch_from_candidate_scores(cand_logits, candidates, batch_options; greedy=true, rng=rng)
        arms["candidate_ranker_sample"] = Dict("selected_index"=>sample_i, "utility"=>utils[sample_i], "candidate_logits"=>cand_logits)
        arms["candidate_ranker_greedy"] = Dict("selected_index"=>greedy_i, "utility"=>utils[greedy_i], "candidate_logits"=>cand_logits)
    end

    oracle_idx = argmax(utils)
    arms["oracle_upper"] = Dict("selected_index"=>oracle_idx, "utility"=>utils[oracle_idx])

    return arms
end

function train_selector(catalogs::Vector{OptionFlowCatalog}; seed::Int, epochs::Int, lr::Float64, label::String)
    isempty(catalogs) && error("cannot train $(label) selector with zero catalogs")
    result = train_option_flow_model(catalogs; config=OptionFlowTrainingConfig(
        n_epochs=epochs,
        learning_rate=lr,
        hidden_dim=64,
        second_hidden_dim=32,
        validation_fraction=length(catalogs) >= 5 ? 0.2 : 0.0,
        seed=seed,
        verbose=false,
    ))
    train_metrics = evaluate_option_flow_model(result["params"], result["train_catalogs"])
    val_metrics = evaluate_option_flow_model(result["params"], result["val_catalogs"])
    logmsg("$(label) train_catalogs=$(length(result["train_catalogs"])) val_catalogs=$(length(result["val_catalogs"])) train_ce_gain=$(round(train_metrics["mean_ce_vs_uniform"], digits=4)) val_ce_gain=$(round(val_metrics["mean_ce_vs_uniform"], digits=4))")
    return merge(result, Dict{String,Any}("train_metrics"=>train_metrics, "val_metrics"=>val_metrics))
end

function advance_state_with_behavior!(state::PTAState, vocab, reward_batch; pool_size::Int, batch_size::Int, rng::AbstractRNG)
    pool = generate_candidate_pool(state, vocab; pool_size=pool_size, rng=rng)
    length(pool) < batch_size && return 0
    # Neutral behavior: half proxy parent, half random, for state diversity.
    idxs = if rand(rng) < 0.5
        sortperm([c.parent_reward for c in pool], rev=true)[1:batch_size]
    else
        shuffle(rng, collect(1:length(pool)))[1:batch_size]
    end
    batch = pool[idxs]
    scores = reward_batch([c.smiles for c in batch])
    added = 0
    for (c, r) in zip(batch, scores)
        added += add_history_entry!(state.history_scores, state.replay, vocab, c.smiles, r) ? 1 : 0
    end
    state.behavior_calls += length(unique([c.smiles for c in batch]))
    return added
end

function generate_states_for_task(task::String, seed::Int, vocab;
                                  n_train::Int, n_heldout::Int,
                                  pool_size::Int, batch_size::Int,
                                  reward_batch)
    rng = MersenneTwister(seed + bounded_hash_int(("pta-states", task)))
    history = Dict{String,Float64}()
    replay = SMILESReplayBuffer(5000)
    before_calls = 0
    added, seed_count = seed_initial_state!(task, history, replay, vocab, reward_batch)
    bootstrap_calls = seed_count
    states = PTAState[]
    total_states = n_train + n_heldout
    for sidx in 1:total_states
        split = sidx <= n_train ? "train" : "heldout"
        push!(states, PTAState(task, seed, sidx, split, copy(history), clone_replay_buffer(replay), bootstrap_calls, before_calls))
        # Advance for the next frozen state.
        temp_state = PTAState(task, seed, sidx, split, history, replay, bootstrap_calls, before_calls)
        advance_state_with_behavior!(temp_state, vocab, reward_batch; pool_size=pool_size, batch_size=batch_size, rng=rng)
        before_calls = temp_state.behavior_calls
    end
    return states, Dict{String,Any}("bootstrap_seed_count"=>seed_count, "bootstrap_entries_added"=>added)
end

function aggregate_eval_rows(rows::Vector{Dict{String,Any}})
    groups = Dict{String,Vector{Dict{String,Any}}}()
    for r in rows
        arm = String(r["arm"])
        groups[arm] = get(groups, arm, Dict{String,Any}[])
        push!(groups[arm], r)
    end
    out = Dict{String,Any}[]
    for (arm, vals) in groups
        utilities = [Float64(v["utility"]) for v in vals]
        rel_top10 = [Float64(v["raw_delta_top10"]) for v in vals]
        push!(out, Dict{String,Any}(
            "arm" => arm,
            "n" => length(vals),
            "mean_utility" => mean(utilities),
            "std_utility" => length(utilities) <= 1 ? 0.0 : std(utilities),
            "mean_raw_delta_top10" => mean(rel_top10),
        ))
    end
    sort!(out, by = r -> -Float64(r["mean_utility"]))
    return out
end

function paired_deltas(rows::Vector{Dict{String,Any}}, baseline::String)
    idx = Dict{Tuple{String,Int,Int,String},Dict{String,Any}}()
    for r in rows
        idx[(String(r["task"]), Int(r["seed"]), Int(r["state_index"]), String(r["arm"]))] = r
    end
    out = Dict{String,Any}[]
    for r in rows
        arm = String(r["arm"])
        arm == baseline && continue
        key = (String(r["task"]), Int(r["seed"]), Int(r["state_index"]), baseline)
        haskey(idx, key) || continue
        b = idx[key]
        u = Float64(r["utility"])
        bu = Float64(b["utility"])
        push!(out, Dict{String,Any}(
            "task"=>String(r["task"]), "seed"=>Int(r["seed"]), "state_index"=>Int(r["state_index"]),
            "arm"=>arm, "baseline"=>baseline, "utility"=>u, "baseline_utility"=>bu,
            "delta"=>u-bu, "relative_delta_pct"=>bu == 0.0 ? NaN : (u / bu - 1.0) * 100.0,
        ))
    end
    return out
end

function gate_decision(rows::Vector{Dict{String,Any}}, agg::Vector{Dict{String,Any}})
    means = Dict(String(r["arm"]) => Float64(r["mean_utility"]) for r in agg)
    sample = get(means, "pta_batch_flow_sample", NaN)
    uniform = get(means, "uniform_batch", NaN)
    proxy_best = maximum([get(means, "proxy_parent_reward", -Inf), get(means, "proxy_diverse", -Inf)])
    ranker_best = maximum([get(means, "candidate_ranker_sample", -Inf), get(means, "candidate_ranker_greedy", -Inf)])
    greedy = get(means, "pta_batch_flow_greedy", NaN)

    deltas_uniform = [d for d in paired_deltas(rows, "uniform_batch") if String(d["arm"]) == "pta_batch_flow_sample"]
    wins_uniform = count(d -> Float64(d["delta"]) > 0.0, deltas_uniform)
    n_pairs = length(deltas_uniform)

    reasons = String[]
    continue_gate = isfinite(sample) && isfinite(uniform) && uniform >= 0.0 &&
        sample > uniform * 1.05 && wins_uniform > n_pairs / 2 &&
        isfinite(proxy_best) && sample >= proxy_best
    if !(isfinite(sample) && isfinite(uniform))
        push!(reasons, "PTA sample or uniform result missing; incomplete.")
    elseif sample <= uniform
        push!(reasons, "pta_batch_flow_sample <= uniform_batch on held-out frozen states.")
    elseif sample <= uniform * 1.05
        push!(reasons, "pta_batch_flow_sample beats uniform but by less than +5% relative.")
    end
    if isfinite(proxy_best) && sample < proxy_best
        push!(reasons, "pta_batch_flow_sample loses best no-oracle proxy baseline.")
    end
    if isfinite(ranker_best) && sample < ranker_best
        push!(reasons, "candidate-ranker ablation beats batch-flow sample; batch-flow claim weak.")
    end
    if isfinite(greedy) && greedy > sample
        push!(reasons, "pta_batch_flow_greedy beats sample; check temperature/calibration or reinterpret as value ranker.")
    end
    if continue_gate
        push!(reasons, "PTA-0 continue gate passed; online PTA-1 may be warranted.")
    elseif isempty(reasons)
        push!(reasons, "No hard stop triggered but continue gate not met; treat as inconclusive.")
    end
    verdict = continue_gate ? "PTA0_CONTINUE_TO_ONLINE_LITE" : "PTA0_STOP_OR_REDESIGN"
    return Dict{String,Any}(
        "verdict" => verdict,
        "continue_gate" => continue_gate,
        "mean_by_arm" => means,
        "sample_vs_uniform_relative" => (isfinite(sample) && isfinite(uniform) && uniform != 0.0) ? sample / uniform - 1.0 : NaN,
        "sample_vs_proxy_best_relative" => (isfinite(sample) && isfinite(proxy_best) && proxy_best != 0.0) ? sample / proxy_best - 1.0 : NaN,
        "sample_vs_ranker_best_relative" => (isfinite(sample) && isfinite(ranker_best) && ranker_best != 0.0) ? sample / ranker_best - 1.0 : NaN,
        "paired_wins_vs_uniform" => wins_uniform,
        "paired_count_vs_uniform" => n_pairs,
        "reasons" => reasons,
    )
end

function save_bundle(path::String, bundle::Dict{String,Any})
    tmp = path * ".tmp"
    serialize(tmp, bundle)
    mv(tmp, path; force=true)
    return path
end

function main()
    mode = strip(get(ENV, "PTA_MODE", "smoke"))
    tasks = parse_csv_strings("PTA_TASKS", mode == "smoke" ? ["qed"] : ["qed", "drd2", "celecoxib_rediscovery"])
    seeds = parse_csv_ints("PTA_SEEDS", mode == "smoke" ? [17] : [17, 23])
    n_train = parse_env_int("PTA_TRAIN_STATES", mode == "smoke" ? 2 : 4)
    n_heldout = parse_env_int("PTA_HELDOUT_STATES", mode == "smoke" ? 1 : 2)
    pool_size = parse_env_int("PTA_POOL_SIZE", mode == "smoke" ? 32 : 48)
    batch_size = parse_env_int("PTA_BATCH_SIZE", 8)
    n_batch_options = parse_env_int("PTA_BATCH_OPTIONS", mode == "smoke" ? 12 : 20)
    epochs = parse_env_int("PTA_EPOCHS", mode == "smoke" ? 80 : 160)
    lr = parse_env_float("PTA_LR", 0.012)
    resume = parse_env_bool("PTA_RESUME", true)

    out_prefix = joinpath(OUTDIR, "pta_$(mode)")
    selector_path = out_prefix * "_selector_bundle.jls"
    result_path = out_prefix * "_results.jls"
    latest_path = joinpath(OUTDIR, "pta_latest_results.jls")

    vocab = SMILESVocabulary()
    logmsg("PTA-GFN feasibility mode=$(mode) tasks=$(tasks) seeds=$(seeds) train_states=$(n_train) heldout=$(n_heldout) pool=$(pool_size) batch=$(batch_size) options=$(n_batch_options)")

    all_train_batch_catalogs = OptionFlowCatalog[]
    all_train_candidate_catalogs = OptionFlowCatalog[]
    heldout_bundles = Dict{String,Any}[]
    state_records = Dict{String,Any}[]
    evidence_calls = Dict{String,Any}()

    for task in tasks, seed in seeds
        reward_one, reward_batch, calls, cache = build_reward_functions(task)
        states, bootstrap_meta = generate_states_for_task(task, seed, vocab;
            n_train=n_train, n_heldout=n_heldout,
            pool_size=pool_size,
            batch_size=batch_size,
            reward_batch=reward_batch)
        before_catalog_calls = calls[]
        for st in states
            rng = MersenneTwister(seed + bounded_hash_int(("pta-catalog", task, st.state_index, mode)))
            bundle = build_batch_and_candidate_catalog(st, vocab, reward_batch;
                pool_size=pool_size,
                batch_size=batch_size,
                n_batch_options=n_batch_options,
                rng=rng,
                budget_scale=300)
            if isnothing(bundle)
                logmsg("catalog skipped task=$(task) seed=$(seed) state=$(st.state_index) split=$(st.split)")
                continue
            end
            rec = Dict{String,Any}(
                "task"=>task, "seed"=>seed, "state_index"=>st.state_index, "split"=>st.split,
                "pool_hash"=>bundle["pool_hash"], "candidate_pool_size"=>bundle["candidate_pool_size"],
                "n_batch_options"=>length(bundle["batch_options"]),
                "history_size"=>length(st.history_scores),
                "history_stats"=>history_top_stats(st.history_scores),
            )
            push!(state_records, rec)
            if st.split == "train"
                push!(all_train_batch_catalogs, bundle["batch_catalog"])
                push!(all_train_candidate_catalogs, bundle["candidate_catalog"])
            else
                push!(heldout_bundles, bundle)
            end
        end
        evidence_calls["$(task)::$(seed)"] = Dict{String,Any}(
            "total_unique_oracle_calls" => calls[],
            "catalog_evidence_calls" => calls[] - before_catalog_calls,
            "cache_size" => length(cache),
            "bootstrap_meta" => bootstrap_meta,
        )
        logmsg("task=$(task) seed=$(seed) calls=$(calls[]) train_catalogs=$(count(r -> r["task"]==task && r["seed"]==seed && r["split"]=="train", state_records)) heldout_catalogs=$(count(r -> r["task"]==task && r["seed"]==seed && r["split"]=="heldout", state_records))")
    end

    batch_selector = nothing
    candidate_selector = nothing
    if resume && isfile(selector_path)
        selector_bundle = deserialize(selector_path)
        batch_selector = selector_bundle["batch_selector"]
        candidate_selector = selector_bundle["candidate_selector"]
        logmsg("Loaded selector bundle $(selector_path)")
    else
        batch_selector = train_selector(all_train_batch_catalogs; seed=17, epochs=epochs, lr=lr, label="batch-flow")
        candidate_selector = train_selector(all_train_candidate_catalogs; seed=23, epochs=epochs, lr=lr, label="candidate-ranker")
        save_bundle(selector_path, Dict{String,Any}(
            "created_at" => string(now()),
            "authoritative_date_note" => AUTHORITATIVE_DATE_NOTE,
            "mode" => mode,
            "schema_version" => PTA_SCHEMA_VERSION,
            "feature_schema_version" => PTA_FEATURE_SCHEMA_VERSION,
            "tasks" => tasks,
            "seeds" => seeds,
            "batch_selector" => batch_selector,
            "candidate_selector" => candidate_selector,
            "train_batch_catalog_count" => length(all_train_batch_catalogs),
            "train_candidate_catalog_count" => length(all_train_candidate_catalogs),
        ))
        logmsg("Saved selector bundle $(selector_path)")
    end

    rows = Dict{String,Any}[]
    pool_equality_records = Dict{String,Any}[]
    for bundle in heldout_bundles
        st = bundle["state"]::PTAState
        rng = MersenneTwister(st.seed + bounded_hash_int(("pta-eval", st.task, st.state_index, mode)))
        arms = evaluate_catalog(bundle, batch_selector, candidate_selector; rng=rng)
        push!(pool_equality_records, Dict{String,Any}(
            "task"=>st.task, "seed"=>st.seed, "state_index"=>st.state_index,
            "pool_hash"=>bundle["pool_hash"],
            "candidate_count"=>length(bundle["candidate_ids"]),
            "candidate_ids"=>bundle["candidate_ids"],
        ))
        for (arm, info) in arms
            idx = Int(info["selected_index"])
            catalog = bundle["batch_catalog"]::OptionFlowCatalog
            meta = catalog.candidates[idx].metadata
            util_detail = meta["utility_detail"]
            push!(rows, Dict{String,Any}(
                "task"=>st.task,
                "seed"=>st.seed,
                "state_index"=>st.state_index,
                "arm"=>arm,
                "selected_index"=>idx,
                "selected_batch_id"=>catalog.candidates[idx].option_id,
                "selected_candidate_ids"=>meta["candidate_ids"],
                "utility"=>Float64(info["utility"]),
                "uniform_expected_utility"=>get(get(arms, "uniform_batch", Dict{String,Any}()), "expected_utility", NaN),
                "raw_delta_top10"=>Float64(util_detail["raw_delta_top10"]),
                "delta_top1"=>Float64(util_detail["delta_top1"]),
                "top10_before"=>Float64(util_detail["top10_before"]),
                "top10_after"=>Float64(util_detail["top10_after"]),
                "unique_scaffold_fraction"=>Float64(get(meta, "unique_scaffold_fraction", 0.0)),
                "unique_parent_fraction"=>Float64(get(meta, "unique_parent_fraction", 0.0)),
                "pool_hash"=>bundle["pool_hash"],
            ))
        end
    end

    agg = aggregate_eval_rows(rows)
    deltas_uniform = paired_deltas(rows, "uniform_batch")
    deltas_proxy_parent = paired_deltas(rows, "proxy_parent_reward")
    deltas_proxy_diverse = paired_deltas(rows, "proxy_diverse")
    gate = gate_decision(rows, agg)

    bundle = Dict{String,Any}(
        "created_at" => string(now()),
        "authoritative_date_note" => AUTHORITATIVE_DATE_NOTE,
        "mode" => mode,
        "schema_version" => PTA_SCHEMA_VERSION,
        "feature_schema_version" => PTA_FEATURE_SCHEMA_VERSION,
        "tasks" => tasks,
        "seeds" => seeds,
        "n_train_states" => n_train,
        "n_heldout_states" => n_heldout,
        "candidate_pool_size" => pool_size,
        "batch_size" => batch_size,
        "batch_options_per_state" => n_batch_options,
        "epochs" => epochs,
        "learning_rate" => lr,
        "evidence_calls" => evidence_calls,
        "train_batch_catalog_count" => length(all_train_batch_catalogs),
        "train_candidate_catalog_count" => length(all_train_candidate_catalogs),
        "heldout_catalog_count" => length(heldout_bundles),
        "state_records" => state_records,
        "pool_equality_records" => pool_equality_records,
        "batch_selector_train_metrics" => batch_selector["train_metrics"],
        "batch_selector_val_metrics" => batch_selector["val_metrics"],
        "candidate_selector_train_metrics" => candidate_selector["train_metrics"],
        "candidate_selector_val_metrics" => candidate_selector["val_metrics"],
        "rows" => rows,
        "aggregate_rows" => agg,
        "paired_delta_vs_uniform_batch" => deltas_uniform,
        "paired_delta_vs_proxy_parent_reward" => deltas_proxy_parent,
        "paired_delta_vs_proxy_diverse" => deltas_proxy_diverse,
        "gate" => gate,
        "limitations" => [
            "PTA-0 is an offline frozen-state exact-pool object/signal test, not online PMO deployment.",
            "Catalog evidence oracle calls are reported separately and are not a fair online PMO budget.",
            "Positive PTA-0 only permits PTA-1 online-lite; it is not a SOTA claim.",
        ],
    )
    save_bundle(result_path, bundle)
    save_bundle(latest_path, bundle)

    println("\n", "="^92)
    println("PTA-GFN PTA-0 SUMMARY")
    println("="^92)
    println(rpad("Arm", 28), rpad("n", 6), rpad("Mean utility", 16), rpad("Std", 12), rpad("Mean ΔTop10", 14))
    println("-"^92)
    for r in agg
        @printf("%-28s%-6d%-16.8f%-12.8f%-14.8f\n", r["arm"], r["n"], r["mean_utility"], r["std_utility"], r["mean_raw_delta_top10"])
    end
    println("\nGATE: ", gate["verdict"])
    for reason in gate["reasons"]
        println("- ", reason)
    end
    logmsg("Saved results: $(abspath(result_path))")
    logmsg("Saved latest: $(abspath(latest_path))")

    if mode == "smoke"
        isempty(heldout_bundles) && error("PTA smoke failed: no heldout catalogs")
        isempty(rows) && error("PTA smoke failed: no evaluation rows")
    end
end

main()
