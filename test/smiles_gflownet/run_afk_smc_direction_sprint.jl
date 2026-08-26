#!/usr/bin/env julia

# AFK-SMC Direction Sprint (v2)
#
# Stage 0 validates an Adaptive Feynman-Kac SMC rare-event population-search
# core for PMO. This is intentionally NOT called AFK-GFN yet: no learned
# GFlowNet twisted proposal is used in this runner.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GFlowNet
using Random
using Serialization
using Statistics
using Dates
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "oracle_bridge.jl"))

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUTDIR = joinpath(ROOT, "checkpoints", "afk_smc_direction")
const AFK_SCHEMA_VERSION = "afk_smc_core_v2"
const AUTHORITATIVE_DATE_NOTE = "User supplied authoritative date/time: Saturday, 2026-06-20 00:54 EDT"
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

const DEFAULT_ARMS = ["uniform_population", "elite_ga", "rank_weighted_ga", "fk_fixed_beta", "fk_adaptive_no_diversity", "afk_smc"]
const FK_ARMS = Set(["fk_fixed_beta", "fk_adaptive_no_diversity", "afk_smc"])

struct AFKParticle
    id::String
    smiles::String
    reward::Float64
    scaffold::String
    genealogy::String
    generation::Int
    source::String
    parent_id::String
    logw::Float64
end

mutable struct AFKState
    task::String
    arm::String
    seed::Int
    budget::Int
    calls_used::Int
    history::Dict{String,Float64}
    oracle_cache::Dict{String,Float64}
    population::Vector{AFKParticle}
    curve::Vector{Dict{String,Any}}
    attempt_ledger::Dict{String,Int}
    ess_curve::Vector{Dict{String,Any}}
    beta_total::Float64
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

function safe_scaffold(smiles::String)
    s = try
        get_scaffold(smiles)
    catch
        ""
    end
    return isempty(s) ? "unknown" : s
end

function topk_mean(scores::AbstractVector{<:Real}; k::Int=10)
    isempty(scores) && return 0.0, 0
    vals = sort(Float64.(scores), rev=true)
    n = min(k, length(vals))
    return mean(vals[1:n]), n
end

function top_stats(history::Dict{String,Float64})
    vals = collect(values(history))
    top10, denom = topk_mean(vals; k=10)
    return Dict{String,Any}(
        "n" => length(vals),
        "top1" => isempty(vals) ? 0.0 : maximum(vals),
        "top10_mean" => top10,
        "top10_denominator" => denom,
    )
end

function record_curve!(state::AFKState; generation::Int, event::String)
    stats = top_stats(state.history)
    push!(state.curve, Dict{String,Any}(
        "calls" => state.calls_used,
        "generation" => generation,
        "event" => event,
        "top1" => stats["top1"],
        "top10_mean" => stats["top10_mean"],
        "top10_denominator" => stats["top10_denominator"],
        "unique_evaluated" => length(state.history),
    ))
end

function normalized_auc(curve::Vector{Dict{String,Any}}, budget::Int; metric::String="top10_mean")
    isempty(curve) && return 0.0
    ordered = sort(curve, by = r -> Int(r["calls"]))
    # Ensure call 0 origin exists.
    if Int(ordered[1]["calls"]) > 0
        ordered = vcat([Dict{String,Any}("calls"=>0, metric=>0.0)], ordered)
    end
    area = 0.0
    prev_x = 0.0
    prev_y = 0.0
    for row in ordered
        x = min(Float64(Int(row["calls"])), Float64(budget))
        y = Float64(get(row, metric, 0.0))
        x >= prev_x || continue
        area += (x - prev_x) * (prev_y + y) / 2.0
        prev_x = x
        prev_y = y
        prev_x >= budget && break
    end
    if prev_x < budget
        area += (Float64(budget) - prev_x) * prev_y
    end
    return area / max(1.0, Float64(budget))
end

function increment!(d::Dict{String,Int}, key::String, n::Int=1)
    d[key] = get(d, key, 0) + n
    return nothing
end

function valid_tokenizable(vocab, smiles::String)
    try
        tokens = encode(vocab, smiles)
        return length(tokens) >= 2
    catch
        return false
    end
end

function add_particle!(state::AFKState, vocab, smiles::String, reward::Float64;
                       generation::Int, source::String, parent_id::String, genealogy::String, logw::Float64=0.0)
    canonical = canonicalize_smiles_identity(smiles)
    isempty(canonical) && return false
    reward <= 0.0 && return false
    valid_tokenizable(vocab, canonical) || return false
    old = get(state.history, canonical, -Inf)
    state.history[canonical] = max(old, reward)
    id = "$(state.task)::$(canonical)"
    push!(state.population, AFKParticle(id, canonical, reward, safe_scaffold(canonical), genealogy, generation, source, parent_id, logw))
    return true
end

function evaluate_unique!(state::AFKState, vocab, smiles_list::Vector{String}; generation::Int, source::String, parent_ids::Vector{String}, genealogies::Vector{String})
    remaining = state.budget - state.calls_used
    remaining <= 0 && return 0
    eval_smiles = String[]
    eval_parent_ids = String[]
    eval_genealogies = String[]
    seen = Set{String}()
    for (idx, smi) in enumerate(smiles_list)
        canonical = canonicalize_smiles_identity(smi)
        if isempty(canonical)
            increment!(state.attempt_ledger, "invalid_canonical")
            continue
        end
        if canonical in seen
            increment!(state.attempt_ledger, "duplicate_within_generation")
            continue
        end
        if haskey(state.history, canonical) || haskey(state.oracle_cache, canonical)
            increment!(state.attempt_ledger, "already_evaluated")
            continue
        end
        if !valid_tokenizable(vocab, canonical)
            increment!(state.attempt_ledger, "tokenization_failed")
            continue
        end
        push!(seen, canonical)
        push!(eval_smiles, canonical)
        push!(eval_parent_ids, parent_ids[min(idx, length(parent_ids))])
        push!(eval_genealogies, genealogies[min(idx, length(genealogies))])
        length(eval_smiles) >= remaining && break
    end
    isempty(eval_smiles) && return 0
    scores = OracleBridge.evaluate_batch(eval_smiles, state.task)
    for (i, (s, r)) in enumerate(zip(eval_smiles, scores))
        state.calls_used += 1
        state.oracle_cache[s] = Float64(r)
        add_particle!(state, vocab, s, Float64(r);
            generation=generation,
            source=source,
            parent_id=eval_parent_ids[i],
            genealogy=eval_genealogies[i],
            logw=0.0)
        record_curve!(state; generation=generation, event=source)
    end
    increment!(state.attempt_ledger, "oracle_evaluated_unique_valid", length(eval_smiles))
    return length(eval_smiles)
end

function initialize_state(task::String, arm::String, seed::Int, budget::Int, vocab)
    OracleBridge.init_oracles!([task]; cache_dir=joinpath(ROOT, "data", "tdc_cache"))
    state = AFKState(task, arm, seed, budget, 0,
        Dict{String,Float64}(), Dict{String,Float64}(), AFKParticle[], Dict{String,Any}[], Dict{String,Int}(), Dict{String,Any}[], 0.0)
    seeds = unique(vcat(DEFAULT_BOOTSTRAP_SEEDS, get(TASK_BOOTSTRAP_SEEDS, task, String[])))
    parent_ids = ["bootstrap" for _ in seeds]
    genealogies = ["seed_$(i)" for i in eachindex(seeds)]
    evaluate_unique!(state, vocab, seeds; generation=0, source="bootstrap", parent_ids=parent_ids, genealogies=genealogies)
    increment!(state.attempt_ledger, "bootstrap_seed_count", length(seeds))
    return state
end

function sample_index_from_probs(rng::AbstractRNG, probs::Vector{Float64})
    total = sum(probs)
    total <= 0.0 && return rand(rng, 1:length(probs))
    cum = cumsum(probs ./ total)
    return clamp(searchsortedfirst(cum, rand(rng)), 1, length(probs))
end

function parent_probs(pop::Vector{AFKParticle}, arm::String)
    n = length(pop)
    n == 0 && return Float64[]
    if arm == "uniform_population" || arm in FK_ARMS
        return fill(1.0 / n, n)
    elseif arm == "elite_ga"
        order = sortperm([p.reward for p in pop], rev=true)
        elite_n = max(1, ceil(Int, n / 4))
        probs = zeros(Float64, n)
        for idx in order[1:elite_n]
            probs[idx] = 1.0 / elite_n
        end
        return probs
    elseif arm == "rank_weighted_ga"
        order = sortperm([p.reward for p in pop], rev=true)
        probs = zeros(Float64, n)
        weights = [1.0 / i for i in 1:n]
        weights ./= sum(weights)
        for (rank, idx) in enumerate(order)
            probs[idx] = weights[rank]
        end
        return probs
    else
        return fill(1.0 / n, n)
    end
end

function choose_parent(pop::Vector{AFKParticle}, arm::String, rng::AbstractRNG)
    probs = parent_probs(pop, arm)
    idx = sample_index_from_probs(rng, probs)
    return pop[idx]
end

function generate_offspring!(state::AFKState, vocab, rng::AbstractRNG;
                             children_per_generation::Int, attempt_multiplier::Int, generation::Int)
    candidates = String[]
    parent_ids = String[]
    genealogies = String[]
    max_attempts = attempt_multiplier * children_per_generation
    attempts = 0
    seen = Set{String}()
    while length(candidates) < children_per_generation && attempts < max_attempts && !isempty(state.population)
        attempts += 1
        increment!(state.attempt_ledger, "attempted_proposals")
        draw = rand(rng)
        if draw < 0.55 || length(state.population) < 2
            p = choose_parent(state.population, state.arm, rng)
            Random.seed!(rand(rng, 1:typemax(Int32)))
            children = smiles_mutate_rdkit(p.smiles; n_mutations=3)
            if isempty(children)
                increment!(state.attempt_ledger, "empty_mutation")
            end
            for child in children
                canonical = canonicalize_smiles_identity(child)
                if isempty(canonical) || canonical in seen || haskey(state.history, canonical)
                    increment!(state.attempt_ledger, "filtered_candidate")
                    continue
                end
                push!(seen, canonical)
                push!(candidates, canonical)
                push!(parent_ids, p.id)
                push!(genealogies, p.genealogy)
                length(candidates) >= children_per_generation && break
            end
        elseif draw < 0.90
            p1 = choose_parent(state.population, state.arm, rng)
            p2 = choose_parent(state.population, state.arm, rng)
            p1.id == p2.id && continue
            Random.seed!(rand(rng, 1:typemax(Int32)))
            children = smiles_crossover_rdkit(p1.smiles, p2.smiles)
            if isempty(children)
                increment!(state.attempt_ledger, "empty_crossover")
            end
            for child in children
                canonical = canonicalize_smiles_identity(child)
                if isempty(canonical) || canonical in seen || haskey(state.history, canonical)
                    increment!(state.attempt_ledger, "filtered_candidate")
                    continue
                end
                push!(seen, canonical)
                push!(candidates, canonical)
                push!(parent_ids, string(p1.id, "|", p2.id))
                # Preserve dominant genealogy by reward.
                push!(genealogies, p1.reward >= p2.reward ? p1.genealogy : p2.genealogy)
                length(candidates) >= children_per_generation && break
            end
        else
            p = choose_parent(state.population, state.arm, rng)
            toks = try encode(vocab, p.smiles) catch; Int[] end
            if length(toks) >= 2
                mtoks = smiles_mutate_tokens(toks, vocab; n_mutations=1)
                smi = try decode(vocab, mtoks) catch; "" end
                canonical = canonicalize_smiles_identity(smi)
                if !isempty(canonical) && !(canonical in seen) && !haskey(state.history, canonical)
                    push!(seen, canonical)
                    push!(candidates, canonical)
                    push!(parent_ids, p.id)
                    push!(genealogies, p.genealogy)
                else
                    increment!(state.attempt_ledger, "filtered_candidate")
                end
            else
                increment!(state.attempt_ledger, "token_fallback_failed")
            end
        end
    end
    if length(candidates) < children_per_generation
        increment!(state.attempt_ledger, "underfilled_generations")
    end
    return candidates, parent_ids, genealogies
end

function normalize_logweights(logw::Vector{Float64})
    isempty(logw) && return Float64[]
    m = maximum(logw)
    w = exp.(logw .- m)
    total = sum(w)
    total <= 0.0 && return fill(1.0 / length(logw), length(logw))
    return w ./ total
end

function ess_fraction_from_logweights(logw::Vector{Float64})
    w = normalize_logweights(logw)
    isempty(w) && return 0.0
    ess = 1.0 / sum(w .^ 2)
    return ess / length(w)
end

function tail_scores(pop::Vector{AFKParticle}; tau::Float64=0.75, smax::Float64=5.0)
    rewards = Float64[p.reward for p in pop]
    if length(rewards) <= 1 || maximum(rewards) == minimum(rewards)
        return zeros(Float64, length(rewards)), Dict("q"=>isempty(rewards) ? 0.0 : rewards[1], "scale"=>1.0, "zero_variance"=>true)
    end
    sorted = sort(rewards)
    qidx = clamp(ceil(Int, tau * length(sorted)), 1, length(sorted))
    q = sorted[qidx]
    q25 = sorted[clamp(ceil(Int, 0.25 * length(sorted)), 1, length(sorted))]
    q75 = sorted[clamp(ceil(Int, 0.75 * length(sorted)), 1, length(sorted))]
    scale = max(q75 - q25, 1.0e-6)
    scores = clamp.((rewards .- q) ./ scale, -smax, smax)
    return scores, Dict("q"=>q, "scale"=>scale, "zero_variance"=>false)
end

function choose_delta_beta(pop::Vector{AFKParticle}, scores::Vector{Float64}, arm::String;
                           target_ess::Float64=0.6, delta_max::Float64=4.0, fixed_delta::Float64=1.0)
    arm == "fk_fixed_beta" && return fixed_delta, ess_fraction_from_logweights([p.logw + fixed_delta * scores[i] for (i,p) in enumerate(pop)])
    if isempty(scores) || maximum(scores) == minimum(scores)
        return 0.0, ess_fraction_from_logweights([p.logw for p in pop])
    end
    base = [p.logw for p in pop]
    hi_ess = ess_fraction_from_logweights(base .+ delta_max .* scores)
    if hi_ess > target_ess
        return delta_max, hi_ess
    end
    lo = 0.0
    hi = delta_max
    for _ in 1:32
        mid = (lo + hi) / 2.0
        ess = ess_fraction_from_logweights(base .+ mid .* scores)
        if ess > target_ess
            lo = mid
        else
            hi = mid
        end
    end
    delta = (lo + hi) / 2.0
    return delta, ess_fraction_from_logweights(base .+ delta .* scores)
end

function unique_by_id_best(pop::Vector{AFKParticle})
    best = Dict{String,AFKParticle}()
    for p in pop
        if !haskey(best, p.id) || p.reward > best[p.id].reward
            best[p.id] = p
        end
    end
    return collect(values(best))
end

function diversity_metrics(pop::Vector{AFKParticle})
    n = length(pop)
    if n == 0
        return Dict{String,Any}("n"=>0, "scaffold_entropy"=>0.0, "genealogy_entropy"=>0.0, "max_scaffold_fraction"=>0.0, "max_genealogy_fraction"=>0.0, "n_scaffolds"=>0, "n_genealogies"=>0)
    end
    function counts(vals)
        d = Dict{String,Int}()
        for v in vals
            d[v] = get(d, v, 0) + 1
        end
        return d
    end
    function norm_entropy(d)
        total = sum(values(d))
        length(d) <= 1 && return 0.0
        ps = [c / total for c in values(d)]
        return -sum(p * log(p) for p in ps if p > 0) / log(max(2, length(d)))
    end
    sc = counts([p.scaffold for p in pop])
    gc = counts([p.genealogy for p in pop])
    return Dict{String,Any}(
        "n" => n,
        "scaffold_entropy" => norm_entropy(sc),
        "genealogy_entropy" => norm_entropy(gc),
        "max_scaffold_fraction" => maximum(values(sc)) / n,
        "max_genealogy_fraction" => maximum(values(gc)) / n,
        "n_scaffolds" => length(sc),
        "n_genealogies" => length(gc),
    )
end

function diversity_guard_select(candidates::Vector{AFKParticle}, N::Int; max_fraction::Float64=0.70)
    uniq = unique_by_id_best(candidates)
    sort!(uniq, by = p -> -p.reward)
    max_count = max(1, floor(Int, max_fraction * N))
    selected = AFKParticle[]
    sc_counts = Dict{String,Int}()
    ge_counts = Dict{String,Int}()
    for p in uniq
        get(sc_counts, p.scaffold, 0) >= max_count && continue
        get(ge_counts, p.genealogy, 0) >= max_count && continue
        push!(selected, p)
        sc_counts[p.scaffold] = get(sc_counts, p.scaffold, 0) + 1
        ge_counts[p.genealogy] = get(ge_counts, p.genealogy, 0) + 1
        length(selected) >= N && break
    end
    if length(selected) < N
        for p in uniq
            any(q -> q.id == p.id, selected) && continue
            push!(selected, p)
            length(selected) >= N && break
        end
    end
    return selected
end

function update_population!(state::AFKState, generation::Int; population_size::Int, target_ess::Float64, delta_max::Float64, fixed_delta::Float64)
    pop = state.population
    isempty(pop) && return
    if state.arm == "uniform_population"
        uniq = unique_by_id_best(pop)
        if length(uniq) > population_size
            state.population = shuffle(MersenneTwister(state.seed + generation), uniq)[1:population_size]
        else
            state.population = uniq
        end
        return
    elseif state.arm == "elite_ga"
        uniq = unique_by_id_best(pop)
        sort!(uniq, by = p -> -p.reward)
        state.population = uniq[1:min(population_size, length(uniq))]
        return
    elseif state.arm == "rank_weighted_ga"
        uniq = unique_by_id_best(pop)
        sort!(uniq, by = p -> -p.reward)
        n = length(uniq)
        weights = [1.0 / i for i in 1:n]
        weights ./= sum(weights)
        rng = MersenneTwister(state.seed + 10_000 + generation)
        selected = AFKParticle[]
        for _ in 1:min(population_size, n)
            push!(selected, uniq[sample_index_from_probs(rng, weights)])
        end
        state.population = selected
        return
    elseif state.arm in FK_ARMS
        scores, score_meta = tail_scores(pop)
        delta, ess_after = choose_delta_beta(pop, scores, state.arm; target_ess=target_ess, delta_max=delta_max, fixed_delta=fixed_delta)
        new_logw = [p.logw + delta * scores[i] for (i,p) in enumerate(pop)]
        probs = normalize_logweights(new_logw)
        weighted = [AFKParticle(p.id, p.smiles, p.reward, p.scaffold, p.genealogy, p.generation, p.source, p.parent_id, new_logw[i]) for (i,p) in enumerate(pop)]
        rng = MersenneTwister(state.seed + 20_000 + generation + bounded_hash_int(state.arm))
        selected = AFKParticle[]
        for _ in 1:min(population_size, length(weighted))
            push!(selected, weighted[sample_index_from_probs(rng, probs)])
        end
        if state.arm == "afk_smc"
            selected = diversity_guard_select(vcat(selected, weighted), population_size)
        end
        state.population = selected
        state.beta_total += delta
        push!(state.ess_curve, Dict{String,Any}(
            "generation" => generation,
            "delta_beta" => delta,
            "beta_total" => state.beta_total,
            "ess_fraction_after" => ess_after,
            "score_meta" => score_meta,
            "diversity" => diversity_metrics(state.population),
        ))
        return
    end
end

function run_arm(task::String, arm::String, seed::Int, vocab;
                 budget::Int, population_size::Int, children_per_generation::Int,
                 attempt_multiplier::Int, target_ess::Float64, delta_max::Float64,
                 fixed_delta::Float64, verbose::Bool)
    rng = MersenneTwister(seed + bounded_hash_int((task, arm, AFK_SCHEMA_VERSION)))
    state = initialize_state(task, arm, seed, budget, vocab)
    update_population!(state, 0; population_size=population_size, target_ess=target_ess, delta_max=delta_max, fixed_delta=fixed_delta)
    generation = 0
    while state.calls_used < budget && !isempty(state.population)
        generation += 1
        candidates, parent_ids, genealogies = generate_offspring!(state, vocab, rng;
            children_per_generation=children_per_generation,
            attempt_multiplier=attempt_multiplier,
            generation=generation)
        if isempty(candidates)
            increment!(state.attempt_ledger, "empty_candidate_generations")
            break
        end
        n_eval = evaluate_unique!(state, vocab, candidates;
            generation=generation,
            source="offspring",
            parent_ids=parent_ids,
            genealogies=genealogies)
        n_eval == 0 && break
        update_population!(state, generation;
            population_size=population_size,
            target_ess=target_ess,
            delta_max=delta_max,
            fixed_delta=fixed_delta)
        generation > 10_000 && break
    end
    final_stats = top_stats(state.history)
    auc = normalized_auc(state.curve, budget; metric="top10_mean")
    div = diversity_metrics(state.population)
    return Dict{String,Any}(
        "task" => task,
        "arm" => arm,
        "seed" => seed,
        "budget" => budget,
        "calls_used" => state.calls_used,
        "generations" => generation,
        "auc_top10" => auc,
        "final_top1" => final_stats["top1"],
        "final_top10_mean" => final_stats["top10_mean"],
        "top10_denominator" => final_stats["top10_denominator"],
        "unique_evaluated" => length(state.history),
        "attempt_ledger" => state.attempt_ledger,
        "curve" => state.curve,
        "ess_curve" => state.ess_curve,
        "diversity" => div,
        "schema_version" => AFK_SCHEMA_VERSION,
    )
end

function mean_or_nan(xs::Vector{Float64})
    isempty(xs) && return NaN
    return mean(xs)
end

function std_or_zero(xs::Vector{Float64})
    length(xs) <= 1 && return 0.0
    return std(xs)
end

function aggregate_rows(rows::Vector{Dict{String,Any}})
    groups = Dict{Tuple{String,String},Vector{Dict{String,Any}}}()
    for r in rows
        key = (String(r["task"]), String(r["arm"]))
        groups[key] = get(groups, key, Dict{String,Any}[])
        push!(groups[key], r)
    end
    out = Dict{String,Any}[]
    for ((task, arm), vals) in groups
        aucs = [Float64(v["auc_top10"]) for v in vals]
        top10 = [Float64(v["final_top10_mean"]) for v in vals]
        top1 = [Float64(v["final_top1"]) for v in vals]
        sc_ent = [Float64(v["diversity"]["scaffold_entropy"]) for v in vals]
        ge_ent = [Float64(v["diversity"]["genealogy_entropy"]) for v in vals]
        maxfam = [Float64(v["diversity"]["max_genealogy_fraction"]) for v in vals]
        push!(out, Dict{String,Any}(
            "task" => task,
            "arm" => arm,
            "n" => length(vals),
            "auc_mean" => mean_or_nan(aucs),
            "auc_std" => std_or_zero(aucs),
            "top10_mean" => mean_or_nan(top10),
            "top1_mean" => mean_or_nan(top1),
            "scaffold_entropy_mean" => mean_or_nan(sc_ent),
            "genealogy_entropy_mean" => mean_or_nan(ge_ent),
            "max_genealogy_fraction_mean" => mean_or_nan(maxfam),
        ))
    end
    sort!(out, by = r -> (String(r["task"]), String(r["arm"])))
    return out
end

function overall_by_arm(agg::Vector{Dict{String,Any}})
    groups = Dict{String,Vector{Dict{String,Any}}}()
    for row in agg
        arm = String(row["arm"])
        groups[arm] = get(groups, arm, Dict{String,Any}[])
        push!(groups[arm], row)
    end
    out = Dict{String,Any}[]
    for (arm, rows) in groups
        aucs = [Float64(r["auc_mean"]) for r in rows]
        top10 = [Float64(r["top10_mean"]) for r in rows]
        push!(out, Dict{String,Any}(
            "arm" => arm,
            "tasks" => [String(r["task"]) for r in rows],
            "mean_auc" => mean_or_nan(aucs),
            "mean_top10" => mean_or_nan(top10),
        ))
    end
    sort!(out, by = r -> -Float64(r["mean_auc"]))
    return out
end

function paired_deltas(rows::Vector{Dict{String,Any}}, baseline::String)
    idx = Dict{Tuple{String,Int,String},Dict{String,Any}}()
    for r in rows
        idx[(String(r["task"]), Int(r["seed"]), String(r["arm"]))] = r
    end
    out = Dict{String,Any}[]
    for r in rows
        arm = String(r["arm"])
        arm == baseline && continue
        key = (String(r["task"]), Int(r["seed"]), baseline)
        haskey(idx, key) || continue
        b = idx[key]
        auc = Float64(r["auc_top10"])
        bauc = Float64(b["auc_top10"])
        top10 = Float64(r["final_top10_mean"])
        btop10 = Float64(b["final_top10_mean"])
        push!(out, Dict{String,Any}(
            "task" => String(r["task"]),
            "seed" => Int(r["seed"]),
            "arm" => arm,
            "baseline" => baseline,
            "auc" => auc,
            "baseline_auc" => bauc,
            "delta_auc" => auc - bauc,
            "relative_delta_pct" => bauc == 0.0 ? NaN : (auc / bauc - 1.0) * 100.0,
            "top10" => top10,
            "baseline_top10" => btop10,
            "relative_top10_delta_pct" => btop10 == 0.0 ? NaN : (top10 / btop10 - 1.0) * 100.0,
        ))
    end
    return out
end

function gate_decision(rows::Vector{Dict{String,Any}}, overall::Vector{Dict{String,Any}})
    means = Dict(String(r["arm"]) => Float64(r["mean_auc"]) for r in overall)
    afk = get(means, "afk_smc", NaN)
    elite = get(means, "elite_ga", NaN)
    rank = get(means, "rank_weighted_ga", NaN)
    fixed = get(means, "fk_fixed_beta", NaN)
    nodiv = get(means, "fk_adaptive_no_diversity", NaN)
    best_core_baseline = maximum([elite, rank, fixed])
    deltas_rank = [d for d in paired_deltas(rows, "rank_weighted_ga") if String(d["arm"]) == "afk_smc"]
    paired_wins_rank = count(d -> Float64(d["delta_auc"]) > 0.0, deltas_rank)
    n_pairs = length(deltas_rank)

    afk_rows = [r for r in rows if String(r["arm"]) == "afk_smc"]
    collapse_rows = [r for r in afk_rows if Float64(r["diversity"]["max_genealogy_fraction"]) > 0.70 || Float64(r["diversity"]["max_scaffold_fraction"]) > 0.70]
    severe_collapse = !isempty(afk_rows) && length(collapse_rows) / length(afk_rows) > 0.5

    continue_gate = isfinite(afk) && isfinite(best_core_baseline) && afk > best_core_baseline * 1.05 && paired_wins_rank > n_pairs / 2 && !severe_collapse
    reasons = String[]
    if !isfinite(afk)
        push!(reasons, "afk_smc result missing.")
    end
    if isfinite(elite) && afk <= elite
        push!(reasons, "afk_smc <= elite_ga.")
    end
    if isfinite(rank) && afk <= rank
        push!(reasons, "afk_smc <= rank_weighted_ga.")
    end
    if isfinite(fixed) && afk <= fixed
        push!(reasons, "afk_smc <= fk_fixed_beta; adaptive component not validated.")
    end
    if severe_collapse
        push!(reasons, "afk_smc has severe scaffold/genealogy collapse in most rows.")
    end
    if continue_gate
        push!(reasons, "AFK-SMC continue gate passed; learned twisted-proposal Sprint 2 may be warranted.")
    elseif isempty(reasons)
        push!(reasons, "No hard stop, but +5%/paired/diversity continue gate not met; treat as inconclusive.")
    end
    verdict = continue_gate ? "AFK_SMC_CORE_SIGNAL_PRESENT" : "AFK_SMC_STOP_OR_INCONCLUSIVE"
    return Dict{String,Any}(
        "verdict" => verdict,
        "continue_gate" => continue_gate,
        "mean_by_arm" => means,
        "afk_vs_best_core_baseline_relative" => (isfinite(afk) && isfinite(best_core_baseline) && best_core_baseline != 0.0) ? afk / best_core_baseline - 1.0 : NaN,
        "afk_vs_elite_relative" => (isfinite(afk) && isfinite(elite) && elite != 0.0) ? afk / elite - 1.0 : NaN,
        "afk_vs_rank_relative" => (isfinite(afk) && isfinite(rank) && rank != 0.0) ? afk / rank - 1.0 : NaN,
        "afk_vs_fixed_beta_relative" => (isfinite(afk) && isfinite(fixed) && fixed != 0.0) ? afk / fixed - 1.0 : NaN,
        "paired_wins_vs_rank_weighted_ga" => paired_wins_rank,
        "paired_count_vs_rank_weighted_ga" => n_pairs,
        "severe_collapse" => severe_collapse,
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
    mode = strip(get(ENV, "AFK_MODE", "smoke"))
    tasks = parse_csv_strings("AFK_TASKS", mode == "smoke" ? ["qed"] : ["qed", "drd2", "celecoxib_rediscovery"])
    seeds = parse_csv_ints("AFK_SEEDS", mode == "smoke" ? [17] : [17, 23])
    arms = parse_csv_strings("AFK_ARMS", DEFAULT_ARMS)
    budget = parse_env_int("AFK_BUDGET", mode == "smoke" ? 128 : 300)
    population_size = parse_env_int("AFK_POPULATION", mode == "smoke" ? 32 : 48)
    children_per_generation = parse_env_int("AFK_CHILDREN", mode == "smoke" ? 16 : 16)
    attempt_multiplier = parse_env_int("AFK_ATTEMPT_MULTIPLIER", 6)
    target_ess = parse_env_float("AFK_TARGET_ESS", 0.6)
    delta_max = parse_env_float("AFK_DELTA_MAX", 4.0)
    fixed_delta = parse_env_float("AFK_FIXED_DELTA", 1.0)
    verbose = parse_env_bool("AFK_VERBOSE", false)

    invalid = [a for a in arms if !(a in DEFAULT_ARMS)]
    isempty(invalid) || error("Unknown AFK arms: $(invalid)")

    logmsg("AFK-SMC direction sprint mode=$(mode) tasks=$(tasks) arms=$(arms) seeds=$(seeds) budget=$(budget) pop=$(population_size) children=$(children_per_generation)")
    vocab = SMILESVocabulary()
    rows = Dict{String,Any}[]
    for seed in seeds, task in tasks, arm in arms
        logmsg("RUN start task=$(task) arm=$(arm) seed=$(seed)")
        start = time()
        try
            row = run_arm(task, arm, seed, vocab;
                budget=budget,
                population_size=population_size,
                children_per_generation=children_per_generation,
                attempt_multiplier=attempt_multiplier,
                target_ess=target_ess,
                delta_max=delta_max,
                fixed_delta=fixed_delta,
                verbose=verbose)
            row["elapsed_sec"] = time() - start
            push!(rows, row)
            logmsg("RUN ok task=$(task) arm=$(arm) seed=$(seed) auc=$(round(row["auc_top10"], digits=6)) top10=$(round(row["final_top10_mean"], digits=4)) top1=$(round(row["final_top1"], digits=4)) calls=$(row["calls_used"]) gen=$(row["generations"])")
        catch e
            bt = catch_backtrace()
            err = sprint(showerror, e, bt)
            logmsg("RUN failed task=$(task) arm=$(arm) seed=$(seed)")
            println(err)
            push!(rows, Dict{String,Any}(
                "task"=>task, "arm"=>arm, "seed"=>seed, "status"=>"failed", "error"=>err,
                "auc_top10"=>NaN, "final_top10_mean"=>NaN, "final_top1"=>NaN,
                "calls_used"=>0, "diversity"=>Dict{String,Any}(), "elapsed_sec"=>time()-start,
            ))
        end
    end

    ok_rows = [r for r in rows if !haskey(r, "status") || r["status"] != "failed"]
    agg = aggregate_rows(ok_rows)
    overall = overall_by_arm(agg)
    gate = gate_decision(ok_rows, overall)
    bundle = Dict{String,Any}(
        "created_at" => string(now()),
        "authoritative_date_note" => AUTHORITATIVE_DATE_NOTE,
        "mode" => mode,
        "schema_version" => AFK_SCHEMA_VERSION,
        "tasks" => tasks,
        "seeds" => seeds,
        "arms" => arms,
        "budget" => budget,
        "population_size" => population_size,
        "children_per_generation" => children_per_generation,
        "attempt_multiplier" => attempt_multiplier,
        "target_ess" => target_ess,
        "delta_max" => delta_max,
        "fixed_delta" => fixed_delta,
        "rows" => rows,
        "aggregate_rows" => agg,
        "overall_by_arm" => overall,
        "paired_delta_vs_elite_ga" => paired_deltas(ok_rows, "elite_ga"),
        "paired_delta_vs_rank_weighted_ga" => paired_deltas(ok_rows, "rank_weighted_ga"),
        "paired_delta_vs_fk_fixed_beta" => paired_deltas(ok_rows, "fk_fixed_beta"),
        "gate" => gate,
        "limitations" => [
            "This is AFK-SMC core validation, not AFK-GFN; no learned GFlowNet proposal is used.",
            "Micro results are direction-selection evidence, not SOTA or official PMO benchmark claims.",
        ],
    )
    out = joinpath(OUTDIR, "afk_smc_$(mode)_results.jls")
    latest = joinpath(OUTDIR, "afk_smc_latest_results.jls")
    save_bundle(out, bundle)
    save_bundle(latest, bundle)

    println("\n", "="^98)
    println("AFK-SMC DIRECTION SPRINT SUMMARY")
    println("="^98)
    println(rpad("Arm", 30), rpad("Mean AUC", 14), rpad("Mean Top10", 14), rpad("Tasks", 30))
    println("-"^98)
    for r in overall
        @printf("%-30s%-14.6f%-14.6f%-30s\n", r["arm"], r["mean_auc"], r["mean_top10"], join(r["tasks"], ","))
    end
    println("\nGATE: ", gate["verdict"])
    for reason in gate["reasons"]
        println("- ", reason)
    end
    logmsg("Saved results: $(abspath(out))")
    logmsg("Saved latest: $(abspath(latest))")

    if mode == "smoke"
        failed = [r for r in rows if haskey(r, "status") && r["status"] == "failed"]
        isempty(failed) || error("Smoke failed for runs: $([(r["task"], r["arm"], r["seed"]) for r in failed])")
    end
end

main()
