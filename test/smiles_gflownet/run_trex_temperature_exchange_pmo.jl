#!/usr/bin/env julia

# TREX-HE Direction Sprint (v2, pre-result exchange-sign amendment)
#
# Stage 0 validates a heuristic temperature-exchange molecular search core.
# This is intentionally NOT TREX-GFN and NOT exact replica-exchange MCMC.

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
const OUTDIR = joinpath(ROOT, "checkpoints", "trex_temperature_exchange")
const TREX_SCHEMA_VERSION = "trex_he_v2_pre_result_exchange_sign_amended"
const AUTHORITATIVE_DATE_NOTE = "User supplied authoritative date/time: Saturday, 2026-06-20 01:18 EDT"
mkpath(OUTDIR)

const BETAS = Float64[0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
const REQUIRED_ARMS = [
    "uniform_population",
    "elite_ga",
    "rank_weighted_ga",
    "single_hot_beta",
    "temperature_ladder_no_exchange",
    "random_exchange_control",
    "trex_exchange",
]
const LADDER_ARMS = Set(["temperature_ladder_no_exchange", "random_exchange_control", "trex_exchange"])
const PRIMARY_BASELINES = Set(["elite_ga", "rank_weighted_ga", "single_hot_beta", "temperature_ladder_no_exchange"])

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

struct TXItem
    id::String
    repr::String
    reward::Float64
    scaffold::String
    genealogy::String
    generation::Int
    source::String
    birth_beta::Float64
end

mutable struct TXState
    domain::String
    task::String
    arm::String
    seed::Int
    budget::Int
    calls_used::Int
    history::Dict{String,Float64}
    item_bank::Dict{String,TXItem}
    replicas::Vector{Vector{TXItem}}
    curve::Vector{Dict{String,Any}}
    ledger::Dict{String,Int}
    exchange_log::Vector{Dict{String,Any}}
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

function increment!(d::Dict{String,Int}, key::String, n::Int=1)
    d[key] = get(d, key, 0) + n
    return nothing
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

function record_curve!(state::TXState; generation::Int, event::String)
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

function live_items(state::TXState)
    isempty(state.replicas) && return TXItem[]
    out = TXItem[]
    for rep in state.replicas
        append!(out, rep)
    end
    return out
end

function unique_by_id_best(items::Vector{TXItem})
    best = Dict{String,TXItem}()
    for item in items
        if !haskey(best, item.id) || item.reward > best[item.id].reward
            best[item.id] = item
        end
    end
    return collect(values(best))
end

function diversity_metrics(items::Vector{TXItem})
    n = length(items)
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
    sc = counts([p.scaffold for p in items])
    gc = counts([p.genealogy for p in items])
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

function robust_z_map(items::Vector{TXItem})
    isempty(items) && return Dict{String,Float64}()
    rewards = sort([p.reward for p in items])
    n = length(rewards)
    med = rewards[clamp(ceil(Int, 0.50 * n), 1, n)]
    q25 = rewards[clamp(ceil(Int, 0.25 * n), 1, n)]
    q75 = rewards[clamp(ceil(Int, 0.75 * n), 1, n)]
    scale = max(q75 - q25, 1e-6)
    z = Dict{String,Float64}()
    for item in items
        z[item.id] = clamp((item.reward - med) / scale, -5.0, 5.0)
    end
    return z
end

function sample_index_from_probs(rng::AbstractRNG, probs::Vector{Float64})
    isempty(probs) && error("cannot sample from empty probabilities")
    total = sum(probs)
    if !(isfinite(total)) || total <= 0.0
        return rand(rng, 1:length(probs))
    end
    cum = cumsum(probs ./ total)
    return clamp(searchsortedfirst(cum, rand(rng)), 1, length(probs))
end

function exp_beta_probs(items::Vector{TXItem}, beta::Float64, zmap::Dict{String,Float64})
    isempty(items) && return Float64[]
    vals = [beta * get(zmap, it.id, 0.0) for it in items]
    m = maximum(vals)
    w = exp.(vals .- m)
    total = sum(w)
    total <= 0.0 && return fill(1.0 / length(items), length(items))
    return w ./ total
end

function single_arm_probs(items::Vector{TXItem}, arm::String, zmap::Dict{String,Float64})
    n = length(items)
    n == 0 && return Float64[]
    if arm == "uniform_population"
        return fill(1.0 / n, n)
    elseif arm == "elite_ga"
        order = sortperm([p.reward for p in items], rev=true)
        elite_n = max(1, ceil(Int, n / 4))
        probs = zeros(Float64, n)
        for idx in order[1:elite_n]
            probs[idx] = 1.0 / elite_n
        end
        return probs
    elseif arm == "rank_weighted_ga"
        order = sortperm([p.reward for p in items], rev=true)
        probs = zeros(Float64, n)
        weights = [1.0 / i for i in 1:n]
        weights ./= sum(weights)
        for (rank, idx) in enumerate(order)
            probs[idx] = weights[rank]
        end
        return probs
    elseif arm == "single_hot_beta"
        return exp_beta_probs(items, 8.0, zmap)
    else
        return fill(1.0 / n, n)
    end
end

function select_population(items::Vector{TXItem}, n_out::Int, arm::String, rng::AbstractRNG; beta::Float64=0.0, zmap::Dict{String,Float64}=Dict{String,Float64}())
    uniq = unique_by_id_best(items)
    isempty(uniq) && return TXItem[]
    if arm == "elite_ga"
        sort!(uniq, by = p -> -p.reward)
        if length(uniq) >= n_out
            return uniq[1:n_out]
        end
        return vcat(uniq, [uniq[rand(rng, 1:length(uniq))] for _ in 1:(n_out - length(uniq))])
    end
    probs = arm in LADDER_ARMS ? exp_beta_probs(uniq, beta, zmap) : single_arm_probs(uniq, arm, zmap)
    return [uniq[sample_index_from_probs(rng, probs)] for _ in 1:n_out]
end

function valid_tokenizable(vocab, smiles::String)
    try
        tokens = encode(vocab, smiles)
        return length(tokens) >= 2
    catch
        return false
    end
end

# -----------------------------------------------------------------------------
# Synthetic landscape
# -----------------------------------------------------------------------------

function bitstring_random(rng::AbstractRNG, n::Int; p::Float64=0.5)
    return String([rand(rng) < p ? '1' : '0' for _ in 1:n])
end

function hamming(a::String, b::String)
    n = min(length(a), length(b))
    d = count(i -> a[i] != b[i], 1:n) + abs(length(a) - length(b))
    return d
end

function synthetic_reward(x::String, family::String)
    n = length(x)
    if family == "funnel"
        low = repeat("0", n)
        high = repeat("1", n)
        broad = 0.62 * exp(-hamming(x, low) / 14.0)
        narrow = 1.00 * exp(-hamming(x, high) / 4.0)
        return max(broad, narrow)
    elseif family == "deceptive_trap"
        total = 0.0
        blocks = n ÷ 4
        for b in 0:(blocks-1)
            block = x[(4b+1):(4b+4)]
            ones = count(==('1'), block)
            total += ones == 4 ? 1.0 : 0.82 * (4 - ones) / 4.0
        end
        return total / blocks
    elseif family == "multi_peak"
        targets = [repeat("1", n), repeat("01", n ÷ 2), repeat("0011", n ÷ 4), repeat("1100", n ÷ 4)]
        heights = [1.0, 0.93, 0.90, 0.88]
        widths = [4.5, 6.0, 6.0, 7.0]
        return maximum(heights[i] * exp(-hamming(x, targets[i]) / widths[i]) for i in eachindex(targets))
    else
        error("Unknown synthetic family: $family")
    end
end

function synthetic_bootstrap(family::String, seed::Int, n::Int)
    rng = MersenneTwister(seed + bounded_hash_int((family, "bootstrap")))
    xs = String[]
    # Bias initial population toward the broad/deceptive basin, with a small random tail.
    for _ in 1:10
        push!(xs, bitstring_random(rng, n; p=0.15))
    end
    for _ in 1:4
        push!(xs, bitstring_random(rng, n; p=0.50))
    end
    push!(xs, repeat("0", n))
    return unique(xs)
end

function synthetic_mutate(x::String, rng::AbstractRNG)
    chars = collect(x)
    flips = rand(rng, 1:3)
    for idx in rand(rng, 1:length(chars), flips)
        chars[idx] = chars[idx] == '1' ? '0' : '1'
    end
    return String(chars)
end

function synthetic_crossover(a::String, b::String, rng::AbstractRNG)
    n = min(length(a), length(b))
    n <= 2 && return a
    cut = rand(rng, 2:(n-1))
    return string(a[1:cut], b[(cut+1):n])
end

# -----------------------------------------------------------------------------
# Evaluation and initialization
# -----------------------------------------------------------------------------

function make_item(domain::String, task::String, repr::String, reward::Float64; generation::Int, source::String, genealogy::String, birth_beta::Float64)
    id = "$(domain)::$(task)::$(repr)"
    scaffold = domain == "synthetic" ? string("bits_", repr[1:min(6, length(repr))]) : safe_scaffold(repr)
    return TXItem(id, repr, reward, scaffold, genealogy, generation, source, birth_beta)
end

function canonicalize_candidate(domain::String, candidate::String)
    if domain == "synthetic"
        return candidate
    else
        return canonicalize_smiles_identity(candidate)
    end
end

function evaluate_candidates!(state::TXState, vocab, candidates::Vector{String}, target_replicas::Vector{Int}, genealogies::Vector{String}, birth_betas::Vector{Float64}; generation::Int, source::String)
    remaining = state.budget - state.calls_used
    remaining <= 0 && return TXItem[]
    eval_repr = String[]
    eval_replica = Int[]
    eval_genealogy = String[]
    eval_beta = Float64[]
    seen = Set{String}()
    for (idx, cand) in enumerate(candidates)
        canonical = canonicalize_candidate(state.domain, cand)
        if isempty(canonical)
            increment!(state.ledger, "invalid_canonical")
            continue
        end
        if canonical in seen
            increment!(state.ledger, "duplicate_within_generation")
            continue
        end
        if haskey(state.history, canonical)
            increment!(state.ledger, "already_evaluated")
            continue
        end
        if state.domain == "pmo" && !valid_tokenizable(vocab, canonical)
            increment!(state.ledger, "tokenization_failed")
            continue
        end
        push!(seen, canonical)
        push!(eval_repr, canonical)
        push!(eval_replica, target_replicas[min(idx, length(target_replicas))])
        push!(eval_genealogy, genealogies[min(idx, length(genealogies))])
        push!(eval_beta, birth_betas[min(idx, length(birth_betas))])
        length(eval_repr) >= remaining && break
    end
    isempty(eval_repr) && return TXItem[]
    scores = if state.domain == "synthetic"
        [synthetic_reward(x, state.task) for x in eval_repr]
    else
        OracleBridge.evaluate_batch(eval_repr, state.task)
    end
    items = TXItem[]
    for (i, (x, r)) in enumerate(zip(eval_repr, scores))
        state.calls_used += 1
        reward = Float64(r)
        state.history[x] = reward
        item = make_item(state.domain, state.task, x, reward;
            generation=generation,
            source=source,
            genealogy=eval_genealogy[i],
            birth_beta=eval_beta[i])
        state.item_bank[x] = item
        push!(items, item)
        rep_idx = clamp(eval_replica[i], 1, length(state.replicas))
        push!(state.replicas[rep_idx], item)
        record_curve!(state; generation=generation, event=source)
    end
    increment!(state.ledger, "oracle_evaluated_unique_valid", length(items))
    return items
end

function initialize_state(domain::String, task::String, arm::String, seed::Int, budget::Int, population_size::Int, replica_size::Int, genome_length::Int, vocab)
    if domain == "pmo"
        OracleBridge.init_oracles!([task]; cache_dir=joinpath(ROOT, "data", "tdc_cache"))
    end
    nrep = arm in LADDER_ARMS ? length(BETAS) : 1
    state = TXState(domain, task, arm, seed, budget, 0,
        Dict{String,Float64}(), Dict{String,TXItem}(), [TXItem[] for _ in 1:nrep],
        Dict{String,Any}[], Dict{String,Int}(), Dict{String,Any}[])
    boots = domain == "synthetic" ? synthetic_bootstrap(task, seed, genome_length) : unique(vcat(DEFAULT_BOOTSTRAP_SEEDS, get(TASK_BOOTSTRAP_SEEDS, task, String[])))
    target_reps = [((i - 1) % nrep) + 1 for i in eachindex(boots)]
    ge = ["seed_$(i)" for i in eachindex(boots)]
    bb = [nrep == 1 ? 8.0 : BETAS[target_reps[i]] for i in eachindex(boots)]
    evaluate_candidates!(state, vocab, boots, target_reps, ge, bb; generation=0, source="bootstrap")
    increment!(state.ledger, "bootstrap_seed_count", length(boots))
    rng = MersenneTwister(seed + bounded_hash_int((domain, task, arm, "fill")))
    bank_items = collect(values(state.item_bank))
    isempty(bank_items) && return state
    if nrep == 1
        while length(state.replicas[1]) < population_size
            push!(state.replicas[1], bank_items[rand(rng, 1:length(bank_items))])
        end
        if length(state.replicas[1]) > population_size
            state.replicas[1] = state.replicas[1][1:population_size]
        end
    else
        for j in 1:nrep
            while length(state.replicas[j]) < replica_size
                push!(state.replicas[j], bank_items[rand(rng, 1:length(bank_items))])
            end
            if length(state.replicas[j]) > replica_size
                state.replicas[j] = state.replicas[j][1:replica_size]
            end
        end
    end
    return state
end

# -----------------------------------------------------------------------------
# Proposal and update
# -----------------------------------------------------------------------------

function choose_parent(items::Vector{TXItem}, rng::AbstractRNG, probs::Vector{Float64})
    isempty(items) && error("empty parent pool")
    return items[sample_index_from_probs(rng, probs)]
end

function propose_synthetic_child(p1::TXItem, p2::Union{TXItem,Nothing}, rng::AbstractRNG)
    if p2 === nothing || rand(rng) < 0.65
        return synthetic_mutate(p1.repr, rng), p1.genealogy
    else
        child = synthetic_crossover(p1.repr, p2.repr, rng)
        child = rand(rng) < 0.5 ? synthetic_mutate(child, rng) : child
        genealogy = p1.reward >= p2.reward ? p1.genealogy : p2.genealogy
        return child, genealogy
    end
end

function propose_pmo_children(p1::TXItem, p2::Union{TXItem,Nothing}, vocab, rng::AbstractRNG, state::TXState)
    out = String[]
    genealogies = String[]
    draw = rand(rng)
    if p2 !== nothing && draw < 0.35
        Random.seed!(rand(rng, 1:typemax(Int32)))
        children = smiles_crossover_rdkit(p1.repr, p2.repr)
        isempty(children) && increment!(state.ledger, "empty_crossover")
        for child in children
            push!(out, child)
            push!(genealogies, p1.reward >= p2.reward ? p1.genealogy : p2.genealogy)
        end
    elseif draw < 0.90
        Random.seed!(rand(rng, 1:typemax(Int32)))
        children = smiles_mutate_rdkit(p1.repr; n_mutations=3)
        isempty(children) && increment!(state.ledger, "empty_mutation")
        for child in children
            push!(out, child)
            push!(genealogies, p1.genealogy)
        end
    else
        toks = try encode(vocab, p1.repr) catch; Int[] end
        if length(toks) >= 2
            mtoks = smiles_mutate_tokens(toks, vocab; n_mutations=1)
            smi = try decode(vocab, mtoks) catch; "" end
            push!(out, smi)
            push!(genealogies, p1.genealogy)
        else
            increment!(state.ledger, "token_fallback_failed")
        end
    end
    return out, genealogies
end

function generate_children!(state::TXState, vocab, rng::AbstractRNG; children_total::Int, attempt_multiplier::Int, generation::Int)
    nrep = length(state.replicas)
    zmap = robust_z_map(live_items(state))
    candidates = String[]
    target_reps = Int[]
    genealogies = String[]
    birth_betas = Float64[]
    requests = if nrep == 1
        [children_total]
    else
        base = children_total ÷ nrep
        rem = children_total - base * nrep
        [base + (j <= rem ? 1 : 0) for j in 1:nrep]
    end
    for rep_idx in 1:nrep
        requested = requests[rep_idx]
        requested <= 0 && continue
        attempts = 0
        max_attempts = attempt_multiplier * requested
        seen_local = Set{String}()
        while count(==(rep_idx), target_reps) < requested && attempts < max_attempts && !isempty(state.replicas[rep_idx])
            attempts += 1
            increment!(state.ledger, "attempted_proposals")
            beta = nrep == 1 ? (state.arm == "single_hot_beta" ? 8.0 : 0.0) : BETAS[rep_idx]
            probs = nrep == 1 ? single_arm_probs(state.replicas[1], state.arm, zmap) : exp_beta_probs(state.replicas[rep_idx], beta, zmap)
            p1 = choose_parent(state.replicas[rep_idx], rng, probs)
            p2 = nothing
            if length(state.replicas[rep_idx]) >= 2 && rand(rng) < 0.35
                p2 = choose_parent(state.replicas[rep_idx], rng, probs)
            end
            raw_children, raw_genealogies = if state.domain == "synthetic"
                child, ge = propose_synthetic_child(p1, p2, rng)
                [child], [ge]
            else
                propose_pmo_children(p1, p2, vocab, rng, state)
            end
            for (raw, ge) in zip(raw_children, raw_genealogies)
                increment!(state.ledger, "generated_candidates")
                canonical = canonicalize_candidate(state.domain, raw)
                if isempty(canonical)
                    increment!(state.ledger, "invalid_canonical")
                    continue
                end
                if canonical in seen_local || haskey(state.history, canonical)
                    increment!(state.ledger, "filtered_candidate")
                    continue
                end
                if state.domain == "pmo" && !valid_tokenizable(vocab, canonical)
                    increment!(state.ledger, "tokenization_failed")
                    continue
                end
                push!(seen_local, canonical)
                push!(candidates, canonical)
                push!(target_reps, rep_idx)
                push!(genealogies, ge)
                push!(birth_betas, beta)
                count(==(rep_idx), target_reps) >= requested && break
            end
        end
        if count(==(rep_idx), target_reps) < requested
            increment!(state.ledger, "underfilled_generations")
        end
    end
    return candidates, target_reps, genealogies, birth_betas
end

function update_populations!(state::TXState, rng::AbstractRNG; population_size::Int, replica_size::Int)
    all_items = live_items(state)
    zmap = robust_z_map(all_items)
    if length(state.replicas) == 1
        state.replicas[1] = select_population(state.replicas[1], population_size, state.arm, rng; zmap=zmap)
    else
        for j in eachindex(state.replicas)
            state.replicas[j] = select_population(state.replicas[j], replica_size, state.arm, rng; beta=BETAS[j], zmap=zmap)
        end
    end
end

function exchange_step!(state::TXState, rng::AbstractRNG; generation::Int, random_swap_p::Float64=0.25)
    state.arm in ("random_exchange_control", "trex_exchange") || return
    nrep = length(state.replicas)
    nrep <= 1 && return
    zmap = robust_z_map(live_items(state))
    for j in 1:(nrep-1)
        isempty(state.replicas[j]) && continue
        isempty(state.replicas[j+1]) && continue
        increment!(state.ledger, "exchange_attempts")
        i_hot = rand(rng, 1:length(state.replicas[j]))
        i_cold = rand(rng, 1:length(state.replicas[j+1]))
        hot = state.replicas[j][i_hot]
        cold = state.replicas[j+1][i_cold]
        z_hot = get(zmap, hot.id, 0.0)
        z_cold = get(zmap, cold.id, 0.0)
        beta_hot = BETAS[j]
        beta_cold = BETAS[j+1]
        log_alpha = (beta_cold - beta_hot) * (z_hot - z_cold)
        accepted = false
        mode = state.arm == "random_exchange_control" ? "random" : "score"
        if state.arm == "random_exchange_control"
            accepted = rand(rng) < random_swap_p
        else
            accepted = log(rand(rng)) < min(0.0, clamp(log_alpha, -20.0, 20.0))
        end
        if accepted
            state.replicas[j][i_hot], state.replicas[j+1][i_cold] = cold, hot
            increment!(state.ledger, "exchange_accepts")
        end
        push!(state.exchange_log, Dict{String,Any}(
            "generation" => generation,
            "pair" => "$(j)-$(j+1)",
            "beta_hot" => beta_hot,
            "beta_cold" => beta_cold,
            "z_hot" => z_hot,
            "z_cold" => z_cold,
            "log_alpha" => log_alpha,
            "accepted" => accepted,
            "mode" => mode,
            "hot_reward" => hot.reward,
            "cold_reward" => cold.reward,
            "hot_genealogy" => hot.genealogy,
            "cold_genealogy" => cold.genealogy,
        ))
    end
end

function run_arm(domain::String, task::String, arm::String, seed::Int, vocab;
                 budget::Int, population_size::Int, children_total::Int,
                 attempt_multiplier::Int, genome_length::Int, verbose::Bool=false)
    replica_size = max(1, population_size ÷ length(BETAS))
    rng = MersenneTwister(seed + bounded_hash_int((TREX_SCHEMA_VERSION, domain, task, arm)))
    state = initialize_state(domain, task, arm, seed, budget, population_size, replica_size, genome_length, vocab)
    update_populations!(state, rng; population_size=population_size, replica_size=replica_size)
    generation = 0
    while state.calls_used < budget && !isempty(live_items(state))
        generation += 1
        candidates, target_reps, genealogies, birth_betas = generate_children!(state, vocab, rng;
            children_total=children_total,
            attempt_multiplier=attempt_multiplier,
            generation=generation)
        if isempty(candidates)
            increment!(state.ledger, "empty_candidate_generations")
            break
        end
        before = state.calls_used
        evaluate_candidates!(state, vocab, candidates, target_reps, genealogies, birth_betas;
            generation=generation,
            source="offspring")
        if state.calls_used == before
            increment!(state.ledger, "zero_eval_generations")
            break
        end
        update_populations!(state, rng; population_size=population_size, replica_size=replica_size)
        exchange_step!(state, rng; generation=generation)
        generation > 10_000 && break
    end
    stats = top_stats(state.history)
    auc = normalized_auc(state.curve, budget; metric="top10_mean")
    items = live_items(state)
    div = diversity_metrics(items)
    ledger = copy(state.ledger)
    attempts = max(1, get(ledger, "attempted_proposals", 0))
    invalids = get(ledger, "invalid_canonical", 0) + get(ledger, "tokenization_failed", 0)
    duplicates = get(ledger, "duplicate_within_generation", 0) + get(ledger, "already_evaluated", 0) + get(ledger, "filtered_candidate", 0)
    exchange_attempts = get(ledger, "exchange_attempts", 0)
    exchange_accepts = get(ledger, "exchange_accepts", 0)
    top_birth = beta_top10_contribution(state)
    return Dict{String,Any}(
        "domain" => domain,
        "task" => task,
        "arm" => arm,
        "seed" => seed,
        "budget" => budget,
        "calls_used" => state.calls_used,
        "generations" => generation,
        "auc_top10" => auc,
        "attempt_normalized_auc_proxy" => auc * min(1.0, budget / attempts),
        "final_top1" => stats["top1"],
        "final_top10_mean" => stats["top10_mean"],
        "top10_denominator" => stats["top10_denominator"],
        "unique_evaluated" => length(state.history),
        "ledger" => ledger,
        "invalid_rate_per_attempt" => invalids / attempts,
        "duplicate_filter_rate_per_attempt" => duplicates / attempts,
        "exchange_acceptance_rate" => exchange_attempts == 0 ? NaN : exchange_accepts / exchange_attempts,
        "curve" => state.curve,
        "exchange_log" => state.exchange_log,
        "diversity" => div,
        "beta_top10_contribution" => top_birth,
        "schema_version" => TREX_SCHEMA_VERSION,
        "exchange_formula" => "log_alpha=(beta_cold-beta_hot)*(z_hot-z_cold); heuristic score exchange, not exact MCMC",
    )
end

function beta_top10_contribution(state::TXState)
    items = collect(values(state.item_bank))
    sort!(items, by = p -> -p.reward)
    top = items[1:min(10, length(items))]
    d = Dict{String,Int}()
    for item in top
        key = isfinite(item.birth_beta) ? string(item.birth_beta) : "unknown"
        d[key] = get(d, key, 0) + 1
    end
    return d
end

# -----------------------------------------------------------------------------
# Aggregation and gates
# -----------------------------------------------------------------------------

function mean_or_nan(xs)
    isempty(xs) && return NaN
    vals = Float64.(xs)
    isempty(vals) && return NaN
    return mean(vals)
end

function std_or_zero(xs)
    vals = Float64.(xs)
    length(vals) <= 1 && return 0.0
    return std(vals)
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
        ex = [Float64(v["exchange_acceptance_rate"]) for v in vals if !isnan(Float64(v["exchange_acceptance_rate"]))]
        invalid = [Float64(v["invalid_rate_per_attempt"]) for v in vals]
        dup = [Float64(v["duplicate_filter_rate_per_attempt"]) for v in vals]
        sc_ent = [Float64(v["diversity"]["scaffold_entropy"]) for v in vals]
        ge_ent = [Float64(v["diversity"]["genealogy_entropy"]) for v in vals]
        max_ge = [Float64(v["diversity"]["max_genealogy_fraction"]) for v in vals]
        push!(out, Dict{String,Any}(
            "task" => task,
            "arm" => arm,
            "n" => length(vals),
            "auc_mean" => mean_or_nan(aucs),
            "auc_std" => std_or_zero(aucs),
            "top10_mean" => mean_or_nan(top10),
            "top1_mean" => mean_or_nan(top1),
            "exchange_acceptance_mean" => mean_or_nan(ex),
            "invalid_rate_mean" => mean_or_nan(invalid),
            "duplicate_filter_rate_mean" => mean_or_nan(dup),
            "scaffold_entropy_mean" => mean_or_nan(sc_ent),
            "genealogy_entropy_mean" => mean_or_nan(ge_ent),
            "max_genealogy_fraction_mean" => mean_or_nan(max_ge),
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

function index_rows(rows::Vector{Dict{String,Any}})
    idx = Dict{Tuple{String,Int,String},Dict{String,Any}}()
    for r in rows
        idx[(String(r["task"]), Int(r["seed"]), String(r["arm"]))] = r
    end
    return idx
end

function paired_deltas(rows::Vector{Dict{String,Any}}, baseline::String)
    idx = index_rows(rows)
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

function best_baseline_pair_rows(rows::Vector{Dict{String,Any}})
    idx = index_rows(rows)
    pairs = unique([(String(r["task"]), Int(r["seed"])) for r in rows])
    out = Dict{String,Any}[]
    for (task, seed) in pairs
        trex_key = (task, seed, "trex_exchange")
        haskey(idx, trex_key) || continue
        best_arm = ""
        best_auc = -Inf
        best_row = nothing
        for arm in PRIMARY_BASELINES
            key = (task, seed, arm)
            if haskey(idx, key)
                auc = Float64(idx[key]["auc_top10"])
                if auc > best_auc
                    best_auc = auc
                    best_arm = arm
                    best_row = idx[key]
                end
            end
        end
        best_row === nothing && continue
        trex = idx[trex_key]
        push!(out, Dict{String,Any}(
            "task" => task,
            "seed" => seed,
            "baseline_arm" => best_arm,
            "trex_auc" => Float64(trex["auc_top10"]),
            "best_baseline_auc" => best_auc,
            "delta_auc" => Float64(trex["auc_top10"]) - best_auc,
            "relative_delta" => best_auc == 0.0 ? NaN : Float64(trex["auc_top10"]) / best_auc - 1.0,
            "trex_top10" => Float64(trex["final_top10_mean"]),
            "best_baseline_top10" => Float64(best_row["final_top10_mean"]),
        ))
    end
    sort!(out, by = r -> (String(r["task"]), Int(r["seed"])))
    return out
end

function gate_decision(rows::Vector{Dict{String,Any}}, overall::Vector{Dict{String,Any}}, mode::String)
    means = Dict(String(r["arm"]) => Float64(r["mean_auc"]) for r in overall)
    trex = get(means, "trex_exchange", NaN)
    getm(arm) = get(means, arm, NaN)
    best_baseline = maximum([getm("elite_ga"), getm("rank_weighted_ga"), getm("single_hot_beta"), getm("temperature_ladder_no_exchange")])
    pair_best = best_baseline_pair_rows(rows)
    paired_wins = count(r -> Float64(r["delta_auc"]) > 0.0, pair_best)
    n_pairs = length(pair_best)
    trex_rows = [r for r in rows if String(r["arm"]) == "trex_exchange"]
    collapse_rows = [r for r in trex_rows if Float64(r["diversity"]["max_genealogy_fraction"]) > 0.70 || Float64(r["diversity"]["max_scaffold_fraction"]) > 0.70]
    severe_collapse = !isempty(trex_rows) && length(collapse_rows) / length(trex_rows) > 0.5
    hidden_cost_bad = false
    if !isempty(trex_rows)
        trex_dup = mean([Float64(r["duplicate_filter_rate_per_attempt"]) for r in trex_rows])
        baseline_rows = [r for r in rows if String(r["arm"]) in PRIMARY_BASELINES]
        base_dup = isempty(baseline_rows) ? 0.0 : mean([Float64(r["duplicate_filter_rate_per_attempt"]) for r in baseline_rows])
        hidden_cost_bad = trex_dup > max(0.30, 2.0 * base_dup + 0.05)
    end
    continue_gate = isfinite(trex) && isfinite(best_baseline) && trex > best_baseline * 1.05 && paired_wins > n_pairs / 2 &&
        trex > getm("temperature_ladder_no_exchange") && trex > getm("single_hot_beta") && trex > getm("random_exchange_control") &&
        !severe_collapse && !hidden_cost_bad
    reasons = String[]
    if !isfinite(trex)
        push!(reasons, "trex_exchange result missing.")
    end
    for arm in ["elite_ga", "rank_weighted_ga", "single_hot_beta", "temperature_ladder_no_exchange", "random_exchange_control"]
        val = getm(arm)
        if isfinite(val) && isfinite(trex) && trex <= val
            push!(reasons, "trex_exchange <= $(arm).")
        end
    end
    severe_collapse && push!(reasons, "trex_exchange has severe scaffold/genealogy collapse in most rows.")
    hidden_cost_bad && push!(reasons, "trex_exchange has much worse duplicate/hidden proposal cost than baselines.")
    if continue_gate
        push!(reasons, "TREX-HE continue gate passed; temperature-conditioned GFN proposal can be planned next.")
    elseif isempty(reasons)
        push!(reasons, "No hard stop, but +5%/paired/ablation/cost continue gate not met; treat as inconclusive.")
    end
    verdict = continue_gate ? "TREX_HE_CORE_SIGNAL_PRESENT" : "TREX_HE_STOP_OR_INCONCLUSIVE"
    return Dict{String,Any}(
        "mode" => mode,
        "verdict" => verdict,
        "continue_gate" => continue_gate,
        "mean_by_arm" => means,
        "trex_vs_best_baseline_relative" => (isfinite(trex) && isfinite(best_baseline) && best_baseline != 0.0) ? trex / best_baseline - 1.0 : NaN,
        "paired_wins_vs_best_baseline" => paired_wins,
        "paired_count_vs_best_baseline" => n_pairs,
        "severe_collapse" => severe_collapse,
        "hidden_cost_bad" => hidden_cost_bad,
        "pair_best_baseline_rows" => pair_best,
        "reasons" => reasons,
    )
end

function synthetic_gate(rows::Vector{Dict{String,Any}})
    agg = aggregate_rows(rows)
    bytask = Dict{String,Dict{String,Float64}}()
    for r in agg
        task = String(r["task"])
        bytask[task] = get(bytask, task, Dict{String,Float64}())
        bytask[task][String(r["arm"])] = Float64(r["auc_mean"])
    end
    pass_tasks = String[]
    fail_tasks = String[]
    for (task, d) in bytask
        trex = get(d, "trex_exchange", NaN)
        single = get(d, "single_hot_beta", NaN)
        noex = get(d, "temperature_ladder_no_exchange", NaN)
        if isfinite(trex) && isfinite(single) && isfinite(noex) && trex > single && trex > noex
            push!(pass_tasks, task)
        else
            push!(fail_tasks, task)
        end
    end
    return Dict{String,Any}(
        "diagnostic_pass" => length(pass_tasks) >= 2,
        "pass_tasks" => sort(pass_tasks),
        "fail_tasks" => sort(fail_tasks),
        "note" => "Synthetic pass is diagnostic only; it cannot justify PMO claims by itself.",
    )
end

function save_bundle(path::String, bundle::Dict{String,Any})
    tmp = path * ".tmp"
    serialize(tmp, bundle)
    mv(tmp, path; force=true)
    return path
end

function run_suite(mode::String)
    if mode == "compile"
        println("TREX compile smoke OK: ", TREX_SCHEMA_VERSION)
        return nothing
    end
    domain = mode == "synthetic" ? "synthetic" : "pmo"
    tasks = if mode == "synthetic"
        parse_csv_strings("TREX_SYNTH_FAMILIES", ["funnel", "deceptive_trap", "multi_peak"])
    elseif mode == "smoke"
        parse_csv_strings("TREX_TASKS", ["qed"])
    else
        parse_csv_strings("TREX_TASKS", ["qed", "drd2", "celecoxib_rediscovery"])
    end
    seeds = parse_csv_ints("TREX_SEEDS", mode == "smoke" ? [17] : [17, 23])
    arms = parse_csv_strings("TREX_ARMS", REQUIRED_ARMS)
    unknown = [a for a in arms if !(a in REQUIRED_ARMS)]
    isempty(unknown) || error("Unknown TREX arms: $(unknown)")
    budget = parse_env_int("TREX_BUDGET", mode == "synthetic" ? 256 : (mode == "smoke" ? 128 : 300))
    population_size = parse_env_int("TREX_POPULATION", 48)
    children_total = parse_env_int("TREX_CHILDREN", mode == "synthetic" ? 24 : 18)
    attempt_multiplier = parse_env_int("TREX_ATTEMPT_MULTIPLIER", 6)
    genome_length = parse_env_int("TREX_GENOME_LENGTH", 32)
    verbose = lowercase(strip(get(ENV, "TREX_VERBOSE", "false"))) in ("1", "true", "yes")
    vocab = SMILESVocabulary()
    logmsg("TREX-HE mode=$(mode) domain=$(domain) tasks=$(tasks) seeds=$(seeds) arms=$(arms) budget=$(budget) population=$(population_size) children=$(children_total)")
    rows = Dict{String,Any}[]
    for seed in seeds, task in tasks, arm in arms
        logmsg("RUN start domain=$(domain) task=$(task) arm=$(arm) seed=$(seed)")
        start = time()
        try
            row = run_arm(domain, task, arm, seed, vocab;
                budget=budget,
                population_size=population_size,
                children_total=children_total,
                attempt_multiplier=attempt_multiplier,
                genome_length=genome_length,
                verbose=verbose)
            row["elapsed_sec"] = time() - start
            push!(rows, row)
            logmsg("RUN ok task=$(task) arm=$(arm) seed=$(seed) auc=$(round(row["auc_top10"], digits=6)) top10=$(round(row["final_top10_mean"], digits=4)) calls=$(row["calls_used"]) gen=$(row["generations"]) exch=$(row["exchange_acceptance_rate"])")
        catch e
            bt = catch_backtrace()
            err = sprint(showerror, e, bt)
            logmsg("RUN failed task=$(task) arm=$(arm) seed=$(seed)")
            println(err)
            push!(rows, Dict{String,Any}(
                "domain"=>domain, "task"=>task, "arm"=>arm, "seed"=>seed, "status"=>"failed", "error"=>err,
                "auc_top10"=>NaN, "final_top10_mean"=>NaN, "final_top1"=>NaN,
                "calls_used"=>0, "diversity"=>Dict{String,Any}("max_genealogy_fraction"=>NaN, "max_scaffold_fraction"=>NaN),
                "elapsed_sec"=>time()-start,
                "exchange_acceptance_rate"=>NaN,
                "invalid_rate_per_attempt"=>NaN,
                "duplicate_filter_rate_per_attempt"=>NaN,
            ))
        end
    end
    ok_rows = [r for r in rows if !haskey(r, "status") || r["status"] != "failed"]
    agg = aggregate_rows(ok_rows)
    overall = overall_by_arm(agg)
    gate = gate_decision(ok_rows, overall, mode)
    synth_gate = mode == "synthetic" ? synthetic_gate(ok_rows) : Dict{String,Any}()
    bundle = Dict{String,Any}(
        "created_at" => string(now()),
        "authoritative_date_note" => AUTHORITATIVE_DATE_NOTE,
        "mode" => mode,
        "domain" => domain,
        "schema_version" => TREX_SCHEMA_VERSION,
        "exchange_formula" => "pre-result amended sign: log_alpha=(beta_cold-beta_hot)*(z_hot-z_cold); heuristic, not exact MCMC",
        "betas" => BETAS,
        "tasks" => tasks,
        "seeds" => seeds,
        "arms" => arms,
        "budget" => budget,
        "population_size" => population_size,
        "children_total" => children_total,
        "attempt_multiplier" => attempt_multiplier,
        "genome_length" => genome_length,
        "rows" => rows,
        "aggregate_rows" => agg,
        "overall_by_arm" => overall,
        "paired_delta_vs_elite_ga" => paired_deltas(ok_rows, "elite_ga"),
        "paired_delta_vs_rank_weighted_ga" => paired_deltas(ok_rows, "rank_weighted_ga"),
        "paired_delta_vs_single_hot_beta" => paired_deltas(ok_rows, "single_hot_beta"),
        "paired_delta_vs_temperature_ladder_no_exchange" => paired_deltas(ok_rows, "temperature_ladder_no_exchange"),
        "paired_delta_vs_random_exchange_control" => paired_deltas(ok_rows, "random_exchange_control"),
        "gate" => gate,
        "synthetic_gate" => synth_gate,
        "limitations" => [
            "TREX-HE is a heuristic temperature-exchange search core, not exact replica-exchange MCMC.",
            "This is not TREX-GFN; no learned GFlowNet proposal is used.",
            "Micro results are direction-selection evidence, not SOTA claims.",
        ],
    )
    outname = mode == "synthetic" ? "trex_synthetic_results.jls" : "trex_$(mode)_results.jls"
    out = joinpath(OUTDIR, outname)
    latest = joinpath(OUTDIR, "trex_latest_results.jls")
    save_bundle(out, bundle)
    save_bundle(latest, bundle)
    println("\n", "="^110)
    println("TREX-HE TEMPERATURE-EXCHANGE SUMMARY — ", uppercase(mode))
    println("="^110)
    println(rpad("Arm", 36), rpad("Mean AUC", 14), rpad("Mean Top10", 14), rpad("Tasks/Families", 36))
    println("-"^110)
    for r in overall
        @printf("%-36s%-14.6f%-14.6f%-36s\n", r["arm"], r["mean_auc"], r["mean_top10"], join(r["tasks"], ","))
    end
    if mode == "synthetic"
        println("\nSYNTHETIC DIAGNOSTIC PASS: ", get(synth_gate, "diagnostic_pass", false))
        println("pass tasks: ", get(synth_gate, "pass_tasks", String[]))
        println("fail tasks: ", get(synth_gate, "fail_tasks", String[]))
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
    return bundle
end

mode = String(strip(get(ENV, "TREX_MODE", "smoke")))
run_suite(mode)
