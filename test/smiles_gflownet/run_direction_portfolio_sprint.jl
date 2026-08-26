#!/usr/bin/env julia

# Direction Portfolio Sprint v2
# Synthetic core-primitive screening for remaining structural algorithm-fusion ideas.
# No PMO micro and no GFlowNet/SOTA claim in this runner.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Random
using Serialization
using Statistics
using Dates
using Printf

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUTDIR = joinpath(ROOT, "checkpoints", "direction_portfolio")
const PORT_SCHEMA_VERSION = "direction_portfolio_v2_reaudited_core_screening"
const AUTHORITATIVE_DATE_NOTE = "User supplied authoritative date/time: Saturday, 2026-06-20 01:45 EDT"
mkpath(OUTDIR)

const CANDIDATES = ["FEBB", "EFR", "SGC", "RG", "AIA", "MCTS", "NCI"]
const TIER0_SEEDS_DEFAULT = [17, 23]
const TIER1_SEEDS_DEFAULT = [17, 23, 31]
const TIER0_BUDGET_DEFAULT = 128
const TIER1_BUDGETS_DEFAULT = [256, 1024]

struct SyntheticProblem
    name::String
    length::Int
    alphabet::Int
    candidate::String
    is_antibias::Bool
    params::Dict{String,Any}
end

mutable struct EvalState
    problem::SyntheticProblem
    budget::Int
    calls::Int
    seen::Dict{String,Float64}
    curve::Vector{Dict{String,Any}}
    ledger::Dict{String,Int}
end

function logmsg(msg)
    println("[", Dates.format(now(), "HH:MM:SS"), "] ", msg)
    flush(stdout)
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

function parse_env_int(name::String, default::Int)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    return parse(Int, raw)
end

function bounded_hash_int(x; modulus::Int=10_000_000)
    return Int(hash(x) % UInt(modulus))
end

function increment!(d::Dict{String,Int}, key::String, n::Int=1)
    d[key] = get(d, key, 0) + n
    return nothing
end

function key_of(x::Vector{Int})
    return join(string.(x), "")
end

function rand_candidate(rng::AbstractRNG, p::SyntheticProblem)
    return [rand(rng, 0:(p.alphabet - 1)) for _ in 1:p.length]
end

function mutate_one(x::Vector{Int}, rng::AbstractRNG, p::SyntheticProblem; n_changes::Int=1)
    y = copy(x)
    for _ in 1:n_changes
        idx = rand(rng, 1:length(y))
        old = y[idx]
        if p.alphabet <= 2
            y[idx] = 1 - old
        else
            vals = collect(0:(p.alphabet - 1))
            deleteat!(vals, findfirst(==(old), vals))
            y[idx] = vals[rand(rng, 1:length(vals))]
        end
    end
    return y
end

function crossover(a::Vector{Int}, b::Vector{Int}, rng::AbstractRNG)
    n = min(length(a), length(b))
    n <= 2 && return copy(a)
    cut = rand(rng, 2:(n - 1))
    return vcat(a[1:cut], b[(cut + 1):n])
end

function hamming(a::Vector{Int}, b::Vector{Int})
    n = min(length(a), length(b))
    return count(i -> a[i] != b[i], 1:n) + abs(length(a) - length(b))
end

function topk_mean(vals::Vector{Float64}; k::Int=10)
    isempty(vals) && return 0.0, 0
    s = sort(vals, rev=true)
    n = min(k, length(s))
    return mean(s[1:n]), n
end

function top_stats(state::EvalState)
    vals = collect(values(state.seen))
    top10, denom = topk_mean(vals; k=10)
    return Dict{String,Any}(
        "top1" => isempty(vals) ? 0.0 : maximum(vals),
        "top10_mean" => top10,
        "top10_denominator" => denom,
        "n" => length(vals),
    )
end

function record_curve!(state::EvalState, event::String)
    stats = top_stats(state)
    push!(state.curve, Dict{String,Any}(
        "calls" => state.calls,
        "event" => event,
        "top1" => stats["top1"],
        "top10_mean" => stats["top10_mean"],
        "top10_denominator" => stats["top10_denominator"],
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

# -----------------------------------------------------------------------------
# Synthetic problem definitions and rewards
# -----------------------------------------------------------------------------

function make_problem(name::String)
    if name == "bounded_basin_tree"
        weights = [1.6,1.45,1.3,1.15,1.0,0.9,0.8,0.7,0.62,0.55,0.5,0.45,0.4,0.35,0.3,0.25]
        target = [1,0,1,1,0,1,0,0,1,0,1,0,1,0,1,1]
        return SyntheticProblem(name, 16, 2, "FEBB", false, Dict{String,Any}("weights"=>weights, "target"=>target, "loose"=>false))
    elseif name == "loose_bound_tree"
        weights = fill(1.0, 16)
        target = [1,0,1,1,0,1,0,0,1,0,1,0,1,0,1,1]
        return SyntheticProblem(name, 16, 2, "FEBB", true, Dict{String,Any}("weights"=>weights, "target"=>target, "loose"=>true))
    elseif name == "fragment_grammar_assembly"
        local_scores = [
            [0.10,0.30,0.55,0.20], [0.15,0.52,0.25,0.05], [0.60,0.15,0.30,0.10], [0.05,0.25,0.18,0.58],
            [0.45,0.20,0.38,0.10], [0.12,0.50,0.22,0.08], [0.48,0.10,0.28,0.35], [0.18,0.44,0.12,0.30]
        ]
        allowed = [0,1,2,3]
        return SyntheticProblem(name, 8, 4, "EFR", false, Dict{String,Any}("local_scores"=>local_scores, "allowed"=>allowed, "missing"=>false))
    elseif name == "missing_fragment_assembly"
        local_scores = [
            [0.10,0.20,0.25,0.92], [0.15,0.18,0.28,0.88], [0.12,0.25,0.20,0.90], [0.05,0.30,0.18,0.85],
            [0.20,0.22,0.35,0.89], [0.12,0.25,0.22,0.91], [0.18,0.10,0.28,0.86], [0.18,0.24,0.12,0.87]
        ]
        # EFR intentionally lacks fragment 3 in this anti-bias family.
        allowed = [0,1,2]
        return SyntheticProblem(name, 8, 4, "EFR", true, Dict{String,Any}("local_scores"=>local_scores, "allowed"=>allowed, "missing"=>true))
    elseif name == "planted_block_spin_glass"
        target = vcat(fill(1,4), fill(0,4), fill(1,4), fill(0,4))
        blocks = [collect(1:4), collect(5:8), collect(9:12), collect(13:16)]
        return SyntheticProblem(name, 16, 2, "SGC", false, Dict{String,Any}("target"=>target, "blocks"=>blocks, "dense"=>false))
    elseif name == "dense_frustrated_spin_glass"
        rng = MersenneTwister(260620)
        J = [i == j ? 0.0 : randn(rng) for i in 1:16, j in 1:16]
        J = (J + J') ./ 2
        blocks = [collect(1:4), collect(5:8), collect(9:12), collect(13:16)]
        return SyntheticProblem(name, 16, 2, "SGC", true, Dict{String,Any}("J"=>J, "blocks"=>blocks, "dense"=>true))
    elseif name == "hierarchical_blocks"
        target_coarse = [1,0,1,0]
        blocks = [collect(1:4), collect(5:8), collect(9:12), collect(13:16)]
        return SyntheticProblem(name, 16, 2, "RG", false, Dict{String,Any}("target_coarse"=>target_coarse, "blocks"=>blocks, "misleading"=>false))
    elseif name == "misleading_hierarchy"
        target = [1,0,0,1, 0,1,1,0, 1,1,0,0, 0,0,1,1]
        blocks = [collect(1:4), collect(5:8), collect(9:12), collect(13:16)]
        return SyntheticProblem(name, 16, 2, "RG", true, Dict{String,Any}("target"=>target, "blocks"=>blocks, "misleading"=>true))
    elseif name == "noisy_surrogate_bandit"
        target = [1,0,1,1,0,1,0,0,1,0,1,0]
        return SyntheticProblem(name, 12, 2, "AIA", false, Dict{String,Any}("target"=>target, "miscalibrated"=>false))
    elseif name == "miscalibrated_uncertainty_bandit"
        target = [0,1,0,0,1,0,1,1,0,1,0,1]
        decoy = [1,1,1,1,1,1,0,0,0,0,0,0]
        return SyntheticProblem(name, 12, 2, "AIA", true, Dict{String,Any}("target"=>target, "decoy"=>decoy, "miscalibrated"=>true))
    elseif name == "delayed_reward_tree"
        target = [1,0,1,1,0,0,1,0,1,0,1,1]
        return SyntheticProblem(name, 12, 2, "MCTS", false, Dict{String,Any}("target"=>target, "shallow"=>false))
    elseif name == "shallow_reward_tree"
        target = [1,1,1,1,0,0,0,0,1,1,0,0]
        return SyntheticProblem(name, 12, 2, "MCTS", true, Dict{String,Any}("target"=>target, "shallow"=>true))
    elseif name == "inverse_channel_motif"
        motif_pos = [3,4,5,10,11]
        motif_vals = [1,1,0,1,0]
        return SyntheticProblem(name, 16, 2, "NCI", false, Dict{String,Any}("motif_pos"=>motif_pos, "motif_vals"=>motif_vals, "misspecified"=>false))
    elseif name == "misspecified_channel_motif"
        motif_pos = [2,7,8,14,15]
        motif_vals = [0,1,1,0,1]
        wrong_pos = [3,4,5,10,11]
        wrong_vals = [1,1,0,1,0]
        return SyntheticProblem(name, 16, 2, "NCI", true, Dict{String,Any}("motif_pos"=>motif_pos, "motif_vals"=>motif_vals, "wrong_pos"=>wrong_pos, "wrong_vals"=>wrong_vals, "misspecified"=>true))
    else
        error("Unknown synthetic problem: $(name)")
    end
end

function reward(p::SyntheticProblem, x::Vector{Int})::Float64
    name = p.name
    if name in ("bounded_basin_tree", "loose_bound_tree")
        w = Vector{Float64}(p.params["weights"])
        target = Vector{Int}(p.params["target"])
        base = sum(w[i] * (x[i] == target[i] ? 1.0 : 0.0) for i in 1:p.length) / sum(w)
        motif = all(x[i] == target[i] for i in 1:4) ? 0.12 : 0.0
        return min(1.0, 0.88 * base + motif)
    elseif name in ("fragment_grammar_assembly", "missing_fragment_assembly")
        local_scores = p.params["local_scores"]
        loc = sum(Float64(local_scores[i][x[i] + 1]) for i in 1:p.length) / p.length
        compat = 0.0
        for i in 1:(p.length - 1)
            compat += ((x[i] + x[i+1]) % 2 == 0) ? 0.035 : -0.015
        end
        triad = (x[2] == 1 && x[4] == 3 && x[6] == 1) ? 0.08 : 0.0
        return clamp(loc + compat + triad, 0.0, 1.0)
    elseif name == "planted_block_spin_glass"
        target = Vector{Int}(p.params["target"])
        blocks = p.params["blocks"]
        block_score = 0.0
        for b in blocks
            matches = count(i -> x[i] == target[i], b)
            block_score += matches == length(b) ? 1.0 : 0.55 * matches / length(b)
        end
        return block_score / length(blocks)
    elseif name == "dense_frustrated_spin_glass"
        J = p.params["J"]
        s = [xi == 1 ? 1.0 : -1.0 for xi in x]
        e = 0.0
        for i in 1:length(s), j in (i+1):length(s)
            e += J[i,j] * s[i] * s[j]
        end
        return 1.0 / (1.0 + exp(-e / 10.0))
    elseif name == "hierarchical_blocks"
        target_coarse = Vector{Int}(p.params["target_coarse"])
        blocks = p.params["blocks"]
        coarse = [sum(x[i] for i in b) >= 2 ? 1 : 0 for b in blocks]
        coarse_score = count(i -> coarse[i] == target_coarse[i], 1:length(blocks)) / length(blocks)
        fine_bonus = sum(max(count(i -> x[i] == target_coarse[j], blocks[j]), 4 - count(i -> x[i] == target_coarse[j], blocks[j])) / 4 for j in 1:length(blocks)) / length(blocks)
        return clamp(0.78 * coarse_score + 0.22 * fine_bonus, 0.0, 1.0)
    elseif name == "misleading_hierarchy"
        target = Vector{Int}(p.params["target"])
        exact = count(i -> x[i] == target[i], 1:p.length) / p.length
        parity_bonus = all((sum(x[i] for i in b) % 2) == 0 for b in p.params["blocks"]) ? 0.12 : 0.0
        return clamp(0.88 * exact + parity_bonus, 0.0, 1.0)
    elseif name == "noisy_surrogate_bandit"
        target = Vector{Int}(p.params["target"])
        base = count(i -> x[i] == target[i], 1:p.length) / p.length
        motif = all(x[i] == target[i] for i in 1:4) ? 0.15 : 0.0
        return clamp(0.85 * base + motif, 0.0, 1.0)
    elseif name == "miscalibrated_uncertainty_bandit"
        target = Vector{Int}(p.params["target"])
        decoy = Vector{Int}(p.params["decoy"])
        true_score = count(i -> x[i] == target[i], 1:p.length) / p.length
        decoy_penalty = count(i -> x[i] == decoy[i], 1:p.length) / p.length
        return clamp(0.92 * true_score + 0.08 * (1.0 - decoy_penalty), 0.0, 1.0)
    elseif name == "delayed_reward_tree"
        target = Vector{Int}(p.params["target"])
        prefix = count(i -> x[i] == target[i], 1:4) == 4 ? 0.35 : 0.0
        suffix = count(i -> x[i] == target[i], 5:p.length) / (p.length - 4)
        return clamp(prefix + 0.65 * suffix, 0.0, 1.0)
    elseif name == "shallow_reward_tree"
        target = Vector{Int}(p.params["target"])
        early = count(i -> x[i] == target[i], 1:4) / 4
        rest = count(i -> x[i] == target[i], 5:p.length) / (p.length - 4)
        return clamp(0.65 * early + 0.35 * rest, 0.0, 1.0)
    elseif name in ("inverse_channel_motif", "misspecified_channel_motif")
        pos = Vector{Int}(p.params["motif_pos"])
        vals = Vector{Int}(p.params["motif_vals"])
        motif_match = count(k -> x[pos[k]] == vals[k], 1:length(pos)) / length(pos)
        smooth = sum(x) / max(1, length(x))
        return clamp(0.82 * motif_match + 0.18 * (1.0 - abs(smooth - 0.5) * 2.0), 0.0, 1.0)
    else
        error("No reward for $(name)")
    end
end

function enumerate_all(p::SyntheticProblem)
    total = p.alphabet ^ p.length
    xs = Vector{Int}[]
    sizehint!(xs, total)
    for n in 0:(total - 1)
        y = Vector{Int}(undef, p.length)
        v = n
        for i in 1:p.length
            y[i] = v % p.alphabet
            v ÷= p.alphabet
        end
        push!(xs, y)
    end
    return xs
end

function exact_stats(p::SyntheticProblem)
    xs = enumerate_all(p)
    vals = [reward(p, x) for x in xs]
    top10, _ = topk_mean(vals; k=10)
    return Dict{String,Any}(
        "exact_top1" => maximum(vals),
        "exact_top10_mean" => top10,
        "space_size" => length(vals),
    )
end

# -----------------------------------------------------------------------------
# Evaluation wrapper
# -----------------------------------------------------------------------------

function new_state(p::SyntheticProblem, budget::Int)
    EvalState(p, budget, 0, Dict{String,Float64}(), Dict{String,Any}[], Dict{String,Int}())
end

function evaluate!(state::EvalState, x::Vector{Int}; event::String="eval")
    state.calls >= state.budget && return false
    key = key_of(x)
    if haskey(state.seen, key)
        increment!(state.ledger, "duplicate_candidates")
        return false
    end
    r = reward(state.problem, x)
    state.calls += 1
    state.seen[key] = r
    record_curve!(state, event)
    return true
end

function finalize_result(candidate::String, algorithm::String, p::SyntheticProblem, state::EvalState, exact::Dict{String,Any}; seed::Int, budget::Int, role::String)
    stats = top_stats(state)
    auc = normalized_auc(state.curve, budget)
    return Dict{String,Any}(
        "candidate" => candidate,
        "algorithm" => algorithm,
        "role" => role,
        "problem" => p.name,
        "is_antibias" => p.is_antibias,
        "seed" => seed,
        "budget" => budget,
        "calls" => state.calls,
        "auc_top10" => auc,
        "final_top1" => stats["top1"],
        "final_top10_mean" => stats["top10_mean"],
        "exact_top1" => exact["exact_top1"],
        "exact_top10_mean" => exact["exact_top10_mean"],
        "optimality_gap_top10" => Float64(exact["exact_top10_mean"]) - Float64(stats["top10_mean"]),
        "ledger" => copy(state.ledger),
        "node_expansions" => get(state.ledger, "node_expansions", 0),
        "candidate_generations" => get(state.ledger, "candidate_generations", 0),
        "rollouts" => get(state.ledger, "rollouts", 0),
        "model_fit_calls" => get(state.ledger, "model_fit_calls", 0),
        "wall_clock_seconds" => get(state.ledger, "wall_clock_ms", 0) / 1000.0,
        "schema_version" => PORT_SCHEMA_VERSION,
    )
end

# -----------------------------------------------------------------------------
# Baseline and candidate algorithms
# -----------------------------------------------------------------------------

function run_random(p::SyntheticProblem, budget::Int, seed::Int; algorithm="random_search", candidate="BASE", role="baseline")
    rng = MersenneTwister(seed + bounded_hash_int((p.name, algorithm)))
    st = new_state(p, budget)
    attempts = 0
    while st.calls < budget && attempts < budget * 20
        attempts += 1
        increment!(st.ledger, "candidate_generations")
        evaluate!(st, rand_candidate(rng, p); event=algorithm)
    end
    return finalize_result(candidate, algorithm, p, st, exact_stats(p); seed=seed, budget=budget, role=role)
end

function run_greedy_local(p::SyntheticProblem, budget::Int, seed::Int; algorithm="greedy_local_search", candidate="BASE", role="baseline")
    rng = MersenneTwister(seed + bounded_hash_int((p.name, algorithm)))
    st = new_state(p, budget)
    x = rand_candidate(rng, p)
    evaluate!(st, x; event="init")
    current = x
    current_r = reward(p, current)
    while st.calls < budget
        best = current
        best_r = current_r
        order = shuffle(rng, collect(1:p.length))
        improved = false
        for idx in order
            y = copy(current)
            if p.alphabet == 2
                y[idx] = 1 - y[idx]
            else
                y[idx] = rand(rng, 0:(p.alphabet-1))
            end
            increment!(st.ledger, "candidate_generations")
            evaluate!(st, y; event=algorithm)
            r = reward(p, y)
            if r > best_r
                best = y
                best_r = r
                improved = true
            end
            st.calls >= budget && break
        end
        if improved
            current = best
            current_r = best_r
        else
            current = rand_candidate(rng, p)
            evaluate!(st, current; event="restart")
            current_r = reward(p, current)
        end
    end
    return finalize_result(candidate, algorithm, p, st, exact_stats(p); seed=seed, budget=budget, role=role)
end

function select_parent(pop::Vector{Vector{Int}}, scores::Vector{Float64}, rng::AbstractRNG; rank_weighted::Bool=false)
    isempty(pop) && error("empty population")
    if !rank_weighted
        # tournament selection
        ids = rand(rng, 1:length(pop), min(3, length(pop)))
        best = ids[argmax(scores[ids])]
        return pop[best]
    end
    order = sortperm(scores, rev=true)
    weights = [1.0 / i for i in 1:length(order)]
    weights ./= sum(weights)
    cum = cumsum(weights)
    rank = searchsortedfirst(cum, rand(rng))
    return pop[order[clamp(rank, 1, length(order))]]
end

function run_ga(p::SyntheticProblem, budget::Int, seed::Int; algorithm="ga_mutation_crossover", candidate="BASE", role="baseline", rank_weighted::Bool=false)
    rng = MersenneTwister(seed + bounded_hash_int((p.name, algorithm)))
    st = new_state(p, budget)
    pop_size = min(24, max(8, budget ÷ 8))
    pop = Vector{Int}[]
    while length(pop) < pop_size && st.calls < budget
        x = rand_candidate(rng, p)
        if evaluate!(st, x; event="init")
            push!(pop, x)
        end
    end
    attempts = 0
    while st.calls < budget && !isempty(pop) && attempts < budget * 100
        attempts += 1
        scores = [reward(p, x) for x in pop]
        p1 = select_parent(pop, scores, rng; rank_weighted=rank_weighted)
        p2 = select_parent(pop, scores, rng; rank_weighted=rank_weighted)
        child = rand(rng) < 0.45 ? crossover(p1, p2, rng) : copy(p1)
        changes = rand(rng) < 0.2 ? 2 : 1
        child = mutate_one(child, rng, p; n_changes=changes)
        if haskey(st.seen, key_of(child)) && rand(rng) < 0.35
            child = rand_candidate(rng, p)
        end
        increment!(st.ledger, "candidate_generations")
        if evaluate!(st, child; event=algorithm)
            push!(pop, child)
            # Keep top pop_size by true reward observed.
            sort!(pop, by = x -> -reward(p, x))
            if length(pop) > pop_size
                pop = pop[1:pop_size]
            end
        end
    end
    attempts >= budget * 100 && increment!(st.ledger, "attempt_cap_hits")
    return finalize_result(candidate, algorithm, p, st, exact_stats(p); seed=seed, budget=budget, role=role)
end

function run_simulated_annealing(p::SyntheticProblem, budget::Int, seed::Int; algorithm="simulated_annealing", candidate="BASE", role="baseline")
    rng = MersenneTwister(seed + bounded_hash_int((p.name, algorithm)))
    st = new_state(p, budget)
    x = rand_candidate(rng, p)
    evaluate!(st, x; event="init")
    rx = reward(p, x)
    t0 = 0.25
    attempts = 0
    while st.calls < budget && attempts < budget * 100
        attempts += 1
        temp = max(0.01, t0 * (1.0 - st.calls / max(1, budget)))
        y = mutate_one(x, rng, p; n_changes=1)
        if haskey(st.seen, key_of(y)) && rand(rng) < 0.35
            y = rand_candidate(rng, p)
        end
        increment!(st.ledger, "candidate_generations")
        evaluate!(st, y; event=algorithm)
        ry = reward(p, y)
        if ry >= rx || rand(rng) < exp((ry - rx) / temp)
            x, rx = y, ry
        end
    end
    attempts >= budget * 100 && increment!(st.ledger, "attempt_cap_hits")
    return finalize_result(candidate, algorithm, p, st, exact_stats(p); seed=seed, budget=budget, role=role)
end

# Prefix upper bound for FEBB.
function prefix_reward_bound(p::SyntheticProblem, prefix::Vector{Int})
    if get(p.params, "loose", false)
        return 1.0
    end
    w = Vector{Float64}(p.params["weights"])
    target = Vector{Int}(p.params["target"])
    k = length(prefix)
    matched = k == 0 ? 0.0 : sum(w[i] * (prefix[i] == target[i] ? 1.0 : 0.0) for i in 1:k)
    remaining = k >= length(w) ? 0.0 : sum(w[(k+1):end])
    motif_possible = true
    for i in 1:min(k, 4)
        if prefix[i] != target[i]
            motif_possible = false
            break
        end
    end
    motif = motif_possible ? 0.12 : 0.0
    return min(1.0, 0.88 * (matched + remaining) / sum(w) + motif)
end

function run_febb(p::SyntheticProblem, budget::Int, seed::Int; pruning::Bool=true, algorithm::String=pruning ? "febb_pruning" : "beam_no_pruning", role::String=pruning ? "candidate" : "ablation")
    st = new_state(p, budget)
    exact = exact_stats(p)
    frontier = [Int[]]
    incumbent = -Inf
    while st.calls < budget && !isempty(frontier)
        bounds = [prefix_reward_bound(p, pref) for pref in frontier]
        idx = argmax(bounds)
        pref = frontier[idx]
        ub = bounds[idx]
        deleteat!(frontier, idx)
        increment!(st.ledger, "node_expansions")
        if pruning && ub <= incumbent
            increment!(st.ledger, "pruned_nodes")
            continue
        end
        if length(pref) == p.length
            if evaluate!(st, pref; event=algorithm)
                incumbent = max(incumbent, reward(p, pref))
            end
        else
            for a in 0:(p.alphabet - 1)
                child = vcat(pref, [a])
                if !pruning || prefix_reward_bound(p, child) > incumbent
                    push!(frontier, child)
                else
                    increment!(st.ledger, "pruned_nodes")
                end
            end
        end
        if length(frontier) > 20_000
            # Keep high-bound frontier only to avoid pathological memory blowup in loose-bound anti-bias.
            b = [prefix_reward_bound(p, pref) for pref in frontier]
            order = sortperm(b, rev=true)[1:10_000]
            frontier = frontier[order]
            increment!(st.ledger, "frontier_truncations")
        end
    end
    return finalize_result("FEBB", algorithm, p, st, exact; seed=seed, budget=budget, role=role)
end

function efr_model_score(p::SyntheticProblem, x::Vector{Int})
    # The exact recombination model only knows local terms and simple compatibility; it is not allowed to query objective for sorting.
    local_scores = p.params["local_scores"]
    loc = sum(Float64(local_scores[i][x[i] + 1]) for i in 1:p.length) / p.length
    compat = sum(((x[i] + x[i+1]) % 2 == 0) ? 0.025 : -0.01 for i in 1:(p.length-1))
    return loc + compat
end

function run_efr(p::SyntheticProblem, budget::Int, seed::Int; algorithm="efr_exact_recombination", role="candidate")
    st = new_state(p, budget)
    exact = exact_stats(p)
    allowed = Vector{Int}(p.params["allowed"])
    candidates = Vector{Int}[]
    function rec(prefix, pos)
        if pos > p.length
            push!(candidates, copy(prefix))
            return
        end
        for a in allowed
            # Simple compatibility constraint: avoid repeated fragment 0 three times in a row.
            if length(prefix) >= 2 && prefix[end] == 0 && prefix[end-1] == 0 && a == 0
                continue
            end
            push!(prefix, a)
            rec(prefix, pos + 1)
            pop!(prefix)
        end
    end
    rec(Int[], 1)
    sort!(candidates, by = x -> -efr_model_score(p, x))
    for x in candidates
        st.calls >= budget && break
        increment!(st.ledger, "candidate_generations")
        evaluate!(st, x; event=algorithm)
    end
    st.ledger["grammar_candidate_count"] = length(candidates)
    st.ledger["grammar_incomplete"] = get(p.params, "missing", false) ? 1 : 0
    return finalize_result("EFR", algorithm, p, st, exact; seed=seed, budget=budget, role=role)
end

function run_blind_fragment_ga(p::SyntheticProblem, budget::Int, seed::Int)
    return run_ga(p, budget, seed; algorithm="blind_fragment_ga", candidate="EFR", role="ablation", rank_weighted=false)
end

function run_sgc(p::SyntheticProblem, budget::Int, seed::Int; cluster::Bool=true, algorithm::String=cluster ? "sgc_cluster_moves" : "local_no_cluster", role::String=cluster ? "candidate" : "ablation")
    rng = MersenneTwister(seed + bounded_hash_int((p.name, algorithm)))
    st = new_state(p, budget)
    exact = exact_stats(p)
    blocks = p.params["blocks"]
    pop = Vector{Int}[]
    for _ in 1:12
        x = rand_candidate(rng, p)
        evaluate!(st, x; event="init") && push!(pop, x)
        st.calls >= budget && break
    end
    attempts = 0
    while st.calls < budget && !isempty(pop) && attempts < budget * 100
        attempts += 1
        parent = pop[rand(rng, 1:length(pop))]
        child = copy(parent)
        if cluster
            b = blocks[rand(rng, 1:length(blocks))]
            for idx in b
                child[idx] = 1 - child[idx]
            end
            increment!(st.ledger, "cluster_moves")
        else
            idx = rand(rng, 1:p.length)
            child[idx] = 1 - child[idx]
            increment!(st.ledger, "local_moves")
        end
        if haskey(st.seen, key_of(child)) && rand(rng) < 0.50
            child = rand_candidate(rng, p)
        end
        increment!(st.ledger, "candidate_generations")
        if evaluate!(st, child; event=algorithm)
            push!(pop, child)
            sort!(pop, by = x -> -reward(p, x))
            pop = pop[1:min(24, length(pop))]
        end
    end
    attempts >= budget * 100 && increment!(st.ledger, "attempt_cap_hits")
    return finalize_result("SGC", algorithm, p, st, exact; seed=seed, budget=budget, role=role)
end

function run_rg(p::SyntheticProblem, budget::Int, seed::Int; coarse::Bool=true, algorithm::String=coarse ? "rg_coarse_to_fine" : "fine_only_search", role::String=coarse ? "candidate" : "ablation")
    rng = MersenneTwister(seed + bounded_hash_int((p.name, algorithm)))
    st = new_state(p, budget)
    exact = exact_stats(p)
    blocks = p.params["blocks"]
    if coarse
        # Enumerate coarse assignments and evaluate representatives first.
        coarse_states = Vector{Int}[]
        for n in 0:(2^length(blocks)-1)
            c = [(n >> (i-1)) & 1 for i in 1:length(blocks)]
            push!(coarse_states, c)
        end
        reps = Vector{Int}[]
        for c in coarse_states
            x = zeros(Int, p.length)
            for (j,b) in enumerate(blocks)
                for idx in b
                    x[idx] = c[j]
                end
            end
            push!(reps, x)
        end
        sort!(reps, by = x -> -reward(p, x))
        for x in reps
            st.calls >= min(budget, 32) && break
            evaluate!(st, x; event="coarse")
        end
        seeds = reps[1:min(4, length(reps))]
        attempts = 0
        while st.calls < budget && attempts < budget * 100
            attempts += 1
            base = seeds[rand(rng, 1:length(seeds))]
            child = mutate_one(base, rng, p; n_changes=rand(rng, 1:3))
            if haskey(st.seen, key_of(child)) && rand(rng) < 0.50
                child = rand_candidate(rng, p)
            end
            increment!(st.ledger, "candidate_generations")
            evaluate!(st, child; event=algorithm)
        end
        attempts >= budget * 100 && increment!(st.ledger, "attempt_cap_hits")
    else
        # Same budget but no coarse representatives.
        attempts = 0
        while st.calls < budget && attempts < budget * 100
            attempts += 1
            x = rand_candidate(rng, p)
            for _ in 1:rand(rng, 0:3)
                x = mutate_one(x, rng, p; n_changes=1)
            end
            increment!(st.ledger, "candidate_generations")
            evaluate!(st, x; event=algorithm)
        end
        attempts >= budget * 100 && increment!(st.ledger, "attempt_cap_hits")
    end
    return finalize_result("RG", algorithm, p, st, exact; seed=seed, budget=budget, role=role)
end

function knn_predict(observed::Vector{Vector{Int}}, scores::Vector{Float64}, x::Vector{Int}; miscalibrated::Bool=false)
    if isempty(observed)
        return 0.5, 1.0
    end
    dists = [hamming(o, x) for o in observed]
    order = sortperm(dists)
    k = min(8, length(order))
    idx = order[1:k]
    weights = [1.0 / (1.0 + dists[i]) for i in idx]
    μ = sum(weights[j] * scores[idx[j]] for j in 1:k) / sum(weights)
    nearest = minimum(dists)
    σ = nearest / max(1, length(x))
    if miscalibrated
        # Overconfident around decoy-like high-density regions.
        σ *= 0.25
    end
    return μ, σ
end

function run_acquisition(p::SyntheticProblem, budget::Int, seed::Int; algorithm="aia_diverse_acquisition", role="candidate")
    rng = MersenneTwister(seed + bounded_hash_int((p.name, algorithm)))
    st = new_state(p, budget)
    exact = exact_stats(p)
    pool = enumerate_all(p)
    shuffle!(rng, pool)
    observed = Vector{Int}[]
    scores = Float64[]
    init_n = min(16, budget)
    for i in 1:init_n
        x = pool[i]
        if evaluate!(st, x; event="init")
            push!(observed, x); push!(scores, reward(p, x))
        end
    end
    miscal = get(p.params, "miscalibrated", false)
    attempts = 0
    while st.calls < budget && attempts < budget * 100
        attempts += 1
        increment!(st.ledger, "model_fit_calls")
        sample_pool = pool[1:min(length(pool), 512)]
        best_x = nothing
        best_a = -Inf
        for x in sample_pool
            haskey(st.seen, key_of(x)) && continue
            μ, σ = knn_predict(observed, scores, x; miscalibrated=miscal)
            a = if algorithm == "aia_diverse_acquisition"
                # UCB plus distance-to-observed diversity. This must beat UCB/TS to avoid rebranding.
                diversity = isempty(observed) ? 1.0 : minimum(hamming(o, x) for o in observed) / p.length
                μ + 1.2 * σ + 0.15 * diversity
            elseif algorithm == "ucb_baseline"
                μ + 1.5 * σ
            elseif algorithm == "thompson_sampling_baseline"
                μ + σ * randn(rng)
            else # greedy surrogate
                μ
            end
            if a > best_a
                best_a = a
                best_x = x
            end
        end
        if best_x === nothing
            for x in shuffle(rng, pool)
                if !haskey(st.seen, key_of(x))
                    best_x = x
                    break
                end
            end
        end
        best_x === nothing && break
        increment!(st.ledger, "candidate_generations")
        if evaluate!(st, best_x; event=algorithm)
            push!(observed, best_x); push!(scores, reward(p, best_x))
        end
    end
    attempts >= budget * 100 && increment!(st.ledger, "attempt_cap_hits")
    cand = algorithm == "aia_diverse_acquisition" ? "AIA" : "AIA"
    role2 = algorithm == "aia_diverse_acquisition" ? role : "baseline"
    return finalize_result(cand, algorithm, p, st, exact; seed=seed, budget=budget, role=role2)
end

function tree_prefix_bound(p::SyntheticProblem, prefix::Vector{Int})
    target = Vector{Int}(p.params["target"])
    k = length(prefix)
    if get(p.params, "shallow", false)
        matched = count(i -> prefix[i] == target[i], 1:k)
        return (matched + (p.length - k)) / p.length
    else
        # Delayed: early prefix only pays if perfect; bound remains optimistic until contradicted.
        early_ok = true
        for i in 1:min(k, 4)
            if prefix[i] != target[i]
                early_ok = false
                break
            end
        end
        suffix_matched = k > 4 ? count(i -> prefix[i] == target[i], 5:k) : 0
        suffix_possible = max(0, p.length - max(k, 4))
        return (early_ok ? 0.35 : 0.0) + 0.65 * (suffix_matched + suffix_possible) / (p.length - 4)
    end
end

function run_beam_prefix(p::SyntheticProblem, budget::Int, seed::Int; algorithm="beam_search", candidate="BASE", role="baseline", beam_width::Int=16)
    st = new_state(p, budget)
    exact = exact_stats(p)
    frontier = [Int[]]
    while st.calls < budget && !isempty(frontier)
        newfront = Vector{Int}[]
        for pref in frontier
            if length(pref) == p.length
                evaluate!(st, pref; event=algorithm)
                st.calls >= budget && break
            else
                for a in 0:(p.alphabet-1)
                    push!(newfront, vcat(pref, [a]))
                    increment!(st.ledger, "node_expansions")
                end
            end
        end
        isempty(newfront) && break
        sort!(newfront, by = pref -> -tree_prefix_bound(p, pref))
        frontier = newfront[1:min(beam_width, length(newfront))]
    end
    return finalize_result(candidate, algorithm, p, st, exact; seed=seed, budget=budget, role=role)
end

function run_mcts(p::SyntheticProblem, budget::Int, seed::Int; algorithm="mcts_lookahead", role="candidate")
    # Minimal UCT-like prefix search. The candidate uses tree bound as rollout prior; plain_mcts uses random rollout.
    rng = MersenneTwister(seed + bounded_hash_int((p.name, algorithm)))
    st = new_state(p, budget)
    exact = exact_stats(p)
    prefixes = [Int[]]
    visits = Dict{String,Int}()
    value_sums = Dict{String,Float64}()
    attempts = 0
    while st.calls < budget && attempts < budget * 100
        attempts += 1
        # Select prefix with optimistic score.
        best_pref = prefixes[1]
        best_score = -Inf
        total_visits = sum(Base.values(visits); init=0) + 1
        for pref in prefixes
            k = key_of(pref)
            v = get(visits, k, 0)
            q = get(value_sums, k, 0.0) / max(1, v)
            prior = algorithm == "mcts_lookahead" ? tree_prefix_bound(p, pref) : 0.5
            u = q + 0.7 * prior + 0.5 * sqrt(log(total_visits + 1) / max(1, v + 1))
            if u > best_score
                best_score = u
                best_pref = pref
            end
        end
        # Expand/rollout.
        pref = copy(best_pref)
        while length(pref) < p.length
            if algorithm == "mcts_lookahead"
                # Mostly choose next bit by prefix bound, but add small exploration to avoid duplicate rollouts.
                options = [vcat(pref, [a]) for a in 0:(p.alphabet-1)]
                pref = rand(rng) < 0.15 ? options[rand(rng, 1:length(options))] : options[argmax([tree_prefix_bound(p, o) for o in options])]
            else
                push!(pref, rand(rng, 0:(p.alphabet-1)))
            end
            increment!(st.ledger, "node_expansions")
            if length(pref) < p.length
                push!(prefixes, copy(pref))
            end
        end
        if haskey(st.seen, key_of(pref)) && rand(rng) < 0.60
            pref = rand_candidate(rng, p)
        end
        increment!(st.ledger, "rollouts")
        evaluate!(st, pref; event=algorithm)
        r = reward(p, pref)
        # Backprop to prefixes on path.
        for klen in 0:p.length
            pr = pref[1:klen]
            kk = key_of(pr)
            visits[kk] = get(visits, kk, 0) + 1
            value_sums[kk] = get(value_sums, kk, 0.0) + r
        end
    end
    attempts >= budget * 100 && increment!(st.ledger, "attempt_cap_hits")
    role2 = algorithm == "mcts_lookahead" ? role : "baseline"
    return finalize_result("MCTS", algorithm, p, st, exact; seed=seed, budget=budget, role=role2)
end

function nci_posterior_score(p::SyntheticProblem, x::Vector{Int}; randomized::Bool=false)
    pos = if get(p.params, "misspecified", false)
        Vector{Int}(p.params["wrong_pos"])
    else
        Vector{Int}(p.params["motif_pos"])
    end
    vals = if get(p.params, "misspecified", false)
        Vector{Int}(p.params["wrong_vals"])
    else
        Vector{Int}(p.params["motif_vals"])
    end
    if randomized
        vals = 1 .- vals
    end
    return count(k -> x[pos[k]] == vals[k], 1:length(pos)) / length(pos)
end

function run_nci(p::SyntheticProblem, budget::Int, seed::Int; algorithm="nci_inverse_posterior", role="candidate")
    rng = MersenneTwister(seed + bounded_hash_int((p.name, algorithm)))
    st = new_state(p, budget)
    exact = exact_stats(p)
    pool = enumerate_all(p)
    if algorithm == "nci_inverse_posterior"
        sort!(pool, by = x -> -nci_posterior_score(p, x; randomized=false))
    elseif algorithm == "random_target_posterior"
        sort!(pool, by = x -> -nci_posterior_score(p, x; randomized=true))
    elseif algorithm == "forward_prior_rerank"
        shuffle!(rng, pool)
        pool = pool[1:min(length(pool), budget * 8)]
        sort!(pool, by = x -> -nci_posterior_score(p, x; randomized=false))
    else
        shuffle!(rng, pool)
    end
    for x in pool
        st.calls >= budget && break
        increment!(st.ledger, "candidate_generations")
        evaluate!(st, x; event=algorithm)
    end
    role2 = algorithm == "nci_inverse_posterior" ? role : "baseline"
    return finalize_result("NCI", algorithm, p, st, exact; seed=seed, budget=budget, role=role2)
end

# -----------------------------------------------------------------------------
# Portfolio orchestration
# -----------------------------------------------------------------------------

const TIER0_PROBLEMS = Dict(
    "FEBB" => ["bounded_basin_tree", "loose_bound_tree"],
    "EFR" => ["fragment_grammar_assembly", "missing_fragment_assembly"],
    "SGC" => ["planted_block_spin_glass", "dense_frustrated_spin_glass"],
    "RG" => ["hierarchical_blocks", "misleading_hierarchy"],
    "AIA" => ["noisy_surrogate_bandit", "miscalibrated_uncertainty_bandit"],
    "MCTS" => ["delayed_reward_tree", "shallow_reward_tree"],
    "NCI" => ["inverse_channel_motif", "misspecified_channel_motif"],
)

function algorithms_for(candidate::String)
    if candidate == "FEBB"
        return [
            ("febb_pruning", (p,b,s)->run_febb(p,b,s; pruning=true)),
            ("beam_no_pruning", (p,b,s)->run_febb(p,b,s; pruning=false)),
            ("ga_mutation_crossover", (p,b,s)->run_ga(p,b,s; algorithm="ga_mutation_crossover", candidate="FEBB", role="baseline")),
            ("random_search", (p,b,s)->run_random(p,b,s; candidate="FEBB")),
        ]
    elseif candidate == "EFR"
        return [
            ("efr_exact_recombination", (p,b,s)->run_efr(p,b,s)),
            ("blind_fragment_ga", (p,b,s)->run_blind_fragment_ga(p,b,s)),
            ("rank_weighted_ga", (p,b,s)->run_ga(p,b,s; algorithm="rank_weighted_ga", candidate="EFR", role="baseline", rank_weighted=true)),
            ("random_search", (p,b,s)->run_random(p,b,s; candidate="EFR")),
        ]
    elseif candidate == "SGC"
        return [
            ("sgc_cluster_moves", (p,b,s)->run_sgc(p,b,s; cluster=true)),
            ("local_no_cluster", (p,b,s)->run_sgc(p,b,s; cluster=false)),
            ("ga_mutation_crossover", (p,b,s)->run_ga(p,b,s; algorithm="ga_mutation_crossover", candidate="SGC", role="baseline")),
            ("simulated_annealing", (p,b,s)->run_simulated_annealing(p,b,s; candidate="SGC")),
            ("random_search", (p,b,s)->run_random(p,b,s; candidate="SGC")),
        ]
    elseif candidate == "RG"
        return [
            ("rg_coarse_to_fine", (p,b,s)->run_rg(p,b,s; coarse=true)),
            ("fine_only_search", (p,b,s)->run_rg(p,b,s; coarse=false)),
            ("ga_mutation_crossover", (p,b,s)->run_ga(p,b,s; algorithm="ga_mutation_crossover", candidate="RG", role="baseline")),
            ("random_search", (p,b,s)->run_random(p,b,s; candidate="RG")),
        ]
    elseif candidate == "AIA"
        return [
            ("aia_diverse_acquisition", (p,b,s)->run_acquisition(p,b,s; algorithm="aia_diverse_acquisition")),
            ("ucb_baseline", (p,b,s)->run_acquisition(p,b,s; algorithm="ucb_baseline")),
            ("thompson_sampling_baseline", (p,b,s)->run_acquisition(p,b,s; algorithm="thompson_sampling_baseline")),
            ("greedy_surrogate", (p,b,s)->run_acquisition(p,b,s; algorithm="greedy_surrogate")),
            ("random_search", (p,b,s)->run_random(p,b,s; candidate="AIA")),
        ]
    elseif candidate == "MCTS"
        return [
            ("mcts_lookahead", (p,b,s)->run_mcts(p,b,s; algorithm="mcts_lookahead")),
            ("plain_mcts", (p,b,s)->run_mcts(p,b,s; algorithm="plain_mcts")),
            ("beam_search", (p,b,s)->run_beam_prefix(p,b,s; candidate="MCTS")),
            ("random_search", (p,b,s)->run_random(p,b,s; candidate="MCTS")),
        ]
    elseif candidate == "NCI"
        return [
            ("nci_inverse_posterior", (p,b,s)->run_nci(p,b,s; algorithm="nci_inverse_posterior")),
            ("forward_prior_rerank", (p,b,s)->run_nci(p,b,s; algorithm="forward_prior_rerank")),
            ("random_target_posterior", (p,b,s)->run_nci(p,b,s; algorithm="random_target_posterior")),
            ("random_search", (p,b,s)->run_random(p,b,s; candidate="NCI")),
        ]
    else
        error("Unknown candidate $(candidate)")
    end
end

function primary_algorithm(candidate::String)
    return Dict(
        "FEBB"=>"febb_pruning",
        "EFR"=>"efr_exact_recombination",
        "SGC"=>"sgc_cluster_moves",
        "RG"=>"rg_coarse_to_fine",
        "AIA"=>"aia_diverse_acquisition",
        "MCTS"=>"mcts_lookahead",
        "NCI"=>"nci_inverse_posterior",
    )[candidate]
end

function nearest_ablation(candidate::String)
    return Dict(
        "FEBB"=>"beam_no_pruning",
        "EFR"=>"blind_fragment_ga",
        "SGC"=>"local_no_cluster",
        "RG"=>"fine_only_search",
        "AIA"=>"ucb_baseline",
        "MCTS"=>"plain_mcts",
        "NCI"=>"forward_prior_rerank",
    )[candidate]
end

function run_rows(candidates::Vector{String}, budgets::Vector{Int}, seeds::Vector{Int})
    rows = Dict{String,Any}[]
    for cand in candidates
        for pname in TIER0_PROBLEMS[cand]
            p = make_problem(pname)
            for budget in budgets, seed in seeds
                for (alg, runner) in algorithms_for(cand)
                    logmsg("RUN cand=$(cand) problem=$(pname) alg=$(alg) budget=$(budget) seed=$(seed)")
                    start = time()
                    try
                        row = runner(p, budget, seed)
                        row["elapsed_sec"] = time() - start
                        push!(rows, row)
                        logmsg("OK cand=$(cand) problem=$(pname) alg=$(alg) auc=$(round(row["auc_top10"], digits=5)) top10=$(round(row["final_top10_mean"], digits=5))")
                    catch e
                        bt = catch_backtrace()
                        err = sprint(showerror, e, bt)
                        logmsg("FAILED cand=$(cand) problem=$(pname) alg=$(alg)")
                        println(err)
                        push!(rows, Dict{String,Any}(
                            "candidate"=>cand, "algorithm"=>alg, "role"=>"failed", "problem"=>pname,
                            "is_antibias"=>p.is_antibias, "seed"=>seed, "budget"=>budget,
                            "status"=>"failed", "error"=>err, "auc_top10"=>NaN, "final_top10_mean"=>NaN,
                            "exact_top10_mean"=>exact_stats(p)["exact_top10_mean"], "node_expansions"=>0,
                            "candidate_generations"=>0, "rollouts"=>0, "model_fit_calls"=>0,
                            "wall_clock_seconds"=>time()-start,
                        ))
                    end
                end
            end
        end
    end
    return rows
end

function group_rows(rows, keys)
    d = Dict{Tuple,Vector{Dict{String,Any}}}()
    for r in rows
        haskey(r, "status") && r["status"] == "failed" && continue
        key = Tuple([r[k] for k in keys])
        d[key] = get(d, key, Dict{String,Any}[])
        push!(d[key], r)
    end
    return d
end

function mean_float(vals)
    isempty(vals) && return NaN
    return mean(Float64.(vals))
end

function summarize_algorithm(rows)
    groups = group_rows(rows, ["candidate", "problem", "algorithm", "budget"])
    out = Dict{String,Any}[]
    for (key, vals) in groups
        cand, problem, alg, budget = key
        aucs = [v["auc_top10"] for v in vals]
        top10 = [v["final_top10_mean"] for v in vals]
        exact = [v["exact_top10_mean"] for v in vals]
        costs = [get(v, "node_expansions", 0) + get(v, "candidate_generations", 0) + get(v, "rollouts", 0) for v in vals]
        push!(out, Dict{String,Any}(
            "candidate" => cand,
            "problem" => problem,
            "algorithm" => alg,
            "budget" => budget,
            "n" => length(vals),
            "auc_mean" => mean_float(aucs),
            "top10_mean" => mean_float(top10),
            "exact_top10_mean" => mean_float(exact),
            "hidden_cost_mean" => mean_float(costs),
            "is_antibias" => vals[1]["is_antibias"],
        ))
    end
    sort!(out, by = r -> (String(r["candidate"]), String(r["problem"]), String(r["algorithm"]), Int(r["budget"])))
    return out
end

function max_or_nan(vals)
    xs = [Float64(v) for v in vals if isfinite(Float64(v))]
    isempty(xs) && return NaN
    return maximum(xs)
end

function candidate_gate(rows)
    summaries = summarize_algorithm(rows)
    bycand = Dict{String,Vector{Dict{String,Any}}}()
    for s in summaries
        cand = String(s["candidate"])
        bycand[cand] = get(bycand, cand, Dict{String,Any}[])
        push!(bycand[cand], s)
    end
    verdicts = Dict{String,Any}[]
    for cand in sort(collect(keys(bycand)))
        primary = primary_algorithm(cand)
        ablation = nearest_ablation(cand)
        ss = bycand[cand]
        intended = [s for s in ss if !Bool(s["is_antibias"])]
        antibias = [s for s in ss if Bool(s["is_antibias"])]
        prim_intended = [s for s in intended if String(s["algorithm"]) == primary]
        abl_intended = [s for s in intended if String(s["algorithm"]) == ablation]
        prim_anti = [s for s in antibias if String(s["algorithm"]) == primary]
        baseline_intended = [s for s in intended if String(s["algorithm"]) != primary]
        baseline_anti = [s for s in antibias if String(s["algorithm"]) != primary]
        primary_auc = mean_float([s["auc_mean"] for s in prim_intended])
        ablation_auc = mean_float([s["auc_mean"] for s in abl_intended])
        best_baseline_auc = max_or_nan([s["auc_mean"] for s in baseline_intended])
        best_anti_baseline = max_or_nan([s["auc_mean"] for s in baseline_anti])
        primary_anti_auc = mean_float([s["auc_mean"] for s in prim_anti])
        exact_auc = mean_float([s["exact_top10_mean"] for s in prim_intended])
        denom = max(1e-8, exact_auc - best_baseline_auc)
        nrr = (primary_auc - best_baseline_auc) / denom
        ablation_pass = isfinite(primary_auc) && isfinite(ablation_auc) && primary_auc > ablation_auc
        beats_best = isfinite(primary_auc) && isfinite(best_baseline_auc) && primary_auc > 1.05 * best_baseline_auc
        anti_survival = !isfinite(primary_anti_auc) || !isfinite(best_anti_baseline) ? false : primary_anti_auc >= 0.95 * best_anti_baseline
        rebranding_penalty = !ablation_pass
        # Hidden cost penalty: compare primary intended cost to best baseline intended cost.
        prim_cost = mean_float([s["hidden_cost_mean"] for s in prim_intended])
        base_cost = mean_float([s["hidden_cost_mean"] for s in baseline_intended])
        hidden_cost_bad = isfinite(prim_cost) && isfinite(base_cost) && prim_cost > 2.0 * max(1.0, base_cost) && nrr < 0.10
        structural_score = 0.0
        structural_score += ablation_pass ? 1.0 : -1.0
        structural_score += beats_best ? 1.0 : -0.5
        structural_score += clamp(nrr, -1.0, 1.0)
        structural_score += anti_survival ? 0.5 : -0.75
        structural_score += hidden_cost_bad ? -1.0 : 0.0
        label = if rebranding_penalty
            "REBRANDED_BASELINE"
        elseif hidden_cost_bad
            "STOP_HIDDEN_COST"
        elseif beats_best && anti_survival && nrr > 0.10
            "PROMOTE_TO_PMO_PLAN"
        elseif ablation_pass && nrr > 0.0
            "SYNTHETIC_ONLY_INTERESTING"
        else
            "STOP_OR_REDESIGN"
        end
        reasons = String[]
        ablation_pass || push!(reasons, "primary <= nearest ablation ($(ablation)); rebranding risk.")
        beats_best || push!(reasons, "primary did not beat best intended-family baseline by >=5%.")
        anti_survival || push!(reasons, "anti-bias holdout failed or missing.")
        hidden_cost_bad && push!(reasons, "hidden expansion/generation cost >2x baseline without enough NRR gain.")
        push!(verdicts, Dict{String,Any}(
            "candidate" => cand,
            "primary_algorithm" => primary,
            "nearest_ablation" => ablation,
            "primary_intended_auc" => primary_auc,
            "ablation_intended_auc" => ablation_auc,
            "best_baseline_intended_auc" => best_baseline_auc,
            "primary_antibias_auc" => primary_anti_auc,
            "best_antibias_baseline_auc" => best_anti_baseline,
            "intended_nrr" => nrr,
            "ablation_pass" => ablation_pass,
            "beats_best_by_5pct" => beats_best,
            "anti_bias_survival" => anti_survival,
            "hidden_cost_bad" => hidden_cost_bad,
            "structural_score" => structural_score,
            "label" => label,
            "reasons" => reasons,
        ))
    end
    sort!(verdicts, by = v -> -Float64(v["structural_score"]))
    return verdicts
end

function save_bundle(path::String, bundle::Dict{String,Any})
    tmp = path * ".tmp"
    serialize(tmp, bundle)
    mv(tmp, path; force=true)
    return path
end

function print_summary(verdicts, summaries, mode)
    println("\n", "="^118)
    println("DIRECTION PORTFOLIO SUMMARY — ", uppercase(mode))
    println("="^118)
    println(rpad("Candidate", 12), rpad("Label", 28), rpad("Score", 10), rpad("Intended AUC", 15), rpad("Ablation AUC", 15), rpad("Best Base", 15), "Reasons")
    println("-"^118)
    for v in verdicts
        @printf("%-12s%-28s%-10.3f%-15.6f%-15.6f%-15.6f%s\n",
            v["candidate"], v["label"], v["structural_score"], v["primary_intended_auc"], v["ablation_intended_auc"], v["best_baseline_intended_auc"], join(v["reasons"], "; "))
    end
end

function run_mode(mode::String)
    if mode == "compile"
        println("Direction portfolio compile OK: ", PORT_SCHEMA_VERSION)
        return nothing
    end
    requested_candidates = parse_csv_strings("PORT_CANDIDATES", CANDIDATES)
    invalid = [c for c in requested_candidates if !(c in CANDIDATES)]
    isempty(invalid) || error("Unknown candidates: $(invalid)")
    if mode == "tier0"
        budgets = [parse_env_int("PORT_BUDGET", TIER0_BUDGET_DEFAULT)]
        seeds = parse_csv_ints("PORT_SEEDS", TIER0_SEEDS_DEFAULT)
    elseif mode == "tier1"
        raw_budgets = parse_csv_ints("PORT_BUDGETS", TIER1_BUDGETS_DEFAULT)
        budgets = raw_budgets
        seeds = parse_csv_ints("PORT_SEEDS", TIER1_SEEDS_DEFAULT)
        # If a tier0 rank exists, default to survivors/promotable-interesting candidates, capped at top 4.
        if get(ENV, "PORT_CANDIDATES", "") == ""
            tier0_path = joinpath(OUTDIR, "portfolio_tier0_results.jls")
            if isfile(tier0_path)
                t0 = deserialize(tier0_path)
                ranked = t0["candidate_verdicts"]
                survivors = [String(v["candidate"]) for v in ranked if String(v["label"]) in ("PROMOTE_TO_PMO_PLAN", "SYNTHETIC_ONLY_INTERESTING")]
                if !isempty(survivors)
                    requested_candidates = survivors[1:min(4, length(survivors))]
                end
            end
        end
    else
        error("Unknown PORT_MODE=$(mode). Use compile, tier0, or tier1.")
    end
    logmsg("Portfolio mode=$(mode) candidates=$(requested_candidates) budgets=$(budgets) seeds=$(seeds)")
    rows = run_rows(requested_candidates, budgets, seeds)
    summaries = summarize_algorithm(rows)
    verdicts = candidate_gate(rows)
    bundle = Dict{String,Any}(
        "created_at" => string(now()),
        "authoritative_date_note" => AUTHORITATIVE_DATE_NOTE,
        "schema_version" => PORT_SCHEMA_VERSION,
        "mode" => mode,
        "candidates" => requested_candidates,
        "budgets" => budgets,
        "seeds" => seeds,
        "rows" => rows,
        "algorithm_summaries" => summaries,
        "candidate_verdicts" => verdicts,
        "limitations" => [
            "Stage 0 tests core primitives only, not learned GFlowNet fusion.",
            "Synthetic screening cannot prove PMO/SOTA performance.",
            "PMO micro requires a separate v2-audited plan for any winner.",
        ],
    )
    out = joinpath(OUTDIR, mode == "tier0" ? "portfolio_tier0_results.jls" : "portfolio_tier1_results.jls")
    latest = joinpath(OUTDIR, "portfolio_latest_results.jls")
    save_bundle(out, bundle)
    save_bundle(latest, bundle)
    if mode == "tier1"
        save_bundle(joinpath(OUTDIR, "portfolio_ranked_candidates.jls"), bundle)
    end
    print_summary(verdicts, summaries, mode)
    logmsg("Saved results: $(abspath(out))")
    logmsg("Saved latest: $(abspath(latest))")
    return bundle
end

mode = String(strip(get(ENV, "PORT_MODE", "compile")))
run_mode(mode)
