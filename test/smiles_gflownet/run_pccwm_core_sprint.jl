#!/usr/bin/env julia

# Proof-Carrying Counterfactual World Model Core Sprint v2
#
# Synthetic/interventional POC only. This is NOT a GFlowNet implementation and
# NOT a PMO/SOTA claim.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Random
using Serialization
using Statistics
using Dates
using Printf
using LinearAlgebra

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUTDIR = joinpath(ROOT, "checkpoints", "pccwm_core")
const PCCWM_SCHEMA_VERSION = "pccwm_core_v2_reaudited_synthetic"
const AUTHORITATIVE_DATE_NOTE = "User supplied authoritative date/time: Saturday, 2026-06-20 11:43 EDT"
mkpath(OUTDIR)

const ARMS = [
    "pccwm_full",
    "no_certificate_cwm",
    "correlational_surrogate",
    "certificate_only_filter",
    "random_intervention",
    "ga_candidate_search",
    "rank_weighted_ga",
    "ucb_surrogate",
    "thompson_surrogate",
]
const STRONG_BASELINES = Set(["ga_candidate_search", "rank_weighted_ga", "ucb_surrogate", "thompson_surrogate"])
const MODEL_ARMS = Set(["pccwm_full", "no_certificate_cwm", "correlational_surrogate", "certificate_only_filter", "ucb_surrogate", "thompson_surrogate"])
const ALPHA = 0.10

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

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

function increment!(d::Dict{String,Int}, k::String, n::Int=1)
    d[k] = get(d, k, 0) + n
    return nothing
end

sigmoid(x) = 1.0 / (1.0 + exp(-x))
clamp01(x) = clamp(x, 0.0, 1.0)
bitvec_key(v::Vector{Int}) = join(v, "")

function quantile_nearest(vals::Vector{Float64}, q::Float64; default::Float64=0.0)
    isempty(vals) && return default
    s = sort(vals)
    idx = clamp(ceil(Int, q * length(s)), 1, length(s))
    return s[idx]
end

function topk_mean(vals::Vector{Float64}; k::Int=10)
    isempty(vals) && return 0.0, 0
    s = sort(vals, rev=true)
    n = min(k, length(s))
    return mean(s[1:n]), n
end

# -----------------------------------------------------------------------------
# World definition
# -----------------------------------------------------------------------------

struct WorldState
    scaffold::Int
    motif::Vector{Int}
    nuisance::Vector{Int}
    confounder::Vector{Int}
end

struct Intervention
    field::Symbol
    idx::Int
    val::Int
end

struct CausalWorld
    task::String
    n_scaffold::Int
    n_motif::Int
    n_nuisance::Int
    n_confounder::Int
    motif_weights::Vector{Float64}
    scaffold_effects::Vector{Float64}
    task_noise::Float64
    identifiable::Bool
    train_scaffolds::Vector{Int}
    test_scaffolds::Vector{Int}
end

function make_world(task::String)
    motif_weights = [0.16, 0.14, -0.12, 0.11, -0.10, 0.08]
    scaffold_effects = [0.00, 0.03, -0.02, 0.05]
    if task == "confounded_motif_ood"
        return CausalWorld(task, 4, 6, 10, 6, motif_weights, scaffold_effects, 0.0, true, [1,2], [3,4])
    elseif task == "scaffold_shift_transfer"
        return CausalWorld(task, 4, 6, 10, 6, motif_weights, scaffold_effects, 0.0, true, [1,2,3], [4])
    elseif task == "nonidentifiable_antibias"
        return CausalWorld(task, 4, 6, 10, 6, motif_weights, scaffold_effects, 0.0, false, [1,2], [3,4])
    else
        error("Unknown PCCWM task: $(task)")
    end
end

function reward(w::CausalWorld, x::WorldState)
    m = x.motif
    base = 0.32 + w.scaffold_effects[x.scaffold]
    motif_term = sum(w.motif_weights[i] * m[i] for i in 1:w.n_motif)
    interaction = 0.11 * (m[1] == 1 && m[2] == 1 ? 1.0 : 0.0) + 0.08 * (m[4] == 1 && m[5] == 0 ? 1.0 : 0.0)
    nuisance = 0.006 * sum(x.nuisance) / max(1, w.n_nuisance)
    # Confounders have no true causal effect. They are spurious in train.
    return clamp01(base + motif_term + interaction + nuisance)
end

function state_key(x::WorldState)
    return string(x.scaffold, "|", bitvec_key(x.motif), "|", bitvec_key(x.nuisance), "|", bitvec_key(x.confounder))
end

function copy_state(x::WorldState)
    WorldState(x.scaffold, copy(x.motif), copy(x.nuisance), copy(x.confounder))
end

function apply_intervention(x::WorldState, u::Intervention)
    y = copy_state(x)
    if u.field == :motif
        y.motif[u.idx] = u.val
    elseif u.field == :nuisance
        y.nuisance[u.idx] = u.val
    elseif u.field == :confounder
        y.confounder[u.idx] = u.val
    else
        error("Unknown intervention field $(u.field)")
    end
    return y
end

function intervention_key(u::Intervention)
    return string(u.field, ":", u.idx, ":", u.val)
end

function all_interventions(w::CausalWorld, x::WorldState; motif_only::Bool=true)
    us = Intervention[]
    for i in 1:w.n_motif
        for val in 0:1
            val != x.motif[i] && push!(us, Intervention(:motif, i, val))
        end
    end
    if !motif_only
        for i in 1:w.n_confounder
            for val in 0:1
                val != x.confounder[i] && push!(us, Intervention(:confounder, i, val))
            end
        end
    end
    return us
end

function sample_state(w::CausalWorld, rng::AbstractRNG; split::String="train")
    scaffolds = split == "train" ? w.train_scaffolds : w.test_scaffolds
    scaffold = scaffolds[rand(rng, 1:length(scaffolds))]
    motif = rand(rng, 0:1, w.n_motif)
    nuisance = rand(rng, 0:1, w.n_nuisance)
    base = WorldState(scaffold, motif, nuisance, zeros(Int, w.n_confounder))
    r = reward(w, base)
    conf = zeros(Int, w.n_confounder)
    if split == "train"
        if w.task == "nonidentifiable_antibias"
            # Perfect collinearity: confounders copy causal motif bits, preventing identification.
            for i in 1:w.n_confounder
                conf[i] = motif[((i - 1) % w.n_motif) + 1]
            end
        else
            # Spurious train-only reward correlation.
            p = sigmoid(7.0 * (r - 0.48))
            for i in 1:w.n_confounder
                conf[i] = rand(rng) < p ? 1 : 0
            end
        end
    else
        if w.task == "confounded_motif_ood"
            # Broken confounding: confounders random / weakly anti-correlated.
            p = sigmoid(-4.0 * (r - 0.48))
            for i in 1:w.n_confounder
                conf[i] = rand(rng) < p ? 1 : 0
            end
        elseif w.task == "nonidentifiable_antibias"
            # Test breaks the train collinearity.
            conf = rand(rng, 0:1, w.n_confounder)
        else
            conf = rand(rng, 0:1, w.n_confounder)
        end
    end
    return WorldState(scaffold, motif, nuisance, conf)
end

# -----------------------------------------------------------------------------
# Training data and models
# -----------------------------------------------------------------------------

struct PairDatum
    x::WorldState
    u::Intervention
    y::WorldState
    delta::Float64
end

struct TrainData
    states::Vector{WorldState}
    rewards::Vector{Float64}
    pairs::Vector{PairDatum}
    cal_pairs::Vector{PairDatum}
end

function make_train_data(w::CausalWorld, seed::Int; n_base::Int=420)
    rng = MersenneTwister(seed + bounded_hash_int((w.task, "train")))
    states = WorldState[]
    rewards = Float64[]
    pairs = PairDatum[]
    cal_pairs = PairDatum[]
    for i in 1:n_base
        x = sample_state(w, rng; split="train")
        push!(states, x); push!(rewards, reward(w, x))
        if w.identifiable
            # Training interventions provide mechanism-level pair evidence.
            for u in all_interventions(w, x; motif_only=true)
                y = apply_intervention(x, u)
                d = reward(w, y) - reward(w, x)
                pd = PairDatum(x, u, y, d)
                if rand(rng) < 0.75
                    push!(pairs, pd)
                else
                    push!(cal_pairs, pd)
                end
            end
        else
            # Anti-bias: no true motif intervention coverage, only spurious confounder interventions.
            for u in all_interventions(w, x; motif_only=false)
                if u.field == :confounder && rand(rng) < 0.20
                    y = apply_intervention(x, u)
                    d = reward(w, y) - reward(w, x)
                    push!(cal_pairs, PairDatum(x, u, y, d))
                end
            end
        end
    end
    return TrainData(states, rewards, pairs, cal_pairs)
end

struct CWMModel
    effect_mean::Dict{String,Float64}
    support::Dict{String,Int}
    q_abs_resid::Float64
    alpha::Float64
end

function fit_cwm(data::TrainData; alpha::Float64=ALPHA)
    sums = Dict{String,Float64}()
    counts = Dict{String,Int}()
    for pd in data.pairs
        k = intervention_key(pd.u)
        sums[k] = get(sums, k, 0.0) + pd.delta
        counts[k] = get(counts, k, 0) + 1
    end
    means = Dict{String,Float64}()
    for (k, s) in sums
        means[k] = s / counts[k]
    end
    residuals = Float64[]
    for pd in vcat(data.pairs, data.cal_pairs)
        k = intervention_key(pd.u)
        if haskey(means, k)
            push!(residuals, abs(means[k] - pd.delta))
        end
    end
    q = quantile_nearest(residuals, 1.0 - alpha; default=1.0)
    return CWMModel(means, counts, q, alpha)
end

function predict_delta(m::CWMModel, u::Intervention)
    k = intervention_key(u)
    return get(m.effect_mean, k, 0.0), get(m.support, k, 0)
end

function certificate(m::CWMModel, x::WorldState, u::Intervention; min_support::Int=20, threshold::Float64=0.0)
    μ, supp = predict_delta(m, u)
    L = μ - m.q_abs_resid
    ok = supp >= min_support && L > threshold
    return Dict{String,Any}(
        "ok" => ok,
        "lower_bound" => L,
        "pred_delta" => μ,
        "support" => supp,
        "calibration_quantile" => m.q_abs_resid,
        "alpha" => m.alpha,
        "trace" => intervention_key(u),
    )
end

function features(w::CausalWorld, x::WorldState)
    # Intercept + scaffold one-hot + motif + nuisance + confounder.
    f = Float64[1.0]
    for s in 1:w.n_scaffold
        push!(f, x.scaffold == s ? 1.0 : 0.0)
    end
    append!(f, Float64.(x.motif))
    append!(f, Float64.(x.nuisance))
    append!(f, Float64.(x.confounder))
    return f
end

struct LinearSurrogate
    β::Vector{Float64}
    residual_q::Float64
end

function fit_linear(w::CausalWorld, states::Vector{WorldState}, rewards::Vector{Float64}; alpha::Float64=ALPHA, λ::Float64=1e-3)
    n = length(states)
    d = length(features(w, states[1]))
    X = zeros(Float64, n, d)
    for i in 1:n
        X[i, :] .= features(w, states[i])
    end
    y = rewards
    β = (X' * X + λ * I(d)) \ (X' * y)
    resid = abs.(X * β .- y)
    q = quantile_nearest(collect(resid), 1.0 - alpha; default=0.5)
    return LinearSurrogate(collect(β), q)
end

function predict(m::LinearSurrogate, w::CausalWorld, x::WorldState)
    return dot(m.β, features(w, x))
end

function fit_ensemble(w::CausalWorld, states::Vector{WorldState}, rewards::Vector{Float64}, rng::AbstractRNG; n_models::Int=7)
    models = LinearSurrogate[]
    n = length(states)
    for _ in 1:n_models
        idx = [rand(rng, 1:n) for _ in 1:n]
        push!(models, fit_linear(w, states[idx], rewards[idx]; alpha=ALPHA, λ=1e-2))
    end
    return models
end

function ensemble_predict(models::Vector{LinearSurrogate}, w::CausalWorld, x::WorldState)
    vals = [predict(m, w, x) for m in models]
    return mean(vals), length(vals) <= 1 ? 0.0 : std(vals)
end

# -----------------------------------------------------------------------------
# Evaluation state and optimization loop
# -----------------------------------------------------------------------------

mutable struct OptState
    w::CausalWorld
    arm::String
    budget::Int
    calls::Int
    seen::Dict{String,Float64}
    pool::Vector{WorldState}
    curve::Vector{Dict{String,Any}}
    ledger::Dict{String,Int}
    cert_records::Vector{Dict{String,Any}}
    proposal_records::Vector{Dict{String,Any}}
    online_states::Vector{WorldState}
    online_rewards::Vector{Float64}
end

function new_opt_state(w::CausalWorld, arm::String, budget::Int)
    return OptState(w, arm, budget, 0, Dict{String,Float64}(), WorldState[], Dict{String,Any}[], Dict{String,Int}(), Dict{String,Any}[], Dict{String,Any}[], WorldState[], Float64[])
end

function record_curve!(st::OptState, event::String)
    vals = collect(values(st.seen))
    top10, denom = topk_mean(Float64.(vals); k=10)
    push!(st.curve, Dict{String,Any}(
        "calls" => st.calls,
        "event" => event,
        "top1" => isempty(vals) ? 0.0 : maximum(vals),
        "top10_mean" => top10,
        "top10_denominator" => denom,
        "unique" => length(vals),
    ))
end

function evaluate!(st::OptState, x::WorldState; event::String="eval")
    st.calls >= st.budget && return false
    k = state_key(x)
    if haskey(st.seen, k)
        increment!(st.ledger, "duplicates")
        return false
    end
    r = reward(st.w, x)
    st.calls += 1
    st.seen[k] = r
    push!(st.pool, x)
    push!(st.online_states, x)
    push!(st.online_rewards, r)
    record_curve!(st, event)
    return true
end

function normalized_auc(curve::Vector{Dict{String,Any}}, budget::Int; metric::String="top10_mean")
    isempty(curve) && return 0.0
    ordered = sort(curve, by = r -> Int(r["calls"]))
    if Int(ordered[1]["calls"]) > 0
        ordered = vcat([Dict{String,Any}("calls"=>0, metric=>0.0)], ordered)
    end
    area = 0.0
    px = 0.0
    py = 0.0
    for row in ordered
        x = min(Float64(Int(row["calls"])), Float64(budget))
        y = Float64(get(row, metric, 0.0))
        x >= px || continue
        area += (x - px) * (py + y) / 2.0
        px = x
        py = y
        px >= budget && break
    end
    if px < budget
        area += (Float64(budget) - px) * py
    end
    return area / max(1.0, Float64(budget))
end

function choose_context(st::OptState, rng::AbstractRNG)
    isempty(st.pool) && error("empty pool")
    if rand(rng) < 0.75
        # Mostly exploit high observed reward contexts.
        top = sort(st.pool, by = x -> -get(st.seen, state_key(x), reward(st.w, x)))
        return top[rand(rng, 1:min(8, length(top)))]
    else
        return st.pool[rand(rng, 1:length(st.pool))]
    end
end

function random_intervention_for(w::CausalWorld, x::WorldState, rng::AbstractRNG)
    us = all_interventions(w, x; motif_only=true)
    return us[rand(rng, 1:length(us))]
end

function mutate_candidate(w::CausalWorld, x::WorldState, rng::AbstractRNG; n_changes::Int=1)
    y = copy_state(x)
    for _ in 1:n_changes
        field = rand(rng) < 0.65 ? :motif : (rand(rng) < 0.5 ? :nuisance : :confounder)
        if field == :motif
            i = rand(rng, 1:w.n_motif); y.motif[i] = 1 - y.motif[i]
        elseif field == :nuisance
            i = rand(rng, 1:w.n_nuisance); y.nuisance[i] = 1 - y.nuisance[i]
        else
            i = rand(rng, 1:w.n_confounder); y.confounder[i] = 1 - y.confounder[i]
        end
    end
    return y
end

function crossover_state(a::WorldState, b::WorldState, rng::AbstractRNG)
    y = copy_state(a)
    for i in eachindex(y.motif)
        rand(rng) < 0.5 && (y.motif[i] = b.motif[i])
    end
    for i in eachindex(y.nuisance)
        rand(rng) < 0.5 && (y.nuisance[i] = b.nuisance[i])
    end
    for i in eachindex(y.confounder)
        rand(rng) < 0.5 && (y.confounder[i] = b.confounder[i])
    end
    return y
end

function generate_candidate(st::OptState, rng::AbstractRNG, data::TrainData, cwm::CWMModel, lin::LinearSurrogate, ensemble::Vector{LinearSurrogate})
    w = st.w
    arm = st.arm
    if arm in ("pccwm_full", "no_certificate_cwm", "correlational_surrogate", "certificate_only_filter", "ucb_surrogate", "thompson_surrogate", "random_intervention")
        contexts = isempty(st.pool) ? [sample_state(w, rng; split="test") for _ in 1:4] : st.pool[shuffle(rng, 1:length(st.pool))[1:min(length(st.pool), 12)]]
        best_score = -Inf
        best_x = contexts[1]
        best_u = random_intervention_for(w, best_x, rng)
        best_cert = Dict{String,Any}("ok"=>false)
        certified_options = 0
        total_options = 0
        for x in contexts
            for u in all_interventions(w, x; motif_only=true)
                total_options += 1
                y = apply_intervention(x, u)
                if haskey(st.seen, state_key(y))
                    continue
                end
                score = 0.0
                cert = Dict{String,Any}("ok"=>false)
                if arm == "pccwm_full"
                    cert = certificate(cwm, x, u)
                    cert["ok"] == true && (certified_options += 1)
                    score = cert["ok"] == true ? Float64(cert["lower_bound"]) : -1.0 + Float64(cert["pred_delta"])
                elseif arm == "no_certificate_cwm"
                    μ, _ = predict_delta(cwm, u)
                    score = μ
                elseif arm == "correlational_surrogate"
                    score = predict(lin, w, y) - predict(lin, w, x)
                elseif arm == "certificate_only_filter"
                    pred_delta = predict(lin, w, y) - predict(lin, w, x)
                    L = pred_delta - lin.residual_q
                    cert = Dict{String,Any}("ok"=>L > 0.0, "lower_bound"=>L, "pred_delta"=>pred_delta, "support"=>length(data.states), "calibration_quantile"=>lin.residual_q)
                    cert["ok"] == true && (certified_options += 1)
                    score = cert["ok"] == true ? L : -1.0 + pred_delta
                elseif arm == "ucb_surrogate"
                    μ, σ = ensemble_predict(ensemble, w, y)
                    score = μ + 1.5 * σ
                elseif arm == "thompson_surrogate"
                    μ, σ = ensemble_predict(ensemble, w, y)
                    score = μ + σ * randn(rng)
                else
                    score = rand(rng)
                end
                if arm == "random_intervention"
                    x0 = choose_context(st, rng); u0 = random_intervention_for(w, x0, rng); return apply_intervention(x0, u0), x0, u0, Dict{String,Any}("ok"=>false), 0, 1
                end
                if score > best_score
                    best_score = score
                    best_x = x
                    best_u = u
                    best_cert = cert
                end
            end
        end
        if arm == "pccwm_full" && certified_options == 0
            increment!(st.ledger, "abstentions")
        end
        return apply_intervention(best_x, best_u), best_x, best_u, best_cert, certified_options, total_options
    elseif arm in ("ga_candidate_search", "rank_weighted_ga")
        # Genetic candidate search with optional rank-weighted parent selection.
        pool = st.pool
        rewards = [get(st.seen, state_key(x), reward(w, x)) for x in pool]
        function pick_parent()
            if arm == "rank_weighted_ga"
                order = sortperm(rewards, rev=true)
                weights = [1.0 / i for i in 1:length(order)]
                weights ./= sum(weights)
                idx = order[searchsortedfirst(cumsum(weights), rand(rng))]
                return pool[idx]
            else
                ids = rand(rng, 1:length(pool), min(3, length(pool)))
                return pool[ids[argmax(rewards[ids])]]
            end
        end
        p1 = pick_parent(); p2 = pick_parent()
        child = rand(rng) < 0.45 ? crossover_state(p1, p2, rng) : copy_state(p1)
        child = mutate_candidate(w, child, rng; n_changes=rand(rng) < 0.2 ? 2 : 1)
        return child, p1, Intervention(:motif, 1, p1.motif[1]), Dict{String,Any}("ok"=>false), 0, 1
    else
        error("Unknown arm $(arm)")
    end
end

function initialize_pool!(st::OptState, rng::AbstractRNG; n_init::Int=16)
    attempts = 0
    while st.calls < min(st.budget, n_init) && attempts < n_init * 20
        attempts += 1
        x = sample_state(st.w, rng; split="test")
        evaluate!(st, x; event="init")
    end
end

function run_arm(task::String, arm::String, seed::Int; budget::Int=200, n_train::Int=420)
    rng = MersenneTwister(seed + bounded_hash_int((task, arm, PCCWM_SCHEMA_VERSION)))
    w = make_world(task)
    data = make_train_data(w, seed; n_base=n_train)
    cwm = fit_cwm(data; alpha=ALPHA)
    lin = fit_linear(w, data.states, data.rewards; alpha=ALPHA)
    ensemble = fit_ensemble(w, data.states, data.rewards, rng; n_models=7)
    st = new_opt_state(w, arm, budget)
    initialize_pool!(st, rng; n_init=min(16, budget ÷ 4))
    attempts = 0
    while st.calls < budget && attempts < budget * 100
        attempts += 1
        child, base, u, cert, certified_options, total_options = generate_candidate(st, rng, data, cwm, lin, ensemble)
        increment!(st.ledger, "proposal_attempts")
        increment!(st.ledger, "certified_options", certified_options)
        increment!(st.ledger, "total_options", total_options)
        true_delta = reward(w, child) - reward(w, base)
        pred_delta = get(cert, "pred_delta", NaN)
        cert_ok = get(cert, "ok", false) == true
        push!(st.proposal_records, Dict{String,Any}(
            "true_delta" => true_delta,
            "pred_delta" => pred_delta,
            "cert_ok" => cert_ok,
            "intervention" => intervention_key(u),
        ))
        if cert_ok
            push!(st.cert_records, Dict{String,Any}(
                "true_delta" => true_delta,
                "pred_delta" => pred_delta,
                "lower_bound" => cert["lower_bound"],
                "support" => cert["support"],
                "ok" => true,
            ))
            true_delta > 0 ? increment!(st.ledger, "certified_hits") : increment!(st.ledger, "false_certificates")
        end
        evaluate!(st, child; event=arm)
    end
    attempts >= budget * 100 && increment!(st.ledger, "attempt_cap_hits")
    # Counterfactual evaluation set.
    cf = counterfactual_metrics(w, data, cwm, lin, seed)
    vals = collect(values(st.seen))
    top10, denom = topk_mean(Float64.(vals); k=10)
    auc = normalized_auc(st.curve, budget)
    cert_count = length(st.cert_records)
    false_count = get(st.ledger, "false_certificates", 0)
    hit_count = get(st.ledger, "certified_hits", 0)
    proposal_count = length(st.proposal_records)
    improvement_count = count(r -> Float64(r["true_delta"]) > 0.0, st.proposal_records)
    return Dict{String,Any}(
        "task" => task,
        "arm" => arm,
        "seed" => seed,
        "budget" => budget,
        "calls" => st.calls,
        "auc_top10" => auc,
        "final_top1" => isempty(vals) ? 0.0 : maximum(vals),
        "final_top10_mean" => top10,
        "top10_denominator" => denom,
        "unique_evaluated" => length(st.seen),
        "certified_count" => cert_count,
        "certified_hit_rate" => cert_count == 0 ? NaN : hit_count / cert_count,
        "false_certificate_rate" => cert_count == 0 ? 0.0 : false_count / cert_count,
        "proposal_hit_rate" => proposal_count == 0 ? NaN : improvement_count / proposal_count,
        "coverage_rate" => proposal_count == 0 ? 0.0 : cert_count / proposal_count,
        "abstention_rate" => max(0, get(st.ledger, "abstentions", 0)) / max(1, proposal_count),
        "counterfactual_metrics" => cf,
        "ledger" => st.ledger,
        "curve" => st.curve,
        "schema_version" => PCCWM_SCHEMA_VERSION,
    )
end

function counterfactual_metrics(w::CausalWorld, data::TrainData, cwm::CWMModel, lin::LinearSurrogate, seed::Int; n_eval::Int=240)
    rng = MersenneTwister(seed + bounded_hash_int((w.task, "cf_eval")))
    cwm_err = Float64[]
    corr_err = Float64[]
    cert_false = 0
    cert_n = 0
    cert_hits = 0
    identifiable_cases = 0
    for _ in 1:n_eval
        x = sample_state(w, rng; split="test")
        for u in all_interventions(w, x; motif_only=true)
            y = apply_intervention(x, u)
            true_delta = reward(w, y) - reward(w, x)
            μ, supp = predict_delta(cwm, u)
            push!(cwm_err, abs(μ - true_delta))
            pred_corr = predict(lin, w, y) - predict(lin, w, x)
            push!(corr_err, abs(pred_corr - true_delta))
            cert = certificate(cwm, x, u)
            if cert["ok"] == true
                cert_n += 1
                true_delta > 0 ? (cert_hits += 1) : (cert_false += 1)
            end
            supp > 0 && (identifiable_cases += 1)
        end
    end
    return Dict{String,Any}(
        "cwm_mae" => mean(cwm_err),
        "correlational_mae" => mean(corr_err),
        "mae_ratio_cwm_over_corr" => mean(cwm_err) / max(1e-8, mean(corr_err)),
        "cf_certified_count" => cert_n,
        "cf_certified_hit_rate" => cert_n == 0 ? NaN : cert_hits / cert_n,
        "cf_false_certificate_rate" => cert_n == 0 ? 0.0 : cert_false / cert_n,
        "cf_coverage" => cert_n / max(1, n_eval * w.n_motif),
        "identifiable_case_count" => identifiable_cases,
    )
end

# -----------------------------------------------------------------------------
# Aggregation and gates
# -----------------------------------------------------------------------------

function mean_or_nan(xs)
    vals = [Float64(x) for x in xs if !isnan(Float64(x))]
    isempty(vals) && return NaN
    return mean(vals)
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
        push!(out, Dict{String,Any}(
            "task" => task,
            "arm" => arm,
            "n" => length(vals),
            "auc_mean" => mean_or_nan([v["auc_top10"] for v in vals]),
            "top10_mean" => mean_or_nan([v["final_top10_mean"] for v in vals]),
            "certified_hit_rate_mean" => mean_or_nan([v["certified_hit_rate"] for v in vals]),
            "false_certificate_rate_mean" => mean_or_nan([v["false_certificate_rate"] for v in vals]),
            "coverage_rate_mean" => mean_or_nan([v["coverage_rate"] for v in vals]),
            "proposal_hit_rate_mean" => mean_or_nan([v["proposal_hit_rate"] for v in vals]),
            "cwm_mae_mean" => mean_or_nan([v["counterfactual_metrics"]["cwm_mae"] for v in vals]),
            "corr_mae_mean" => mean_or_nan([v["counterfactual_metrics"]["correlational_mae"] for v in vals]),
        ))
    end
    sort!(out, by = r -> (String(r["task"]), String(r["arm"])))
    return out
end

function overall_by_arm(agg::Vector{Dict{String,Any}})
    groups = Dict{String,Vector{Dict{String,Any}}}()
    for r in agg
        arm = String(r["arm"])
        groups[arm] = get(groups, arm, Dict{String,Any}[])
        push!(groups[arm], r)
    end
    out = Dict{String,Any}[]
    for (arm, vals) in groups
        push!(out, Dict{String,Any}(
            "arm" => arm,
            "mean_auc" => mean_or_nan([v["auc_mean"] for v in vals]),
            "mean_top10" => mean_or_nan([v["top10_mean"] for v in vals]),
            "mean_certified_hit_rate" => mean_or_nan([v["certified_hit_rate_mean"] for v in vals]),
            "mean_false_certificate_rate" => mean_or_nan([v["false_certificate_rate_mean"] for v in vals]),
            "mean_coverage_rate" => mean_or_nan([v["coverage_rate_mean"] for v in vals]),
            "mean_proposal_hit_rate" => mean_or_nan([v["proposal_hit_rate_mean"] for v in vals]),
        ))
    end
    sort!(out, by = r -> -Float64(r["mean_auc"]), rev=false)
    return out
end

function paired_best_baseline(rows::Vector{Dict{String,Any}})
    idx = Dict{Tuple{String,Int,String},Dict{String,Any}}()
    for r in rows
        idx[(String(r["task"]), Int(r["seed"]), String(r["arm"]))] = r
    end
    pairs = unique([(String(r["task"]), Int(r["seed"])) for r in rows])
    out = Dict{String,Any}[]
    for (task, seed) in pairs
        pkey = (task, seed, "pccwm_full")
        haskey(idx, pkey) || continue
        p = idx[pkey]
        best_arm = ""
        best_auc = -Inf
        best = nothing
        for arm in STRONG_BASELINES
            key = (task, seed, arm)
            if haskey(idx, key)
                auc = Float64(idx[key]["auc_top10"])
                if auc > best_auc
                    best_auc = auc; best_arm = arm; best = idx[key]
                end
            end
        end
        best === nothing && continue
        push!(out, Dict{String,Any}(
            "task"=>task, "seed"=>seed, "best_baseline_arm"=>best_arm,
            "pccwm_auc"=>Float64(p["auc_top10"]), "best_baseline_auc"=>best_auc,
            "delta_auc"=>Float64(p["auc_top10"]) - best_auc,
            "relative_delta"=>best_auc == 0 ? NaN : Float64(p["auc_top10"]) / best_auc - 1.0,
        ))
    end
    return out
end

function gate_decision(rows::Vector{Dict{String,Any}}, overall::Vector{Dict{String,Any}})
    byarm = Dict(String(r["arm"]) => r for r in overall)
    p = byarm["pccwm_full"]
    corr_maes = [r["counterfactual_metrics"]["correlational_mae"] for r in rows if String(r["arm"]) == "pccwm_full" && String(r["task"]) != "nonidentifiable_antibias"]
    cwm_maes = [r["counterfactual_metrics"]["cwm_mae"] for r in rows if String(r["arm"]) == "pccwm_full" && String(r["task"]) != "nonidentifiable_antibias"]
    mechanism_ratio = mean_or_nan(cwm_maes) / max(1e-8, mean_or_nan(corr_maes))
    mechanism_gate = mechanism_ratio < 0.80
    false_rate = Float64(p["mean_false_certificate_rate"])
    cert_hit = Float64(p["mean_certified_hit_rate"])
    no_cert_hit = Float64(get(byarm, "no_certificate_cwm", Dict("mean_proposal_hit_rate"=>NaN))["mean_proposal_hit_rate"])
    cert_gate = false_rate <= ALPHA + 0.05 && isfinite(cert_hit) && isfinite(no_cert_hit) && cert_hit > 1.10 * no_cert_hit
    best_baseline_auc = maximum([Float64(byarm[a]["mean_auc"]) for a in STRONG_BASELINES if haskey(byarm, a)])
    p_auc = Float64(p["mean_auc"])
    paired = paired_best_baseline(rows)
    paired_wins = count(x -> Float64(x["delta_auc"]) > 0, paired)
    opt_gate = p_auc > 1.05 * best_baseline_auc && paired_wins > length(paired) / 2
    abstention = Float64(p["mean_coverage_rate"])
    wins_by_abstain = abstention < 0.05 && opt_gate
    continue_gate = mechanism_gate && cert_gate && opt_gate && !wins_by_abstain
    reasons = String[]
    mechanism_gate || push!(reasons, "mechanism gate failed: CWM MAE not >=20% better than correlational surrogate on OOD.")
    cert_gate || push!(reasons, "certificate gate failed: false rate/hit-rate criteria not met.")
    opt_gate || push!(reasons, "optimization gate failed: PCCWM did not beat strong GA/rank/UCB/TS baselines by >=5% with paired majority.")
    wins_by_abstain && push!(reasons, "PCCWM appears to win by excessive abstention/low coverage.")
    continue_gate && push!(reasons, "PCCWM core passes all synthetic gates; separate PMO/GFN plan warranted.")
    return Dict{String,Any}(
        "verdict" => continue_gate ? "PCCWM_CORE_SIGNAL_PRESENT" : "PCCWM_STOP_OR_REDESIGN",
        "continue_gate" => continue_gate,
        "mechanism_gate" => mechanism_gate,
        "certificate_gate" => cert_gate,
        "optimization_gate" => opt_gate,
        "mechanism_mae_ratio" => mechanism_ratio,
        "pccwm_mean_auc" => p_auc,
        "best_strong_baseline_auc" => best_baseline_auc,
        "pccwm_vs_best_relative" => p_auc / best_baseline_auc - 1.0,
        "paired_wins_vs_best_baseline" => paired_wins,
        "paired_count" => length(paired),
        "pccwm_false_certificate_rate" => false_rate,
        "pccwm_certified_hit_rate" => cert_hit,
        "no_certificate_proposal_hit_rate" => no_cert_hit,
        "pccwm_coverage_rate" => abstention,
        "paired_rows" => paired,
        "reasons" => reasons,
    )
end

function save_bundle(path::String, bundle::Dict{String,Any})
    tmp = path * ".tmp"
    serialize(tmp, bundle)
    mv(tmp, path; force=true)
    return path
end

function print_summary(overall, gate, mode)
    println("\n", "="^112)
    println("PCCWM CORE SUMMARY — ", uppercase(mode))
    println("="^112)
    println(rpad("Arm", 30), rpad("Mean AUC", 14), rpad("Mean Top10", 14), rpad("CertHit", 12), rpad("FalseCert", 12), rpad("Coverage", 12))
    println("-"^112)
    for r in overall
        @printf("%-30s%-14.6f%-14.6f%-12.4f%-12.4f%-12.4f\n", r["arm"], r["mean_auc"], r["mean_top10"], r["mean_certified_hit_rate"], r["mean_false_certificate_rate"], r["mean_coverage_rate"])
    end
    println("\nGATE: ", gate["verdict"])
    for reason in gate["reasons"]
        println("- ", reason)
    end
end

function run_suite(mode::String)
    if mode == "compile"
        println("PCCWM compile OK: ", PCCWM_SCHEMA_VERSION)
        return nothing
    end
    tasks = parse_csv_strings("PCCWM_TASKS", mode == "smoke" ? ["confounded_motif_ood"] : ["confounded_motif_ood", "scaffold_shift_transfer", "nonidentifiable_antibias"])
    seeds = parse_csv_ints("PCCWM_SEEDS", mode == "smoke" ? [17] : [17, 23])
    budget = parse_env_int("PCCWM_BUDGET", mode == "smoke" ? 80 : 200)
    arms = parse_csv_strings("PCCWM_ARMS", ARMS)
    invalid = [a for a in arms if !(a in ARMS)]
    isempty(invalid) || error("Unknown PCCWM arms: $(invalid)")
    rows = Dict{String,Any}[]
    logmsg("PCCWM mode=$(mode) tasks=$(tasks) seeds=$(seeds) budget=$(budget) arms=$(arms)")
    for seed in seeds, task in tasks, arm in arms
        logmsg("RUN task=$(task) arm=$(arm) seed=$(seed)")
        start = time()
        try
            row = run_arm(task, arm, seed; budget=budget)
            row["elapsed_sec"] = time() - start
            push!(rows, row)
            logmsg("OK task=$(task) arm=$(arm) auc=$(round(row["auc_top10"], digits=5)) top10=$(round(row["final_top10_mean"], digits=5)) cert=$(row["certified_count"])")
        catch e
            bt = catch_backtrace()
            err = sprint(showerror, e, bt)
            logmsg("FAILED task=$(task) arm=$(arm) seed=$(seed)")
            println(err)
            push!(rows, Dict{String,Any}("task"=>task, "arm"=>arm, "seed"=>seed, "status"=>"failed", "error"=>err, "auc_top10"=>NaN, "final_top10_mean"=>NaN))
        end
    end
    ok_rows = [r for r in rows if !haskey(r, "status")]
    agg = aggregate_rows(ok_rows)
    overall = overall_by_arm(agg)
    gate = gate_decision(ok_rows, overall)
    bundle = Dict{String,Any}(
        "created_at" => string(now()),
        "authoritative_date_note" => AUTHORITATIVE_DATE_NOTE,
        "schema_version" => PCCWM_SCHEMA_VERSION,
        "mode" => mode,
        "tasks" => tasks,
        "seeds" => seeds,
        "budget" => budget,
        "arms" => arms,
        "rows" => rows,
        "aggregate_rows" => agg,
        "overall_by_arm" => overall,
        "gate" => gate,
        "limitations" => [
            "Synthetic/interventional POC only; no PMO or molecular claim.",
            "Stage 0 core only; no learned GFlowNet is used.",
            "Certificate is for synthetic counterfactual improvement, not chemistry validity.",
        ],
    )
    out = joinpath(OUTDIR, "pccwm_$(mode)_results.jls")
    latest = joinpath(OUTDIR, "pccwm_latest_results.jls")
    save_bundle(out, bundle)
    save_bundle(latest, bundle)
    print_summary(overall, gate, mode)
    logmsg("Saved results: $(abspath(out))")
    logmsg("Saved latest: $(abspath(latest))")
    return bundle
end

mode = String(strip(get(ENV, "PCCWM_MODE", "compile")))
run_suite(mode)
