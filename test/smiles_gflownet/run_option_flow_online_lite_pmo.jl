#!/usr/bin/env julia

# Option-Flow Online-Lite PMO POC.
#
# This runner tests whether the E3 strict same-snapshot object transfers to
# actual sequential search. It deliberately separates:
#   O1: warm-started online option-selector proof;
#   O2: total-budget-fair proof where selector training calls reduce online calls.
#
# Deployable arms never evaluate all schemas before choosing. The optional
# oracle_upper arm is diagnostic-only and excluded from headline gates.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GFlowNet
using Random
using Serialization
using Statistics
using Dates
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "oracle_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "training", "option_flow_dataset.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "training", "option_flow_model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "training", "option_flow_loss.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "training", "option_flow_training.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "training", "option_flow_real_catalog.jl"))

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUTDIR = joinpath(ROOT, "checkpoints", "option_flow_online_lite_pmo")
mkpath(OUTDIR)

const TARGET_SMILES = Dict(
    "albuterol_similarity" => "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "celecoxib_rediscovery" => "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "mestranol_similarity" => "C#C[C@]1(O)CC[C@H]2[C@@H]3CCc4cc(OC)ccc4[C@H]3CC[C@@]21C",
    "thiothixene_rediscovery" => "C=C(c1ccc(S(=O)(=O)N2CCN(C)CC2)cc1)c1cc2c(s1)Cc1ccccc1-2",
)

const DEFAULT_BOOTSTRAP_SEEDS = [
    "CCO", "CCN", "CCC", "CCCl", "CC(=O)O", "c1ccccc1", "c1ccncc1", "CC(C)O",
]

const TASK_BOOTSTRAP_SEEDS = Dict(
    "qed" => ["CCO", "CCN", "CC(=O)O", "c1ccccc1", "CC(C)O", "COc1ccccc1"],
    "drd2" => ["N1CCCCC1", "CN1CCNCC1", "c1ccncc1", "CCN(CC)CC", "c1ccccc1Cl"],
    "gsk3b" => ["c1ccncc1", "Nc1ncccc1", "c1ccccc1", "CCN", "CC(=O)N"],
    "jnk3" => ["c1ccncc1", "Nc1ncccc1", "c1ccc(F)cc1", "CCN", "CC(=O)N"],
    "albuterol_similarity" => ["CC(C)(C)N", "NCCO", "Oc1ccccc1", "Oc1ccc(O)cc1", "CC(C)O"],
    "celecoxib_rediscovery" => ["Cc1ccccc1", "NS(=O)(=O)c1ccccc1", "FC(F)(F)c1ccccc1", "c1ccn[nH]1", "c1ccccc1S(N)(=O)=O"],
)

struct StrictOptionSpec
    name::String
    operator_override::Union{Nothing,Symbol}
    horizon::Int
    max_candidates::Int
    min_reward_ratio::Float64
    allow_crossover::Bool
end

const DEPLOYABLE_ARMS = Set(["uniform_schema", "heuristic_mixed_h2", "prior_best_schema", "option_flow_sample", "option_flow_greedy"])
const SELECTOR_ARMS = Set(["prior_best_schema", "option_flow_sample", "option_flow_greedy"])

function logmsg(msg)
    println("[", Dates.format(now(), "HH:MM:SS"), "] ", msg)
    flush(stdout)
end

function parse_env_int(name::String, default::Int)
    value = strip(get(ENV, name, ""))
    isempty(value) && return default
    return parse(Int, value)
end

function parse_env_float(name::String, default::Float64)
    value = strip(get(ENV, name, ""))
    isempty(value) && return default
    return parse(Float64, value)
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

function option_specs()
    return StrictOptionSpec[
        StrictOptionSpec("mutate_h1", :mutate, 1, 8, 0.15, true),
        StrictOptionSpec("mutate_h2", :mutate, 2, 8, 0.15, true),
        StrictOptionSpec("crossover_h1", :crossover, 1, 8, 0.15, true),
        StrictOptionSpec("crossover_h2", :crossover, 2, 8, 0.15, true),
        StrictOptionSpec("mixed_h1", nothing, 1, 8, 0.15, true),
        StrictOptionSpec("mixed_h2", nothing, 2, 8, 0.15, true),
    ]
end

function clone_frontier_buffer(buffer::MolecularFrontierBuffer)
    clone = MolecularFrontierBuffer(buffer.max_size)
    for entry in buffer.entries
        add_to_frontier!(clone, entry.smiles;
            reward=entry.reward,
            source=entry.source,
            parent_smiles=entry.parent_smiles,
            operator=entry.operator,
            tb_delta_abs=entry.tb_delta_abs)
    end
    return clone
end

function replace_frontier!(dest::MolecularFrontierBuffer, src::MolecularFrontierBuffer)
    dest.entries = [MolecularFrontierEntry(e.smiles, e.scaffold, e.reward, e.source, e.parent_smiles, e.operator, e.novelty_score, e.tb_delta_abs, e.visits) for e in src.entries]
    dest.seen_smiles = copy(src.seen_smiles)
    dest.scaffold_counts = copy(src.scaffold_counts)
    dest.needs_refresh = true
    return nothing
end

function build_reward_functions(task_name::String, budget::Int)
    OracleBridge.init_oracles!([task_name]; cache_dir=joinpath(ROOT, "data", "tdc_cache"))
    cache = Dict{String,Float64}()
    calls = Ref(0)

    function reward_batch(smiles_list::Vector{String})
        uncached = String[]
        seen = Set{String}()
        for smiles in smiles_list
            canonical = canonicalize_smiles_identity(smiles)
            isempty(canonical) && continue
            if !haskey(cache, canonical) && !(canonical in seen) && calls[] < budget
                push!(uncached, smiles)
                push!(seen, canonical)
            end
        end
        if !isempty(uncached)
            remaining = max(0, budget - calls[])
            eval_list = uncached[1:min(end, remaining)]
            scores = OracleBridge.evaluate_batch(eval_list, task_name)
            for (s, score) in zip(eval_list, scores)
                cache[canonicalize_smiles_identity(s)] = Float64(score)
            end
            calls[] += length(eval_list)
        end
        return Float64[get(cache, canonicalize_smiles_identity(s), 0.0) for s in smiles_list]
    end

    function reward_one(smiles::String)
        canonical = canonicalize_smiles_identity(smiles)
        haskey(cache, canonical) && return cache[canonical]
        scores = reward_batch([smiles])
        return isempty(scores) ? 0.0 : scores[1]
    end

    return reward_one, reward_batch, calls, cache
end

function seed_frontier!(frontier::MolecularFrontierBuffer, task_name::String, reward_batch)
    seeds = unique(vcat(DEFAULT_BOOTSTRAP_SEEDS, get(TASK_BOOTSTRAP_SEEDS, task_name, String[])))
    scores = reward_batch(seeds)
    added = 0
    for (smiles, reward) in zip(seeds, scores)
        reward <= 0.0 && continue
        before = length(frontier)
        add_to_frontier!(frontier, smiles; reward=reward, source=:seed, parent_smiles=nothing, operator=:seed)
        added += length(frontier) > before ? 1 : 0
    end
    return added
end

function strict_state_features(frontier::MolecularFrontierBuffer, task_name::String, task_names::Vector{String}, budget_remaining::Int, snapshot_index::Int; budget_scale::Int=1000)
    summary = frontier_quality_summary(frontier; topk=10)
    task_onehot = zeros(Float32, length(task_names))
    idx = findfirst(==(task_name), task_names)
    idx === nothing || (task_onehot[idx] = 1.0f0)
    return Float32.(vcat(task_onehot, Float32[
        Float32(summary["size"]) / 512.0f0,
        Float32(summary["top1"]),
        Float32(summary["top10_mean"]),
        Float32(summary["n_scaffolds"]) / 128.0f0,
        Float32(get(summary, "graph_unique_count", summary["size"])) / 128.0f0,
        Float32(budget_remaining) / Float32(max(budget_scale, 1)),
        Float32(snapshot_index) / 32.0f0,
        Float32(summary["top1"] - summary["top10_mean"]),
    ]))
end

function strict_option_features(spec::StrictOptionSpec, option_index::Int, n_specs::Int)
    op = spec.operator_override
    return Float32[
        op == :mutate ? 1.0f0 : 0.0f0,
        op == :crossover ? 1.0f0 : 0.0f0,
        isnothing(op) ? 1.0f0 : 0.0f0,
        Float32(spec.horizon) / 4.0f0,
        Float32(spec.max_candidates) / 16.0f0,
        Float32(spec.min_reward_ratio),
        spec.allow_crossover ? 1.0f0 : 0.0f0,
        Float32(option_index) / Float32(max(n_specs, 1)),
    ]
end

function run_option_from_snapshot(base_frontier::MolecularFrontierBuffer,
                                  task_name::String,
                                  target_smiles::Union{Nothing,String},
                                  reward_one,
                                  reward_batch,
                                  vocab,
                                  spec::StrictOptionSpec;
                                  snapshot_index::Int,
                                  option_index::Int,
                                  base_seed::Int,
                                  budget_remaining::Int)
    Random.seed!(base_seed + 1000 * snapshot_index + option_index)
    local_frontier = clone_frontier_buffer(base_frontier)
    trajectory_buffer = EditTrajectoryBuffer(10000)
    diagnostics_buffer = HierarchicalEditDiagnosticsBuffer(10000)
    config = HierarchicalEditConfig(;
        horizon=spec.horizon,
        frontier_snapshot_size=min(64, max(length(local_frontier), 1)),
        allow_crossover=spec.allow_crossover,
        allow_fragment_ops=false,
        max_operator_candidates=spec.max_candidates,
        topk_tracking=10,
        max_step_attempts=3,
        operators=nothing,
        min_exploration_per_operator=2,
        multi_child_min_reward_ratio=spec.min_reward_ratio,
        operator_prior_strength=4.0,
    )
    episode = run_hierarchical_edit_episode!(local_frontier,
        trajectory_buffer,
        reward_one,
        vocab;
        reward_fn_batch=reward_batch,
        diagnostics_buffer=diagnostics_buffer,
        config=config,
        target_smiles=target_smiles,
        budget_remaining=budget_remaining,
        created_at_step=snapshot_index,
        task_name=task_name,
        episode_id="online-$(task_name)-snap$(snapshot_index)-$(spec.name)-$(option_index)",
        operator_override=spec.operator_override,
    )
    utility_sum = sum(max(0.0, step.frontier_utility_delta) for step in episode.steps; init=0.0)
    raw_delta_sum = sum(step.frontier_utility_delta for step in episode.steps; init=0.0)
    return Dict{String,Any}(
        "spec" => spec,
        "episode" => episode,
        "frontier" => local_frontier,
        "utility" => utility_sum,
        "raw_delta_sum" => raw_delta_sum,
        "best_reward" => episode.best_reward,
        "commits_applied" => episode.commits_applied,
        "steps" => length(episode.steps),
        "improved_topk" => episode.improved_topk,
    )
end

function build_strict_catalog_for_snapshot(frontier::MolecularFrontierBuffer,
                                           task_name::String,
                                           task_names::Vector{String},
                                           target_smiles::Union{Nothing,String},
                                           reward_one,
                                           reward_batch,
                                           vocab,
                                           specs::Vector{StrictOptionSpec};
                                           snapshot_index::Int,
                                           base_seed::Int,
                                           budget_remaining::Int,
                                           budget_scale::Int=1000)
    snapshot = create_frontier_snapshot(frontier;
        max_entries=min(64, max(length(frontier), 1)),
        target_smiles=target_smiles,
        budget_remaining=budget_remaining,
        created_at_step=snapshot_index)
    state_features = strict_state_features(frontier, task_name, task_names, budget_remaining, snapshot_index; budget_scale=budget_scale)
    option_features = Vector{Vector{Float32}}()
    utilities = Float64[]
    option_ids = String[]
    metadata = Dict{String,Any}[]
    option_results = Dict{String,Any}[]

    for (i, spec) in enumerate(specs)
        result = run_option_from_snapshot(frontier, task_name, target_smiles, reward_one, reward_batch, vocab, spec;
            snapshot_index=snapshot_index,
            option_index=i,
            base_seed=base_seed,
            budget_remaining=budget_remaining)
        push!(option_results, result)
        push!(option_features, strict_option_features(spec, i, length(specs)))
        push!(utilities, Float64(result["utility"]))
        push!(option_ids, spec.name)
        push!(metadata, Dict{String,Any}(
            "evidence_level" => "online_pretrain_strict_same_snapshot",
            "task_name" => task_name,
            "snapshot_id" => string(snapshot.snapshot_id),
            "option_name" => spec.name,
            "operator_override" => isnothing(spec.operator_override) ? "mixed" : string(spec.operator_override),
            "horizon" => spec.horizon,
            "max_candidates" => spec.max_candidates,
            "min_reward_ratio" => spec.min_reward_ratio,
            "utility" => result["utility"],
            "raw_delta_sum" => result["raw_delta_sum"],
            "best_reward" => result["best_reward"],
            "commits_applied" => result["commits_applied"],
            "steps" => result["steps"],
            "improved_topk" => result["improved_topk"],
        ))
    end

    catalog = make_option_flow_catalog(task_name, snapshot.snapshot_id, state_features, option_features, utilities;
        option_ids=option_ids,
        metadata=metadata,
        epsilon=1.0e-6)
    return catalog, option_results, snapshot
end

function advance_frontier_with_best!(frontier::MolecularFrontierBuffer, option_results::Vector{Dict{String,Any}})
    isempty(option_results) && return "none"
    utilities = [Float64(r["utility"]) for r in option_results]
    best_idx = argmax(utilities)
    best_frontier = option_results[best_idx]["frontier"]::MolecularFrontierBuffer
    replace_frontier!(frontier, best_frontier)
    spec = option_results[best_idx]["spec"]::StrictOptionSpec
    return spec.name
end

function make_scoring_catalog(frontier::MolecularFrontierBuffer,
                              task_name::String,
                              task_names::Vector{String},
                              specs::Vector{StrictOptionSpec};
                              snapshot_index::Int,
                              budget_remaining::Int,
                              budget_scale::Int=1000)
    state_features = strict_state_features(frontier, task_name, task_names, budget_remaining, snapshot_index; budget_scale=budget_scale)
    option_features = [strict_option_features(spec, i, length(specs)) for (i, spec) in enumerate(specs)]
    utilities = ones(Float64, length(specs))
    option_ids = [spec.name for spec in specs]
    metas = [Dict{String,Any}("online_scoring_only" => true, "option_name" => spec.name) for spec in specs]
    return make_option_flow_catalog(task_name, UInt64(abs(hash((task_name, snapshot_index, budget_remaining))) % typemax(UInt64)),
        state_features, option_features, utilities; option_ids=option_ids, metadata=metas, epsilon=1.0e-6)
end

function train_catalog_bundle(tasks::Vector{String}, task_names::Vector{String}, specs::Vector{StrictOptionSpec};
                              budget_per_task::Int,
                              snapshots::Int,
                              base_seed::Int,
                              vocab,
                              budget_scale::Int=1000)
    catalogs = OptionFlowCatalog[]
    per_task = Dict{String,Any}()
    total_calls = 0
    for task in tasks
        target = get(TARGET_SMILES, task, nothing)
        reward_one, reward_batch, calls, cache = build_reward_functions(task, budget_per_task)
        frontier = MolecularFrontierBuffer(512)
        added = seed_frontier!(frontier, task, reward_batch)
        task_catalogs = OptionFlowCatalog[]
        advanced_by = String[]
        for snapshot_index in 1:snapshots
            length(frontier) >= 2 || break
            calls[] >= budget_per_task && break
            budget_remaining = max(0, budget_per_task - calls[])
            catalog, option_results, _ = build_strict_catalog_for_snapshot(frontier, task, task_names, target,
                reward_one, reward_batch, vocab, specs;
                snapshot_index=snapshot_index,
                base_seed=base_seed,
                budget_remaining=budget_remaining,
                budget_scale=budget_scale)
            push!(catalogs, catalog)
            push!(task_catalogs, catalog)
            chosen = advance_frontier_with_best!(frontier, option_results)
            push!(advanced_by, chosen)
        end
        task_calls = calls[]
        total_calls += task_calls
        per_task[task] = Dict{String,Any}(
            "calls" => task_calls,
            "cache_size" => length(cache),
            "seed_entries_added" => added,
            "catalogs" => length(task_catalogs),
            "advanced_by" => advanced_by,
            "catalog_stats" => option_flow_catalog_stats(task_catalogs),
        )
        logmsg("pretrain task=$(task) catalogs=$(length(task_catalogs)) calls=$(task_calls) added=$(added)")
    end
    return Dict{String,Any}(
        "catalogs" => catalogs,
        "per_task" => per_task,
        "total_calls" => total_calls,
        "stats" => option_flow_catalog_stats(catalogs),
    )
end

function train_selector(catalogs::Vector{OptionFlowCatalog}; seed::Int, epochs::Int, lr::Float64)
    isempty(catalogs) && error("cannot train selector with zero catalogs")
    result = train_option_flow_model(catalogs; config=OptionFlowTrainingConfig(
        n_epochs=epochs,
        learning_rate=lr,
        hidden_dim=64,
        second_hidden_dim=32,
        validation_fraction=length(catalogs) >= 4 ? 0.25 : 0.0,
        seed=seed,
        verbose=false,
    ))
    val_metrics = evaluate_option_flow_model(result["params"], result["val_catalogs"])
    train_metrics = evaluate_option_flow_model(result["params"], result["train_catalogs"])
    logmsg("selector train_catalogs=$(length(result["train_catalogs"])) val_catalogs=$(length(result["val_catalogs"])) train_ce_gain=$(round(train_metrics["mean_ce_vs_uniform"], digits=4)) val_ce_gain=$(round(val_metrics["mean_ce_vs_uniform"], digits=4))")
    return merge(result, Dict{String,Any}(
        "train_metrics" => train_metrics,
        "val_metrics" => val_metrics,
    ))
end

function prior_best_option_id(catalogs::Vector{OptionFlowCatalog})
    sums = Dict{String,Float64}()
    counts = Dict{String,Int}()
    for catalog in catalogs
        for candidate in catalog.candidates
            sums[candidate.option_id] = get(sums, candidate.option_id, 0.0) + candidate.utility
            counts[candidate.option_id] = get(counts, candidate.option_id, 0) + 1
        end
    end
    isempty(sums) && return "mixed_h2"
    means = Dict(k => sums[k] / max(counts[k], 1) for k in keys(sums))
    return collect(keys(means))[argmax([means[k] for k in keys(means)])]
end

function choose_spec(arm::String,
                     frontier::MolecularFrontierBuffer,
                     task_name::String,
                     task_names::Vector{String},
                     specs::Vector{StrictOptionSpec},
                     selector,
                     prior_best::String;
                     snapshot_index::Int,
                     budget_remaining::Int,
                     rng::AbstractRNG,
                     budget_scale::Int=1000)
    if arm == "uniform_schema"
        idx = rand(rng, 1:length(specs))
        return idx, fill(1.0 / length(specs), length(specs)), zeros(Float32, length(specs))
    elseif arm == "heuristic_mixed_h2"
        idx = something(findfirst(s -> s.name == "mixed_h2", specs), length(specs))
        return idx, [i == idx ? 1.0 : 0.0 for i in 1:length(specs)], zeros(Float32, length(specs))
    elseif arm == "prior_best_schema"
        idx = something(findfirst(s -> s.name == prior_best, specs), length(specs))
        return idx, [i == idx ? 1.0 : 0.0 for i in 1:length(specs)], zeros(Float32, length(specs))
    elseif arm == "option_flow_sample" || arm == "option_flow_greedy"
        isnothing(selector) && error("selector required for $(arm)")
        catalog = make_scoring_catalog(frontier, task_name, task_names, specs;
            snapshot_index=snapshot_index,
            budget_remaining=budget_remaining,
            budget_scale=budget_scale)
        logits = Float32.(option_flow_logits(selector["params"], catalog))
        probs = Float64.(option_flow_probs_from_logits(logits))
        if arm == "option_flow_greedy"
            idx = argmax(logits)
        else
            cumulative = cumsum(probs ./ max(sum(probs), eps(Float64)))
            idx = clamp(searchsortedfirst(cumulative, rand(rng)), 1, length(specs))
        end
        return idx, probs, logits
    else
        error("unknown deployable arm: $(arm)")
    end
end

function tracker_row(frontier::MolecularFrontierBuffer, calls::Int; step::Int, selected::String="initial", utility::Float64=0.0, raw_delta::Float64=0.0, commits::Int=0, new_calls::Int=0)
    summary = frontier_quality_summary(frontier; topk=10)
    return Dict{String,Any}(
        "step" => step,
        "calls" => calls,
        "top1" => Float64(summary["top1"]),
        "top10_mean" => Float64(summary["top10_mean"]),
        "frontier_size" => Int(summary["size"]),
        "n_scaffolds" => Int(summary["n_scaffolds"]),
        "selected_option" => selected,
        "utility" => utility,
        "raw_delta" => raw_delta,
        "commits" => commits,
        "new_calls" => new_calls,
    )
end

function normalized_auc(rows::Vector{Dict{String,Any}}, metric::String)
    length(rows) >= 1 || return 0.0
    ordered = sort(rows, by = r -> Int(r["calls"]))
    if length(ordered) == 1
        return Float64(ordered[1][metric])
    end
    area = 0.0
    for i in 2:length(ordered)
        x0 = Float64(ordered[i-1]["calls"])
        x1 = Float64(ordered[i]["calls"])
        y0 = Float64(ordered[i-1][metric])
        y1 = Float64(ordered[i][metric])
        x1 < x0 && continue
        area += (x1 - x0) * (y0 + y1) / 2.0
    end
    span = Float64(max(1, Int(ordered[end]["calls"]) - Int(ordered[1]["calls"])))
    return area / span
end

function summarize_online_rows(rows::Vector{Dict{String,Any}}, frontier::MolecularFrontierBuffer, calls::Int, cache::Dict{String,Float64})
    final = rows[end]
    option_counts = Dict{String,Int}()
    total_commits = 0
    total_utility = 0.0
    for row in rows
        opt = String(row["selected_option"])
        opt == "initial" && continue
        option_counts[opt] = get(option_counts, opt, 0) + 1
        total_commits += Int(row["commits"])
        total_utility += Float64(row["utility"])
    end
    source_summary = frontier_source_summary(frontier; topk=10)
    return Dict{String,Any}(
        "calls" => calls,
        "cache_size" => length(cache),
        "auc_top10_mean" => normalized_auc(rows, "top10_mean"),
        "auc_top1" => normalized_auc(rows, "top1"),
        "final_top1" => Float64(final["top1"]),
        "final_top10_mean" => Float64(final["top10_mean"]),
        "final_frontier_size" => Int(final["frontier_size"]),
        "final_n_scaffolds" => Int(final["n_scaffolds"]),
        "unique_evaluated" => length(cache),
        "option_counts" => option_counts,
        "total_commits" => total_commits,
        "total_positive_utility" => total_utility,
        "source_summary" => source_summary,
    )
end

function run_online_arm(task_name::String,
                        task_names::Vector{String},
                        arm::String,
                        specs::Vector{StrictOptionSpec},
                        selector,
                        prior_best::String;
                        online_budget::Int,
                        seed::Int,
                        max_steps::Int,
                        vocab,
                        budget_scale::Int=1000)
    target = get(TARGET_SMILES, task_name, nothing)
    reward_one, reward_batch, calls, cache = build_reward_functions(task_name, online_budget)
    rng = MersenneTwister(seed + abs(hash((task_name, arm))) % 10_000_000)
    frontier = MolecularFrontierBuffer(512)
    seed_added = seed_frontier!(frontier, task_name, reward_batch)
    rows = Dict{String,Any}[tracker_row(frontier, calls[]; step=0)]
    action_logs = Dict{String,Any}[]
    zero_call_streak = 0

    for step in 1:max_steps
        calls[] >= online_budget && break
        length(frontier) >= 2 || break
        before_calls = calls[]
        budget_remaining = max(0, online_budget - calls[])

        idx, probs, logits = choose_spec(arm, frontier, task_name, task_names, specs, selector, prior_best;
            snapshot_index=step,
            budget_remaining=budget_remaining,
            rng=rng,
            budget_scale=budget_scale)
        spec = specs[idx]
        result = run_option_from_snapshot(frontier, task_name, target, reward_one, reward_batch, vocab, spec;
            snapshot_index=step,
            option_index=idx,
            base_seed=seed,
            budget_remaining=budget_remaining)
        replace_frontier!(frontier, result["frontier"]::MolecularFrontierBuffer)
        new_calls = calls[] - before_calls
        new_calls == 0 ? (zero_call_streak += 1) : (zero_call_streak = 0)
        push!(rows, tracker_row(frontier, calls[];
            step=step,
            selected=spec.name,
            utility=Float64(result["utility"]),
            raw_delta=Float64(result["raw_delta_sum"]),
            commits=Int(result["commits_applied"]),
            new_calls=new_calls))
        push!(action_logs, Dict{String,Any}(
            "step" => step,
            "arm" => arm,
            "selected_option" => spec.name,
            "selected_index" => idx,
            "probs" => probs,
            "logits" => logits,
            "calls_before" => before_calls,
            "calls_after" => calls[],
            "new_calls" => new_calls,
            "utility" => Float64(result["utility"]),
            "raw_delta_sum" => Float64(result["raw_delta_sum"]),
            "best_reward" => Float64(result["best_reward"]),
            "commits" => Int(result["commits_applied"]),
        ))
        zero_call_streak >= 8 && break
    end

    summary = summarize_online_rows(rows, frontier, calls[], cache)
    return Dict{String,Any}(
        "task" => task_name,
        "arm" => arm,
        "seed" => seed,
        "online_budget" => online_budget,
        "seed_entries_added" => seed_added,
        "summary" => summary,
        "trajectory" => rows,
        "actions" => action_logs,
        "deployable" => arm in DEPLOYABLE_ARMS,
        "diagnostic_only" => arm == "oracle_upper",
    )
end

function run_oracle_upper_arm(task_name::String,
                              task_names::Vector{String},
                              specs::Vector{StrictOptionSpec};
                              online_budget::Int,
                              seed::Int,
                              max_steps::Int,
                              vocab,
                              budget_scale::Int=1000)
    target = get(TARGET_SMILES, task_name, nothing)
    reward_one, reward_batch, calls, cache = build_reward_functions(task_name, online_budget)
    frontier = MolecularFrontierBuffer(512)
    seed_added = seed_frontier!(frontier, task_name, reward_batch)
    rows = Dict{String,Any}[tracker_row(frontier, calls[]; step=0)]
    action_logs = Dict{String,Any}[]

    for step in 1:max_steps
        calls[] >= online_budget && break
        length(frontier) >= 2 || break
        before_calls = calls[]
        budget_remaining = max(0, online_budget - calls[])
        option_results = Dict{String,Any}[]
        for (idx, spec) in enumerate(specs)
            calls[] >= online_budget && break
            push!(option_results, run_option_from_snapshot(frontier, task_name, target, reward_one, reward_batch, vocab, spec;
                snapshot_index=step,
                option_index=idx,
                base_seed=seed,
                budget_remaining=max(0, online_budget - calls[])))
        end
        isempty(option_results) && break
        utilities = [Float64(r["utility"]) for r in option_results]
        best_local = argmax(utilities)
        best_result = option_results[best_local]
        spec = best_result["spec"]::StrictOptionSpec
        replace_frontier!(frontier, best_result["frontier"]::MolecularFrontierBuffer)
        new_calls = calls[] - before_calls
        push!(rows, tracker_row(frontier, calls[];
            step=step,
            selected=spec.name,
            utility=Float64(best_result["utility"]),
            raw_delta=Float64(best_result["raw_delta_sum"]),
            commits=Int(best_result["commits_applied"]),
            new_calls=new_calls))
        push!(action_logs, Dict{String,Any}(
            "step" => step,
            "arm" => "oracle_upper",
            "selected_option" => spec.name,
            "utilities" => utilities,
            "calls_before" => before_calls,
            "calls_after" => calls[],
            "new_calls" => new_calls,
        ))
        new_calls == 0 && break
    end

    summary = summarize_online_rows(rows, frontier, calls[], cache)
    return Dict{String,Any}(
        "task" => task_name,
        "arm" => "oracle_upper",
        "seed" => seed,
        "online_budget" => online_budget,
        "seed_entries_added" => seed_added,
        "summary" => summary,
        "trajectory" => rows,
        "actions" => action_logs,
        "deployable" => false,
        "diagnostic_only" => true,
    )
end

function mean_or_nan(values::Vector{Float64})
    isempty(values) && return NaN
    return mean(values)
end

function std_or_zero(values::Vector{Float64})
    length(values) <= 1 && return 0.0
    return std(values)
end

function aggregate_runs(runs::Vector{Dict{String,Any}})
    groups = Dict{Tuple{String,String},Vector{Dict{String,Any}}}()
    for run in runs
        key = (String(run["task"]), String(run["arm"]))
        groups[key] = get(groups, key, Dict{String,Any}[])
        push!(groups[key], run)
    end
    rows = Dict{String,Any}[]
    for ((task, arm), vals) in groups
        aucs = [Float64(v["summary"]["auc_top10_mean"]) for v in vals]
        top1 = [Float64(v["summary"]["final_top1"]) for v in vals]
        top10 = [Float64(v["summary"]["final_top10_mean"]) for v in vals]
        unique = [Float64(v["summary"]["unique_evaluated"]) for v in vals]
        scaff = [Float64(v["summary"]["final_n_scaffolds"]) for v in vals]
        commits = [Float64(v["summary"]["total_commits"]) for v in vals]
        push!(rows, Dict{String,Any}(
            "task" => task,
            "arm" => arm,
            "n" => length(vals),
            "deployable" => Bool(vals[1]["deployable"]),
            "auc_top10_mean_mean" => mean_or_nan(aucs),
            "auc_top10_mean_std" => std_or_zero(aucs),
            "final_top1_mean" => mean_or_nan(top1),
            "final_top10_mean_mean" => mean_or_nan(top10),
            "unique_evaluated_mean" => mean_or_nan(unique),
            "n_scaffolds_mean" => mean_or_nan(scaff),
            "total_commits_mean" => mean_or_nan(commits),
        ))
    end
    sort!(rows, by = r -> (String(r["task"]), String(r["arm"])))
    return rows
end

function by_arm_overall(agg_rows::Vector{Dict{String,Any}})
    groups = Dict{String,Vector{Dict{String,Any}}}()
    for row in agg_rows
        Bool(row["deployable"]) || continue
        arm = String(row["arm"])
        groups[arm] = get(groups, arm, Dict{String,Any}[])
        push!(groups[arm], row)
    end
    out = Dict{String,Any}[]
    for (arm, rows) in groups
        aucs = [Float64(r["auc_top10_mean_mean"]) for r in rows if isfinite(Float64(r["auc_top10_mean_mean"]))]
        top10 = [Float64(r["final_top10_mean_mean"]) for r in rows if isfinite(Float64(r["final_top10_mean_mean"]))]
        push!(out, Dict{String,Any}(
            "arm" => arm,
            "tasks" => [String(r["task"]) for r in rows],
            "mean_auc_top10" => mean_or_nan(aucs),
            "mean_final_top10" => mean_or_nan(top10),
        ))
    end
    sort!(out, by = r -> -Float64(r["mean_auc_top10"]))
    return out
end

function online_gates(agg_rows::Vector{Dict{String,Any}}; protocol::String)
    task_rows = Dict{String,Dict{String,Dict{String,Any}}}()
    for row in agg_rows
        Bool(row["deployable"]) || continue
        task = String(row["task"])
        arm = String(row["arm"])
        task_rows[task] = get(task_rows, task, Dict{String,Dict{String,Any}}())
        task_rows[task][arm] = row
    end
    sample_wins_uniform = 0
    sample_wins_heuristic = 0
    sample_available = 0
    task_details = Dict{String,Any}()
    for (task, arms) in task_rows
        haskey(arms, "option_flow_sample") || continue
        sample_available += 1
        sample_auc = Float64(arms["option_flow_sample"]["auc_top10_mean_mean"])
        uniform_auc = haskey(arms, "uniform_schema") ? Float64(arms["uniform_schema"]["auc_top10_mean_mean"]) : NaN
        heuristic_auc = haskey(arms, "heuristic_mixed_h2") ? Float64(arms["heuristic_mixed_h2"]["auc_top10_mean_mean"]) : NaN
        win_u = isfinite(uniform_auc) && sample_auc > uniform_auc
        win_h = isfinite(heuristic_auc) && sample_auc > heuristic_auc
        sample_wins_uniform += win_u ? 1 : 0
        sample_wins_heuristic += win_h ? 1 : 0
        task_details[task] = Dict("sample_auc"=>sample_auc, "uniform_auc"=>uniform_auc, "heuristic_auc"=>heuristic_auc, "wins_uniform"=>win_u, "wins_heuristic"=>win_h)
    end
    gate_a = sample_available > 0 && sample_wins_uniform >= ceil(Int, sample_available * 2/3) && sample_wins_heuristic >= ceil(Int, sample_available * 2/3)
    celecoxib_detail = get(task_details, "celecoxib_rediscovery", nothing)
    structural_positive = celecoxib_detail isa Dict && Bool(get(celecoxib_detail, "wins_uniform", false)) && Bool(get(celecoxib_detail, "wins_heuristic", false))
    pass = gate_a || structural_positive
    verdict = if protocol == "O2"
        pass ? "ONLINE_FAIR_BUDGET_SIGNAL_PRESENT" : "ONLINE_SIGNAL_WEAK_UNDER_TOTAL_BUDGET"
    else
        pass ? "ONLINE_SELECTOR_USEFUL_BUT_FAIRNESS_OPEN" : "ONLINE_DEPLOYMENT_SIGNAL_WEAK_DESPITE_E3_OBJECT"
    end
    return Dict{String,Any}(
        "protocol" => protocol,
        "sample_available_tasks" => sample_available,
        "sample_wins_uniform" => sample_wins_uniform,
        "sample_wins_heuristic" => sample_wins_heuristic,
        "gate_a_online_usefulness" => gate_a,
        "structural_positive" => structural_positive,
        "pass" => pass,
        "verdict" => verdict,
        "task_details" => task_details,
    )
end

function run_protocol_o1(tasks, arms, seeds, budget, specs, vocab; train_budget::Int, train_snapshots::Int, epochs::Int, lr::Float64, max_steps::Int)
    logmsg("O1 warm-start protocol tasks=$(tasks) arms=$(arms) seeds=$(seeds) online_budget=$(budget) train_budget_per_task=$(train_budget)")
    task_names = sort(tasks)
    train_bundle = train_catalog_bundle(tasks, task_names, specs;
        budget_per_task=train_budget,
        snapshots=train_snapshots,
        base_seed=404,
        vocab=vocab,
        budget_scale=max(budget, train_budget))
    selector = train_selector(train_bundle["catalogs"]; seed=17, epochs=epochs, lr=lr)
    prior_best = prior_best_option_id(train_bundle["catalogs"])
    logmsg("O1 prior_best_schema=$(prior_best)")
    runs = Dict{String,Any}[]
    for seed in seeds, task in tasks, arm in arms
        if arm == "oracle_upper"
            run = run_oracle_upper_arm(task, task_names, specs;
                online_budget=budget,
                seed=seed,
                max_steps=max_steps,
                vocab=vocab,
                budget_scale=max(budget, train_budget))
        else
            run = run_online_arm(task, task_names, arm, specs, selector, prior_best;
                online_budget=budget,
                seed=seed,
                max_steps=max_steps,
                vocab=vocab,
                budget_scale=max(budget, train_budget))
        end
        s = run["summary"]
        logmsg("O1 run task=$(task) arm=$(arm) seed=$(seed) calls=$(s["calls"]) auc=$(round(s["auc_top10_mean"], digits=4)) top1=$(round(s["final_top1"], digits=4)) top10=$(round(s["final_top10_mean"], digits=4)) unique=$(s["unique_evaluated"])")
        push!(runs, run)
    end
    agg = aggregate_runs(runs)
    gate = online_gates(agg; protocol="O1")
    return Dict{String,Any}(
        "protocol" => "O1_warm_started_online_selector",
        "tasks" => tasks,
        "arms" => arms,
        "seeds" => seeds,
        "online_budget" => budget,
        "training_budget_per_task" => train_budget,
        "training_budget_counted_in_headline" => false,
        "training_catalog_bundle" => train_bundle,
        "selector_train_metrics" => selector["train_metrics"],
        "selector_val_metrics" => selector["val_metrics"],
        "prior_best_schema" => prior_best,
        "runs" => runs,
        "aggregate_rows" => agg,
        "overall_by_arm" => by_arm_overall(agg),
        "gate" => gate,
    )
end

function run_protocol_o2(tasks, arms, seeds, total_budget, specs, vocab; train_fraction::Float64, train_snapshots::Int, epochs::Int, lr::Float64, max_steps::Int)
    logmsg("O2 total-budget protocol tasks=$(tasks) arms=$(arms) seeds=$(seeds) total_budget=$(total_budget) train_fraction=$(train_fraction)")
    task_names_global = sort(tasks)
    runs = Dict{String,Any}[]
    training_records = Dict{String,Any}[]
    for seed in seeds, task in tasks
        selector_cache = nothing
        prior_best = "mixed_h2"
        train_budget = max(1, round(Int, total_budget * train_fraction))
        online_budget_selector = max(1, total_budget - train_budget)
        for arm in arms
            if arm in SELECTOR_ARMS
                if isnothing(selector_cache)
                    # Variant A: same-task training budget is explicitly charged to selector-dependent arms.
                    train_bundle = train_catalog_bundle([task], [task], specs;
                        budget_per_task=train_budget,
                        snapshots=train_snapshots,
                        base_seed=404 + seed,
                        vocab=vocab,
                        budget_scale=total_budget)
                    selector_cache = train_selector(train_bundle["catalogs"]; seed=seed, epochs=epochs, lr=lr)
                    prior_best = prior_best_option_id(train_bundle["catalogs"])
                    push!(training_records, Dict{String,Any}(
                        "task" => task,
                        "seed" => seed,
                        "train_budget" => train_budget,
                        "actual_train_calls" => train_bundle["total_calls"],
                        "online_budget_after_training" => online_budget_selector,
                        "prior_best_schema" => prior_best,
                        "catalog_stats" => train_bundle["stats"],
                        "train_metrics" => selector_cache["train_metrics"],
                        "val_metrics" => selector_cache["val_metrics"],
                    ))
                end
                run = run_online_arm(task, [task], arm, specs, selector_cache, prior_best;
                    online_budget=online_budget_selector,
                    seed=seed,
                    max_steps=max_steps,
                    vocab=vocab,
                    budget_scale=total_budget)
                run["total_budget"] = total_budget
                run["charged_training_budget"] = train_budget
                run["actual_training_calls"] = get(training_records[end], "actual_train_calls", train_budget)
            else
                if arm == "oracle_upper"
                    run = run_oracle_upper_arm(task, task_names_global, specs;
                        online_budget=total_budget,
                        seed=seed,
                        max_steps=max_steps,
                        vocab=vocab,
                        budget_scale=total_budget)
                else
                    run = run_online_arm(task, task_names_global, arm, specs, nothing, prior_best;
                        online_budget=total_budget,
                        seed=seed,
                        max_steps=max_steps,
                        vocab=vocab,
                        budget_scale=total_budget)
                end
                run["total_budget"] = total_budget
                run["charged_training_budget"] = 0
                run["actual_training_calls"] = 0
            end
            s = run["summary"]
            logmsg("O2 run task=$(task) arm=$(arm) seed=$(seed) online_calls=$(s["calls"]) train_charged=$(run["charged_training_budget"]) auc=$(round(s["auc_top10_mean"], digits=4)) top1=$(round(s["final_top1"], digits=4)) top10=$(round(s["final_top10_mean"], digits=4))")
            push!(runs, run)
        end
    end
    agg = aggregate_runs(runs)
    gate = online_gates(agg; protocol="O2")
    return Dict{String,Any}(
        "protocol" => "O2_total_budget_fair_variant_A",
        "tasks" => tasks,
        "arms" => arms,
        "seeds" => seeds,
        "total_budget" => total_budget,
        "train_fraction" => train_fraction,
        "selector_online_budget" => max(1, total_budget - max(1, round(Int, total_budget * train_fraction))),
        "training_budget_counted_in_headline" => true,
        "training_records" => training_records,
        "runs" => runs,
        "aggregate_rows" => agg,
        "overall_by_arm" => by_arm_overall(agg),
        "gate" => gate,
    )
end

function main()
    mode = strip(get(ENV, "OPTION_FLOW_ONLINE_MODE", "smoke"))
    default_tasks = mode == "smoke" ? ["qed"] : ["qed", "drd2", "celecoxib_rediscovery"]
    default_arms = mode == "smoke" ? ["uniform_schema", "option_flow_sample"] : ["uniform_schema", "heuristic_mixed_h2", "prior_best_schema", "option_flow_sample", "option_flow_greedy"]
    tasks = parse_csv_strings("OPTION_FLOW_ONLINE_TASKS", default_tasks)
    arms = parse_csv_strings("OPTION_FLOW_ONLINE_ARMS", default_arms)
    seeds = parse_csv_ints("OPTION_FLOW_ONLINE_SEEDS", mode == "smoke" ? [17] : [17, 23])
    budget = parse_env_int("OPTION_FLOW_ONLINE_BUDGET", mode == "smoke" ? 150 : 400)
    train_budget = parse_env_int("OPTION_FLOW_ONLINE_TRAIN_BUDGET", mode == "smoke" ? 220 : max(250, budget))
    train_fraction = parse_env_float("OPTION_FLOW_ONLINE_TRAIN_FRACTION", 0.30)
    train_snapshots = parse_env_int("OPTION_FLOW_ONLINE_TRAIN_SNAPSHOTS", mode == "smoke" ? 2 : 3)
    epochs = parse_env_int("OPTION_FLOW_ONLINE_EPOCHS", mode == "smoke" ? 60 : 140)
    lr = parse_env_float("OPTION_FLOW_ONLINE_LR", 0.012)
    max_steps = parse_env_int("OPTION_FLOW_ONLINE_MAX_STEPS", max(12, ceil(Int, budget / 4)))
    specs = option_specs()
    vocab = SMILESVocabulary()

    invalid = [a for a in arms if !(a in DEPLOYABLE_ARMS || a == "oracle_upper")]
    isempty(invalid) || error("unknown arms: $(invalid)")

    logmsg("Option-Flow online-lite PMO mode=$(mode) tasks=$(tasks) arms=$(arms) seeds=$(seeds) budget=$(budget) max_steps=$(max_steps)")

    result = if mode == "o2_total_budget"
        run_protocol_o2(tasks, arms, seeds, budget, specs, vocab;
            train_fraction=train_fraction,
            train_snapshots=train_snapshots,
            epochs=epochs,
            lr=lr,
            max_steps=max_steps)
    else
        run_protocol_o1(tasks, arms, seeds, budget, specs, vocab;
            train_budget=train_budget,
            train_snapshots=train_snapshots,
            epochs=epochs,
            lr=lr,
            max_steps=max_steps)
    end

    verdict = result["gate"]["verdict"]
    bundle = Dict{String,Any}(
        "created_at" => string(now()),
        "authoritative_date_note" => "User supplied authoritative date/time: Thursday, 2026-06-18 12:28 EDT",
        "mode" => mode,
        "protocol_result" => result,
        "verdict" => verdict,
        "limitations" => [
            "Online-lite proof, not official 10K/23-task PMO SOTA benchmark.",
            "O1 warm-start selector training oracle calls are reported separately and not counted in O1 headline online budget.",
            "O2 Variant A counts same-task selector-training budget by reducing Option-Flow online budget; this is fairer but harsher and still small-scale.",
            "Pure HE option selection is not yet full integrated GFlowNet PMO unless O3 integration is added separately.",
        ],
    )
    out = joinpath(OUTDIR, "option_flow_online_lite_pmo_$(mode)_results.jls")
    latest = joinpath(OUTDIR, "option_flow_online_lite_pmo_latest_results.jls")
    serialize(out, bundle)
    serialize(latest, bundle)
    logmsg("VERDICT=$(verdict)")
    logmsg("Saved results: $(abspath(out))")
    logmsg("Saved latest: $(abspath(latest))")
end

main()
