#!/usr/bin/env julia

# Option-Flow E3 strict same-snapshot generated-catalog POC.
#
# This is the first direct real-molecule test of:
#     Pθ(ω | S_t) ∝ U(ω; S_t)
#
# For each frozen frontier S_t, clone the frontier and run multiple bounded HE
# option schemas from the exact same state. Each option is scored by realized
# frontier utility in its clone. The resulting catalog is strict: candidates
# share the same snapshot_id/state features.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GFlowNet
using Random
using Serialization
using Statistics
using Dates

include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "oracle_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "training", "option_flow_dataset.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "training", "option_flow_model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "training", "option_flow_loss.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "training", "option_flow_training.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "training", "option_flow_real_catalog.jl"))

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUTDIR = joinpath(ROOT, "checkpoints", "option_flow_strict_e3_poc")
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

function logmsg(msg)
    println("[", Dates.format(now(), "HH:MM:SS"), "] ", msg)
    flush(stdout)
end

function parse_env_int(name::String, default::Int)
    value = get(ENV, name, "")
    isempty(value) && return default
    return parse(Int, value)
end

function parse_env_float(name::String, default::Float64)
    value = get(ENV, name, "")
    isempty(value) && return default
    return parse(Float64, value)
end

function parse_tasks(default::Vector{String})
    raw = strip(get(ENV, "OPTION_FLOW_E3_TASKS", ""))
    isempty(raw) && return default
    return [String(strip(x)) for x in split(raw, ',') if !isempty(strip(x))]
end

function parse_seeds(default::Vector{Int})
    raw = strip(get(ENV, "OPTION_FLOW_E3_TRAIN_SEEDS", ""))
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

function strict_state_features(frontier::MolecularFrontierBuffer, task_name::String, task_names::Vector{String}, budget_remaining::Int, snapshot_index::Int)
    summary = frontier_quality_summary(frontier; topk=10)
    task_onehot = zeros(Float32, length(task_names))
    idx = findfirst(==(task_name), task_names)
    idx === nothing || (task_onehot[idx] = 1.0f0)
    return Float32.(vcat(task_onehot, Float32[
        Float32(summary["size"]) / 512.0f0,
        Float32(summary["top1"]),
        Float32(summary["top10_mean"]),
        Float32(summary["n_scaffolds"]) / 128.0f0,
        Float32(get(summary, "graph_unique_count", 0.0)) / 128.0f0,
        Float32(budget_remaining) / 1000.0f0,
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
        episode_id="e3-$(task_name)-snap$(snapshot_index)-$(spec.name)-$(option_index)",
        operator_override=spec.operator_override,
    )
    utility_sum = sum(max(0.0, step.frontier_utility_delta) for step in episode.steps; init=0.0)
    raw_delta_sum = sum(step.frontier_utility_delta for step in episode.steps; init=0.0)
    best_reward = episode.best_reward
    return Dict{String,Any}(
        "spec" => spec,
        "episode" => episode,
        "frontier" => local_frontier,
        "trajectory_entries" => trajectory_buffer.entries,
        "diagnostics" => Dict(
            "decision_logs" => diagnostics_buffer.logs,
            "proposal_logs" => diagnostics_buffer.proposal_logs,
            "basin_logs" => diagnostics_buffer.basin_logs,
            "parent_logs" => diagnostics_buffer.parent_logs,
            "operator_logs" => diagnostics_buffer.operator_logs,
        ),
        "utility" => utility_sum,
        "raw_delta_sum" => raw_delta_sum,
        "best_reward" => best_reward,
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
                                           budget_remaining::Int)
    snapshot = create_frontier_snapshot(frontier;
        max_entries=min(64, max(length(frontier), 1)),
        target_smiles=target_smiles,
        budget_remaining=budget_remaining,
        created_at_step=snapshot_index)
    state_features = strict_state_features(frontier, task_name, task_names, budget_remaining, snapshot_index)
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
            "evidence_level" => "E3_strict_same_snapshot",
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
    frontier.entries = [MolecularFrontierEntry(e.smiles, e.scaffold, e.reward, e.source, e.parent_smiles, e.operator, e.novelty_score, e.tb_delta_abs, e.visits) for e in best_frontier.entries]
    frontier.seen_smiles = copy(best_frontier.seen_smiles)
    frontier.scaffold_counts = copy(best_frontier.scaffold_counts)
    frontier.needs_refresh = true
    spec = option_results[best_idx]["spec"]::StrictOptionSpec
    return spec.name
end

function strict_catalog_stats(catalogs::Vector{OptionFlowCatalog})
    base = option_flow_catalog_stats(catalogs)
    strict_ok = all(length(unique(c.snapshot_id for _ in c.candidates)) == 1 for c in catalogs)
    utilities = [option_flow_utilities(c) for c in catalogs]
    spreads = [maximum(u) - minimum(u) for u in utilities if !isempty(u)]
    return merge(base, Dict{String,Any}(
        "evidence_level" => "E3_strict_same_snapshot",
        "strict_same_snapshot_catalogs" => strict_ok,
        "mean_utility_spread" => isempty(spreads) ? 0.0 : mean(spreads),
        "positive_spread_catalogs" => count(>(1.0e-8), spreads),
    ))
end

function train_and_evaluate_strict(catalogs::Vector{OptionFlowCatalog}; seeds::Vector{Int}, epochs::Int, lr::Float64)
    seed_results = Dict{String,Any}[]
    for seed in seeds
        result = train_option_flow_model(catalogs; config=OptionFlowTrainingConfig(
            n_epochs=epochs,
            learning_rate=lr,
            hidden_dim=64,
            second_hidden_dim=32,
            validation_fraction=0.30,
            seed=seed,
            verbose=false,
        ))
        val_metrics = evaluate_real_option_flow_model(result["params"], result["val_catalogs"])
        train_metrics = evaluate_real_option_flow_model(result["params"], result["train_catalogs"])
        gate = e1_summary_proxy_gate(val_metrics)
        push!(seed_results, Dict{String,Any}(
            "seed" => seed,
            "train_catalog_count" => length(result["train_catalogs"]),
            "val_catalog_count" => length(result["val_catalogs"]),
            "train_metrics" => train_metrics,
            "val_metrics" => val_metrics,
            "gate" => gate,
            "history" => result["history"],
            "param_count" => result["param_count"],
        ))
        logmsg("train_seed=$(seed) val_ce_gain=$(round(val_metrics["mean_ce_vs_uniform"], digits=4)) util_lift=$(round(val_metrics["mean_expected_utility_lift"], digits=4)) top_lift=$(round(val_metrics["mean_top_quartile_lift"], digits=4)) rank=$(round(val_metrics["mean_rank_correlation"], digits=4)) gate=$(gate["pass"])")
    end
    return seed_results
end

function aggregate_seed_results(seed_results::Vector{Dict{String,Any}})
    function vals(path...)
        out = Float64[]
        for sr in seed_results
            cur = sr
            ok = true
            for key in path
                if cur isa AbstractDict && haskey(cur, key)
                    cur = cur[key]
                else
                    ok = false
                    break
                end
            end
            ok && isfinite(Float64(cur)) && push!(out, Float64(cur))
        end
        return out
    end
    function ms(path...)
        v = vals(path...)
        isempty(v) && return Dict("mean"=>NaN, "std"=>NaN, "n"=>0)
        return Dict("mean"=>mean(v), "std"=>length(v)>1 ? std(v) : 0.0, "n"=>length(v))
    end
    gates = [Bool(get(sr["gate"], "pass", false)) for sr in seed_results]
    return Dict{String,Any}(
        "n_seeds" => length(seed_results),
        "gate_pass_count" => count(identity, gates),
        "majority_pass" => count(identity, gates) >= ceil(Int, length(gates)/2),
        "ce_gain" => ms("val_metrics", "mean_ce_vs_uniform"),
        "expected_utility_lift" => ms("val_metrics", "mean_expected_utility_lift"),
        "expected_utility_lift_fraction" => ms("val_metrics", "mean_expected_utility_lift_fraction"),
        "top_quartile_lift" => ms("val_metrics", "mean_top_quartile_lift"),
        "rank_correlation" => ms("val_metrics", "mean_rank_correlation"),
        "entropy" => ms("val_metrics", "mean_entropy"),
        "uniform_ce" => ms("val_metrics", "mean_uniform_ce"),
        "model_greedy_utility" => ms("val_metrics", "greedy_model", "mean_expected_utility"),
        "oracle_greedy_utility" => ms("val_metrics", "oracle_greedy", "mean_expected_utility"),
    )
end

function strict_e3_gate(agg::Dict{String,Any})
    ce = Float64(agg["ce_gain"]["mean"])
    util = Float64(agg["expected_utility_lift"]["mean"])
    top = Float64(agg["top_quartile_lift"]["mean"])
    rank = Float64(agg["rank_correlation"]["mean"])
    entropy = Float64(agg["entropy"]["mean"])
    uniform_entropy = Float64(agg["uniform_ce"]["mean"])
    pass = ce > 0 && util > 0 && top > 0.02 && rank > 0 && entropy >= 0.40 * uniform_entropy && Bool(agg["majority_pass"])
    return Dict{String,Any}(
        "gate" => "E3_strict_same_snapshot",
        "pass" => pass,
        "ce_positive" => ce > 0,
        "utility_lift_positive" => util > 0,
        "top_lift_gt_0p02" => top > 0.02,
        "rank_positive" => rank > 0,
        "entropy_noncollapse" => entropy >= 0.40 * uniform_entropy,
        "majority_seeds_pass" => Bool(agg["majority_pass"]),
    )
end

function main()
    mode = get(ENV, "OPTION_FLOW_E3_MODE", "strict_e3")
    tasks = parse_tasks(["qed", "drd2", "celecoxib_rediscovery"])
    n_snapshots = parse_env_int("OPTION_FLOW_E3_SNAPSHOTS", mode == "smoke" ? 2 : 5)
    oracle_budget = parse_env_int("OPTION_FLOW_E3_ORACLE_BUDGET", mode == "smoke" ? 250 : 900)
    train_epochs = parse_env_int("OPTION_FLOW_E3_EPOCHS", mode == "smoke" ? 80 : 240)
    lr = parse_env_float("OPTION_FLOW_E3_LR", 0.012)
    base_seed = parse_env_int("OPTION_FLOW_E3_SEED", 404)
    train_seeds = parse_seeds(mode == "smoke" ? [17] : [17, 23, 31])
    specs = option_specs()
    vocab = SMILESVocabulary()

    logmsg("Option-Flow strict E3 POC mode=$(mode) tasks=$(tasks) snapshots=$(n_snapshots) specs=$(length(specs))")
    catalogs = OptionFlowCatalog[]
    per_task = Dict{String,Any}()
    task_names = sort(tasks)

    for task in tasks
        target = get(TARGET_SMILES, task, nothing)
        logmsg("Task $(task): initializing oracle")
        reward_one, reward_batch, calls, cache = build_reward_functions(task, oracle_budget)
        frontier = MolecularFrontierBuffer(512)
        added = seed_frontier!(frontier, task, reward_batch)
        logmsg("Task $(task): seed frontier added=$(added) frontier=$(length(frontier)) calls=$(calls[])")
        task_catalogs = OptionFlowCatalog[]
        task_options = Vector{Dict{String,Any}}()
        advanced_by = String[]
        for snapshot_index in 1:n_snapshots
            length(frontier) >= 2 || break
            budget_remaining = max(0, oracle_budget - calls[])
            catalog, option_results, snapshot = build_strict_catalog_for_snapshot(frontier, task, task_names, target,
                reward_one, reward_batch, vocab, specs;
                snapshot_index=snapshot_index,
                base_seed=base_seed,
                budget_remaining=budget_remaining)
            push!(catalogs, catalog)
            push!(task_catalogs, catalog)
            push!(task_options, Dict{String,Any}(
                "snapshot_index" => snapshot_index,
                "snapshot_id" => string(snapshot.snapshot_id),
                "utilities" => option_flow_utilities(catalog),
                "target_probs" => catalog.target_probs,
                "option_metadata" => [c.metadata for c in catalog.candidates],
            ))
            chosen = advance_frontier_with_best!(frontier, option_results)
            push!(advanced_by, chosen)
            logmsg("Task $(task) snapshot=$(snapshot_index) id=$(snapshot.snapshot_id) utilities=$(round.(option_flow_utilities(catalog), digits=4)) advance=$(chosen) frontier=$(length(frontier)) calls=$(calls[])")
        end
        per_task[task] = Dict{String,Any}(
            "catalog_stats" => strict_catalog_stats(task_catalogs),
            "n_oracle_calls" => calls[],
            "cache_size" => length(cache),
            "advanced_by" => advanced_by,
            "snapshots" => task_options,
        )
    end

    stats = strict_catalog_stats(catalogs)
    logmsg("Generated strict catalogs=$(length(catalogs)) candidates=$(get(stats, "n_candidates", 0)) mean_spread=$(round(get(stats, "mean_utility_spread", 0.0), digits=4))")
    min_required_catalogs = mode == "smoke" ? 2 : 3
    length(catalogs) >= min_required_catalogs || error("Too few strict catalogs generated: $(length(catalogs))")

    seed_results = train_and_evaluate_strict(catalogs; seeds=train_seeds, epochs=train_epochs, lr=lr)
    aggregate = aggregate_seed_results(seed_results)
    gate = strict_e3_gate(aggregate)
    verdict = gate["pass"] ? "E3_STRICT_OBJECT_SIGNAL_PRESENT" : "E3_STRICT_OBJECT_SIGNAL_WEAK_OR_ABSENT"
    logmsg("VERDICT=$(verdict) gate=$(gate)")

    bundle = Dict{String,Any}(
        "created_at" => string(now()),
        "authoritative_date_note" => "User supplied authoritative date/time: Thursday, 2026-06-11 17:17 EDT",
        "mode" => mode,
        "evidence_level" => "E3_strict_same_snapshot",
        "tasks" => tasks,
        "option_specs" => [Dict("name"=>s.name, "operator_override"=>isnothing(s.operator_override) ? "mixed" : string(s.operator_override), "horizon"=>s.horizon, "max_candidates"=>s.max_candidates, "min_reward_ratio"=>s.min_reward_ratio, "allow_crossover"=>s.allow_crossover) for s in specs],
        "catalog_stats" => stats,
        "per_task" => per_task,
        "seed_results" => seed_results,
        "aggregate" => aggregate,
        "gate" => gate,
        "verdict" => verdict,
        "limitations" => [
            "Small E3 POC, not 23-task PMO/SOTA benchmark.",
            "Option candidates are controlled HE schemas from cloned same-state frontiers, not yet a learned online option generator.",
            "Frontier advances between snapshots use oracle-best option clone for dataset generation; each catalog itself remains strict same-snapshot.",
        ],
    )
    out = joinpath(OUTDIR, "option_flow_strict_e3_poc_$(mode)_results.jls")
    latest = joinpath(OUTDIR, "option_flow_strict_e3_poc_latest_results.jls")
    serialize(out, bundle)
    serialize(latest, bundle)
    logmsg("Saved results: $(abspath(out))")
    logmsg("Saved latest: $(abspath(latest))")
end

main()
