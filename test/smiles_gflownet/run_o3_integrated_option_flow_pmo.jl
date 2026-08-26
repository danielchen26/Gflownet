#!/usr/bin/env julia

# O3 Integrated Option-Flow PMO-Lite
#
# This runner tests Option-Flow as an HE episode selector inside the real
# run_smiles_pmo_task PMO loop. It uses one locked O3 schema menu for strict
# catalog training, uniform option selection, and Option-Flow deployment.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GFlowNet
using Random
using Serialization
using Statistics
using Dates
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "applications", "smiles_gflownet.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "rdkit_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "oracle_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "oracle_manager.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "pmo_benchmark.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "training", "option_flow_dataset.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "training", "option_flow_model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "training", "option_flow_loss.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "training", "option_flow_training.jl"))

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUTDIR = joinpath(ROOT, "checkpoints", "o3_integrated_option_flow_pmo")
const MENU_VERSION = "o3_schema_menu_v1"
const SELECTOR_METADATA_VERSION = "o3_selector_metadata_v1"
const AUTHORITATIVE_DATE_NOTE = "User supplied authoritative date/time: Friday, 2026-06-19 12:25 EDT"

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

const O3_TASK_VOCAB = ["qed", "drd2", "celecoxib_rediscovery", "unknown_holdout"]
const CONFIRMATORY_ARMS = Set(["tb_only", "heuristic_he_default", "uniform_option_he", "option_flow_sample_he", "option_flow_greedy_he"])
const SELECTOR_ARMS = Set(["option_flow_sample_he", "option_flow_greedy_he"])
const OPTION_MENU_ARMS = Set(["uniform_option_he", "option_flow_sample_he", "option_flow_greedy_he"])

struct O3OptionSpec
    name::String
    operator_override::Union{Nothing,Symbol}
    horizon::Int
    max_operator_candidates::Int
    min_exploration_per_operator::Int
    multi_child_min_reward_ratio::Float64
    allow_crossover::Bool
    operator_prior_strength::Float64
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

function load_pretrain()
    checkpoint_path = joinpath(ROOT, "checkpoints", "pretrain", "final.jls")
    isfile(checkpoint_path) || error("Pretrained checkpoint not found: $(checkpoint_path)")
    checkpoint = deserialize(checkpoint_path)
    pretrained_params = checkpoint["params"]
    pretrained_states = checkpoint["states"]
    vocab = SMILESVocabulary()
    vocab_size = haskey(checkpoint, "vocab_size") ? Int(checkpoint["vocab_size"]) : Int(size(pretrained_params.output.layer_2.weight, 1))
    policy_model, _, _ = create_smiles_policy(; vocab_size=vocab_size, hidden_dim=512, embed_dim=128, n_layers=3)
    return Dict{String,Any}(
        "checkpoint_path" => checkpoint_path,
        "pretrained_params" => pretrained_params,
        "pretrained_states" => pretrained_states,
        "vocab" => vocab,
        "vocab_size" => vocab_size,
        "policy_model" => policy_model,
    )
end

function make_base_he_config()
    return HierarchicalEditConfig(;
        horizon=8,
        topk_tracking=10,
        allow_crossover=true,
        min_exploration_per_operator=3,
        multi_child_min_reward_ratio=0.2,
        operator_prior_strength=4.0,
    )
end

function o3_option_specs()
    return O3OptionSpec[
        O3OptionSpec("mutate_h2", :mutate, 2, 8, 2, 0.2, true, 4.0),
        O3OptionSpec("crossover_h2", :crossover, 2, 8, 2, 0.2, true, 4.0),
        O3OptionSpec("mixed_h2", nothing, 2, 8, 2, 0.2, true, 4.0),
        O3OptionSpec("mixed_h4", nothing, 4, 8, 3, 0.2, true, 4.0),
        O3OptionSpec("heuristic_default_h8", nothing, 8, 8, 3, 0.2, true, 4.0),
    ]
end

function config_from_spec(spec::O3OptionSpec, base_config::HierarchicalEditConfig=make_base_he_config())
    return HierarchicalEditConfig(;
        horizon=spec.horizon,
        frontier_snapshot_size=base_config.frontier_snapshot_size,
        allow_crossover=spec.allow_crossover,
        allow_fragment_ops=base_config.allow_fragment_ops,
        max_operator_candidates=spec.max_operator_candidates,
        topk_tracking=base_config.topk_tracking,
        max_step_attempts=base_config.max_step_attempts,
        operators=base_config.operators,
        min_exploration_per_operator=spec.min_exploration_per_operator,
        multi_child_min_reward_ratio=spec.multi_child_min_reward_ratio,
        operator_prior_strength=spec.operator_prior_strength,
        use_operator_adaptation=base_config.use_operator_adaptation,
        operator_sampling_weights=base_config.operator_sampling_weights,
        basin_candidate_limit=base_config.basin_candidate_limit,
        use_learned_basin=base_config.use_learned_basin,
        learned_basin_controller=base_config.learned_basin_controller,
        parent_candidate_limit=base_config.parent_candidate_limit,
        use_learned_parent=base_config.use_learned_parent,
        learned_parent_controller=base_config.learned_parent_controller,
        use_learned_operator=base_config.use_learned_operator,
        learned_operator_controller=base_config.learned_operator_controller,
    )
end

function heuristic_equivalence_check(specs::Vector{O3OptionSpec})
    base = make_base_he_config()
    idx = findfirst(s -> s.name == "heuristic_default_h8", specs)
    idx === nothing && return Dict{String,Any}("ok" => false, "reason" => "missing heuristic_default_h8")
    spec = specs[idx]
    cfg = config_from_spec(spec, base)
    comparisons = Dict{String,Bool}(
        "horizon" => cfg.horizon == base.horizon == 8,
        "topk_tracking" => cfg.topk_tracking == base.topk_tracking == 10,
        "allow_crossover" => cfg.allow_crossover == base.allow_crossover == true,
        "min_exploration_per_operator" => cfg.min_exploration_per_operator == base.min_exploration_per_operator == 3,
        "multi_child_min_reward_ratio" => cfg.multi_child_min_reward_ratio == base.multi_child_min_reward_ratio == 0.2,
        "operator_prior_strength" => cfg.operator_prior_strength == base.operator_prior_strength == 4.0,
        "allow_fragment_ops" => cfg.allow_fragment_ops == base.allow_fragment_ops == false,
        "max_operator_candidates" => cfg.max_operator_candidates == base.max_operator_candidates,
    )
    return Dict{String,Any}(
        "ok" => all(values(comparisons)),
        "menu_version" => MENU_VERSION,
        "spec_name" => spec.name,
        "comparisons" => comparisons,
        "base" => Dict("horizon"=>base.horizon, "topk_tracking"=>base.topk_tracking, "allow_crossover"=>base.allow_crossover, "min_exploration_per_operator"=>base.min_exploration_per_operator, "multi_child_min_reward_ratio"=>base.multi_child_min_reward_ratio, "operator_prior_strength"=>base.operator_prior_strength, "max_operator_candidates"=>base.max_operator_candidates),
        "schema" => Dict("horizon"=>cfg.horizon, "topk_tracking"=>cfg.topk_tracking, "allow_crossover"=>cfg.allow_crossover, "min_exploration_per_operator"=>cfg.min_exploration_per_operator, "multi_child_min_reward_ratio"=>cfg.multi_child_min_reward_ratio, "operator_prior_strength"=>cfg.operator_prior_strength, "max_operator_candidates"=>cfg.max_operator_candidates),
    )
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
        if !isempty(uncached) && calls[] < budget
            remaining = max(0, budget - calls[])
            eval_list = uncached[1:min(length(uncached), remaining)]
            if !isempty(eval_list)
                scores = OracleBridge.evaluate_batch(eval_list, task_name)
                for (s, score) in zip(eval_list, scores)
                    cache[canonicalize_smiles_identity(s)] = Float64(score)
                end
                calls[] += length(eval_list)
            end
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

function o3_task_features(task_name::String, task_vocab::Vector{String})
    task_onehot = zeros(Float32, length(task_vocab))
    idx = findfirst(==(task_name), task_vocab)
    if idx === nothing
        holdout = findfirst(==("unknown_holdout"), task_vocab)
        holdout === nothing || (task_onehot[holdout] = 1.0f0)
    else
        task_onehot[idx] = 1.0f0
    end
    return task_onehot
end

function strict_state_features(frontier::MolecularFrontierBuffer,
                               task_name::String,
                               task_vocab::Vector{String},
                               budget_remaining::Int,
                               created_at_step::Int;
                               budget_scale::Int=1000)
    summary = frontier_quality_summary(frontier; topk=10)
    task_onehot = o3_task_features(task_name, task_vocab)
    return Float32.(vcat(task_onehot, Float32[
        Float32(summary["size"]) / 512.0f0,
        Float32(summary["top1"]),
        Float32(summary["top10_mean"]),
        Float32(get(summary, "n_scaffolds", 0)) / 128.0f0,
        Float32(get(summary, "graph_unique_count", summary["size"])) / 128.0f0,
        Float32(budget_remaining) / Float32(max(budget_scale, 1)),
        Float32(created_at_step) / 1000.0f0,
        Float32(summary["top1"] - summary["top10_mean"]),
    ]))
end

function strict_option_features(spec::O3OptionSpec, option_index::Int, n_specs::Int)
    op = spec.operator_override
    return Float32[
        op == :mutate ? 1.0f0 : 0.0f0,
        op == :crossover ? 1.0f0 : 0.0f0,
        isnothing(op) ? 1.0f0 : 0.0f0,
        Float32(spec.horizon) / 8.0f0,
        Float32(spec.max_operator_candidates) / 16.0f0,
        Float32(spec.min_exploration_per_operator) / 5.0f0,
        Float32(spec.multi_child_min_reward_ratio),
        spec.allow_crossover ? 1.0f0 : 0.0f0,
        Float32(spec.operator_prior_strength) / 4.0f0,
        Float32(option_index) / Float32(max(n_specs, 1)),
    ]
end

function run_option_from_snapshot(base_frontier::MolecularFrontierBuffer,
                                  task_name::String,
                                  target_smiles::Union{Nothing,String},
                                  reward_one,
                                  reward_batch,
                                  vocab,
                                  spec::O3OptionSpec;
                                  snapshot_index::Int,
                                  option_index::Int,
                                  base_seed::Int,
                                  budget_remaining::Int)
    Random.seed!(base_seed + 1000 * snapshot_index + 37 * option_index)
    local_frontier = clone_frontier_buffer(base_frontier)
    trajectory_buffer = EditTrajectoryBuffer(10000)
    diagnostics_buffer = HierarchicalEditDiagnosticsBuffer(10000)
    config = config_from_spec(spec, make_base_he_config())
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
        episode_id="o3-train-$(task_name)-snap$(snapshot_index)-$(spec.name)-$(option_index)",
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
                                           task_vocab::Vector{String},
                                           target_smiles::Union{Nothing,String},
                                           reward_one,
                                           reward_batch,
                                           vocab,
                                           specs::Vector{O3OptionSpec};
                                           snapshot_index::Int,
                                           base_seed::Int,
                                           budget_remaining::Int,
                                           budget_scale::Int=1000)
    snapshot = create_frontier_snapshot(frontier;
        max_entries=min(64, max(length(frontier), 1)),
        target_smiles=target_smiles,
        budget_remaining=budget_remaining,
        created_at_step=snapshot_index)
    state_features = strict_state_features(frontier, task_name, task_vocab, budget_remaining, snapshot_index; budget_scale=budget_scale)
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
            "evidence_level" => "o3_exact_menu_strict_same_snapshot",
            "menu_version" => MENU_VERSION,
            "task_name" => task_name,
            "snapshot_id" => string(snapshot.snapshot_id),
            "option_name" => spec.name,
            "operator_override" => isnothing(spec.operator_override) ? "mixed" : string(spec.operator_override),
            "horizon" => spec.horizon,
            "max_operator_candidates" => spec.max_operator_candidates,
            "min_exploration_per_operator" => spec.min_exploration_per_operator,
            "multi_child_min_reward_ratio" => spec.multi_child_min_reward_ratio,
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
    spec = option_results[best_idx]["spec"]::O3OptionSpec
    return spec.name
end

function make_scoring_catalog(frontier::MolecularFrontierBuffer,
                              task_name::String,
                              task_vocab::Vector{String},
                              specs::Vector{O3OptionSpec};
                              created_at_step::Int,
                              budget_remaining::Int,
                              budget_scale::Int=1000)
    state_features = strict_state_features(frontier, task_name, task_vocab, budget_remaining, created_at_step; budget_scale=budget_scale)
    option_features = [strict_option_features(spec, i, length(specs)) for (i, spec) in enumerate(specs)]
    utilities = ones(Float64, length(specs))
    option_ids = [spec.name for spec in specs]
    metas = [Dict{String,Any}("online_scoring_only" => true, "menu_version" => MENU_VERSION, "option_name" => spec.name) for spec in specs]
    snapshot_id = UInt64(hash((task_name, created_at_step, budget_remaining, MENU_VERSION)))
    return make_option_flow_catalog(task_name, snapshot_id, state_features, option_features, utilities;
        option_ids=option_ids,
        metadata=metas,
        epsilon=1.0e-6)
end

function train_catalog_bundle(tasks::Vector{String},
                              task_vocab::Vector{String},
                              specs::Vector{O3OptionSpec};
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
            catalog, option_results, _ = build_strict_catalog_for_snapshot(frontier, task, task_vocab, target,
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
        logmsg("O3 pretrain task=$(task) catalogs=$(length(task_catalogs)) calls=$(task_calls) added=$(added) advanced_by=$(advanced_by)")
    end
    return Dict{String,Any}(
        "menu_version" => MENU_VERSION,
        "task_vocab" => task_vocab,
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

function entropy_from_probs(probs::AbstractVector{<:Real})
    total = sum(Float64.(probs))
    total <= 0.0 && return 0.0
    normalized = Float64.(probs) ./ total
    return -sum(p > 0.0 ? p * log(p) : 0.0 for p in normalized)
end

function sample_index_from_probs(rng::AbstractRNG, probs::Vector{Float64})
    total = sum(probs)
    total <= 0.0 && return rand(rng, 1:length(probs))
    cumulative = cumsum(probs ./ total)
    return clamp(searchsortedfirst(cumulative, rand(rng)), 1, length(probs))
end

function choose_spec_for_selector(policy::String,
                                  frontier::MolecularFrontierBuffer,
                                  task_name::String,
                                  task_vocab::Vector{String},
                                  specs::Vector{O3OptionSpec},
                                  selector_model;
                                  created_at_step::Int,
                                  budget_remaining::Int,
                                  rng::AbstractRNG,
                                  budget_scale::Int=1000)
    if policy == "uniform"
        probs = fill(1.0 / length(specs), length(specs))
        logits = zeros(Float64, length(specs))
        idx = sample_index_from_probs(rng, probs)
        return idx, probs, logits
    elseif policy == "option_flow_sample" || policy == "option_flow_greedy"
        isnothing(selector_model) && error("selector model required for $(policy)")
        catalog = make_scoring_catalog(frontier, task_name, task_vocab, specs;
            created_at_step=created_at_step,
            budget_remaining=budget_remaining,
            budget_scale=budget_scale)
        logits = Float64.(option_flow_logits(selector_model["params"], catalog))
        probs = Float64.(option_flow_probs_from_logits(logits))
        idx = policy == "option_flow_greedy" ? argmax(logits) : sample_index_from_probs(rng, probs)
        return idx, probs, logits
    else
        error("unknown selector policy: $(policy)")
    end
end

function make_he_selector(policy::String,
                          specs::Vector{O3OptionSpec},
                          task_vocab::Vector{String},
                          selector_model;
                          seed::Int,
                          budget_scale::Int)
    rng = MersenneTwister(seed + bounded_hash_int((policy, MENU_VERSION)))
    return function(frontier_buffer, task_name, budget_remaining, created_at_step, phase, episode_index, base_he_config)
        idx, probs, logits = choose_spec_for_selector(policy, frontier_buffer, task_name, task_vocab, specs, selector_model;
            created_at_step=created_at_step,
            budget_remaining=budget_remaining,
            rng=rng,
            budget_scale=budget_scale)
        spec = specs[idx]
        selected_config = config_from_spec(spec, base_he_config)
        metadata = Dict{String,Any}(
            "schema_version" => SELECTOR_METADATA_VERSION,
            "menu_version" => MENU_VERSION,
            "policy" => policy,
            "selected_schema" => spec.name,
            "selected_index" => idx,
            "option_ids" => [s.name for s in specs],
            "probs" => probs,
            "logits" => logits,
            "entropy" => entropy_from_probs(probs),
            "phase" => phase,
            "episode_index" => episode_index,
            "created_at_step" => created_at_step,
            "budget_remaining" => budget_remaining,
            "task_name" => task_name,
            "selected_horizon" => spec.horizon,
            "operator_override" => isnothing(spec.operator_override) ? "mixed" : string(spec.operator_override),
        )
        return selected_config, spec.operator_override, metadata
    end
end

function pmo_result_summary(result)
    return Dict{String,Any}(
        "task_name" => result.task_name,
        "auc_top10" => Float64(result.auc_top10),
        "top1" => Float64(result.top1),
        "top10_mean" => Float64(result.top10_mean),
        "diversity" => Float64(result.diversity),
        "n_oracle_calls" => Int(result.n_oracle_calls),
        "unique_molecules" => Int(result.unique_molecules),
        "oracle_call_breakdown" => result.oracle_call_breakdown,
        "provenance_summary" => result.provenance_summary,
        "artifact_paths" => result.artifact_paths,
        "diagnostics_summary" => result.diagnostics_summary,
    )
end

function selector_metadata_summary(result_summary::Dict{String,Any})
    diagnostics = get(result_summary, "diagnostics_summary", Dict{String,Any}())
    episodes = get(diagnostics, "episode_summaries", Dict{String,Any}[])
    selected_counts = Dict{String,Int}()
    policy_counts = Dict{String,Int}()
    active_count = 0
    version_ok = true
    menu_ok = true
    for ep in episodes
        meta = get(ep, "he_selector_metadata", Dict{String,Any}())
        isempty(meta) && (version_ok = false; continue)
        active = Bool(get(meta, "selector_active", false))
        active && (active_count += 1)
        schema = String(get(meta, "selected_schema", "unknown"))
        policy = String(get(meta, "policy", "unknown"))
        selected_counts[schema] = get(selected_counts, schema, 0) + 1
        policy_counts[policy] = get(policy_counts, policy, 0) + 1
        version_ok &= String(get(meta, "schema_version", "")) == SELECTOR_METADATA_VERSION
        if active
            menu_ok &= String(get(meta, "menu_version", "")) == MENU_VERSION
        end
    end
    return Dict{String,Any}(
        "episode_count" => length(episodes),
        "selector_active_count" => active_count,
        "selected_schema_counts" => selected_counts,
        "policy_counts" => policy_counts,
        "metadata_version_ok" => version_ok,
        "menu_version_ok_for_active" => menu_ok,
    )
end

function arm_flags(arm::String)
    if arm == "tb_only"
        return Dict(:use_hierarchical_edit => false, :selector_policy => nothing)
    elseif arm == "heuristic_he_default"
        return Dict(:use_hierarchical_edit => true, :selector_policy => nothing)
    elseif arm == "uniform_option_he"
        return Dict(:use_hierarchical_edit => true, :selector_policy => "uniform")
    elseif arm == "option_flow_sample_he"
        return Dict(:use_hierarchical_edit => true, :selector_policy => "option_flow_sample")
    elseif arm == "option_flow_greedy_he"
        return Dict(:use_hierarchical_edit => true, :selector_policy => "option_flow_greedy")
    else
        error("unknown O3 arm: $(arm)")
    end
end

function run_one(task::String,
                 arm::String,
                 seed::Int,
                 pretrain::Dict{String,Any},
                 specs::Vector{O3OptionSpec},
                 task_vocab::Vector{String},
                 selector_model;
                 budget::Int,
                 n_iters::Int,
                 batch_size::Int,
                 replay_ratio::Int,
                 he_warmup_episodes::Int,
                 he_episodes_per_segment::Int,
                 he_budget_fraction::Float64,
                 verbose::Bool)
    flags = arm_flags(arm)
    he_enabled = Bool(flags[:use_hierarchical_edit])
    policy = flags[:selector_policy]
    target_smi = get(TARGET_SMILES, task, nothing)
    Random.seed!(seed + bounded_hash_int((task, arm, MENU_VERSION)))
    he_artifact_dir = he_enabled ? joinpath(OUTDIR, "artifacts", arm, task, "seed$(seed)") : nothing
    !isnothing(he_artifact_dir) && mkpath(he_artifact_dir)
    he_selector = nothing
    if he_enabled && !isnothing(policy)
        he_selector = make_he_selector(String(policy), specs, task_vocab, selector_model;
            seed=seed + bounded_hash_int((task, arm)),
            budget_scale=budget)
    end
    he_run_context = Dict{String,Any}(
        "task_name" => task,
        "config_name" => arm,
        "run_index" => seed,
        "o3_arm" => arm,
        "seed" => seed,
        "menu_version" => MENU_VERSION,
        "selector_policy" => isnothing(policy) ? "none" : String(policy),
    )

    start_time = time()
    try
        result = run_smiles_pmo_task(task;
            budget=budget,
            pretrained_params=deepcopy(pretrain["pretrained_params"]),
            pretrained_states=deepcopy(pretrain["pretrained_states"]),
            vocab=pretrain["vocab"],
            policy_model=pretrain["policy_model"],
            training_mode=:tb,
            use_qgfn=false,
            use_boosting=false,
            use_replay=true,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            n_iterations=n_iters,
            track_frontier=true,
            target_smiles=target_smi,
            target_seed=!isnothing(target_smi),
            target_seed_augmentations=4,
            use_hierarchical_edit=he_enabled,
            he_warmup_episodes=he_enabled ? he_warmup_episodes : 0,
            he_episodes_per_segment=he_enabled ? he_episodes_per_segment : 0,
            he_budget_fraction=he_budget_fraction,
            he_config=he_enabled ? make_base_he_config() : HierarchicalEditConfig(),
            he_artifact_dir=he_artifact_dir,
            he_run_context=he_run_context,
            he_episode_selector=he_selector,
            verbose=verbose,
        )
        elapsed = time() - start_time
        summary = pmo_result_summary(result)
        metadata_summary = selector_metadata_summary(summary)
        return Dict{String,Any}(
            "status" => "ok",
            "task" => task,
            "arm" => arm,
            "seed" => seed,
            "budget" => budget,
            "n_iterations" => n_iters,
            "batch_size" => batch_size,
            "replay_ratio" => replay_ratio,
            "he_warmup_episodes" => he_enabled ? he_warmup_episodes : 0,
            "he_episodes_per_segment" => he_enabled ? he_episodes_per_segment : 0,
            "he_budget_fraction" => he_enabled ? he_budget_fraction : 0.0,
            "flags" => Dict(String(k) => v for (k, v) in flags),
            "elapsed_sec" => elapsed,
            "result_summary" => summary,
            "selector_metadata_summary" => metadata_summary,
            "error" => nothing,
        )
    catch e
        elapsed = time() - start_time
        bt = catch_backtrace()
        return Dict{String,Any}(
            "status" => "failed",
            "task" => task,
            "arm" => arm,
            "seed" => seed,
            "budget" => budget,
            "n_iterations" => n_iters,
            "batch_size" => batch_size,
            "replay_ratio" => replay_ratio,
            "he_warmup_episodes" => he_enabled ? he_warmup_episodes : 0,
            "he_episodes_per_segment" => he_enabled ? he_episodes_per_segment : 0,
            "he_budget_fraction" => he_enabled ? he_budget_fraction : 0.0,
            "flags" => Dict(String(k) => v for (k, v) in flags),
            "elapsed_sec" => elapsed,
            "result_summary" => nothing,
            "selector_metadata_summary" => Dict{String,Any}(),
            "error" => sprint(showerror, e, bt),
        )
    end
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
    for row in rows
        row["status"] == "ok" || continue
        key = (String(row["task"]), String(row["arm"]))
        groups[key] = get(groups, key, Dict{String,Any}[])
        push!(groups[key], row)
    end
    out = Dict{String,Any}[]
    for ((task, arm), vals) in groups
        aucs = [Float64(v["result_summary"]["auc_top10"]) for v in vals]
        top1 = [Float64(v["result_summary"]["top1"]) for v in vals]
        top10 = [Float64(v["result_summary"]["top10_mean"]) for v in vals]
        calls = [Float64(v["result_summary"]["n_oracle_calls"]) for v in vals]
        uniques = [Float64(v["result_summary"]["unique_molecules"]) for v in vals]
        he_episodes = [Float64(get(v["selector_metadata_summary"], "episode_count", 0)) for v in vals]
        push!(out, Dict{String,Any}(
            "task" => task,
            "arm" => arm,
            "n_ok" => length(vals),
            "auc_mean" => mean_or_nan(aucs),
            "auc_std" => std_or_zero(aucs),
            "top1_mean" => mean_or_nan(top1),
            "top10_mean" => mean_or_nan(top10),
            "calls_mean" => mean_or_nan(calls),
            "unique_mean" => mean_or_nan(uniques),
            "he_episode_mean" => mean_or_nan(he_episodes),
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
        aucs = [Float64(r["auc_mean"]) for r in rows if isfinite(Float64(r["auc_mean"]))]
        top10s = [Float64(r["top10_mean"]) for r in rows if isfinite(Float64(r["top10_mean"]))]
        push!(out, Dict{String,Any}(
            "arm" => arm,
            "tasks" => [String(r["task"]) for r in rows],
            "mean_auc" => mean_or_nan(aucs),
            "mean_top10" => mean_or_nan(top10s),
        ))
    end
    sort!(out, by = r -> -Float64(r["mean_auc"]))
    return out
end

function paired_deltas(rows::Vector{Dict{String,Any}}, baseline_arm::String)
    ok_rows = [r for r in rows if r["status"] == "ok"]
    index = Dict{Tuple{String,Int,String},Dict{String,Any}}()
    for r in ok_rows
        index[(String(r["task"]), Int(r["seed"]), String(r["arm"]))] = r
    end
    deltas = Dict{String,Any}[]
    for r in ok_rows
        arm = String(r["arm"])
        arm == baseline_arm && continue
        key = (String(r["task"]), Int(r["seed"]), baseline_arm)
        haskey(index, key) || continue
        base = index[key]
        auc = Float64(r["result_summary"]["auc_top10"])
        base_auc = Float64(base["result_summary"]["auc_top10"])
        top10 = Float64(r["result_summary"]["top10_mean"])
        base_top10 = Float64(base["result_summary"]["top10_mean"])
        push!(deltas, Dict{String,Any}(
            "task" => String(r["task"]),
            "seed" => Int(r["seed"]),
            "arm" => arm,
            "baseline_arm" => baseline_arm,
            "auc" => auc,
            "baseline_auc" => base_auc,
            "delta_auc" => auc - base_auc,
            "relative_delta_pct" => base_auc == 0.0 ? NaN : (auc / base_auc - 1.0) * 100.0,
            "top10" => top10,
            "baseline_top10" => base_top10,
            "relative_top10_delta_pct" => base_top10 == 0.0 ? NaN : (top10 / base_top10 - 1.0) * 100.0,
        ))
    end
    return deltas
end

function selection_distribution(rows::Vector{Dict{String,Any}})
    counts = Dict{String,Int}()
    per_arm = Dict{String,Dict{String,Int}}()
    for row in rows
        row["status"] == "ok" || continue
        arm = String(row["arm"])
        meta = get(row, "selector_metadata_summary", Dict{String,Any}())
        selected = get(meta, "selected_schema_counts", Dict{String,Int}())
        arm_counts = get(per_arm, arm, Dict{String,Int}())
        for (schema, c) in selected
            s = String(schema)
            n = Int(c)
            counts[s] = get(counts, s, 0) + n
            arm_counts[s] = get(arm_counts, s, 0) + n
        end
        per_arm[arm] = arm_counts
    end
    return Dict{String,Any}("overall" => counts, "by_arm" => per_arm)
end

function o3_gate(rows::Vector{Dict{String,Any}}, agg::Vector{Dict{String,Any}})
    overall = Dict(String(r["arm"]) => r for r in overall_by_arm(agg))
    failures = [r for r in rows if r["status"] != "ok"]
    sample_auc = haskey(overall, "option_flow_sample_he") ? Float64(overall["option_flow_sample_he"]["mean_auc"]) : NaN
    sample_top10 = haskey(overall, "option_flow_sample_he") ? Float64(overall["option_flow_sample_he"]["mean_top10"]) : NaN
    heuristic_auc = haskey(overall, "heuristic_he_default") ? Float64(overall["heuristic_he_default"]["mean_auc"]) : NaN
    heuristic_top10 = haskey(overall, "heuristic_he_default") ? Float64(overall["heuristic_he_default"]["mean_top10"]) : NaN
    uniform_auc = haskey(overall, "uniform_option_he") ? Float64(overall["uniform_option_he"]["mean_auc"]) : NaN

    delta_vs_he_rel = isfinite(sample_auc) && isfinite(heuristic_auc) && heuristic_auc != 0.0 ? (sample_auc / heuristic_auc - 1.0) : NaN
    top10_vs_he_rel = isfinite(sample_top10) && isfinite(heuristic_top10) && heuristic_top10 != 0.0 ? (sample_top10 / heuristic_top10 - 1.0) : NaN

    paired_he = paired_deltas(rows, "heuristic_he_default")
    sample_he_pairs = [d for d in paired_he if String(d["arm"]) == "option_flow_sample_he"]
    paired_wins = count(d -> Float64(d["delta_auc"]) > 0.0, sample_he_pairs)
    task_win_counts = Dict{String,Bool}()
    for d in sample_he_pairs
        t = String(d["task"])
        task_win_counts[t] = get(task_win_counts, t, false) || Float64(d["delta_auc"]) > 0.0
    end
    task_wins = count(identity, values(task_win_counts))
    n_tasks = length(keys(task_win_counts))
    sample_schema_counts = Dict{String,Int}()
    for row in rows
        row["status"] == "ok" || continue
        String(row["arm"]) == "option_flow_sample_he" || continue
        meta = get(row, "selector_metadata_summary", Dict{String,Any}())
        selected = get(meta, "selected_schema_counts", Dict{String,Int}())
        for (schema, c) in selected
            s = String(schema)
            sample_schema_counts[s] = get(sample_schema_counts, s, 0) + Int(c)
        end
    end
    sample_schema_total = sum(values(sample_schema_counts); init=0)
    sample_schema_unique = count(>(0), values(sample_schema_counts))
    sample_schema_max_fraction = sample_schema_total == 0 ? 1.0 : maximum(collect(values(sample_schema_counts))) / sample_schema_total
    no_total_option_collapse = sample_schema_unique >= 2 && sample_schema_max_fraction < 0.95

    structural_raw = false
    for d in sample_he_pairs
        if String(d["task"]) == "celecoxib_rediscovery"
            structural_raw |= Float64(d["relative_delta_pct"]) >= 5.0 && Float64(d["relative_top10_delta_pct"]) >= -2.0
        end
    end
    structural = structural_raw && no_total_option_collapse

    learned_not_better_than_uniform = isfinite(sample_auc) && isfinite(uniform_auc) && sample_auc <= uniform_auc
    eligible_o3b_raw = isfinite(delta_vs_he_rel) && delta_vs_he_rel >= 0.02 &&
                       (task_wins >= ceil(Int, max(n_tasks, 1) * 2 / 3) || paired_wins >= ceil(Int, max(length(sample_he_pairs), 1) / 2)) &&
                       isfinite(top10_vs_he_rel) && top10_vs_he_rel >= -0.02
    eligible_o3b = eligible_o3b_raw && !learned_not_better_than_uniform
    downgrade = false
    reasons = String[]
    if !isempty(failures)
        push!(reasons, "Some O3 runs failed; verdict incomplete until failures are explained.")
    end
    if isfinite(sample_auc) && isfinite(heuristic_auc) && sample_auc <= heuristic_auc && !structural
        downgrade = true
        push!(reasons, "option_flow_sample_he did not beat heuristic_he_default on mean AUC and no structural-specialist win was established.")
    end
    if learned_not_better_than_uniform
        downgrade = true
        push!(reasons, "option_flow_sample_he did not beat uniform_option_he; learned selector adds no value over random menu sampling.")
    end
    if eligible_o3b
        push!(reasons, "option_flow_sample_he passed O3a warm-start threshold; run O3b fair/transfer before any mainline restoration.")
    elseif structural
        push!(reasons, "Overall restoration not established, but structural-specialist criterion is provisionally positive on celecoxib_rediscovery.")
    elseif isempty(reasons)
        push!(reasons, "No decisive branch triggered; inspect paired deltas and schema distributions.")
    end

    verdict = if !isempty(failures)
        "O3_INCOMPLETE_WITH_FAILURES"
    elseif learned_not_better_than_uniform && structural
        "O3_MAINLINE_DOWNGRADE_WITH_STRUCTURAL_AUXILIARY_SIGNAL"
    elseif learned_not_better_than_uniform || downgrade
        "O3_MAINLINE_DOWNGRADE"
    elseif eligible_o3b
        "O3A_POSITIVE_REQUIRES_O3B"
    elseif structural
        "O3_STRUCTURAL_SPECIALIST_SIGNAL"
    else
        "O3_INCONCLUSIVE"
    end

    return Dict{String,Any}(
        "verdict" => verdict,
        "eligible_for_o3b" => eligible_o3b,
        "eligible_for_o3b_raw_before_uniform_gate" => eligible_o3b_raw,
        "learned_not_better_than_uniform" => learned_not_better_than_uniform,
        "structural_specialist" => structural,
        "structural_specialist_raw_before_collapse_check" => structural_raw,
        "no_total_option_collapse" => no_total_option_collapse,
        "sample_schema_counts" => sample_schema_counts,
        "sample_schema_max_fraction" => sample_schema_max_fraction,
        "downgrade" => downgrade,
        "sample_mean_auc" => sample_auc,
        "heuristic_mean_auc" => heuristic_auc,
        "uniform_mean_auc" => uniform_auc,
        "sample_vs_heuristic_relative" => delta_vs_he_rel,
        "sample_vs_heuristic_top10_relative" => top10_vs_he_rel,
        "paired_sample_vs_heuristic_count" => length(sample_he_pairs),
        "paired_sample_wins_heuristic" => paired_wins,
        "task_wins_heuristic" => task_wins,
        "task_win_counts" => task_win_counts,
        "failed_runs" => length(failures),
        "reasons" => reasons,
    )
end

function print_summary(agg, overall, gate)
    println("\n", "="^96)
    println("O3 INTEGRATED OPTION-FLOW PMO SUMMARY")
    println("="^96)
    println(rpad("Task", 26), rpad("Arm", 24), rpad("n", 5), rpad("AUC mean", 12), rpad("AUC std", 12), rpad("Top1", 10), rpad("Top10", 10), rpad("Calls", 10), rpad("HE eps", 8))
    println("-"^96)
    for row in agg
        @printf("%-26s%-24s%-5d%-12.6f%-12.6f%-10.4f%-10.4f%-10.1f%-8.1f\n",
            row["task"], row["arm"], row["n_ok"], row["auc_mean"], row["auc_std"], row["top1_mean"], row["top10_mean"], row["calls_mean"], row["he_episode_mean"])
    end
    println("\nOVERALL BY ARM")
    for row in overall
        @printf("%-24s mean_auc=%.6f mean_top10=%.6f tasks=%s\n", row["arm"], row["mean_auc"], row["mean_top10"], join(row["tasks"], ","))
    end
    println("\nO3 GATE")
    println("verdict=", gate["verdict"])
    for r in gate["reasons"]
        println("- ", r)
    end
end

function existing_keys(rows::Vector{Dict{String,Any}})
    return Set((String(r["task"]), String(r["arm"]), Int(r["seed"])) for r in rows)
end

function save_bundle(path::String, bundle::Dict{String,Any})
    tmp = path * ".tmp"
    serialize(tmp, bundle)
    mv(tmp, path; force=true)
    return path
end

function metadata_roundtrip_check(path::String)
    isfile(path) || return Dict{String,Any}("ok" => false, "reason" => "missing bundle")
    bundle = deserialize(path)
    rows = get(bundle, "rows", Dict{String,Any}[])
    active_rows = [r for r in rows if r["status"] == "ok" && String(r["arm"]) in OPTION_MENU_ARMS]
    if isempty(active_rows)
        return Dict{String,Any}("ok" => false, "reason" => "no active selector/menu rows")
    end
    ok_rows = 0
    for row in active_rows
        meta = get(row, "selector_metadata_summary", Dict{String,Any}())
        if Int(get(meta, "episode_count", 0)) > 0 && Bool(get(meta, "metadata_version_ok", false)) && Bool(get(meta, "menu_version_ok_for_active", false))
            ok_rows += 1
        end
    end
    return Dict{String,Any}(
        "ok" => ok_rows == length(active_rows),
        "checked_rows" => length(active_rows),
        "ok_rows" => ok_rows,
    )
end

function main()
    mode = strip(get(ENV, "O3_MODE", "smoke"))
    default_tasks = mode == "smoke" ? ["qed"] : ["qed", "drd2", "celecoxib_rediscovery"]
    default_arms = mode == "smoke" ? ["heuristic_he_default", "uniform_option_he", "option_flow_sample_he"] : ["tb_only", "heuristic_he_default", "uniform_option_he", "option_flow_sample_he", "option_flow_greedy_he"]
    tasks = parse_csv_strings("O3_TASKS", default_tasks)
    arms = parse_csv_strings("O3_ARMS", default_arms)
    seeds = parse_csv_ints("O3_SEEDS", mode == "smoke" ? [17] : [17, 23])
    budget = parse_env_int("O3_BUDGET", mode == "smoke" ? 192 : 300)
    n_iters = parse_env_int("O3_ITERS", mode == "smoke" ? 3 : 3)
    batch_size = parse_env_int("O3_BATCH", mode == "smoke" ? 8 : 8)
    replay_ratio = parse_env_int("O3_REPLAY_RATIO", 2)
    he_warmup_episodes = parse_env_int("O3_HE_WARMUP_EPISODES", 2)
    he_episodes_per_segment = parse_env_int("O3_HE_EPISODES_PER_SEGMENT", 1)
    he_budget_fraction = parse_env_float("O3_HE_BUDGET_FRACTION", 0.15)
    train_budget = parse_env_int("O3_TRAIN_BUDGET", mode == "smoke" ? 240 : max(300, budget))
    train_snapshots = parse_env_int("O3_TRAIN_SNAPSHOTS", mode == "smoke" ? 2 : 3)
    epochs = parse_env_int("O3_EPOCHS", mode == "smoke" ? 70 : 140)
    lr = parse_env_float("O3_LR", 0.012)
    verbose = parse_env_bool("O3_VERBOSE", true)
    resume = parse_env_bool("O3_RESUME", true)
    rerun_failed = parse_env_bool("O3_RERUN_FAILED", false)

    invalid = [a for a in arms if !(a in CONFIRMATORY_ARMS)]
    isempty(invalid) || error("Unknown O3 arms: $(invalid)")

    specs = o3_option_specs()
    eq_check = heuristic_equivalence_check(specs)
    Bool(eq_check["ok"]) || error("heuristic_default_h8 equivalence check failed: $(eq_check)")
    task_vocab = copy(O3_TASK_VOCAB)

    logmsg("O3 integrated PMO mode=$(mode) tasks=$(tasks) arms=$(arms) seeds=$(seeds) budget=$(budget) iters=$(n_iters) batch=$(batch_size) menu=$(MENU_VERSION)")
    logmsg("heuristic_default_h8 equivalence ok")
    pretrain = load_pretrain()
    logmsg("Loaded checkpoint $(pretrain["checkpoint_path"]) vocab_size=$(pretrain["vocab_size"])")

    partial_path = joinpath(OUTDIR, "o3_$(mode)_partial_results.jls")
    final_path = joinpath(OUTDIR, "o3_$(mode)_results.jls")
    latest_path = joinpath(OUTDIR, "o3_latest_results.jls")
    selector_artifact_path = joinpath(OUTDIR, "o3_$(mode)_selector_bundle.jls")

    needs_selector = any(a -> a in SELECTOR_ARMS, arms)
    needs_menu_training = any(a -> a in OPTION_MENU_ARMS, arms)
    train_bundle = Dict{String,Any}(
        "menu_version" => MENU_VERSION,
        "task_vocab" => task_vocab,
        "catalogs" => OptionFlowCatalog[],
        "per_task" => Dict{String,Any}(),
        "total_calls" => 0,
        "stats" => option_flow_catalog_stats(OptionFlowCatalog[]),
    )
    selector_model = nothing
    if needs_menu_training
        if resume && isfile(selector_artifact_path)
            selector_artifact = deserialize(selector_artifact_path)
            train_bundle = selector_artifact["train_bundle"]
            selector_model = get(selector_artifact, "selector_model", nothing)
            logmsg("Loaded selector artifact $(selector_artifact_path)")
        else
            train_bundle = train_catalog_bundle(tasks, task_vocab, specs;
                budget_per_task=train_budget,
                snapshots=train_snapshots,
                base_seed=404,
                vocab=pretrain["vocab"],
                budget_scale=max(budget, train_budget))
            if needs_selector
                selector_model = train_selector(train_bundle["catalogs"]; seed=17, epochs=epochs, lr=lr)
            end
            save_bundle(selector_artifact_path, Dict{String,Any}(
                "created_at" => string(now()),
                "authoritative_date_note" => AUTHORITATIVE_DATE_NOTE,
                "mode" => mode,
                "menu_version" => MENU_VERSION,
                "task_vocab" => task_vocab,
                "tasks" => tasks,
                "train_budget" => train_budget,
                "train_snapshots" => train_snapshots,
                "epochs" => epochs,
                "learning_rate" => lr,
                "train_bundle" => train_bundle,
                "selector_model" => selector_model,
            ))
            logmsg("Saved selector artifact $(selector_artifact_path)")
        end
    end
    rows = Dict{String,Any}[]
    if resume && isfile(partial_path)
        prior = deserialize(partial_path)
        rows = get(prior, "rows", Dict{String,Any}[])
        logmsg("Resume loaded $(length(rows)) partial rows from $(partial_path)")
    end

    for seed in seeds, task in tasks, arm in arms
        keyset = existing_keys(rerun_failed ? [r for r in rows if r["status"] == "ok"] : rows)
        key = (task, arm, seed)
        if key in keyset
            logmsg("SKIP existing task=$(task) arm=$(arm) seed=$(seed)")
            continue
        end
        logmsg("RUN start task=$(task) arm=$(arm) seed=$(seed)")
        row = run_one(task, arm, seed, pretrain, specs, task_vocab, selector_model;
            budget=budget,
            n_iters=n_iters,
            batch_size=batch_size,
            replay_ratio=replay_ratio,
            he_warmup_episodes=he_warmup_episodes,
            he_episodes_per_segment=he_episodes_per_segment,
            he_budget_fraction=he_budget_fraction,
            verbose=verbose)
        push!(rows, row)
        partial_agg = aggregate_rows(rows)
        partial_overall = overall_by_arm(partial_agg)
        partial_gate = o3_gate(rows, partial_agg)
        partial_bundle = Dict{String,Any}(
            "created_at" => string(now()),
            "authoritative_date_note" => AUTHORITATIVE_DATE_NOTE,
            "mode" => mode,
            "partial" => true,
            "tasks" => tasks,
            "arms" => arms,
            "seeds" => seeds,
            "budget" => budget,
            "n_iterations" => n_iters,
            "batch_size" => batch_size,
            "replay_ratio" => replay_ratio,
            "he_warmup_episodes" => he_warmup_episodes,
            "he_episodes_per_segment" => he_episodes_per_segment,
            "he_budget_fraction" => he_budget_fraction,
            "menu_version" => MENU_VERSION,
            "selector_metadata_version" => SELECTOR_METADATA_VERSION,
            "task_vocab" => task_vocab,
            "schema_menu" => [Dict("name"=>s.name, "operator_override"=>isnothing(s.operator_override) ? "mixed" : string(s.operator_override), "horizon"=>s.horizon, "max_operator_candidates"=>s.max_operator_candidates, "min_exploration_per_operator"=>s.min_exploration_per_operator, "multi_child_min_reward_ratio"=>s.multi_child_min_reward_ratio, "allow_crossover"=>s.allow_crossover, "operator_prior_strength"=>s.operator_prior_strength) for s in specs],
            "heuristic_equivalence_check" => eq_check,
            "selector_training" => Dict(
                "training_budget_counted_in_headline" => false,
                "budget_per_task" => train_budget,
                "snapshots" => train_snapshots,
                "epochs" => epochs,
                "learning_rate" => lr,
                "bundle_summary" => Dict(k => v for (k, v) in train_bundle if k != "catalogs"),
                "train_metrics" => isnothing(selector_model) ? Dict{String,Any}() : selector_model["train_metrics"],
                "val_metrics" => isnothing(selector_model) ? Dict{String,Any}() : selector_model["val_metrics"],
            ),
            "rows" => rows,
            "aggregate_rows" => partial_agg,
            "overall_by_arm" => partial_overall,
            "paired_delta_vs_heuristic_he_default" => paired_deltas(rows, "heuristic_he_default"),
            "paired_delta_vs_uniform_option_he" => paired_deltas(rows, "uniform_option_he"),
            "paired_delta_vs_tb_only" => paired_deltas(rows, "tb_only"),
            "selection_distribution" => selection_distribution(rows),
            "gate" => partial_gate,
        )
        save_bundle(partial_path, partial_bundle)
        if row["status"] == "ok"
            s = row["result_summary"]
            m = row["selector_metadata_summary"]
            logmsg("RUN ok task=$(task) arm=$(arm) seed=$(seed) auc=$(round(s["auc_top10"], digits=6)) top1=$(round(s["top1"], digits=4)) top10=$(round(s["top10_mean"], digits=4)) calls=$(s["n_oracle_calls"]) unique=$(s["unique_molecules"]) he_eps=$(get(m, "episode_count", 0)) elapsed=$(round(row["elapsed_sec"], digits=1))")
        else
            logmsg("RUN failed task=$(task) arm=$(arm) seed=$(seed) elapsed=$(round(row["elapsed_sec"], digits=1))")
            println(row["error"])
            flush(stdout)
        end
    end

    agg = aggregate_rows(rows)
    overall = overall_by_arm(agg)
    gate = o3_gate(rows, agg)
    bundle = Dict{String,Any}(
        "created_at" => string(now()),
        "authoritative_date_note" => AUTHORITATIVE_DATE_NOTE,
        "mode" => mode,
        "partial" => false,
        "tasks" => tasks,
        "arms" => arms,
        "seeds" => seeds,
        "budget" => budget,
        "n_iterations" => n_iters,
        "batch_size" => batch_size,
        "replay_ratio" => replay_ratio,
        "he_warmup_episodes" => he_warmup_episodes,
        "he_episodes_per_segment" => he_episodes_per_segment,
        "he_budget_fraction" => he_budget_fraction,
        "menu_version" => MENU_VERSION,
        "selector_metadata_version" => SELECTOR_METADATA_VERSION,
        "task_vocab" => task_vocab,
        "schema_menu" => [Dict("name"=>s.name, "operator_override"=>isnothing(s.operator_override) ? "mixed" : string(s.operator_override), "horizon"=>s.horizon, "max_operator_candidates"=>s.max_operator_candidates, "min_exploration_per_operator"=>s.min_exploration_per_operator, "multi_child_min_reward_ratio"=>s.multi_child_min_reward_ratio, "allow_crossover"=>s.allow_crossover, "operator_prior_strength"=>s.operator_prior_strength) for s in specs],
        "heuristic_equivalence_check" => eq_check,
        "selector_training" => Dict(
            "training_budget_counted_in_headline" => false,
            "budget_per_task" => train_budget,
            "snapshots" => train_snapshots,
            "epochs" => epochs,
            "learning_rate" => lr,
            "bundle_summary" => Dict(k => v for (k, v) in train_bundle if k != "catalogs"),
            "train_metrics" => isnothing(selector_model) ? Dict{String,Any}() : selector_model["train_metrics"],
            "val_metrics" => isnothing(selector_model) ? Dict{String,Any}() : selector_model["val_metrics"],
        ),
        "rows" => rows,
        "aggregate_rows" => agg,
        "overall_by_arm" => overall,
        "paired_delta_vs_heuristic_he_default" => paired_deltas(rows, "heuristic_he_default"),
        "paired_delta_vs_uniform_option_he" => paired_deltas(rows, "uniform_option_he"),
        "paired_delta_vs_tb_only" => paired_deltas(rows, "tb_only"),
        "selection_distribution" => selection_distribution(rows),
        "gate" => gate,
        "limitations" => [
            "O3a warm-start selector training oracle calls are reported separately and are not counted in the online PMO budget.",
            "O3a can downgrade the direction if it loses, but cannot restore mainline without O3b fair/transfer follow-up.",
            "This is a 3-task PMO-lite test, not a 23-task/10K PMO benchmark or SOTA claim.",
        ],
    )
    save_bundle(final_path, bundle)
    save_bundle(latest_path, bundle)
    roundtrip = metadata_roundtrip_check(final_path)
    bundle["metadata_roundtrip_check"] = roundtrip
    save_bundle(final_path, bundle)
    save_bundle(latest_path, bundle)

    print_summary(agg, overall, gate)
    logmsg("metadata_roundtrip_check=$(roundtrip)")
    logmsg("Saved results: $(abspath(final_path))")
    logmsg("Saved latest: $(abspath(latest_path))")

    if mode == "smoke"
        failed = [r for r in rows if r["arm"] in arms && r["status"] != "ok"]
        isempty(failed) || error("O3 smoke failed runs: $([(r["task"], r["arm"], r["seed"]) for r in failed])")
        Bool(roundtrip["ok"]) || error("O3 smoke metadata roundtrip failed: $(roundtrip)")
    end
end

main()
