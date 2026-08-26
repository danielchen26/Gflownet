#!/usr/bin/env julia
# Benchmark Truth Sprint — matched-search comparison for current CAFE-GFN
#
# Purpose:
# 1. Measure the strongest realistic performance of the CURRENT architecture
# 2. Compare matched PMO configs under the SAME search stack
# 3. Support Stage B′ heuristic HE integration audits with explicit provenance/accounting
#
# Defaults remain configurable via env vars:
#   PMO_BUDGET=10000
#   PMO_RUNS=1
#   PMO_TASKS=qed,drd2,gsk3b,jnk3,albuterol_similarity,celecoxib_rediscovery
#   PMO_CONFIGS=tb,rwmle,tb_seeded,rwmle_seeded
#   TRUTH_SPRINT_LOGDIR=checkpoints/truth_sprint
#
# Stage B′ recommended execution (explicit envs, not implicit defaults):
#   PMO_BUDGET=3000
#   PMO_RUNS=3
#   PMO_CONFIGS=tb,tb_he_full_locked
#   HE_BUDGET_FRACTION=0.15 (or 0.30 / 0.40 during fraction lock)

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GFlowNet
using Serialization
using Statistics: mean, std

include(joinpath(@__DIR__, "..", "..", "src", "applications", "smiles_gflownet.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "rdkit_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "oracle_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "oracle_manager.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "pmo_benchmark.jl"))

function parse_list_env(name::String, default::Vector{String})
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    return [String(strip(x)) for x in split(raw, ',') if !isempty(strip(x))]
end

function parse_bool_env(name::String, default::Bool)
    raw = lowercase(strip(get(ENV, name, default ? "1" : "0")))
    return raw in ("1", "true", "yes", "y", "on")
end

const DEFAULT_TASKS = [
    "qed",
    "drd2",
    "gsk3b",
    "jnk3",
    "albuterol_similarity",
    "celecoxib_rediscovery",
]

const STRUCTURAL_TASKS = Set(["albuterol_similarity", "celecoxib_rediscovery"])
const PROPERTY_TASKS = Set(["qed", "drd2", "gsk3b", "jnk3"])

const TARGET_SMILES = Dict(
    "albuterol_similarity" => "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "mestranol_similarity" => "C#CC1(O)CCC2C3CCc4cc(OC)ccc4C3CCC21C",
    "celecoxib_rediscovery" => "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "thiothixene_rediscovery" => "O=C(C1CCN(CCO)CC1)c1ccc(C(F)(F)F)c(I)c1",
)

const GENETIC_GFN_REF = Dict(
    "qed" => 0.948,
    "drd2" => 0.974,
    "gsk3b" => 0.960,
    "jnk3" => 0.780,
    "albuterol_similarity" => 0.949,
    "celecoxib_rediscovery" => 0.837,
)

const HE_BUDGET_FRACTION = parse(Float64, get(ENV, "HE_BUDGET_FRACTION", "0.15"))
const HE_WARMUP_EPISODES = parse(Int, get(ENV, "HE_WARMUP_EPISODES", "8"))
const HE_EPISODES_PER_SEGMENT = parse(Int, get(ENV, "HE_EPISODES_PER_SEGMENT", "4"))
const HE_HORIZON = parse(Int, get(ENV, "HE_HORIZON", "3"))
const HE_TOPK_TRACKING = parse(Int, get(ENV, "HE_TOPK_TRACKING", "10"))
const HE_MAX_OPERATOR_CANDIDATES = parse(Int, get(ENV, "HE_MAX_OPERATOR_CANDIDATES", "8"))
const HE_MAX_STEP_ATTEMPTS = parse(Int, get(ENV, "HE_MAX_STEP_ATTEMPTS", "3"))
const HE_MIN_EXPLORATION_PER_OPERATOR = parse(Int, get(ENV, "HE_MIN_EXPLORATION_PER_OPERATOR", "5"))
const HE_MULTI_CHILD_MIN_REWARD_RATIO = parse(Float64, get(ENV, "HE_MULTI_CHILD_MIN_REWARD_RATIO", "0.2"))
const HE_OPERATOR_PRIOR_STRENGTH = parse(Float64, get(ENV, "HE_OPERATOR_PRIOR_STRENGTH", "4.0"))
const HE_ALLOW_CROSSOVER = parse_bool_env("HE_ALLOW_CROSSOVER", true)

const PMO_BATCH_SIZE = parse(Int, get(ENV, "PMO_BATCH_SIZE", "64"))
const PMO_REPLAY_RATIO = parse(Int, get(ENV, "PMO_REPLAY_RATIO", "8"))
const PMO_N_ITERATIONS = parse(Int, get(ENV, "PMO_N_ITERATIONS", "25"))
const PMO_GA_PER_STEP = parse_bool_env("PMO_GA_PER_STEP", true)
const PMO_GA_CROSSOVER = parse(Int, get(ENV, "PMO_GA_CROSSOVER", "4"))
const PMO_GA_MUTATION = parse(Int, get(ENV, "PMO_GA_MUTATION", "4"))
const FRONTIER_BOOTSTRAP_SAMPLES = parse(Int, get(ENV, "FRONTIER_BOOTSTRAP_SAMPLES", "0"))
const FRONTIER_BOOTSTRAP_MIN_ENTRIES = parse(Int, get(ENV, "FRONTIER_BOOTSTRAP_MIN_ENTRIES", "2"))

function build_he_config()::HierarchicalEditConfig
    return HierarchicalEditConfig(;
        horizon=HE_HORIZON,
        topk_tracking=HE_TOPK_TRACKING,
        allow_crossover=HE_ALLOW_CROSSOVER,
        allow_fragment_ops=false,
        max_operator_candidates=HE_MAX_OPERATOR_CANDIDATES,
        max_step_attempts=HE_MAX_STEP_ATTEMPTS,
        min_exploration_per_operator=HE_MIN_EXPLORATION_PER_OPERATOR,
        multi_child_min_reward_ratio=HE_MULTI_CHILD_MIN_REWARD_RATIO,
        operator_prior_strength=HE_OPERATOR_PRIOR_STRENGTH,
        use_learned_basin=false,
        use_learned_parent=false,
        use_learned_operator=false,
    )
end

const CONFIG_LIBRARY = Dict(
    "tb" => Dict(
        "label" => "TB matched",
        "training_mode" => :tb,
        "target_seed" => false,
        "he_mode" => :off,
    ),
    "rwmle" => Dict(
        "label" => "RWMLE matched",
        "training_mode" => :rwmle,
        "target_seed" => false,
        "he_mode" => :off,
    ),
    "tb_seeded" => Dict(
        "label" => "TB matched + target seed",
        "training_mode" => :tb,
        "target_seed" => true,
        "he_mode" => :off,
    ),
    "rwmle_seeded" => Dict(
        "label" => "RWMLE matched + target seed",
        "training_mode" => :rwmle,
        "target_seed" => true,
        "he_mode" => :off,
    ),
    "tb_he_full" => Dict(
        "label" => "TB + heuristic HE (full)",
        "training_mode" => :tb,
        "target_seed" => false,
        "he_mode" => :full,
    ),
    "tb_he_full_locked" => Dict(
        "label" => "TB + heuristic HE (full locked)",
        "training_mode" => :tb,
        "target_seed" => false,
        "he_mode" => :full,
    ),
    "tb_he_warmup" => Dict(
        "label" => "TB + heuristic HE (warmup only)",
        "training_mode" => :tb,
        "target_seed" => false,
        "he_mode" => :warmup,
    ),
    "tb_he_warmup_locked" => Dict(
        "label" => "TB + heuristic HE (warmup only locked)",
        "training_mode" => :tb,
        "target_seed" => false,
        "he_mode" => :warmup,
    ),
)

const BUDGET = parse(Int, get(ENV, "PMO_BUDGET", "10000"))
const N_RUNS = parse(Int, get(ENV, "PMO_RUNS", "1"))
const TASKS = parse_list_env("PMO_TASKS", DEFAULT_TASKS)
const CONFIG_KEYS = parse_list_env("PMO_CONFIGS", ["tb", "rwmle", "tb_seeded", "rwmle_seeded"])
const LOGDIR = abspath(get(ENV, "TRUTH_SPRINT_LOGDIR", joinpath(@__DIR__, "..", "..", "checkpoints", "truth_sprint")))
mkpath(LOGDIR)
const LOGFILE = joinpath(LOGDIR, "truth_sprint.log")
const SUMMARY_INDENT = repeat(" ", 28)

function logmsg(msg::String)
    open(LOGFILE, "a") do f
        println(f, msg)
    end
    println(msg)
    flush(stdout)
end

function he_mode_settings(he_mode::Symbol)
    if he_mode == :full
        return true, HE_WARMUP_EPISODES, HE_EPISODES_PER_SEGMENT
    elseif he_mode == :warmup
        return true, HE_WARMUP_EPISODES, 0
    end
    return false, 0, 0
end

function source_fraction(result::PMOResult, scope_key::String, source_key::String)::Float64
    summary = result.provenance_summary
    nested = get(summary, scope_key, Dict{String,Float64}())
    return Float64(get(nested, source_key, 0.0))
end

function breakdown_value(result::PMOResult, key::String)::Float64
    return Float64(get(result.oracle_call_breakdown, key, 0))
end

function summarize_runs(runs)::Dict{String,Any}
    aucs = Float64[r.auc_top10 for r in runs]
    top1s = Float64[r.top1 for r in runs]
    top10s = Float64[r.top10_mean for r in runs]
    diversities = Float64[r.diversity for r in runs]
    unique_counts = Float64[r.unique_molecules for r in runs]

    breakdown_key_set = Set{String}()
    for r in runs
        for k in keys(r.oracle_call_breakdown)
            push!(breakdown_key_set, String(k))
        end
    end
    breakdown_keys = sort(collect(breakdown_key_set))
    breakdown_means = Dict{String,Float64}(k => mean(Float64[get(r.oracle_call_breakdown, k, 0) for r in runs]) for k in breakdown_keys)

    topk_fraction_means = Dict{String,Float64}(
        key => mean(Float64[source_fraction(r, "topk_source_fractions", key) for r in runs])
        for key in ("tb", "ga", "he", "seed", "bootstrap")
    )
    frontier_fraction_means = Dict{String,Float64}(
        key => mean(Float64[source_fraction(r, "overall_source_fractions", key) for r in runs])
        for key in ("tb", "ga", "he", "seed", "bootstrap")
    )

    he_diag_means = Dict{String,Float64}()
    for key in ("episode_count", "total_he_calls", "total_commits", "total_frontier_gain", "unique_basin_count")
        vals = Float64[Float64(diagnostics_value(r, "run_capacity", key, 0.0)) for r in runs]
        he_diag_means[key] = isempty(vals) ? 0.0 : mean(vals)
    end

    return Dict(
        "auc_mean" => mean(aucs),
        "auc_std" => length(aucs) > 1 ? std(aucs) : 0.0,
        "top1_mean" => mean(top1s),
        "top1_std" => length(top1s) > 1 ? std(top1s) : 0.0,
        "top10_mean" => mean(top10s),
        "top10_std" => length(top10s) > 1 ? std(top10s) : 0.0,
        "diversity_mean" => mean(diversities),
        "unique_mean" => mean(unique_counts),
        "oracle_call_breakdown_mean" => breakdown_means,
        "topk_source_fraction_mean" => topk_fraction_means,
        "frontier_source_fraction_mean" => frontier_fraction_means,
        "he_diagnostics_mean" => he_diag_means,
    )
end

function family_mean(task_summaries::Dict{String,Any}, family_tasks::Set{String})
    values = Float64[]
    for (task, summary) in task_summaries
        task in family_tasks || continue
        push!(values, Float64(summary["auc_mean"]))
    end
    return isempty(values) ? 0.0 : mean(values)
end

function format_breakdown_mean(summary::Dict{String,Any})::String
    breakdown = get(summary, "oracle_call_breakdown_mean", Dict{String,Float64}())
    ordered_keys = ["seed", "frontier_bootstrap", "model", "ga", "he_warmup", "he_interleaved", "unattributed", "total"]
    parts = String[]
    for key in ordered_keys
        haskey(breakdown, key) || continue
        push!(parts, "$(key)=$(round(breakdown[key], digits=1))")
    end
    return join(parts, ", ")
end

function format_source_fracs(fracs::Dict{String,Float64})::String
    return "tb=$(round(get(fracs, "tb", 0.0), digits=3)), ga=$(round(get(fracs, "ga", 0.0), digits=3)), he=$(round(get(fracs, "he", 0.0), digits=3)), seed=$(round(get(fracs, "seed", 0.0), digits=3)), bootstrap=$(round(get(fracs, "bootstrap", 0.0), digits=3))"
end

function diagnostics_value(result::PMOResult, section::String, key::String, default)
    section_dict = get(result.diagnostics_summary, section, Dict{String,Any}())
    return get(section_dict, key, default)
end

function format_he_diag_mean(summary::Dict{String,Any})::String
    he_diag = get(summary, "he_diagnostics_mean", Dict{String,Float64}())
    isempty(he_diag) && return "none"
    return "episodes=$(round(get(he_diag, "episode_count", 0.0), digits=1)), commits=$(round(get(he_diag, "total_commits", 0.0), digits=1)), frontier_gain=$(round(get(he_diag, "total_frontier_gain", 0.0), digits=4)), basins=$(round(get(he_diag, "unique_basin_count", 0.0), digits=1))"
end

function closure_tier(tb_summary::Dict{String,Any}, cfg_summary::Dict{String,Any}, tasks::Vector{String})
    task_deltas = Float64[Float64(cfg_summary[task]["auc_mean"]) - Float64(tb_summary[task]["auc_mean"]) for task in tasks]
    overall_delta = mean(task_deltas)
    positive_tasks = count(>(0.0), task_deltas)
    structural_delta = family_mean(cfg_summary, STRUCTURAL_TASKS) - family_mean(tb_summary, STRUCTURAL_TASKS)
    property_delta = family_mean(cfg_summary, PROPERTY_TASKS) - family_mean(tb_summary, PROPERTY_TASKS)
    property_rel = abs(family_mean(tb_summary, PROPERTY_TASKS)) > 1e-8 ? property_delta / family_mean(tb_summary, PROPERTY_TASKS) : 0.0
    property_rel_deltas = Float64[]
    for task in tasks
        task in PROPERTY_TASKS || continue
        tb_auc = Float64(tb_summary[task]["auc_mean"])
        cfg_auc = Float64(cfg_summary[task]["auc_mean"])
        push!(property_rel_deltas, abs(tb_auc) > 1e-8 ? (cfg_auc - tb_auc) / tb_auc : 0.0)
    end
    worst_property_rel = isempty(property_rel_deltas) ? 0.0 : minimum(property_rel_deltas)
    catastrophic_single = worst_property_rel < -0.15
    catastrophic_family = property_rel < -0.10
    he_frontier_values = Float64[get(get(cfg_summary[task], "frontier_source_fraction_mean", Dict{String,Float64}()), "he", 0.0) for task in tasks]
    mean_he_frontier = isempty(he_frontier_values) ? 0.0 : mean(he_frontier_values)

    if overall_delta > 0 && structural_delta > 0 && !catastrophic_single && !catastrophic_family && positive_tasks >= 3 && mean_he_frontier >= 0.05
        return Dict("tier" => "Tier 1 — Strong Closure", "overall_delta" => overall_delta, "structural_delta" => structural_delta, "property_delta" => property_delta, "positive_tasks" => positive_tasks, "mean_he_frontier" => mean_he_frontier, "catastrophic_single" => catastrophic_single, "catastrophic_family" => catastrophic_family)
    elseif overall_delta > -0.01 && structural_delta > 0 && !catastrophic_single && !catastrophic_family && mean_he_frontier >= 0.03
        return Dict("tier" => "Tier 2 — Conditional Closure", "overall_delta" => overall_delta, "structural_delta" => structural_delta, "property_delta" => property_delta, "positive_tasks" => positive_tasks, "mean_he_frontier" => mean_he_frontier, "catastrophic_single" => catastrophic_single, "catastrophic_family" => catastrophic_family)
    end

    return Dict("tier" => "Tier 3 — No Closure", "overall_delta" => overall_delta, "structural_delta" => structural_delta, "property_delta" => property_delta, "positive_tasks" => positive_tasks, "mean_he_frontier" => mean_he_frontier, "catastrophic_single" => catastrophic_single, "catastrophic_family" => catastrophic_family)
end

logmsg("=" ^ 78)
logmsg("BENCHMARK TRUTH SPRINT — MATCHED SEARCH CAFE-GFN")
logmsg("=" ^ 78)
logmsg("Log dir: $LOGDIR")
logmsg("Budget per task: $BUDGET")
logmsg("Runs per task: $N_RUNS")
logmsg("Tasks: $(join(TASKS, ", "))")
logmsg("Configs: $(join(CONFIG_KEYS, ", "))")
logmsg("Search stack: batch=$(PMO_BATCH_SIZE), replay=$(PMO_REPLAY_RATIO), per-step GA=$(PMO_GA_PER_STEP), crossover=$(PMO_GA_CROSSOVER), mutation=$(PMO_GA_MUTATION), iters=$(PMO_N_ITERATIONS)")
logmsg("Bootstrap env: samples=$(FRONTIER_BOOTSTRAP_SAMPLES), min_entries=$(FRONTIER_BOOTSTRAP_MIN_ENTRIES)")
logmsg("HE env: fraction=$(HE_BUDGET_FRACTION), warmup=$(HE_WARMUP_EPISODES), per_segment=$(HE_EPISODES_PER_SEGMENT), horizon=$(HE_HORIZON), allow_crossover=$(HE_ALLOW_CROSSOVER)")

checkpoint_path = joinpath(@__DIR__, "..", "..", "checkpoints", "pretrain", "final.jls")
if !isfile(checkpoint_path)
    error("Pretrained checkpoint not found: $checkpoint_path")
end

logmsg("Loading checkpoint: $checkpoint_path")
checkpoint = deserialize(checkpoint_path)
pretrained_params = checkpoint["params"]
pretrained_states = checkpoint["states"]
vocab = SMILESVocabulary()
actual_vocab_size = size(pretrained_params.output.layer_2.weight, 1)
policy_model, _, _ = create_smiles_policy(; vocab_size=actual_vocab_size, hidden_dim=512, embed_dim=128, n_layers=3)
logmsg("Checkpoint loaded (vocab_size=$actual_vocab_size)")

all_results = Dict{String, Any}()
summary_by_config = Dict{String, Any}()

for config_key in CONFIG_KEYS
    haskey(CONFIG_LIBRARY, config_key) || error("Unknown config key: $config_key")
    config = CONFIG_LIBRARY[config_key]
    label = config["label"]
    mode = config["training_mode"]
    seed_enabled = config["target_seed"]
    he_mode = config["he_mode"]
    use_he, he_warmup_eps, he_segment_eps = he_mode_settings(he_mode)

    logmsg("\n" * "=" ^ 78)
    logmsg("CONFIG: $label")
    logmsg("=" ^ 78)
    logmsg("Mode=$(mode) | target_seed=$(seed_enabled) | HE=$(use_he ? String(he_mode) : "off") | HE fraction=$(use_he ? string(HE_BUDGET_FRACTION) : "0.0")")

    config_results = Dict{String, Any}()
    config_summary = Dict{String, Any}()

    for task in TASKS
        task_runs = Any[]
        logmsg("\n--- Task: $task ---")

        for run_idx in 1:N_RUNS
            target_smi = get(TARGET_SMILES, task, nothing)
            use_seed = seed_enabled && !isnothing(target_smi)
            he_config = use_he ? build_he_config() : HierarchicalEditConfig()
            logmsg("Run $run_idx/$N_RUNS | mode=$(mode) | target_seed=$(use_seed) | HE=$(use_he ? String(he_mode) : "off")")
            start_t = time()

            run_artifact_dir = use_he ? joinpath(LOGDIR, "he_artifacts", _artifact_safe_component(config_key), _artifact_safe_component(task), "run$(run_idx)") : nothing
            result = run_smiles_pmo_task(task;
                budget=BUDGET,
                pretrained_params=deepcopy(pretrained_params),
                pretrained_states=deepcopy(pretrained_states),
                vocab=vocab,
                policy_model=policy_model,
                training_mode=mode,
                use_replay=true,
                replay_ratio=PMO_REPLAY_RATIO,
                batch_size=PMO_BATCH_SIZE,
                n_iterations=PMO_N_ITERATIONS,
                ga_per_step=PMO_GA_PER_STEP,
                ga_crossover=PMO_GA_CROSSOVER,
                ga_mutation=PMO_GA_MUTATION,
                track_frontier=true,
                frontier_bootstrap_samples=FRONTIER_BOOTSTRAP_SAMPLES,
                frontier_bootstrap_min_entries=FRONTIER_BOOTSTRAP_MIN_ENTRIES,
                target_smiles=target_smi,
                target_seed=use_seed,
                target_seed_augmentations=8,
                verbose=true,
                use_hierarchical_edit=use_he,
                he_warmup_episodes=he_warmup_eps,
                he_episodes_per_segment=he_segment_eps,
                he_budget_fraction=use_he ? HE_BUDGET_FRACTION : 0.0,
                he_config=he_config,
                he_artifact_dir=run_artifact_dir,
                he_run_context=Dict{String,Any}(
                    "task_name" => task,
                    "config_name" => config_key,
                    "run_index" => run_idx,
                ),
            )

            push!(task_runs, result)
            elapsed = round(time() - start_t, digits=1)
            ref = get(GENETIC_GFN_REF, task, 0.0)
            delta = round(result.auc_top10 - ref, digits=4)
            delta_str = delta >= 0 ? "+$(delta)" : string(delta)
            topk_fracs = get(result.provenance_summary, "topk_source_fractions", Dict{String,Float64}())
            breakdown = result.oracle_call_breakdown
            tb_top10 = round(get(topk_fracs, "tb", 0.0), digits=3)
            ga_top10 = round(get(topk_fracs, "ga", 0.0), digits=3)
            he_top10 = round(get(topk_fracs, "he", 0.0), digits=3)
            bootstrap_top10 = round(get(topk_fracs, "bootstrap", 0.0), digits=3)
            seed_calls = get(breakdown, "seed", 0)
            bootstrap_calls = get(breakdown, "frontier_bootstrap", 0)
            model_calls = get(breakdown, "model", 0)
            ga_calls = get(breakdown, "ga", 0)
            he_warmup_calls = get(breakdown, "he_warmup", 0)
            he_interleaved_calls = get(breakdown, "he_interleaved", 0)
            total_calls = get(breakdown, "total", result.n_oracle_calls)
            logmsg(
                "Result: AUC=$(round(result.auc_top10, digits=4)) | Top1=$(round(result.top1, digits=4)) | Top10=$(round(result.top10_mean, digits=4)) | Ref=$(round(ref, digits=3)) | Δ=$delta_str | top10[tb=$(tb_top10),ga=$(ga_top10),he=$(he_top10),bootstrap=$(bootstrap_top10)] | calls[seed=$(seed_calls),boot=$(bootstrap_calls),model=$(model_calls),ga=$(ga_calls),he_w=$(he_warmup_calls),he_i=$(he_interleaved_calls),total=$(total_calls)] | $(elapsed)s"
            )
            if use_he && !isempty(result.artifact_paths)
                episode_summary_path = get(result.artifact_paths, "episode_summary", "none")
                capacity_summary_path = get(result.artifact_paths, "capacity_summary", "none")
                logmsg("HE artifacts: episode_summary=$(episode_summary_path) | capacity_summary=$(capacity_summary_path)")
            end
        end

        config_results[task] = task_runs
        config_summary[task] = summarize_runs(task_runs)
    end

    all_results[config_key] = config_results
    summary_by_config[config_key] = config_summary
end

logmsg("\n\n" * "=" ^ 78)
logmsg("TRUTH SPRINT SUMMARY")
logmsg("=" ^ 78)
for config_key in CONFIG_KEYS
    config = CONFIG_LIBRARY[config_key]
    label = config["label"]
    config_summary = summary_by_config[config_key]
    logmsg("\n--- $label ---")
    total_auc = 0.0
    total_ref = 0.0
    n = 0
    for task in TASKS
        summary = config_summary[task]
        auc_mean = Float64(summary["auc_mean"])
        auc_std = Float64(summary["auc_std"])
        top1_mean = Float64(summary["top1_mean"])
        top10_mean = Float64(summary["top10_mean"])
        ref = get(GENETIC_GFN_REF, task, 0.0)
        total_auc += auc_mean
        total_ref += ref
        n += 1
        delta = round(auc_mean - ref, digits=4)
        delta_str = delta >= 0 ? "+$(delta)" : string(delta)
        logmsg("$(rpad(task, 26)) AUC=$(round(auc_mean, digits=4)) ± $(round(auc_std, digits=4)) | Top1=$(round(top1_mean, digits=4)) | Top10=$(round(top10_mean, digits=4)) | Ref=$(round(ref, digits=3)) | Δ=$delta_str")
        logmsg("$(SUMMARY_INDENT)Calls: $(format_breakdown_mean(summary))")
        logmsg("$(SUMMARY_INDENT)Top10 provenance: $(format_source_fracs(get(summary, "topk_source_fraction_mean", Dict{String,Float64}())))")
        logmsg("$(SUMMARY_INDENT)Frontier provenance: $(format_source_fracs(get(summary, "frontier_source_fraction_mean", Dict{String,Float64}())))")
        he_diag = get(summary, "he_diagnostics_mean", Dict{String,Float64}())
        get(he_diag, "total_he_calls", 0.0) > 0 && logmsg("$(SUMMARY_INDENT)HE diagnostics: $(format_he_diag_mean(summary))")
    end
    if n > 0
        structural_auc = family_mean(config_summary, STRUCTURAL_TASKS)
        property_auc = family_mean(config_summary, PROPERTY_TASKS)
        logmsg("Structural-family mean AUC: $(round(structural_auc, digits=4))")
        logmsg("Property-family mean AUC: $(round(property_auc, digits=4))")
        logmsg("Total ($(n) tasks): $(round(total_auc, digits=4)) | Ref=$(round(total_ref, digits=4))")
        logmsg("Per-task avg: $(round(total_auc / n, digits=4)) | Projected PMO-23=$(round((total_auc / n) * 23, digits=2))")
    end
end

if "tb" in CONFIG_KEYS
    tb_summary = summary_by_config["tb"]
    logmsg("\n" * "=" ^ 78)
    logmsg("DELTAS VS TB BASELINE")
    logmsg("=" ^ 78)
    for config_key in CONFIG_KEYS
        config_key == "tb" && continue
        config = CONFIG_LIBRARY[config_key]
        label = config["label"]
        cfg_summary = summary_by_config[config_key]
        logmsg("\n--- $label vs TB matched ---")

        property_rel_deltas = Float64[]
        for task in TASKS
            tb_auc = Float64(tb_summary[task]["auc_mean"])
            cfg_auc = Float64(cfg_summary[task]["auc_mean"])
            delta = cfg_auc - tb_auc
            rel = abs(tb_auc) > 1e-8 ? delta / tb_auc : 0.0
            task in PROPERTY_TASKS && push!(property_rel_deltas, rel)
            logmsg("$(rpad(task, 26)) ΔAUC=$(round(delta, digits=4)) | rel=$(round(rel * 100, digits=2))%")
        end

        structural_delta = family_mean(cfg_summary, STRUCTURAL_TASKS) - family_mean(tb_summary, STRUCTURAL_TASKS)
        property_delta = family_mean(cfg_summary, PROPERTY_TASKS) - family_mean(tb_summary, PROPERTY_TASKS)
        property_rel = abs(family_mean(tb_summary, PROPERTY_TASKS)) > 1e-8 ? property_delta / family_mean(tb_summary, PROPERTY_TASKS) : 0.0
        worst_property_rel = isempty(property_rel_deltas) ? 0.0 : minimum(property_rel_deltas)
        catastrophic_single = worst_property_rel < -0.15
        catastrophic_family = property_rel < -0.10

        logmsg("Structural-family ΔAUC=$(round(structural_delta, digits=4))")
        logmsg("Property-family ΔAUC=$(round(property_delta, digits=4)) | rel=$(round(property_rel * 100, digits=2))%")
        logmsg("Worst single property relative Δ=$(round(worst_property_rel * 100, digits=2))%")
        logmsg("Catastrophic harm flags: single=$(catastrophic_single), family=$(catastrophic_family)")
        tier = closure_tier(tb_summary, cfg_summary, TASKS)
        tier_name = tier["tier"]
        tier_overall = tier["overall_delta"]
        tier_structural = tier["structural_delta"]
        tier_property = tier["property_delta"]
        tier_positive_tasks = tier["positive_tasks"]
        tier_mean_he_frontier = tier["mean_he_frontier"]
        logmsg("Closure tier: $(tier_name)")
        logmsg("$(SUMMARY_INDENT)overall Δ=$(round(tier_overall, digits=4)) | structural Δ=$(round(tier_structural, digits=4)) | property Δ=$(round(tier_property, digits=4)) | positive tasks=$(tier_positive_tasks) | mean HE frontier=$(round(tier_mean_he_frontier, digits=4))")
    end
end

closure_by_config = Dict{String,Any}()
if "tb" in CONFIG_KEYS
    tb_summary = summary_by_config["tb"]
    for config_key in CONFIG_KEYS
        config_key == "tb" && continue
        closure_by_config[config_key] = closure_tier(tb_summary, summary_by_config[config_key], TASKS)
    end
end

save_path = joinpath(LOGDIR, "truth_sprint_results.jls")
serialize(save_path, Dict(
    "budget" => BUDGET,
    "runs" => N_RUNS,
    "tasks" => TASKS,
    "configs" => CONFIG_KEYS,
    "he_settings" => Dict(
        "fraction" => HE_BUDGET_FRACTION,
        "warmup_episodes" => HE_WARMUP_EPISODES,
        "episodes_per_segment" => HE_EPISODES_PER_SEGMENT,
        "horizon" => HE_HORIZON,
        "topk_tracking" => HE_TOPK_TRACKING,
        "max_operator_candidates" => HE_MAX_OPERATOR_CANDIDATES,
        "max_step_attempts" => HE_MAX_STEP_ATTEMPTS,
        "min_exploration_per_operator" => HE_MIN_EXPLORATION_PER_OPERATOR,
        "multi_child_min_reward_ratio" => HE_MULTI_CHILD_MIN_REWARD_RATIO,
        "operator_prior_strength" => HE_OPERATOR_PRIOR_STRENGTH,
        "allow_crossover" => HE_ALLOW_CROSSOVER,
    ),
    "results" => all_results,
    "summary_by_config" => summary_by_config,
    "closure_by_config" => closure_by_config,
))
logmsg("\nSaved results to: $save_path")
flush(stdout)
