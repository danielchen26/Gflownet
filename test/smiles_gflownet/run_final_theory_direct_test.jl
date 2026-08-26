# ============================================================================
# Final Theory Direct Test
# ============================================================================
#
# Minimal decisive system-level comparison:
#   Arm A: pure TB
#   Arm B: TB + task-aware learned edit control + conservative gate
#
# Usage:
#   julia --project=. test/smiles_gflownet/run_final_theory_direct_test.jl
#   FINAL_THEORY_BUDGET=256 FINAL_THEORY_REPEATS=2 julia --project=. test/smiles_gflownet/run_final_theory_direct_test.jl

using Pkg
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(PROJECT_ROOT)

using GFlowNet
using Random
using Serialization
using Statistics
using Dates

include(joinpath(PROJECT_ROOT, "src", "applications", "smiles_gflownet.jl"))
include(joinpath(PROJECT_ROOT, "src", "utils", "visualization", "python", "rdkit_bridge.jl"))
include(joinpath(PROJECT_ROOT, "src", "utils", "visualization", "python", "oracle_bridge.jl"))
include(joinpath(PROJECT_ROOT, "src", "utils", "visualization", "core", "oracle_manager.jl"))
include(joinpath(PROJECT_ROOT, "src", "utils", "visualization", "core", "pmo_benchmark.jl"))

const SIBLING_MAIN_ROOT = normpath(joinpath(PROJECT_ROOT, "..", "Gflownet"))

function _first_existing_path(paths::Vector{String}; want_dir::Bool=false)
    for p in paths
        isempty(p) && continue
        pp = normpath(p)
        if want_dir
            isdir(pp) && return pp
        else
            isfile(pp) && return pp
        end
    end
    return ""
end

function _env_or_empty(key::String)
    return haskey(ENV, key) ? strip(ENV[key]) : ""
end

function _safe_mean(xs)
    vals = Float64[x for x in xs]
    isempty(vals) && return 0.0
    return mean(vals)
end

function _safe_std(xs)
    vals = Float64[x for x in xs]
    length(vals) <= 1 && return 0.0
    return std(vals)
end

function _selection_reason_counts(logs)
    counts = Dict{String,Int}()
    for log in logs
        reason = String(getfield(log, :selection_reason))
        counts[reason] = get(counts, reason, 0) + 1
    end
    return counts
end

function _decision_key(log)
    return string(getfield(log, :episode_id), "|", getfield(log, :step_index), "|", getfield(log, :attempt_index))
end

function _summarize_learned_policy(result::PMOResult)::Dict{String,Any}
    raw_diag_path = get(result.artifact_paths, "raw_diagnostics", "")
    isempty(raw_diag_path) && return Dict{String,Any}(
        "available" => false,
        "parent_override_rate" => 0.0,
        "operator_override_rate" => 0.0,
        "override_reward_delta_mean" => 0.0,
        "override_frontier_delta_mean" => 0.0,
        "override_enters_topk_rate" => 0.0,
        "parent_mean_entropy" => 0.0,
        "operator_mean_entropy" => 0.0,
        "selection_reason_counts" => Dict{String,Any}(),
    )

    diags = deserialize(raw_diag_path)
    decision_logs = get(diags, "decision_logs", Any[])
    parent_logs = get(diags, "parent_logs", Any[])
    operator_logs = get(diags, "operator_logs", Any[])

    parent_override_keys = Set(_decision_key(log) for log in parent_logs if getfield(log, :override_applied))
    operator_override_keys = Set(_decision_key(log) for log in operator_logs if getfield(log, :override_applied))
    any_override_keys = union(parent_override_keys, operator_override_keys)

    override_decisions = [log for log in decision_logs if _decision_key(log) in any_override_keys]
    no_override_decisions = [log for log in decision_logs if !(_decision_key(log) in any_override_keys)]

    return Dict{String,Any}(
        "available" => true,
        "parent_total" => length(parent_logs),
        "operator_total" => length(operator_logs),
        "parent_override_rate" => _safe_mean(getfield(log, :override_applied) ? 1.0 : 0.0 for log in parent_logs),
        "operator_override_rate" => _safe_mean(getfield(log, :override_applied) ? 1.0 : 0.0 for log in operator_logs),
        "parent_mean_entropy" => _safe_mean(getfield(log, :learned_entropy) for log in parent_logs),
        "operator_mean_entropy" => _safe_mean(getfield(log, :learned_entropy) for log in operator_logs),
        "override_reward_delta_mean" => _safe_mean(getfield(log, :reward_delta) for log in override_decisions),
        "override_frontier_delta_mean" => _safe_mean(getfield(log, :frontier_utility_delta) for log in override_decisions),
        "override_enters_topk_rate" => _safe_mean(getfield(log, :enters_topk) ? 1.0 : 0.0 for log in override_decisions),
        "no_override_reward_delta_mean" => _safe_mean(getfield(log, :reward_delta) for log in no_override_decisions),
        "no_override_frontier_delta_mean" => _safe_mean(getfield(log, :frontier_utility_delta) for log in no_override_decisions),
        "no_override_enters_topk_rate" => _safe_mean(getfield(log, :enters_topk) ? 1.0 : 0.0 for log in no_override_decisions),
        "selection_reason_counts" => Dict(
            "parent" => _selection_reason_counts(parent_logs),
            "operator" => _selection_reason_counts(operator_logs),
        ),
    )
end

function _summarize_run(result::PMOResult)::Dict{String,Any}
    topk_fracs = get(result.provenance_summary, "topk_source_fractions", Dict{String,Float64}())
    overall_fracs = get(result.provenance_summary, "overall_source_fractions", Dict{String,Float64}())
    call_breakdown = result.oracle_call_breakdown
    learned_summary = _summarize_learned_policy(result)
    run_capacity = get(result.diagnostics_summary, "run_capacity", Dict{String,Any}())

    return Dict{String,Any}(
        "auc_top10" => result.auc_top10,
        "top1" => result.top1,
        "top10_mean" => result.top10_mean,
        "diversity" => result.diversity,
        "n_oracle_calls" => result.n_oracle_calls,
        "he_top10_fraction" => get(topk_fracs, "he", 0.0),
        "he_frontier_fraction" => get(overall_fracs, "he", 0.0),
        "tb_top10_fraction" => get(topk_fracs, "tb", 0.0),
        "seed_top10_fraction" => get(topk_fracs, "seed", 0.0),
        "he_calls" => get(call_breakdown, "he_warmup", 0) + get(call_breakdown, "he_interleaved", 0),
        "model_calls" => get(call_breakdown, "model", 0),
        "ga_calls" => get(call_breakdown, "ga", 0),
        "episode_count" => Int(get(run_capacity, "episode_count", 0)),
        "total_commits" => Int(get(run_capacity, "total_commits", 0)),
        "total_frontier_gain" => Float64(get(run_capacity, "total_frontier_gain", 0.0)),
        "parent_override_rate" => get(learned_summary, "parent_override_rate", 0.0),
        "operator_override_rate" => get(learned_summary, "operator_override_rate", 0.0),
        "parent_mean_entropy" => get(learned_summary, "parent_mean_entropy", 0.0),
        "operator_mean_entropy" => get(learned_summary, "operator_mean_entropy", 0.0),
        "override_reward_delta_mean" => get(learned_summary, "override_reward_delta_mean", 0.0),
        "override_frontier_delta_mean" => get(learned_summary, "override_frontier_delta_mean", 0.0),
        "override_enters_topk_rate" => get(learned_summary, "override_enters_topk_rate", 0.0),
        "selection_reason_counts" => get(learned_summary, "selection_reason_counts", Dict{String,Any}()),
    )
end

function _mean_metric(records, key::String)
    return _safe_mean(get(record, key, 0.0) for record in records)
end

function _build_base_he_config()
    return HierarchicalEditConfig(;
        horizon=3,
        allow_crossover=true,
        allow_fragment_ops=false,
        max_step_attempts=3,
        min_exploration_per_operator=5,
        multi_child_min_reward_ratio=0.2,
        operator_prior_strength=4.0,
        use_learned_basin=false,
        use_learned_parent=false,
        use_learned_operator=false,
    )
end

function _build_final_theory_he_config(controller::EditTBPolicyController)
    return HierarchicalEditConfig(;
        horizon=3,
        allow_crossover=true,
        allow_fragment_ops=false,
        max_step_attempts=3,
        min_exploration_per_operator=5,
        multi_child_min_reward_ratio=0.2,
        operator_prior_strength=4.0,
        use_learned_basin=true,
        learned_basin_controller=controller,
        use_learned_parent=true,
        learned_parent_controller=controller,
        use_learned_operator=true,
        learned_operator_controller=controller,
    )
end

const BUDGET = parse(Int, get(ENV, "FINAL_THEORY_BUDGET", "256"))
const N_REPEATS = parse(Int, get(ENV, "FINAL_THEORY_REPEATS", "2"))
const THEORY_EPOCHS = parse(Int, get(ENV, "FINAL_THEORY_EPOCHS", "40"))
const PMO_BATCH_SIZE = parse(Int, get(ENV, "PMO_BATCH_SIZE", "32"))
const PMO_REPLAY_RATIO = parse(Int, get(ENV, "PMO_REPLAY_RATIO", "4"))
const PMO_N_ITERATIONS = parse(Int, get(ENV, "PMO_N_ITERATIONS", "8"))
const PMO_GA_CROSSOVER = parse(Int, get(ENV, "PMO_GA_CROSSOVER", "0"))
const PMO_GA_MUTATION = parse(Int, get(ENV, "PMO_GA_MUTATION", "0"))
const FRONTIER_BOOTSTRAP_SAMPLES = parse(Int, get(ENV, "FINAL_THEORY_BOOTSTRAP_SAMPLES", "12"))
const HE_WARMUP_EPISODES = parse(Int, get(ENV, "HE_WARMUP_EPISODES", "2"))
const HE_EPISODES_PER_SEGMENT = parse(Int, get(ENV, "HE_EPISODES_PER_SEGMENT", "1"))
const HE_BUDGET_FRACTION = parse(Float64, get(ENV, "FINAL_THEORY_HE_BUDGET_FRACTION", "0.15"))
const TASKS = [String(strip(t)) for t in split(get(ENV, "FINAL_THEORY_TASKS", "qed,drd2,celecoxib_rediscovery"), ',') if !isempty(strip(t))]
const LOGDIR = get(ENV, "FINAL_THEORY_LOGDIR", joinpath(PROJECT_ROOT, "checkpoints", "final_theory_direct_test"))
const ARTIFACT_ROOT = let env_path = _env_or_empty("FINAL_THEORY_ARTIFACT_ROOT")
    resolved = _first_existing_path(String[
        env_path,
        joinpath(PROJECT_ROOT, "checkpoints", "truth_sprint_stage_b_f015_truth_tasksharded"),
        joinpath(SIBLING_MAIN_ROOT, "checkpoints", "truth_sprint_stage_b_f015_truth_tasksharded"),
    ]; want_dir=true)
    isempty(resolved) && error("Could not locate truth-sprint HE artifacts. Set FINAL_THEORY_ARTIFACT_ROOT.")
    resolved
end
const CHECKPOINT_PATH = let env_path = _env_or_empty("FINAL_THEORY_PRETRAIN_CHECKPOINT")
    resolved = _first_existing_path(String[
        env_path,
        joinpath(PROJECT_ROOT, "checkpoints", "pretrain", "final.jls"),
        joinpath(SIBLING_MAIN_ROOT, "checkpoints", "pretrain", "final.jls"),
    ])
    isempty(resolved) && error("Could not locate pretrained checkpoint. Set FINAL_THEORY_PRETRAIN_CHECKPOINT.")
    resolved
end
const TARGET_SMILES = Dict(
    "celecoxib_rediscovery" => "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
)
const ARM_NAMES = ["tb_baseline", "final_theory_v1"]

logmsg(msg) = println("[$(Dates.format(now(), "HH:MM:SS"))] $msg")

mkpath(LOGDIR)

logmsg("Loading pretrained checkpoint: $CHECKPOINT_PATH")
checkpoint = deserialize(CHECKPOINT_PATH)
const PRETRAINED_PARAMS = checkpoint["params"]
const PRETRAINED_STATES = checkpoint["states"]
const VOCAB = SMILESVocabulary()
actual_vocab_size = size(PRETRAINED_PARAMS.output.layer_2.weight, 1)
const POLICY_MODEL, _, _ = create_smiles_policy(; vocab_size=actual_vocab_size, hidden_dim=512, embed_dim=128, n_layers=3)

logmsg("Loading HE artifacts from: $ARTIFACT_ROOT")
train_config = EditTBConfig(
    n_epochs=THEORY_EPOCHS,
    reward_mode=:top10_delta,
    include_heuristic_scores=true,
    heuristic_interpolation=0.5,
    entropy_floor=0.05,
    task_conditioning=true,
)
dataset = load_edit_tb_dataset(ARTIFACT_ROOT; config=train_config)
logmsg("Dataset: $(dataset.stats["n_trajectories"]) trajectories, $(dataset.stats["n_steps"]) steps, tasks=$(join(dataset.task_names, ", "))")

rng = Random.MersenneTwister(42)
model, params, states = init_edit_policy(rng; config=train_config, task_names=dataset.task_names)
best_params, best_log_Z, history = train_edit_policy!(
    model, params, states, dataset, train_config;
    verbose=true, rng=Random.MersenneTwister(123)
)
controller = EditTBPolicyController(model, best_params, states, train_config; task_names=dataset.task_names)
policy_ckpt = joinpath(LOGDIR, "final_theory_v1_policy.jls")
save_edit_policy(policy_ckpt, best_params, best_log_Z, history, train_config)
logmsg("Saved final-theory candidate policy: $policy_ckpt")

function _run_arm(arm_name::String, task::AbstractString, run_idx::Int)
    task = String(task)
    target_smi = get(TARGET_SMILES, task, nothing)
    use_seed = !isnothing(target_smi)
    use_hierarchical_edit = arm_name != "tb_baseline"
    track_frontier = use_hierarchical_edit
    bootstrap_samples = use_hierarchical_edit ? FRONTIER_BOOTSTRAP_SAMPLES : 0
    he_config = if use_hierarchical_edit
        set_edit_tb_task!(controller, task)
        controller._last_basin = nothing
        _build_final_theory_he_config(controller)
    else
        _build_base_he_config()
    end

    run_artifact_dir = use_hierarchical_edit ? joinpath(LOGDIR, "artifacts", arm_name, task, "run$(run_idx)") : nothing
    !isnothing(run_artifact_dir) && mkpath(run_artifact_dir)

    result = run_smiles_pmo_task(task;
        budget=BUDGET,
        pretrained_params=deepcopy(PRETRAINED_PARAMS),
        pretrained_states=deepcopy(PRETRAINED_STATES),
        vocab=VOCAB,
        policy_model=POLICY_MODEL,
        training_mode=:tb,
        use_replay=true,
        replay_ratio=PMO_REPLAY_RATIO,
        batch_size=PMO_BATCH_SIZE,
        n_iterations=PMO_N_ITERATIONS,
        ga_per_step=false,
        ga_crossover=PMO_GA_CROSSOVER,
        ga_mutation=PMO_GA_MUTATION,
        track_frontier=track_frontier,
        frontier_bootstrap_samples=bootstrap_samples,
        frontier_bootstrap_min_entries=2,
        target_smiles=target_smi,
        target_seed=use_seed,
        target_seed_augmentations=8,
        verbose=true,
        use_hierarchical_edit=use_hierarchical_edit,
        he_warmup_episodes=HE_WARMUP_EPISODES,
        he_episodes_per_segment=HE_EPISODES_PER_SEGMENT,
        he_budget_fraction=HE_BUDGET_FRACTION,
        he_config=he_config,
        he_artifact_dir=run_artifact_dir,
        he_run_context=Dict{String,Any}(
            "task_name" => task,
            "config_name" => arm_name,
            "run_index" => run_idx,
            "theory_candidate" => arm_name == "final_theory_v1",
        ),
        allow_learned_he=(arm_name == "final_theory_v1"),
    )

    summary = _summarize_run(result)
    logmsg("  [$arm_name][$task][run$(run_idx)] auc=$(round(summary["auc_top10"], digits=4)) top10=$(round(summary["top10_mean"], digits=4)) he_top10=$(round(summary["he_top10_fraction"], digits=3)) he_calls=$(summary["he_calls"]) parent_override=$(round(summary["parent_override_rate"], digits=3)) operator_override=$(round(summary["operator_override_rate"], digits=3))")
    return Dict{String,Any}(
        "result" => result,
        "summary" => summary,
    )
end

logmsg("============================================================")
logmsg("Final Theory Direct Test")
logmsg("Budget=$BUDGET | Repeats=$N_REPEATS | Tasks=$(join(TASKS, ", "))")
logmsg("Arms=$(join(ARM_NAMES, ", "))")
logmsg("============================================================")

all_results = Dict{String, Dict{String, Vector{Dict{String,Any}}}}()
for arm_name in ARM_NAMES
    task_results = Dict{String, Vector{Dict{String,Any}}}()
    logmsg("--- Arm: $arm_name ---")
    for task in TASKS
        runs = Dict{String,Any}[]
        for run_idx in 1:N_REPEATS
            push!(runs, _run_arm(arm_name, task, run_idx))
        end
        task_results[task] = runs
    end
    all_results[arm_name] = task_results
end

summary_rows = Dict{String,Any}[]
for arm_name in ARM_NAMES
    for task in TASKS
        runs = all_results[arm_name][task]
        summaries = [r["summary"] for r in runs]
        push!(summary_rows, Dict{String,Any}(
            "arm" => arm_name,
            "task" => task,
            "auc_mean" => _mean_metric(summaries, "auc_top10"),
            "auc_std" => _safe_std(get(summary, "auc_top10", 0.0) for summary in summaries),
            "top10_mean" => _mean_metric(summaries, "top10_mean"),
            "top1_mean" => _mean_metric(summaries, "top1"),
            "he_top10_fraction" => _mean_metric(summaries, "he_top10_fraction"),
            "he_frontier_fraction" => _mean_metric(summaries, "he_frontier_fraction"),
            "he_calls" => _mean_metric(summaries, "he_calls"),
            "parent_override_rate" => _mean_metric(summaries, "parent_override_rate"),
            "operator_override_rate" => _mean_metric(summaries, "operator_override_rate"),
            "override_reward_delta_mean" => _mean_metric(summaries, "override_reward_delta_mean"),
            "override_frontier_delta_mean" => _mean_metric(summaries, "override_frontier_delta_mean"),
        ))
    end
end

baseline_rows = Dict(row["task"] => row for row in summary_rows if row["arm"] == "tb_baseline")
candidate_rows = Dict(row["task"] => row for row in summary_rows if row["arm"] == "final_theory_v1")
deltas = Dict{String,Dict{String,Float64}}()
for task in TASKS
    b = baseline_rows[task]
    c = candidate_rows[task]
    deltas[task] = Dict(
        "auc_delta" => c["auc_mean"] - b["auc_mean"],
        "top10_delta" => c["top10_mean"] - b["top10_mean"],
        "top1_delta" => c["top1_mean"] - b["top1_mean"],
    )
end

mean_auc_delta = _safe_mean(v["auc_delta"] for v in values(deltas))
mean_top10_delta = _safe_mean(v["top10_delta"] for v in values(deltas))
celecoxib_top10_delta = get(get(deltas, "celecoxib_rediscovery", Dict{String,Float64}()), "top10_delta", 0.0)
celecoxib_auc_delta = get(get(deltas, "celecoxib_rediscovery", Dict{String,Float64}()), "auc_delta", 0.0)
mean_he_top10 = _safe_mean(row["he_top10_fraction"] for row in summary_rows if row["arm"] == "final_theory_v1")
mean_override_frontier_delta = _safe_mean(row["override_frontier_delta_mean"] for row in summary_rows if row["arm"] == "final_theory_v1")
mean_parent_override = _safe_mean(row["parent_override_rate"] for row in summary_rows if row["arm"] == "final_theory_v1")
mean_operator_override = _safe_mean(row["operator_override_rate"] for row in summary_rows if row["arm"] == "final_theory_v1")
catastrophic_stress = (celecoxib_top10_delta <= -0.10) || (celecoxib_auc_delta <= -0.10)
provenance_alive = mean_he_top10 > 0.0
nonnegative_headline = mean_auc_delta >= 0.0
quality_nonnegative = mean_top10_delta >= 0.0
override_not_noise = mean_override_frontier_delta >= 0.0

verdict = if nonnegative_headline && !catastrophic_stress && provenance_alive && quality_nonnegative && override_not_noise
    "WORKING_ENOUGH_TO_SCALE"
elseif !catastrophic_stress && provenance_alive
    "PROMISING_BUT_INCONCLUSIVE"
else
    "CURRENTLY_FALSIFIED"
end

logmsg("============================================================")
logmsg("FINAL THEORY DIRECT TEST SUMMARY")
for row in summary_rows
    logmsg("$(rpad(row["arm"], 18)) $(rpad(row["task"], 24)) auc=$(round(row["auc_mean"], digits=4))±$(round(row["auc_std"], digits=4)) top10=$(round(row["top10_mean"], digits=4)) top1=$(round(row["top1_mean"], digits=4)) he_top10=$(round(row["he_top10_fraction"], digits=3)) he_calls=$(round(row["he_calls"], digits=1))")
end
logmsg("--- Deltas: final_theory_v1 - tb_baseline ---")
for task in TASKS
    δ = deltas[task]
    logmsg("$task: ΔAUC=$(round(δ["auc_delta"], digits=4)) ΔTop10=$(round(δ["top10_delta"], digits=4)) ΔTop1=$(round(δ["top1_delta"], digits=4))")
end
logmsg("--- Verdict ---")
logmsg("VERDICT=$verdict")
logmsg("mean_auc_delta=$(round(mean_auc_delta, digits=4)) mean_top10_delta=$(round(mean_top10_delta, digits=4))")
logmsg("mean_he_top10=$(round(mean_he_top10, digits=4)) parent_override=$(round(mean_parent_override, digits=4)) operator_override=$(round(mean_operator_override, digits=4))")
logmsg("mean_override_frontier_delta=$(round(mean_override_frontier_delta, digits=4)) catastrophic_stress=$(catastrophic_stress)")

results_file = joinpath(LOGDIR, "final_theory_direct_test_results.jls")
serialize(results_file, Dict(
    "timestamp" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
    "budget" => BUDGET,
    "repeats" => N_REPEATS,
    "tasks" => TASKS,
    "arms" => ARM_NAMES,
    "artifact_root" => ARTIFACT_ROOT,
    "pretrain_checkpoint" => CHECKPOINT_PATH,
    "train_config" => train_config,
    "dataset_stats" => dataset.stats,
    "training_history" => history,
    "policy_checkpoint" => policy_ckpt,
    "summary_rows" => summary_rows,
    "deltas" => deltas,
    "verdict" => verdict,
    "mean_auc_delta" => mean_auc_delta,
    "mean_top10_delta" => mean_top10_delta,
    "mean_he_top10" => mean_he_top10,
    "mean_override_frontier_delta" => mean_override_frontier_delta,
    "all_results" => all_results,
))
logmsg("Saved results: $results_file")
logmsg("Done.")
