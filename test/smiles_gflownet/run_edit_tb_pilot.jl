# ============================================================================
# Edit-TB Pilot Evaluation Runner
# ============================================================================
#
# Staged execution (per critique):
#   Step 1: Tiny online smoke (2 tasks, 1 repeat, budget 256)
#   Step 2: Full 3-task × 5-config matrix (budget 3000)
#
# Pre-registered headline reward: top10_delta
# Pre-registered heuristic_tuned config (NOT tuned after seeing results):
#   operator_prior_strength=6.0, min_exploration_per_operator=8
#
# Usage:
#   EDIT_TB_MODE=smoke julia --project=. test/smiles_gflownet/run_edit_tb_pilot.jl
#   EDIT_TB_MODE=full  julia --project=. test/smiles_gflownet/run_edit_tb_pilot.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GFlowNet
using Random
using Serialization
using Statistics
using Dates

# PMO benchmark infrastructure (not exported from GFlowNet)
include(joinpath(@__DIR__, "..", "..", "src", "applications", "smiles_gflownet.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "rdkit_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "oracle_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "oracle_manager.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "pmo_benchmark.jl"))

# ============================================================================
# Environment Configuration
# ============================================================================

const MODE = get(ENV, "EDIT_TB_MODE", "smoke")  # "smoke" or "full"
const BUDGET = parse(Int, get(ENV, "EDIT_TB_BUDGET", MODE == "smoke" ? "256" : "3000"))
const N_REPEATS = parse(Int, get(ENV, "EDIT_TB_REPEATS", MODE == "smoke" ? "1" : "3"))
const ARTIFACT_ROOT = get(ENV, "EDIT_TB_ARTIFACT_ROOT",
    joinpath(@__DIR__, "..", "..", "checkpoints", "truth_sprint_stage_b_f015_truth_tasksharded"))
const LOGDIR = get(ENV, "EDIT_TB_LOGDIR",
    joinpath(@__DIR__, "..", "..", "checkpoints", "edit_tb_pilot_$(MODE)"))
const EDIT_TB_EPOCHS = parse(Int, get(ENV, "EDIT_TB_EPOCHS", "50"))

# Smoke: celecoxib + drd2 only; Full: add albuterol
const TASKS = if MODE == "smoke"
    ["celecoxib_rediscovery", "drd2"]
else
    ["celecoxib_rediscovery", "drd2", "albuterol_similarity"]
end

# Smoke: 3 configs; Full: 5 configs
const CONFIGS_TO_RUN = if MODE == "smoke"
    ["heuristic_baseline", "learned_rwmle", "interpolated_050"]
else
    ["heuristic_baseline", "learned_rwmle", "learned_rwmle_no_heuristic",
     "interpolated_050", "heuristic_tuned"]
end

# HE standard parameters (match truth sprint)
const HE_BUDGET_FRACTION = 0.15
const HE_WARMUP_EPISODES = parse(Int, get(ENV, "HE_WARMUP_EPISODES", "4"))
const HE_EPISODES_PER_SEGMENT = parse(Int, get(ENV, "HE_EPISODES_PER_SEGMENT", "2"))
const HE_HORIZON = 3
const HE_MAX_STEP_ATTEMPTS = 3
const FRONTIER_BOOTSTRAP_SAMPLES = 32

# PMO training params (match truth sprint, with env var overrides for quick smoke)
const PMO_BATCH_SIZE = parse(Int, get(ENV, "PMO_BATCH_SIZE", "64"))
const PMO_REPLAY_RATIO = parse(Int, get(ENV, "PMO_REPLAY_RATIO", "8"))
const PMO_N_ITERATIONS = parse(Int, get(ENV, "PMO_N_ITERATIONS", "25"))
const PMO_GA_PER_STEP = true
const PMO_GA_CROSSOVER = parse(Int, get(ENV, "PMO_GA_CROSSOVER", "4"))
const PMO_GA_MUTATION = parse(Int, get(ENV, "PMO_GA_MUTATION", "4"))

# Target molecules for seeded tasks
const TARGET_SMILES = Dict(
    "albuterol_similarity" => "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "celecoxib_rediscovery" => "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
)

logmsg(s) = println("[$(Dates.format(now(), "HH:MM:SS"))] $s")

# ============================================================================
# Load Pretrained Model
# ============================================================================

checkpoint_path = joinpath(@__DIR__, "..", "..", "checkpoints", "pretrain", "final.jls")
if !isfile(checkpoint_path)
    error("Pretrained checkpoint not found: $checkpoint_path")
end

logmsg("Loading pretrained checkpoint: $checkpoint_path")
checkpoint = deserialize(checkpoint_path)
const PRETRAINED_PARAMS = checkpoint["params"]
const PRETRAINED_STATES = checkpoint["states"]
const VOCAB = SMILESVocabulary()
actual_vocab_size = size(PRETRAINED_PARAMS.output.layer_2.weight, 1)
const POLICY_MODEL, _, _ = create_smiles_policy(; vocab_size=actual_vocab_size, hidden_dim=512, embed_dim=128, n_layers=3)
logmsg("Checkpoint loaded (vocab_size=$actual_vocab_size)")

# ============================================================================
# Train Edit-TB Policy (offline, before any online evaluation)
# ============================================================================

logmsg("Loading HE artifacts from: $ARTIFACT_ROOT")
if !isdir(ARTIFACT_ROOT)
    error("Artifact root not found: $ARTIFACT_ROOT")
end

# Train with heuristic scores (for learned_rwmle and interpolated configs)
logmsg("Training edit-TB policy (with heuristic features)...")
rng = Random.MersenneTwister(42)
config_h = EditTBConfig(n_epochs=EDIT_TB_EPOCHS, reward_mode=:top10_delta,
                        include_heuristic_scores=true, heuristic_interpolation=0.5)
model_h, params_h, states_h = init_edit_policy(rng; config=config_h)
dataset_h = load_edit_tb_dataset(ARTIFACT_ROOT; config=config_h)
logmsg("Dataset: $(dataset_h.stats["n_trajectories"]) trajectories, $(dataset_h.stats["n_steps"]) steps")
best_params_h, best_log_Z_h, history_h = train_edit_policy!(
    model_h, params_h, states_h, dataset_h, config_h;
    verbose=true, rng=Random.MersenneTwister(123)
)
logmsg("With-heuristic training done. Best val=$(round(minimum(history_h["val_loss"]), digits=4))")

# Train without heuristic scores (for no_heuristic ablation)
trained_no_h = nothing
if "learned_rwmle_no_heuristic" in CONFIGS_TO_RUN
    logmsg("Training edit-TB policy (without heuristic features)...")
    config_no_h = EditTBConfig(n_epochs=EDIT_TB_EPOCHS, reward_mode=:top10_delta,
                               include_heuristic_scores=false, heuristic_interpolation=1.0)
    model_no_h, params_no_h, states_no_h = init_edit_policy(Random.MersenneTwister(42); config=config_no_h)
    dataset_no_h = load_edit_tb_dataset(ARTIFACT_ROOT; config=config_no_h)
    best_params_no_h, best_log_Z_no_h, history_no_h = train_edit_policy!(
        model_no_h, params_no_h, states_no_h, dataset_no_h, config_no_h;
        verbose=true, rng=Random.MersenneTwister(123)
    )
    trained_no_h = (model=model_no_h, params=best_params_no_h, states=states_no_h, config=config_no_h)
    logmsg("No-heuristic training done. Best val=$(round(minimum(history_no_h["val_loss"]), digits=4))")
end

# ============================================================================
# Config Builders
# ============================================================================

function build_base_he_config()
    HierarchicalEditConfig(;
        horizon=HE_HORIZON,
        allow_crossover=true,
        allow_fragment_ops=false,
        max_step_attempts=HE_MAX_STEP_ATTEMPTS,
        min_exploration_per_operator=5,
        multi_child_min_reward_ratio=0.2,
        operator_prior_strength=4.0,
        use_learned_basin=false,
        use_learned_parent=false,
        use_learned_operator=false,
    )
end

function build_learned_he_config(controller::EditTBPolicyController)
    HierarchicalEditConfig(;
        horizon=HE_HORIZON,
        allow_crossover=true,
        allow_fragment_ops=false,
        max_step_attempts=HE_MAX_STEP_ATTEMPTS,
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

# PRE-REGISTERED heuristic_tuned: more exploration, stronger prior
# NOT tuned after seeing pilot results.
function build_tuned_he_config()
    HierarchicalEditConfig(;
        horizon=HE_HORIZON,
        allow_crossover=true,
        allow_fragment_ops=false,
        max_step_attempts=HE_MAX_STEP_ATTEMPTS,
        min_exploration_per_operator=8,  # more exploration
        multi_child_min_reward_ratio=0.15,  # slightly more permissive
        operator_prior_strength=6.0,  # stronger prior
        use_learned_basin=false,
        use_learned_parent=false,
        use_learned_operator=false,
    )
end

function build_config(config_name::String)
    if config_name == "heuristic_baseline"
        return (
            he_config=build_base_he_config(),
            label="Heuristic baseline",
        )
    elseif config_name == "learned_rwmle"
        # λ=0.5 interpolation with heuristic features
        edit_config = EditTBConfig(include_heuristic_scores=true, heuristic_interpolation=0.5,
                                   entropy_floor=0.05)
        controller = EditTBPolicyController(model_h, best_params_h, states_h, edit_config)
        return (
            he_config=build_learned_he_config(controller),
            label="Learned RWMLE (λ=0.5)",
        )
    elseif config_name == "learned_rwmle_no_heuristic"
        if trained_no_h === nothing
            error("No-heuristic model not trained")
        end
        edit_config = EditTBConfig(include_heuristic_scores=false, heuristic_interpolation=1.0,
                                   entropy_floor=0.05)
        controller = EditTBPolicyController(trained_no_h.model, trained_no_h.params,
                                            trained_no_h.states, edit_config)
        return (
            he_config=build_learned_he_config(controller),
            label="Learned RWMLE (no heuristic)",
        )
    elseif config_name == "interpolated_050"
        # Same model but λ=0.5 (safest deploy candidate)
        edit_config = EditTBConfig(include_heuristic_scores=true, heuristic_interpolation=0.5,
                                   entropy_floor=0.1)
        controller = EditTBPolicyController(model_h, best_params_h, states_h, edit_config)
        return (
            he_config=build_learned_he_config(controller),
            label="Interpolated λ=0.5 (safe)",
        )
    elseif config_name == "heuristic_tuned"
        return (
            he_config=build_tuned_he_config(),
            label="Heuristic tuned (pre-registered)",
        )
    else
        error("Unknown config: $config_name")
    end
end

# ============================================================================
# Run Evaluation
# ============================================================================

mkpath(LOGDIR)
logmsg("=" ^ 60)
logmsg("Edit-TB Pilot Evaluation — MODE=$MODE")
logmsg("Budget=$BUDGET, Repeats=$N_REPEATS, Tasks=$(TASKS)")
logmsg("Configs=$(CONFIGS_TO_RUN)")
logmsg("Logdir=$LOGDIR")
logmsg("=" ^ 60)

all_results = Dict{String, Dict{String, Vector}}()

for config_name in CONFIGS_TO_RUN
    cfg = build_config(config_name)
    logmsg("\n--- Config: $config_name ($(cfg.label)) ---")

    config_results = Dict{String, Vector}()

    for task in TASKS
        task_runs = []
        target_smi = get(TARGET_SMILES, task, nothing)
        use_seed = !isnothing(target_smi)

        for run_idx in 1:N_REPEATS
            logmsg("  $task run $run_idx/$N_REPEATS ...")
            start_t = time()

            run_artifact_dir = joinpath(LOGDIR, "artifacts", config_name, task, "run$(run_idx)")
            mkpath(run_artifact_dir)

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
                ga_per_step=PMO_GA_PER_STEP,
                ga_crossover=PMO_GA_CROSSOVER,
                ga_mutation=PMO_GA_MUTATION,
                track_frontier=true,
                frontier_bootstrap_samples=FRONTIER_BOOTSTRAP_SAMPLES,
                frontier_bootstrap_min_entries=2,
                target_smiles=target_smi,
                target_seed=use_seed,
                target_seed_augmentations=8,
                verbose=false,
                use_hierarchical_edit=true,
                he_warmup_episodes=HE_WARMUP_EPISODES,
                he_episodes_per_segment=HE_EPISODES_PER_SEGMENT,
                he_budget_fraction=HE_BUDGET_FRACTION,
                he_config=cfg.he_config,
                he_artifact_dir=run_artifact_dir,
                he_run_context=Dict{String,Any}(
                    "task_name" => task,
                    "config_name" => config_name,
                    "run_index" => run_idx,
                ),
                allow_learned_he=(config_name != "heuristic_baseline" && config_name != "heuristic_tuned"),
            )

            push!(task_runs, result)
            elapsed = round(time() - start_t, digits=1)

            breakdown = result.oracle_call_breakdown
            he_w = get(breakdown, "he_warmup", 0)
            he_i = get(breakdown, "he_interleaved", 0)
            topk_fracs = get(result.provenance_summary, "topk_source_fractions", Dict{String,Float64}())
            he_top10 = round(get(topk_fracs, "he", 0.0), digits=3)

            logmsg("    AUC=$(round(result.auc_top10, digits=4)) Top1=$(round(result.top1, digits=4)) " *
                   "Top10=$(round(result.top10_mean, digits=4)) | " *
                   "he_calls=$(he_w+he_i) he_top10=$(he_top10) | $(elapsed)s")
        end

        config_results[task] = task_runs
    end

    all_results[config_name] = config_results
end

# ============================================================================
# Summary Report
# ============================================================================

logmsg("\n" * "=" ^ 60)
logmsg("PILOT RESULTS SUMMARY (headline: top10_delta)")
logmsg("=" ^ 60)

# Collect summary table
summary_rows = []
for config_name in CONFIGS_TO_RUN
    results = all_results[config_name]
    for task in TASKS
        runs = results[task]
        aucs = [r.auc_top10 for r in runs]
        top10s = [r.top10_mean for r in runs]
        top1s = [r.top1 for r in runs]
        push!(summary_rows, (
            config=config_name, task=task,
            auc_mean=mean(aucs), auc_std=N_REPEATS > 1 ? std(aucs) : 0.0,
            top10_mean=mean(top10s), top10_std=N_REPEATS > 1 ? std(top10s) : 0.0,
            top1_mean=mean(top1s),
        ))
    end
end

# Print table
logmsg(rpad("Config", 30) * rpad("Task", 25) * rpad("AUC", 20) * rpad("Top10", 20) * "Top1")
logmsg("-" ^ 95)
for row in summary_rows
    auc_str = "$(round(row.auc_mean, digits=4))"
    if row.auc_std > 0
        auc_str *= " ± $(round(row.auc_std, digits=4))"
    end
    top10_str = "$(round(row.top10_mean, digits=4))"
    if row.top10_std > 0
        top10_str *= " ± $(round(row.top10_std, digits=4))"
    end
    logmsg(rpad(row.config, 30) * rpad(row.task, 25) * rpad(auc_str, 20) * rpad(top10_str, 20) * "$(round(row.top1_mean, digits=4))")
end

# Compute deltas vs heuristic baseline
if "heuristic_baseline" in CONFIGS_TO_RUN
    logmsg("\n--- Deltas vs Heuristic Baseline ---")
    baseline = all_results["heuristic_baseline"]
    for config_name in CONFIGS_TO_RUN
        config_name == "heuristic_baseline" && continue
        results = all_results[config_name]
        logmsg("$config_name:")
        for task in TASKS
            b_top10 = mean(r.top10_mean for r in baseline[task])
            l_top10 = mean(r.top10_mean for r in results[task])
            delta = round(l_top10 - b_top10, digits=4)
            sign = delta >= 0 ? "+" : ""
            logmsg("  $task: ΔTop10 = $sign$delta")
        end
    end
end

# ============================================================================
# Verdict
# ============================================================================

logmsg("\n--- PILOT VERDICT ---")
if "heuristic_baseline" in CONFIGS_TO_RUN
    let baseline = all_results["heuristic_baseline"],
        n_better = 0, n_catastrophic = 0, best_config = "", best_delta = -Inf

        for config_name in CONFIGS_TO_RUN
            config_name == "heuristic_baseline" && continue
            results = all_results[config_name]
            total_delta = 0.0
            for task in TASKS
                b_top10 = mean(r.top10_mean for r in baseline[task])
                l_top10 = mean(r.top10_mean for r in results[task])
                d = l_top10 - b_top10
                total_delta += d
                if d > 0.005
                    n_better += 1
                end
                if d < -0.05
                    n_catastrophic += 1
                end
            end
            if total_delta > best_delta
                best_delta = total_delta
                best_config = config_name
            end
        end

        if n_catastrophic > 0
            logmsg("VERDICT: FAILURE — catastrophic regression detected")
        elseif n_better == 0
            logmsg("VERDICT: NOT ALIVE — no learned config beats heuristic on any task")
        elseif n_better >= 2
            logmsg("VERDICT: WORTH SCALING — learned policy beats heuristic on ≥2 task-config pairs")
            logmsg("Best config: $best_config (total ΔTop10 = $(round(best_delta, digits=4)))")
        else
            logmsg("VERDICT: ALIVE — learned policy shows encouraging behavior")
            logmsg("Best config: $best_config (total ΔTop10 = $(round(best_delta, digits=4)))")
        end
    end
end

# Save results
results_file = joinpath(LOGDIR, "edit_tb_pilot_results.jls")
serialize(results_file, Dict(
    "mode" => MODE,
    "budget" => BUDGET,
    "n_repeats" => N_REPEATS,
    "tasks" => TASKS,
    "configs" => CONFIGS_TO_RUN,
    "all_results" => all_results,
    "training_history_h" => history_h,
    "timestamp" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
))
logmsg("Results saved: $results_file")
logmsg("Done.")
