#!/usr/bin/env julia
# CAFE-GFN v3 Principled Fix — Phase 1 Validation
#
# Tests 3 key tasks (QED, DRD2, GSK3b) with principled fixes:
#   A) Unified raw reward storage (fixes double-exponentiation R^β² → R^β)
#   B) Additive GA composition (BRICS GA fallback when Graph GA fails)
#   C) Correct config metadata
#
# Expected: match or beat baseline (QED≈0.900, DRD2≈0.780, GSK3b≈0.342)

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GFlowNet

# --- Logging ---
const LOGDIR = joinpath(@__DIR__, "..", "..", "checkpoints", "v3_validation")
mkpath(LOGDIR)
const LOGFILE = joinpath(LOGDIR, "validation.log")

function logmsg(msg::String)
    open(LOGFILE, "a") do f
        println(f, msg)
    end
    println(msg)
    flush(stdout)
end

# Include infrastructure
include(joinpath(@__DIR__, "..", "..", "src", "applications", "smiles_gflownet.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "rdkit_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "oracle_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "oracle_manager.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "pmo_benchmark.jl"))

using Serialization
using Statistics: mean

logmsg("=" ^ 70)
logmsg("CAFE-GFN v3 — Principled Fix Validation (3 tasks)")
logmsg("=" ^ 70)
logmsg("Fixes applied:")
logmsg("  A) Unified raw reward storage (no double-exponentiation)")
logmsg("  B) Additive GA composition (BRICS fallback)")
logmsg("  C) Correct config metadata")

# --- Load pretrained checkpoint ---
checkpoint_path = joinpath(@__DIR__, "..", "..", "checkpoints", "pretrain", "final.jls")
if !isfile(checkpoint_path)
    error("Pretrained checkpoint not found: $checkpoint_path")
end

println("Loading checkpoint: $checkpoint_path")
checkpoint = deserialize(checkpoint_path)
pretrained_params = checkpoint["params"]
pretrained_states = checkpoint["states"]
vocab = SMILESVocabulary()

actual_vocab_size = size(pretrained_params.output.layer_2.weight, 1)
policy_model, _, _ = create_smiles_policy(;
    vocab_size=actual_vocab_size, hidden_dim=512, embed_dim=128, n_layers=3)

logmsg("Checkpoint loaded (vocab_size=$actual_vocab_size)")

# --- 3 key tasks covering smooth, sparse, and structural ---
TASKS = ["qed", "drd2", "gsk3b"]
BUDGET = 3000

# Baselines for comparison (from previous PMO benchmark)
BASELINES = Dict(
    "qed" => 0.900,
    "drd2" => 0.780,
    "gsk3b" => 0.342,
)

# v2.0 buggy results for comparison
V2_BUGGY = Dict(
    "qed" => 0.886,
    "drd2" => 0.625,
    "gsk3b" => 0.188,
)

logmsg("\nConfiguration (validated baseline defaults):")
logmsg("  Budget: $BUDGET per task")
logmsg("  Training mode: TB v10")
logmsg("  LR: 3e-5, β: 8.0, KL: 0.01, lr_z: 3e-4")
logmsg("  Batch: 32, Replay ratio: 4")
logmsg("  Graph GA: ENABLED (with BRICS fallback)")
logmsg("  Augmentation: ENABLED")
flush(stdout)

# --- Run validation ---
results = Dict{String, Any}()
total_start = time()

for (i, task) in enumerate(TASKS)
    logmsg("\n" * "=" ^ 70)
    logmsg("[$i/$(length(TASKS))] Task: $task")
    logmsg("=" ^ 70)

    task_start = time()

    result = run_smiles_pmo_task(task;
        budget=BUDGET,
        pretrained_params=pretrained_params,
        pretrained_states=pretrained_states,
        vocab=vocab,
        policy_model=policy_model,
        training_mode=:tb,
        constructive_only=true,
        reward_weighted=true,
        use_replay=true,
        use_augmentation=true,
        use_graph_ga=true,
        verbose=true,
    )

    results[task] = result
    elapsed = round(time() - task_start, digits=1)

    baseline = get(BASELINES, task, 0.0)
    buggy = get(V2_BUGGY, task, 0.0)
    auc = round(result.auc_top10, digits=4)
    delta_base = round(auc - baseline, digits=4)
    delta_bug = round(auc - buggy, digits=4)

    logmsg(">>> $task: AUC=$auc (vs baseline=$baseline Δ=$(delta_base > 0 ? "+" : "")$delta_base, vs buggy=$buggy Δ=$(delta_bug > 0 ? "+" : "")$delta_bug) [$(elapsed)s]")
end

# --- Summary ---
total_elapsed = round(time() - total_start, digits=1)
logmsg("\n" * "=" ^ 70)
logmsg("V3 PRINCIPLED FIX — VALIDATION RESULTS ($total_elapsed s)")
logmsg("=" ^ 70)

total_auc = 0.0
for task in TASKS
    r = results[task]
    auc = round(r.auc_top10, digits=4)
    baseline = get(BASELINES, task, 0.0)
    buggy = get(V2_BUGGY, task, 0.0)
    total_auc += auc
    logmsg("  $task: AUC=$auc  Top1=$(round(r.top1, digits=3))  (baseline=$baseline, buggy=$buggy)")
end
logmsg("\n  3-task total AUC: $(round(total_auc, digits=4))")
logmsg("  3-task baseline:  $(round(sum(get(BASELINES, t, 0.0) for t in TASKS), digits=4))")
logmsg("  3-task buggy:     $(round(sum(get(V2_BUGGY, t, 0.0) for t in TASKS), digits=4))")

# Save results
save_path = joinpath(LOGDIR, "v3_validation_results.jls")
serialize(save_path, Dict(
    "results" => results,
    "total_auc" => total_auc,
    "config" => Dict(
        "budget" => BUDGET,
        "version" => "v3_principled_fix",
        "fixes" => ["raw_reward_storage", "additive_ga", "correct_metadata"],
    ),
))
logmsg("\nResults saved to: $save_path")
flush(stdout)
