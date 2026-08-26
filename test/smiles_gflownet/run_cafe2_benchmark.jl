#!/usr/bin/env julia
# CAFE-GFN 2.0 Benchmark: 8-Task PMO Evaluation
#
# Tests the complete framework:
# - TB v10 with gradient surgery stack
# - Graph GA (Jensen 2019) between segments
# - Scaffold-aware reward shaping
# - Adaptive β via novelty rate
# - SMILES augmentation
# - Rank-based replay (ratio=8)
#
# Compares against:
# - Previous baseline (no Graph GA, no scaffold-aware): ~10.46 projected AUC
# - SOTA Genetic GFN: 16.213

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GFlowNet

# --- Log file for monitoring (bypasses Julia stdout buffering) ---
const LOGFILE = joinpath(@__DIR__, "..", "..", "checkpoints", "cafe2_benchmark", "benchmark.log")
mkpath(dirname(LOGFILE))
function logmsg(msg::String)
    open(LOGFILE, "a") do f
        println(f, msg)
    end
    println(msg)
    flush(stdout)
end

# Include SMILES GFlowNet + Oracle infrastructure
include(joinpath(@__DIR__, "..", "..", "src", "applications", "smiles_gflownet.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "rdkit_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "oracle_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "oracle_manager.jl"))

# Include PMO benchmark runner
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "pmo_benchmark.jl"))

using Serialization
using Statistics: mean

logmsg("=" ^ 70)
logmsg("CAFE-GFN 2.0 — 8-Task PMO Benchmark")
logmsg("=" ^ 70)

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

println("Checkpoint loaded (vocab_size=$actual_vocab_size)")

# --- 8 key tasks ---
# Covers smooth (QED), sparse (DRD2, JNK3, GSK3β), and structural (similarity)
TASKS = [
    "qed",
    "drd2",
    "jnk3",
    "gsk3b",
    "albuterol_similarity",
    "mestranol_similarity",
    "celecoxib_rediscovery",
    "thiothixene_rediscovery",
]

# Task → target SMILES mapping for structural tasks
TARGET_SMILES = Dict(
    "albuterol_similarity" => "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "mestranol_similarity" => "C#CC1(O)CCC2C3CCc4cc(OC)ccc4C3CCC21C",
    "celecoxib_rediscovery" => "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "thiothixene_rediscovery" => "O=C(C1CCN(CCO)CC1)c1ccc(C(F)(F)F)c(I)c1",
)

# --- Configuration ---
BUDGET = 3000  # 3K budget for faster iteration

println("\nConfiguration:")
println("  Budget: $BUDGET per task")
println("  Training mode: TB v10 (validated defaults)")
println("  LR: 3e-5, β: 8.0, KL: 0.01, lr_z: 3e-4")
println("  Batch: 32, Replay ratio: 4")
println("  Graph GA: ENABLED (between segments)")
println("  Augmentation: ENABLED (between segments)")
flush(stdout)

# --- Run benchmark ---
results = Dict{String, Any}()
total_start = time()

for (i, task) in enumerate(TASKS)
    logmsg("\n" * "=" ^ 70)
    logmsg("[$i/$(length(TASKS))] Task: $task")
    logmsg("=" ^ 70)

    target_smi = get(TARGET_SMILES, task, nothing)
    task_start = time()

    result = run_smiles_pmo_task(task;
        budget=BUDGET,
        pretrained_params=pretrained_params,
        pretrained_states=pretrained_states,
        vocab=vocab,
        policy_model=policy_model,
        training_mode=:tb,
        # --- Training settings (validated defaults from PMO baseline QED=0.900) ---
        # learning_rate=3e-5 (default), reward_exponent=8.0 (default),
        # kl_weight=0.01 (default), lr_z=lr*10=3e-4 (default),
        # batch_size=32 (default), replay_ratio=4 (default)
        constructive_only=true,
        reward_weighted=true,
        use_replay=true,
        use_augmentation=true,
        # --- CAFE-GFN 2.0: Graph GA search between segments ---
        use_graph_ga=true,
        target_smiles=target_smi,
        # --- Misc ---
        verbose=true,
    )

    results[task] = result
    elapsed = round(time() - task_start, digits=1)
    logmsg(">>> $task: AUC=$(round(result.auc_top10, digits=4)), Top1=$(round(result.top1, digits=4)), Top10=$(round(result.top10_mean, digits=4)) [$(elapsed)s]")
end

# --- Summary ---
total_elapsed = round(time() - total_start, digits=1)
logmsg("\n" * "=" ^ 70)
logmsg("CAFE-GFN 2.0 — BENCHMARK RESULTS ($(total_elapsed)s)")
logmsg("=" ^ 70)
logmsg("")

total_auc = 0.0
baseline_scores = Dict(
    "qed" => 0.900, "drd2" => 0.780, "jnk3" => 0.107, "gsk3b" => 0.342,
    "albuterol_similarity" => 0.559, "mestranol_similarity" => 0.388,
    "celecoxib_rediscovery" => 0.306, "thiothixene_rediscovery" => 0.255,
)

logmsg("Task                     | AUC    | Top1   | Top10  | Baseline | Δ")
logmsg("-" ^ 75)
for task in TASKS
    r = results[task]
    base = get(baseline_scores, task, 0.0)
    delta = r.auc_top10 - base
    delta_str = delta >= 0 ? "+$(round(delta, digits=3))" : "$(round(delta, digits=3))"
    logmsg("$(rpad(task, 25))| $(lpad(round(r.auc_top10, digits=4), 6)) | $(lpad(round(r.top1, digits=4), 6)) | $(lpad(round(r.top10_mean, digits=4), 6)) | $(lpad(round(base, digits=3), 8)) | $delta_str")
    total_auc += r.auc_top10
end
logmsg("-" ^ 75)
logmsg("Total (8 tasks):           $(round(total_auc, digits=4))")
logmsg("Per-task average:          $(round(total_auc / length(TASKS), digits=4))")
logmsg("")

# Projection to 23 tasks
baseline_total = sum(values(baseline_scores))
baseline_23 = 10.46  # Previous projected 23-task score
improvement_ratio = total_auc / baseline_total
projected_23 = baseline_23 * improvement_ratio
logmsg("Previous 8-task total:     $(round(baseline_total, digits=3))")
logmsg("Improvement ratio:         $(round(improvement_ratio, digits=3))×")
logmsg("Projected 23-task score:   $(round(projected_23, digits=1))")
logmsg("SOTA (Genetic GFN):        16.213")
logmsg("")

if projected_23 > 16.213
    println("🎯 PROJECTED TO BEAT SOTA!")
elseif projected_23 > 15.0
    println("Close to SOTA — Tier 3 refinements may close the gap")
else
    println("Below target — investigate per-task bottlenecks")
end

# Save results
save_dir = joinpath(@__DIR__, "..", "..", "checkpoints", "cafe2_benchmark")
mkpath(save_dir)
save_path = joinpath(save_dir, "cafe2_8task_results.jls")
serialize(save_path, Dict(
    "results" => results,
    "total_auc" => total_auc,
    "projected_23" => projected_23,
    "config" => Dict(
        "budget" => BUDGET,
        "lr" => 3e-5,
        "beta" => 8.0,
        "lr_z_multiplier" => 10.0,
        "kl" => 0.01,
        "use_graph_ga" => true,
        "use_scaffold_aware" => false,
        "use_augmentation" => true,
        "replay_ratio" => 4,
        "batch_size" => 32,
        "version" => "v3_principled_fix",
    ),
))
println("\nResults saved to: $save_path")
flush(stdout)
