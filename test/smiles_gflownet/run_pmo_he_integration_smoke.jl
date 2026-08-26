#!/usr/bin/env julia
# PMO + Hierarchical Edit Integration Smoke Test
#
# Stage B validation: verify that HE warmup + per-segment HE episodes
# run end-to-end inside the PMO pipeline without errors.
#
# Usage:
#   julia test/smiles_gflownet/run_pmo_he_integration_smoke.jl
#   PMO_BUDGET=1024 PMO_TASKS=albuterol_similarity julia test/smiles_gflownet/run_pmo_he_integration_smoke.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GFlowNet
using Serialization
using Statistics: mean

include(joinpath(@__DIR__, "..", "..", "src", "applications", "smiles_gflownet.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "rdkit_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "oracle_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "oracle_manager.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "pmo_benchmark.jl"))

const TARGET_SMILES = Dict(
    "albuterol_similarity" => "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "celecoxib_rediscovery" => "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "mestranol_similarity" => "C#C[C@]1(O)CC[C@H]2[C@@H]3CCc4cc(OC)ccc4[C@H]3CC[C@@]21C",
    "thiothixene_rediscovery" => "C=C(c1ccc(S(=O)(=O)N2CCN(C)CC2)cc1)c1cc2c(s1)Cc1ccccc1-2",
)

const DEFAULT_TASKS = ["celecoxib_rediscovery"]
const BUDGET = parse(Int, get(ENV, "PMO_BUDGET", "64"))
const N_ITERS = parse(Int, get(ENV, "PMO_ITERS", "3"))
const BATCH_SZ = parse(Int, get(ENV, "PMO_BATCH", "8"))

function parse_tasks()
    raw = strip(get(ENV, "PMO_TASKS", ""))
    isempty(raw) && return DEFAULT_TASKS
    return [String(strip(x)) for x in split(raw, ',') if !isempty(strip(x))]
end

const TASKS = parse_tasks()

# --- Load pretrained checkpoint ---
checkpoint_path = joinpath(@__DIR__, "..", "..", "checkpoints", "pretrain", "final.jls")
if !isfile(checkpoint_path)
    error("Pretrained checkpoint not found: $checkpoint_path")
end

checkpoint = deserialize(checkpoint_path)
pretrained_params = checkpoint["params"]
pretrained_states = checkpoint["states"]
vocab = SMILESVocabulary()
actual_vocab_size = size(pretrained_params.output.layer_2.weight, 1)
policy_model, _, _ = create_smiles_policy(; vocab_size=actual_vocab_size, hidden_dim=512, embed_dim=128, n_layers=3)

println("=" ^ 70)
println("PMO + HE INTEGRATION SMOKE TEST")
println("=" ^ 70)
println("Budget: $BUDGET | Iters: $N_ITERS | Batch: $BATCH_SZ | Tasks: $(join(TASKS, ", "))")
println("HE warmup: 2 episodes | HE per-segment: 1 episode | HE budget fraction: 15%")
flush(stdout)

# --- Run configs: baseline (no HE) and with HE ---
configs = [
    ("baseline", false),
    ("pmo_he", true),
]

all_results = Dict{String, Any}()

for (config_name, use_he) in configs
    println("\n" * "=" ^ 70)
    println("CONFIG: $config_name (HE=$(use_he))")
    println("=" ^ 70)
    flush(stdout)

    for task in TASKS
        target_smi = get(TARGET_SMILES, task, nothing)
        println("\n--- $task ($config_name) ---")
        flush(stdout)
        start_t = time()

        he_config = use_he ? HierarchicalEditConfig(;
            horizon=8,
            topk_tracking=10,
            allow_crossover=true,
            min_exploration_per_operator=3,
            multi_child_min_reward_ratio=0.2,
            operator_prior_strength=4.0,
        ) : HierarchicalEditConfig()

        result = run_smiles_pmo_task(task;
            budget=BUDGET,
            pretrained_params=deepcopy(pretrained_params),
            pretrained_states=deepcopy(pretrained_states),
            vocab=vocab,
            policy_model=policy_model,
            training_mode=:tb,
            use_replay=true,
            replay_ratio=2,
            batch_size=BATCH_SZ,
            n_iterations=N_ITERS,
            ga_per_step=false,
            track_frontier=true,
            target_smiles=target_smi,
            target_seed=!isnothing(target_smi),
            target_seed_augmentations=4,
            verbose=true,
            # HE integration
            use_hierarchical_edit=use_he,
            he_warmup_episodes=use_he ? 2 : 0,
            he_episodes_per_segment=use_he ? 1 : 0,
            he_budget_fraction=0.15,
            he_config=he_config,
        )

        elapsed = round(time() - start_t, digits=1)
        println("$config_name [$task] AUC=$(round(result.auc_top10, digits=4)) | Top1=$(round(result.top1, digits=4)) | Top10=$(round(result.top10_mean, digits=4)) | Calls=$(result.n_oracle_calls) | $(elapsed)s")
        flush(stdout)

        key = "$(config_name)_$(task)"
        all_results[key] = result
    end
end

# --- Summary comparison ---
println("\n" * "=" ^ 70)
println("COMPARISON SUMMARY (budget=$BUDGET)")
println("=" ^ 70)
println(rpad("Task", 28) * rpad("Config", 12) * rpad("AUC", 10) * rpad("Top1", 10) * rpad("Top10", 10) * "Calls")
println("-" ^ 80)

for task in TASKS
    for (config_name, _) in configs
        key = "$(config_name)_$(task)"
        r = all_results[key]
        println(rpad(task, 28) * rpad(config_name, 12) * rpad(round(r.auc_top10, digits=4), 10) * rpad(round(r.top1, digits=4), 10) * rpad(round(r.top10_mean, digits=4), 10) * string(r.n_oracle_calls))
    end
end

# --- Delta summary ---
println("\n" * "-" ^ 70)
println("HE DELTAS:")
for task in TASKS
    base = all_results["baseline_$(task)"]
    he = all_results["pmo_he_$(task)"]
    δ_auc = round(he.auc_top10 - base.auc_top10, digits=4)
    δ_top1 = round(he.top1 - base.top1, digits=4)
    δ_str = δ_auc >= 0 ? "+$δ_auc" : "$δ_auc"
    println("  $task: AUC Δ=$δ_str | Top1 Δ=$(δ_top1 >= 0 ? "+" : "")$δ_top1")
end

# Save results
outdir = joinpath(@__DIR__, "..", "..", "checkpoints", "pmo_he_integration")
mkpath(outdir)
serialize(joinpath(outdir, "smoke_test_results.jls"), all_results)
println("\nResults saved to $outdir/smoke_test_results.jls")
flush(stdout)
