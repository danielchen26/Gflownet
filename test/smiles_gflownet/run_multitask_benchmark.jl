# Multi-Task PMO Benchmark: CAFE-GFN vs SOTA
#
# Representative tasks across difficulty levels:
#   Easy:   qed, albuterol_similarity, celecoxib_rediscovery
#   Medium: drd2, gsk3b, jnk3
#   Hard:   isomers_c7h8n2o2, zaleplon_mpo
#
# Two configs: Baseline (v10) and β-Schedule (with global budget ramping)
# Budget: 3000 per task for speed

using GFlowNet
using Serialization
using Statistics

# Include all SMILES GFlowNet code
include(joinpath(@__DIR__, "..", "..", "src", "applications", "smiles_gflownet.jl"))

# Include Oracle infrastructure
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "rdkit_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "oracle_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "oracle_manager.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "pmo_benchmark.jl"))

println("=" ^ 70)
println("MULTI-TASK PMO BENCHMARK: CAFE-GFN vs SOTA")
println("Budget: 3000 per task | Configs: Baseline + β-Schedule")
println("=" ^ 70)
println()

# Load pretrained checkpoint
println("Loading pretrained checkpoint...")
flush(stdout)
checkpoint = deserialize(joinpath(@__DIR__, "..", "..", "checkpoints", "pretrain", "final.jls"))
pretrained_params = checkpoint["params"]
pretrained_states = checkpoint["states"]
vocab = SMILESVocabulary()
actual_vocab_size = size(pretrained_params.output.layer_2.weight, 1)
policy_model, _, _ = create_smiles_policy(; vocab_size=actual_vocab_size, hidden_dim=512, embed_dim=128, n_layers=3)
println("Loaded. Vocab size: $(vocab.size), Model output: $actual_vocab_size")
flush(stdout)

# PMO tasks (representative selection)
tasks = [
    "qed",                      # Easy: drug-likeness
    "albuterol_similarity",     # Easy: molecular similarity
    "celecoxib_rediscovery",    # Easy: molecular rediscovery
    "drd2",                     # Medium: DRD2 inhibitor prediction
    "gsk3b",                    # Medium: GSK3β inhibitor prediction
    "jnk3",                     # Medium: JNK3 inhibitor prediction
    "mestranol_similarity",     # Easy: similarity
    "thiothixene_rediscovery",  # Easy: rediscovery
]

# Genetic GFN per-task reference scores (from paper Table 1, lexicographic order)
# Tasks #1-23 in the paper. We map by position in alphabetical order.
genetic_gfn_ref = Dict(
    "albuterol_similarity" => 0.949,    # #1
    "celecoxib_rediscovery" => 0.837,   # #3
    "drd2" => 0.974,                    # #5
    "gsk3b" => 0.960,                   # #7
    "jnk3" => 0.780,                    # #10
    "mestranol_similarity" => 0.720,    # #13
    "qed" => 0.948,                     # #16
    "thiothixene_rediscovery" => 0.660, # #20
)

configs = [
    ("Baseline", :none, false),
    ("beta-Schedule", :linear_ramp, false),
]

all_results = Dict{String, Vector{Tuple{String, Any}}}()

for (config_name, beta_sched, delta_pri) in configs
    println("\n" * "=" ^ 70)
    println("CONFIG: $config_name")
    println("=" ^ 70)
    flush(stdout)

    task_results = []

    for task in tasks
        println("\n--- Task: $task ($config_name) ---")
        flush(stdout)

        try
            r = run_smiles_pmo_task(task;
                budget=3000,
                pretrained_params=deepcopy(pretrained_params),
                pretrained_states=deepcopy(pretrained_states),
                vocab=vocab,
                policy_model=policy_model,
                training_mode=:tb,
                use_replay=true,
                replay_ratio=2,
                batch_size=32,
                verbose=true,
                beta_schedule=beta_sched,
                beta_start=0.0,
                beta_end=8.0,
                delta_priority_replay=delta_pri,
            )
            push!(task_results, (task, r))
            ref = get(genetic_gfn_ref, task, 0.0)
            delta = r.auc_top10 - ref
            delta_str = delta >= 0 ? "+$(round(delta, digits=3))" : "$(round(delta, digits=3))"
            println(">>> [$config_name] $task: AUC=$(round(r.auc_top10, digits=4)), Top1=$(round(r.top1, digits=4)), Top10=$(round(r.top10_mean, digits=4)) | GenGFN=$(round(ref, digits=3)) Δ=$delta_str")
        catch e
            println(">>> [$config_name] $task: FAILED ($e)")
            push!(task_results, (task, nothing))
        end
        flush(stdout)
    end

    all_results[config_name] = task_results
end

# Final comparison
println("\n\n" * "=" ^ 70)
println("FINAL MULTI-TASK PMO COMPARISON")
println("=" ^ 70)
println()

for (config_name, _...) in configs
    task_results = all_results[config_name]
    println("\n--- $config_name ---")
    println("Task                      CAFE-GFN   GenGFN   Delta")
    println("-" ^ 60)
    sum_auc = 0.0
    sum_ref = 0.0
    n_valid = 0
    for (task, r) in task_results
        if r !== nothing
            ref = get(genetic_gfn_ref, task, 0.0)
            delta = r.auc_top10 - ref
            delta_str = delta >= 0 ? "+$(round(delta, digits=3))" : "$(round(delta, digits=3))"
            println("$(rpad(task, 26))$(lpad(string(round(r.auc_top10, digits=4)), 8))  $(lpad(string(round(ref, digits=3)), 7))  $(lpad(delta_str, 7))")
            sum_auc += r.auc_top10
            sum_ref += ref
            n_valid += 1
        else
            println("$(rpad(task, 26))  FAILED")
        end
    end
    println("-" ^ 60)
    if n_valid > 0
        per_task_avg = sum_auc / n_valid
        projected_23 = per_task_avg * 23
        println("Sum ($n_valid tasks):        $(round(sum_auc, digits=4))  $(round(sum_ref, digits=3))")
        println("Per-task average:        $(round(per_task_avg, digits=4))  $(round(sum_ref/n_valid, digits=3))")
        println("Projected 23-task sum:   $(round(projected_23, digits=2))")
        println("Genetic GFN 23-task:     16.213")
        if projected_23 > 16.213
            println("*** PROJECTED TO BEAT SOTA by $(round(projected_23 - 16.213, digits=2)) ***")
        else
            println("Gap to SOTA: $(round(16.213 - projected_23, digits=2))")
        end
    end
end

println()

# Save results
mkpath(joinpath(@__DIR__, "..", "..", "checkpoints", "novel_directions"))
serialize(joinpath(@__DIR__, "..", "..", "checkpoints", "novel_directions", "multitask_comparison.jls"), all_results)
println("Results saved to checkpoints/novel_directions/multitask_comparison.jls")
flush(stdout)
