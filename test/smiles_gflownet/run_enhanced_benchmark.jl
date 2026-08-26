# Enhanced Multi-Task PMO Benchmark v3: GA+Replay Enhancement
#
# Key insight: β=50 fails on hard tasks with 3K budget (reward signal vanishes).
# Instead, keep our proven β=8 but add the GA+replay enhancements:
#   1. Per-step GA: 16 offspring/iter (8 crossover + 8 mutation)  [was: ~0.6/iter]
#   2. Replay intensity: 8× replay ratio                          [was: 2×]
#   3. Learning rate: 1e-4                                         [was: 3e-5]
#   4. log_Z LR multiplier: 100 (giving lr_z = 0.01)             [was: 10 (giving 3e-4)]
#   5. KL: 0.001                                                   [was: 0.01]
#
# This isolates the effect of structural search (GA) + replay intensity
# without confounding with aggressive reward shaping that breaks hard tasks.

using GFlowNet
using Serialization
using Statistics

include(joinpath(@__DIR__, "..", "..", "src", "applications", "smiles_gflownet.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "rdkit_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "oracle_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "oracle_manager.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "pmo_benchmark.jl"))

println("=" ^ 70)
println("ENHANCED MULTI-TASK PMO BENCHMARK (v3)")
println("GA+Replay Enhancement | β=8 (proven) | Budget: 3000")
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

# All 8 representative tasks
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

# Reference scores
genetic_gfn_ref = Dict(
    "albuterol_similarity" => 0.949, "celecoxib_rediscovery" => 0.837,
    "drd2" => 0.974, "gsk3b" => 0.960, "jnk3" => 0.780,
    "mestranol_similarity" => 0.720, "qed" => 0.948, "thiothixene_rediscovery" => 0.660,
)

prev_baseline = Dict(
    "qed" => 0.900, "drd2" => 0.780, "albuterol_similarity" => 0.559,
    "mestranol_similarity" => 0.388, "gsk3b" => 0.342,
    "celecoxib_rediscovery" => 0.306, "thiothixene_rediscovery" => 0.255, "jnk3" => 0.107,
)

println("CONFIG: GA+Replay Enhanced (β=8 proven)")
println("  LR=1e-4, β=8.0, lr_z_mult=100, KL=0.001")
println("  replay=4×, batch=32, GA/step=true (4+4)")
println()
flush(stdout)

task_results = []

for task in tasks
    println("\n" * "-" ^ 70)
    println("Task: $task")
    println("-" ^ 70)
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
            verbose=true,
            # GA+Replay enhancement with proven β=8
            learning_rate=1e-4,
            reward_exponent=8.0,
            lr_z_multiplier=100.0,
            kl_weight=0.001,
            replay_ratio=4,
            batch_size=32,
            ga_per_step=true,
            ga_crossover=4,
            ga_mutation=4,
        )
        push!(task_results, (task, r))
        ref = get(genetic_gfn_ref, task, 0.0)
        prev = get(prev_baseline, task, 0.0)
        println(">>> $task: AUC=$(round(r.auc_top10, digits=4)), Top1=$(round(r.top1, digits=4)), Top10=$(round(r.top10_mean, digits=4))")
        println("    vs GenGFN=$(round(ref, digits=3)) Δ=$(round(r.auc_top10 - ref, digits=3)), vs Prev=$(round(prev, digits=3)) Δ=$(round(r.auc_top10 - prev, digits=3))")
    catch e
        println(">>> $task: FAILED")
        println("    Error: $e")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt; backtrace=true)
            println()
        end
        push!(task_results, (task, nothing))
    end
    flush(stdout)
end

# Final comparison table
println("\n\n" * "=" ^ 70)
println("FINAL ENHANCED MULTI-TASK PMO COMPARISON")
println("=" ^ 70)
println()
println("Task                      Enhanced   PrevBase  GenGFN   Δ(GenGFN)  Δ(Prev)")
println("-" ^ 75)
sum_auc = 0.0; sum_ref = 0.0; sum_prev = 0.0; n_valid = 0
for (task, r) in task_results
    if r !== nothing
        ref = get(genetic_gfn_ref, task, 0.0)
        prev = get(prev_baseline, task, 0.0)
        delta_ref = r.auc_top10 - ref
        delta_prev = r.auc_top10 - prev
        d_ref_str = delta_ref >= 0 ? "+$(round(delta_ref, digits=3))" : "$(round(delta_ref, digits=3))"
        d_prev_str = delta_prev >= 0 ? "+$(round(delta_prev, digits=3))" : "$(round(delta_prev, digits=3))"
        println("$(rpad(task, 26))$(lpad(string(round(r.auc_top10, digits=4)), 8))  $(lpad(string(round(prev, digits=3)), 8))  $(lpad(string(round(ref, digits=3)), 7))   $(lpad(d_ref_str, 7))    $(lpad(d_prev_str, 7))")
        sum_auc += r.auc_top10; sum_ref += ref; sum_prev += prev; n_valid += 1
    else
        println("$(rpad(task, 26))  FAILED")
    end
end
println("-" ^ 75)
if n_valid > 0
    per_task_avg = sum_auc / n_valid
    projected_23 = per_task_avg * 23
    prev_projected = (sum_prev / n_valid) * 23
    println("Sum ($n_valid tasks):        $(round(sum_auc, digits=4))  $(round(sum_prev, digits=3))  $(round(sum_ref, digits=3))")
    println("Per-task average:        $(round(per_task_avg, digits=4))  $(round(sum_prev/n_valid, digits=3))  $(round(sum_ref/n_valid, digits=3))")
    println("Projected 23-task sum:   $(round(projected_23, digits=2))  $(round(prev_projected, digits=2))")
    println("Genetic GFN 23-task:     16.213")
    println("Improvement over prev:   $(round(projected_23 - prev_projected, digits=2))")
    if projected_23 > 16.213
        println("*** PROJECTED TO BEAT SOTA by $(round(projected_23 - 16.213, digits=2)) ***")
    else
        println("Gap to SOTA: $(round(16.213 - projected_23, digits=2))")
    end
end
println()
mkpath(joinpath(@__DIR__, "..", "..", "checkpoints", "novel_directions"))
serialize(joinpath(@__DIR__, "..", "..", "checkpoints", "novel_directions", "enhanced_benchmark_v3.jls"), task_results)
println("Results saved to checkpoints/novel_directions/enhanced_benchmark_v3.jls")
flush(stdout)
