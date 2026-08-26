# QED PMO Benchmark (FAST): Compare Novel Features vs Baseline
# Budget: 3000 (30% of standard PMO) — enough for relative comparison
# replay_ratio=2 for speed

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
println("QED PMO BENCHMARK (FAST): Comparing Novel Features vs Baseline")
println("Budget: 3000 | replay_ratio: 2")
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

configs = [
    ("Baseline (v10)", :none, false),
    ("beta-Schedule (0->8)", :linear_ramp, false),
    ("|delta|-Replay", :none, true),
    ("Combined (A+B)", :linear_ramp, true),
]

results = []
for (name, beta_sched, delta_pri) in configs
    println("\n" * "-" ^ 70)
    println("CONFIG: $name")
    println("-" ^ 70)
    flush(stdout)

    r = run_smiles_pmo_task("qed";
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
    push!(results, (name, r))
    println(">>> $name AUC=$(round(r.auc_top10, digits=4)), Top1=$(round(r.top1, digits=4)), Top10=$(round(r.top10_mean, digits=4)), Unique=$(r.unique_molecules)")
    flush(stdout)
end

# Final comparison
println("\n\n" * "=" ^ 70)
println("FINAL COMPARISON: QED PMO (3K budget, fast mode)")
println("=" ^ 70)
println()
println("Config                     AUC      Top1     Top10    Unique  Diversity")
println("-" ^ 70)
for (name, r) in results
    println("$(rpad(name, 27))$(lpad(string(round(r.auc_top10, digits=4)), 7))  $(lpad(string(round(r.top1, digits=4)), 7))  $(lpad(string(round(r.top10_mean, digits=4)), 7))  $(lpad(string(r.unique_molecules), 7))  $(lpad(string(round(r.diversity, digits=4)), 7))")
end
println()
println("SOTA Reference (Genetic GFN per-task avg): ~0.704")
println("Our previous best (TB v10, 10K): 0.9355")
println("Our previous RWMLE (10K): 0.9373")

best = results[argmax([c[2].auc_top10 for c in results])]
println("\nBEST: $(best[1]) AUC = $(round(best[2].auc_top10, digits=4))")

mkpath(joinpath(@__DIR__, "..", "..", "checkpoints", "novel_directions"))
serialize(joinpath(@__DIR__, "..", "..", "checkpoints", "novel_directions", "qed_comparison_fast.jls"), results)
println("Results saved to checkpoints/novel_directions/qed_comparison_fast.jls")
flush(stdout)
