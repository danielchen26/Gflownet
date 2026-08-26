# Tier 1 Grid Search: Training Intensity Optimization
#
# Goal: Find optimal hyperparameters for frozen-GRU TB on QED (3K budget)
#
# Phase 1: LR × constructive_only (fixed β=8, rw=true)
#   LR ∈ {1e-4, 2e-4, 3e-4} × constructive_only ∈ {true, false}
#   → 6 configs
#
# Phase 2: β × reward_weighted (using best LR+constr from Phase 1)
#   β ∈ {1, 2, 4, 8} × reward_weighted (5 configs)
#   → 5 configs
#
# Fixed: KL=0.001, replay_ratio=8, batch=64, lr_z=0.05, log_z_grad_clip=1.0

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
println("TIER 1 GRID SEARCH: Training Intensity on QED (3K budget)")
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

const BUDGET = 3000
const TASK = "qed"

# Common fixed parameters
const FIXED = Dict(
    :kl_weight => 0.001,
    :replay_ratio => 8,
    :batch_size => 64,
    :lr_z => 0.05,
    :log_z_grad_clip => 1.0,
    :warmup_iters => 0,
    :n_iterations => 25,
    :use_replay => true,
)

# ═══════════════════════════════════════════════════════════════════════
# Phase 1: LR × constructive_only (fixed β=8, reward_weighted=true)
# ═══════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("PHASE 1: LR × constructive_only (β=8, rw=true)")
println("=" ^ 70)
flush(stdout)

phase1_configs = [
    (lr=1e-4, constr=true),
    (lr=1e-4, constr=false),
    (lr=2e-4, constr=true),
    (lr=2e-4, constr=false),
    (lr=3e-4, constr=true),
    (lr=3e-4, constr=false),
]

phase1_results = []
for (i, cfg) in enumerate(phase1_configs)
    name = "P1-$(i): lr=$(cfg.lr), constr=$(cfg.constr)"
    println("\n" * "-" ^ 70)
    println("[$i/$(length(phase1_configs))] $name")
    println("-" ^ 70)
    flush(stdout)

    r = run_smiles_pmo_task(TASK;
        budget=BUDGET,
        pretrained_params=pretrained_params,
        pretrained_states=pretrained_states,
        vocab=vocab,
        policy_model=policy_model,
        training_mode=:tb,
        learning_rate=cfg.lr,
        reward_exponent=8.0,
        constructive_only=cfg.constr,
        reward_weighted=true,
        kl_weight=FIXED[:kl_weight],
        use_replay=FIXED[:use_replay],
        replay_ratio=FIXED[:replay_ratio],
        batch_size=FIXED[:batch_size],
        lr_z=FIXED[:lr_z],
        log_z_grad_clip=FIXED[:log_z_grad_clip],
        warmup_iters=FIXED[:warmup_iters],
        n_iterations=FIXED[:n_iterations],
        verbose=true,
    )
    push!(phase1_results, (name=name, cfg=cfg, result=r))
    println(">>> $name → AUC=$(round(r.auc_top10, digits=4)), Top1=$(round(r.top1, digits=4)), Top10=$(round(r.top10_mean, digits=4))")
    flush(stdout)
end

# Phase 1 summary
println("\n" * "=" ^ 70)
println("PHASE 1 RESULTS")
println("=" ^ 70)
println("Config                                  AUC      Top1     Top10    Unique")
println("-" ^ 70)
for p in phase1_results
    println("$(rpad(p.name, 40))$(lpad(string(round(p.result.auc_top10, digits=4)), 7))  $(lpad(string(round(p.result.top1, digits=4)), 7))  $(lpad(string(round(p.result.top10_mean, digits=4)), 7))  $(lpad(string(p.result.unique_molecules), 7))")
end
flush(stdout)

# Find best Phase 1 config
best_p1_idx = argmax([p.result.auc_top10 for p in phase1_results])
best_p1 = phase1_results[best_p1_idx]
println("\nBest Phase 1: $(best_p1.name) → AUC=$(round(best_p1.result.auc_top10, digits=4))")
println("Using lr=$(best_p1.cfg.lr), constructive_only=$(best_p1.cfg.constr) for Phase 2")
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════
# Phase 2: β × reward_weighted (using best LR + constr from Phase 1)
# ═══════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("PHASE 2: β × reward_weighted (lr=$(best_p1.cfg.lr), constr=$(best_p1.cfg.constr))")
println("=" ^ 70)
flush(stdout)

phase2_configs = [
    (beta=1.0, rw=false),   # Standard TB (no reward shaping at all)
    (beta=1.0, rw=true),    # RW-TB with β=1 (mild weighting)
    (beta=2.0, rw=true),    # RW-TB with β=2
    (beta=4.0, rw=true),    # RW-TB with β=4
    (beta=8.0, rw=true),    # RW-TB with β=8 (our v10 default)
]

phase2_results = []
for (i, cfg) in enumerate(phase2_configs)
    name = "P2-$(i): β=$(cfg.beta), rw=$(cfg.rw)"
    println("\n" * "-" ^ 70)
    println("[$i/$(length(phase2_configs))] $name")
    println("-" ^ 70)
    flush(stdout)

    r = run_smiles_pmo_task(TASK;
        budget=BUDGET,
        pretrained_params=pretrained_params,
        pretrained_states=pretrained_states,
        vocab=vocab,
        policy_model=policy_model,
        training_mode=:tb,
        learning_rate=best_p1.cfg.lr,
        reward_exponent=cfg.beta,
        constructive_only=best_p1.cfg.constr,
        reward_weighted=cfg.rw,
        kl_weight=FIXED[:kl_weight],
        use_replay=FIXED[:use_replay],
        replay_ratio=FIXED[:replay_ratio],
        batch_size=FIXED[:batch_size],
        lr_z=FIXED[:lr_z],
        log_z_grad_clip=FIXED[:log_z_grad_clip],
        warmup_iters=FIXED[:warmup_iters],
        n_iterations=FIXED[:n_iterations],
        verbose=true,
    )
    push!(phase2_results, (name=name, cfg=cfg, result=r))
    println(">>> $name → AUC=$(round(r.auc_top10, digits=4)), Top1=$(round(r.top1, digits=4)), Top10=$(round(r.top10_mean, digits=4))")
    flush(stdout)
end

# Phase 2 summary
println("\n" * "=" ^ 70)
println("PHASE 2 RESULTS")
println("=" ^ 70)
println("Config                                  AUC      Top1     Top10    Unique")
println("-" ^ 70)
for p in phase2_results
    println("$(rpad(p.name, 40))$(lpad(string(round(p.result.auc_top10, digits=4)), 7))  $(lpad(string(round(p.result.top1, digits=4)), 7))  $(lpad(string(round(p.result.top10_mean, digits=4)), 7))  $(lpad(string(p.result.unique_molecules), 7))")
end
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════
# Final Summary
# ═══════════════════════════════════════════════════════════════════════
println("\n\n" * "=" ^ 70)
println("TIER 1 GRID SEARCH — FINAL SUMMARY")
println("=" ^ 70)

all_results = vcat(phase1_results, phase2_results)
println("\nAll configs ranked by AUC:")
println("-" ^ 70)
sorted = sort(all_results, by=p -> p.result.auc_top10, rev=true)
for (rank, p) in enumerate(sorted)
    marker = rank == 1 ? " ★" : ""
    println("  #$rank  $(rpad(p.name, 40)) AUC=$(round(p.result.auc_top10, digits=4))$marker")
end

best = sorted[1]
println("\n" * "=" ^ 70)
println("BEST CONFIG: $(best.name)")
println("  AUC top-10 = $(round(best.result.auc_top10, digits=4))")
println("  Top-1 = $(round(best.result.top1, digits=4))")
println("  Top-10 mean = $(round(best.result.top10_mean, digits=4))")
println("  Unique molecules = $(best.result.unique_molecules)")
println("=" ^ 70)
println()
println("Previous baseline (TB v10, 3K budget): AUC=0.9044")
println("Improvement: $(round((best.result.auc_top10 - 0.9044) / 0.9044 * 100, digits=1))%")
flush(stdout)

# Save results
mkpath(joinpath(@__DIR__, "..", "..", "checkpoints", "tier1_grid_search"))
serialize(joinpath(@__DIR__, "..", "..", "checkpoints", "tier1_grid_search", "grid_search_results.jls"),
    Dict("phase1" => phase1_results, "phase2" => phase2_results, "best" => best))
println("\nResults saved to checkpoints/tier1_grid_search/grid_search_results.jls")
flush(stdout)
