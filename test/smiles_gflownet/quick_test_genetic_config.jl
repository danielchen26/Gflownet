#!/usr/bin/env julia
# Quick test: β=50 + lr_z=0.1 + MSE with our validated architecture
# Isolates the hyperparameter changes from architectural changes

using Pkg
Pkg.activate(".")
using GFlowNet
using Serialization, Statistics

checkpoint_path = joinpath(@__DIR__, "..", "..", "checkpoints", "pretrain", "final.jls")
checkpoint = deserialize(checkpoint_path)
pretrained_params = checkpoint["params"]
pretrained_states = checkpoint["states"]
vocab = SMILESVocabulary()
policy_model, _, _ = create_smiles_policy(; vocab_size=checkpoint["vocab_size"], hidden_dim=512, embed_dim=128, n_layers=3)
println("Checkpoint loaded"); flush(stdout)

include(joinpath(@__DIR__, "..", "..", "src", "applications", "smiles_gflownet.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "rdkit_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "oracle_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "oracle_manager.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "pmo_benchmark.jl"))
println("Modules loaded"); flush(stdout)

# === Test 1: β=50, lr_z=0.1, MSE — keep our validated architecture ===
println("\n" * "="^60)
println("TEST 1: β=50, lr_z=0.1, MSE, freeze_gru+unfreeze_top, batch=32")
println("="^60); flush(stdout)

t0 = time()
r1 = run_smiles_pmo_task("qed";
    budget=300,
    pretrained_params=pretrained_params,
    pretrained_states=pretrained_states,
    vocab=vocab,
    policy_model=policy_model,
    training_mode=:tb,
    # Key changes: β, lr_z, loss
    reward_exponent=50.0,        # Genetic GFN: 50 (was 8)
    lr_z=0.1,                    # Genetic GFN: 0.1 (was 3e-4)
    log_z_grad_clip=100.0,       # Effectively no clipping
    loss_type=:mse,              # Genetic GFN: MSE (was shifted_cosh)
    # Keep our validated settings
    learning_rate=3e-5,          # Our validated default
    kl_weight=0.01,              # Our validated default
    constructive_only=true,      # Our gradient surgery
    reward_weighted=true,        # Our gradient surgery
    batch_size=32,               # Our validated default
    use_replay=true,
    replay_ratio=4,              # Our validated default
    ga_per_step=false,           # No GA overhead for speed
    use_augmentation=false,
    verbose=true,
)
println("\n>>> TEST 1: AUC=$(round(r1.auc_top10, digits=4)), Top1=$(round(r1.top1, digits=4)), Top10=$(round(r1.top10_mean, digits=4)) [$(round(time()-t0, digits=1))s]")
flush(stdout)

# === Test 2: Same as Test 1 but with our original β=8 for comparison ===
println("\n" * "="^60)
println("TEST 2 (control): β=8, lr_z=3e-4, shifted_cosh — our baseline")
println("="^60); flush(stdout)

t0 = time()
r2 = run_smiles_pmo_task("qed";
    budget=300,
    pretrained_params=pretrained_params,
    pretrained_states=pretrained_states,
    vocab=vocab,
    policy_model=policy_model,
    training_mode=:tb,
    # Our validated baseline settings
    constructive_only=true,
    reward_weighted=true,
    use_replay=true,
    verbose=true,
)
println("\n>>> TEST 2 (control): AUC=$(round(r2.auc_top10, digits=4)), Top1=$(round(r2.top1, digits=4)), Top10=$(round(r2.top10_mean, digits=4)) [$(round(time()-t0, digits=1))s]")
flush(stdout)

println("\n" * "="^60)
println("COMPARISON")
println("="^60)
println("Test 1 (β=50+MSE):    AUC=$(round(r1.auc_top10, digits=4))")
println("Test 2 (β=8+cosh):    AUC=$(round(r2.auc_top10, digits=4))")
println("Improvement: $(round(r1.auc_top10 - r2.auc_top10, digits=4))")
