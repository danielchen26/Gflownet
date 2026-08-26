#!/usr/bin/env julia
# Diagnostic micro test: β=50+MSE vs β=8+shifted_cosh
# Tests Genetic GFN-style config (β=50, lr_z=0.1, MSE, no reward_weighted)
# vs our validated baseline (β=8, shifted_cosh, reward_weighted)

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

# Load RDKit bridge (defines module RDKitBridge)
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "rdkit_bridge.jl"))
RDKitBridge.init_rdkit!()  # Must explicitly initialize!
println("RDKit initialized"); flush(stdout)

# QED reward using properly initialized RDKitBridge
function qed_reward(smiles::String)::Float64
    try
        props = RDKitBridge.compute_mol_properties(smiles)
        if isnothing(props) || isnan(props.qed)
            return 0.0
        end
        return Float64(props.qed)
    catch
        return 0.0
    end
end

# === DIAGNOSTIC: Verify QED works ===
println("\n--- QED Diagnostic ---")
for smi in ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    score = qed_reward(smi)
    println("  QED('$smi') = $(round(score, digits=4))")
end; flush(stdout)

# Verify pretrained model generates valid SMILES
println("\n--- Sampling Diagnostic ---")
let n_valid = 0, qed_scores = Float64[]
    for i in 1:8
        smi, tok, _ = sample_smiles_autoregressive(
            policy_model, pretrained_params, pretrained_states, vocab;
            max_length=150, temperature=1.0, constrained=false)
        if !isempty(smi) && length(tok) >= 2
            r = qed_reward(smi)
            if r > 0.0
                n_valid += 1
                push!(qed_scores, r)
                i <= 3 && println("  Sample $i: QED=$(round(r, digits=3))")
            end
        end
    end
    println("Validity: $n_valid/8")
    if !isempty(qed_scores)
        println("QED: mean=$(round(mean(qed_scores), digits=3)), range=$(round(minimum(qed_scores), digits=3))-$(round(maximum(qed_scores), digits=3))")
        for β in [8.0, 50.0]
            ws = [q^β for q in qed_scores]
            println("  β=$β: R^β range = $(minimum(ws)) .. $(maximum(ws))")
        end
    end
end; flush(stdout)

ref_params = deepcopy(pretrained_params)
ref_states = deepcopy(pretrained_states)

# === TEST A: β=50, lr_z=0.1, MSE, NO reward_weighted (Genetic GFN style) ===
println("\n" * "="^60)
println("TEST A: β=50, MSE, no RW, no surgery (Genetic GFN style)")
println("="^60); flush(stdout)

config_a = GFlowNet.FinetuningConfig(
    n_iterations=3,
    sample_batch_size=8,
    learning_rate=3e-5,
    kl_weight=0.01,
    kl_decay_schedule=:none,
    loss_type=:mse,
    reward_exponent=50.0,
    min_reward=0.01,
    training_mode=:tb,
    constructive_only=false,    # Genetic GFN: no surgery
    reward_weighted=false,      # Genetic GFN: no reward weighting
    freeze_gru=true,
    unfreeze_top_gru=true,
    use_replay=false,
    log_frequency=1,
    lr_z=0.1,
    log_z_grad_clip=100.0,
)

t0 = time()
result_a = GFlowNet.finetune_smiles_gflownet(
    policy_model, vocab, deepcopy(pretrained_params), pretrained_states,
    ref_params, ref_states,
    qed_reward, config_a;
    verbose=true,
)
elapsed_a = round(time() - t0, digits=1)
println("\n>>> TEST A: unique=$(length(result_a.history.unique_smiles)), log_Z=$(round(Float64(result_a.log_Z), digits=2)) [$(elapsed_a)s]")
flush(stdout)

# === TEST B: β=8, shifted_cosh + reward_weighted (our validated baseline) ===
println("\n" * "="^60)
println("TEST B: β=8, shifted_cosh, +RW, +surgery (our baseline)")
println("="^60); flush(stdout)

config_b = GFlowNet.FinetuningConfig(
    n_iterations=3,
    sample_batch_size=8,
    learning_rate=3e-5,
    kl_weight=0.01,
    kl_decay_schedule=:none,
    loss_type=:shifted_cosh,
    reward_exponent=8.0,
    min_reward=0.01,
    training_mode=:tb,
    constructive_only=true,
    reward_weighted=true,
    freeze_gru=true,
    unfreeze_top_gru=true,
    use_replay=false,
    log_frequency=1,
    lr_z=3e-4,
    log_z_grad_clip=5.0,
)

t0 = time()
result_b = GFlowNet.finetune_smiles_gflownet(
    policy_model, vocab, deepcopy(pretrained_params), pretrained_states,
    ref_params, ref_states,
    qed_reward, config_b;
    verbose=true,
)
elapsed_b = round(time() - t0, digits=1)
println("\n>>> TEST B: unique=$(length(result_b.history.unique_smiles)), log_Z=$(round(Float64(result_b.log_Z), digits=2)) [$(elapsed_b)s]")
flush(stdout)

# === COMPARISON ===
println("\n" * "="^60)
println("COMPARISON")
println("="^60)
println("A (β=50+MSE, Genetic GFN): unique=$(length(result_a.history.unique_smiles)), log_Z=$(round(Float64(result_a.log_Z), digits=2))")
println("B (β=8+cosh, our baseline): unique=$(length(result_b.history.unique_smiles)), log_Z=$(round(Float64(result_b.log_Z), digits=2))")
flush(stdout)
