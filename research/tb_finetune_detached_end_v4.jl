#!/usr/bin/env julia
# TB Fine-Tuning with Detached-END v4: Proper log_Z Tracking
#
# Key fix: periodic log_Z re-estimation every 25 iters inside training loop.
# This solves the v3 failure where log_Z was stuck (20.81→20.82) while true Z
# changed by 2-3 nats, causing policy divergence.
#
# Config: LR=1e-5, KL=1.0 (conservative, proven to work in v1)
# Single continuous segment of 300 iters (Adam momentum preserved, log_Z tracked)

using Pkg; Pkg.activate(".")
using GFlowNet
using Serialization
using Statistics
using Printf
using Random

println("=" ^ 70)
println("TB Fine-Tuning with Detached-END v4 (Proper log_Z Tracking)")
println("=" ^ 70)

# =============================================================================
# Setup
# =============================================================================
println("\nLoading pretrained checkpoint...")
pretrained = load_pretrained_checkpoint("checkpoints/pretrain/final.jls")

println("Loading ZINC vocabulary...")
smiles_data = load_zinc_smiles("data/zinc/250k_rndm_zinc_drugs_clean_3.csv"; max_molecules=50000)
vocab = SMILESVocabulary()
prepare_zinc_dataset(vocab, smiles_data)

actual_vocab_size = size(pretrained.params.output.layer_2.weight, 1)
println("  Vocab size (pretrained): $actual_vocab_size")

model, _, init_states = create_smiles_policy(;
    vocab_size=actual_vocab_size, hidden_dim=512, embed_dim=128, n_layers=3
)

# QED oracle
using PythonCall
rdkit = pyimport("rdkit.Chem")
rdkit_qed = pyimport("rdkit.Chem.QED")
function qed_oracle(smi::String)::Float64
    mol = rdkit.MolFromSmiles(smi)
    if pyis(mol, pybuiltins.None)
        return 0.0
    end
    try
        return pyconvert(Float64, rdkit_qed.qed(mol))
    catch
        return 0.0
    end
end

function evaluate_model(model, ps, states, vocab, oracle_fn; n_samples=200, temp=0.8)
    qeds = Float64[]
    valid = 0
    unique_smiles = Set{String}()
    for _ in 1:n_samples
        smi, _, _ = sample_smiles_autoregressive(
            model, ps, states, vocab;
            max_length=150, temperature=temp, constrained=true
        )
        q = oracle_fn(smi)
        if q > 0.0
            valid += 1
            push!(qeds, q)
            push!(unique_smiles, smi)
        end
    end
    sort!(qeds; rev=true)
    n = length(qeds)
    return (
        valid=valid, total=n_samples, validity=100.0*valid/n_samples,
        qeds=qeds, n_unique=length(unique_smiles),
        mean_qed=n > 0 ? mean(qeds) : 0.0,
        median_qed=n > 0 ? qeds[div(n,2)+1] : 0.0,
        geq_07=count(>=(0.7), qeds),
        geq_08=count(>=(0.8), qeds),
        geq_09=count(>=(0.9), qeds),
        top5=n >= 5 ? qeds[1:5] : qeds
    )
end

function print_eval(name, stats)
    @printf("  %-30s: valid=%d/%d (%.1f%%)  unique=%d  QED: mean=%.3f  >=0.7=%d  >=0.8=%d  >=0.9=%d\n",
        name, stats.valid, stats.total, stats.validity, stats.n_unique,
        stats.mean_qed, stats.geq_07, stats.geq_08, stats.geq_09)
    if length(stats.top5) > 0
        println("    Top 5: ", join([@sprintf("%.3f", q) for q in stats.top5], ", "))
    end
end

# =============================================================================
# Baseline
# =============================================================================
println("\n" * "=" ^ 70)
println("BASELINE")
println("=" ^ 70)
baseline = evaluate_model(model, pretrained.params, init_states, vocab, qed_oracle)
print_eval("pretrained", baseline)

# =============================================================================
# TB Fine-Tuning v4
# =============================================================================
println("\n" * "=" ^ 70)
println("TB FINE-TUNING v4 (Detached-END, log_Z re-estimation, 300 iters)")
println("=" ^ 70)

ref_params = deepcopy(pretrained.params)
ref_states = deepcopy(init_states)

# Conservative hyperparameters (same as v1 which preserved validity)
# Now with proper log_Z tracking inside the loop
config = FinetuningConfig(;
    n_iterations=300,
    sample_batch_size=32,
    learning_rate=1e-5,
    gradient_clip_norm=1.0,
    kl_weight=1.0,
    kl_decay_schedule=:none,
    loss_type=:shifted_cosh,
    cosh_threshold=2.0,
    max_length=150,
    temperature=1.0,
    epsilon=0.05,
    log_frequency=25,       # Log every 25 iters (aligns with log_Z re-estimation)
    reward_exponent=4.0,
    min_reward=0.01,
    training_mode=:tb
)

println("  LR=$(config.learning_rate)")
println("  KL=$(config.kl_weight) (constant)")
println("  beta=$(config.reward_exponent)")
println("  DETACHED-END: YES")
println("  log_Z re-estimation: every 25 iters")
println("  Total iters: $(config.n_iterations) (single continuous segment)")

# Run TB fine-tuning (single call — preserves Adam momentum for 300 iters)
result = finetune_smiles_gflownet(
    model, vocab, deepcopy(pretrained.params), init_states,
    ref_params, ref_states,
    qed_oracle, config;
    verbose=true
)

# =============================================================================
# Evaluate at final checkpoint
# =============================================================================
println("\n--- Final Evaluation (iter 300) ---")
final_stats = evaluate_model(model, result.params, init_states, vocab, qed_oracle)
print_eval("tb_v4_iter300", final_stats)

# Save checkpoint
checkpoint_dir = "checkpoints/finetune_tb_detached_v4"
mkpath(checkpoint_dir)
open("$checkpoint_dir/tb_v4_iter300.jls", "w") do f
    Serialization.serialize(f, Dict(
        "params" => result.params,
        "iter" => 300,
        "log_Z" => result.log_Z,
        "eval" => (mean_qed=final_stats.mean_qed, validity=final_stats.validity,
                  geq_07=final_stats.geq_07, geq_08=final_stats.geq_08, geq_09=final_stats.geq_09)
    ))
end
println("  Saved: $checkpoint_dir/tb_v4_iter300.jls")

# =============================================================================
# Summary
# =============================================================================
println("\n" * "=" ^ 70)
println("SUMMARY: TB Fine-Tuning v4")
println("=" ^ 70)

@printf("  %-30s  valid%%  unique  QED_mean  >=0.7  >=0.8  >=0.9\n", "Checkpoint")
@printf("  %-30s  ------  ------  --------  -----  -----  -----\n", "-" ^ 30)
@printf("  %-30s  %5.1f%%  %5d   %6.3f   %5d  %5d  %5d\n",
    "pretrained", baseline.validity, baseline.n_unique,
    baseline.mean_qed, baseline.geq_07, baseline.geq_08, baseline.geq_09)

delta = final_stats.mean_qed - baseline.mean_qed
@printf("  %-30s  %5.1f%%  %5d   %6.3f   %5d  %5d  %5d  (delta=%+.3f)\n",
    "TB v4 iter 300", final_stats.validity, final_stats.n_unique,
    final_stats.mean_qed, final_stats.geq_07, final_stats.geq_08, final_stats.geq_09, delta)

# Compare with earlier results
println("\nComparison with earlier experiments:")
println("  RWMLE iter 50:     QED 0.823 (+9%), validity 92.5%, >=0.9: 34")
println("  TB v1 iter 100:    QED 0.744 (-2%), validity 87.5%, >=0.9: 9")
println("  TB v3 iter 100:    QED 0.571 (COLLAPSED — no log_Z re-estimation)")
println("  TB v4 iter 300:    QED $(round(final_stats.mean_qed, digits=3)), validity $(round(final_stats.validity, digits=1))%")

println("\nDone!")
