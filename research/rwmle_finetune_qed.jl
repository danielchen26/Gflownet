#!/usr/bin/env julia
# RWMLE Fine-Tuning Experiment: QED Optimization
#
# This is the grammar-preserving alternative to TB fine-tuning.
# Key insight: RWMLE gradient always points in the "increase probability" direction,
# matching the pretraining gradient. TB gradient can DECREASE token probabilities,
# which destroys the SMILES grammar encoded in GRU weights.
#
# Experiment plan:
#   1. Load pretrained checkpoint
#   2. Run RWMLE fine-tuning with QED oracle (100 iters, with validation every 25)
#   3. Compare pretrained vs RWMLE at key checkpoints
#   4. Measure: validity, QED distribution, molecule diversity

using Pkg; Pkg.activate(".")
using GFlowNet
using Serialization
using Statistics
using Printf
using Random

# =============================================================================
# Setup
# =============================================================================
println("=" ^ 70)
println("RWMLE Fine-Tuning Experiment: QED Optimization")
println("=" ^ 70)

# Load pretrained checkpoint
println("\nLoading pretrained checkpoint...")
pretrained = load_pretrained_checkpoint("checkpoints/pretrain/final.jls")

# Build vocab from ZINC
println("Loading ZINC vocabulary...")
smiles_data = load_zinc_smiles("data/zinc/250k_rndm_zinc_drugs_clean_3.csv"; max_molecules=50000)
vocab = SMILESVocabulary()
prepare_zinc_dataset(vocab, smiles_data)
println("  Vocab size: $(vocab.size)")

# Create model
model, _, init_states = create_smiles_policy(;
    vocab_size=vocab.size, hidden_dim=512, embed_dim=128, n_layers=3
)

# QED oracle (via RDKit)
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

# =============================================================================
# Baseline: Evaluate pretrained model
# =============================================================================
println("\n" * "=" ^ 70)
println("BASELINE: Pretrained Model (200 samples, T=0.8, constrained)")
println("=" ^ 70)

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
    @printf("  %-20s: valid=%d/%d (%.1f%%)  unique=%d  QED: mean=%.3f  median=%.3f  ≥0.7=%d  ≥0.8=%d  ≥0.9=%d\n",
        name, stats.valid, stats.total, stats.validity, stats.n_unique,
        stats.mean_qed, stats.median_qed, stats.geq_07, stats.geq_08, stats.geq_09)
    if length(stats.top5) > 0
        println("    Top 5: ", join([@sprintf("%.3f", q) for q in stats.top5], ", "))
    end
end

baseline = evaluate_model(model, pretrained.params, init_states, vocab, qed_oracle)
print_eval("pretrained", baseline)

# =============================================================================
# RWMLE Fine-Tuning
# =============================================================================
println("\n" * "=" ^ 70)
println("RWMLE FINE-TUNING")
println("=" ^ 70)

# Configuration: conservative to preserve grammar
config = FinetuningConfig(;
    n_iterations=100,
    sample_batch_size=32,
    learning_rate=3e-5,        # Slightly higher than TB since RWMLE is gentler
    gradient_clip_norm=1.0,
    kl_weight=1.0,             # Same KL as failed TB run for fair comparison
    kl_decay_schedule=:none,   # Constant KL — no decay (cosine→0 caused collapse)
    loss_type=:shifted_cosh,   # Only matters for TB, ignored in RWMLE
    cosh_threshold=2.0,
    max_length=150,
    temperature=1.0,
    epsilon=0.05,
    log_frequency=10,
    reward_exponent=4.0,       # β=4 strongly focuses on high-QED molecules
    min_reward=0.01,
    training_mode=:rwmle       # The key difference!
)

# Freeze reference policy
ref_params = deepcopy(pretrained.params)
ref_states = deepcopy(init_states)

# Run fine-tuning with checkpointing
println("\nRunning RWMLE fine-tuning ($(config.n_iterations) iters)...")
println("  reward_exponent=β=$(config.reward_exponent) (focus on high-QED)")
println("  KL=$(config.kl_weight) (constant, no decay)")
println("  LR=$(config.learning_rate)")

# Checkpoint evaluation schedule
checkpoint_iters = [25, 50, 75, 100]
checkpoint_results = Dict{Int, Any}()

# Manual fine-tuning loop with checkpointing
current_params = deepcopy(pretrained.params)

# Run in segments
for (seg_idx, target_iter) in enumerate(checkpoint_iters)
    prev_iter = seg_idx > 1 ? checkpoint_iters[seg_idx - 1] : 0
    seg_iters = target_iter - prev_iter

    seg_config = FinetuningConfig(;
        n_iterations=seg_iters,
        sample_batch_size=config.sample_batch_size,
        learning_rate=config.learning_rate,
        gradient_clip_norm=config.gradient_clip_norm,
        kl_weight=config.kl_weight,
        kl_decay_schedule=config.kl_decay_schedule,
        loss_type=config.loss_type,
        cosh_threshold=config.cosh_threshold,
        max_length=config.max_length,
        temperature=config.temperature,
        epsilon=config.epsilon,
        log_frequency=config.log_frequency,
        reward_exponent=config.reward_exponent,
        min_reward=config.min_reward,
        training_mode=:rwmle
    )

    result = finetune_smiles_gflownet(
        model, vocab, current_params, init_states,
        ref_params, ref_states,
        qed_oracle, seg_config;
        log_Z_init=0.0,
        verbose=true
    )

    global current_params = result.params

    # Evaluate at this checkpoint
    println("\n--- Evaluation at iter $target_iter ---")
    stats = evaluate_model(model, current_params, init_states, vocab, qed_oracle)
    print_eval("rwmle_iter$target_iter", stats)
    checkpoint_results[target_iter] = stats

    # Early stopping check: if quality degrades significantly, stop
    if stats.mean_qed < baseline.mean_qed * 0.8
        println("\n⚠ EARLY STOP: QED degraded to $(round(stats.mean_qed, digits=3)) " *
                "(< 80% of baseline $(round(baseline.mean_qed, digits=3)))")
        break
    end

    # Save checkpoint
    checkpoint_dir = "checkpoints/finetune_rwmle_qed"
    mkpath(checkpoint_dir)
    open("$checkpoint_dir/rwmle_qed_iter$(target_iter).jls", "w") do f
        Serialization.serialize(f, Dict(
            "params" => current_params,
            "iter" => target_iter,
            "config" => config,
            "eval" => (mean_qed=stats.mean_qed, validity=stats.validity,
                      geq_07=stats.geq_07, geq_08=stats.geq_08, geq_09=stats.geq_09)
        ))
    end
    println("  Saved checkpoint: $checkpoint_dir/rwmle_qed_iter$(target_iter).jls")
end

# =============================================================================
# Summary: Compare all checkpoints
# =============================================================================
println("\n" * "=" ^ 70)
println("SUMMARY: RWMLE Fine-Tuning Results")
println("=" ^ 70)

@printf("  %-20s  valid%%  unique  QED_mean  QED_med  ≥0.7  ≥0.8  ≥0.9\n", "Checkpoint")
@printf("  %-20s  ------  ------  --------  -------  ----  ----  ----\n", "-" ^ 20)
@printf("  %-20s  %5.1f%%  %5d   %6.3f   %6.3f   %4d  %4d  %4d\n",
    "pretrained", baseline.validity, baseline.n_unique,
    baseline.mean_qed, baseline.median_qed,
    baseline.geq_07, baseline.geq_08, baseline.geq_09)

for iter in sort(collect(keys(checkpoint_results)))
    s = checkpoint_results[iter]
    delta_qed = s.mean_qed - baseline.mean_qed
    @printf("  %-20s  %5.1f%%  %5d   %6.3f   %6.3f   %4d  %4d  %4d  (Δ=%+.3f)\n",
        "rwmle_iter$iter", s.validity, s.n_unique,
        s.mean_qed, s.median_qed,
        s.geq_07, s.geq_08, s.geq_09, delta_qed)
end

# Find best checkpoint
if !isempty(checkpoint_results)
    best_iter = argmax(iter -> checkpoint_results[iter].mean_qed, collect(keys(checkpoint_results)))
    best = checkpoint_results[best_iter]
    println("\nBest checkpoint: iter $best_iter")
    println("  QED mean: $(round(baseline.mean_qed, digits=3)) → $(round(best.mean_qed, digits=3)) " *
            "($(round((best.mean_qed - baseline.mean_qed) / baseline.mean_qed * 100, digits=1))%)")
    println("  QED ≥ 0.9: $(baseline.geq_09) → $(best.geq_09)")
end

println("\nDone!")
