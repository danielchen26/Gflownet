#!/usr/bin/env julia
# TB Fine-Tuning with Detached-END Gradient Surgery
#
# This is REAL GFlowNet TB fine-tuning with the ORIGINAL single-head architecture.
# No architecture change — same model, same sampling, same everything.
#
# The key insight: at non-terminal positions, decompose the probability as:
#   log P(a_t | s_t) = log P_atom(a_t | not END) + log(1 - P_END)
# and STOP the gradient through log(1 - P_END).
#
# This means:
#   - TB gradient only adjusts atom selection at non-terminal steps
#   - END token is NOT suppressed by TB gradient (the root cause of failure!)
#   - END token IS optimized at the terminal position (full gradient)
#   - KL regularization maintains termination behavior from pretraining
#
# Mathematically identical forward pass. Only the gradient routing changes.

using Pkg; Pkg.activate(".")
using GFlowNet
using Serialization
using Statistics
using Printf
using Random

println("=" ^ 70)
println("TB Fine-Tuning with Detached-END Gradient Surgery")
println("=" ^ 70)

# =============================================================================
# 1. Load pretrained checkpoint and vocab
# =============================================================================
println("\nLoading pretrained checkpoint...")
pretrained = load_pretrained_checkpoint("checkpoints/pretrain/final.jls")

println("Loading ZINC vocabulary...")
smiles_data = load_zinc_smiles("data/zinc/250k_rndm_zinc_drugs_clean_3.csv"; max_molecules=50000)
vocab = SMILESVocabulary()
prepare_zinc_dataset(vocab, smiles_data)

# Use vocab size from pretrained params (may differ from rebuilt vocab)
actual_vocab_size = size(pretrained.params.output.layer_2.weight, 1)
println("  Vocab size (pretrained): $actual_vocab_size")

# Create model with ORIGINAL architecture (single-head, no term_head)
model, _, init_states = create_smiles_policy(;
    vocab_size=actual_vocab_size, hidden_dim=512, embed_dim=128, n_layers=3
)
println("  Architecture: single-head (has_term_head=$(has_term_head(model)))")

# =============================================================================
# 2. QED oracle
# =============================================================================
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
# 3. Evaluation helper
# =============================================================================
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
# 4. Baseline evaluation
# =============================================================================
println("\n" * "=" ^ 70)
println("BASELINE: Pretrained Model (200 samples, T=0.8, constrained)")
println("=" ^ 70)

baseline = evaluate_model(model, pretrained.params, init_states, vocab, qed_oracle)
print_eval("pretrained (single-head)", baseline)

# =============================================================================
# 5. TB Fine-Tuning with Detached-END
# =============================================================================
println("\n" * "=" ^ 70)
println("TB FINE-TUNING (Detached-END Gradient Surgery)")
println("=" ^ 70)

ref_params = deepcopy(pretrained.params)
ref_states = deepcopy(init_states)

# Configuration — same hyperparameters as previous experiments for fair comparison
tb_config = FinetuningConfig(;
    n_iterations=100,
    sample_batch_size=32,
    learning_rate=1e-5,
    gradient_clip_norm=1.0,
    kl_weight=1.0,
    kl_decay_schedule=:none,   # Constant KL (cosine->0 caused collapse)
    loss_type=:shifted_cosh,
    cosh_threshold=2.0,
    max_length=150,
    temperature=1.0,
    epsilon=0.05,
    log_frequency=10,
    reward_exponent=4.0,
    min_reward=0.01,
    training_mode=:tb          # REAL GFlowNet TB loss
)

checkpoint_iters = [25, 50, 75, 100]
checkpoint_results = Dict{Int, Any}()
current_params = deepcopy(pretrained.params)

println("\nRunning TB fine-tuning ($(tb_config.n_iterations) iters)...")
println("  loss_type=shifted_cosh, cosh_threshold=2.0")
println("  reward_exponent=beta=$(tb_config.reward_exponent)")
println("  KL=$(tb_config.kl_weight) (constant)")
println("  LR=$(tb_config.learning_rate)")
println("  DETACHED-END: YES (gradient surgery, no architecture change)")

for (seg_idx, target_iter) in enumerate(checkpoint_iters)
    prev_iter = seg_idx > 1 ? checkpoint_iters[seg_idx - 1] : 0
    seg_iters = target_iter - prev_iter

    seg_config = FinetuningConfig(;
        n_iterations=seg_iters,
        sample_batch_size=tb_config.sample_batch_size,
        learning_rate=tb_config.learning_rate,
        gradient_clip_norm=tb_config.gradient_clip_norm,
        kl_weight=tb_config.kl_weight,
        kl_decay_schedule=tb_config.kl_decay_schedule,
        loss_type=tb_config.loss_type,
        cosh_threshold=tb_config.cosh_threshold,
        max_length=tb_config.max_length,
        temperature=tb_config.temperature,
        epsilon=tb_config.epsilon,
        log_frequency=tb_config.log_frequency,
        reward_exponent=tb_config.reward_exponent,
        min_reward=tb_config.min_reward,
        training_mode=:tb
    )

    result = finetune_smiles_gflownet(
        model, vocab, current_params, init_states,
        ref_params, ref_states,
        qed_oracle, seg_config;
        verbose=true
    )

    global current_params = result.params

    # Evaluate
    println("\n--- Evaluation at iter $target_iter ---")
    stats = evaluate_model(model, current_params, init_states, vocab, qed_oracle)
    print_eval("tb_detached_iter$target_iter", stats)
    checkpoint_results[target_iter] = stats

    # Early stopping: catastrophic degradation
    if stats.mean_qed < baseline.mean_qed * 0.5
        println("\nEARLY STOP: QED collapsed to $(round(stats.mean_qed, digits=3))")
        break
    end

    # Save checkpoint
    checkpoint_dir = "checkpoints/finetune_tb_detached_end"
    mkpath(checkpoint_dir)
    open("$checkpoint_dir/tb_detached_iter$(target_iter).jls", "w") do f
        Serialization.serialize(f, Dict(
            "params" => current_params,
            "iter" => target_iter,
            "model" => "single-head with detached-END TB",
            "eval" => (mean_qed=stats.mean_qed, validity=stats.validity,
                      geq_07=stats.geq_07, geq_08=stats.geq_08, geq_09=stats.geq_09)
        ))
    end
    println("  Saved: $checkpoint_dir/tb_detached_iter$(target_iter).jls")
end

# =============================================================================
# 6. Summary
# =============================================================================
println("\n" * "=" ^ 70)
println("SUMMARY: TB Fine-Tuning with Detached-END Gradient Surgery")
println("=" ^ 70)

@printf("  %-30s  valid%%  unique  QED_mean  >=0.7  >=0.8  >=0.9\n", "Checkpoint")
@printf("  %-30s  ------  ------  --------  -----  -----  -----\n", "-" ^ 30)
@printf("  %-30s  %5.1f%%  %5d   %6.3f   %5d  %5d  %5d\n",
    "pretrained (single-head)", baseline.validity, baseline.n_unique,
    baseline.mean_qed, baseline.geq_07, baseline.geq_08, baseline.geq_09)

for iter in sort(collect(keys(checkpoint_results)))
    s = checkpoint_results[iter]
    delta = s.mean_qed - baseline.mean_qed
    @printf("  %-30s  %5.1f%%  %5d   %6.3f   %5d  %5d  %5d  (delta=%+.3f)\n",
        "TB detached iter $iter", s.validity, s.n_unique,
        s.mean_qed, s.geq_07, s.geq_08, s.geq_09, delta)
end

println("\nKey question: Does QED improve while validity is preserved?")
println("If yes -> TB works with detached-END gradient surgery -> REAL GFlowNet!")
println("Done!")
