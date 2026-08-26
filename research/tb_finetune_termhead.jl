#!/usr/bin/env julia
# TB Fine-Tuning with Separate Termination Head
#
# This is the REAL GFlowNet fine-tuning — Trajectory Balance with proper architecture.
#
# The key insight: the original model has atoms and END in the SAME softmax.
# TB gradient boosts atoms at ~35 positions → suppresses END → model can't terminate.
#
# Fix: separate termination head (Dense(512→1, sigmoid)) decouples "when to stop"
# from "which atom to emit". TB gradient for atoms literally cannot affect termination.
#
# Pipeline:
#   1. Load pretrained checkpoint (single-head)
#   2. Transfer weights to dual-head architecture (atom_head + term_head)
#   3. Short MLE warmup to calibrate term_head (10 iters)
#   4. TB fine-tuning with QED oracle
#   5. Compare against pretrained baseline

using Pkg; Pkg.activate(".")
using GFlowNet
using Serialization
using Statistics
using Printf
using Random

println("=" ^ 70)
println("TB Fine-Tuning with Separate Termination Head")
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
println("  Vocab size (current): $(vocab.size)")

# Get actual vocab size from pretrained params (may differ from current vocab)
actual_vocab_size = size(pretrained.params.output.layer_2.weight, 1)
println("  Vocab size (pretrained): $actual_vocab_size")

# Create old model (single-head) for reference — must match pretrained params
old_model, _, old_states = create_smiles_policy(;
    vocab_size=actual_vocab_size, hidden_dim=512, embed_dim=128, n_layers=3
)

# =============================================================================
# 2. Transfer weights to dual-head architecture
# =============================================================================
println("\nTransferring weights to dual-head architecture...")
model_config = (vocab_size=actual_vocab_size, hidden_dim=512, embed_dim=128, n_layers=3)
new_model, new_params, new_states = convert_to_term_head_params(
    pretrained.params, old_states, model_config
)
println("  has_term_head: $(has_term_head(new_model))")
println("  Term head weight shape: $(size(new_params.term_head.weight))")
println("  Term head bias: $(new_params.term_head.bias)")

# =============================================================================
# 3. Verify: dual-head model should produce similar outputs
# =============================================================================
println("\n" * "=" ^ 70)
println("VERIFICATION: Dual-head vs single-head similarity")
println("=" ^ 70)

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
    @printf("  %-25s: valid=%d/%d (%.1f%%)  unique=%d  QED: mean=%.3f  ≥0.7=%d  ≥0.8=%d  ≥0.9=%d\n",
        name, stats.valid, stats.total, stats.validity, stats.n_unique,
        stats.mean_qed, stats.geq_07, stats.geq_08, stats.geq_09)
    if length(stats.top5) > 0
        println("    Top 5: ", join([@sprintf("%.3f", q) for q in stats.top5], ", "))
    end
end

# Evaluate both models
println("\nSingle-head (pretrained):")
baseline_old = evaluate_model(old_model, pretrained.params, old_states, vocab, qed_oracle)
print_eval("single-head pretrained", baseline_old)

println("\nDual-head (transferred):")
baseline_new = evaluate_model(new_model, new_params, new_states, vocab, qed_oracle)
print_eval("dual-head transferred", baseline_new)

# =============================================================================
# 4. Short MLE warmup to calibrate termination head
# =============================================================================
println("\n" * "=" ^ 70)
println("MLE WARMUP (calibrate termination head)")
println("=" ^ 70)

# Encode a small batch of ZINC molecules for warmup
warmup_smiles = smiles_data[1:min(5000, length(smiles_data))]
warmup_seqs = [encode(vocab, smi; add_special_tokens=true) for smi in warmup_smiles]
warmup_seqs = filter(s -> length(s) >= 3, warmup_seqs)
println("  Warmup sequences: $(length(warmup_seqs))")

warmup_config = PretrainingConfig(;
    n_epochs=5,
    batch_size=256,
    tb_batch_size=0,     # No TB during warmup, just MLE
    learning_rate=5e-5,  # Lower LR to fine-tune without disrupting GRU
    mle_weight=1.0,
    tb_weight=0.0,
    gradient_clip_norm=1.0,
    max_length=150,
    log_frequency=10,
    checkpoint_frequency=0,  # No checkpoints
    use_batched_loss=true
)

println("Running MLE warmup (5 epochs, calibrating term_head)...")
warmup_result = pretrain_smiles_gflownet(
    new_model, vocab, new_params, new_states,
    warmup_seqs, warmup_config;
    verbose=true
)
warmup_params = warmup_result.params

println("\nAfter warmup:")
baseline_warmup = evaluate_model(new_model, warmup_params, new_states, vocab, qed_oracle)
print_eval("dual-head after warmup", baseline_warmup)

# =============================================================================
# 5. TB Fine-Tuning (the real GFlowNet training!)
# =============================================================================
println("\n" * "=" ^ 70)
println("TB FINE-TUNING (Trajectory Balance — REAL GFlowNet)")
println("=" ^ 70)

# Reference policy for KL regularization
ref_params = deepcopy(warmup_params)
ref_states = deepcopy(new_states)

# TB config — same hyperparameters as the failed single-head experiment
# but now with separate termination head, TB gradient should work correctly
tb_config = FinetuningConfig(;
    n_iterations=100,
    sample_batch_size=32,
    learning_rate=1e-5,
    gradient_clip_norm=1.0,
    kl_weight=1.0,
    kl_decay_schedule=:none,  # Constant KL (cosine→0 caused collapse)
    loss_type=:shifted_cosh,
    cosh_threshold=2.0,
    max_length=150,
    temperature=1.0,
    epsilon=0.05,
    log_frequency=10,
    reward_exponent=4.0,
    min_reward=0.01,
    training_mode=:tb        # The REAL GFlowNet loss!
)

# Checkpoint and evaluate every 25 iters
checkpoint_iters = [25, 50, 75, 100]
checkpoint_results = Dict{Int, Any}()
current_params = deepcopy(warmup_params)

println("\nRunning TB fine-tuning ($(tb_config.n_iterations) iters)...")
println("  loss_type=shifted_cosh, cosh_threshold=2.0")
println("  reward_exponent=β=$(tb_config.reward_exponent)")
println("  KL=$(tb_config.kl_weight) (constant)")
println("  LR=$(tb_config.learning_rate)")
println("  SEPARATE TERMINATION HEAD: YES ✓")

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
        new_model, vocab, current_params, new_states,
        ref_params, ref_states,
        qed_oracle, seg_config;
        verbose=true
    )

    global current_params = result.params

    # Evaluate
    println("\n--- Evaluation at iter $target_iter ---")
    stats = evaluate_model(new_model, current_params, new_states, vocab, qed_oracle)
    print_eval("tb_termhead_iter$target_iter", stats)
    checkpoint_results[target_iter] = stats

    # Early stopping: catastrophic degradation
    if stats.mean_qed < baseline_warmup.mean_qed * 0.5
        println("\n⚠ EARLY STOP: QED collapsed to $(round(stats.mean_qed, digits=3))")
        break
    end

    # Save checkpoint
    checkpoint_dir = "checkpoints/finetune_tb_termhead"
    mkpath(checkpoint_dir)
    open("$checkpoint_dir/tb_termhead_iter$(target_iter).jls", "w") do f
        Serialization.serialize(f, Dict(
            "params" => current_params,
            "iter" => target_iter,
            "model" => "dual-head with term_head",
            "eval" => (mean_qed=stats.mean_qed, validity=stats.validity,
                      geq_07=stats.geq_07, geq_08=stats.geq_08, geq_09=stats.geq_09)
        ))
    end
    println("  Saved: $checkpoint_dir/tb_termhead_iter$(target_iter).jls")
end

# =============================================================================
# 6. Summary
# =============================================================================
println("\n" * "=" ^ 70)
println("SUMMARY: TB Fine-Tuning with Separate Termination Head")
println("=" ^ 70)

@printf("  %-25s  valid%%  unique  QED_mean  ≥0.7  ≥0.8  ≥0.9\n", "Checkpoint")
@printf("  %-25s  ------  ------  --------  ----  ----  ----\n", "-" ^ 25)
@printf("  %-25s  %5.1f%%  %5d   %6.3f   %4d  %4d  %4d\n",
    "pretrained (single-head)", baseline_old.validity, baseline_old.n_unique,
    baseline_old.mean_qed, baseline_old.geq_07, baseline_old.geq_08, baseline_old.geq_09)
@printf("  %-25s  %5.1f%%  %5d   %6.3f   %4d  %4d  %4d\n",
    "after warmup (dual-head)", baseline_warmup.validity, baseline_warmup.n_unique,
    baseline_warmup.mean_qed, baseline_warmup.geq_07, baseline_warmup.geq_08, baseline_warmup.geq_09)

for iter in sort(collect(keys(checkpoint_results)))
    s = checkpoint_results[iter]
    delta = s.mean_qed - baseline_warmup.mean_qed
    @printf("  %-25s  %5.1f%%  %5d   %6.3f   %4d  %4d  %4d  (Δ=%+.3f)\n",
        "TB iter $iter", s.validity, s.n_unique,
        s.mean_qed, s.geq_07, s.geq_08, s.geq_09, delta)
end

println("\nKey question: Does QED improve WITHOUT validity collapsing?")
println("If yes → TB works with separate termination head → REAL GFlowNet fine-tuning!")
println("Done!")
