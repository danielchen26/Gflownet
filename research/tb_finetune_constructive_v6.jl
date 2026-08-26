#!/usr/bin/env julia
# TB Fine-Tuning v6: Constructive-Only TB
#
# ROOT CAUSE OF QED DEGRADATION:
# Standard TB gradient has two components per molecule:
#   1. δ < 0 (model under-produces): gradient INCREASES probability → constructive
#   2. δ > 0 (model over-produces): gradient DECREASES probability → DESTRUCTIVE
# The destructive component corrupts SMILES grammar because decreasing probability
# of valid-but-mediocre molecules disrupts shared GRU weights.
#
# FIX: Skip molecules with δ > 0 (constructive_only=true).
# Only train on under-produced high-reward molecules.
# Grammar preserved because we never push probabilities DOWN.
#
# Evidence from v5: QED 0.754 → 0.637 at iter 150 with standard TB.
# Validity preserved (82%) thanks to detached-END, but atom selection corrupted.

using Pkg; Pkg.activate(".")
using GFlowNet
using Serialization
using Statistics
using Printf
using Random

flush(stdout)

println("=" ^ 70)
println("TB Fine-Tuning v6: Constructive-Only TB (Grammar-Preserving)")
println("=" ^ 70)
flush(stdout)

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
    flush(stdout)
end

# =============================================================================
# Baseline
# =============================================================================
println("\n" * "=" ^ 70)
println("BASELINE")
println("=" ^ 70)
flush(stdout)
baseline = evaluate_model(model, pretrained.params, init_states, vocab, qed_oracle)
print_eval("pretrained", baseline)

# =============================================================================
# v6: Constructive-Only TB, 12 × 25-iter segments
# =============================================================================
println("\n" * "=" ^ 70)
println("v6: CONSTRUCTIVE-ONLY TB (12×25 segments, 300 total iters)")
println("=" ^ 70)

ref_params = deepcopy(pretrained.params)
ref_states = deepcopy(init_states)

seg_config = FinetuningConfig(;
    n_iterations=25,
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
    log_frequency=5,
    reward_exponent=4.0,
    min_reward=0.01,
    training_mode=:tb,
    constructive_only=true      # THE KEY FIX
)

println("  LR=$(seg_config.learning_rate)")
println("  KL=$(seg_config.kl_weight) (constant)")
println("  beta=$(seg_config.reward_exponent)")
println("  DETACHED-END: YES")
println("  CONSTRUCTIVE-ONLY: YES  ← new in v6")
println("  Segments: 12 × 25 iters = 300 total")
println("  Evaluation: every 2 segments (50 iters)")
flush(stdout)

n_segments = 12
checkpoint_results = Dict{Int, Any}()
current_params = deepcopy(pretrained.params)
best_qed = 0.0
best_iter = 0
best_params = nothing

for seg in 1:n_segments
    total_iter = seg * 25
    println("\n" * "-" ^ 50)
    println("Segment $seg/$(n_segments): iters $(total_iter-24)-$total_iter")
    println("-" ^ 50)
    flush(stdout)

    result = finetune_smiles_gflownet(
        model, vocab, current_params, init_states,
        ref_params, ref_states,
        qed_oracle, seg_config;
        verbose=true
    )

    global current_params = result.params
    println("  Segment $seg done. Final log_Z: $(round(Float64(result.log_Z), digits=2))")
    flush(stdout)

    # Evaluate every 2 segments, plus seg 1 and final
    if seg % 2 == 0 || seg == 1 || seg == n_segments
        println("\n--- Evaluation at iter $total_iter ---")
        flush(stdout)
        stats = evaluate_model(model, current_params, init_states, vocab, qed_oracle)
        print_eval("v6_iter$total_iter", stats)
        checkpoint_results[total_iter] = stats

        # Track best
        if stats.mean_qed > best_qed && stats.validity > 70.0
            global best_qed = stats.mean_qed
            global best_iter = total_iter
            global best_params = deepcopy(current_params)
        end

        # Save checkpoint
        checkpoint_dir = "checkpoints/finetune_tb_v6"
        mkpath(checkpoint_dir)
        open("$checkpoint_dir/tb_v6_iter$(total_iter).jls", "w") do f
            Serialization.serialize(f, Dict(
                "params" => current_params,
                "iter" => total_iter,
                "eval" => (mean_qed=stats.mean_qed, validity=stats.validity,
                          geq_07=stats.geq_07, geq_08=stats.geq_08, geq_09=stats.geq_09)
            ))
        end
        println("  Saved checkpoint")
        flush(stdout)

        # Early stopping
        if stats.validity < 60.0
            println("\nEARLY STOP: Validity collapsed to $(round(stats.validity, digits=1))%")
            flush(stdout)
            break
        end
        if stats.mean_qed < baseline.mean_qed * 0.5
            println("\nEARLY STOP: QED collapsed to $(round(stats.mean_qed, digits=3))")
            flush(stdout)
            break
        end
    end
end

# Save best checkpoint
if best_params !== nothing
    checkpoint_dir = "checkpoints/finetune_tb_v6"
    mkpath(checkpoint_dir)
    open("$checkpoint_dir/tb_v6_best.jls", "w") do f
        Serialization.serialize(f, Dict(
            "params" => best_params,
            "iter" => best_iter,
            "mean_qed" => best_qed
        ))
    end
    println("\n  Best checkpoint saved: iter $best_iter, QED $(round(best_qed, digits=3))")
end

# =============================================================================
# Summary
# =============================================================================
println("\n" * "=" ^ 70)
println("SUMMARY: v6 Constructive-Only TB")
println("=" ^ 70)

@printf("  %-30s  valid%%  unique  QED_mean  >=0.7  >=0.8  >=0.9\n", "Checkpoint")
@printf("  %-30s  ------  ------  --------  -----  -----  -----\n", "-" ^ 30)
@printf("  %-30s  %5.1f%%  %5d   %6.3f   %5d  %5d  %5d\n",
    "pretrained", baseline.validity, baseline.n_unique,
    baseline.mean_qed, baseline.geq_07, baseline.geq_08, baseline.geq_09)

for iter in sort(collect(keys(checkpoint_results)))
    s = checkpoint_results[iter]
    delta = s.mean_qed - baseline.mean_qed
    @printf("  %-30s  %5.1f%%  %5d   %6.3f   %5d  %5d  %5d  (delta=%+.3f)\n",
        "v6 iter $iter", s.validity, s.n_unique,
        s.mean_qed, s.geq_07, s.geq_08, s.geq_09, delta)
end

println("\n  Best: iter $best_iter, QED $(round(best_qed, digits=3))")

# Historical comparison
println("\nAll experiments:")
println("  RWMLE iter 50:     QED 0.823 (+9%), validity 92.5%")
println("  TB v1  iter 25:    QED 0.744 (-1%), validity 89.0%")
println("  TB v5  iter 50:    QED 0.742 (-2%), validity 88.5%  (standard TB, degraded to 0.637 at iter 150)")
if !isempty(checkpoint_results)
    bi = best_iter
    bs = checkpoint_results[bi]
    @printf("  TB v6  iter %d:  QED %.3f (%+.1f%%), validity %.1f%%  (CONSTRUCTIVE-ONLY)\n",
        bi, bs.mean_qed, 100.0*(bs.mean_qed - baseline.mean_qed)/baseline.mean_qed, bs.validity)
end

println("\nDone!")
flush(stdout)
