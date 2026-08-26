#!/usr/bin/env julia
# TB Fine-Tuning v8: Aggressive Freeze-GRU with Full TB Gradient
#
# v7 results: Best at iter 25 (QED 0.758, +1.5%), then slowly degraded.
# Constructive-only filter was too restrictive — only ~50% of gradient signal used.
#
# v8 strategy: FULL TB gradient (both constructive AND destructive) on output layer.
# Since GRU is frozen, the output layer is the only thing being modified.
# Destructive gradient on the output layer is SAFE because:
#   - Grammar is encoded in GRU hidden states, not the output mapping
#   - Pushing probability AWAY from low-QED tokens is actually what we want
#   - The output layer has 334K params — enough capacity to handle both directions
#
# Changes vs v7:
#   - constructive_only = FALSE (full TB gradient)
#   - LR = 2e-4 (4× more aggressive for output layer)
#   - KL = 0.3 (even less regularization — GRU frozen = safe)
#   - β = 8.0 (strong focus on top QED)
#   - Short run: 4 × 25 = 100 iters (keep best, don't overtrain)

using Pkg; Pkg.activate(".")
using GFlowNet
using Serialization
using Statistics
using Printf
using Random

flush(stdout)

println("=" ^ 70)
println("TB Fine-Tuning v8: Aggressive Freeze-GRU, Full TB Gradient")
println("=" ^ 70)
flush(stdout)

# Setup
println("\nLoading pretrained checkpoint...")
pretrained = load_pretrained_checkpoint("checkpoints/pretrain/final.jls")

println("Loading ZINC vocabulary...")
smiles_data = load_zinc_smiles("data/zinc/250k_rndm_zinc_drugs_clean_3.csv"; max_molecules=50000)
vocab = SMILESVocabulary()
prepare_zinc_dataset(vocab, smiles_data)

actual_vocab_size = size(pretrained.params.output.layer_2.weight, 1)
println("  Vocab size: $actual_vocab_size")

model, _, init_states = create_smiles_policy(;
    vocab_size=actual_vocab_size, hidden_dim=512, embed_dim=128, n_layers=3
)

# QED oracle
using PythonCall
rdkit = pyimport("rdkit.Chem")
rdkit_qed = pyimport("rdkit.Chem.QED")
function qed_oracle(smi::String)::Float64
    mol = rdkit.MolFromSmiles(smi)
    pyis(mol, pybuiltins.None) && return 0.0
    try; return pyconvert(Float64, rdkit_qed.qed(mol)); catch; return 0.0; end
end

function evaluate_model(model, ps, states, vocab, oracle_fn; n_samples=200, temp=0.8)
    qeds = Float64[]; valid = 0; unique_smiles = Set{String}()
    for _ in 1:n_samples
        smi, _, _ = sample_smiles_autoregressive(model, ps, states, vocab;
            max_length=150, temperature=temp, constrained=true)
        q = oracle_fn(smi)
        if q > 0.0; valid += 1; push!(qeds, q); push!(unique_smiles, smi); end
    end
    sort!(qeds; rev=true); n = length(qeds)
    return (valid=valid, total=n_samples, validity=100.0*valid/n_samples,
        qeds=qeds, n_unique=length(unique_smiles),
        mean_qed=n > 0 ? mean(qeds) : 0.0,
        geq_07=count(>=(0.7), qeds), geq_08=count(>=(0.8), qeds),
        geq_09=count(>=(0.9), qeds),
        top5=n >= 5 ? qeds[1:5] : qeds)
end

function print_eval(name, stats)
    @printf("  %-30s: valid=%d/%d (%.1f%%)  unique=%d  QED: mean=%.3f  >=0.7=%d  >=0.8=%d  >=0.9=%d\n",
        name, stats.valid, stats.total, stats.validity, stats.n_unique,
        stats.mean_qed, stats.geq_07, stats.geq_08, stats.geq_09)
    length(stats.top5) > 0 && println("    Top 5: ", join([@sprintf("%.3f", q) for q in stats.top5], ", "))
    flush(stdout)
end

# Baseline
println("\n" * "=" ^ 70); println("BASELINE"); println("=" ^ 70); flush(stdout)
baseline = evaluate_model(model, pretrained.params, init_states, vocab, qed_oracle)
print_eval("pretrained", baseline)

# v8 Configuration
println("\n" * "=" ^ 70)
println("v8: FREEZE GRU + FULL TB GRADIENT (aggressive, 100 iters)")
println("=" ^ 70)

ref_params = deepcopy(pretrained.params)
ref_states = deepcopy(init_states)

seg_config = FinetuningConfig(;
    n_iterations=25,
    sample_batch_size=64,        # Larger batch for more stable gradient
    learning_rate=2e-4,          # 4× more aggressive than v7
    gradient_clip_norm=1.0,
    kl_weight=0.3,               # Lower KL: GRU frozen = grammar safe
    kl_decay_schedule=:none,
    loss_type=:shifted_cosh,
    cosh_threshold=2.0,
    max_length=150,
    temperature=1.0,
    epsilon=0.05,
    log_frequency=5,
    reward_exponent=8.0,         # Strong focus on top QED
    min_reward=0.01,
    training_mode=:tb,
    constructive_only=false,     # FULL TB gradient (both directions)
    freeze_gru=true              # GRU frozen = grammar safe
)

println("  LR=$(seg_config.learning_rate)  (4× higher than v7)")
println("  KL=$(seg_config.kl_weight)  (lower than v7's 0.5)")
println("  beta=$(seg_config.reward_exponent)")
println("  batch_size=$(seg_config.sample_batch_size)  (2× larger for stable gradients)")
println("  FREEZE-GRU: YES")
println("  CONSTRUCTIVE-ONLY: NO  (full TB gradient on output layer)")
println("  Total: 4 × 25 = 100 iters")
flush(stdout)

n_segments = 4
checkpoint_results = Dict{Int, Any}()
current_params = deepcopy(pretrained.params)
best_qed = 0.0; best_iter = 0; best_params = nothing

for seg in 1:n_segments
    total_iter = seg * 25
    println("\n" * "-" ^ 50)
    println("Segment $seg/$n_segments: iters $(total_iter-24)-$total_iter")
    println("-" ^ 50); flush(stdout)

    result = finetune_smiles_gflownet(
        model, vocab, current_params, init_states,
        ref_params, ref_states, qed_oracle, seg_config; verbose=true)

    global current_params = result.params
    println("  Segment $seg done. Final log_Z: $(round(Float64(result.log_Z), digits=2))")
    flush(stdout)

    # Evaluate every segment
    println("\n--- Evaluation at iter $total_iter ---"); flush(stdout)
    stats = evaluate_model(model, current_params, init_states, vocab, qed_oracle)
    print_eval("v8_iter$total_iter", stats)
    checkpoint_results[total_iter] = stats

    if stats.mean_qed > best_qed && stats.validity > 70.0
        global best_qed = stats.mean_qed
        global best_iter = total_iter
        global best_params = deepcopy(current_params)
    end

    # Save checkpoint
    checkpoint_dir = "checkpoints/finetune_tb_v8"
    mkpath(checkpoint_dir)
    open("$checkpoint_dir/tb_v8_iter$(total_iter).jls", "w") do f
        Serialization.serialize(f, Dict("params" => current_params, "iter" => total_iter,
            "eval" => (mean_qed=stats.mean_qed, validity=stats.validity,
                      geq_07=stats.geq_07, geq_08=stats.geq_08, geq_09=stats.geq_09)))
    end
    println("  Saved checkpoint"); flush(stdout)
end

# Save best
if best_params !== nothing
    checkpoint_dir = "checkpoints/finetune_tb_v8"
    open("$checkpoint_dir/tb_v8_best.jls", "w") do f
        Serialization.serialize(f, Dict("params" => best_params, "iter" => best_iter, "mean_qed" => best_qed))
    end
end

# Summary
println("\n" * "=" ^ 70)
println("SUMMARY: v8 Aggressive Freeze-GRU + Full TB")
println("=" ^ 70)
@printf("  %-25s  valid%%  unique  QED_mean  >=0.7  >=0.8  >=0.9\n", "Checkpoint")
@printf("  %-25s  ------  ------  --------  -----  -----  -----\n", "-" ^ 25)
@printf("  %-25s  %5.1f%%  %5d   %6.3f   %5d  %5d  %5d\n",
    "pretrained", baseline.validity, baseline.n_unique,
    baseline.mean_qed, baseline.geq_07, baseline.geq_08, baseline.geq_09)
for iter in sort(collect(keys(checkpoint_results)))
    s = checkpoint_results[iter]; delta = s.mean_qed - baseline.mean_qed
    @printf("  %-25s  %5.1f%%  %5d   %6.3f   %5d  %5d  %5d  (delta=%+.3f)\n",
        "v8 iter $iter", s.validity, s.n_unique, s.mean_qed, s.geq_07, s.geq_08, s.geq_09, delta)
end
println("\n  Best: iter $best_iter, QED $(round(best_qed, digits=3))")
println("\nComparison:")
println("  RWMLE:   QED 0.823 (+9%)")
println("  TB v7:   QED 0.758 (+1.5%) at iter 25  [freeze GRU, constructive-only]")
if best_iter > 0 && !isempty(checkpoint_results)
    bs = checkpoint_results[best_iter]
    @printf("  TB v8:   QED %.3f (%+.1f%%) at iter %d  [freeze GRU, full TB]\n",
        bs.mean_qed, 100.0*(bs.mean_qed - baseline.mean_qed)/baseline.mean_qed, best_iter)
end
println("\nDone!"); flush(stdout)
