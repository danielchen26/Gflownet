#!/usr/bin/env julia
# TB Fine-Tuning v10: Partial Unfreeze (gru_3 + output) + Reward-Weighted TB
#
# v9 results: reward-weighted TB achieved QED 0.797 (+3.3%) at iter 175.
# 2× better than v7 (+1.5%), but still half of RWMLE's +9%.
#
# Hypothesis: output-layer-only (334K params, 7.4%) has limited capacity
# to reshape the distribution. Unfreezing gru_3 (top GRU layer) adds ~2M params
# for ~7× more capacity. The top GRU layer is more about output distribution
# shaping than low-level grammar encoding.
#
# v10 strategy:
#   - freeze_gru=true + unfreeze_top_gru=true (unfreeze gru_3 + output)
#   - constructive_only=true (grammar-preserving direction)
#   - reward_weighted=true (RWMLE-style focus)
#   - Lower LR=3e-5 (more params being updated = need less aggressive LR)
#   - KL=1.0 (higher regularization since gru_3 is unfrozen)
#   - β=8.0

using Pkg; Pkg.activate(".")
using GFlowNet
using Serialization
using Statistics
using Printf
using Random

flush(stdout)

println("=" ^ 70)
println("TB Fine-Tuning v10: Partial Unfreeze (gru_3) + Reward-Weighted TB")
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

# Count parameters by layer
function count_params(x)
    if x isa AbstractArray; return length(x)
    elseif x isa NamedTuple; return sum(count_params(v) for v in values(x))
    else; return 0; end
end
total_p = count_params(pretrained.params)
output_p = count_params(pretrained.params.output)
gru3_p = count_params(pretrained.params.gru_3)
println("  Total params: $total_p")
println("  Output layer: $output_p ($(round(100.0*output_p/total_p, digits=1))%)")
println("  GRU-3 params: $gru3_p ($(round(100.0*gru3_p/total_p, digits=1))%)")
println("  Unfrozen (gru_3 + output): $(gru3_p + output_p) ($(round(100.0*(gru3_p+output_p)/total_p, digits=1))%)")
println("  Frozen (embed + gru + gru_2): $(total_p - gru3_p - output_p)")

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

# v10 Configuration
println("\n" * "=" ^ 70)
println("v10: PARTIAL UNFREEZE (gru_3 + output) + REWARD-WEIGHTED TB")
println("=" ^ 70)

ref_params = deepcopy(pretrained.params)
ref_states = deepcopy(init_states)

seg_config = FinetuningConfig(;
    n_iterations=25,
    sample_batch_size=32,
    learning_rate=3e-5,          # Lower than v9's 5e-5 (more params = less LR)
    gradient_clip_norm=1.0,
    kl_weight=1.0,               # Higher KL: gru_3 unfrozen needs more regularization
    kl_decay_schedule=:none,
    loss_type=:shifted_cosh,
    cosh_threshold=2.0,
    max_length=150,
    temperature=1.0,
    epsilon=0.05,
    log_frequency=5,
    reward_exponent=8.0,
    min_reward=0.01,
    training_mode=:tb,
    constructive_only=true,
    freeze_gru=true,
    reward_weighted=true,
    unfreeze_top_gru=true        # NEW: also unfreeze gru_3
)

println("  LR=$(seg_config.learning_rate)  (lower: more params being updated)")
println("  KL=$(seg_config.kl_weight)  (higher: gru_3 needs regularization)")
println("  beta=$(seg_config.reward_exponent)")
println("  FREEZE-GRU: YES (but gru_3 unfrozen)")
println("  UNFREEZE-TOP-GRU: YES  ← new in v10")
println("  CONSTRUCTIVE-ONLY: YES")
println("  REWARD-WEIGHTED: YES")
println("  Total: 8 × 25 = 200 iters")
flush(stdout)

n_segments = 8
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
    print_eval("v10_iter$total_iter", stats)
    checkpoint_results[total_iter] = stats

    if stats.mean_qed > best_qed && stats.validity > 70.0
        global best_qed = stats.mean_qed
        global best_iter = total_iter
        global best_params = deepcopy(current_params)
    end

    # Save checkpoint
    checkpoint_dir = "checkpoints/finetune_tb_v10"
    mkpath(checkpoint_dir)
    open("$checkpoint_dir/tb_v10_iter$(total_iter).jls", "w") do f
        Serialization.serialize(f, Dict("params" => current_params, "iter" => total_iter,
            "eval" => (mean_qed=stats.mean_qed, validity=stats.validity,
                      geq_07=stats.geq_07, geq_08=stats.geq_08, geq_09=stats.geq_09)))
    end
    println("  Saved checkpoint"); flush(stdout)

    # Early stopping
    if stats.validity < 60.0
        println("\nEARLY STOP: Validity collapsed to $(round(stats.validity, digits=1))%")
        flush(stdout); break
    end
end

# Save best
if best_params !== nothing
    checkpoint_dir = "checkpoints/finetune_tb_v10"
    open("$checkpoint_dir/tb_v10_best.jls", "w") do f
        Serialization.serialize(f, Dict("params" => best_params, "iter" => best_iter, "mean_qed" => best_qed))
    end
end

# Summary
println("\n" * "=" ^ 70)
println("SUMMARY: v10 Partial Unfreeze + Reward-Weighted TB")
println("=" ^ 70)
@printf("  %-25s  valid%%  unique  QED_mean  >=0.7  >=0.8  >=0.9\n", "Checkpoint")
@printf("  %-25s  ------  ------  --------  -----  -----  -----\n", "-" ^ 25)
@printf("  %-25s  %5.1f%%  %5d   %6.3f   %5d  %5d  %5d\n",
    "pretrained", baseline.validity, baseline.n_unique,
    baseline.mean_qed, baseline.geq_07, baseline.geq_08, baseline.geq_09)
for iter in sort(collect(keys(checkpoint_results)))
    s = checkpoint_results[iter]; delta = s.mean_qed - baseline.mean_qed
    @printf("  %-25s  %5.1f%%  %5d   %6.3f   %5d  %5d  %5d  (delta=%+.3f)\n",
        "v10 iter $iter", s.validity, s.n_unique, s.mean_qed, s.geq_07, s.geq_08, s.geq_09, delta)
end
println("\n  Best: iter $best_iter, QED $(round(best_qed, digits=3))")
println("\nComparison:")
println("  RWMLE:   QED 0.823 (+9%)")
println("  TB v7:   QED 0.758 (+1.5%) at iter 25   [freeze ALL GRU]")
println("  TB v9:   QED 0.797 (+3.3%) at iter 175  [freeze ALL GRU + reward-weighted]")
if best_iter > 0 && !isempty(checkpoint_results)
    bs = checkpoint_results[best_iter]
    @printf("  TB v10:  QED %.3f (%+.1f%%) at iter %d   [partial unfreeze gru_3 + reward-weighted]\n",
        bs.mean_qed, 100.0*(bs.mean_qed - baseline.mean_qed)/baseline.mean_qed, best_iter)
end
println("\nDone!"); flush(stdout)
