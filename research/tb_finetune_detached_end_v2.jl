#!/usr/bin/env julia
# TB Fine-Tuning with Detached-END v2: Extended Training
#
# v1 showed: validity PRESERVED (87-93%) but QED not improving yet.
# The issue: 25-iter segments reset Adam optimizer state, and 100 total iters
# isn't enough for TB to converge.
#
# v2 changes:
#   - Larger segments (100 iters each) to preserve Adam momentum
#   - 300 total iterations
#   - Slightly higher LR (3e-5 vs 1e-5) to speed convergence
#   - Lower KL (0.5 vs 1.0) to allow more TB deviation from pretrained

using Pkg; Pkg.activate(".")
using GFlowNet
using Serialization
using Statistics
using Printf
using Random

println("=" ^ 70)
println("TB Fine-Tuning with Detached-END v2 (Extended Training)")
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
# TB Fine-Tuning with Detached-END (v2: Extended)
# =============================================================================
println("\n" * "=" ^ 70)
println("TB FINE-TUNING v2: Extended (300 iters, LR=3e-5, KL=0.5)")
println("=" ^ 70)

ref_params = deepcopy(pretrained.params)
ref_states = deepcopy(init_states)

# Tuned config: higher LR, lower KL, same architecture
base_config = FinetuningConfig(;
    n_iterations=100,           # Per segment (100 iters = good Adam momentum)
    sample_batch_size=32,
    learning_rate=3e-5,         # 3x higher than v1 (1e-5) to speed convergence
    gradient_clip_norm=1.0,
    kl_weight=0.5,              # Half of v1 (1.0) — allow more TB deviation
    kl_decay_schedule=:none,
    loss_type=:shifted_cosh,
    cosh_threshold=2.0,
    max_length=150,
    temperature=1.0,
    epsilon=0.05,
    log_frequency=10,
    reward_exponent=4.0,
    min_reward=0.01,
    training_mode=:tb
)

println("  LR=$(base_config.learning_rate)")
println("  KL=$(base_config.kl_weight) (constant)")
println("  beta=$(base_config.reward_exponent)")
println("  DETACHED-END: YES")
println("  Segments: 3 x 100 iters = 300 total")

checkpoint_results = Dict{Int, Any}()
current_params = deepcopy(pretrained.params)

for (seg_idx, target_iter) in enumerate([100, 200, 300])
    println("\n" * "-" ^ 50)
    println("Segment $seg_idx: iters $(target_iter-99)-$target_iter")
    println("-" ^ 50)

    result = finetune_smiles_gflownet(
        model, vocab, current_params, init_states,
        ref_params, ref_states,
        qed_oracle, base_config;
        verbose=true
    )

    global current_params = result.params

    # Evaluate
    println("\n--- Evaluation at iter $target_iter ---")
    stats = evaluate_model(model, current_params, init_states, vocab, qed_oracle)
    print_eval("tb_v2_iter$target_iter", stats)
    checkpoint_results[target_iter] = stats

    # Early stopping
    if stats.mean_qed < baseline.mean_qed * 0.5
        println("\nEARLY STOP: QED collapsed")
        break
    end

    # Save checkpoint
    checkpoint_dir = "checkpoints/finetune_tb_detached_v2"
    mkpath(checkpoint_dir)
    open("$checkpoint_dir/tb_v2_iter$(target_iter).jls", "w") do f
        Serialization.serialize(f, Dict(
            "params" => current_params,
            "iter" => target_iter,
            "eval" => (mean_qed=stats.mean_qed, validity=stats.validity,
                      geq_07=stats.geq_07, geq_08=stats.geq_08, geq_09=stats.geq_09)
        ))
    end
    println("  Saved: $checkpoint_dir/tb_v2_iter$(target_iter).jls")
end

# =============================================================================
# Summary
# =============================================================================
println("\n" * "=" ^ 70)
println("SUMMARY: TB Fine-Tuning v2 (Detached-END, Extended)")
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
        "TB v2 iter $iter", s.validity, s.n_unique,
        s.mean_qed, s.geq_07, s.geq_08, s.geq_09, delta)
end

println("\nDone!")
