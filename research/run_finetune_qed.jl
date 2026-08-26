#!/usr/bin/env julia
# CAFE-GFN Fine-Tuning with QED Oracle
#
# Usage:
#   julia --project research/run_finetune_qed.jl          # Full run (500 iters)
#   julia --project research/run_finetune_qed.jl --quick   # Quick test (50 iters)

using GFlowNet
using Lux
using Random
using Statistics
using Serialization
using PythonCall

const QUICK_MODE = "--quick" in ARGS

const rdkit_chem = pyimport("rdkit.Chem")
const rdkit_qed  = pyimport("rdkit.Chem.QED")

function qed_oracle(smiles::String)::Float64
    mol = rdkit_chem.MolFromSmiles(smiles)
    if pyis(mol, pybuiltins.None)
        return 0.0
    end
    try
        return pyconvert(Float64, rdkit_qed.qed(mol))
    catch
        return 0.0
    end
end

function main()
    N_ITERATIONS    = QUICK_MODE ? 100  : 500
    SAMPLE_BATCH    = QUICK_MODE ? 16   : 32
    LEARNING_RATE   = 1e-5
    KL_WEIGHT       = 1.0           # Strong KL to prevent mode collapse
    KL_DECAY        = :none         # Keep KL constant (cosine→0 caused collapse)
    TEMPERATURE     = 1.0           # T=1.0 for training
    EVAL_TEMP       = 0.8           # T=0.8 for evaluation
    REWARD_EXP      = 2.0           # R^2 shaping
    MIN_REWARD      = 0.01          # Floor for invalid molecules
    COSH_THRESHOLD  = 2.0           # Sane threshold for shifted-cosh
    LOG_FREQ        = QUICK_MODE ? 20   : 25
    CHECKPOINT_DIR  = "checkpoints/finetune_qed"
    PRETRAIN_CKPT   = "checkpoints/pretrain/final.jls"
    ZINC_PATH       = "data/zinc/250k_rndm_zinc_drugs_clean_3.csv"
    VOCAB_BUILD_N   = QUICK_MODE ? 5000 : 50000

    println("QED oracle ready (aspirin=$(round(qed_oracle("CC(=O)Oc1ccccc1C(=O)O"), digits=3)))")

    # Load pretrained model
    ckpt = load_pretrained_checkpoint(PRETRAIN_CKPT)
    smiles_list = load_zinc_smiles(ZINC_PATH; max_molecules=VOCAB_BUILD_N)
    vocab = SMILESVocabulary()
    ds = prepare_zinc_dataset(vocab, smiles_list)
    println("Vocab: $(vocab.size) tokens from $(length(smiles_list)) molecules")

    model, _, st = create_smiles_policy(;
        vocab_size=vocab.size, hidden_dim=512, embed_dim=128, n_layers=3
    )
    params = ckpt.params
    states = st
    ref_params = deepcopy(params)
    ref_states = deepcopy(states)

    # Pre-flight baseline
    println("\n--- Baseline (T=$EVAL_TEMP, constrained=true) ---")
    baseline_valid = 0
    baseline_qeds = Float64[]
    for i in 1:100
        smi, _, _ = sample_smiles_autoregressive(model, params, states, vocab;
            max_length=150, temperature=EVAL_TEMP, constrained=true)
        if !isempty(smi)
            r = qed_oracle(smi)
            push!(baseline_qeds, r)
            if r > 0.0; baseline_valid += 1; end
        end
    end
    valid_base = filter(>(0.0), baseline_qeds)
    println("  Valid: $baseline_valid/100, QED mean=$(round(mean(valid_base), digits=3)), ≥0.7: $(count(≥(0.7), valid_base))")

    # Fine-tune
    println("\n" * "=" ^ 60)
    println("Fine-Tuning: iters=$N_ITERATIONS, batch=$SAMPLE_BATCH, LR=$LEARNING_RATE")
    println("  R^$REWARD_EXP, min_R=$MIN_REWARD, cosh_th=$COSH_THRESHOLD")
    println("  KL=$KL_WEIGHT ($KL_DECAY), T_train=$TEMPERATURE")
    println("=" ^ 60)

    config = FinetuningConfig(
        n_iterations      = N_ITERATIONS,
        sample_batch_size = SAMPLE_BATCH,
        learning_rate     = LEARNING_RATE,
        kl_weight         = KL_WEIGHT,
        kl_decay_schedule = KL_DECAY,
        loss_type         = :shifted_cosh,
        cosh_threshold    = COSH_THRESHOLD,
        temperature       = TEMPERATURE,
        epsilon           = 0.02,
        log_frequency     = LOG_FREQ,
        reward_exponent   = REWARD_EXP,
        min_reward        = MIN_REWARD,
    )

    t_start = time()
    result = finetune_smiles_gflownet(
        model, vocab, params, states,
        ref_params, ref_states,
        qed_oracle,
        config;
        log_Z_init=0.0,
        verbose=true,
    )
    elapsed = (time() - t_start) / 60.0
    println("  Time: $(round(elapsed, digits=1)) min")

    # Post-training evaluation
    println("\n" * "=" ^ 60)
    println("Post-Training (T=$EVAL_TEMP, constrained=true)")
    println("=" ^ 60)

    eval_valid = 0
    eval_unique = Set{String}()
    eval_qeds = Float64[]
    eval_smiles = String[]

    for i in 1:200
        smi, _, _ = sample_smiles_autoregressive(model, result.params, states, vocab;
            max_length=150, temperature=EVAL_TEMP, constrained=true)
        if !isempty(smi)
            r = qed_oracle(smi)
            push!(eval_qeds, r)
            if r > 0.0
                eval_valid += 1
                push!(eval_unique, smi)
                if length(eval_smiles) < 20; push!(eval_smiles, smi); end
            end
        end
    end

    valid_eval = filter(>(0.0), eval_qeds)
    println("  Valid: $eval_valid/200 ($(round(100*eval_valid/200, digits=1))%)")
    println("  Unique: $(length(eval_unique))/$eval_valid")
    if !isempty(valid_eval)
        println("  QED mean=$(round(mean(valid_eval), digits=3)), max=$(round(maximum(valid_eval), digits=3))")
        println("  QED ≥0.7: $(count(≥(0.7), valid_eval)), ≥0.8: $(count(≥(0.8), valid_eval)), ≥0.9: $(count(≥(0.9), valid_eval))")
    end

    println("\nTop molecules:")
    scored = [(smi, qed_oracle(smi)) for smi in eval_smiles]
    sort!(scored, by=x->-x[2])
    for (i, (smi, r)) in enumerate(scored[1:min(10, length(scored))])
        println("  $i. QED=$(round(r, digits=3)) | $smi")
    end

    h = result.history
    if length(h.combined_losses) > 10
        println("\nTraining: loss $(round(h.combined_losses[1], digits=2)) → $(round(h.combined_losses[end], digits=2))")
        println("  reward $(round(h.mean_rewards[1], digits=4)) → $(round(h.mean_rewards[end], digits=4))")
        println("  log_Z $(round(h.log_Z_values[1], digits=2)) → $(round(h.log_Z_values[end], digits=2))")
        println("  unique: $(length(h.unique_smiles))")
    end

    # Save
    mkpath(CHECKPOINT_DIR)
    ckpt_path = joinpath(CHECKPOINT_DIR, "finetune_qed_final.jls")
    Serialization.serialize(ckpt_path, Dict(
        "params" => result.params, "states" => states, "log_Z" => result.log_Z,
        "history" => result.history, "vocab_size" => vocab.size,
        "vocab_token_to_idx" => vocab.token_to_idx,
        "vocab_idx_to_token" => vocab.idx_to_token,
        "config" => Dict("oracle" => "QED", "n_iterations" => N_ITERATIONS,
            "learning_rate" => LEARNING_RATE, "kl_weight" => KL_WEIGHT,
            "reward_exponent" => REWARD_EXP, "min_reward" => MIN_REWARD,
            "cosh_threshold" => COSH_THRESHOLD),
    ))
    println("\nSaved: $ckpt_path")
end

main()
