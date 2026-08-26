#!/usr/bin/env julia
# CAFE-GFN Pretraining on ZINC 250K
#
# Usage:
#   julia --threads=16 research/run_pretrain.jl [--quick]
#
# Flags:
#   --quick    Run on 1000 molecules, 2 epochs (validation only, ~2 min)
#   (default)  Full ZINC 250K, 10 epochs (~2.5 hours on M4 Max)
#
# Output:
#   checkpoints/pretrain/checkpoint_iter{N}.jls
#   checkpoints/pretrain/final.jls

using Pkg
Pkg.activate(".")

using GFlowNet
using Lux, Zygote, Random, Statistics, Dates
using AppleAccelerate
using Serialization

println("=" ^ 60)
println("CAFE-GFN Pretraining — ZINC 250K")
println("=" ^ 60)
println("  Julia: $(VERSION)")
println("  Threads: $(Threads.nthreads())")
println("  Time: $(Dates.now())")

# =============================================================================
# Configuration
# =============================================================================

quick_mode = "--quick" in ARGS

ZINC_PATH = joinpath(@__DIR__, "..", "data", "zinc", "250k_rndm_zinc_drugs_clean_3.csv")
CHECKPOINT_DIR = joinpath(@__DIR__, "..", "checkpoints", "pretrain")
FINAL_CHECKPOINT = joinpath(CHECKPOINT_DIR, "final.jls")

if quick_mode
    println("\n⚡ QUICK MODE: 1000 molecules, 2 epochs")
    MAX_MOLECULES = 1000
    N_EPOCHS = 2
    BATCH_SIZE = 64
    TB_WEIGHT = 0.0  # MLE-only for quick validation
    LOG_FREQ = 5
    CHECKPOINT_FREQ = 0  # No checkpoints in quick mode
else
    println("\n📦 FULL MODE: ZINC 250K, 10 epochs")
    MAX_MOLECULES = 0  # All
    N_EPOCHS = 10
    BATCH_SIZE = 256
    TB_WEIGHT = 0.0  # Start with MLE-only (TB added in phase 2)
    LOG_FREQ = 100
    CHECKPOINT_FREQ = 500
end

# =============================================================================
# Data Loading
# =============================================================================

println("\n--- Loading ZINC Data ---")
@time raw_smiles = load_zinc_smiles(ZINC_PATH; max_molecules=MAX_MOLECULES)
println("  Loaded: $(length(raw_smiles)) SMILES")
println("  Examples: $(raw_smiles[1]), $(raw_smiles[2])")

# =============================================================================
# Vocabulary & Tokenization
# =============================================================================

println("\n--- Building Vocabulary ---")
vocab = SMILESVocabulary()

@time dataset = prepare_zinc_dataset(vocab, raw_smiles;
    max_length=150, min_length=3)

println("  Vocab size: $(dataset.stats["vocab_size"])")
println("  Valid sequences: $(dataset.stats["total_valid"]) / $(dataset.stats["total_loaded"])")
println("  Avg length: $(round(dataset.stats["avg_length"], digits=1))")
println("  Max length: $(dataset.stats["max_observed_length"])")
println("  Skipped (long): $(dataset.stats["skipped_long"])")
println("  Skipped (short): $(dataset.stats["skipped_short"])")

sequences = dataset.sequences

# =============================================================================
# Model Creation
# =============================================================================

println("\n--- Creating Model ---")
rng = Random.MersenneTwister(42)

model, ps, st = create_smiles_policy(
    vocab_size=vocab.size,
    embed_dim=128,
    hidden_dim=512,
    n_layers=3,
    rng=rng
)

n_params = sum(length(v) for v in Iterators.flatten(
    (values(ps.embedding), values(ps.gru), values(ps.gru_2),
     values(ps.gru_3), values(ps.output))
) if v isa AbstractArray)
println("  Architecture: Embedding($(vocab.size)→128) → GRU(128→512)×3 → Dense(512→$(vocab.size))")
println("  Parameters: ~$(round(n_params / 1000, digits=1))K")

# Quick forward pass sanity check
test_seqs = [sequences[1], sequences[2]]
test_loss = compute_mle_loss_batched(model, test_seqs, ps, st)
println("  Initial loss (2 seqs): $(round(test_loss, digits=4))")

# Gradient check
lv, gs = Zygote.withgradient(p -> compute_mle_loss_batched(model, test_seqs, p, st), ps)
println("  Gradient check: $(gs[1] !== nothing ? "OK" : "FAILED")")

# =============================================================================
# Pre-training Sampling Check
# =============================================================================

println("\n--- Pre-training Samples (random policy) ---")
for i in 1:5
    smi, tokens, lp = sample_smiles_autoregressive(model, ps, st, vocab; max_length=80)
    println("  [$i] \"$smi\" ($(length(tokens)) tokens, logP=$(round(lp, digits=2)))")
end

# =============================================================================
# Pretraining
# =============================================================================

println("\n--- Starting Pretraining ---")
config = PretrainingConfig(
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    tb_batch_size=16,
    learning_rate=1e-4,
    mle_weight=1.0,
    tb_weight=TB_WEIGHT,
    gradient_clip_norm=1.0,
    loss_type=:shifted_cosh,
    cosh_threshold=15.0,
    max_length=150,
    log_frequency=LOG_FREQ,
    checkpoint_frequency=CHECKPOINT_FREQ,
    checkpoint_dir=CHECKPOINT_DIR,
    use_batched_loss=true
)

@time result = pretrain_smiles_gflownet(model, vocab, ps, st, sequences, config)

# =============================================================================
# Post-training Evaluation
# =============================================================================

println("\n--- Post-training Evaluation ---")
trained_ps = result.params

# Sample molecules
println("  Sampling 200 molecules...")
n_eval = 200
valid_smiles = String[]
all_tokens = Vector{Int}[]
for _ in 1:n_eval
    smi, tokens, _ = sample_smiles_autoregressive(model, trained_ps, st, vocab; max_length=150)
    if !isempty(smi)
        push!(valid_smiles, smi)
        push!(all_tokens, tokens)
    end
end

non_empty_rate = length(valid_smiles) / n_eval * 100
unique_rate = length(Set(valid_smiles)) / max(length(valid_smiles), 1) * 100
avg_len = isempty(all_tokens) ? 0.0 : mean(length.(all_tokens))

println("  Non-empty: $(length(valid_smiles))/$n_eval ($(round(non_empty_rate, digits=1))%)")
println("  Unique: $(length(Set(valid_smiles))) ($(round(unique_rate, digits=1))%)")
println("  Avg token length: $(round(avg_len, digits=1))")

# Show sample molecules
println("\n  Sample generated molecules:")
for (i, smi) in enumerate(valid_smiles[1:min(10, length(valid_smiles))])
    println("    [$i] $smi")
end

# Loss comparison
final_loss = compute_mle_loss_batched(model, sequences[1:min(100, length(sequences))],
                                       trained_ps, st)
println("\n  Final MLE loss (100 seqs): $(round(final_loss, digits=4))")

# =============================================================================
# Save Final Checkpoint
# =============================================================================

println("\n--- Saving Final Checkpoint ---")
mkpath(CHECKPOINT_DIR)
Serialization.serialize(FINAL_CHECKPOINT, Dict(
    "params" => trained_ps,
    "states" => st,
    "log_Z" => result.log_Z,
    "history" => result.history,
    "vocab_size" => vocab.size,
    "vocab_token_to_idx" => vocab.token_to_idx,
    "vocab_idx_to_token" => vocab.idx_to_token,
    "config" => Dict(
        "n_epochs" => N_EPOCHS,
        "batch_size" => BATCH_SIZE,
        "n_sequences" => length(sequences),
    )
))
println("  Saved to: $FINAL_CHECKPOINT")

# =============================================================================
# Summary
# =============================================================================

println("\n" * "=" ^ 60)
println("PRETRAINING COMPLETE")
println("=" ^ 60)
hist = result.history
if !isempty(hist.combined_losses)
    println("  Initial loss: $(round(hist.combined_losses[1], digits=4))")
    println("  Final loss: $(round(hist.combined_losses[end], digits=4))")
    println("  Best loss: $(round(minimum(hist.combined_losses), digits=4))")
end
println("  Log Z: $(round(result.log_Z, digits=4))")
println("  Non-empty rate: $(round(non_empty_rate, digits=1))%")
println("  Unique molecules: $(length(Set(valid_smiles)))")
if !isempty(hist.step_times_ms)
    println("  Avg step time: $(round(mean(hist.step_times_ms), digits=1))ms")
    total_min = sum(hist.step_times_ms) / 60000
    println("  Total training time: $(round(total_min, digits=1)) min")
end
println("  Checkpoint: $FINAL_CHECKPOINT")
println("=" ^ 60)
