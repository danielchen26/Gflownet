#!/usr/bin/env julia
# Deep diagnostic: What did TB fine-tuning actually do to the model?
#
# We compare pretrained vs finetuned at multiple levels:
# 1. Parameter-level: How much did weights change?
# 2. Token-level: How did conditional distributions shift?
# 3. Molecule-level: What changed about generated molecules?

using Pkg; Pkg.activate(".")
using GFlowNet
using Serialization
using Statistics
using NNlib
using Printf

println("Loading pretrained checkpoint...")
pretrained = load_pretrained_checkpoint("checkpoints/pretrain/final.jls")
println("Loading finetuned checkpoint...")
finetuned_ckpt = open("checkpoints/finetune_qed/finetune_qed_final.jls") do f
    Serialization.deserialize(f)
end

# Build vocab from ZINC
smiles_data = load_zinc_smiles("data/zinc/250k_rndm_zinc_drugs_clean_3.csv"; max_molecules=50000)
vocab = SMILESVocabulary()
prepare_zinc_dataset(vocab, smiles_data)

# Create model
model, _, init_states = create_smiles_policy(;
    vocab_size=vocab.size, hidden_dim=512, embed_dim=128, n_layers=3
)

pre_params = pretrained.params
ft_params = finetuned_ckpt["params"]

# =============================================================================
# 1. Parameter-level analysis: How much did each layer change?
# =============================================================================
println("\n" * "=" ^ 70)
println("1. PARAMETER DRIFT ANALYSIS")
println("=" ^ 70)

function param_stats(pre, ft, name)
    pre_flat = collect(Iterators.flatten(pre))
    ft_flat = collect(Iterators.flatten(ft))
    if length(pre_flat) != length(ft_flat)
        println("  $name: SIZE MISMATCH $(length(pre_flat)) vs $(length(ft_flat))")
        return
    end
    diff = ft_flat .- pre_flat
    rel_diff = abs.(diff) ./ (abs.(pre_flat) .+ 1f-8)
    @printf("  %-20s  n=%7d  |Δ|_mean=%.6f  |Δ|_max=%.6f  rel_mean=%.4f%%  rel_max=%.2f%%\n",
        name, length(diff), mean(abs.(diff)), maximum(abs.(diff)),
        mean(rel_diff)*100, maximum(rel_diff)*100)
end

# Compare each parameter group
for key in keys(pre_params)
    pre_p = pre_params[key]
    ft_p = ft_params[key]
    if pre_p isa NamedTuple
        for subkey in keys(pre_p)
            param_stats(pre_p[subkey], ft_p[subkey], "$key.$subkey")
        end
    else
        param_stats(pre_p, ft_p, string(key))
    end
end

# =============================================================================
# 2. Token-level analysis: How did conditional distributions shift?
# =============================================================================
println("\n" * "=" ^ 70)
println("2. TOKEN-LEVEL DISTRIBUTION SHIFT")
println("=" ^ 70)

# Take 10 well-known SMILES and compute per-token KL divergence
test_smiles = [
    "c1ccccc1",         # benzene
    "CC(=O)O",          # acetic acid
    "CC(=O)Oc1ccccc1OC(C)=O",  # aspirin
    "CC(C)Cc1ccc(cc1)C(C)CC",   # ibuprofen-like
    "O=C(O)c1ccccc1O",   # salicylic acid
]

for smi in test_smiles
    tokens = encode(vocab, smi; add_special_tokens=true)
    if isempty(tokens) || length(tokens) < 3
        println("  $smi → encoding failed")
        continue
    end

    # Get per-step log probs under both models
    pre_log_prob, pre_steps = compute_log_probs_teacher_forced(model, tokens, pre_params, init_states)
    ft_log_prob, ft_steps = compute_log_probs_teacher_forced(model, tokens, ft_params, init_states)

    # Per-step KL divergence
    n_steps = min(length(pre_steps), length(ft_steps))
    if n_steps == 0
        continue
    end

    step_kl = [ft_steps[i] - pre_steps[i] for i in 1:n_steps]
    max_shift_idx = argmax(abs.(step_kl))
    max_shift_token = vocab.idx_to_token[tokens[max_shift_idx + 1]]  # +1 because targets are shifted

    @printf("  %-40s  log_PF: %.1f→%.1f (Δ=%.1f)  max_shift: step %d (token '%s', Δ=%.2f)\n",
        smi[1:min(40,length(smi))],
        Float64(pre_log_prob), Float64(ft_log_prob),
        Float64(ft_log_prob - pre_log_prob),
        max_shift_idx, max_shift_token, step_kl[max_shift_idx])
end

# =============================================================================
# 3. Full distribution comparison at each position
# =============================================================================
println("\n" * "=" ^ 70)
println("3. PER-POSITION FULL DISTRIBUTION SHIFT (benzene: c1ccccc1)")
println("=" ^ 70)

tokens = encode(vocab, "c1ccccc1"; add_special_tokens=true)
if !isempty(tokens)
    layers = model.layers
    input_tokens = tokens[1:end-1] .+ 1
    target_tokens = tokens[2:end]

    for (name, ps) in [("pretrained", pre_params), ("finetuned", ft_params)]
        println("\n  [$name] Per-step top-3 tokens:")
        embedded, _ = layers.embedding(input_tokens, ps.embedding, init_states.embedding)

        h1 = zeros(Float32, 512)
        h2 = zeros(Float32, 512)
        h3 = zeros(Float32, 512)

        for t in 1:length(target_tokens)
            embed_t = embedded[:, t]
            (h1, _), _ = layers.gru_cell((embed_t, (h1,)), ps.gru, init_states.gru)
            (h2, _), _ = layers.extra_gru_cells[1]((h1, (h2,)), ps.gru_2, init_states.gru_2)
            (h3, _), _ = layers.extra_gru_cells[2]((h2, (h3,)), ps.gru_3, init_states.gru_3)

            logits, _ = layers.output_dense(h3, ps.output, init_states.output)
            probs = softmax(logits)
            top3_idx = sortperm(collect(probs); rev=true)[1:3]

            target = target_tokens[t]
            target_tok = get(vocab.idx_to_token, target, "?")
            target_prob = probs[target + 1]

            top3_str = join([@sprintf("%s:%.3f", get(vocab.idx_to_token, i-1, "?"), probs[i])
                            for i in top3_idx], "  ")
            @printf("    step %2d: target='%s' P=%.4f  | top3: %s\n",
                t, target_tok, target_prob, top3_str)
        end
    end
end

# =============================================================================
# 4. Sampling comparison
# =============================================================================
println("\n" * "=" ^ 70)
println("4. SAMPLING COMPARISON (200 molecules each, T=0.8, constrained=true)")
println("=" ^ 70)

# QED oracle
using PythonCall
rdkit = pyimport("rdkit.Chem")
rdkit_qed = pyimport("rdkit.Chem.QED")
function qed_oracle(smi)
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

for (name, ps) in [("pretrained", pre_params), ("finetuned", ft_params)]
    qeds = Float64[]
    valid = 0
    total = 200
    for _ in 1:total
        smi, _, _ = sample_smiles_autoregressive(
            model, ps, init_states, vocab;
            max_length=150, temperature=0.8, constrained=true
        )
        q = qed_oracle(smi)
        if q > 0.0
            valid += 1
            push!(qeds, q)
        end
    end
    sort!(qeds; rev=true)
    n = length(qeds)
    @printf("  %-12s: valid=%d/%d (%.1f%%)  QED mean=%.3f  median=%.3f  ≥0.7=%d  ≥0.8=%d  ≥0.9=%d\n",
        name, valid, total, 100*valid/total,
        n > 0 ? mean(qeds) : 0.0,
        n > 0 ? qeds[div(n,2)+1] : 0.0,
        count(>=(0.7), qeds),
        count(>=(0.8), qeds),
        count(>=(0.9), qeds))
    if n >= 5
        println("    Top 5: ", join([@sprintf("%.3f", q) for q in qeds[1:5]], ", "))
    end
end

# =============================================================================
# 5. Specific failure modes
# =============================================================================
println("\n" * "=" ^ 70)
println("5. WHAT KIND OF MOLECULES DOES THE FINETUNED MODEL GENERATE?")
println("=" ^ 70)

println("\nFinetuned model samples (first 20):")
for i in 1:20
    smi, tokens, lp = sample_smiles_autoregressive(
        model, ft_params, init_states, vocab;
        max_length=150, temperature=0.8, constrained=true
    )
    q = qed_oracle(smi)
    @printf("  %2d. len=%-3d QED=%.3f | %s\n", i, length(smi), q, smi[1:min(80,length(smi))])
end

println("\nPretrained model samples (first 20):")
for i in 1:20
    smi, tokens, lp = sample_smiles_autoregressive(
        model, pre_params, init_states, vocab;
        max_length=150, temperature=0.8, constrained=true
    )
    q = qed_oracle(smi)
    @printf("  %2d. len=%-3d QED=%.3f | %s\n", i, length(smi), q, smi[1:min(80,length(smi))])
end
