#!/usr/bin/env julia
# Test constrained decoding + temperature sweep
using GFlowNet, Random, Lux, PythonCall

function test_mode(model, params, st, vocab, rdkit, total; constrained, temp=1.0)
    valid = 0
    unique_smiles = Set{String}()
    error_counts = Dict{String,Int}()

    for i in 1:total
        smi, _, _ = sample_smiles_autoregressive(model, params, st, vocab;
            max_length=150, temperature=temp, constrained=constrained)
        if !isempty(smi)
            mol = rdkit.MolFromSmiles(smi)
            if !pyis(mol, pybuiltins.None)
                valid += 1
                push!(unique_smiles, smi)
            else
                # Classify
                open_p = count(c -> c == '(', smi)
                close_p = count(c -> c == ')', smi)
                if open_p != close_p
                    error_counts["parens"] = get(error_counts, "parens", 0) + 1
                else
                    ring_digits = Dict{Char,Int}()
                    for c in smi
                        isdigit(c) && (ring_digits[c] = get(ring_digits, c, 0) + 1)
                    end
                    if any(v -> v % 2 != 0, values(ring_digits))
                        error_counts["rings"] = get(error_counts, "rings", 0) + 1
                    else
                        error_counts["kekulize"] = get(error_counts, "kekulize", 0) + 1
                    end
                end
            end
        end
    end
    uniqueness = length(unique_smiles) / max(valid, 1)
    return valid, uniqueness, error_counts
end

function main()
    rdkit = pyimport("rdkit.Chem")

    ckpt = load_pretrained_checkpoint("checkpoints/pretrain/final.jls")
    smiles_list = load_zinc_smiles("data/zinc/250k_rndm_zinc_drugs_clean_3.csv"; max_molecules=5000)
    vocab = SMILESVocabulary()
    ds = prepare_zinc_dataset(vocab, smiles_list)
    model, _, st = create_smiles_policy(; vocab_size=vocab.size, hidden_dim=512, embed_dim=128, n_layers=3)

    total = 500

    configs = [
        ("Unconstrained T=1.0",  false, 1.0),
        ("Constrained T=1.0",    true,  1.0),
        ("Constrained T=0.9",    true,  0.9),
        ("Constrained T=0.8",    true,  0.8),
        ("Constrained T=0.7",    true,  0.7),
        ("Constrained T=0.6",    true,  0.6),
        ("Constrained T=0.5",    true,  0.5),
    ]

    println("=" ^ 75)
    println("Validity + Diversity Sweep (n=$total each)")
    println("=" ^ 75)
    println()

    for (name, con, temp) in configs
        print("$name... ")
        valid, uniq, errs = test_mode(model, ckpt.params, st, vocab, rdkit, total;
            constrained=con, temp=temp)
        pct = round(100*valid/total, digits=1)
        uniq_pct = round(100*uniq, digits=1)
        println("$pct% valid, $uniq_pct% unique")

        err_parts = String[]
        for k in ["parens", "rings", "kekulize"]
            if haskey(errs, k)
                push!(err_parts, "$k=$(errs[k])")
            end
        end
        if !isempty(err_parts)
            println("    errors: $(join(err_parts, ", "))")
        end
    end
end

main()
