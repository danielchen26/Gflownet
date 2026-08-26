# ZINC 250K Data Loader for CAFE-GFN Pretraining
# Loads and preprocesses SMILES strings from the ZINC 250K dataset
#
# The ZINC 250K dataset contains ~250,000 drug-like molecules
# commonly used for molecular generation benchmarks.

# =============================================================================
# Data Loading
# =============================================================================

"""
    load_zinc_smiles(filepath::String; max_molecules::Int=0)

Load SMILES strings from a ZINC 250K CSV/TSV file.

# Arguments
- `filepath`: Path to the ZINC data file (CSV with 'smiles' column, or one SMILES per line)
- `max_molecules`: Maximum number of molecules to load (0 = all)

# Returns
Vector{String} of SMILES strings
"""
function load_zinc_smiles(filepath::String; max_molecules::Int=0)::Vector{String}
    if !isfile(filepath)
        error("ZINC data file not found: $filepath. " *
              "Download from: https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv")
    end

    smiles_list = String[]
    open(filepath) do f
        header = readline(f)  # Skip header

        # Detect format: CSV with 'smiles' column or plain SMILES
        header_lower = lowercase(header)
        smiles_col_idx = 1  # Default: first column

        if occursin(",", header)
            # CSV format — find the smiles column
            cols = split(header, ",")
            for (i, col) in enumerate(cols)
                if lowercase(strip(col, ['"', ' '])) == "smiles"
                    smiles_col_idx = i
                    break
                end
            end
        elseif !occursin('\t', header) && !occursin(',', header)
            # Single column, no header — the first line IS a SMILES
            push!(smiles_list, strip(header))
        end

        for line in eachline(f)
            line = strip(line)
            if isempty(line)
                continue
            end

            smiles = if occursin(",", line)
                parts = split(line, ",")
                if smiles_col_idx <= length(parts)
                    strip(parts[smiles_col_idx], ['"', ' '])
                else
                    continue
                end
            elseif occursin("\t", line)
                strip(split(line, "\t")[1], ['"', ' '])
            else
                strip(line, ['"', ' '])
            end

            if !isempty(smiles)
                push!(smiles_list, smiles)
            end

            if max_molecules > 0 && length(smiles_list) >= max_molecules
                break
            end
        end
    end

    return smiles_list
end

# =============================================================================
# Dataset Preparation
# =============================================================================

"""
    prepare_zinc_dataset(vocab::SMILESVocabulary, smiles_list::Vector{String};
                         max_length::Int=150, min_length::Int=3)

Tokenize and filter SMILES strings for training.

# Arguments
- `vocab`: SMILES vocabulary (will be expanded with new tokens)
- `smiles_list`: Raw SMILES strings
- `max_length`: Maximum token sequence length (including START/END)
- `min_length`: Minimum token sequence length (filters too-short molecules)

# Returns
Named tuple with:
- `sequences`: Vector{Vector{Int}} of token sequences
- `vocab`: Updated vocabulary
- `stats`: Dict with dataset statistics
"""
function prepare_zinc_dataset(vocab::SMILESVocabulary, smiles_list::Vector{String};
                              max_length::Int=150, min_length::Int=3)
    sequences = Vector{Int}[]
    skipped_long = 0
    skipped_short = 0
    total_tokens = 0

    for smiles in smiles_list
        tokens = encode(vocab, smiles; add_special_tokens=true)

        if length(tokens) > max_length
            skipped_long += 1
            continue
        end

        if length(tokens) < min_length
            skipped_short += 1
            continue
        end

        push!(sequences, tokens)
        total_tokens += length(tokens)
    end

    stats = Dict{String, Any}(
        "total_loaded" => length(smiles_list),
        "total_valid" => length(sequences),
        "skipped_long" => skipped_long,
        "skipped_short" => skipped_short,
        "vocab_size" => vocab.size,
        "avg_length" => total_tokens / max(length(sequences), 1),
        "max_observed_length" => isempty(sequences) ? 0 : maximum(length.(sequences)),
    )

    return (sequences=sequences, vocab=vocab, stats=stats)
end

# =============================================================================
# Checkpoint Loading
# =============================================================================

"""
    load_pretrained_checkpoint(filepath::String)

Load a pretrained SMILES GFlowNet checkpoint.

# Returns
Named tuple with: params, states, log_Z, vocab_size, vocab (if saved), history, config
"""
function load_pretrained_checkpoint(filepath::String)
    ckpt = Serialization.deserialize(filepath)

    # Reconstruct vocab if mappings were saved
    vocab = if haskey(ckpt, "vocab_token_to_idx") && haskey(ckpt, "vocab_idx_to_token")
        v = SMILESVocabulary()
        v.token_to_idx = ckpt["vocab_token_to_idx"]
        v.idx_to_token = ckpt["vocab_idx_to_token"]
        v.size = ckpt["vocab_size"]
        v
    else
        nothing  # Old checkpoint without vocab — must rebuild from data
    end

    return (
        params=ckpt["params"],
        states=get(ckpt, "states", nothing),
        log_Z=get(ckpt, "log_Z", 0.0),
        vocab_size=ckpt["vocab_size"],
        vocab=vocab,
        history=get(ckpt, "history", nothing),
        config=get(ckpt, "config", nothing),
    )
end

# =============================================================================
# Batch Iterator
# =============================================================================

"""
    create_batch_iterator(sequences::Vector{Vector{Int}}, batch_size::Int;
                          max_length::Int=150, shuffle::Bool=true)

Create a batched iterator over tokenized sequences.

# Returns
Channel that yields (input_batch, target_batch, lengths) tuples where:
- input_batch: Matrix{Int} (max_length, batch_size) — input tokens
- target_batch: Matrix{Int} (max_length, batch_size) — target tokens (shifted by 1)
- lengths: Vector{Int} — actual sequence lengths
"""
function create_batch_iterator(sequences::Vector{Vector{Int}}, batch_size::Int;
                               max_length::Int=150, shuffle::Bool=true)
    indices = collect(1:length(sequences))
    if shuffle
        Random.shuffle!(indices)
    end

    batches = []
    for start_idx in 1:batch_size:length(indices)
        end_idx = min(start_idx + batch_size - 1, length(indices))
        batch_indices = indices[start_idx:end_idx]
        actual_batch_size = length(batch_indices)

        input_batch = fill(PAD_TOKEN, max_length, actual_batch_size)
        target_batch = fill(PAD_TOKEN, max_length, actual_batch_size)
        lengths = Int[]

        for (j, idx) in enumerate(batch_indices)
            seq = sequences[idx]
            seq_len = min(length(seq), max_length + 1)

            # Input: all tokens except last
            input_len = min(seq_len - 1, max_length)
            input_batch[1:input_len, j] = seq[1:input_len]

            # Target: all tokens except first (shifted by 1)
            target_len = min(seq_len - 1, max_length)
            target_batch[1:target_len, j] = seq[2:target_len+1]

            push!(lengths, input_len)
        end

        push!(batches, (input=input_batch, target=target_batch, lengths=lengths))
    end

    return batches
end
