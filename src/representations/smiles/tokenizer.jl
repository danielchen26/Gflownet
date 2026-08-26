# SMILES Tokenizer for CAFE-GFN
# Regex-based tokenizer for converting SMILES strings to token sequences
#
# Reference: Weininger (1988) SMILES specification
# The tokenizer handles: atoms (C, N, O, S, F, Cl, Br, etc.), bonds (=, #, :),
# branches ((, )), ring closures (digits, %XX), charges (+, -), and bracket atoms [...]

# =============================================================================
# SMILES Vocabulary
# =============================================================================

"""
    SMILESVocabulary

Mapping between SMILES tokens and integer indices.

# Special Tokens
- PAD (0): Padding for fixed-length sequences
- START (1): Beginning of sequence
- END (2): End of sequence

# Token Categories
- Organic atoms: C, N, O, S, F, P, Cl, Br, I, B, Si, Se, etc.
- Aromatic atoms: c, n, o, s, p, se, te
- Bonds: -, =, #, :, /, \\
- Branches: (, )
- Ring closures: 0-9, %10-%99
- Bracket notation: [, ], +, -, @, @@, H
- Charges and counts embedded in bracket atoms
"""
mutable struct SMILESVocabulary
    token_to_idx::Dict{String, Int}
    idx_to_token::Dict{Int, String}
    size::Int

    function SMILESVocabulary()
        token_to_idx = Dict{String, Int}()
        idx_to_token = Dict{Int, String}()

        # Special tokens
        tokens = String[
            "[PAD]",   # 0
            "[START]", # 1
            "[END]",   # 2
        ]

        # Organic subset atoms (implicit H)
        append!(tokens, ["B", "C", "N", "O", "P", "S", "F", "I"])

        # Two-letter organic atoms
        append!(tokens, ["Cl", "Br", "Si", "Se"])

        # Aromatic atoms
        append!(tokens, ["b", "c", "n", "o", "p", "s", "se", "te"])

        # Bonds
        append!(tokens, ["-", "=", "#", ":", "/", "\\"])

        # Branches
        append!(tokens, ["(", ")"])

        # Ring closure digits
        append!(tokens, [string(d) for d in 0:9])

        # Two-digit ring closures (%10-%39 covers most cases)
        append!(tokens, ["%$(d)" for d in 10:39])

        # Bracket notation components
        append!(tokens, ["[", "]", "+", "++", "-", "--", "@", "@@", "H"])

        # Common bracket atoms (pre-tokenized for efficiency)
        append!(tokens, [
            "[C]", "[N]", "[O]", "[S]", "[P]", "[F]", "[Cl]", "[Br]", "[I]",
            "[C@@H]", "[C@H]", "[C@@]", "[C@]",
            "[N+]", "[N-]", "[O-]", "[S+]", "[S-]",
            "[nH]", "[NH]", "[OH]", "[SH]",
            "[Si]", "[Se]", "[B-]", "[n]", "[o]", "[s]",
            "[C-]", "[c]", "[NH2+]", "[NH3+]", "[n+]",
            "[2H]", "[3H]", "[11C]", "[13C]", "[14C]", "[15N]", "[18F]",
        ])

        # Build mappings (deduplicate)
        seen = Set{String}()
        idx = 0
        clean_tokens = String[]
        for t in tokens
            if !(t in seen)
                push!(seen, t)
                push!(clean_tokens, t)
                token_to_idx[t] = idx
                idx_to_token[idx] = t
                idx += 1
            end
        end

        new(token_to_idx, idx_to_token, idx)
    end
end

"""Number of tokens in vocabulary"""
Base.length(vocab::SMILESVocabulary) = vocab.size

"""Check if a token exists in the vocabulary"""
has_token(vocab::SMILESVocabulary, token::String) = haskey(vocab.token_to_idx, token)

"""Get index for a token, adding it if not present"""
function get_or_add_token!(vocab::SMILESVocabulary, token::String)::Int
    if haskey(vocab.token_to_idx, token)
        return vocab.token_to_idx[token]
    end
    # Add new token
    idx = vocab.size
    vocab.token_to_idx[token] = idx
    vocab.idx_to_token[idx] = token
    vocab.size += 1
    return idx
end

# Special token indices
const PAD_TOKEN = 0
const START_TOKEN = 1
const END_TOKEN = 2

# =============================================================================
# SMILES Tokenization Regex
# =============================================================================

# Regex pattern for splitting SMILES into tokens
# Order matters: try longer matches first
# 1. Bracket atoms: [...]  (captures entire bracket expression)
# 2. Two-letter organic atoms: Cl, Br, Si, Se
# 3. Stereo: @@, @
# 4. Two-digit ring closures: %XX
# 5. Single characters: atoms, bonds, branches, digits
const SMILES_TOKEN_REGEX = r"\[[^\]]+\]|Br|Cl|Si|Se|se|te|@@|@|%\d{2}|[BCNOPSFIbcnops]|[=#:/\\()\-\+]|\d|."

"""
    tokenize_smiles(smiles::String)

Split a SMILES string into individual tokens.

# Examples
```julia
tokenize_smiles("CCO")       # ["C", "C", "O"]
tokenize_smiles("c1ccccc1")  # ["c", "1", "c", "c", "c", "c", "c", "1"]
tokenize_smiles("[C@@H](O)F") # ["[C@@H]", "(", "O", ")", "F"]
```
"""
function tokenize_smiles(smiles::String)::Vector{String}
    tokens = String[]
    for m in eachmatch(SMILES_TOKEN_REGEX, smiles)
        push!(tokens, m.match)
    end
    return tokens
end

# =============================================================================
# Encode / Decode
# =============================================================================

"""
    encode(vocab::SMILESVocabulary, smiles::String; add_special_tokens=true)

Encode a SMILES string into a sequence of token indices.

# Arguments
- `vocab`: SMILES vocabulary
- `smiles`: SMILES string to encode
- `add_special_tokens`: Whether to prepend START and append END tokens

# Returns
Vector{Int} of token indices
"""
function encode(vocab::SMILESVocabulary, smiles::String; add_special_tokens::Bool=true)::Vector{Int}
    tokens = tokenize_smiles(smiles)
    indices = Int[]

    if add_special_tokens
        push!(indices, START_TOKEN)
    end

    for token in tokens
        idx = get_or_add_token!(vocab, token)
        push!(indices, idx)
    end

    if add_special_tokens
        push!(indices, END_TOKEN)
    end

    return indices
end

"""
    decode(vocab::SMILESVocabulary, indices::Vector{Int}; strip_special=true)

Decode a sequence of token indices back into a SMILES string.

# Arguments
- `vocab`: SMILES vocabulary
- `indices`: Vector of token indices
- `strip_special`: Whether to remove START, END, PAD tokens

# Returns
SMILES string
"""
function decode(vocab::SMILESVocabulary, indices::Vector{Int}; strip_special::Bool=true)::String
    tokens = String[]
    for idx in indices
        if strip_special && idx in (PAD_TOKEN, START_TOKEN, END_TOKEN)
            continue
        end
        if haskey(vocab.idx_to_token, idx)
            push!(tokens, vocab.idx_to_token[idx])
        end
    end
    return join(tokens)
end

"""
    pad_sequence(indices::Vector{Int}, max_length::Int; pad_value=PAD_TOKEN)

Pad or truncate a token sequence to fixed length.
"""
function pad_sequence(indices::Vector{Int}, max_length::Int; pad_value::Int=PAD_TOKEN)::Vector{Int}
    if length(indices) >= max_length
        return indices[1:max_length]
    else
        return vcat(indices, fill(pad_value, max_length - length(indices)))
    end
end

"""
    batch_encode(vocab::SMILESVocabulary, smiles_list::Vector{String};
                 max_length::Int=150, add_special_tokens::Bool=true)

Encode a batch of SMILES strings into a padded matrix.

# Returns
Matrix{Int} of shape (max_length, batch_size)
"""
function batch_encode(vocab::SMILESVocabulary, smiles_list::Vector{String};
                      max_length::Int=150, add_special_tokens::Bool=true)::Matrix{Int}
    batch_size = length(smiles_list)
    result = fill(PAD_TOKEN, max_length, batch_size)

    for (j, smiles) in enumerate(smiles_list)
        indices = encode(vocab, smiles; add_special_tokens=add_special_tokens)
        len = min(length(indices), max_length)
        result[1:len, j] = indices[1:len]
    end

    return result
end
