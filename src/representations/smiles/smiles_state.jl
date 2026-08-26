# SMILES State and Action Types for CAFE-GFN
# Implements the GFlowNet AbstractState/AbstractAction interface for SMILES-level generation
#
# Key design: P_B = 1 (uniform backward policy) because the SMILES DAG is a tree —
# each prefix has exactly one parent (remove the last token).

using ..GFlowNet: AbstractState, AbstractAction

# =============================================================================
# SMILES State
# =============================================================================

"""
    SMILESState <: AbstractState

State representing a partial SMILES string under construction.

# Fields
- `tokens::Vector{Int}` — Token index sequence (immutable for Zygote safety)
- `max_length::Int` — Maximum allowed sequence length
- `vocab_size::Int` — Size of the vocabulary (for action space)

# GFlowNet Properties
- Initial state: [START_TOKEN] (single start token)
- Terminal state: last token is END_TOKEN, or length >= max_length
- P_B = 1: SMILES DAG is a tree (each prefix has exactly 1 parent)
"""
struct SMILESState <: AbstractState
    tokens::Vector{Int}
    max_length::Int
    vocab_size::Int
end

"""Create initial SMILES state with just the START token."""
function create_initial_smiles_state(; max_length::Int=150, vocab_size::Int=100)
    return SMILESState([START_TOKEN], max_length, vocab_size)
end

# Equality and hashing (required by AbstractState interface)
Base.:(==)(a::SMILESState, b::SMILESState) = a.tokens == b.tokens
Base.hash(a::SMILESState, h::UInt) = hash(a.tokens, h)

# =============================================================================
# SMILES Action
# =============================================================================

"""
    SMILESTokenAction <: AbstractAction

Action that appends a single token to the SMILES sequence.

# Fields
- `token_idx::Int` — Vocabulary index of the token to append
"""
struct SMILESTokenAction <: AbstractAction
    token_idx::Int
end

Base.:(==)(a::SMILESTokenAction, b::SMILESTokenAction) = a.token_idx == b.token_idx
Base.hash(a::SMILESTokenAction, h::UInt) = hash(a.token_idx, h)

"""Create all possible token actions for a vocabulary."""
function create_smiles_actions(vocab_size::Int)::Vector{SMILESTokenAction}
    # Actions for all non-special tokens (skip PAD=0, START=1, but include END=2)
    # END action is how the model terminates generation
    actions = SMILESTokenAction[]
    for idx in 2:(vocab_size-1)  # Include END (2), skip PAD (0) and START (1)
        push!(actions, SMILESTokenAction(idx))
    end
    return actions
end

# =============================================================================
# GFlowNet Interface Implementation
# =============================================================================

"""Check if the SMILES state is terminal (END token reached or max length)."""
function GFlowNet.is_terminal_state(state::SMILESState)::Bool
    if isempty(state.tokens)
        return false
    end
    # Terminal if last token is END or max length reached
    return state.tokens[end] == END_TOKEN || length(state.tokens) >= state.max_length
end

"""Check if a token action is applicable to the current state."""
function GFlowNet.is_applicable(action::SMILESTokenAction, state::SMILESState)::Bool
    # Cannot act on terminal states
    if GFlowNet.is_terminal_state(state)
        return false
    end
    # Cannot add START token
    if action.token_idx == START_TOKEN
        return false
    end
    # Cannot add PAD token
    if action.token_idx == PAD_TOKEN
        return false
    end
    # All other tokens (including END) are valid
    return action.token_idx < state.vocab_size
end

"""
    apply_action(action::SMILESTokenAction, state::SMILESState) → SMILESState

Append a token to the SMILES sequence (pure functional, no mutation for Zygote).
"""
function GFlowNet.apply_action(action::SMILESTokenAction, state::SMILESState)::SMILESState
    new_tokens = vcat(state.tokens, [action.token_idx])
    return SMILESState(new_tokens, state.max_length, state.vocab_size)
end

"""
    state_to_features(state::SMILESState) → Vector{Float32}

Convert SMILES state to feature vector for neural network input.

For GRU-based policy, this returns the last token as a one-hot vector.
The GRU maintains hidden state across the sequence internally.
"""
function GFlowNet.state_to_features(state::SMILESState)::Vector{Float32}
    # One-hot encoding of the last token
    features = zeros(Float32, state.vocab_size)
    if !isempty(state.tokens)
        last_token = state.tokens[end]
        if 0 <= last_token < state.vocab_size
            features[last_token + 1] = 1.0f0  # 1-indexed
        end
    end
    return features
end

"""
    reward(state::SMILESState) → Float64

Placeholder reward function. Returns 1.0 for any terminal state.
The actual reward is computed by the oracle (e.g., docking score, QED)
and will be set externally during training.
"""
function GFlowNet.reward(state::SMILESState)::Float64
    if !GFlowNet.is_terminal_state(state)
        return 0.0
    end
    return 1.0  # Placeholder — overridden by oracle during training
end

# =============================================================================
# SMILES Conversion Utilities
# =============================================================================

"""
    state_to_smiles(state::SMILESState, vocab::SMILESVocabulary)

Convert a SMILES state back to a SMILES string.
"""
function state_to_smiles(state::SMILESState, vocab::SMILESVocabulary)::String
    return decode(vocab, state.tokens; strip_special=true)
end

"""
    smiles_to_state(vocab::SMILESVocabulary, smiles::String;
                    max_length::Int=150)

Convert a SMILES string to a terminal SMILES state.
"""
function smiles_to_state(vocab::SMILESVocabulary, smiles::String;
                         max_length::Int=150)::SMILESState
    tokens = encode(vocab, smiles; add_special_tokens=true)
    return SMILESState(tokens, max_length, vocab.size)
end

"""
    get_token_sequence(state::SMILESState)

Get the token sequence without special tokens (for display/debug).
"""
function get_token_sequence(state::SMILESState)::Vector{Int}
    return filter(t -> t != PAD_TOKEN && t != START_TOKEN && t != END_TOKEN, state.tokens)
end

"""Current sequence length (excluding special tokens)."""
sequence_length(state::SMILESState) = length(get_token_sequence(state))
