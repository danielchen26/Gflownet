# GRU Forward Policy for SMILES GFlowNet (CAFE-GFN)
# Autoregressive policy using Lux.jl GRU layers
#
# Architecture: Embedding → Multi-layer GRU → Dense → Logits
#
# Two modes of operation:
# 1. Teacher-forced (training): Full sequence processed via Lux.Recurrence (Zygote-safe)
# 2. Autoregressive (inference): Token-by-token with manual hidden state (outside Zygote)
#
# Key insight: P_B = 1 for SMILES (tree DAG), so we only need forward policy.

using Lux
using Random
using Zygote
using ComponentArrays
using NNlib

using ..GFlowNet: AbstractState, AbstractAction, ForwardPolicy

# =============================================================================
# Architecture Creation
# =============================================================================

"""
    create_smiles_gru_layers(vocab_size, embed_dim, hidden_dim, n_layers)

Create the individual Lux layers for a SMILES GRU policy.

# Returns
Named tuple of: (embedding, gru_cell, output_dense, hidden_proj)
"""
function create_smiles_gru_layers(;
    vocab_size::Int=100,
    embed_dim::Int=128,
    hidden_dim::Int=512,
    n_layers::Int=3,
    use_term_head::Bool=false
)
    # Embedding layer: token index → dense vector
    embedding = Lux.Embedding(vocab_size => embed_dim)

    # GRU cell: operates on embedded tokens
    # For multi-layer, we use a single large GRU and project between layers
    # Lux.GRUCell handles the recurrence internally
    gru_cell = Lux.GRUCell(embed_dim => hidden_dim)

    # Additional GRU layers (if n_layers > 1)
    extra_gru_cells = [Lux.GRUCell(hidden_dim => hidden_dim) for _ in 2:n_layers]

    # Output projection: hidden state → vocabulary logits
    output_dense = Lux.Chain(
        Lux.Dense(hidden_dim => hidden_dim, tanh),
        Lux.Dense(hidden_dim => vocab_size)  # Raw logits over vocabulary
    )

    # Separate termination head (optional, for TB-safe fine-tuning)
    # Decouples "when to stop" from "which atom to emit" so TB gradient
    # can boost atoms without suppressing END through softmax competition.
    term_head = use_term_head ? Lux.Dense(hidden_dim => 1) : nothing

    return (
        embedding=embedding,
        gru_cell=gru_cell,
        extra_gru_cells=extra_gru_cells,
        output_dense=output_dense,
        term_head=term_head,
        n_layers=n_layers,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )
end

"""
    create_smiles_policy(; vocab_size=100, embed_dim=128, hidden_dim=512, n_layers=3, rng=Random.default_rng())

Create a complete SMILES forward policy with initialized parameters.

# Returns
Tuple of (policy_model, parameters, states) following Lux conventions.
The policy_model is a NamedTuple containing the Lux layers.
Parameters and states are ComponentArrays/NamedTuples for Zygote.
"""
function create_smiles_policy(;
    vocab_size::Int=100,
    embed_dim::Int=128,
    hidden_dim::Int=512,
    n_layers::Int=3,
    use_term_head::Bool=false,
    rng=Random.default_rng()
)
    # Create layers
    layers = create_smiles_gru_layers(;
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        use_term_head=use_term_head
    )

    # Initialize parameters for each component
    embed_ps, embed_st = Lux.setup(rng, layers.embedding)
    gru_ps, gru_st = Lux.setup(rng, layers.gru_cell)
    output_ps, output_st = Lux.setup(rng, layers.output_dense)

    # Extra GRU layers
    extra_gru_ps = []
    extra_gru_st = []
    for cell in layers.extra_gru_cells
        ps, st = Lux.setup(rng, cell)
        push!(extra_gru_ps, ps)
        push!(extra_gru_st, st)
    end

    # Termination head (optional)
    term_ps = nothing
    term_st = nothing
    if use_term_head && !isnothing(layers.term_head)
        term_ps, term_st = Lux.setup(rng, layers.term_head)
    end

    # Build parameter and state NamedTuples
    if n_layers == 1
        params = (
            embedding=embed_ps,
            gru=gru_ps,
            output=output_ps,
        )
        states = (
            embedding=embed_st,
            gru=gru_st,
            output=output_st,
        )
    else
        # Create named entries for extra GRU layers
        key_names = ntuple(i -> Symbol("gru_$(i+1)"), n_layers - 1)
        extra_params = NamedTuple{key_names}(ntuple(i -> extra_gru_ps[i], n_layers - 1))
        extra_states = NamedTuple{key_names}(ntuple(i -> extra_gru_st[i], n_layers - 1))

        params = merge(
            (embedding=embed_ps, gru=gru_ps),
            extra_params,
            (output=output_ps,)
        )
        states = merge(
            (embedding=embed_st, gru=gru_st),
            extra_states,
            (output=output_st,)
        )
    end

    # Add termination head params if present
    if !isnothing(term_ps)
        params = merge(params, (term_head=term_ps,))
        states = merge(states, (term_head=term_st,))
    end

    # Create the policy model struct that holds layers + metadata
    policy_model = SMILESPolicyModel(layers, n_layers, vocab_size, embed_dim, hidden_dim)

    return policy_model, params, states
end

"""
    has_term_head(model)

Check if model has a separate termination head (for TB-safe fine-tuning).
Works with any model that has a `.layers` NamedTuple.
"""
has_term_head(model) =
    hasproperty(model.layers, :term_head) && !isnothing(model.layers.term_head)

# =============================================================================
# SMILES Policy Model (callable struct for Lux compatibility)
# =============================================================================

"""
    SMILESPolicyModel

Callable model struct wrapping the GRU SMILES policy.
Implements the Lux model interface: `(model)(x, ps, st) -> (y, st_new)`.

When called with a feature vector (from state_to_features), it returns
logits over the vocabulary. This is the "single-step" mode used by
the standard GFlowNet sampling loop.
"""
struct SMILESPolicyModel
    layers::NamedTuple
    n_layers::Int
    vocab_size::Int
    embed_dim::Int
    hidden_dim::Int
end

"""
    (model::SMILESPolicyModel)(features, ps, st)

Single-step forward pass: feature vector → logits.
This is called by the standard GFlowNet sampling loop via
`model.forward_policy.model(features, params.forward, model.states.forward)`.

The features are a one-hot vector of the last token. We embed it,
pass through GRU (using zero hidden state), and project to logits.

NOTE: For efficient autoregressive generation, use `sample_smiles_autoregressive`
which maintains the GRU hidden state across steps.
"""
function (model::SMILESPolicyModel)(features::AbstractVector, ps, st)
    layers = model.layers

    # Find the token index from one-hot features
    token_idx = Zygote.@ignore begin
        idx = argmax(features)
        idx  # 1-indexed position in one-hot
    end

    # Create token index array for embedding lookup
    token_array = Zygote.@ignore [token_idx]

    # Embed the token
    embedded, embed_st = layers.embedding(token_array, ps.embedding, st.embedding)
    # embedded is (embed_dim, 1) matrix → take column
    embed_vec = embedded[:, 1]

    # Single GRU step with zero hidden state
    hidden = zeros(Float32, model.hidden_dim)
    (new_hidden, _), gru_st = layers.gru_cell((embed_vec, (hidden,)), ps.gru, st.gru)

    # Extra GRU layers
    current_hidden = new_hidden
    for i in 2:model.n_layers
        layer_key = Symbol("gru_$i")
        layer_hidden = zeros(Float32, model.hidden_dim)
        (current_hidden, _), _ = layers.extra_gru_cells[i-1](
            (current_hidden, (layer_hidden,)), ps[layer_key], st[layer_key]
        )
    end

    # Project to vocabulary logits
    logits, output_st = layers.output_dense(current_hidden, ps.output, st.output)

    # Return logits as vector (Lux convention: (output, state))
    return logits, st
end

# =============================================================================
# Teacher-Forced Log Probabilities (for TB Training)
# =============================================================================

"""
    compute_log_probs_teacher_forced(model, token_sequence, ps, st)

Compute log probabilities of each token in a sequence under teacher forcing.
This is the core computation for Trajectory Balance loss.

# Arguments
- `model::SMILESPolicyModel`: The GRU policy
- `token_sequence::Vector{Int}`: Full token sequence [START, t1, t2, ..., END]
- `ps`: Model parameters (Lux)
- `st`: Model states (Lux)

# Returns
Tuple of (total_log_prob, per_step_log_probs)
- total_log_prob: Sum of log P(t_{i+1} | t_{1:i}) for all steps
- per_step_log_probs: Vector of individual step log probabilities

# Zygote Safety
This function is designed for use inside Zygote.withgradient. All hidden states
are maintained via variable reassignment (not array mutation), and logits are
computed inside the time loop to avoid storing intermediate outputs.
GRU layers are unrolled (supports up to 3 layers) for AD compatibility.
"""
function compute_log_probs_teacher_forced(
    model::SMILESPolicyModel,
    token_sequence::Vector{Int},
    ps, st;
    detach_end_nonterminal::Bool=false
)
    layers = model.layers
    seq_length = length(token_sequence) - 1  # Number of predictions to make

    if seq_length <= 0
        return 0.0, Float64[]
    end

    Zygote.@ignore begin
        if model.n_layers > 3
            error("compute_log_probs_teacher_forced supports up to 3 GRU layers. Got $(model.n_layers).")
        end
    end

    # Embed all input tokens (0-indexed tokens → 1-indexed for Lux.Embedding)
    input_tokens = token_sequence[1:end-1] .+ 1
    target_tokens = token_sequence[2:end]

    # Embed input tokens: shape (embed_dim, seq_length)
    embedded, _ = layers.embedding(input_tokens, ps.embedding, st.embedding)

    # Initialize hidden states via variable reassignment (Zygote-safe, no array mutation)
    h1 = zeros(Float32, model.hidden_dim)
    h2 = zeros(Float32, model.hidden_dim)
    h3 = zeros(Float32, model.hidden_dim)

    # Check if model has separate termination head
    use_term = has_term_head(model)

    # Pre-compute END mask for atom softmax (used only with term_head)
    end_mask = Zygote.@ignore begin
        m = zeros(Float32, model.vocab_size)
        m[END_TOKEN + 1] = -Inf32
        m
    end

    total_log_prob = 0.0
    per_step_log_probs = Zygote.@ignore Float64[]

    for t in 1:seq_length
        embed_t = embedded[:, t]

        # Layer 1 GRU (variable reassignment preserves gradient flow through time)
        (h1, _), _ = layers.gru_cell((embed_t, (h1,)), ps.gru, st.gru)
        final_h = h1

        # Layer 2 GRU (if present)
        if model.n_layers >= 2
            (h2, _), _ = layers.extra_gru_cells[1]((final_h, (h2,)), ps.gru_2, st.gru_2)
            final_h = h2
        end

        # Layer 3 GRU (if present)
        if model.n_layers >= 3
            (h3, _), _ = layers.extra_gru_cells[2]((final_h, (h3,)), ps.gru_3, st.gru_3)
            final_h = h3
        end

        # Compute logits directly from final hidden state (no intermediate storage)
        logits, _ = layers.output_dense(final_h, ps.output, st.output)

        target_idx = target_tokens[t] + 1  # 0-indexed token → 1-indexed array

        if use_term
            # === DUAL-HEAD MODE ===
            # Separate termination decision from atom selection.
            # TB gradient for atoms CANNOT affect termination (separate parameters).
            #
            # Numerically stable log-sigmoid:
            #   log(σ(z)) = -softplus(-z)
            #   log(1-σ(z)) = -softplus(z)
            term_logit_vec, _ = layers.term_head(final_h, ps.term_head, st.term_head)
            term_z = term_logit_vec[1]

            is_end = Zygote.@ignore (target_tokens[t] == END_TOKEN)

            if is_end
                # Terminal step: log P_F = log P(terminate)
                step_log_prob = -NNlib.softplus(-term_z)  # log(sigmoid(z))
            else
                # Non-terminal step: log P_F = log(1 - P_term) + log P_atom(a_t)
                log_p_cont = -NNlib.softplus(term_z)  # log(1 - sigmoid(z))
                # Mask END from atom softmax (decoupled!)
                masked_logits = logits .+ end_mask
                atom_log_probs = masked_logits .- NNlib.logsumexp(masked_logits)
                step_log_prob = log_p_cont + atom_log_probs[target_idx]
            end
        elseif detach_end_nonterminal
            # === DETACHED-END MODE (gradient surgery for TB fine-tuning) ===
            # Same single-head architecture, but at non-terminal positions:
            #   log P(a_t|s_t) = log P_atom(a_t|not END) + sg[log(1 - P_END)]
            # where sg[] = stop-gradient. This prevents TB gradient from
            # suppressing END token at the ~38 non-terminal positions.
            # At the terminal position, full gradient flows through END.
            # KL regularization (not using this flag) maintains END from pretraining.
            is_end = Zygote.@ignore (target_tokens[t] == END_TOKEN)

            if is_end
                # Terminal step: full gradient through log P(END)
                log_probs = logits .- NNlib.logsumexp(logits)
                step_log_prob = log_probs[target_idx]
            else
                # Non-terminal step: decompose into atom selection + continuation
                # 1. Atom probability in non-END subspace (full gradient for atom logits)
                masked_logits = logits .+ end_mask  # end_mask[END+1] = -Inf32
                atom_log_probs = masked_logits .- NNlib.logsumexp(masked_logits)
                log_p_atom = atom_log_probs[target_idx]

                # 2. Continuation probability (stop gradient — maintained by KL)
                log_continue = Zygote.@ignore begin
                    full_lse = NNlib.logsumexp(logits)
                    p_end = exp(logits[END_TOKEN + 1] - full_lse)
                    Float32(log(max(1.0f0 - p_end, 1.0f-8)))
                end

                step_log_prob = log_p_atom + log_continue
            end
        else
            # === ORIGINAL MODE (backward compatible) ===
            log_probs = logits .- NNlib.logsumexp(logits)
            step_log_prob = if 1 <= target_idx <= model.vocab_size
                log_probs[target_idx]
            else
                Zygote.@ignore(-Inf)
            end
        end

        total_log_prob += step_log_prob
        Zygote.@ignore push!(per_step_log_probs, Float64(step_log_prob))
    end

    return total_log_prob, per_step_log_probs
end

# =============================================================================
# Autoregressive Sampling (for Inference)
# =============================================================================

"""
    sample_smiles_autoregressive(model, ps, st, vocab;
        max_length=150, temperature=1.0, epsilon=0.0,
        constrained=true,
        q_net=nothing, q_params=nothing, q_states=nothing,
        p_quantile=0.0, collect_transitions=false)

Sample a complete SMILES string autoregressively.
This maintains GRU hidden state across steps for proper sequence generation.

# Arguments
- `model::SMILESPolicyModel`: The GRU policy
- `ps`: Model parameters
- `st`: Model states
- `vocab::SMILESVocabulary`: Vocabulary for decoding
- `max_length`: Maximum sequence length
- `temperature`: Sampling temperature (1.0 = standard, <1 = greedy, >1 = diverse)
- `epsilon`: ε-uniform exploration rate
- `constrained`: If true (default), enforce structural SMILES constraints:
  balanced parentheses, matched ring closures, no END with open structures

# QGFN Arguments (optional)
- `q_net`: QFunctionNetwork for Q-value guided masking
- `q_params`: Q-function parameters
- `q_states`: Q-function Lux states
- `p_quantile`: Masking threshold in [0,1] (0 = disabled)
- `collect_transitions`: If true, return transition data for Q-training

# Returns
- Standard: (smiles_string, token_sequence, log_prob)
- With collect_transitions: (smiles_string, token_sequence, log_prob, transitions)
"""
function sample_smiles_autoregressive(
    model::SMILESPolicyModel, ps, st, vocab::SMILESVocabulary;
    max_length::Int=150,
    temperature::Float64=1.0,
    epsilon::Float64=0.0,
    constrained::Bool=true,
    # QGFN parameters (optional)
    q_net=nothing,
    q_params=nothing,
    q_states=nothing,
    p_quantile::Float64=0.0,
    collect_transitions::Bool=false
)
    layers = model.layers

    # Initialize hidden states for all GRU layers
    hiddens = [zeros(Float32, model.hidden_dim) for _ in 1:model.n_layers]

    tokens = [START_TOKEN]
    total_log_prob = 0.0
    current_token = START_TOKEN

    # QGFN: determine if Q-masking is active
    use_q_masking = !isnothing(q_net) && !isnothing(q_params) && p_quantile > 0.0

    # QGFN: transition buffer for Q-training
    transitions = collect_transitions ? NamedTuple[] : nothing

    # Constrained decoding state: track structural SMILES invariants
    paren_depth = 0
    open_rings = Set{Int}()  # Ring digits currently open (single-digit 0-9)

    # Pre-compute token indices for constraint masks (0-indexed token IDs)
    open_paren_idx = get(vocab.token_to_idx, "(", -1)
    close_paren_idx = get(vocab.token_to_idx, ")", -1)
    end_token_idx = END_TOKEN
    ring_digit_indices = Dict{Int,Int}()  # token_idx → digit value
    for d in 0:9
        idx = get(vocab.token_to_idx, string(d), -1)
        if idx >= 0
            ring_digit_indices[idx] = d
        end
    end

    # Check if model has separate termination head
    use_term = has_term_head(model)

    for step in 1:max_length
        # Embed current token
        embedded, _ = layers.embedding([current_token + 1], ps.embedding, st.embedding)
        embed_vec = embedded[:, 1]

        # QGFN: save pre-step hidden state for transition collection
        prev_hidden = collect_transitions ? copy(hiddens[model.n_layers]) : nothing

        # Process through GRU layers
        input_vec = embed_vec
        for layer_idx in 1:model.n_layers
            if layer_idx == 1
                (hiddens[1], _), _ = layers.gru_cell(
                    (input_vec, (hiddens[1],)), ps.gru, st.gru
                )
                input_vec = hiddens[1]
            else
                layer_key = Symbol("gru_$layer_idx")
                (hiddens[layer_idx], _), _ = layers.extra_gru_cells[layer_idx-1](
                    (input_vec, (hiddens[layer_idx],)), ps[layer_key], st[layer_key]
                )
                input_vec = hiddens[layer_idx]
            end
        end

        # Compute logits
        logits, _ = layers.output_dense(input_vec, ps.output, st.output)

        # === SEPARATE TERMINATION HEAD ===
        if use_term
            # Two-step decision: first terminate or continue, then which atom
            term_logit_vec, _ = layers.term_head(input_vec, ps.term_head, st.term_head)
            term_prob = Float64(NNlib.sigmoid(term_logit_vec[1]))

            # Constrained: block termination if unclosed parentheses
            if constrained && paren_depth > 0
                term_prob = 0.0
            end

            # Decide: terminate or continue?
            if rand() < term_prob
                # Terminate
                next_token = END_TOKEN
                step_log_prob = log(term_prob + 1e-30)
                push!(tokens, next_token)
                total_log_prob += step_log_prob

                # QGFN transition
                if collect_transitions && !isnothing(prev_hidden)
                    push!(transitions, (
                        hidden=prev_hidden, action=next_token,
                        next_hidden=copy(hiddens[model.n_layers]), is_terminal=true
                    ))
                end
                break
            end

            # Continue: sample atom from masked distribution (END excluded)
            logits[end_token_idx + 1] = -Inf  # Mask END from atom softmax

            # QGFN: Apply Q-value masking
            if use_q_masking
                q_values, _ = compute_q_values(q_net, hiddens[model.n_layers], q_params, q_states)
                applicable_mask = trues(length(logits))
                applicable_mask[PAD_TOKEN + 1] = false
                applicable_mask[START_TOKEN + 1] = false
                applicable_mask[END_TOKEN + 1] = false
                logits = apply_q_masking(logits, q_values, p_quantile; applicable_mask=applicable_mask)
            end

            # Constrained decoding (atom-level constraints)
            if constrained
                if paren_depth == 0 && close_paren_idx >= 0
                    logits[close_paren_idx + 1] = -Inf
                end
                remaining = max_length - step
                if remaining <= paren_depth + 1 && paren_depth > 0
                    for tok_0idx in 0:(model.vocab_size - 1)
                        tok_1idx = tok_0idx + 1
                        is_close_paren = (tok_0idx == close_paren_idx)
                        if !is_close_paren
                            logits[tok_1idx] = -Inf
                        end
                    end
                end
            end

            # Apply temperature
            if temperature != 1.0
                logits = logits ./ Float32(temperature)
            end

            # Mask special tokens
            logits[PAD_TOKEN + 1] = -Inf
            logits[START_TOKEN + 1] = -Inf

            # Compute atom probabilities
            log_probs = logits .- NNlib.logsumexp(logits)
            probs = exp.(log_probs)

            # ε-uniform exploration
            if epsilon > 0.0
                n_valid = sum(isfinite.(logits))
                uniform_prob = 1.0f0 / max(n_valid, 1)
                valid_mask = isfinite.(logits)
                probs = (1.0f0 - Float32(epsilon)) .* probs .+ Float32(epsilon) .* uniform_prob .* valid_mask
                probs = probs ./ sum(probs)
            end

            # Sample atom
            cumulative = cumsum(probs)
            r = rand(Float32)
            next_token_1indexed = findfirst(p -> p >= r, cumulative)
            if isnothing(next_token_1indexed)
                next_token_1indexed = length(probs)
            end
            next_token = next_token_1indexed - 1

            # Log prob: log(1 - P_term) + log P_atom(a)
            log_p_cont = log(1.0 - term_prob + 1e-30)
            step_log_prob = log_p_cont + Float64(log_probs[next_token_1indexed])

            push!(tokens, next_token)
            total_log_prob += step_log_prob

        else
            # === ORIGINAL MODE (no term_head, backward compatible) ===

            # QGFN: Apply Q-value masking before temperature/exploration
            if use_q_masking
                q_values, _ = compute_q_values(q_net, hiddens[model.n_layers], q_params, q_states)
                applicable_mask = trues(length(logits))
                applicable_mask[PAD_TOKEN + 1] = false
                applicable_mask[START_TOKEN + 1] = false
                logits = apply_q_masking(logits, q_values, p_quantile; applicable_mask=applicable_mask)
            end

            # Constrained decoding: enforce structural SMILES validity
            if constrained
                if paren_depth == 0 && close_paren_idx >= 0
                    logits[close_paren_idx + 1] = -Inf
                end
                if paren_depth > 0 && end_token_idx >= 0
                    logits[end_token_idx + 1] = -Inf
                end
                remaining = max_length - step
                if remaining <= paren_depth + 1 && paren_depth > 0
                    for tok_0idx in 0:(model.vocab_size - 1)
                        tok_1idx = tok_0idx + 1
                        is_close_paren = (tok_0idx == close_paren_idx)
                        is_end = (tok_0idx == end_token_idx && paren_depth <= 1)
                        if !(is_close_paren || is_end)
                            logits[tok_1idx] = -Inf
                        end
                    end
                end
            end

            # Apply temperature
            if temperature != 1.0
                logits = logits ./ Float32(temperature)
            end

            logits[PAD_TOKEN + 1] = -Inf
            logits[START_TOKEN + 1] = -Inf

            log_probs = logits .- NNlib.logsumexp(logits)
            probs = exp.(log_probs)

            if epsilon > 0.0
                n_valid = sum(isfinite.(logits))
                uniform_prob = 1.0f0 / n_valid
                valid_mask = isfinite.(logits)
                probs = (1.0f0 - Float32(epsilon)) .* probs .+ Float32(epsilon) .* uniform_prob .* valid_mask
                probs = probs ./ sum(probs)
            end

            cumulative = cumsum(probs)
            r = rand(Float32)
            next_token_1indexed = findfirst(p -> p >= r, cumulative)
            if isnothing(next_token_1indexed)
                next_token_1indexed = length(probs)
            end
            next_token = next_token_1indexed - 1

            push!(tokens, next_token)
            total_log_prob += Float64(log_probs[next_token_1indexed])
        end

        # Update constraint state
        if constrained
            if next_token == open_paren_idx
                paren_depth += 1
            elseif next_token == close_paren_idx
                paren_depth = max(0, paren_depth - 1)
            elseif haskey(ring_digit_indices, next_token)
                d = ring_digit_indices[next_token]
                if d in open_rings
                    delete!(open_rings, d)
                else
                    push!(open_rings, d)
                end
            end
        end

        # QGFN: collect transition for Q-training
        if !use_term && collect_transitions && !isnothing(prev_hidden)
            is_terminal = (next_token == END_TOKEN)
            push!(transitions, (
                hidden=prev_hidden,
                action=next_token,
                next_hidden=copy(hiddens[model.n_layers]),
                is_terminal=is_terminal
            ))
        elseif use_term && collect_transitions && !isnothing(prev_hidden)
            # Already collected for termination case above (break);
            # collect for continuation case here
            push!(transitions, (
                hidden=prev_hidden,
                action=next_token,
                next_hidden=copy(hiddens[model.n_layers]),
                is_terminal=false
            ))
        end

        current_token = next_token

        # Stop if END token (original mode only; term_head mode breaks above)
        if !use_term && next_token == END_TOKEN
            break
        end
    end

    # Decode to SMILES string
    smiles = decode(vocab, tokens; strip_special=true)

    if collect_transitions
        return smiles, tokens, total_log_prob, transitions
    else
        return smiles, tokens, total_log_prob
    end
end

# =============================================================================
# Batch Sampling
# =============================================================================

"""
    sample_smiles_batch(model, ps, st, vocab, batch_size; kwargs...)

Sample a batch of SMILES strings.

# Returns
Vector of (smiles, tokens, log_prob) tuples
"""
function sample_smiles_batch(
    model::SMILESPolicyModel, ps, st, vocab::SMILESVocabulary,
    batch_size::Int; kwargs...
)
    results = Vector{Tuple{String, Vector{Int}, Float64}}(undef, batch_size)
    Threads.@threads for i in 1:batch_size
        results[i] = sample_smiles_autoregressive(model, ps, st, vocab; kwargs...)
    end
    return results
end

# =============================================================================
# Batched MLE Loss (Critical Performance Optimization)
# =============================================================================
#
# Processes ALL sequences in a batch simultaneously through the GRU.
# Instead of 64 sequential forward passes, this does 1 batched pass
# where hidden state is (hidden_dim, batch_size) matrix.
#
# Speedup: ~50-100x for MLE pretraining on ZINC 250K.
# Zygote-safe: all differentiable operations, padding masked via one-hot targets.

"""
    compute_mle_loss_batched(model, sequences, ps, st)

Compute batched autoregressive MLE loss: L = -(1/N) Σ_t Σ_j log P(x_{t+1,j} | x_{1:t,j})

This is the **critical optimization** for pretraining performance.
Instead of processing each sequence independently (O(batch × seq_len) serial steps),
this processes the entire batch in parallel (O(seq_len) steps with batch-wide matrix ops).

# Arguments
- `model::SMILESPolicyModel`: The GRU policy
- `sequences::Vector{Vector{Int}}`: Batch of token sequences (0-indexed)
- `ps`: Model parameters (Lux)
- `st`: Model states (Lux)

# Returns
Average negative log-likelihood per token (Float32, differentiable)

# Performance
- GRU operations: (embed_dim, batch) × (hidden, batch) → fully utilizes BLAS/GPU
- Embedding: single batched lookup for all tokens in all sequences
- Padding: masked via one-hot targets (no wasted gradient computation)
"""
function compute_mle_loss_batched(
    model::SMILESPolicyModel,
    sequences::Vector{Vector{Int}},
    ps, st
)
    batch_size = length(sequences)
    if batch_size == 0
        return 0.0f0
    end

    # Prepare all non-differentiable data upfront (one allocation, not per-step)
    seq_lengths, max_len, input_flat, target_selectors, total_tokens = Zygote.@ignore begin
        lens = Int[length(seq) - 1 for seq in sequences]
        ml = maximum(lens)
        vs = model.vocab_size

        # Flatten input tokens for single embedding call
        # Input: 1-indexed for Lux.Embedding (original 0-indexed + 1)
        inp = fill(Int(PAD_TOKEN + 1), ml * batch_size)

        # Pre-compute ALL target selectors as 3D tensor: (vocab_size, batch_size, max_len)
        # This avoids per-step allocation inside the time loop
        selectors = zeros(Float32, vs, batch_size, ml)

        for (j, seq) in enumerate(sequences)
            sl = lens[j]
            offset = (j - 1) * ml
            for t in 1:sl
                inp[offset + t] = seq[t] + 1     # 0→1 indexed
                target_idx = seq[t + 1] + 1       # next token, 0→1 indexed
                if 1 <= target_idx <= vs
                    selectors[target_idx, j, t] = 1.0f0
                end
            end
        end

        total_tok = Float32(sum(lens))
        lens, ml, inp, selectors, total_tok
    end

    if total_tokens == 0.0f0
        return 0.0f0
    end

    # === BATCHED EMBEDDING ===
    # Single embedding call for ALL tokens in ALL sequences
    flat_embedded, _ = model.layers.embedding(input_flat, ps.embedding, st.embedding)
    embedded = reshape(flat_embedded, model.embed_dim, max_len, batch_size)

    # === BATCHED GRU ===
    # Hidden states are matrices: (hidden_dim, batch_size)
    h1 = zeros(Float32, model.hidden_dim, batch_size)
    h2 = zeros(Float32, model.hidden_dim, batch_size)
    h3 = zeros(Float32, model.hidden_dim, batch_size)

    # Check for separate termination head
    use_term = has_term_head(model)

    # Pre-compute terminal step masks for dual-head mode
    term_masks = Zygote.@ignore begin
        if use_term
            # For each (batch, time), is the target END?
            # term_mask[j, t] = 1 if target is END, 0 otherwise
            tmask = zeros(Float32, batch_size, max_len)
            for (j, seq) in enumerate(sequences)
                sl = seq_lengths[j]
                for t in 1:sl
                    if seq[t + 1] == END_TOKEN
                        tmask[j, t] = 1.0f0
                    end
                end
            end
            tmask
        else
            nothing
        end
    end

    # Pre-compute END mask for atom softmax
    # Use -1e10 instead of -Inf to avoid NaN from -Inf * 0 in IEEE 754
    # when the product of atom_log_probs (with -Inf at END) and zeroed selector is computed
    end_mask_vec = Zygote.@ignore begin
        m = zeros(Float32, model.vocab_size)
        m[END_TOKEN + 1] = -1.0f10
        m
    end

    total_nll = 0.0f0

    for t in 1:max_len
        # Embeddings for this time step: (embed_dim, batch_size)
        embed_t = embedded[:, t, :]

        # Layer 1 GRU — batched
        (h1, _), _ = model.layers.gru_cell((embed_t, (h1,)), ps.gru, st.gru)
        final_h = h1

        # Layer 2 GRU (if present)
        if model.n_layers >= 2
            (h2, _), _ = model.layers.extra_gru_cells[1]((final_h, (h2,)), ps.gru_2, st.gru_2)
            final_h = h2
        end

        # Layer 3 GRU (if present)
        if model.n_layers >= 3
            (h3, _), _ = model.layers.extra_gru_cells[2]((final_h, (h3,)), ps.gru_3, st.gru_3)
            final_h = h3
        end

        # Compute logits: (vocab_size, batch_size)
        logits, _ = model.layers.output_dense(final_h, ps.output, st.output)

        if use_term
            # === DUAL-HEAD BATCHED ===
            # Termination logits: (1, batch_size)
            term_logits, _ = model.layers.term_head(final_h, ps.term_head, st.term_head)
            # log P(term) and log P(cont) per sample: (1, batch_size)
            log_p_term = -NNlib.softplus.(-term_logits)   # log sigmoid
            log_p_cont = -NNlib.softplus.(term_logits)    # log (1 - sigmoid)

            # Atom log probs: mask END, then log softmax
            masked_logits = logits .+ end_mask_vec  # broadcasts (V,) over (V, B)
            atom_log_probs = masked_logits .- NNlib.logsumexp(masked_logits; dims=1)

            # For terminal targets: NLL = -log P(term)
            # For non-terminal targets: NLL = -log P(cont) - log P_atom(a)
            target_sel_t = Zygote.@ignore @view target_selectors[:, :, t]
            atom_nll = -sum(atom_log_probs .* target_sel_t)

            # Continuation NLL: -log(1-P_term) for non-terminal positions
            # Terminal NLL: -log(P_term) for terminal positions
            tm = Zygote.@ignore @view term_masks[:, t:t]  # (B, 1) → transpose to (1, B)
            tm_t = Zygote.@ignore permutedims(tm)  # (1, B)
            term_nll = -sum(log_p_term .* tm_t .+ log_p_cont .* (1.0f0 .- tm_t))

            # For terminal positions, atom_nll includes a -Inf · 0 = 0 contribution
            # (target selector is zero at padding positions, and END target maps to
            #  masked -Inf but target_sel has 1 at END index — we need to zero it out)
            # Fix: remove atom NLL contribution for terminal steps
            # atom_sel without END: zero out END row in selector for terminal steps
            # Actually, target_sel_t already has 1.0 at END_TOKEN+1 for terminal steps,
            # and atom_log_probs[END_TOKEN+1, :] = -Inf. Product = -Inf · 1 = -Inf.
            # We need to handle this. Easiest: zero out END row in target_selectors.
            # But target_selectors is pre-computed... let me use a mask.
            #
            # Better approach: compute atom_nll only for non-terminal, term_nll for terminal
            # Use target_sel with END row zeroed:
            non_end_sel_t = Zygote.@ignore begin
                sel = copy(target_selectors[:, :, t])
                sel[END_TOKEN + 1, :] .= 0.0f0
                sel
            end
            atom_nll_safe = -sum(atom_log_probs .* non_end_sel_t)

            total_nll += atom_nll_safe + term_nll
        else
            # === ORIGINAL MODE ===
            log_probs = logits .- NNlib.logsumexp(logits; dims=1)
            target_sel_t = Zygote.@ignore @view target_selectors[:, :, t]
            total_nll -= sum(log_probs .* target_sel_t)
        end
    end

    return total_nll / total_tokens
end

# =============================================================================
# Weight Transfer: Pretrained → Separate Termination Head
# =============================================================================

"""
    convert_to_term_head_params(old_params, old_states, model_config)

Convert pretrained single-head parameters to dual-head (with term_head).

Extracts the END token weights from the output layer's final Dense(hidden→V)
to initialize the termination head Dense(hidden→1). The atom head keeps the
full output layer unchanged (END is masked at runtime).

# Arguments
- `old_params`: Parameters from pretrained single-head model
- `old_states`: States from pretrained single-head model
- `model_config`: NamedTuple with vocab_size, hidden_dim, embed_dim, n_layers

# Returns
(new_params, new_states) with term_head initialized from END weights
"""
function convert_to_term_head_params(old_params, old_states, model_config;
                                      rng=Random.default_rng())
    # Derive actual vocab_size from pretrained params' output layer weight (V, H)
    # This avoids mismatch when vocab is rebuilt with a different subset
    actual_vocab_size = size(old_params.output.layer_2.weight, 1)

    # Create new model with term_head to get the right structure
    new_model, new_params_template, new_states_template = create_smiles_policy(;
        vocab_size=actual_vocab_size,
        hidden_dim=model_config.hidden_dim,
        embed_dim=model_config.embed_dim,
        n_layers=model_config.n_layers,
        use_term_head=true,
        rng=rng
    )

    # Start with old params (copy all existing weights)
    new_params = deepcopy(old_params)
    new_states = deepcopy(old_states)

    # Initialize term_head for sigmoid — learn termination from scratch.
    #
    # The END row from the softmax output layer is NOT suitable for a sigmoid because:
    # softmax context: P(END) = exp(z_END) / Σexp(z_i) ≈ 2% (relative to 140 tokens)
    # sigmoid context: P(term) = σ(z_END) ≈ 50% if z_END ≈ 0 (absolute)
    #
    # Instead: zero weights + calibrated bias. MLE warmup learns state-dependent
    # termination from ZINC data, which is the right supervision signal.
    # bias = -3.0 gives σ(-3.0) ≈ 4.7%, reasonable for ~20-40 token molecules.
    hidden_dim = model_config.hidden_dim
    term_weight = zeros(Float32, 1, hidden_dim)
    term_bias = Float32[-3.0f0]

    # Add term_head to params
    term_head_ps = (weight=term_weight, bias=term_bias)
    term_head_st = new_states_template.term_head  # Empty state for Dense

    new_params = merge(new_params, (term_head=term_head_ps,))
    new_states = merge(new_states, (term_head=term_head_st,))

    return new_model, new_params, new_states
end
