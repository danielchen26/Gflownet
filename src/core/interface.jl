# High-Level Interface for GFlowNet - Model Creation and Sampling
# Clean implementation with training functions moved to training/

using Lux
using ComponentArrays
using Optimisers
using Zygote
using Random
using Statistics

using ..GFlowNet: AbstractState, AbstractAction, GFlowNetModel, Trajectory
using ..GFlowNet: ForwardPolicy, BackwardPolicy, FlowEstimator
using ..GFlowNet: PartitionFunctionMethod, SIMPLE_ESTIMATION, LEARNABLE_ESTIMATION
using ..GFlowNet: SamplingConfig, SamplingStrategy
using ..GFlowNet: STOCHASTIC_SAMPLING, GREEDY_SAMPLING, TEMPERATURE_SAMPLING
using ..GFlowNet: get_applicable_actions, is_terminal_state

# =============================================================================
# Model Creation - Following Official Lux Patterns
# =============================================================================

"""
    create_gflownet(initial_state, all_actions; kwargs...)

Create a GFlowNet model using proper Lux+ComponentArray+Zygote patterns.

# Arguments
- `initial_state`: Starting state s₀
- `all_actions`: Complete action space
- `state_dim`: Dimension of state features
- `hidden_dim=64`: Hidden layer size for neural networks
- `learning_rate=0.01`: Learning rate for optimizer
- `include_backward=false`: Whether to include backward policy for DETAILED_BALANCE
- `include_flow_estimator=false`: Whether to include flow estimator for DIRECT_FLOW/FLOW_MATCHING
- `partition_function_method=SIMPLE_ESTIMATION`: How to handle partition function Z:
  - `SIMPLE_ESTIMATION`: Z = 1 (default, fixed)
  - `LEARNABLE_ESTIMATION`: Learn Z as trainable parameter (recommended for complex environments)
- `rng=Random.default_rng()`: Random number generator

# Returns
`GFlowNetModel` with all components initialized

# Example
```julia
# With fixed Z (default)
model = create_gflownet(
    initial_state, all_actions;
    state_dim = 10,
    hidden_dim = 64
)

# With learnable Z (recommended)
model = create_gflownet(
    initial_state, all_actions;
    state_dim = 10,
    hidden_dim = 64,
    partition_function_method = LEARNABLE_ESTIMATION
)
```

Follows official Lux documentation patterns for gradient computation compatibility.
"""
function create_gflownet(
    initial_state::AbstractState,
    all_actions::Vector{<:AbstractAction};
    state_dim::Int,
    hidden_dim::Int = 64,
    learning_rate::Float64 = 0.01,
    include_backward::Bool = false,
    include_flow_estimator::Bool = false,
    partition_function_method::PartitionFunctionMethod = SIMPLE_ESTIMATION,
    rng = Random.default_rng()
)
    n_actions = length(all_actions)

    # Create neural networks using official Lux patterns
    forward_policy, forward_ps, forward_st = create_forward_policy(state_dim, hidden_dim, n_actions, rng)
    
    # Optionally create flow estimator
    if include_flow_estimator
        flow_estimator, flow_ps, flow_st = create_flow_estimator(state_dim, hidden_dim, rng)
    else
        flow_estimator = nothing
        flow_ps = nothing
        flow_st = nothing
    end
    
    # Initialize partition function parameter based on method
    log_partition_function = if partition_function_method == LEARNABLE_ESTIMATION
        0.0  # Initialize log Z to 0 (Z = 1)
    else
        nothing  # For SIMPLE_ESTIMATION, SAMPLING_ESTIMATION, etc.
    end

    # Build parameters and states based on which components are included.
    #
    # log_Z MUST match the network parameter element type. Lux gives Float32 params,
    # and a Float64 scalar promotes the ENTIRE ComponentArray to Float64 -- doubling
    # memory and silently changing the numerics of every layer. That surfaced as
    # `compute_forward_logits` returning Vector{Float64} once LEARNABLE_ESTIMATION
    # became the default, breaking test_core_functions.jl:131 which asserts Float32.
    # Derived rather than hardcoded, so a Float64 network stays Float64.
    param_eltype = eltype(ComponentArray(forward = forward_ps))
    # log_partition_function is `nothing` unless LEARNABLE_ESTIMATION, so the
    # conversion has to be guarded -- converting unconditionally threw
    # MethodError: no method matching Float32(::Nothing) for every SIMPLE model.
    logz_init = isnothing(log_partition_function) ? nothing :
                param_eltype(log_partition_function)

    if include_backward && include_flow_estimator
        # Both backward policy and flow estimator
        backward_policy, backward_ps, backward_st = create_backward_policy(state_dim, hidden_dim, rng)
        
        parameters = if partition_function_method == LEARNABLE_ESTIMATION
            ComponentArray(
                forward = forward_ps,
                backward = backward_ps,
                flow = flow_ps,
                log_Z = logz_init
            )
        else
            ComponentArray(
                forward = forward_ps,
                backward = backward_ps,
                flow = flow_ps
            )
        end
        
        states = (forward = forward_st, backward = backward_st, flow = flow_st)
        
    elseif include_backward && !include_flow_estimator
        # Only backward policy
        backward_policy, backward_ps, backward_st = create_backward_policy(state_dim, hidden_dim, rng)
        
        parameters = if partition_function_method == LEARNABLE_ESTIMATION
            ComponentArray(
                forward = forward_ps,
                backward = backward_ps,
                log_Z = logz_init
            )
        else
            ComponentArray(
                forward = forward_ps,
                backward = backward_ps
            )
        end
        
        states = (forward = forward_st, backward = backward_st)
        
    elseif !include_backward && include_flow_estimator
        # Only flow estimator
        backward_policy = nothing
        
        parameters = if partition_function_method == LEARNABLE_ESTIMATION
            ComponentArray(
                forward = forward_ps,
                flow = flow_ps,
                log_Z = logz_init
            )
        else
            ComponentArray(
                forward = forward_ps,
                flow = flow_ps
            )
        end
        
        states = (forward = forward_st, flow = flow_st)
        
    else
        # Neither backward policy nor flow estimator
        backward_policy = nothing
        
        parameters = if partition_function_method == LEARNABLE_ESTIMATION
            ComponentArray(
                forward = forward_ps,
                log_Z = logz_init
            )
        else
            ComponentArray(
                forward = forward_ps
            )
        end
        
        states = (forward = forward_st,)
    end

    # Setup optimizer (official Lux pattern)
    opt = Optimisers.Adam(learning_rate)
    optimizer = Optimisers.setup(opt, parameters)

    # Create model
    return GFlowNetModel(
        initial_state,
        all_actions,
        forward_policy,
        backward_policy,
        flow_estimator,
        log_partition_function,
        parameters,
        optimizer,
        states
    )
end

# =============================================================================
# Optimiser Learning Rate - Accessor and Mutator
# =============================================================================

"""
    optimizer_learning_rate(model::GFlowNetModel)::Float64

Return the step size the model's optimiser will ACTUALLY use for the network
parameters.

`Optimisers.setup` bakes the rate into the optimiser state at construction, so
`create_gflownet(learning_rate = ...)` is the only thing that determines it until
something changes it. This accessor exists because there was previously no way to
ask a model what rate it was training at, and that is what let
`TrainingConfig.learning_rate` sit dead for so long: the two numbers could
disagree by any factor and nothing could observe the disagreement. The library's
own defaults disagree -- `create_gflownet` defaults to 0.01 while
`TrainingConfig` defaults to 1e-3, a 10x gap -- so the verbose banner printing
`config.learning_rate` was reporting a rate no parameter was moving at.

Throws if the optimiser tree carries more than one distinct rate, which cannot
arise from `create_gflownet` and means the state was assembled by hand.
"""
function optimizer_learning_rate(model::GFlowNetModel)::Float64
    etas = Float64[]
    Optimisers.fmap(model.optimizer; exclude = x -> x isa Optimisers.Leaf) do leaf
        push!(etas, Float64(leaf.rule.eta))
        leaf
    end

    if isempty(etas)
        throw(ArgumentError("model.optimizer contains no Optimisers.Leaf; cannot read a learning rate"))
    end

    unique_etas = unique(etas)
    if length(unique_etas) > 1
        throw(ArgumentError(
            "model.optimizer carries $(length(unique_etas)) distinct learning rates " *
            "$(unique_etas); a single scalar rate is not defined for this model"
        ))
    end

    return unique_etas[1]
end

"""
    set_learning_rate!(model::GFlowNetModel, learning_rate::Real)

Set the step size the model's optimiser uses for the network parameters, and
return the model.

Implemented with `Optimisers.adjust`, which rewrites the rule's `eta` while
PRESERVING the accumulated moment buffers. That distinction matters: rebuilding
the optimiser with `Optimisers.setup` would reset Adam's first and second moments
to zero, and doing that on every step would leave Adam permanently on its
bias-corrected first step -- an update of fixed magnitude eta in the direction of
the gradient's sign, which is a different algorithm, not a rate change. So the
rate is adjusted, never re-setup.
"""
function set_learning_rate!(model::GFlowNetModel, learning_rate::Real)
    learning_rate > 0 || throw(ArgumentError("learning_rate must be positive, got $learning_rate"))
    model.optimizer = Optimisers.adjust(model.optimizer, Float64(learning_rate))
    return model
end

"""
    create_forward_policy(state_dim::Int, hidden_dim::Int, n_actions::Int, rng)

Create forward policy neural network following Lux patterns.
"""
function create_forward_policy(state_dim::Int, hidden_dim::Int, n_actions::Int, rng)
    # Standard Lux Chain
    policy_net = Lux.Chain(
        Lux.Dense(state_dim => hidden_dim, tanh),
        Lux.Dense(hidden_dim => hidden_dim, tanh),
        Lux.Dense(hidden_dim => n_actions)  # Raw logits
    )

    ps, st = Lux.setup(rng, policy_net)
    return ForwardPolicy(policy_net), ps, st
end

"""
    create_flow_estimator(state_dim::Int, hidden_dim::Int, rng)

Create flow estimation neural network following Lux patterns.
"""
function create_flow_estimator(state_dim::Int, hidden_dim::Int, rng)
    # Standard Lux Chain
    flow_net = Lux.Chain(
        Lux.Dense(state_dim => hidden_dim, tanh),
        Lux.Dense(hidden_dim => hidden_dim, tanh),
        Lux.Dense(hidden_dim => 1)  # Single flow value
    )

    ps, st = Lux.setup(rng, flow_net)
    return FlowEstimator(flow_net), ps, st
end

"""
    create_backward_policy(state_dim::Int, hidden_dim::Int, rng)

Create backward policy neural network for computing P_B(s|s').

Uses joint state representation: takes concatenated features of (source, target) states
and outputs a single probability value.
"""
function create_backward_policy(state_dim::Int, hidden_dim::Int, rng)
    # Input dimension is 2 * state_dim for joint representation
    input_dim = 2 * state_dim
    
    # Standard Lux Chain
    backward_net = Lux.Chain(
        Lux.Dense(input_dim => hidden_dim, tanh),
        Lux.Dense(hidden_dim => hidden_dim, tanh),
        Lux.Dense(hidden_dim => 1)  # Single probability logit
    )

    ps, st = Lux.setup(rng, backward_net)
    return BackwardPolicy(backward_net), ps, st
end

# =============================================================================
# MOGFN Model Creation — Preference-Conditioned GFlowNet (Gap 5)
# =============================================================================

"""
    create_preference_encoder(n_objectives::Int, embed_dim::Int, rng)

Create a preference encoder MLP that maps w ∈ R^K → R^d.
Shared between forward policy, backward policy, and Z network.
"""
function create_preference_encoder(n_objectives::Int, embed_dim::Int, rng)
    encoder = Lux.Chain(
        Lux.Dense(n_objectives => embed_dim, Lux.relu),
        Lux.Dense(embed_dim => embed_dim, Lux.relu)
    )
    ps, st = Lux.setup(rng, encoder)
    return encoder, ps, st
end

"""
    create_z_network(embed_dim::Int, rng)

Create a Z(w) network: maps preference embedding → scalar log Z(w).
Replaces the scalar log_Z parameter in MOGFN mode.
"""
function create_z_network(embed_dim::Int, rng)
    z_net = Lux.Chain(
        Lux.Dense(embed_dim => embed_dim, Lux.relu),
        Lux.Dense(embed_dim => 1)  # Scalar log Z output
    )
    ps, st = Lux.setup(rng, z_net)
    return z_net, ps, st
end

"""
    create_mogfn_gflownet(initial_state, all_actions; kwargs...)

Create a MOGFN-PC (preference-conditioned) GFlowNet model (Gap 5, ICML 2023).

The forward policy input is [state_features; embed(w)] where embed(w)
is a learned preference embedding. Z(w) is a network that outputs
log Z conditioned on the same embedding.

# Arguments
- `initial_state`: Starting state s₀
- `all_actions`: Complete action space
- `state_dim`: Base dimension of state features (without preference)
- `hidden_dim=256`: Hidden layer size
- `learning_rate=0.001`: Learning rate
- `n_objectives=4`: Number of objectives K
- `preference_dim=64`: Preference embedding dimension d
- `include_backward=false`: Include backward policy
- `rng`: Random number generator
"""
function create_mogfn_gflownet(
    initial_state::AbstractState,
    all_actions::Vector{<:AbstractAction};
    state_dim::Int,
    hidden_dim::Int = 256,
    learning_rate::Float64 = 0.001,
    n_objectives::Int = 4,
    preference_dim::Int = 64,
    include_backward::Bool = false,
    rng = Random.default_rng()
)
    n_actions = length(all_actions)

    # Augmented state dim: state features + preference embedding
    augmented_state_dim = state_dim + preference_dim

    # Create components
    forward_policy, forward_ps, forward_st = create_forward_policy(augmented_state_dim, hidden_dim, n_actions, rng)
    preference_encoder, pref_ps, pref_st = create_preference_encoder(n_objectives, preference_dim, rng)
    z_net, z_ps, z_st = create_z_network(preference_dim, rng)

    if include_backward
        # Backward policy also conditioned on preferences
        backward_input_dim = 2 * state_dim + preference_dim
        backward_net = Lux.Chain(
            Lux.Dense(backward_input_dim => hidden_dim, tanh),
            Lux.Dense(hidden_dim => hidden_dim, tanh),
            Lux.Dense(hidden_dim => 1)
        )
        backward_ps_raw, backward_st = Lux.setup(rng, backward_net)
        backward_policy = BackwardPolicy(backward_net)

        parameters = ComponentArray(
            forward = forward_ps,
            backward = backward_ps_raw,
            preference = pref_ps,
            z_net = z_ps
        )
        states = (forward = forward_st, backward = backward_st, preference = pref_st, z_net = z_st)
    else
        backward_policy = nothing

        parameters = ComponentArray(
            forward = forward_ps,
            preference = pref_ps,
            z_net = z_ps
        )
        states = (forward = forward_st, preference = pref_st, z_net = z_st)
    end

    # Setup optimizer
    opt = Optimisers.Adam(learning_rate)
    optimizer = Optimisers.setup(opt, parameters)

    return GFlowNetModel(
        initial_state,
        all_actions,
        forward_policy,
        backward_policy,
        nothing,          # flow_estimator
        nothing,          # log_partition_function (replaced by z_network)
        parameters,
        optimizer,
        states,
        preference_encoder,
        z_net
    )
end

# =============================================================================
# Trajectory Sampling - Zygote-Safe Implementation
# =============================================================================

"""
    sample_trajectory(model::GFlowNetModel; config::SamplingConfig = SamplingConfig())

Sample trajectory using Zygote-safe operations.
"""
function sample_trajectory(model::GFlowNetModel; config = SamplingConfig())
    # Handle both SamplingConfig and named tuple configs
    if isa(config, NamedTuple)
        # Convert named tuple to SamplingConfig
        acyclic_rate = get(config, :acyclic_rate, 0.0)
        strategy = get(config, :strategy, :stochastic)
        temperature = get(config, :temperature, 1.0)

        # Convert strategy symbol to enum
        strategy_enum = if strategy == :stochastic
            STOCHASTIC_SAMPLING
        elseif strategy == :greedy
            GREEDY_SAMPLING
        elseif strategy == :temperature
            TEMPERATURE_SAMPLING
        else
            STOCHASTIC_SAMPLING
        end

        config = SamplingConfig(
            strategy=strategy_enum,
            temperature=temperature,
            acyclic_rate=acyclic_rate
        )
    elseif !isa(config, SamplingConfig)
        config = SamplingConfig()
    end
    trajectory_states = [model.initial_state]
    trajectory_actions = AbstractAction[]

    current_state = model.initial_state
    steps = 0

    # Simple acyclic control: track visited states if acyclic_rate > 0
    visited_states = Set()
    if config.acyclic_rate > 0.0
        push!(visited_states, current_state)
    end

    while !is_terminal_state(current_state) && steps < config.max_trajectory_length
        steps += 1

        # Get applicable actions - computed fresh each time (on-demand approach)
        applicable_actions = Zygote.@ignore get_applicable_actions(current_state, model.all_actions)

        # Apply simple acyclic control if enabled
        if config.acyclic_rate > 0.0
            # Filter out actions that would lead to visited states
            applicable_actions = filter(applicable_actions) do action
                next_state = compute_next_state(action, current_state)
                next_state ∉ visited_states || rand() > config.acyclic_rate
            end
        end

        if isempty(applicable_actions)
            break
        end

        # Sample an action that ACTUALLY ADVANCES the state.
        #
        # An action leaving the state unchanged is not a DAG edge. A domain can
        # approve an action in `is_applicable` and then fail to perform it, returning
        # the state untouched -- the fragment domain does exactly that when RDKit
        # rejects a join (molecular_generation.jl:545 and :552 both `return state`).
        # Measured on the real fragment library with RDKit working: 57 of 1838
        # applied actions, 3.1%, were silent self-loops.
        #
        # Recording one corrupts the structure Trajectory Balance depends on. The
        # graph stops being acyclic; the trajectory gains a step that moved nowhere;
        # and `is_valid_backward_transition(s, s)` becomes true, so the state becomes
        # ITS OWN PARENT. That inflates the parent set P_B normalises over, feeding a
        # wrong n(x) straight into the TB optimum -- the same path-count bias that was
        # measured on the grid as Z = 78 instead of 19.
        #
        # Resampling only on a detected self-loop keeps the common path free: filtering
        # upfront would need one compute_next_state for EVERY action every step, which
        # in the molecular domain is 51 RDKit joins per step.
        #
        # Terminating actions are exempt: they legitimately produce a state that may
        # compare equal on the fields `==` inspects.
        action = nothing
        next_state = current_state
        candidates = applicable_actions
        while true
            candidate = sample_action_from_policy(model, current_state, candidates; config = config)
            proposed = compute_next_state(candidate, current_state)
            if proposed != current_state || is_terminal_state(proposed)
                action = candidate
                next_state = proposed
                break
            end
            candidates = Zygote.@ignore filter(a -> a != candidate, candidates)
            isempty(candidates) && break
        end

        # Every applicable action was a self-loop; this state cannot advance.
        action === nothing && break

        # Update visited states for acyclic control
        if config.acyclic_rate > 0.0
            push!(visited_states, next_state)
        end

        # Update trajectory (non-mutating pattern for Zygote)
        trajectory_actions = [trajectory_actions..., action]
        trajectory_states = [trajectory_states..., next_state]

        current_state = next_state
    end

    return Trajectory(trajectory_states, trajectory_actions)
end

"""
    sample_action_from_policy(model, state, applicable_actions; config)

Sample action from forward policy using Zygote-safe operations.
"""
function sample_action_from_policy(model::GFlowNetModel, state::AbstractState,
                                  applicable_actions::Vector{<:AbstractAction};
                                  config::SamplingConfig = SamplingConfig())

    # Get state features
    features = state_to_features(state)

    # Compute forward logits using proper Lux call
    logits_vec, _ = model.forward_policy.model(features, model.parameters.forward, model.states.forward)

    # Find applicable indices (discrete logic - non-differentiable)
    applicable_indices = Zygote.@ignore [i for (i, a) in enumerate(model.all_actions) if a in applicable_actions]

    if isempty(applicable_indices)
        # Fallback: return first applicable action
        return applicable_actions[1]
    end

    # Extract logits for applicable actions
    applicable_logits = logits_vec[applicable_indices]

    # Apply temperature if needed
    if config.strategy == TEMPERATURE_SAMPLING
        applicable_logits = applicable_logits ./ config.temperature
    end

    # Convert to probabilities using numerically stable softmax
    max_logit = maximum(applicable_logits)
    exp_logits = exp.(applicable_logits .- max_logit)
    probs = exp_logits ./ sum(exp_logits)

    # ε-Uniform Exploration Mixing (Standard GFlowNet practice)
    # P(a|s) = (1-ε) × P_F(a|s) + ε × Uniform(applicable_actions)
    # Reference: Malkin et al. (2022), Shen et al. (ICML 2023)
    if config.epsilon > 0.0
        n_actions = length(probs)
        uniform_prob = 1.0 / n_actions
        probs = (1.0 - config.epsilon) .* probs .+ config.epsilon * uniform_prob
    end

    # Sample action based on strategy
    if config.strategy == GREEDY_SAMPLING
        action_idx = argmax(probs)
    else
        # Stochastic sampling
        cumulative = cumsum(probs)
        r = rand()
        action_idx = findfirst(p -> p >= r, cumulative)
        if isnothing(action_idx)
            action_idx = length(probs)
        end
    end

    return applicable_actions[action_idx]
end

# =============================================================================
# MOGFN Trajectory Sampling — Preference-Conditioned (Gap 5)
# =============================================================================

"""
    sample_mogfn_trajectory(model::GFlowNetModel, w::Vector{Float64};
                            config::SamplingConfig = SamplingConfig())

Sample a trajectory from a MOGFN-PC model conditioned on preference vector w.

The preference is embedded once via the shared encoder, then concatenated
to state features at each step for the forward policy.

# Arguments
- `model`: MOGFN GFlowNet model (must have preference_encoder)
- `w`: Preference vector (sums to ~1.0, length K)
- `config`: Sampling configuration
"""
function sample_mogfn_trajectory(model::GFlowNetModel, w::Vector{Float64};
                                  config = SamplingConfig())
    if isa(config, NamedTuple)
        config = SamplingConfig(
            strategy = get(config, :strategy, STOCHASTIC_SAMPLING),
            temperature = get(config, :temperature, 1.0),
            epsilon = get(config, :epsilon, 0.0),
            max_trajectory_length = get(config, :max_trajectory_length, 100)
        )
    elseif !isa(config, SamplingConfig)
        config = SamplingConfig()
    end

    # Embed preference vector once (reused for all steps)
    w_f32 = Float32.(w)
    w_embed, _ = model.preference_encoder(w_f32, model.parameters.preference, model.states.preference)

    trajectory_states = [model.initial_state]
    trajectory_actions = AbstractAction[]
    current_state = model.initial_state
    steps = 0

    while !is_terminal_state(current_state) && steps < config.max_trajectory_length
        steps += 1

        applicable_actions = get_applicable_actions(current_state, model.all_actions)
        if isempty(applicable_actions)
            break
        end

        # Sample action with augmented features
        action = sample_mogfn_action(model, current_state, applicable_actions, w_embed; config=config)

        next_state = apply_action(action, current_state)

        trajectory_actions = [trajectory_actions..., action]
        trajectory_states = [trajectory_states..., next_state]
        current_state = next_state
    end

    return Trajectory(trajectory_states, trajectory_actions)
end

"""
    sample_mogfn_action(model, state, applicable_actions, w_embed; config)

Sample an action from MOGFN forward policy using [state_features; w_embed] input.
"""
function sample_mogfn_action(model::GFlowNetModel, state::AbstractState,
                              applicable_actions::Vector{<:AbstractAction},
                              w_embed::AbstractVector;
                              config::SamplingConfig = SamplingConfig())

    features = state_to_features(state)
    augmented_features = vcat(features, w_embed)

    logits_vec, _ = model.forward_policy.model(augmented_features, model.parameters.forward, model.states.forward)

    applicable_indices = [i for (i, a) in enumerate(model.all_actions) if a in applicable_actions]

    if isempty(applicable_indices)
        return applicable_actions[1]
    end

    applicable_logits = logits_vec[applicable_indices]

    if config.strategy == TEMPERATURE_SAMPLING
        applicable_logits = applicable_logits ./ config.temperature
    end

    max_logit = maximum(applicable_logits)
    exp_logits = exp.(applicable_logits .- max_logit)
    probs = exp_logits ./ sum(exp_logits)

    # ε-uniform exploration
    if config.epsilon > 0.0
        n_actions = length(probs)
        uniform_prob = 1.0 / n_actions
        probs = (1.0 - config.epsilon) .* probs .+ config.epsilon * uniform_prob
    end

    if config.strategy == GREEDY_SAMPLING
        action_idx = argmax(probs)
    else
        cumulative = cumsum(probs)
        r = rand()
        action_idx = findfirst(p -> p >= r, cumulative)
        if isnothing(action_idx)
            action_idx = length(probs)
        end
    end

    return applicable_actions[action_idx]
end

# =============================================================================
# Domain Interface Declarations - Required by Applications
# =============================================================================

"""Convert state to feature vector - must return Vector{Float32}"""
function state_to_features end

"""Check if state is terminal"""
function is_terminal_state end

"""Compute reward for state (positive for terminals)"""
function reward end

"""Check if action is applicable from state"""
function is_applicable end

"""Apply action to state, returning new state"""
function apply_action end

# =============================================================================
# Backward Trajectory Sampling - TLM Support (ICLR 2025)
# =============================================================================

"""
    sample_backward_trajectory(model::GFlowNetModel, terminal_state::AbstractState;
                               config::SamplingConfig = SamplingConfig())

Sample a trajectory backwards from a terminal state to the initial state using the backward policy.

# Mathematical Foundation (TLM - ICLR 2025)
The backward policy P_B(s|s') is trained to implicitly encode path counts:
    P_B(s|s') ≈ n(s) / n(s')
where n(s) = number of paths from initial state to s.

By sampling backwards from terminal states (which are sampled ∝ R), we:
1. Bypass the 70:1 forward path asymmetry
2. Generate diverse trajectories that would be rarely sampled forward
3. Use these for training to improve mode coverage

# Arguments
- `model::GFlowNetModel`: The GFlowNet model (must have backward_policy)
- `terminal_state::AbstractState`: Terminal state to sample backward from
- `config::SamplingConfig`: Sampling configuration

# Returns
`Trajectory` with states/actions from initial to terminal (forward direction)
"""
function sample_backward_trajectory(model::GFlowNetModel, terminal_state::AbstractState;
                                    config = SamplingConfig())

    if isnothing(model.backward_policy)
        throw(ArgumentError("Backward trajectory sampling requires backward_policy. Use include_backward=true in create_gflownet"))
    end

    # Build trajectory in reverse order (terminal → initial), then reverse at end
    # Using push! + reverse! is O(n) vs pushfirst! which is O(n²)
    reverse_states = [terminal_state]
    reverse_actions = AbstractAction[]

    current_state = terminal_state
    steps = 0
    max_steps = config.max_trajectory_length

    while current_state != model.initial_state && steps < max_steps
        steps += 1

        # Find all parent states that can transition to current_state
        parent_candidates = find_parent_states(model, current_state)

        if isempty(parent_candidates)
            # Cannot find parents - may be at initial state or disconnected
            break
        end

        # Sample parent state using backward policy
        parent_state, action = sample_parent_from_backward_policy(
            model, current_state, parent_candidates; config=config
        )

        # Add to trajectory (in reverse order)
        push!(reverse_states, parent_state)
        push!(reverse_actions, action)

        current_state = parent_state
    end

    # Reverse to get forward direction (initial → terminal)
    reverse!(reverse_states)
    reverse!(reverse_actions)

    return Trajectory(reverse_states, reverse_actions)
end

"""
    find_parent_states(model::GFlowNetModel, target_state::AbstractState)

Find all states that can transition to target_state.

Returns vector of (parent_state, action) tuples.
"""
function find_parent_states(model::GFlowNetModel, target_state::AbstractState)
    # This is a simple implementation that works for grid worlds
    # For more complex domains, this would need domain-specific implementation

    parents = Tuple{AbstractState, AbstractAction}[]

    # Try each action from potential parent states
    # For grid world: parent is one step behind in the direction of the action
    for action in model.all_actions
        # Try to find a parent state where applying action gives target
        parent = find_parent_for_action(target_state, action)
        if !isnothing(parent)
            # Verify the transition is valid
            if is_applicable(action, parent) && apply_action(action, parent) == target_state
                push!(parents, (parent, action))
            end
        end
    end

    return parents
end

"""
    find_parent_for_action(target_state, action)

For a given target state and action, find the parent state that would produce target via action.
Domain-specific implementation.
"""
function find_parent_for_action(target_state::AbstractState, action::AbstractAction)
    # Default implementation - requires domain override
    # For grid world, this is handled by specialized methods
    return nothing
end

"""
    sample_parent_from_backward_policy(model, target_state, parent_candidates; config)

Sample a parent state using the backward policy P_B(parent|target).
"""
function sample_parent_from_backward_policy(model::GFlowNetModel, target_state::AbstractState,
                                            parent_candidates::Vector{<:Tuple}; config = SamplingConfig())

    if isempty(parent_candidates)
        error("No parent candidates for backward sampling")
    end

    if length(parent_candidates) == 1
        return parent_candidates[1]
    end

    # Compute backward probabilities for each parent
    probs = Float64[]
    for (parent, action) in parent_candidates
        prob = compute_backward_probability(
            model.backward_policy, target_state, parent,
            model.parameters.backward, model.states.backward,
            model.all_actions
        )
        push!(probs, max(prob, 1e-8))
    end

    # Normalize
    prob_sum = sum(probs)
    if prob_sum > 1e-8
        probs ./= prob_sum
    else
        probs = fill(1.0 / length(probs), length(probs))
    end

    # Apply epsilon exploration if configured
    if config.epsilon > 0.0
        uniform_prob = 1.0 / length(probs)
        probs = (1.0 - config.epsilon) .* probs .+ config.epsilon * uniform_prob
    end

    # Sample
    cumulative = cumsum(probs)
    r = rand()
    idx = findfirst(p -> p >= r, cumulative)
    if isnothing(idx)
        idx = length(probs)
    end

    return parent_candidates[idx]
end

"""
    sample_backward_trajectories_from_terminals(model::GFlowNetModel, terminal_states::Vector,
                                                n_samples::Int; config = SamplingConfig())

Sample backward trajectories from given terminal states.
Terminal states should be sampled proportionally to their rewards.

# Arguments
- `model`: GFlowNet model with backward policy
- `terminal_states`: Vector of terminal states to sample from
- `n_samples`: Number of backward trajectories to generate
- `config`: Sampling configuration

# Returns
Vector of Trajectory objects
"""
function sample_backward_trajectories_from_terminals(model::GFlowNetModel,
                                                     terminal_states::Vector,
                                                     n_samples::Int;
                                                     config = SamplingConfig())

    trajectories = Trajectory[]

    for _ in 1:n_samples
        # Randomly select a terminal state
        terminal = terminal_states[rand(1:length(terminal_states))]
        traj = sample_backward_trajectory(model, terminal; config=config)
        push!(trajectories, traj)
    end

    return trajectories
end

# =============================================================================
# Gap 4: Reaction-Based GFlowNet Factory
# =============================================================================

"""
    reaction_state_dim(; n_reactions = 17, fp_dim = 1024, n_scalar_features = 8) → Int

Feature-vector width for reaction-based molecular states: a Morgan fingerprint,
a one-hot over reaction templates, and a block of scalar descriptors.

This is the only place the reaction state width is computed.
`create_reaction_gflownet` defaults `state_dim` to it, and the server-side
`REACTION_STATE_DIM` calls it, so changing `n_reactions` can no longer desync
the network's input layer from the real feature width.
"""
reaction_state_dim(; n_reactions::Int = 17, fp_dim::Int = 1024,
                     n_scalar_features::Int = 8) =
    fp_dim + n_reactions + n_scalar_features

"""
    create_reaction_gflownet(; kwargs...) → GFlowNetModel

Create a GFlowNet model for reaction-based molecular generation.
Uses a two-head architecture:
- Reaction head: selects which reaction template (or terminate)
- Reactant head: produces embedding for dot-product scoring of building blocks
"""
function create_reaction_gflownet(;
    n_reactions::Int = 17,
    fp_dim::Int = 1024,
    n_scalar_features::Int = 8,
    # Julia evaluates keyword defaults left to right, so n_reactions, fp_dim and
    # n_scalar_features must be declared above this line. Previously this was the
    # literal 1049, so create_reaction_gflownet(n_reactions=20) built a network
    # sized for 17 reactions.
    state_dim::Int = reaction_state_dim(; n_reactions, fp_dim, n_scalar_features),
    hidden_dim::Int = 256,
    learning_rate::Float64 = 0.001,
    initial_state::Union{Nothing, AbstractState} = nothing,
    all_actions::Union{Nothing, Vector{<:AbstractAction}} = nothing,
    rng = Random.default_rng()
)
    # Combined forward policy: state → (reaction logits + reactant embedding)
    # Output: [n_reactions+1 logits | fp_dim embedding]
    output_dim = n_reactions + 1 + fp_dim

    forward_policy = Lux.Chain(
        Lux.Dense(state_dim => hidden_dim, Lux.relu),
        Lux.Dense(hidden_dim => hidden_dim, Lux.relu),
        Lux.Dense(hidden_dim => output_dim),
    )

    fp_ps, fp_st = Lux.setup(rng, forward_policy)

    parameters = ComponentArray(forward = fp_ps)
    states = (forward = fp_st,)

    # Setup optimizer
    opt = Optimisers.Adam(learning_rate)
    optimizer = Optimisers.setup(opt, parameters)

    # Use provided initial state/actions or create minimal placeholders
    # (the server will replace these with real ReactionMolState/ReactionAction at training time)
    init_state = if initial_state !== nothing
        initial_state
    else
        # Minimal placeholder — a grid state works as any AbstractState subtype
        GridState(1, 1, false)
    end

    actions = if all_actions !== nothing
        all_actions
    else
        # Placeholder terminate action
        AbstractAction[Terminate()]
    end

    return GFlowNetModel(
        init_state,
        actions,
        ForwardPolicy(forward_policy),
        nothing,          # backward_policy
        nothing,          # flow_estimator
        nothing,          # log_partition_function
        parameters,
        optimizer,
        states,
    )
end

# =============================================================================
# Exports
# =============================================================================

export create_gflownet, create_forward_policy, create_backward_policy, create_flow_estimator
export create_mogfn_gflownet, create_preference_encoder, create_z_network
export create_reaction_gflownet
export sample_trajectory, sample_action_from_policy
export sample_mogfn_trajectory, sample_mogfn_action
export sample_backward_trajectory, sample_backward_trajectories_from_terminals
