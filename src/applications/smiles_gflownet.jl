# SMILES GFlowNet Factory (CAFE-GFN)
# High-level API for creating and configuring a complete SMILES-level GFlowNet
#
# This integrates all CAFE-GFN components:
# - GRU forward policy (autoregressive SMILES generation)
# - Shifted-Cosh TB loss (mode-covering + mode-seeking)
# - QGFN Q-function (inference-time control)
# - KL regularization (pretrain-aware fine-tuning)
# - Sequential boosting (ensemble diversity)

using Lux
using Random
using ComponentArrays
using Optimisers

using ..GFlowNet: GFlowNetModel, ForwardPolicy, AbstractState, AbstractAction
using ..GFlowNet: SHIFTED_COSH_TB, LEARNABLE_ESTIMATION

# =============================================================================
# SMILES GFlowNet Configuration
# =============================================================================

"""
    SMILESGFlowNetConfig

Configuration for creating a SMILES GFlowNet.
"""
struct SMILESGFlowNetConfig
    # Architecture
    vocab_size::Int
    embed_dim::Int
    hidden_dim::Int
    n_gru_layers::Int

    # Training
    learning_rate::Float64
    z_learning_rate_multiplier::Float64
    gradient_clip_norm::Float64

    # CAFE-GFN features
    include_q_function::Bool
    q_hidden_dim::Int
    max_sequence_length::Int

    function SMILESGFlowNetConfig(;
        vocab_size::Int=100,
        embed_dim::Int=128,
        hidden_dim::Int=512,
        n_gru_layers::Int=3,
        learning_rate::Float64=1e-4,
        z_learning_rate_multiplier::Float64=2.0,
        gradient_clip_norm::Float64=1.0,
        include_q_function::Bool=true,
        q_hidden_dim::Int=512,
        max_sequence_length::Int=150
    )
        new(vocab_size, embed_dim, hidden_dim, n_gru_layers,
            learning_rate, z_learning_rate_multiplier, gradient_clip_norm,
            include_q_function, q_hidden_dim, max_sequence_length)
    end
end

# =============================================================================
# Factory Function
# =============================================================================

"""
    create_smiles_gflownet(; config=SMILESGFlowNetConfig(), rng=Random.default_rng(),
                             pretrained_params=nothing)

Create a complete SMILES GFlowNet model ready for training.

# Arguments
- `config`: SMILESGFlowNetConfig with architecture and training settings
- `rng`: Random number generator
- `pretrained_params`: Optional pretrained parameters (from pretraining step)

# Returns
Named tuple with:
- `model`: GFlowNetModel
- `policy_model`: SMILESPolicyModel (for autoregressive operations)
- `vocab`: SMILESVocabulary
- `q_function`: Optional QFunctionNetwork
- `q_params`: Optional Q-function parameters
- `q_states`: Optional Q-function states
- `config`: The config used
"""
function create_smiles_gflownet(;
    config::SMILESGFlowNetConfig=SMILESGFlowNetConfig(),
    rng=Random.default_rng(),
    pretrained_params=nothing
)
    # Create vocabulary
    vocab = SMILESVocabulary()

    # Create GRU policy
    policy_model, policy_ps, policy_st = create_smiles_policy(;
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_gru_layers,
        rng=rng
    )

    # Create initial state and action space
    initial_state = create_initial_smiles_state(;
        max_length=config.max_sequence_length,
        vocab_size=config.vocab_size
    )
    all_actions = create_smiles_actions(config.vocab_size)

    # Use pretrained params if available
    if !isnothing(pretrained_params)
        policy_ps = pretrained_params
    end

    # Build parameters ComponentArray with learnable log_Z
    log_Z = 0.0  # Initialize Z = 1

    parameters = ComponentArray(
        forward=policy_ps,
        log_Z=log_Z
    )

    states = (forward=policy_st,)

    # Create optimizer
    opt = Optimisers.Adam(config.learning_rate)
    optimizer = Optimisers.setup(opt, parameters)

    # Wrap in ForwardPolicy
    forward_policy = ForwardPolicy(policy_model)

    # Create GFlowNetModel
    model = GFlowNetModel(
        initial_state,
        all_actions,
        forward_policy,
        nothing,   # backward_policy (P_B = 1 for SMILES)
        nothing,   # flow_estimator
        log_Z,
        parameters,
        optimizer,
        states
    )

    # Optional Q-function
    q_function = nothing
    q_params = nothing
    q_states = nothing
    q_optimizer = nothing

    if config.include_q_function
        q_function, q_params, q_states = create_q_function(
            config.hidden_dim, config.vocab_size; rng=rng
        )
        q_opt = Optimisers.Adam(config.learning_rate)
        q_optimizer = Optimisers.setup(q_opt, q_params)
    end

    return (
        model=model,
        policy_model=policy_model,
        vocab=vocab,
        q_function=q_function,
        q_params=q_params,
        q_states=q_states,
        q_optimizer=q_optimizer,
        config=config,
    )
end

# =============================================================================
# SMILES GFlowNet Training Config Factory
# =============================================================================

"""
    create_smiles_training_config(; kwargs...)

Create a TrainingConfig pre-configured for SMILES GFlowNet training.

Uses sensible defaults based on the CAFE-GFN framework:
- Shifted-Cosh TB loss
- Reduced Z learning rate (2x instead of 10x)
- Gradient clipping at 1.0
- KL regularization enabled
"""
function create_smiles_training_config(;
    n_iterations::Int=1000,
    batch_size::Int=64,
    learning_rate::Float64=1e-4,
    loss_type::Symbol=:shifted_cosh,
    cosh_delta_threshold::Float64=2.0,    # FIXED: 15.0 → 2.0 (cosh(15)≈1.6M was catastrophic)
    z_learning_rate_multiplier::Float64=2.0,
    gradient_clip_norm::Float64=1.0,
    kl_weight::Float64=0.01,               # LOWERED: 0.1 → 0.01 (Genetic GFN uses 0.001)
    kl_decay_schedule::Symbol=:none,        # CHANGED: :cosine → :none (cosine→0 causes collapse)
    epsilon::Float64=0.05,
    epsilon_decay::Bool=true,
    use_replay_buffer::Bool=true,
    replay_buffer_size::Int=10000,
    replay_ratio::Float64=0.25,
    entropy_weight::Float64=0.02,
    q_masking_quantile::Float64=0.0,
    kwargs...
)
    return GFlowNet.TrainingConfig(
        objective=SHIFTED_COSH_TB,
        partition_function_method=LEARNABLE_ESTIMATION,
        n_iterations=n_iterations,
        batch_size=batch_size,
        learning_rate=learning_rate,
        loss_type=loss_type,
        cosh_delta_threshold=cosh_delta_threshold,
        z_learning_rate_multiplier=z_learning_rate_multiplier,
        gradient_clip_norm=gradient_clip_norm,
        kl_weight=kl_weight,
        kl_decay_schedule=kl_decay_schedule,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        use_replay_buffer=use_replay_buffer,
        replay_buffer_size=replay_buffer_size,
        replay_ratio=replay_ratio,
        entropy_weight=entropy_weight,
        q_masking_quantile=q_masking_quantile;
        kwargs...
    )
end
