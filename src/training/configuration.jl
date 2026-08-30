# Training Configuration
# Mathematical foundations for GFlowNet training configuration and hyperparameters

# =============================================================================
# Training Objective Enumerations
# =============================================================================

"""
    TrainingObjective

Enumeration of GFlowNet training objectives.

# Mathematical Foundation
Different training objectives correspond to different ways of enforcing
the fundamental flow conservation equation F(s) = Σ_{s'} P_F(s'|s) * F(s'):

- TRAJECTORY_BALANCE: Global balance via trajectory probabilities
- DETAILED_BALANCE: Local balance via edge-wise flow conservation
- FLOW_MATCHING: Direct flow estimation with conservation constraints
- SUB_TRAJECTORY_BALANCE: Intermediate-scale balance conditions
- COMBINED_OBJECTIVES: Weighted combination of multiple objectives
"""
@enum TrainingObjective begin
    TRAJECTORY_BALANCE
    DETAILED_BALANCE
    FLOW_MATCHING
    SUB_TRAJECTORY_BALANCE
    DIRECT_FLOW_OBJECTIVE
    COMBINED_OBJECTIVES
    TRAJECTORY_LIKELIHOOD_MAXIMIZATION  # TLM (ICLR 2025) - learns backward policy
    MULTI_OBJECTIVE_TB                  # MOGFN-PC (Gap 5) - preference-conditioned TB
end

"""
    PartitionFunctionMethod

Methods for estimating the partition function Z = F(s₀).

# Mathematical Foundation
The partition function appears in trajectory balance objectives:
∏P_F(s'|s) * Z = R(s_T)

# Available Methods

## LEARNABLE_ESTIMATION (DEFAULT)
- Learns Z = exp(log_Z) as a trainable parameter
- Improves exploration (~42% better mode discovery)
- REQUIRED for SUB_TRAJECTORY_BALANCE, which anchors F(s_0) = Z; with Z pinned the
  sampler collapses onto a single terminal state (measured TV 0.9474)

## SIMPLE_ESTIMATION
- Sets Z = 1 (fixed)
- Correct ONLY when the rewards sum to 1. The trajectory-balance residual is
  (log Z + sum log P_F - log R - sum log P_B)^2; summing its optimum over all
  trajectories reaching x gives p(x) = R(x)/Z, and p is a distribution, so a fixed
  Z = 1 is satisfiable iff sum_x R(x) = 1. Otherwise the residual has a floor it
  can never close: measured 28.021 -> 36.939, i.e. RISING, on a 3-action grid.
- That statement REQUIRES sum_tau P_B(tau|x) = 1, i.e. a normalised backward policy.
  It was silently false for forward-only models until losses.jl:535 was repaired:
  the loss dropped its backward term entirely, which is P_B == 1 unnormalised, and
  the satisfiability condition was the path-weighted sum_x n_paths(x) R(x) instead.
  On the 3x3 grid that is 78.0 against a true Z of 19.0. Uniform-over-parents is now
  used when no backward policy exists, so the plain condition above holds again for
  every configuration.
- Was the default. That made the library's primary objective unsatisfiable out of
  the box for 26 callers, which is why the default is now LEARNABLE_ESTIMATION.

## SAMPLING_ESTIMATION (REJECTED -- not implemented)
- Would estimate Z ~ (1/N) sum_i R(s_i^T) / prod P_F(s_i)
- Nothing implements it: no log_Z is allocated and none is updated, so selecting it
  silently pins Z = 1. validate_training_config now throws rather than let a caller
  believe Monte-Carlo estimation is happening.

## ADAPTIVE_ESTIMATION (REJECTED -- not implemented)
- Would switch methods based on training progress
- Same silent no-op, same refusal.

# Example
```julia
# Enable learnable partition function
config = TrainingConfig(
    partition_function_method = LEARNABLE_ESTIMATION,
    # ... other parameters
)
```
"""
@enum PartitionFunctionMethod begin
    SIMPLE_ESTIMATION
    SAMPLING_ESTIMATION
    LEARNABLE_ESTIMATION
    ADAPTIVE_ESTIMATION
end

"""
    OptimizationMethod

Neural network optimization algorithms.

# Mathematical Foundation
Different optimization strategies for minimizing L(θ):
- ADAM: Adaptive moment estimation with bias correction
- RMSPROP: Root mean square propagation
- SGD: Stochastic gradient descent
- ADAMW: Adam with weight decay regularization
"""
@enum OptimizationMethod begin
    ADAM
    RMSPROP
    SGD
    ADAMW
end

# =============================================================================
# Core Training Configuration
# =============================================================================

"""
    TrainingConfig

Comprehensive configuration for GFlowNet training.

# Mathematical Foundation
Controls all aspects of the optimization process:
min_θ 𝔼[L(θ, τ)] where L is the chosen objective and τ are trajectories.

# Core Parameters
- `objective::TrainingObjective`: Loss function L to minimize
- `n_iterations::Int`: Number of optimization steps
- `batch_size::Int`: Number of trajectories per gradient estimate
- `learning_rate::Float64`: Step size α in θ_{t+1} = θ_t - α∇L(θ_t)

# Regularization Parameters
- `entropy_weight::Float64`: Entropy regularization coefficient (default 0.01)
    - **Breaking change**: Default changed from 0.0 to 0.01 for better mode discovery
    - 0.0 = no entropy regularization (original TB behavior)
    - 0.01-0.1 = recommended range for exploration (AISTATS 2024)
    - Adds -λH(π) to loss, encouraging diverse policies and preventing mode collapse
    - Set to 0.0 explicitly if you need exact backward compatibility
- `parameter_regularization::Float64`: L2 regularization on parameters
- `gradient_clip_norm::Float64`: Maximum gradient norm for stability

# Exploration Parameters (Critical for Mode Discovery!)
- `temperature::Float64`: Temperature T in softmax: P ∝ exp(logits/T)
- `epsilon::Float64`: ε-uniform exploration rate (default 0.05, standard GFlowNet practice)
- `epsilon_decay::Bool`: Whether to linearly anneal epsilon to 0 over training

# ε-Uniform Exploration (Standard Practice)
During trajectory sampling:
    P(a|s) = (1 - ε) × P_F(a|s) + ε × Uniform(valid_actions)

This is essential for mode discovery in GFlowNet (Malkin et al. 2022, Shen et al. ICML 2023).
Without it, TB training gets stuck in local minima and fails to discover all reward modes.

# Training Control
- `validation_frequency::Int`: Steps between validation evaluations
- `checkpoint_frequency::Int`: Steps between model checkpoints
- `early_stopping_patience::Int`: Steps to wait without improvement
- `verbose::Bool`: Enable progress logging
"""
struct TrainingConfig
    # Core training parameters
    objective::TrainingObjective
    partition_function_method::PartitionFunctionMethod
    optimization_method::OptimizationMethod

    # Training schedule
    n_iterations::Int
    batch_size::Int
    learning_rate::Float64

    # Regularization
    entropy_weight::Float64
    parameter_regularization::Float64
    gradient_clip_norm::Float64

    # Temperature and exploration
    temperature::Float64
    exploration_noise::Float64
    epsilon::Float64          # ε-uniform exploration rate (standard: 0.05)
    epsilon_decay::Bool       # Whether to anneal epsilon to 0 over training

    # Monitoring and control
    validation_frequency::Int
    checkpoint_frequency::Int
    early_stopping_patience::Int
    early_stopping_threshold::Float64
    verbose::Bool

    # Advanced configuration
    sub_trajectory_length::Int
    # z_learning_rate_multiplier: Scales the log_Z gradient for faster partition function convergence
    # Reference: Peptide generation paper (bioRxiv 2026) recommends 10x for better results
    # Implementation: gradient scaling before optimizer update (lr_effective = lr × multiplier)
    z_learning_rate_multiplier::Float64

    # Experience Replay (JMLR 2023: GFlowNet Foundations - Off-policy learning)
    use_replay_buffer::Bool              # Whether to use experience replay
    replay_buffer_size::Int              # Maximum buffer capacity
    replay_ratio::Float64                # Ratio of replay vs fresh samples (0.5 = 50% each)
    replay_priority_alpha::Float64       # Priority exponent (0 = uniform, 1 = full priority)

    # TLM: Trajectory Likelihood Maximization (ICLR 2025)
    # Paper: "Optimizing Backward Policies in GFlowNets via Trajectory Likelihood Maximization"
    # Key insight: Max-entropy backward policy is P_B(s|s') ∝ n(s)/n(s') where n(s) = #paths to s
    # This directly addresses path asymmetry by learning path counts implicitly
    tlm_backward_weight::Float64         # Weight for backward policy loss (λ in paper, default 1.0)
    tlm_update_frequency::Int            # Update backward policy every N iterations (default 1)
    tlm_entropy_coeff::Float64           # Entropy coefficient for backward policy (default 0.01)

    # MOGFN-PC: Multi-Objective GFlowNet with Preference Conditioning (Gap 5, ICML 2023)
    # Conditions the policy on a preference vector w ∈ Δ^K (K-simplex) so that a single
    # trained model generates molecules for ANY preference weighting at inference time.
    mogfn_n_objectives::Int              # Number of objectives K (default 4: QED, SA, LogP, MW)
    mogfn_preference_dim::Int            # Preference embedding dimension (default 64)
    mogfn_dirichlet_alpha::Float64       # Dirichlet concentration for preference sampling (default 1.0)

    function TrainingConfig(;
        objective::TrainingObjective=TRAJECTORY_BALANCE,
        # LEARNABLE_ESTIMATION, not SIMPLE_ESTIMATION, and this is a BUG FIX rather
        # than a preference.
        #
        # SIMPLE_ESTIMATION pins Z = 1. The Trajectory Balance residual is then
        # (sum log P_F - log R - sum log P_B)^2, which is zero only if
        # P_F(tau) = R(x) P_B(tau|x) for every trajectory. Summing that over all
        # trajectories reaching x gives p(x) = R(x), so sum_x p(x) = sum_x R(x). The
        # left side is 1 because p is a distribution, so the objective is
        # SATISFIABLE ONLY IF sum_x R(x) = 1 -- which is essentially never.
        #
        # So the previous default made the library's primary objective provably
        # unsatisfiable. It was not a harmless conservative choice; it silently
        # degraded every caller that took the default. Measured: examples/
        # cycle_problem_demo.jl trained with the old default produced a sampler whose
        # mean reward was IDENTICAL to the untrained one, 8.125 vs 8.125, with the
        # loss RISING 28.021 -> 36.939. The only change was this method, after which
        # the same demo goes 3.025 -> 15.575, a 5.1x improvement.
        # test_flow_matching_comprehensive.jl showed the same shape: TB stuck at 21.17
        # after 200 iterations and WORSE at 1000 (25.46), because sharpening the
        # policy only grows a residual it can never close.
        #
        # SIMPLE_ESTIMATION remains available for the case it is actually correct for,
        # rewards that genuinely sum to 1. The cost of the new default is one scalar
        # parameter.
        partition_function_method::PartitionFunctionMethod=LEARNABLE_ESTIMATION,
        optimization_method::OptimizationMethod=ADAM,
        n_iterations::Int=1000,
        batch_size::Int=32,
        learning_rate::Float64=1e-3,
        entropy_weight::Float64=0.01,  # AISTATS 2024: GFlowNets as Entropy-Regularized RL recommends 0.01
        parameter_regularization::Float64=1e-4,
        gradient_clip_norm::Float64=1.0,
        temperature::Float64=1.0,
        exploration_noise::Float64=0.0,
        epsilon::Float64=0.05,
        epsilon_decay::Bool=true,
        validation_frequency::Int=100,
        checkpoint_frequency::Int=500,
        early_stopping_patience::Int=200,
        early_stopping_threshold::Float64=1e-6,
        verbose::Bool=true,
        sub_trajectory_length::Int=10,
        z_learning_rate_multiplier::Float64=10.0,  # 10x faster Z learning per peptide generation paper
        # Experience Replay parameters
        use_replay_buffer::Bool=false,
        replay_buffer_size::Int=10000,
        replay_ratio::Float64=0.5,  # 50% replay, 50% fresh
        replay_priority_alpha::Float64=0.6,
        # TLM parameters (ICLR 2025)
        tlm_backward_weight::Float64=1.0,      # Weight for backward likelihood loss
        tlm_update_frequency::Int=1,            # Update backward every N iterations
        tlm_entropy_coeff::Float64=0.01,        # Entropy coefficient for backward policy
        # MOGFN-PC parameters (Gap 5, ICML 2023)
        mogfn_n_objectives::Int=4,              # Number of objectives (QED, SA, LogP, MW)
        mogfn_preference_dim::Int=64,           # Preference embedding dimension
        mogfn_dirichlet_alpha::Float64=1.0      # Dirichlet concentration (1.0 = uniform simplex)
    )
        # Validation
        if n_iterations <= 0
            throw(ArgumentError("n_iterations must be positive"))
        end
        if batch_size <= 0
            throw(ArgumentError("batch_size must be positive"))
        end
        if learning_rate <= 0
            throw(ArgumentError("learning_rate must be positive"))
        end
        if temperature <= 0
            throw(ArgumentError("temperature must be positive"))
        end
        if gradient_clip_norm <= 0
            throw(ArgumentError("gradient_clip_norm must be positive"))
        end
        if entropy_weight < 0 || parameter_regularization < 0
            throw(ArgumentError("regularization weights must be non-negative"))
        end
        if !(0.0 <= epsilon <= 1.0)
            throw(ArgumentError("epsilon must be in [0.0, 1.0]"))
        end
        if validation_frequency <= 0 || checkpoint_frequency <= 0
            throw(ArgumentError("monitoring frequencies must be positive"))
        end
        if early_stopping_patience <= 0
            throw(ArgumentError("early_stopping_patience must be positive"))
        end
        if sub_trajectory_length <= 0
            throw(ArgumentError("sub_trajectory_length must be positive"))
        end
        if z_learning_rate_multiplier <= 0
            throw(ArgumentError("z_learning_rate_multiplier must be positive"))
        end
        # Replay buffer validation
        if replay_buffer_size <= 0
            throw(ArgumentError("replay_buffer_size must be positive"))
        end
        if !(0.0 <= replay_ratio <= 1.0)
            throw(ArgumentError("replay_ratio must be in [0.0, 1.0]"))
        end
        if !(0.0 <= replay_priority_alpha <= 1.0)
            throw(ArgumentError("replay_priority_alpha must be in [0.0, 1.0]"))
        end
        # TLM validation
        if tlm_backward_weight < 0
            throw(ArgumentError("tlm_backward_weight must be non-negative"))
        end
        if tlm_update_frequency <= 0
            throw(ArgumentError("tlm_update_frequency must be positive"))
        end
        if tlm_entropy_coeff < 0
            throw(ArgumentError("tlm_entropy_coeff must be non-negative"))
        end
        # MOGFN validation
        if mogfn_n_objectives <= 0
            throw(ArgumentError("mogfn_n_objectives must be positive"))
        end
        if mogfn_preference_dim <= 0
            throw(ArgumentError("mogfn_preference_dim must be positive"))
        end
        if mogfn_dirichlet_alpha <= 0
            throw(ArgumentError("mogfn_dirichlet_alpha must be positive"))
        end

        new(objective, partition_function_method, optimization_method,
            n_iterations, batch_size, learning_rate,
            entropy_weight, parameter_regularization, gradient_clip_norm,
            temperature, exploration_noise, epsilon, epsilon_decay,
            validation_frequency, checkpoint_frequency, early_stopping_patience, early_stopping_threshold,
            verbose, sub_trajectory_length, z_learning_rate_multiplier,
            use_replay_buffer, replay_buffer_size, replay_ratio, replay_priority_alpha,
            tlm_backward_weight, tlm_update_frequency, tlm_entropy_coeff,
            mogfn_n_objectives, mogfn_preference_dim, mogfn_dirichlet_alpha)
    end
end

# =============================================================================
# Training State and Progress Tracking
# =============================================================================

"""
    TrainingState

Tracks the current state of training progress.

# Fields
- `iteration::Int`: Current training iteration
- `best_loss::Float64`: Best validation loss achieved
- `patience_counter::Int`: Iterations since last improvement
- `training_losses::Vector{Float64}`: History of training losses
- `validation_losses::Vector{Float64}`: History of validation losses
- `learning_rates::Vector{Float64}`: History of learning rates
- `start_time::DateTime`: Training start timestamp
"""
mutable struct TrainingState
    iteration::Int
    best_loss::Float64
    patience_counter::Int
    training_losses::Vector{Float64}
    validation_losses::Vector{Float64}
    learning_rates::Vector{Float64}
    start_time::DateTime

    function TrainingState()
        new(0, Inf, 0, Float64[], Float64[], Float64[], now())
    end
end

"""
    TrainingMetrics

Comprehensive metrics collected during training.

# Fields
- `loss_components::Dict{String,Float64}`: Breakdown of loss components
- `gradient_norms::Vector{Float64}`: History of gradient norms
- `sampling_stats::Dict{String,Any}`: Trajectory sampling statistics
- `flow_conservation_score::Float64`: Flow conservation validation score
- `partition_function_estimate::Float64`: Current Z estimate
"""
struct TrainingMetrics
    loss_components::Dict{String,Float64}
    gradient_norms::Vector{Float64}
    sampling_stats::Dict{String,Any}
    flow_conservation_score::Float64
    partition_function_estimate::Float64

    function TrainingMetrics()
        new(Dict{String,Float64}(), Float64[], Dict{String,Any}(), 0.0, 1.0)
    end
end

# =============================================================================
# Configuration Validation and Utilities
# =============================================================================

"""
    validate_training_config(config::TrainingConfig, model::GFlowNetModel)

Validate that training configuration is compatible with model.

# Mathematical Validation
Checks that:
1. Objective requirements match model capabilities
2. Parameter ranges are mathematically valid
3. Hyperparameters are in reasonable ranges
4. Model components exist for chosen objective

# Throws
- `ArgumentError` if configuration is invalid
"""
function validate_training_config(config::TrainingConfig, model::GFlowNetModel)
    # ONE up-front check against ONE declarative table, reporting EVERY missing
    # piece at once.
    #
    # This function was previously unusable: line 412 read `model.dag.states`, and
    # GFlowNetModel has no `dag` field, so any caller that got that far threw. That
    # is why nothing called it, and why the real requirements ended up scattered as
    # five separate `throw`s buried inside loss branches
    # (balance.jl:105, balance.jl:128, losses.jl:328, losses.jl:423,
    # training.jl:102). Discovering a missing component by crashing on iteration 1 --
    # or worse, by having train_gflownet swallow the exception and record NaN for
    # every iteration -- is the failure mode that let several demos "run" while
    # learning nothing.
    #
    # It was also WRONG where it did work: it claimed DETAILED_BALANCE needs only a
    # backward policy, and get_objective_requirements claimed SUB_TRAJECTORY_BALANCE
    # needs only a forward policy when it needs four things.
    #
    # TWO SEVERITIES, and the split is deliberate. A missing component that makes the
    # objective DEGENERATE is an error; one that merely makes it SUBOPTIMAL is a
    # warning. Measured basis: SubTB with a fixed Z collapses onto a single terminal
    # state with probability 1.000 on every seed tried (TV 0.9474), so it errors. TB
    # with a fixed Z is unsatisfiable but still trains and still moves the policy, and
    # is correct outright when the rewards sum to 1, so it warns. Refusing to run it
    # would break legitimate code-path tests for no mathematical gain.
    # SAMPLING_ESTIMATION and ADAPTIVE_ESTIMATION are EXPORTED enum values that do
    # nothing. Both are documented "Not implemented" above, only LEARNABLE_ESTIMATION
    # allocates log_Z (interface.jl:89, 114, 135, 154, 173), and only it updates log_Z
    # (training.jl:63). So selecting either silently pins Z = 1 -- the caller believes
    # they chose Monte-Carlo estimation or adaptive switching and instead gets the
    # unsatisfiable fixed-Z regime.
    #
    # Three live examples were doing exactly this: feature_acquisition/main.jl:1610
    # paired ADAPTIVE with SUB_TRAJECTORY_BALANCE, i.e. the configuration measured to
    # collapse onto one terminal state (TV 0.9474); causal_discovery.jl:148 selected
    # SAMPLING with the comment "Complex graph spaces need sampling", claiming a
    # benefit from a no-op. Both "passed" because Z = 1 does not crash.
    #
    # Refuse rather than warn: unlike a fixed Z chosen deliberately, there is no
    # configuration in which these two do what their names say.
    if config.partition_function_method in (SAMPLING_ESTIMATION, ADAPTIVE_ESTIMATION)
        throw(ArgumentError(
            "partition_function_method = $(config.partition_function_method) is NOT " *
            "IMPLEMENTED -- it silently pins Z = 1. Use LEARNABLE_ESTIMATION to train " *
            "Z as a parameter, or SIMPLE_ESTIMATION to pin Z = 1 deliberately (only " *
            "correct when the rewards sum to 1)."))
    end

    reqs = _objective_component_requirements(config.objective)
    missing_parts = String[]

    if reqs.backward && isnothing(model.backward_policy)
        push!(missing_parts, "a backward policy (include_backward = true)")
    end
    if reqs.flow && isnothing(model.flow_estimator)
        push!(missing_parts, "a flow estimator (include_flow_estimator = true)")
    end
    if reqs.learnable_z && !haskey(model.parameters, :log_Z)
        push!(missing_parts,
              "a learnable partition function " *
              "(partition_function_method = LEARNABLE_ESTIMATION)")
    end

    if get(reqs, :warn_learnable_z, false) && !haskey(model.parameters, :log_Z)
        @warn "$(config.objective) is running with a FIXED partition function Z = 1" reason = reqs.why fix = "partition_function_method = LEARNABLE_ESTIMATION"
    end

    if !isempty(missing_parts)
        throw(ArgumentError(
            "$(config.objective) cannot be trained on this model. Missing: " *
            join(missing_parts, "; ") * ". " * reqs.why
        ))
    end

    # The partition-function method has to agree between model and config, because
    # train_step! reads it from the CONFIG when deciding whether to update log_Z.
    # Setting it on only one of the two silently freezes Z at its initial value, which
    # looks like slow convergence rather than a misconfiguration -- and a model can
    # carry a log_Z parameter that never moves, which is the most confusing state of
    # all.
    if config.partition_function_method != LEARNABLE_ESTIMATION &&
       haskey(model.parameters, :log_Z)
        if reqs.learnable_z
            throw(ArgumentError(
                "$(config.objective) requires partition_function_method = " *
                "LEARNABLE_ESTIMATION on the TrainingConfig as well as on the model; " *
                "got $(config.partition_function_method). train_step! reads the " *
                "method from the config, so Z would stay frozen at its initial value."
            ))
        else
            @warn "Model has a learnable log_Z but the config says $(config.partition_function_method), so Z will NOT be updated" fix = "set partition_function_method = LEARNABLE_ESTIMATION on the TrainingConfig too"
        end
    end

    # Numeric sanity. Warnings, not errors: these are unusual, not impossible.
    if config.temperature < 0.1 || config.temperature > 10.0
        @warn "Temperature $(config.temperature) is outside typical range [0.1, 10.0]"
    end

    if config.learning_rate > 1e-1
        @warn "Learning rate $(config.learning_rate) is quite high, may cause instability"
    end

    if config.entropy_weight > 1.0
        @warn "High entropy weight $(config.entropy_weight) may prevent convergence"
    end

    if config.parameter_regularization > 1e-1
        @warn "High regularization $(config.parameter_regularization) may prevent learning"
    end

    return nothing
end

"""
    _objective_component_requirements(objective) -> NamedTuple

THE single source of truth for what each objective needs from a model.

Every requirement here is one that the corresponding loss branch actually
dereferences, verified by reading the branch -- not a guess. `why` explains the
mathematical reason, so a failure tells the caller what to fix and why it matters.
"""
function _objective_component_requirements(objective::TrainingObjective)
    if objective == TRAJECTORY_BALANCE
        # learnable_z is a WARNING for TB, not an error. TB with Z pinned to 1 is
        # unsatisfiable but it still runs and still moves the policy, so refusing to
        # train would break legitimate code-path tests and anyone whose rewards
        # genuinely sum to 1. Contrast SUB_TRAJECTORY_BALANCE below, where a fixed Z
        # collapses the sampler onto one state deterministically -- that one errors.
        return (backward = false, flow = false, learnable_z = false,
                warn_learnable_z = true,
                why = "TB balances log Z + sum log P_F against log R + sum log P_B. " *
                      "With Z pinned to 1 the residual cannot reach zero unless the " *
                      "rewards sum to 1, so the objective is unsatisfiable and " *
                      "training barely moves the sampler.")

    elseif objective == DETAILED_BALANCE
        # Flow estimator is OPTIONAL, not required. losses.jl falls back to the
        # recursive flow() when there is none: F is then non-trainable, so the flow
        # gradient is zero, but the objective still works and the reward still enters
        # through the terminal boundary F(x) = R(x). Claiming it as required broke
        # five existing DB tests that deliberately exercise that path.
        return (backward = true, flow = false, learnable_z = false,
                why = "DB enforces P_F(s'|s) F(s) = P_B(s|s') F(s'), so it needs P_B. " *
                      "A flow estimator is optional: with one, F is learned; without " *
                      "one, F is the recursive flow and carries no gradient. Z does " *
                      "not appear in DB at all.")

    elseif objective == FLOW_MATCHING
        # Flow estimator is genuinely REQUIRED here, unlike for DB. FM's entire
        # content is the conservation law on F, and losses.jl has no non-estimator
        # path that can train it -- without one there is nothing to optimise.
        return (backward = false, flow = true, learnable_z = false,
                why = "FM enforces sum over parents of F(p) P_F(s|p) = F(s) with " *
                      "F(x) = R(x), so F must exist and be trainable. P_B does not " *
                      "appear in the residual, and Z does not either -- F(s0) plays " *
                      "that role.")

    elseif objective == SUB_TRAJECTORY_BALANCE
        return (backward = true, flow = true, learnable_z = true,
                why = "SubTB balances log F(s_i) + sum log P_F against " *
                      "log F(s_j) + sum log P_B, anchored at F(s_0) = Z and " *
                      "F(x) = R(x). Under a fixed Z = 1 the root anchor contradicts " *
                      "the true partition function and the sampler collapses onto a " *
                      "single terminal state.")

    elseif objective == TRAJECTORY_LIKELIHOOD_MAXIMIZATION
        # Same severity reasoning as TB: it shares TB's trajectory residual, so a
        # fixed Z makes it unsatisfiable but not degenerate. Warn, do not refuse.
        return (backward = true, flow = false, learnable_z = false,
                warn_learnable_z = true,
                why = "TLM trains the backward policy directly, so P_B must exist, " *
                      "and it shares TB's trajectory residual, so a fixed Z leaves " *
                      "that residual with a floor it cannot close.")

    elseif objective == MULTI_OBJECTIVE_TB
        return (backward = false, flow = false, learnable_z = false,
                why = "MOGFN needs a preference encoder and a Z network, which only " *
                      "create_mogfn_gflownet builds; that is checked in its own loss " *
                      "branch because the components are not fields of a plain model.")

    elseif objective == DIRECT_FLOW_OBJECTIVE
        return (backward = false, flow = false, learnable_z = false,
                why = "DIRECT_FLOW_OBJECTIVE is disabled: its loss was constant with " *
                      "respect to the parameters, so training under it did nothing. " *
                      "Use TRAJECTORY_BALANCE.")

    else
        return (backward = false, flow = false, learnable_z = false,
                why = "No declared component requirements.")
    end
end

"""
    get_objective_requirements(objective::TrainingObjective)::Vector{String}

Get model component requirements for a training objective.

# Returns
Vector of strings naming required model components.
"""
function get_objective_requirements(objective::TrainingObjective)::Vector{String}
    # DERIVED from _objective_component_requirements, never restated. The two used to
    # disagree: this function claimed SUB_TRAJECTORY_BALANCE needs only
    # ["forward_policy"], when its loss dereferences the backward policy, the flow
    # estimator AND log_Z. Duplicated truth tables drift, and this one had.
    reqs = _objective_component_requirements(objective)
    parts = ["forward_policy"]           # every objective reads P_F
    reqs.backward    && push!(parts, "backward_policy")
    reqs.flow        && push!(parts, "flow_estimator")
    (reqs.learnable_z || get(reqs, :warn_learnable_z, false)) && push!(parts, "learnable_log_Z")

    # MOGFN's extra components are not fields of a plain GFlowNetModel, so they are
    # named here rather than in the boolean table.
    if objective == MULTI_OBJECTIVE_TB
        append!(parts, ["preference_encoder", "z_network"])
    end

    return parts
end

"""
    estimate_training_time(config::TrainingConfig, model::GFlowNetModel)::Float64

Estimate training time in seconds based on configuration and model size.

# Mathematical Foundation
Estimates based on:
- Model complexity: O(|S| + |A| + network_params)
- Batch computation: O(batch_size * avg_trajectory_length)
- Training iterations: O(n_iterations)

# Returns
Estimated training time in seconds.
"""
function estimate_training_time(config::TrainingConfig, model::GFlowNetModel)::Float64
    # Base time per iteration (empirical estimate)
    base_time_per_iteration = 0.1  # seconds

    # Scale by model complexity
    n_states = length(model.dag.states)
    n_actions = length(model.dag.actions)
    complexity_factor = log(n_states + n_actions + 1000) / log(1000)  # Normalized log scale

    # Scale by batch size
    batch_factor = config.batch_size / 32  # Normalize to batch_size=32

    # Scale by objective complexity
    objective_factors = Dict(
        TRAJECTORY_BALANCE => 1.0,
        DETAILED_BALANCE => 1.5,
        FLOW_MATCHING => 1.3,
        SUB_TRAJECTORY_BALANCE => 1.2,
        COMBINED_OBJECTIVES => 2.0
    )
    objective_factor = get(objective_factors, config.objective, 1.0)

    time_per_iteration = base_time_per_iteration * complexity_factor * batch_factor * objective_factor
    total_time = time_per_iteration * config.n_iterations

    return total_time
end

"""
    create_optimizer(method::OptimizationMethod, learning_rate::Float64)

Create optimizer instance for the specified method.

# Mathematical Foundation
Different optimizers implement different update rules:
- Adam: Uses adaptive learning rates with momentum
- RMSProp: Scales learning rate by running average of gradients
- SGD: Simple gradient descent with optional momentum
"""
function create_optimizer(method::OptimizationMethod, learning_rate::Float64)
    if method == ADAM
        return Optimisers.Adam(learning_rate)
    elseif method == RMSPROP
        return Optimisers.RMSProp(learning_rate)
    elseif method == SGD
        return Optimisers.Descent(learning_rate)
    elseif method == ADAMW
        return Optimisers.AdamW(learning_rate)
    else
        throw(ArgumentError("Unknown optimization method: $method"))
    end
end

# =============================================================================
# Configuration Presets
# =============================================================================

"""
    create_default_config(objective::TrainingObjective = TRAJECTORY_BALANCE)::TrainingConfig

Create default training configuration for a specific objective.

# Mathematical Foundation
Provides reasonable default hyperparameters based on empirical GFlowNet research:
- Learning rates in range [1e-4, 1e-2]
- Batch sizes that balance computation and gradient noise
- Regularization that prevents overfitting without hampering learning
"""
function create_default_config(objective::TrainingObjective=TRAJECTORY_BALANCE)::TrainingConfig
    base_config = Dict{Symbol,Any}(
        :objective => objective,
        :n_iterations => 1000,
        :batch_size => 32,
        :learning_rate => 1e-3,
        :verbose => true
    )

    # Objective-specific adjustments
    if objective == DETAILED_BALANCE
        base_config[:learning_rate] = 5e-4  # More conservative for DB
        base_config[:entropy_weight] = 0.01  # Help exploration
    elseif objective == FLOW_MATCHING
        base_config[:parameter_regularization] = 1e-3  # Stronger regularization
    elseif objective == SUB_TRAJECTORY_BALANCE
        base_config[:sub_trajectory_length] = 5
    elseif objective == COMBINED_OBJECTIVES
        base_config[:n_iterations] = 1500  # More iterations for complex objective
    end

    return TrainingConfig(; base_config...)
end

"""
    create_fast_config()::TrainingConfig

Create configuration optimized for fast training (development/testing).
"""
function create_fast_config()::TrainingConfig
    return TrainingConfig(
        n_iterations=100,
        batch_size=16,
        learning_rate=1e-2,
        validation_frequency=20,
        checkpoint_frequency=50,
        verbose=true
    )
end

"""
    create_robust_config()::TrainingConfig

Create configuration optimized for robust, stable training.
"""
function create_robust_config()::TrainingConfig
    return TrainingConfig(
        n_iterations=2000,
        batch_size=64,
        learning_rate=5e-4,
        entropy_weight=0.001,
        parameter_regularization=1e-4,
        gradient_clip_norm=0.5,
        early_stopping_patience=300,
        validation_frequency=50,
        verbose=true
    )
end

# =============================================================================
# Display Methods
# =============================================================================

function Base.show(io::IO, objective::TrainingObjective)
    objective_names = Dict(
        TRAJECTORY_BALANCE => "Trajectory Balance",
        DETAILED_BALANCE => "Detailed Balance",
        FLOW_MATCHING => "Flow Matching",
        SUB_TRAJECTORY_BALANCE => "Sub-Trajectory Balance",
        DIRECT_FLOW_OBJECTIVE => "Direct Flow Objective",
        COMBINED_OBJECTIVES => "Combined Objectives",
        TRAJECTORY_LIKELIHOOD_MAXIMIZATION => "TLM (ICLR 2025)",
        MULTI_OBJECTIVE_TB => "MOGFN-PC (Gap 5)"
    )
    print(io, get(objective_names, objective, "Unknown Objective"))
end

function Base.show(io::IO, method::PartitionFunctionMethod)
    method_names = Dict(
        SIMPLE_ESTIMATION => "Simple Estimation",
        SAMPLING_ESTIMATION => "Sampling Estimation",
        LEARNABLE_ESTIMATION => "Learnable Parameter",
        ADAPTIVE_ESTIMATION => "Adaptive Estimation"
    )
    print(io, get(method_names, method, "Unknown Method"))
end

function Base.show(io::IO, opt_method::OptimizationMethod)
    print(io, string(opt_method))
end

function Base.show(io::IO, config::TrainingConfig)
    print(io, "TrainingConfig($(config.objective), lr=$(config.learning_rate), batch=$(config.batch_size), iter=$(config.n_iterations), ε=$(config.epsilon))")
end

function Base.show(io::IO, ::MIME"text/plain", config::TrainingConfig)
    println(io, "GFlowNet Training Configuration:")
    println(io, "  Objective: $(config.objective)")
    println(io, "  Optimization: $(config.optimization_method)")
    println(io, "  Iterations: $(config.n_iterations)")
    println(io, "  Batch size: $(config.batch_size)")
    println(io, "  Learning rate: $(config.learning_rate)")
    println(io, "  Temperature: $(config.temperature)")
    println(io, "  Epsilon (ε-uniform): $(config.epsilon)$(config.epsilon_decay ? " (annealed)" : "")")
    if config.entropy_weight > 0
        println(io, "  Entropy weight: $(config.entropy_weight)")
    end
    if config.parameter_regularization > 0
        println(io, "  L2 regularization: $(config.parameter_regularization)")
    end
    println(io, "  Gradient clipping: $(config.gradient_clip_norm)")
    println(io, "  Early stopping patience: $(config.early_stopping_patience)")
end

function Base.show(io::IO, state::TrainingState)
    elapsed = now() - state.start_time
    print(io, "TrainingState(iter=$(state.iteration), best_loss=$(round(state.best_loss, digits=4)), elapsed=$elapsed)")
end
