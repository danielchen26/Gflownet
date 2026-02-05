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
end

"""
    PartitionFunctionMethod

Methods for estimating the partition function Z = F(s₀).

# Mathematical Foundation
The partition function appears in trajectory balance objectives:
∏P_F(s'|s) * Z = R(s_T)

# Available Methods

## SIMPLE_ESTIMATION
- Sets Z = 1 (fixed)
- Valid when starting from a fixed initial state
- Default method for simplicity

## LEARNABLE_ESTIMATION (Recommended)
- Learns Z = exp(log_Z) as a trainable parameter
- Improves exploration (~42% better mode discovery)
- Ensures theoretical correctness of trajectory balance
- Prepares for future multi-start GFlowNets

## SAMPLING_ESTIMATION (Not implemented)
- Would estimate Z ≈ (1/N) Σᵢ R(sᵢᵀ) / ∏P_F(sᵢ)
- Monte Carlo estimation from samples

## ADAPTIVE_ESTIMATION (Not implemented)
- Would switch methods based on training progress

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
    # NOTE: z_learning_rate_multiplier is defined but NOT YET IMPLEMENTED
    # It exists for API compatibility with literature (peptide generation paper recommends 10x)
    # but requires optimizer refactoring to work properly. Use default for now.
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

    function TrainingConfig(;
        objective::TrainingObjective=TRAJECTORY_BALANCE,
        partition_function_method::PartitionFunctionMethod=SIMPLE_ESTIMATION,
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
        tlm_entropy_coeff::Float64=0.01         # Entropy coefficient for backward policy
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

        new(objective, partition_function_method, optimization_method,
            n_iterations, batch_size, learning_rate,
            entropy_weight, parameter_regularization, gradient_clip_norm,
            temperature, exploration_noise, epsilon, epsilon_decay,
            validation_frequency, checkpoint_frequency, early_stopping_patience, early_stopping_threshold,
            verbose, sub_trajectory_length, z_learning_rate_multiplier,
            use_replay_buffer, replay_buffer_size, replay_ratio, replay_priority_alpha,
            tlm_backward_weight, tlm_update_frequency, tlm_entropy_coeff)
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
    # Check objective compatibility
    if config.objective == DETAILED_BALANCE && isnothing(model.backward_policy)
        throw(ArgumentError("Detailed balance requires backward policy"))
    end

    if config.objective == FLOW_MATCHING && isnothing(model.flow_estimator)
        throw(ArgumentError("Flow matching requires flow estimator"))
    end

    if config.objective == TRAJECTORY_LIKELIHOOD_MAXIMIZATION && isnothing(model.backward_policy)
        throw(ArgumentError("TLM requires backward policy - use include_backward_policy=true in create_gflownet"))
    end

    # Check mathematical constraints
    if config.temperature < 0.1 || config.temperature > 10.0
        @warn "Temperature $(config.temperature) is outside typical range [0.1, 10.0]"
    end

    if config.learning_rate > 1e-1
        @warn "Learning rate $(config.learning_rate) is quite high, may cause instability"
    end

    if config.batch_size > length(model.dag.states)
        @warn "Batch size larger than number of states in DAG"
    end

    # Check regularization
    if config.entropy_weight > 1.0
        @warn "High entropy weight $(config.entropy_weight) may prevent convergence"
    end

    if config.parameter_regularization > 1e-1
        @warn "High regularization $(config.parameter_regularization) may prevent learning"
    end
end

"""
    get_objective_requirements(objective::TrainingObjective)::Vector{String}

Get model component requirements for a training objective.

# Returns
Vector of strings naming required model components.
"""
function get_objective_requirements(objective::TrainingObjective)::Vector{String}
    if objective == TRAJECTORY_BALANCE
        return ["forward_policy"]
    elseif objective == DETAILED_BALANCE
        return ["forward_policy", "backward_policy"]
    elseif objective == FLOW_MATCHING
        return ["forward_policy", "flow_estimator"]
    elseif objective == SUB_TRAJECTORY_BALANCE
        return ["forward_policy"]
    elseif objective == COMBINED_OBJECTIVES
        return ["forward_policy"]  # May require others depending on weights
    elseif objective == TRAJECTORY_LIKELIHOOD_MAXIMIZATION
        return ["forward_policy", "backward_policy"]  # TLM requires both policies
    else
        return String[]
    end
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
- RMSprop: Scales learning rate by running average of gradients
- SGD: Simple gradient descent with optional momentum
"""
function create_optimizer(method::OptimizationMethod, learning_rate::Float64)
    if method == ADAM
        return Optimisers.Adam(learning_rate)
    elseif method == RMSPROP
        return Optimisers.RMSprop(learning_rate)
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
        TRAJECTORY_LIKELIHOOD_MAXIMIZATION => "TLM (ICLR 2025)"
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
