# Trajectory Sampling Configuration - On-Demand Approach Compatible
# Configuration types and utilities for GFlowNet trajectory sampling

using Random
using StatsBase: Weights, sample
using Zygote

# =============================================================================
# Sampling Strategy Types and Configuration
# =============================================================================

"""
    SamplingStrategy

Enumeration of trajectory sampling strategies.

# Mathematical Foundation
Different approaches to trajectory sampling in GFlowNet:
- `:stochastic`: Sample according to policy probabilities P_F(a|s)
- `:greedy`: Always choose highest probability action (deterministic)
- `:temperature`: Temperature-scaled sampling for exploration control
"""
@enum SamplingStrategy begin
    STOCHASTIC_SAMPLING
    GREEDY_SAMPLING
    TEMPERATURE_SAMPLING
end

"""
    SamplingConfig

Configuration for trajectory sampling algorithms.

# Fields
- `strategy::SamplingStrategy`: Sampling strategy to use
- `max_trajectory_length::Int`: Maximum allowed trajectory length
- `temperature::Float64`: Temperature for temperature sampling (> 0.0)
- `epsilon::Float64`: ε-uniform exploration rate (0.0 to 1.0, default 0.0)
- `enable_early_stopping::Bool`: Stop if stuck in non-terminal state
- `validation_mode::Bool`: Enable additional validation during sampling
- `acyclic_rate::Float64`: Rate for acyclic trajectory enforcement

# ε-Uniform Exploration
When `epsilon > 0`, action sampling uses:
    P(a|s) = (1 - ε) × P_F(a|s) + ε × Uniform(applicable_actions)

This is the standard exploration technique in GFlowNet (Malkin et al. 2022, ICML 2023).
Typical values: ε = 0.05 for training, ε = 0 for evaluation.
"""
struct SamplingConfig
    strategy::SamplingStrategy
    max_trajectory_length::Int
    temperature::Float64
    epsilon::Float64
    enable_early_stopping::Bool
    validation_mode::Bool
    acyclic_rate::Float64

    function SamplingConfig(;
        strategy::SamplingStrategy=STOCHASTIC_SAMPLING,
        max_trajectory_length::Int=100,
        temperature::Float64=1.0,
        epsilon::Float64=0.0,
        enable_early_stopping::Bool=true,
        validation_mode::Bool=false,
        acyclic_rate::Float64=0.0
    )
        if max_trajectory_length <= 0
            throw(ArgumentError("max_trajectory_length must be positive"))
        end
        if temperature <= 0.0
            throw(ArgumentError("temperature must be positive"))
        end
        if !(0.0 <= epsilon <= 1.0)
            throw(ArgumentError("epsilon must be in [0.0, 1.0]"))
        end
        if !(0.0 <= acyclic_rate <= 1.0)
            throw(ArgumentError("acyclic_rate must be in [0.0, 1.0]"))
        end

        new(strategy, max_trajectory_length, temperature, epsilon, enable_early_stopping, validation_mode, acyclic_rate)
    end
end

# =============================================================================
# Sampling Utilities - Compatible with On-Demand Approach
# =============================================================================

"""
    validate_trajectory_step(current_state, action, next_state, applicable_actions)

Validate a single trajectory step for consistency.
"""
function validate_trajectory_step(current_state, action, next_state, applicable_actions)
    # Check that action was applicable
    if action ∉ applicable_actions
        throw(ArgumentError("Action $action was not in applicable actions $applicable_actions"))
    end

    # Check that the transition is correct
    expected_next_state = apply_action(action, current_state)
    if expected_next_state != next_state
        throw(ArgumentError("Expected next state $expected_next_state but got $next_state"))
    end

    # Check that we don't apply actions to terminal states
    if is_terminal_state(current_state)
        throw(ArgumentError("Cannot apply action to terminal state $current_state"))
    end
end

"""
    compute_trajectory_probability(model::GFlowNetModel, trajectory::Trajectory, params)

Compute the log probability of a trajectory under the forward policy.

# Mathematical Foundation
Computes log P_F(τ) = Σ_t log P_F(a_t | s_t)
"""
function compute_trajectory_probability(model::GFlowNetModel, trajectory::Trajectory, params)
    if isempty(trajectory.actions)
        return 0.0
    end

    log_prob = 0.0

    for i in 1:length(trajectory.actions)
        state = trajectory.states[i]
        action = trajectory.actions[i]

        # Get applicable actions for this state
        applicable_actions = get_applicable_actions(state, model.all_actions)

        if isempty(applicable_actions)
            return -Inf  # Invalid trajectory
        end

        # Compute forward probability for this action
        action_prob = forward_probability(
            model.forward_policy, state, action,
            params.forward, model.states.forward, model.all_actions
        )

        if action_prob <= 0
            return -Inf  # Invalid action
        end

        log_prob += log(action_prob)
    end

    return log_prob
end

"""
    create_default_sampling_config(; strategy=STOCHASTIC_SAMPLING)

Create a default sampling configuration with standard settings.
"""
function create_default_sampling_config(; strategy=STOCHASTIC_SAMPLING, acyclic_rate=0.0)
    return SamplingConfig(
        strategy=strategy,
        max_trajectory_length=100,
        temperature=1.0,
        enable_early_stopping=true,
        validation_mode=false,
        acyclic_rate=acyclic_rate
    )
end

"""
    create_greedy_sampling_config()

Create a sampling configuration for greedy (deterministic) sampling.
"""
function create_greedy_sampling_config(; acyclic_rate=0.0)
    return SamplingConfig(
        strategy=GREEDY_SAMPLING,
        max_trajectory_length=100,
        temperature=1.0,  # Not used for greedy
        enable_early_stopping=true,
        validation_mode=false,
        acyclic_rate=acyclic_rate
    )
end

"""
    create_exploration_sampling_config(; temperature=1.0, epsilon=0.05)

Create a sampling configuration for exploration with ε-uniform mixing.

This is the standard GFlowNet exploration strategy (Malkin et al. 2022, ICML 2023).
With probability ε, sample uniformly; otherwise sample from policy.

# Arguments
- `temperature::Float64 = 1.0`: Temperature scaling (1.0 = no scaling)
- `epsilon::Float64 = 0.05`: ε-uniform exploration rate (standard value)
- `acyclic_rate::Float64 = 0.0`: Rate for acyclic trajectory enforcement

# References
- Malkin et al. (2022) "Trajectory Balance" uses ε=0.05 as default
- Shen et al. (ICML 2023) "Towards Understanding and Improving GFlowNet Training"
"""
function create_exploration_sampling_config(; temperature::Float64=1.0, epsilon::Float64=0.05, acyclic_rate::Float64=0.0)
    return SamplingConfig(
        strategy = temperature != 1.0 ? TEMPERATURE_SAMPLING : STOCHASTIC_SAMPLING,
        max_trajectory_length=100,
        temperature=temperature,
        epsilon=epsilon,
        enable_early_stopping=true,
        validation_mode=false,
        acyclic_rate=acyclic_rate
    )
end



# =============================================================================
# Trajectory Analysis Utilities
# =============================================================================

"""
    analyze_trajectory_diversity(trajectories::Vector{Trajectory})

Analyze the diversity of a collection of trajectories.
"""
function analyze_trajectory_diversity(trajectories::Vector{Trajectory})
    if isempty(trajectories)
        return (unique_trajectories=0, unique_terminals=0, avg_length=0.0)
    end

    # Count unique trajectories (by terminal state)
    terminal_states = Set()
    total_length = 0

    for traj in trajectories
        if !isempty(traj.states)
            push!(terminal_states, traj.states[end])
            total_length += length(traj.actions)
        end
    end

    unique_terminals = length(terminal_states)
    avg_length = total_length / length(trajectories)

    return (
        unique_trajectories=length(unique(trajectories)),
        unique_terminals=unique_terminals,
        avg_length=avg_length
    )
end

"""
    filter_valid_trajectories(trajectories::Vector{Trajectory})

Filter trajectories to only include valid ones (ending in terminal states).
"""
function filter_valid_trajectories(trajectories::Vector{Trajectory})
    return filter(trajectories) do traj
        !isempty(traj.states) && is_terminal_state(traj.states[end])
    end
end

# =============================================================================
# Display Functions
# =============================================================================

function Base.show(io::IO, strategy::SamplingStrategy)
    strategy_name = if strategy == STOCHASTIC_SAMPLING
        "Stochastic"
    elseif strategy == GREEDY_SAMPLING
        "Greedy"
    elseif strategy == TEMPERATURE_SAMPLING
        "Temperature"
    else
        "Unknown"
    end
    print(io, "$(strategy_name) Sampling")
end

function Base.show(io::IO, config::SamplingConfig)
    print(io, "SamplingConfig($(config.strategy), max_length=$(config.max_trajectory_length)")
    if config.strategy == TEMPERATURE_SAMPLING
        print(io, ", temp=$(config.temperature)")
    end
    if config.epsilon > 0.0
        print(io, ", ε=$(config.epsilon)")
    end
    if config.acyclic_rate > 0.0
        print(io, ", acyclic_rate=$(config.acyclic_rate)")
    end
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", config::SamplingConfig)
    println(io, "SamplingConfig:")
    println(io, "  Strategy: $(config.strategy)")
    println(io, "  Max length: $(config.max_trajectory_length)")
    println(io, "  Temperature: $(config.temperature)")
    println(io, "  Epsilon (ε-uniform): $(config.epsilon)")
    println(io, "  Early stopping: $(config.enable_early_stopping)")
    println(io, "  Validation mode: $(config.validation_mode)")
    println(io, "  Acyclic rate: $(config.acyclic_rate)")
end

# =============================================================================
# Exports
# =============================================================================

export SamplingStrategy, STOCHASTIC_SAMPLING, GREEDY_SAMPLING, TEMPERATURE_SAMPLING
export SamplingConfig
export validate_trajectory_step, compute_trajectory_probability
export create_default_sampling_config, create_greedy_sampling_config, create_exploration_sampling_config
export analyze_trajectory_diversity, filter_valid_trajectories

# =============================================================================
# Architecture Notes
# =============================================================================

"""
This simplified sampling module provides only configuration and utility functions.
The actual sampling implementations are in interface.jl using the on-demand approach.

Key principles:
1. ✅ NO DUPLICATE FUNCTIONS - Avoids method overwriting during precompilation
2. ✅ ON-DEMAND COMPATIBLE - All utilities work with the new architecture
3. ✅ CLEAN SEPARATION - Configuration here, implementation in interface.jl
4. ✅ MATHEMATICAL CONSISTENCY - All probability computations are well-defined

The sampling functions in interface.jl use these configurations but implement
the actual trajectory sampling using on-demand computation of applicable actions
and state transitions, eliminating the need for explicit DAG construction.
"""
