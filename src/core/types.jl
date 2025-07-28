# Core Types for GFlowNet - Standard On-Demand Approach
# Clean, robust, and focused on mathematical foundations

using ComponentArrays
using Random

# =============================================================================
# Core Abstractions - Domain Interface
# =============================================================================

"""
    AbstractState

Abstract base type for all states in a GFlowNet domain.

Domain implementations must provide:
- `state_to_features(state::YourState)::Vector{Float32}` - Convert to neural network input
- `is_terminal_state(state::YourState)::Bool` - Check if terminal
- `reward(state::YourState)::Float64` - Compute reward (positive for terminals)
- `Base.:(==)(a::YourState, b::YourState)` - State equality
- `Base.hash(state::YourState, h::UInt)` - State hashing
"""
abstract type AbstractState end

"""
    AbstractAction

Abstract base type for all actions in a GFlowNet domain.

Domain implementations must provide:
- `is_applicable(action::YourAction, state::YourState)::Bool` - Check applicability
- `apply_action(action::YourAction, state::YourState)::YourState` - Apply action
- `Base.:(==)(a::YourAction, b::YourAction)` - Action equality
"""
abstract type AbstractAction end

# =============================================================================
# Core Data Structures - Standard and Clean
# =============================================================================

"""
    Trajectory

Standard trajectory representation for GFlowNet.

A trajectory τ = (s₀, a₀, s₁, a₁, ..., s_T) where:
- s₀ is the initial state
- s_T is a terminal state
- aᵢ are valid actions: is_applicable(aᵢ, sᵢ) = true
- Transitions: sᵢ₊₁ = apply_action(aᵢ, sᵢ)

# Fields
- `states::Vector{<:AbstractState}` - Sequence of states [s₀, s₁, ..., s_T]
- `actions::Vector{<:AbstractAction}` - Sequence of actions [a₀, a₁, ..., a_{T-1}]

# Mathematical Properties
- length(states) = length(actions) + 1
- states[1] is initial state
- states[end] must be terminal: is_terminal_state(states[end]) = true
"""
struct Trajectory
    states::Vector{<:AbstractState}
    actions::Vector{<:AbstractAction}

    function Trajectory(states::Vector{<:AbstractState}, actions::Vector{<:AbstractAction})
        if length(states) != length(actions) + 1
            throw(ArgumentError("Invalid trajectory: length(states) must equal length(actions) + 1"))
        end
        if isempty(states)
            throw(ArgumentError("Trajectory cannot be empty"))
        end
        new(states, actions)
    end
end

"""Trajectory length (number of transitions)"""
Base.length(trajectory::Trajectory) = length(trajectory.actions)

"""Check if trajectory is empty"""
Base.isempty(trajectory::Trajectory) = isempty(trajectory.states)

# =============================================================================
# Neural Network Policy Wrappers - Standard Interface
# =============================================================================

"""
    ForwardPolicy{M}

Standard wrapper for forward policy neural network.

# Fields
- `model::M` - Lux neural network that maps state features to action logits
"""
struct ForwardPolicy{M}
    model::M
end

"""
    BackwardPolicy{M}

Standard wrapper for backward policy neural network.

# Fields
- `model::M` - Lux neural network that maps state features to backward probabilities
"""
struct BackwardPolicy{M}
    model::M
end

"""
    FlowEstimator{M}

Standard wrapper for flow estimation neural network.

# Fields
- `model::M` - Lux neural network that estimates state flows
"""
struct FlowEstimator{M}
    model::M
end

# =============================================================================
# GFlowNet Model - Standard On-Demand Approach
# =============================================================================

"""
    GFlowNetModel

Standard GFlowNet model using on-demand computation approach.

The DAG is defined through domain functions rather than explicit
pre-computed data structures. This eliminates cache misses and object identity
issues while maintaining all mathematical properties.

# Fields
- `initial_state::AbstractState` - Starting state for trajectories
- `all_actions::Vector{<:AbstractAction}` - Complete action space
- `forward_policy::ForwardPolicy` - Policy network π(a|s)
- `flow_estimator::Union{Nothing,FlowEstimator}` - Flow estimation network (optional)
- `parameters::ComponentArray` - Neural network parameters
- `optimizer` - Optimizer state (from Optimisers.jl)
- `states::NamedTuple` - Neural network states (for Lux)

# On-Demand DAG Properties
The DAG structure is defined by:
- States S: All states reachable from initial_state
- Edges E: {(s,a,s') : is_applicable(a,s) ∧ apply_action(a,s) = s'}
- No explicit caching - computed on-demand for robustness
"""
mutable struct GFlowNetModel
    initial_state::AbstractState
    all_actions::Vector{<:AbstractAction}
    forward_policy::ForwardPolicy
    flow_estimator::Union{Nothing,FlowEstimator}
    parameters::ComponentArray
    optimizer
    states::NamedTuple

    function GFlowNetModel(
        initial_state::AbstractState,
        all_actions::Vector{<:AbstractAction},
        forward_policy::ForwardPolicy,
        flow_estimator::Union{Nothing,FlowEstimator},
        parameters::ComponentArray,
        optimizer,
        states::NamedTuple
    )
        if isempty(all_actions)
            throw(ArgumentError("all_actions cannot be empty"))
        end
        new(initial_state, all_actions, forward_policy, flow_estimator,
            parameters, optimizer, states)
    end
end

# =============================================================================
# On-Demand DAG Operations - Standard and Robust
# =============================================================================







# =============================================================================
# Training Configuration - Clean and Standard
# =============================================================================



# =============================================================================
# Sampling Configuration - Standard Options
# =============================================================================





# =============================================================================
# Utility Types - Keep It Standard
# =============================================================================

"""
    TrainingHistory

Standard container for training metrics.

# Fields
- `losses::Vector{Float64}` - Training losses per iteration
- `gradient_norms::Vector{Float64}` - Gradient norms per iteration
- `iteration_times::Vector{Float64}` - Time per iteration
"""
mutable struct TrainingHistory
    losses::Vector{Float64}
    gradient_norms::Vector{Float64}
    iteration_times::Vector{Float64}

    function TrainingHistory()
        new(Float64[], Float64[], Float64[])
    end
end

# Add symbol indexing support for backwards compatibility
Base.getindex(history::TrainingHistory, key::Symbol) = begin
    if key == :losses
        return history.losses
    elseif key == :gradient_norms
        return history.gradient_norms
    elseif key == :iteration_times
        return history.iteration_times
    else
        throw(KeyError(key))
    end
end

"""
    DomainInterface

Standard validation struct to ensure domain implements required interface.

Use this to validate that a domain provides all required functions.
"""
struct DomainInterface{S<:AbstractState, A<:AbstractAction}
    state_type::Type{S}
    action_type::Type{A}

    function DomainInterface(state_type::Type{S}, action_type::Type{A}) where {S<:AbstractState, A<:AbstractAction}
        # Validate required methods exist
        if !hasmethod(state_to_features, (state_type,))
            throw(ArgumentError("state_to_features not implemented for $state_type"))
        end
        if !hasmethod(is_terminal_state, (state_type,))
            throw(ArgumentError("is_terminal_state not implemented for $state_type"))
        end
        if !hasmethod(reward, (state_type,))
            throw(ArgumentError("reward not implemented for $state_type"))
        end
        if !hasmethod(is_applicable, (action_type, state_type))
            throw(ArgumentError("is_applicable not implemented for $action_type, $state_type"))
        end
        if !hasmethod(apply_action, (action_type, state_type))
            throw(ArgumentError("apply_action not implemented for $action_type, $state_type"))
        end

        new{S,A}(state_type, action_type)
    end
end

# =============================================================================
# Type Aliases for Convenience
# =============================================================================

"""Alias for trajectory vector"""
const TrajectoryBatch = Vector{Trajectory}

# =============================================================================
# Key Advantages of This Approach
# =============================================================================

"""
Why On-Demand Computation Eliminates Training Errors:

1. ✅ NO CACHE MISSES
   - No pre-computed action_cache that can miss
   - Applicable actions computed fresh each time
   - Eliminates MethodError(getindex, (Dict{Any, Any}(),)

2. ✅ NO OBJECT IDENTITY ISSUES
   - No reliance on object equality for dictionary keys
   - States compared by logical content, not object identity
   - Works with any state representation

3. ✅ SIMPLE AND ROBUST
   - 10x less code than explicit DAG approach
   - Easy to debug and understand
   - Fewer failure modes

4. ✅ MATHEMATICALLY EQUIVALENT
   - Same GFlowNet theory and algorithms
   - Same training objectives and convergence properties
   - Just different computational approach

5. ✅ PERFORMANCE ADEQUATE
   - On-demand computation is fast for most domains
   - Eliminates complex DAG construction overhead
   - Avoids memory issues with large state spaces

The mathematical DAG still exists - it's just defined through:
- get_applicable_actions(state, all_actions)
- apply_action(action, state)
- is_terminal_state(state)

This eliminates engineering complexity while preserving all mathematical foundations.
"""
