# Base Policy Definitions and Shared Functionality
# This file contains the core abstractions for all GFlowNet policies

using ..GFlowNet: AbstractState
using Random
using Distributions: Categorical
using Lux
using Optimisers

# =============================================================================
# Core Policy Interface
# =============================================================================

"""
    state_to_features(state::AbstractState)

Convert a state to a feature vector suitable for neural network input.
This function should be implemented by domain-specific state types.

# Arguments
- `state`: The state to convert to features

# Returns
- Feature vector (typically Vector{Float32})

# Example
```julia
function state_to_features(state::MoleculeState)
    features = Float32[
        length(state.data.atoms),
        length(state.data.bonds),
        state.complete ? 1.0 : 0.0
    ]
    return features
end
```
"""
function state_to_features(state::AbstractState)
    error("state_to_features not implemented for $(typeof(state))")
end

# =============================================================================
# Common Utility Functions for Policies
# =============================================================================

"""
    normalize_probabilities(logits::AbstractVector)

Convert logits to normalized probabilities using softmax with numerical stability.

# Arguments
- `logits`: Raw logits from neural network

# Returns
- Normalized probability vector
"""
function normalize_probabilities(logits::AbstractVector)
    # Numerical stability: subtract maximum before softmax
    max_logit = maximum(logits)
    exp_logits = exp.(logits .- max_logit)
    return exp_logits ./ sum(exp_logits)
end

"""
    sample_from_probabilities(probs::AbstractVector, rng=nothing)

Sample an index from a probability distribution.

# Arguments
- `probs`: Probability vector (should sum to 1)
- `rng`: Random number generator (optional)

# Returns
- Sampled index
"""
function sample_from_probabilities(probs::AbstractVector, rng=nothing)
    if isnothing(rng)
        rng = Random.default_rng()
    end
    
    # Use categorical distribution for sampling
    return rand(rng, Categorical(probs))
end

"""
    clamp_probabilities(probs::AbstractVector, min_prob::Float64=1e-10)

Ensure all probabilities are above a minimum threshold to prevent numerical issues.

# Arguments
- `probs`: Probability vector
- `min_prob`: Minimum allowed probability

# Returns
- Clamped and renormalized probability vector
"""
function clamp_probabilities(probs::AbstractVector, min_prob::Float64=1e-10)
    clamped = max.(probs, min_prob)
    return clamped ./ sum(clamped)
end

"""
    validate_policy_output(output, expected_size::Int)

Validate that policy neural network output has the expected dimensions.

# Arguments
- `output`: Output from policy network
- `expected_size`: Expected output dimension

# Throws
- `DimensionMismatch` if output size doesn't match expected
"""
function validate_policy_output(output, expected_size::Int)
    actual_size = length(output)
    if actual_size != expected_size
        throw(DimensionMismatch("Policy output size $actual_size doesn't match expected $expected_size"))
    end
end

# =============================================================================
# Policy Parameter Management
# =============================================================================

"""
    initialize_policy_parameters(model, rng=nothing)

Initialize parameters for a policy neural network using Lux conventions.

# Arguments
- `model`: Lux neural network model
- `rng`: Random number generator (optional)

# Returns
- Tuple of (parameters, state) for the model
"""
function initialize_policy_parameters(model, rng=nothing)
    if isnothing(rng)
        rng = Random.default_rng()
    end
    
    return Lux.setup(rng, model)
end

"""
    update_policy_parameters(old_params, gradients, optimizer)

Update policy parameters using gradients and optimizer.

# Arguments
- `old_params`: Current parameters
- `gradients`: Computed gradients
- `optimizer`: Optimizer state

# Returns
- Tuple of (new_optimizer, new_parameters)
"""
function update_policy_parameters(old_params, gradients, optimizer)
    return Optimisers.update(optimizer, old_params, gradients)
end

# =============================================================================
# Error Handling and Validation
# =============================================================================

"""
    PolicyError

Exception type for policy-related errors.
"""
struct PolicyError <: Exception
    message::String
end

"""
    validate_state_for_policy(state::AbstractState, policy_type::String)

Validate that a state is suitable for use with a particular policy type.

# Arguments
- `state`: State to validate
- `policy_type`: Name of policy type for error messages

# Throws
- `PolicyError` if state is invalid
"""
function validate_state_for_policy(state::AbstractState, policy_type::String)
    # Basic validation - can be extended by specific policy types
    if isnothing(state)
        throw(PolicyError("Cannot use nothing state with $policy_type"))
    end
    
    # Check if state_to_features is implemented
    try
        features = state_to_features(state)
        if isempty(features)
            throw(PolicyError("state_to_features returned empty vector for $policy_type"))
        end
    catch MethodError
        throw(PolicyError("state_to_features not implemented for $(typeof(state)) used with $policy_type"))
    end
end

# =============================================================================
# Performance Monitoring
# =============================================================================

"""
    PolicyMetrics

Structure to track policy performance metrics during training.
"""
mutable struct PolicyMetrics
    n_forward_calls::Int
    n_backward_calls::Int
    n_flow_calls::Int
    total_forward_time::Float64
    total_backward_time::Float64
    total_flow_time::Float64
    last_reset_time::Float64
end

"""
    PolicyMetrics()

Create a new policy metrics tracker.
"""
function PolicyMetrics()
    current_time = time()
    return PolicyMetrics(0, 0, 0, 0.0, 0.0, 0.0, current_time)
end

"""
    reset_policy_metrics!(metrics::PolicyMetrics)

Reset all policy metrics to zero.
"""
function reset_policy_metrics!(metrics::PolicyMetrics)
    metrics.n_forward_calls = 0
    metrics.n_backward_calls = 0
    metrics.n_flow_calls = 0
    metrics.total_forward_time = 0.0
    metrics.total_backward_time = 0.0
    metrics.total_flow_time = 0.0
    metrics.last_reset_time = time()
end

"""
    get_policy_statistics(metrics::PolicyMetrics)

Get summary statistics for policy performance.

# Returns
- Dictionary with performance statistics
"""
function get_policy_statistics(metrics::PolicyMetrics)
    elapsed_time = time() - metrics.last_reset_time
    
    return Dict(
        :total_calls => metrics.n_forward_calls + metrics.n_backward_calls + metrics.n_flow_calls,
        :forward_calls => metrics.n_forward_calls,
        :backward_calls => metrics.n_backward_calls,
        :flow_calls => metrics.n_flow_calls,
        :avg_forward_time => metrics.n_forward_calls > 0 ? metrics.total_forward_time / metrics.n_forward_calls : 0.0,
        :avg_backward_time => metrics.n_backward_calls > 0 ? metrics.total_backward_time / metrics.n_backward_calls : 0.0,
        :avg_flow_time => metrics.n_flow_calls > 0 ? metrics.total_flow_time / metrics.n_flow_calls : 0.0,
        :elapsed_time => elapsed_time,
        :calls_per_second => elapsed_time > 0 ? (metrics.n_forward_calls + metrics.n_backward_calls + metrics.n_flow_calls) / elapsed_time : 0.0
    )
end 