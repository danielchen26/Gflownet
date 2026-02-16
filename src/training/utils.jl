# Training Utilities for GFlowNet
# Helper functions for training, validation, and gradient management

using Statistics
using ComponentArrays

using ..GFlowNet: GFlowNetModel, Trajectory, SamplingConfig
using ..GFlowNet: sample_trajectory, is_terminal_state

# =============================================================================
# Trajectory Utilities
# =============================================================================

"""
    sample_trajectory_batch(model, batch_size; config)

Sample multiple trajectories efficiently.
"""
function sample_trajectory_batch(model::GFlowNetModel, batch_size::Int;
                                config::SamplingConfig = SamplingConfig())
    return [sample_trajectory(model; config = config) for _ in 1:batch_size]
end

"""
    is_valid_trajectory(trajectory)

Check if trajectory is valid.
"""
function is_valid_trajectory(trajectory::Trajectory)
    return !isempty(trajectory.states) &&
           length(trajectory.states) == length(trajectory.actions) + 1 &&
           is_terminal_state(trajectory.states[end])
end

# =============================================================================
# Gradient Utilities
# =============================================================================

"""
    any_invalid(gradients)

Check if gradients contain invalid values.
"""
function any_invalid(gradients)
    if gradients isa AbstractArray
        return any(isnan, gradients) || any(isinf, gradients)
    end
    for grad in values(gradients)
        if grad isa AbstractArray
            if any(isnan, grad) || any(isinf, grad)
                return true
            end
        end
    end
    return false
end

"""
    compute_gradient_norm(gradients)

Compute L2 norm of gradients with proper ComponentArray support.
"""
function compute_gradient_norm(gradients)
    norm_squared = 0.0

    function add_gradient_contribution!(obj)
        if obj isa AbstractArray && !isempty(obj)
            norm_squared += sum(abs2, obj; init=0.0)
        elseif obj isa NamedTuple
            for value in values(obj)
                add_gradient_contribution!(value)
            end
        elseif hasproperty(obj, :axes) && hasmethod(values, (typeof(obj),))
            # ComponentArray or similar structure
            try
                for value in values(obj)
                    add_gradient_contribution!(value)
                end
            catch
                # Fallback: try to access as NamedTuple-like
                try
                    for key in keys(obj)
                        add_gradient_contribution!(getproperty(obj, key))
                    end
                catch
                    # Last resort: treat as array if possible
                    if obj isa AbstractArray && !isempty(obj)
                        norm_squared += sum(abs2, obj; init=0.0)
                    end
                end
            end
        end
    end

    try
        add_gradient_contribution!(gradients)
    catch e
        @warn "Error computing gradient norm: $e"
        return 0.0
    end

    return sqrt(max(norm_squared, 0.0))
end