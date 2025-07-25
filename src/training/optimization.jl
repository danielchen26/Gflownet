# Optimization and Gradient Computation for GFlowNet Training
# This file contains functions for computing gradients and updating model parameters

using ..GFlowNet: GFlowNetModel, Trajectory
using Zygote
using Optimisers
using ComponentArrays
using Lux
using NNlib: logsumexp

# Import objective functions for gradient computation
using ..GFlowNet: trajectory_balance_loss, trajectory_balance_loss_grad, general_trajectory_balance_loss_grad
using ..GFlowNet: sub_trajectory_balance_loss_grad, flow_consistency_loss_grad
using ..GFlowNet: FlowConsistencyMode, STATE_LEVEL

"""
    update_model_parameters!(model::GFlowNetModel, config::TrainingConfig, trajectories::Vector{<:Trajectory})

Update model parameters based on gradients computed for the given trajectories.

# Arguments
- `model`: GFlowNet model to update
- `config`: Training configuration
- `trajectories`: Batch of trajectories for gradient computation
"""
function update_model_parameters!(model::GFlowNetModel, config::TrainingConfig, trajectories::Vector{<:Trajectory})
    # Compute gradients based on objective
    if config.objective == TRAJECTORY_BALANCE
        grad = trajectory_balance_loss_grad(model, trajectories)
    elseif config.objective == GENERAL_TRAJECTORY_BALANCE
        grad = general_trajectory_balance_loss_grad(model, trajectories)
    elseif config.objective in [SUB_TRAJECTORY_BALANCE, HIERARCHICAL_SUB_TB, ADAPTIVE_SUB_TB]
        grad = sub_trajectory_balance_loss_grad(model, trajectories)
    elseif config.objective == FLOW_CONSISTENCY
        mode = get(config.sub_trajectory_config, :flow_consistency_mode, STATE_LEVEL)
        grad = flow_consistency_loss_grad(model, trajectories; mode=mode)
    else
        error("Unknown training objective: $(config.objective)")
    end
    
    # Apply gradients to update parameters with gradient clipping
    max_grad_norm = get(config.sub_trajectory_config, :max_grad_norm, 1.0)
    if max_grad_norm > 0.0 && !isnothing(grad)
        grad_norm = clip_gradients!(grad, max_grad_norm)
        if grad_norm > max_grad_norm
            @debug "Clipped gradients: norm $grad_norm → $max_grad_norm"
        end
    end
    
    apply_gradients!(model, grad, config.learning_rate)
end

"""
    apply_gradients!(model::GFlowNetModel, gradients, learning_rate::Float64)

Apply gradients to update model parameters.

# Arguments
- `model`: GFlowNet model to update
- `gradients`: Computed gradients
- `learning_rate`: Learning rate for parameter updates
"""
function apply_gradients!(model::GFlowNetModel, gradients, learning_rate::Float64)
    # Handle case where gradients is Nothing
    if isnothing(gradients)
        @warn "Gradients are Nothing, skipping parameter update"
        return
    end
    
    # Ensure parameters are ComponentArray for consistent gradient operations
    if !(model.parameters isa ComponentArray)
        @warn "Converting model parameters to ComponentArray for gradient compatibility"
        model.parameters = to_component_array(model.parameters)
    end
    
    # Create new optimizer and parameter tuples
    new_optimizer = model.optimizer
    new_parameters = model.parameters
    
    # Update forward policy
    if !isnothing(model.forward_policy) && haskey(gradients, :forward) && !isnothing(gradients.forward)
        result = Optimisers.update(model.optimizer.forward, model.parameters.forward, gradients.forward)
        
        if !isnothing(result) && length(result) == 2
            new_optimizer = merge(new_optimizer, (forward = result[1],))
            new_parameters = merge(new_parameters, (forward = result[2],))
        end
    end
    
    # Update backward policy if it exists
    if !isnothing(model.backward_policy) && haskey(gradients, :backward) && !isnothing(gradients.backward)
        result = Optimisers.update(model.optimizer.backward, model.parameters.backward, gradients.backward)
        
        if !isnothing(result) && length(result) == 2
            new_optimizer = merge(new_optimizer, (backward = result[1],))
            new_parameters = merge(new_parameters, (backward = result[2],))
        end
    end
    
    # Update flow estimator if it exists
    if !isnothing(model.flow_estimator) && haskey(gradients, :flow) && !isnothing(gradients.flow)
        result = Optimisers.update(model.optimizer.flow, model.parameters.flow, gradients.flow)
        
        if !isnothing(result) && length(result) == 2
            new_optimizer = merge(new_optimizer, (flow = result[1],))
            new_parameters = merge(new_parameters, (flow = result[2],))
        end
    end
    
    # Set the new optimizer and parameters
    model.optimizer = new_optimizer
    model.parameters = new_parameters
    
    # Clear flow cache when parameters change
    clear_flow_cache!()
end

"""
    compute_gradient_norm(gradients)

Compute the L2 norm of gradients for monitoring training stability.

# Arguments
- `gradients`: Computed gradients (typically a NamedTuple)

# Returns
- L2 norm of all gradients combined
"""
function compute_gradient_norm(gradients)
    total_norm_squared = 0.0
    
    for (name, grad) in pairs(gradients)
        if !isnothing(grad)
            # Handle different gradient structures
            if grad isa AbstractArray
                total_norm_squared += sum(abs2, grad)
            elseif grad isa NamedTuple
                # Recursively compute norm for nested structures
                total_norm_squared += compute_gradient_norm(grad)
            else
                # For scalar gradients
                total_norm_squared += abs2(grad)
            end
        end
    end
    
    return sqrt(total_norm_squared)
end

"""
    clip_gradients!(gradients, max_norm::Float64)

Clip gradients to prevent exploding gradients during training.
Handles nested NamedTuple structures recursively.

# Arguments
- `gradients`: Gradients to clip (modified in-place)
- `max_norm`: Maximum allowed gradient norm

# Returns
- Actual gradient norm before clipping
"""
function clip_gradients!(gradients, max_norm::Float64)
    current_norm = compute_gradient_norm(gradients)
    
    if current_norm > max_norm
        scale_factor = max_norm / current_norm
        
        # Recursively scale all gradients including nested structures
        _clip_gradients_recursive!(gradients, scale_factor)
    end
    
    return current_norm
end

"""
    _clip_gradients_recursive!(gradients, scale_factor::Float64)

Helper function to recursively clip gradients in nested structures.
"""
function _clip_gradients_recursive!(gradients, scale_factor::Float64)
    for (name, grad) in pairs(gradients)
        if !isnothing(grad)
            if grad isa AbstractArray
                # Scale array gradients in-place
                grad .*= scale_factor
            elseif grad isa NamedTuple
                # Recursively handle nested structures
                _clip_gradients_recursive!(grad, scale_factor)
            elseif grad isa Number
                # Handle scalar gradients - ComponentArrays should handle this automatically
                # Skip scalar clipping as ComponentArrays manages this internally
                continue
            end
        end
    end
end

"""
    setup_optimizers(model::GFlowNetModel, learning_rate::Float64=0.001; optimizer_type=:adam)

Setup optimizers for all components of a GFlowNet model.

# Arguments
- `model`: GFlowNet model
- `learning_rate`: Learning rate for optimizers
- `optimizer_type`: Type of optimizer (:adam, :sgd, :rmsprop)

# Returns
- NamedTuple of optimizers for each component
"""
function setup_optimizers(model::GFlowNetModel, learning_rate::Float64=0.001; optimizer_type=:adam)
    # Choose optimizer
    if optimizer_type == :adam
        opt_fn = () -> Optimisers.Adam(learning_rate)
    elseif optimizer_type == :sgd
        opt_fn = () -> Optimisers.Descent(learning_rate)
    elseif optimizer_type == :rmsprop
        opt_fn = () -> Optimisers.RMSProp(learning_rate)
    else
        error("Unknown optimizer type: $optimizer_type")
    end
    
    # Setup optimizers for each component
    optimizers = []
    
    # Forward policy optimizer
    if !isnothing(model.forward_policy)
        push!(optimizers, :forward => opt_fn())
    end
    
    # Backward policy optimizer
    if !isnothing(model.backward_policy)
        push!(optimizers, :backward => opt_fn())
    end
    
    # Flow estimator optimizer
    if !isnothing(model.flow_estimator)
        push!(optimizers, :flow => opt_fn())
    end
    
    return NamedTuple(optimizers)
end

# =============================================================================
# Legacy Functions for Backward Compatibility
# =============================================================================

"""
    compute_loss_and_grad(model::GFlowNetModel, trajectories::Vector{<:Trajectory})

Legacy function for backward compatibility with examples.
Use modern train_gflownet() interface for new code.
"""
function compute_loss_and_grad(model::GFlowNetModel, trajectories::Vector{<:Trajectory})
    # SYSTEMATIC FIX: Pre-compute all state indices and transitions outside the differentiable function
    # This avoids Zygote differentiation issues with custom struct operations

    # Comprehensive trajectory validation to prevent type issues
    if !isa(trajectories, Vector{<:Trajectory})
        error("trajectories must be Vector{<:Trajectory}, got $(typeof(trajectories))")
    end
    
    for (i, traj) in enumerate(trajectories)
        if !isa(traj, Trajectory)
            error("Trajectory $i is not a Trajectory, got $(typeof(traj))")
        end
        if isempty(traj.states)
            error("Trajectory $i has empty states")
        end
    end

    # Pre-compute trajectory data to avoid struct operations in differentiable function
    trajectory_data = Vector{Vector{Tuple{Vector{Float32}, Vector{Int}, Int}}}()

    for traj in trajectories
        traj_data = Vector{Tuple{Vector{Float32}, Vector{Int}, Int}}()
        for i in 1:(length(traj.states)-1)
            current_state = traj.states[i]
            next_state = traj.states[i+1]

            # Pre-compute state features (these are just Float32 arrays)
            features = state_to_features(current_state)

            # Pre-compute next state indices (avoid struct operations in loss function)
            next_states = get_next_states(model.dag, current_state)

            if !isempty(next_states) && next_state in next_states
                next_state_indices = [model.dag.state_to_idx[s] for s in next_states]
                target_idx = findfirst(s -> s == next_state, next_states)

                if !isnothing(target_idx)
                    push!(traj_data, (features, next_state_indices, target_idx))
                end
            end
        end
        push!(trajectory_data, traj_data)
    end

    total_transitions = sum(length(td) for td in trajectory_data)

    # CRITICAL: Check if we have any valid transitions for gradient computation
    if total_transitions == 0
        @warn "No valid transitions found for gradient computation! Returning zero loss and nothing gradients."
        return 0.0f0, nothing
    end

    # Differentiable loss function that only operates on numeric data
    function loss_fn(params)
        total_loss = 0.0f0
        valid_trajectories = 0

        for traj_data in trajectory_data
            if isempty(traj_data)
                continue
            end

            log_prob = 0.0f0

            for (features, next_state_indices, target_idx) in traj_data
                # Reshape features for neural network
                features_matrix = reshape(features, :, 1)

                # Forward pass through neural network
                logits, _ = model.forward_policy.model(features_matrix, params.forward, model.states.forward)
                logits = vec(logits)

                # Extract relevant logits for possible next states
                relevant_logits = logits[next_state_indices]

                # Compute log probabilities
                log_probs = relevant_logits .- logsumexp(relevant_logits)

                # Add log probability of the actual transition
                log_prob += log_probs[target_idx]
            end

            # Simple loss: negative log probability
            total_loss += -log_prob
            valid_trajectories += 1
        end

        return valid_trajectories > 0 ? total_loss / valid_trajectories : 0.0f0
    end

    # Compute gradients
    loss, grad = Zygote.withgradient(loss_fn, model.parameters)
    return loss, grad[1]
end

"""
    apply_optimizer!(model::GFlowNetModel, grad; max_grad_norm::Float64=1.0)

Legacy function for backward compatibility with examples.
PREFERRED: Use modern train_gflownet() interface for new code.

This function is used by:
- examples/grid_world/grid_world.jl
- examples/feature_acquisition/ (legacy versions)

# Arguments
- `model`: GFlowNet model to update
- `grad`: Gradients to apply (should be NamedTuple with forward/backward/flow components)
- `max_grad_norm`: Maximum gradient norm for clipping
"""
function apply_optimizer!(model::GFlowNetModel, grad; max_grad_norm::Float64=1.0)
    # Handle case where gradients is Nothing
    if isnothing(grad)
        @warn "Gradients are Nothing, skipping parameter update"
        return model
    end



    # Add gradient clipping integration with improved nested structure support
    # Clip gradients to prevent exploding gradients
    if max_grad_norm > 0.0
        grad_norm = clip_gradients!(grad, max_grad_norm)
        if grad_norm > max_grad_norm
            @debug "Clipped gradients: norm $grad_norm → $max_grad_norm"
        end
    end

    # Create new optimizer and parameter structures
    new_optimizer = model.optimizer
    new_parameters = model.parameters

    # Update forward policy
    if !isnothing(model.forward_policy) && haskey(grad, :forward) && !isnothing(grad.forward)
        result = Optimisers.update(model.optimizer.forward, model.parameters.forward, grad.forward)

        if !isnothing(result) && length(result) == 2
            # Handle both NamedTuple and ComponentArray cases
            if isa(new_optimizer, NamedTuple)
                new_optimizer = merge(new_optimizer, (forward = result[1],))
            else
                new_optimizer = (forward = result[1], flow = new_optimizer.flow)
            end

            if isa(new_parameters, ComponentArray)
                # For ComponentArray, update the forward component directly
                new_parameters = ComponentArray(
                    forward = result[2],
                    flow = new_parameters.flow
                )
            else
                new_parameters = merge(new_parameters, (forward = result[2],))
            end
        end
    end
    
    # Update backward policy if it exists
    if !isnothing(model.backward_policy) && haskey(grad, :backward) && !isnothing(grad.backward)
        result = Optimisers.update(model.optimizer.backward, model.parameters.backward, grad.backward)
        
        if !isnothing(result) && length(result) == 2
            new_optimizer = merge(new_optimizer, (backward = result[1],))
            new_parameters = merge(new_parameters, (backward = result[2],))
        end
    end
    
    # Update flow estimator if it exists
    if !isnothing(model.flow_estimator) && haskey(grad, :flow) && !isnothing(grad.flow)
        result = Optimisers.update(model.optimizer.flow, model.parameters.flow, grad.flow)

        if !isnothing(result) && length(result) == 2
            # Handle both NamedTuple and ComponentArray cases
            if isa(new_optimizer, NamedTuple)
                new_optimizer = merge(new_optimizer, (flow = result[1],))
            else
                new_optimizer = (forward = new_optimizer.forward, flow = result[1])
            end

            if isa(new_parameters, ComponentArray)
                # For ComponentArray, update the flow component directly
                new_parameters = ComponentArray(
                    forward = new_parameters.forward,
                    flow = result[2]
                )
            else
                new_parameters = merge(new_parameters, (flow = result[2],))
            end
        end
    end
    
    # Set the new optimizer and parameters
    model.optimizer = new_optimizer
    model.parameters = new_parameters
    
    # OPTIMIZED: Clear flow cache when parameters change
    clear_flow_cache!()
    
    return model
end

"""
    create_adam_optimizer(learning_rate::Float64=0.001)

Create an Adam optimizer with the specified learning rate.
Legacy function for examples.
"""
function create_adam_optimizer(learning_rate::Float64=0.001)
    return Optimisers.Adam(learning_rate)
end

"""
    update_parameters_with_gradients!(parameters, gradients, optimizer_state, learning_rate::Float64)

Update parameters using gradients and optimizer state.
Legacy function for examples.
"""
function update_parameters_with_gradients!(parameters, gradients, optimizer_state, learning_rate::Float64)
    result = Optimisers.update(optimizer_state, parameters, gradients)
    
    if !isnothing(result) && length(result) == 2
        return result[1], result[2]  # new_optimizer_state, new_parameters
    else
        return optimizer_state, parameters  # No update
    end
end 