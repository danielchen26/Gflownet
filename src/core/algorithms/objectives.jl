# Core GFlowNet Training Objectives
# This file unifies all training objectives for GFlowNet models

# Import external dependencies only - internal GFlowNet types are available in module context
using Random
using Zygote
using LinearAlgebra: norm
using NNlib: softmax

# Core objective functions are defined below

# =============================================================================
# Flow Consistency Objectives (Unified detailed balance and flow matching)
# =============================================================================

"""
    FlowConsistencyMode

Enum for different levels of flow consistency enforcement.
"""
@enum FlowConsistencyMode begin
    EDGE_LEVEL      # Detailed Balance: edge-by-edge consistency  
    STATE_LEVEL     # Flow Matching: state-by-state consistency
    MIXED_LEVEL     # Combination of both approaches
end

"""
    flow_consistency_loss(model::GFlowNetModel, trajectories::Vector{<:Trajectory}; 
                         mode::FlowConsistencyMode=STATE_LEVEL)

Unified flow consistency loss function that can operate at different granularity levels.

This function consolidates what were previously separate "detailed balance" and "flow matching"
objectives, recognizing that they are mathematically related concepts with different granularities.

# Arguments
- `model`: The GFlowNet model
- `trajectories`: Vector of trajectories (for interface consistency)
- `mode`: Level of consistency enforcement

# Modes
- `EDGE_LEVEL`: Enforces F(s) * P_F(s→s') = F(s') * P_B(s'→s) for each edge (former "detailed balance")
- `STATE_LEVEL`: Enforces ∑incoming_flow = ∑outgoing_flow for each state (former "flow matching")
- `MIXED_LEVEL`: Combines both edge and state level constraints
"""
function flow_consistency_loss(model::GFlowNetModel, trajectories::Vector{<:Trajectory}; 
                             mode::FlowConsistencyMode=STATE_LEVEL)
    if mode == EDGE_LEVEL
        return edge_level_consistency_loss(model)
    elseif mode == STATE_LEVEL  
        return state_level_consistency_loss(model)
    elseif mode == MIXED_LEVEL
        edge_loss = edge_level_consistency_loss(model)
        state_loss = state_level_consistency_loss(model)
        return 0.5 * edge_loss + 0.5 * state_loss
    else
        error("Unknown flow consistency mode: $mode")
    end
end

"""
    edge_level_consistency_loss(model::GFlowNetModel)

Enforce flow consistency at the edge level (formerly "detailed balance").

For each edge (s, s'), enforces: F(s) * P_F(s → s') = F(s') * P_B(s' → s)
This is the most fine-grained consistency constraint.
"""
function edge_level_consistency_loss(model::GFlowNetModel)
    if isnothing(model.backward_policy)
        @warn "Edge-level consistency requires backward policy. Falling back to state-level."
        return state_level_consistency_loss(model)
    end
    
    total_loss = 0.0
    
    # For each edge in the graph
    for source in model.dag.states
        for target in get_next_states(model.dag, source)
            # Skip edges to terminal sink
            if target == model.dag.terminal_sink
                continue
            end
            
            # Forward flow: F(s) * P_F(s → s')
            forward_flow = flow(model, source) * forward_transition_prob(model, source, target)
            
            # Backward flow: F(s') * P_B(s' → s)
            backward_flow = flow(model, target) * backward_transition_prob(model, target, source)
            
            # Squared difference (edge-level consistency)
            diff = forward_flow - backward_flow
            total_loss += diff^2
        end
    end
    
    return total_loss
end

"""
    state_level_consistency_loss(model::GFlowNetModel)

Enforce flow consistency at the state level (formerly "flow matching").

For each non-terminal state, enforces: ∑incoming_flow = ∑outgoing_flow
This is an aggregated version of edge-level consistency.
"""
function state_level_consistency_loss(model::GFlowNetModel)
    total_loss = 0.0
    
    # Exclude initial and terminal states
    non_terminal_states = filter(s -> s != model.dag.initial_state && 
                                 s ∉ model.dag.terminal_states && 
                                 s != model.dag.terminal_sink,
                                 model.dag.states)
    
    for state in non_terminal_states
        # Sum of incoming flows
        incoming_flow = sum(edge_flow(model, prev, state) 
                           for prev in get_previous_states(model.dag, state))
        
        # Sum of outgoing flows
        outgoing_flow = sum(edge_flow(model, state, next) 
                           for next in get_next_states(model.dag, state))
        
        # Squared difference (state-level consistency)
        diff = incoming_flow - outgoing_flow
        total_loss += diff^2
    end
    
    return total_loss
end

# =============================================================================
# Trajectory Balance Objectives
# =============================================================================

"""
    trajectory_balance_loss(model::GFlowNetModel, trajectories::Vector{Trajectory})

Compute the Trajectory Balance loss for the GFlowNet.

The trajectory balance objective directly relates the probability of a complete trajectory τ
to the reward of the terminal state:
    P_F(τ) = R(s_τ) / Z
where Z is the partition function, P_F is the product of forward transition probabilities,
and R is the reward function.

This implementation requires all core functions to be properly implemented.
"""
function trajectory_balance_loss(model::GFlowNetModel, trajectories::Vector{Trajectory})
    total_loss = 0.0
    n_trajectories = length(trajectories)

    if n_trajectories == 0
        return 0.0
    end

    for trajectory in trajectories
        # Last state in trajectory (before sink)
        final_state = trajectory.states[end]

        # Product of forward probabilities along the trajectory
        forward_prob_product = 1.0
        for i in 1:(length(trajectory.states)-1)
            source = trajectory.states[i]
            target = trajectory.states[i+1]

            prob = forward_transition_prob(model, source, target)
            forward_prob_product *= prob
        end

        # Compute the reward of the final state
        final_reward = reward(final_state)

        # Compute Z (partition function)
        Z = if isnothing(model.partition_function)
            estimate_partition_function(model)
        else
            model.partition_function
        end

        # Compute the ratio (should be 1 for perfect balance)
        ratio = (Z * forward_prob_product) / final_reward

        # Squared log error (numerically stable)
        log_ratio = log(ratio)
        total_loss += log_ratio^2
    end

    return total_loss / n_trajectories
end

"""
    general_trajectory_balance_loss(model::GFlowNetModel, trajectories::Vector{<:Trajectory})

Compute the General Trajectory Balance loss for the GFlowNet.

The general trajectory balance objective enforces the complete balance equation:
    Z * P_F(τ) = R(s_τ) * P_B(τ)
where P_B(τ) is the product of backward transition probabilities.

This is suitable for problems where states can have multiple parents (non-deterministic backward paths).
"""
function general_trajectory_balance_loss(model::GFlowNetModel, trajectories::Vector{<:Trajectory})
    if isnothing(model.backward_policy)
        error("General trajectory balance requires a backward policy. Use trajectory_balance_loss for simplified version.")
    end
    
    total_loss = 0.0
    n_trajectories = length(trajectories)
    
    for trajectory in trajectories
        # Last state in trajectory (before sink)
        final_state = trajectory.states[end]
        
        # Product of forward probabilities along the trajectory
        forward_prob_product = 1.0
        for i in 1:(length(trajectory.states)-1)
            source = trajectory.states[i]
            target = trajectory.states[i+1]
            prob = forward_transition_prob(model, source, target)
            forward_prob_product *= prob
        end
        
        # Product of backward probabilities along the trajectory
        backward_prob_product = 1.0
        for i in length(trajectory.states):-1:2
            source = trajectory.states[i-1]
            target = trajectory.states[i]
            prob = backward_transition_prob(model, target, source)
            backward_prob_product *= prob
        end
        
        # Compute the reward of the final state
        final_reward = reward(final_state)
        
        # Compute Z (partition function)
        Z = isnothing(model.partition_function) ? 
            estimate_partition_function(model) : model.partition_function
        
        # General trajectory balance: Z * P_F(τ) = R(s_τ) * P_B(τ)
        ratio = (Z * forward_prob_product) / (final_reward * backward_prob_product)
        
        # Squared log error
        log_ratio = log(ratio)
        total_loss += log_ratio^2
    end
    
    return total_loss / n_trajectories
end

# =============================================================================
# Sub-Trajectory Balance Objectives
# =============================================================================

"""
    sub_trajectory_balance_loss(model::GFlowNetModel, trajectories::Vector{<:Trajectory}; 
                               min_length::Int=2, max_length::Union{Int,Nothing}=nothing,
                               n_subtrajectories::Int=5)

Compute the Sub-Trajectory Balance loss for the GFlowNet.

Sub-Trajectory Balance applies the trajectory balance condition to sub-trajectories,
providing better credit assignment for long trajectories by enforcing balance
at multiple scales.

For any sub-trajectory τ_{i:j} = (s_i, s_{i+1}, ..., s_j), the balance condition is:
    F(s_i) * ∏_{t=i}^{j-1} P_F(s_{t+1} | s_t) = F(s_j)

# Arguments
- `model`: The GFlowNet model
- `trajectories`: Vector of complete trajectories
- `min_length`: Minimum length of sub-trajectories to consider
- `max_length`: Maximum length of sub-trajectories (default: no limit)
- `n_subtrajectories`: Number of sub-trajectories to sample per complete trajectory
"""
function sub_trajectory_balance_loss(model::GFlowNetModel, trajectories::Vector{<:Trajectory}; 
                                   min_length::Int=2, max_length::Union{Int,Nothing}=nothing,
                                   n_subtrajectories::Int=5)
    
    total_loss = 0.0
    total_subtrajectories = 0
    
    for trajectory in trajectories
        trajectory_length = length(trajectory.states)
        
        # Skip very short trajectories
        if trajectory_length < min_length + 1  # +1 because we need at least min_length + 1 states
            continue
        end
        
        # Sample sub-trajectories from this trajectory
        for _ in 1:n_subtrajectories
            # Randomly select start and end indices
            start_idx = rand(1:(trajectory_length - min_length))
            max_end = min(trajectory_length, start_idx + (isnothing(max_length) ? trajectory_length : max_length) - 1)
            end_idx = rand((start_idx + min_length - 1):max_end)
            
            # Extract sub-trajectory
            sub_states = trajectory.states[start_idx:end_idx]
            
            # Compute flow at start state
            start_flow = flow(model, sub_states[1])
            
            # Compute flow at end state  
            end_flow = flow(model, sub_states[end])
            
            # Compute product of forward probabilities along sub-trajectory
            forward_prob_product = 1.0
            for i in 1:(length(sub_states)-1)
                source = sub_states[i]
                target = sub_states[i+1]
                prob = forward_transition_prob(model, source, target)
                prob = max(prob, 1e-10)  # Prevent numerical issues
                forward_prob_product *= prob
            end
            
            # Sub-trajectory balance condition: F(s_i) * P_F(τ_{i:j}) = F(s_j)
            if start_flow > 1e-10 && end_flow > 1e-10
                predicted_end_flow = start_flow * forward_prob_product
                ratio = predicted_end_flow / end_flow
                
                # Squared log error
                log_ratio = log(max(ratio, 1e-10))
                total_loss += log_ratio^2
                total_subtrajectories += 1
            end
        end
    end
    
    return total_subtrajectories > 0 ? total_loss / total_subtrajectories : 0.0
end

"""
    hierarchical_sub_trajectory_balance_loss(model::GFlowNetModel, trajectories::Vector{<:Trajectory};
                                           scales::Vector{Int}=[2, 4, 8])

Compute Sub-Trajectory Balance loss at multiple hierarchical scales.

This version computes sub-trajectory balance at different length scales,
providing multi-scale credit assignment.

# Arguments
- `model`: The GFlowNet model
- `trajectories`: Vector of complete trajectories
- `scales`: Vector of sub-trajectory lengths to consider
"""
function hierarchical_sub_trajectory_balance_loss(model::GFlowNetModel, trajectories::Vector{<:Trajectory};
                                                 scales::Vector{Int}=[2, 4, 8])
    
    total_loss = 0.0
    total_weight = 0.0
    
    for scale in scales
        scale_weight = 1.0 / scale  # Weight smaller scales more heavily
        scale_loss = sub_trajectory_balance_loss(model, trajectories; 
                                                min_length=scale, max_length=scale,
                                                n_subtrajectories=max(1, 10 ÷ scale))
        
        total_loss += scale_weight * scale_loss
        total_weight += scale_weight
    end
    
    return total_weight > 0 ? total_loss / total_weight : 0.0
end

"""
    adaptive_sub_trajectory_balance_loss(model::GFlowNetModel, trajectories::Vector{<:Trajectory};
                                        difficulty_threshold::Float64=0.1)

Compute Sub-Trajectory Balance loss with adaptive sub-trajectory selection.

This version focuses on sub-trajectories where the model is performing poorly,
providing targeted credit assignment where it's most needed.

# Arguments
- `model`: The GFlowNet model
- `trajectories`: Vector of complete trajectories  
- `difficulty_threshold`: Threshold for considering a sub-trajectory "difficult"
"""
function adaptive_sub_trajectory_balance_loss(model::GFlowNetModel, trajectories::Vector{<:Trajectory};
                                             difficulty_threshold::Float64=0.1)
    
    total_loss = 0.0
    total_subtrajectories = 0
    
    for trajectory in trajectories
        trajectory_length = length(trajectory.states)
        
        if trajectory_length < 3
            continue
        end
        
        # Evaluate all possible sub-trajectories and select difficult ones
        for start_idx in 1:(trajectory_length-1)
            for end_idx in (start_idx+1):trajectory_length
                sub_states = trajectory.states[start_idx:end_idx]
                
                # Compute flows and forward probabilities
                start_flow = flow(model, sub_states[1])
                end_flow = flow(model, sub_states[end])
                
                forward_prob_product = 1.0
                for i in 1:(length(sub_states)-1)
                    source = sub_states[i]
                    target = sub_states[i+1]
                    prob = forward_transition_prob(model, source, target)
                    forward_prob_product *= prob
                end
                
                if start_flow > 1e-10 && end_flow > 1e-10
                    # Check if this sub-trajectory is "difficult"
                    predicted_end_flow = start_flow * forward_prob_product
                    error = abs(log(predicted_end_flow / end_flow))
                    
                    if error > difficulty_threshold
                        # Include this sub-trajectory in the loss
                        log_ratio = log(max(predicted_end_flow / end_flow, 1e-10))
                        total_loss += log_ratio^2
                        total_subtrajectories += 1
                    end
                end
            end
        end
    end
    
    return total_subtrajectories > 0 ? total_loss / total_subtrajectories : 0.0
end

# =============================================================================
# Gradient Computation Functions
# =============================================================================

"""
    trajectory_balance_loss_grad(model::GFlowNetModel, trajectories::Vector{<:Trajectory})

Compute the gradient of the Trajectory Balance loss for optimization using Zygote.jl.
Following proper Lux.jl patterns with ComponentArrays for parameter handling.
"""
function trajectory_balance_loss_grad(model::GFlowNetModel, trajectories::Vector{<:Trajectory})
    # FIXED: Simplified gradient computation without ComponentArray dependency

    # Define loss function that works directly with model parameters
    function loss_fn(params)
        # Compute forward pass with current parameters
        loss_value = 0.0f0

        for traj in trajectories
            # Compute trajectory balance loss for the entire trajectory
            log_prob_sum = 0.0f0

            # Accumulate log probabilities for all transitions in the trajectory
            for i in 1:(length(traj.states)-1)
                current_state = traj.states[i]
                next_state = traj.states[i+1]

                # Get forward probability using current parameters
                if !isnothing(model.forward_policy)
                    # Extract features for neural network
                    state_features = state_to_features(current_state)

                    # Forward pass through neural network with current parameters
                    logits, _ = model.forward_policy.model(state_features, params.forward, model.states.forward)

                    # Compute transition probability
                    next_states = get_next_states(model.dag, current_state)
                    if !isempty(next_states)
                        next_state_indices = [model.dag.state_to_idx[s] for s in next_states]
                        relevant_logits = logits[next_state_indices]
                        relevant_logits = clamp.(relevant_logits, -20.0f0, 20.0f0)
                        log_probs = logsoftmax(relevant_logits)  # Use logsoftmax for numerical stability

                        # Find probability of the actual transition
                        next_idx = findfirst(s -> s == next_state, next_states)
                        if !isnothing(next_idx)
                            log_prob_sum += log_probs[next_idx]
                        end
                    end
                end
            end

            # Now compute trajectory balance loss for this complete trajectory
            R = reward(traj.states[end])
            R_safe = max(R, 1f-8)  # Ensure positive reward

            # Trajectory balance: log(Z) + sum(log P_F) - log(R) ≈ 0
            # We minimize squared error: (log(Z) + sum(log P_F) - log(R))^2
            log_Z = 0.0f0  # log(1) = 0, will be improved with proper partition function

            trajectory_balance = log_Z + log_prob_sum - log(R_safe)
            loss_value += trajectory_balance^2
        end

        return loss_value / length(trajectories)
    end

    # Compute gradients using Zygote
    gradients = gradient(loss_fn, model.parameters)
    
    if !isnothing(gradients[1])
        # Return the gradients directly (they're already in the correct structure)
        return gradients[1]
    else
        return nothing
    end
end

"""
    general_trajectory_balance_loss_grad(model::GFlowNetModel, trajectories::Vector{<:Trajectory})

Compute the gradient of the General Trajectory Balance loss for optimization.
"""
function general_trajectory_balance_loss_grad(model::GFlowNetModel, trajectories::Vector{<:Trajectory})
    if isnothing(model.backward_policy)
        error("General trajectory balance requires a backward policy.")
    end
    
    # Define the loss function that takes parameters and returns scalar loss
    function loss_fn(ps)
        # Create updated model with new parameters (using intuitive keyword arguments)
        updated_model = GFlowNetModel(
            dag = model.dag,
            forward_policy = model.forward_policy,
            backward_policy = model.backward_policy,
            flow_estimator = model.flow_estimator,
            partition_function = model.partition_function,
            objectives = model.objectives,
            optimizer = model.optimizer,
            parameters = ps,  # Updated parameters
            states = model.states
        )
        return general_trajectory_balance_loss(updated_model, trajectories)
    end
    
    # Use Zygote to compute gradients
    gradients = gradient(loss_fn, model.parameters)
    
    return gradients[1]
end

"""
    sub_trajectory_balance_loss_grad(model::GFlowNetModel, trajectories::Vector{<:Trajectory}; kwargs...)

Compute the gradient of the Sub-Trajectory Balance loss for optimization.
"""
function sub_trajectory_balance_loss_grad(model::GFlowNetModel, trajectories::Vector{<:Trajectory}; kwargs...)
    # Define the loss function that takes parameters and returns scalar loss
    function loss_fn(ps)
        # Create updated model with new parameters (using intuitive keyword arguments)
        updated_model = GFlowNetModel(
            dag = model.dag,
            forward_policy = model.forward_policy,
            backward_policy = model.backward_policy,
            flow_estimator = model.flow_estimator,
            partition_function = model.partition_function,
            objectives = model.objectives,
            optimizer = model.optimizer,
            parameters = ps,  # Updated parameters
            states = model.states
        )
        return sub_trajectory_balance_loss(updated_model, trajectories; kwargs...)
    end
    
    # Use Zygote to compute gradients
    gradients = gradient(loss_fn, model.parameters)
    
    return gradients[1]
end

"""
    flow_consistency_loss_grad(model::GFlowNetModel, trajectories::Vector{<:Trajectory}; 
                              mode::FlowConsistencyMode=STATE_LEVEL)

Compute gradients for the flow consistency loss.
"""
function flow_consistency_loss_grad(model::GFlowNetModel, trajectories::Vector{<:Trajectory}; 
                                   mode::FlowConsistencyMode=STATE_LEVEL)
    # Define the loss function that takes parameters and returns scalar loss
    function loss_fn(ps)
        # Create updated model with new parameters (using intuitive keyword arguments)
        updated_model = GFlowNetModel(
            dag = model.dag,
            forward_policy = model.forward_policy,
            backward_policy = model.backward_policy,
            flow_estimator = model.flow_estimator,
            partition_function = model.partition_function,
            objectives = model.objectives,
            optimizer = model.optimizer,
            parameters = ps,  # Updated parameters
            states = model.states
        )
        return flow_consistency_loss(updated_model, trajectories; mode=mode)
    end
    
    # Compute gradient using Zygote
    gradients = gradient(loss_fn, model.parameters)
    
    return gradients[1]
end

# =============================================================================
# Parameter-variant Functions for Automatic Differentiation
# =============================================================================
# 
# Note: These functions have been removed because they were causing confusion
# and infinite recursion. Gradient computation is now handled directly in the
# *_grad functions above using proper Zygote.jl patterns.

# =============================================================================
# Legacy Compatibility Functions
# =============================================================================

# Note: Former detailed_balance.jl and flow_matching.jl functionality
# is now unified in this file as flow_consistency_loss() with different modes:
# - EDGE_LEVEL: Former detailed balance (edge-by-edge consistency)
# - STATE_LEVEL: Former flow matching (state-by-state consistency)  
# - MIXED_LEVEL: Combination of both approaches

"""
    detailed_balance_loss(model::GFlowNetModel, trajectories::Vector{<:Trajectory})

Legacy function name for edge-level flow consistency.
Use flow_consistency_loss(model, trajectories; mode=EDGE_LEVEL) instead.
"""
function detailed_balance_loss(model::GFlowNetModel, trajectories::Vector{<:Trajectory})
    @warn "detailed_balance_loss is deprecated. Use flow_consistency_loss(model, trajectories; mode=EDGE_LEVEL) instead."
    return flow_consistency_loss(model, trajectories; mode=EDGE_LEVEL)
end

"""
    flow_matching_loss(model::GFlowNetModel, trajectories::Vector{<:Trajectory})

Legacy function name for state-level flow consistency.
Use flow_consistency_loss(model, trajectories; mode=STATE_LEVEL) instead.
"""
function flow_matching_loss(model::GFlowNetModel, trajectories::Vector{<:Trajectory})
    @warn "flow_matching_loss is deprecated. Use flow_consistency_loss(model, trajectories; mode=STATE_LEVEL) instead."
    return flow_consistency_loss(model, trajectories; mode=STATE_LEVEL)
end 