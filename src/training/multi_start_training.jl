# Training integration for Multi-Start GFlowNets
# Handles loss computation with per-initial-state partition functions

using Zygote
using Statistics
using Optimisers
using ..GFlowNet: TrainingConfig, TrainingHistory, TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING
using ..GFlowNet: compute_gradient_norm, any_invalid, clear_flow_cache!
using ..GFlowNet: forward_transition_probability, backward_transition_probability
using ..GFlowNet: flow, flow_estimate, forward_action_probabilities
using ..GFlowNet: get_applicable_actions, is_valid_trajectory, is_terminal_state

# =============================================================================
# Loss Computation for Multi-Start Models
# =============================================================================

"""
    compute_trajectory_loss_multi_start(model, trajectories_with_idx, params, config)

Compute loss for trajectories from multi-start model.

Each trajectory is paired with its initial state index to use the correct Z.
"""
function compute_trajectory_loss_multi_start(
    model::MultiStartGFlowNetModel,
    trajectories_with_idx::Vector{Tuple{Trajectory, Int}},
    params,
    config::TrainingConfig
)
    if config.objective == TRAJECTORY_BALANCE
        # Filter valid trajectories
        valid_data = Zygote.@ignore begin
            [(traj, idx) for (traj, idx) in trajectories_with_idx if is_valid_trajectory(traj)]
        end
        
        if isempty(valid_data)
            return 0.0
        end
        
        # Compute losses with correct log Z for each trajectory
        losses = [compute_single_trajectory_loss_multi_start(model, traj, idx, params) 
                 for (traj, idx) in valid_data]
        
        # Filter out infinite losses
        finite_losses = filter(!isinf, losses)
        
        if isempty(finite_losses)
            return Inf
        end
        
        return mean(finite_losses)
        
    elseif config.objective == DETAILED_BALANCE
        # Extract state pairs from trajectories
        state_pairs = Tuple{AbstractState, AbstractState}[]
        
        for (traj, _) in trajectories_with_idx
            if !is_valid_trajectory(traj)
                continue
            end
            
            for i in 1:(length(traj.states)-1)
                push!(state_pairs, (traj.states[i], traj.states[i+1]))
            end
        end
        
        if isempty(state_pairs)
            return 0.0
        end
        
        # Detailed balance loss doesn't depend on initial state
        return compute_detailed_balance_loss_batch(model, state_pairs, params)
        
    elseif config.objective == FLOW_MATCHING
        # Extract non-terminal states
        states = AbstractState[]
        
        for (traj, _) in trajectories_with_idx
            if !is_valid_trajectory(traj)
                continue
            end
            
            for state in traj.states[1:end-1]
                if !is_terminal_state(state)
                    push!(states, state)
                end
            end
        end
        
        # Remove duplicates
        states = unique(states)
        
        if isempty(states)
            return 0.0
        end
        
        # Flow matching loss doesn't depend on initial state
        return compute_flow_matching_loss_batch(model, states, params)
        
    else
        throw(ArgumentError("Unsupported objective: $(config.objective)"))
    end
end

"""
    compute_single_trajectory_loss_multi_start(model, trajectory, initial_idx, params)

Compute trajectory balance loss using the correct log Z for the initial state.
"""
function compute_single_trajectory_loss_multi_start(
    model::MultiStartGFlowNetModel,
    trajectory::Trajectory,
    initial_idx::Int,
    params
)
    # Compute log probability of trajectory
    log_prob_sum = 0.0
    
    for i in 1:(length(trajectory.states)-1)
        state = trajectory.states[i]
        action = trajectory.actions[i]
        
        # Get state features
        features = state_to_features(state)
        
        # Compute forward logits
        logits_vec, _ = model.forward_policy.model(features, params.forward, model.states.forward)
        
        # Get applicable actions
        applicable_actions = Zygote.@ignore get_applicable_actions(state, model.all_actions)
        
        if isempty(applicable_actions)
            return Inf
        end
        
        # Find indices
        action_idx = Zygote.@ignore findfirst(a -> a == action, model.all_actions)
        applicable_indices = Zygote.@ignore [i for (i, a) in enumerate(model.all_actions) if a in applicable_actions]
        
        if isnothing(action_idx) || !(action_idx in applicable_indices)
            return Inf
        end
        
        # Compute log probability
        applicable_logits = logits_vec[applicable_indices]
        log_probs = applicable_logits .- logsumexp(applicable_logits)
        
        action_pos = Zygote.@ignore findfirst(==(action_idx), applicable_indices)
        if isnothing(action_pos)
            return Inf
        end
        
        log_prob_sum += log_probs[action_pos]
    end
    
    # Get terminal reward
    terminal_state = trajectory.states[end]
    terminal_reward = Zygote.@ignore reward(terminal_state)
    
    if terminal_reward <= 0
        terminal_reward = 1e-8
    end
    
    # Use the correct log Z for this trajectory's initial state
    log_Z = params.log_Z[initial_idx]
    
    # Trajectory balance: (log Z + log P_F(τ) - log R)²
    trajectory_balance_error = log_Z + log_prob_sum - log(terminal_reward)
    
    return trajectory_balance_error^2
end

# =============================================================================
# Training Loop for Multi-Start Models
# =============================================================================

"""
    train_gflownet(model::MultiStartGFlowNetModel, config::TrainingConfig; kwargs...)

Train multi-start GFlowNet with per-initial-state partition functions.
"""
function train_gflownet(
    model::MultiStartGFlowNetModel,
    config::TrainingConfig;
    verbose::Bool = false,
    callback = nothing
)
    history = GFlowNet.TrainingHistory()
    
    # Track per-initial-state statistics
    initial_state_counts = zeros(Int, length(model.initial_states))
    initial_state_rewards = [Float64[] for _ in 1:length(model.initial_states)]
    
    if verbose
        println("🚀 Starting Multi-Start GFlowNet training...")
        println("   Configuration:")
        println("     - Objective: $(config.objective)")
        println("     - Initial states: $(length(model.initial_states))")
        println("     - Iterations: $(config.n_iterations)")
        println("     - Batch size: $(config.batch_size)")
    end
    
    for iteration in 1:config.n_iterations
        start_time = time()
        
        try
            # Sample trajectories with initial state tracking
            trajectories_with_idx = [sample_trajectory(model) for _ in 1:config.batch_size]
            
            # Update statistics
            for (traj, idx) in trajectories_with_idx
                initial_state_counts[idx] += 1
                if is_valid_trajectory(traj)
                    terminal_reward = reward(traj.states[end])
                    push!(initial_state_rewards[idx], terminal_reward)
                end
            end
            
            # Compute loss and gradients
            loss_val, gradient_norm = train_step_multi_start!(model, trajectories_with_idx, config)
            
            # Record metrics
            push!(history.losses, loss_val)
            push!(history.gradient_norms, gradient_norm)
            push!(history.iteration_times, time() - start_time)
            
            # Verbose output
            if verbose && (iteration % config.validation_frequency == 0)
                println("\n   Iteration $iteration:")
                println("     - Loss: $(round(loss_val, digits=4))")
                println("     - Gradient norm: $(round(gradient_norm, digits=4))")
                
                # Show initial state distribution
                probs = get_initial_state_distribution(model)
                println("     - Initial state distribution:")
                for (i, p) in enumerate(probs)
                    count = initial_state_counts[i]
                    avg_reward = isempty(initial_state_rewards[i]) ? 0.0 : mean(initial_state_rewards[i])
                    println("       State $i: P=$(round(p, digits=3)), Count=$count, Avg R=$(round(avg_reward, digits=3))")
                end
            end
            
            # Callback
            if !isnothing(callback)
                callback(model, history, iteration)
            end
            
        catch e
            push!(history.losses, NaN)
            push!(history.gradient_norms, NaN)
            push!(history.iteration_times, time() - start_time)
            
            if verbose
                println("   ⚠️  Training error at iteration $iteration: $e")
            end
        end
    end
    
    if verbose
        println("\n   ✅ Training completed!")
        println("     - Final loss: $(round(history.losses[end], digits=4))")
        println("     - Total time: $(round(sum(history.iteration_times), digits=1))s")
        
        # Final initial state distribution
        probs = get_initial_state_distribution(model)
        println("     - Final initial state distribution:")
        for (i, p) in enumerate(probs)
            println("       State $i: P=$(round(p, digits=3))")
        end
    end
    
    return history
end

"""
    train_step_multi_start!(model, trajectories_with_idx, config)

Perform single training step for multi-start model.
"""
function train_step_multi_start!(
    model::MultiStartGFlowNetModel,
    trajectories_with_idx::Vector{Tuple{Trajectory, Int}},
    config::TrainingConfig
)
    # Define loss function
    loss_function = ps -> begin
        Zygote.@ignore clear_flow_cache!()
        compute_trajectory_loss_multi_start(model, trajectories_with_idx, ps, config)
    end
    
    # Compute gradients
    loss_val, grads = Zygote.withgradient(loss_function, model.parameters)
    
    # Check for valid gradients
    if grads[1] === nothing || any_invalid(grads[1])
        return Inf, 0.0
    end
    
    # Compute gradient norm
    gradient_norm = compute_gradient_norm(grads[1])
    
    # Update parameters
    optimizer_state, parameters = Optimisers.update(model.optimizer, model.parameters, grads[1])
    
    # Update model
    model.optimizer = optimizer_state
    model.parameters = parameters
    
    # Synchronize log partition functions
    if haskey(parameters, :log_Z)
        model.log_partition_functions = parameters.log_Z
    end
    
    return loss_val, gradient_norm
end

# =============================================================================
# Helper Functions
# =============================================================================

# Use logsumexp from interface.jl instead of redefining
using ..GFlowNet: logsumexp

# Batch loss functions for multi-start models
function compute_detailed_balance_loss_batch(model::MultiStartGFlowNetModel, state_pairs, params)
    if isempty(state_pairs)
        return 0.0
    end
    
    total_loss = 0.0
    valid_pairs = 0
    
    for (source, target) in state_pairs
        # Check if transition is valid
        applicable_actions = get_applicable_actions(source, model.all_actions)
        can_transition = false
        for action in applicable_actions
            if apply_action(action, source) == target
                can_transition = true
                break
            end
        end
        
        if !can_transition || is_terminal_state(source)
            continue
        end
        
        # Compute forward probability
        forward_prob = forward_transition_probability(model, source, target)
        
        # Compute backward probability
        backward_prob = if isnothing(model.backward_policy)
            1.0
        else
            backward_transition_probability(model, target, source)
        end
        
        # Compute flows
        source_flow = Zygote.@ignore flow(model, source)
        target_flow = Zygote.@ignore flow(model, target)
        
        # Detailed balance loss
        left_side = log(max(forward_prob, 1e-8)) + log(max(source_flow, 1e-8))
        right_side = log(max(backward_prob, 1e-8)) + log(max(target_flow, 1e-8))
        
        loss = (left_side - right_side)^2
        total_loss += loss
        valid_pairs += 1
    end
    
    return valid_pairs > 0 ? total_loss / valid_pairs : 0.0
end

function compute_flow_matching_loss_batch(model::MultiStartGFlowNetModel, states, params)
    if isempty(states)
        return 0.0
    end
    
    total_loss = 0.0
    valid_states = 0
    
    for state in states
        if is_terminal_state(state)
            continue
        end
        
        # Get flow estimate from neural network
        estimated_flow = flow_estimate(
            model.flow_estimator, state,
            params.flow, model.states.flow
        )
        
        # Compute expected flow
        expected_flow = Zygote.@ignore begin
            applicable_actions = get_applicable_actions(state, model.all_actions)
            if isempty(applicable_actions)
                0.0
            else
                action_probs = forward_action_probabilities(
                    model.forward_policy, state, model.all_actions,
                    params.forward, model.states.forward
                )
                
                flow_sum = 0.0
                for (idx, action) in enumerate(model.all_actions)
                    if action in applicable_actions
                        next_state = apply_action(action, state)
                        next_flow = flow(model, next_state)
                        flow_sum += action_probs[idx] * next_flow
                    end
                end
                flow_sum
            end
        end
        
        # Flow matching loss
        loss = (estimated_flow - expected_flow)^2
        total_loss += loss
        valid_states += 1
    end
    
    return valid_states > 0 ? total_loss / valid_states : 0.0
end