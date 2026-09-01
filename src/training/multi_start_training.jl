# Training integration for Multi-Start GFlowNets
# Handles loss computation with per-initial-state partition functions

using Zygote
using Statistics
using Optimisers
using ..GFlowNet: TrainingConfig, TrainingHistory, TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING
using ..GFlowNet: compute_gradient_norm, any_invalid, clear_flow_cache!
using ..GFlowNet: forward_transition_probability, backward_transition_probability
# Needed for the P_B term in both the TB and DB losses below. backward_parent_states is
# the uniform-over-parents fallback used when no backward policy exists.
using ..GFlowNet: compute_backward_probability, backward_parent_states
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
        # WHICH state pairs appear is structural -- it depends on the sampled trajectories,
        # not on the parameters -- so the collection must be built outside the gradient.
        # Without @ignore, Zygote refuses the push! with
        # "Mutating arrays is not supported", the training loop catches it, and the run
        # reports a full history of NaN. The TB branch above already does this.
        state_pairs = Zygote.@ignore begin
            pairs = Tuple{AbstractState, AbstractState}[]
            for (traj, _) in trajectories_with_idx
                is_valid_trajectory(traj) || continue
                for i in 1:(length(traj.states)-1)
                    push!(pairs, (traj.states[i], traj.states[i+1]))
                end
            end
            pairs
        end

        if isempty(state_pairs)
            return 0.0
        end
        
        # Detailed balance loss doesn't depend on initial state
        return compute_detailed_balance_loss_batch(model, state_pairs, params)
        
    elseif config.objective == FLOW_MATCHING
        # Same reason as the DETAILED_BALANCE branch: the state SET is structural.
        states = Zygote.@ignore begin
            acc = AbstractState[]
            for (traj, _) in trajectories_with_idx
                is_valid_trajectory(traj) || continue
                for state in traj.states[1:end-1]
                    is_terminal_state(state) || push!(acc, state)
                end
            end
            unique(acc)
        end

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
    log_backward_sum = 0.0
    
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

        # P_B(s_i | s_{i+1}). Two corrections live here.
        #
        # (1) The term was ABSENT ENTIRELY -- not guarded, not conditional, simply not
        #     written -- so this loss was P_B == 1 unconditionally. That is a distribution
        #     only where every state has one parent, and the single-start version of the same
        #     omission made TB converge to sum_x n_paths(x) R(x) instead of sum_x R(x):
        #     measured 77.928 against a true 19.0 on the 3x3 grid. See losses.jl:535.
        #
        # (2) The first fix normalised over `backward_parent_states`, the GLOBAL parent set.
        #     Wrong for any start that is not the source: a child outside that start's cone
        #     contributes parents this DAG cannot reach, sum_tau P_B(tau|x) leaks below 1
        #     (measured 0.125), and Z collapses -- start (2,2) on a 4x4 landed on 10.25
        #     against a true 25.0, a 59% error, while the start-(1,1) control arm was exact.
        #     `reachable_parent_count` filters to the cone. See core/multi_start.jl.
        child = trajectory.states[i + 1]
        if !isnothing(model.backward_policy) && haskey(params, :backward)
            pb = compute_backward_probability(
                model.backward_policy, child, trajectory.states[i],
                params.backward, model.states.backward, model.all_actions
            )
            log_backward_sum += log(max(pb, 1e-8))
        else
            n_parents = Zygote.@ignore reachable_parent_count(model, child, initial_idx)
            n_parents > 1 && (log_backward_sum += -log(n_parents))
        end
    end

    # Get terminal reward
    terminal_state = trajectory.states[end]
    terminal_reward = Zygote.@ignore reward(terminal_state)

    if terminal_reward <= 0
        terminal_reward = 1e-8
    end

    # Use the correct log Z for this trajectory's initial state
    log_Z = params.log_Z[initial_idx]

    # Trajectory balance: (log Z + log P_F(tau) - log R - log P_B(tau|x))^2
    trajectory_balance_error = log_Z + log_prob_sum - log(terminal_reward) - log_backward_sum
    
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
    # DETAILED_BALANCE and FLOW_MATCHING are IMPLEMENTED for multi-start models now. They
    # used to throw MethodError on every iteration -- no `forward_transition_probability`
    # or `flow` method existed for this type -- and the loop below caught it and pushed
    # NaN, so a full-length history of NaN was returned instead of an error. Measured 0 of
    # 10 finite losses for both.
    #
    # The missing piece was a flow definition, and it turned out not to need one: F(s) is
    # START-INDEPENDENT. The recursion F(s) = sum_children F(c) P_B(s|c) with F(x) = R(x)
    # at terminals reads only the action set and the backward policy, never the initial
    # state. Only F(s_0^i) = Z_i is per-start, which the log_Z VECTOR already holds. See
    # core/multi_start.jl for the three methods, which delegate to the same policy-context
    # recursion flows.jl exposes, so there is no second copy to drift.
    #
    # The objective whitelist stays as a guard against the failure MODE, not the objective:
    # anything not handled by compute_trajectory_loss_multi_start must say so here rather
    # than be discovered as NaN inside the loop.
    if !(config.objective in (TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING))
        throw(ArgumentError(
            "$(config.objective) is not implemented for MultiStartGFlowNetModel. " *
            "Supported: TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING. Refused here " *
            "rather than inside the training loop, which converts any throw into a NaN " *
            "entry and reports the run as complete."))
    end

    # Reachability is a function of the initial state, the action set and the domain's
    # applicability rules -- never of parameters -- so it is computed once per run and reused
    # across iterations. It is NOT safe across runs: grid world installs a global GRID_CONFIG
    # at model construction, so a model built after another of a different size would inherit
    # a stale cone. Cleared here rather than never.
    GFlowNet.clear_reachability_cache!()

    # DETAILED_BALANCE and FLOW_MATCHING need a finite F(s), which they get from the
    # conservation recursion -- and that recursion has no solution on a cyclic state graph:
    # the backward transfer matrix has spectral radius exactly 1, sigma_min(I - W') is
    # 1.5e-16, and the partial sums for sum_tau P_B(tau|x) diverge as 2.0/12.0/49.5/249.5
    # at horizons 10/50/200/1000 where the value must be 1.
    #
    # `compute_recursive_flow` detects the cycle and throws. Probe it HERE, once, rather than
    # letting it throw on iteration 1: the loop below catches everything and pushes NaN, so a
    # throw from inside is invisible and the run reports as complete with a full history of
    # NaN. That is the same failure mode this whole function has been repaired for twice.
    #
    # TRAJECTORY_BALANCE is deliberately NOT probed. It carries Z as a learned parameter, so
    # cycles are absorbed into log_Z and it trains -- it simply has no analytic ground truth
    # there, which is documented at the allow_all_moves keyword rather than refused.
    if config.objective in (DETAILED_BALANCE, FLOW_MATCHING)
        try
            compute_recursive_flow(model, model.initial_states[1])
        catch e
            e isa ArgumentError || rethrow()
            throw(ArgumentError(
                "$(config.objective) cannot be trained on this model: " * e.msg))
        end
    end

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

# Use logsumexp from losses.jl
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
        
        # P_F and P_B MUST be computed from `params` -- the vector being differentiated --
        # not from `model.parameters`. The convenience wrappers
        # forward_transition_probability / backward_transition_probability read the model's
        # STORED parameters, so using them here made the loss independent of `params` and
        # the gradient exactly 0.0: the objective could not train at all. Same defect the
        # single-start DB path carried until its flows were made differentiable
        # (see src/training/losses.jl:151).
        probs = forward_action_probabilities(model.forward_policy, source,
                                            model.all_actions, params.forward,
                                            model.states.forward)
        applicable = Zygote.@ignore get_applicable_actions(source, model.all_actions)
        idx = Zygote.@ignore findfirst(a -> a in applicable && apply_action(a, source) == target,
                                       model.all_actions)
        isnothing(idx) && continue
        forward_prob = probs[idx]

        # P_B: uniform over parents when there is no backward policy. NOT 1.0 -- that is a
        # distribution only where every state has a unique parent, and this path is
        # reachable by default since create_multi_start_gflownet defaults
        # include_backward = false (multi_start.jl:257) and multi-start training does not go
        # through validate_training_config (MultiStartGFlowNetModel is not a GFlowNetModel).
        backward_prob = if isnothing(model.backward_policy) || !haskey(params, :backward)
            parents = Zygote.@ignore backward_parent_states(target, model.all_actions)
            isempty(parents) ? 1.0 : 1.0 / length(parents)
        else
            compute_backward_probability(model.backward_policy, target, source,
                                         params.backward, model.states.backward,
                                         model.all_actions)
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