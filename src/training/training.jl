# Main training loop and infrastructure for GFlowNet
# Consolidated from core/interface.jl for better organization

using Zygote
using Statistics
using Optimisers
using ComponentArrays
using Random

using ..GFlowNet: AbstractState, AbstractAction, GFlowNetModel, Trajectory
using ..GFlowNet: TrainingConfig, TrainingHistory, TrainingObjective
using ..GFlowNet: TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING, TRAJECTORY_LIKELIHOOD_MAXIMIZATION, MULTI_OBJECTIVE_TB, SHIFTED_COSH_TB
using ..GFlowNet: SamplingConfig
using ..GFlowNet: state_to_features, reward, is_terminal_state, is_applicable
using ..GFlowNet: get_applicable_actions, apply_action
using ..GFlowNet: forward_action_probabilities, compute_backward_probability
using ..GFlowNet: forward_transition_probability, backward_transition_probability
using ..GFlowNet: flow, flow_estimate, clear_flow_cache!

# =============================================================================
# Main Training Loop
# =============================================================================

"""
    train_gflownet(model::GFlowNetModel, config::TrainingConfig; kwargs...)

Train GFlowNet using proper Lux+Zygote patterns with optional learnable partition function.

# Arguments
- `model::GFlowNetModel`: The GFlowNet model to train
- `config::TrainingConfig`: Training configuration including:
  - `objective`: Training objective (e.g., TRAJECTORY_BALANCE)
  - `partition_function_method`: How to handle Z (SIMPLE_ESTIMATION or LEARNABLE_ESTIMATION)
  - `n_iterations`: Number of training iterations
  - `batch_size`: Batch size for trajectory sampling
  - `learning_rate`: Learning rate for optimizer

# Keyword Arguments
- `verbose::Bool=false`: Whether to print training progress
- `validation_data=nothing`: Optional validation data
- `callback=nothing`: Optional callback function(model, history, iteration)

# Returns
`TrainingHistory` containing:
- `losses`: Training loss per iteration
- `partition_function_estimates`: Z values over time (if using LEARNABLE_ESTIMATION)
- Other metrics based on configuration

# Example
```julia
# Train with learnable partition function
config = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    partition_function_method = LEARNABLE_ESTIMATION,
    n_iterations = 1000,
    batch_size = 64,
    learning_rate = 0.001
)

history = train_gflownet(model, config; verbose=true)

# Access learned Z
if model.partition_function_method == LEARNABLE_ESTIMATION
    learned_Z = exp(model.parameters.log_Z)
    println("Learned partition function: \$learned_Z")
end
```
"""
function train_gflownet(model::GFlowNetModel, config::TrainingConfig; verbose::Bool = false)
    history = TrainingHistory()

    # Initialize replay buffer if configured (JMLR 2023: Off-policy learning)
    replay_buffer = if config.use_replay_buffer
        GFlowNet.ReplayBuffer(config.replay_buffer_size; alpha=config.replay_priority_alpha)
    else
        nothing
    end

    if verbose
        println("🚀 Starting GFlowNet training...")
        println("   Configuration:")
        println("     - Objective: $(config.objective)")
        println("     - Iterations: $(config.n_iterations)")
        println("     - Batch size: $(config.batch_size)")
        println("     - Learning rate: $(config.learning_rate)")
        println("     - Temperature: $(config.temperature)")
        println("     - Epsilon (ε-uniform): $(config.epsilon)$(config.epsilon_decay ? " (annealed)" : "")")
        println("     - Entropy weight: $(config.entropy_weight)")
        if config.z_learning_rate_multiplier != 1.0
            println("     - Z learning rate multiplier: $(config.z_learning_rate_multiplier)x")
        end
        if config.use_replay_buffer
            println("     - Replay buffer: $(config.replay_buffer_size) capacity, $(Int(config.replay_ratio * 100))% replay")
        end
        # MOGFN-specific output
        if config.objective == MULTI_OBJECTIVE_TB
            println("   MOGFN-PC (Gap 5) Settings:")
            println("     - Objectives: $(config.mogfn_n_objectives)")
            println("     - Preference embedding dim: $(config.mogfn_preference_dim)")
            println("     - Dirichlet alpha: $(config.mogfn_dirichlet_alpha)")
            if isnothing(model.preference_encoder)
                println("   WARNING: MOGFN requires preference_encoder!")
                println("      Use create_mogfn_gflownet() or create_mogfn_molecular_gflownet()")
            else
                println("   Preference encoder detected")
                println("   Z(w) network detected")
            end
        end
        # TLM-specific output
        if config.objective == TRAJECTORY_LIKELIHOOD_MAXIMIZATION
            println("   TLM (ICLR 2025) Settings:")
            println("     - Backward weight (λ): $(config.tlm_backward_weight)")
            println("     - Backward entropy coeff: $(config.tlm_entropy_coeff)")
            println("     - Update frequency: every $(config.tlm_update_frequency) iteration(s)")
            if isnothing(model.backward_policy)
                println("   ⚠️  WARNING: TLM requires backward policy!")
                println("      Use include_backward_policy=true in create_gflownet")
            else
                println("   ✓ Backward policy detected")
            end
        end
    end

    for iteration in 1:config.n_iterations
        start_time = time()

        try
            # Compute current epsilon (annealed if epsilon_decay is true)
            # Anneal linearly from epsilon to 0 over training
            current_epsilon = if config.epsilon_decay
                config.epsilon * (1.0 - (iteration - 1) / config.n_iterations)
            else
                config.epsilon
            end

            # Create sampling config with ε-uniform exploration
            # This is the CRITICAL exploration mechanism for mode discovery
            sampling_config = GFlowNet.SamplingConfig(
                strategy = config.temperature != 1.0 ? GFlowNet.TEMPERATURE_SAMPLING : GFlowNet.STOCHASTIC_SAMPLING,
                temperature = config.temperature,
                epsilon = current_epsilon,
                max_trajectory_length = 100
            )

            # Sample fresh trajectories with ε-uniform exploration
            if config.objective == MULTI_OBJECTIVE_TB
                # MOGFN-PC (Gap 5): Sample with preference-conditioned policy
                # Each batch uses a single freshly sampled preference vector w from Dirichlet(α)
                n_obj = config.mogfn_n_objectives
                alpha = config.mogfn_dirichlet_alpha
                w = if alpha == 1.0
                    gammas = [randexp() for _ in 1:n_obj]
                    gammas ./ sum(gammas)
                else
                    gammas = Float64[]
                    for _ in 1:n_obj
                        d = alpha - 1.0/3.0
                        c = 1.0 / sqrt(9.0 * max(d, 1e-8))
                        g = alpha >= 1.0 ? begin
                            local v, x, u
                            while true
                                x = randn(); v = (1.0 + c * x)^3
                                if v > 0.0
                                    u = rand()
                                    if u < 1.0 - 0.0331 * x^4 || log(u) < 0.5 * x^2 + d * (1.0 - v + log(v))
                                        break
                                    end
                                end
                            end
                            d * v
                        end : randexp()  # fallback for alpha < 1
                        push!(gammas, g)
                    end
                    gammas ./ sum(gammas)
                end
                fresh_trajectories = [GFlowNet.sample_mogfn_trajectory(model, w; config=sampling_config)
                                      for _ in 1:config.batch_size]
            else
                fresh_trajectories = [GFlowNet.sample_trajectory(model; config=sampling_config) for _ in 1:config.batch_size]
            end

            # TLM (ICLR 2025): Add backward-sampled trajectories for extreme path asymmetry
            # This is the key mechanism that solves the 70:1 mode collapse problem
            if config.objective == TRAJECTORY_LIKELIHOOD_MAXIMIZATION && !isnothing(model.backward_policy)
                # Sample backward trajectories from terminal states
                # This bypasses the forward path asymmetry by starting from terminals
                backward_trajectories = try
                    # Get terminal states from recent forward trajectories
                    terminal_states = [traj.states[end] for traj in fresh_trajectories
                                       if !isempty(traj.states) && GFlowNet.is_terminal_state(traj.states[end])]

                    if !isempty(terminal_states)
                        # Sample backward from these terminals
                        n_backward = min(config.batch_size ÷ 2, length(terminal_states))
                        GFlowNet.sample_backward_trajectories_from_terminals(
                            model, terminal_states, n_backward; config=sampling_config
                        )
                    else
                        Trajectory[]
                    end
                catch e
                    # Backward sampling may fail if parent finding isn't implemented for the domain
                    # Log warning to help with debugging (not silent failure)
                    if verbose && iteration == 1
                        @warn "TLM backward sampling failed (will use forward-only): $e"
                    end
                    Trajectory[]
                end

                # Add backward trajectories to training batch
                if !isempty(backward_trajectories)
                    fresh_trajectories = vcat(fresh_trajectories, backward_trajectories)
                end
            end

            # Mix fresh and replay samples if using replay buffer
            training_data = if !isnothing(replay_buffer) && length(replay_buffer) >= config.batch_size
                # Add fresh trajectories to buffer with reward-based priority
                # Higher reward → higher priority → more likely to be replayed
                # This helps retain high-reward modes (critical for mode collapse prevention)
                for traj in fresh_trajectories
                    traj_reward = GFlowNet.reward(traj.states[end])
                    priority = GFlowNet.compute_trajectory_priority(traj_reward)
                    GFlowNet.add!(replay_buffer, traj, priority)
                end

                # Compute number of fresh vs replay samples
                n_replay = round(Int, config.batch_size * config.replay_ratio)
                n_fresh = config.batch_size - n_replay

                # Sample from replay buffer with importance weights
                replay_trajs, replay_weights, _ = GFlowNet.sample_with_weights(replay_buffer, n_replay)

                # Combine fresh and replay with weights
                # Fresh samples have weight 1.0 (on-policy), replay samples have importance weights
                # Handle edge case where n_fresh=0 (replay_ratio=1.0)
                fresh_subset = n_fresh > 0 ? fresh_trajectories[1:min(n_fresh, length(fresh_trajectories))] : Trajectory[]
                combined_trajs = vcat(fresh_subset, replay_trajs)
                combined_weights = vcat(ones(length(fresh_subset)), replay_weights)

                (trajectories=combined_trajs, weights=combined_weights, use_weights=true)
            else
                # No replay buffer or not enough samples yet - just add to buffer and use fresh
                if !isnothing(replay_buffer)
                    for traj in fresh_trajectories
                        traj_reward = GFlowNet.reward(traj.states[end])
                        GFlowNet.add!(replay_buffer, traj, GFlowNet.compute_trajectory_priority(traj_reward))
                    end
                end
                (trajectories=fresh_trajectories, weights=ones(length(fresh_trajectories)), use_weights=false)
            end

            training_trajectories = training_data.trajectories

            # Compute loss and gradients - use weighted version if replay samples included
            loss_val, gradient_norm = if training_data.use_weights
                train_step_weighted!(model, training_data.trajectories, training_data.weights, config)
            else
                train_step!(model, training_data.trajectories, config)
            end

            # Record metrics
            push!(history.losses, loss_val)
            push!(history.gradient_norms, gradient_norm)
            push!(history.iteration_times, time() - start_time)

            # Verbose output
            if verbose && (iteration % config.validation_frequency == 0)
                avg_loss = mean(filter(!isnan, history.losses[max(1, end-4):end]))
                println("   Iteration $iteration:")
                println("     - Loss: $(round(loss_val, digits=4))")
                println("     - Avg Loss (5): $(isnan(avg_loss) ? "NaN" : round(avg_loss, digits=4))")
                println("     - Gradient norm: $(round(gradient_norm, digits=4))")
                println("     - Time: $(round(time() - start_time, digits=3))s")
                println("     - Trajectories: $(length(training_trajectories))")
                if !isnothing(replay_buffer)
                    println("     - Replay buffer size: $(length(replay_buffer))")
                end
            end

        catch e
            # Record failed iteration
            push!(history.losses, NaN)
            push!(history.gradient_norms, NaN)
            push!(history.iteration_times, time() - start_time)

            if verbose
                println("   ⚠️  Training error at iteration $iteration: $e")
            end
        end
    end

    if verbose
        successful_iterations = count(!isnan, history.losses)
        final_loss = isempty(filter(!isnan, history.losses)) ? NaN : filter(!isnan, history.losses)[end]
        total_time = sum(history.iteration_times)

        println("   ✅ Training completed:")
        println("     - Final loss: $(isnan(final_loss) ? "NaN" : round(final_loss, digits=4))")
        println("     - Total time: $(round(total_time, digits=1))s")
        println("     - Successful iterations: $successful_iterations/$(config.n_iterations)")
    end

    return history
end

"""
    train_step!(model, trajectories, config)

Perform single training step using official Lux+Zygote pattern.
"""
function train_step!(model::GFlowNetModel, trajectories::Vector{Trajectory}, config::TrainingConfig)

    # Define loss function following official Lux pattern
    loss_function = ps -> begin
        # Clear flow cache before gradient computation to avoid mutation issues
        Zygote.@ignore clear_flow_cache!()
        compute_trajectory_loss(model, trajectories, ps, config)
    end

    # Compute gradients using official Zygote pattern
    loss_val, grads = Zygote.withgradient(loss_function, model.parameters)

    # Check for valid gradients
    if grads[1] === nothing || any_invalid(grads[1])
        return Inf, 0.0
    end

    # Compute gradient norm (before scaling)
    gradient_norm = compute_gradient_norm(grads[1])

    # Apply z_learning_rate_multiplier by scaling the log_Z gradient
    # This effectively gives Z a higher learning rate: lr_Z = lr * multiplier
    # Reference: Peptide generation paper (bioRxiv 2026) recommends 10x for faster Z convergence
    scaled_grads = if haskey(grads[1], :log_Z) && config.z_learning_rate_multiplier != 1.0
        scale_z_gradient(grads[1], config.z_learning_rate_multiplier)
    else
        grads[1]
    end

    # Apply gradient clipping (critical for Shifted-Cosh stability)
    # The gradient_clip_norm field existed in config but was never applied — fixing this latent bug
    scaled_norm = compute_gradient_norm(scaled_grads)
    if scaled_norm > config.gradient_clip_norm
        clip_scale = config.gradient_clip_norm / scaled_norm
        scaled_grads = clip_scale .* scaled_grads
    end

    # Update parameters using Optimisers.jl
    optimizer_state, parameters = Optimisers.update(model.optimizer, model.parameters, scaled_grads)

    # Update model state (mutation after gradient computation is safe)
    model.optimizer = optimizer_state
    model.parameters = parameters

    # Synchronize log_partition_function field with parameter if using LEARNABLE_ESTIMATION
    if haskey(parameters, :log_Z)
        model.log_partition_function = parameters.log_Z
    end

    return loss_val, gradient_norm
end

"""
    scale_z_gradient(grads, multiplier::Float64)

Scale the log_Z gradient by the given multiplier.
This effectively increases the learning rate for the partition function Z.

# Mathematical Foundation
By scaling ∇log_Z by multiplier M before the optimizer update:
    log_Z_new = log_Z - lr × M × ∇log_Z

This is equivalent to using learning rate (lr × M) for Z while keeping
the standard learning rate lr for all other parameters.

# Arguments
- `grads`: Gradient structure from Zygote (ComponentVector or NamedTuple)
- `multiplier::Float64`: Learning rate multiplier for Z (e.g., 10.0)

# Returns
New gradient structure with scaled log_Z gradient
"""
function scale_z_gradient(grads, multiplier::Float64)
    if !haskey(grads, :log_Z)
        return grads
    end

    # For ComponentArrays (the actual type from Zygote with our parameters),
    # we need to create a copy and modify in place
    if grads isa ComponentArrays.ComponentVector
        # Create a copy to avoid mutation
        scaled_grads = copy(grads)
        scaled_grads.log_Z = grads.log_Z * multiplier
        return scaled_grads
    elseif grads isa NamedTuple
        # For NamedTuple, use merge
        scaled_log_Z = grads.log_Z * multiplier
        return merge(grads, (log_Z = scaled_log_Z,))
    else
        # Fallback: try direct property access
        try
            scaled_grads = copy(grads)
            scaled_grads.log_Z = grads.log_Z * multiplier
            return scaled_grads
        catch
            return grads
        end
    end
end

"""
    train_step_weighted!(model, trajectories, weights, config)

Perform single training step with importance-weighted loss for off-policy learning.

This function applies importance sampling corrections when learning from
trajectories sampled under a different (older) policy, such as from a replay buffer.

# Arguments
- `model::GFlowNetModel`: The GFlowNet model to train
- `trajectories::Vector{Trajectory}`: Training trajectories
- `weights::Vector{Float64}`: Importance weights for each trajectory
- `config::TrainingConfig`: Training configuration

# Returns
Tuple of (loss_value, gradient_norm)
"""
function train_step_weighted!(model::GFlowNetModel, trajectories::Vector{Trajectory},
                             weights::Vector{Float64}, config::TrainingConfig)

    # Define weighted loss function
    loss_function = ps -> begin
        Zygote.@ignore clear_flow_cache!()
        compute_weighted_trajectory_loss(model, trajectories, weights, ps, config)
    end

    # Compute gradients
    loss_val, grads = Zygote.withgradient(loss_function, model.parameters)

    # Check for valid gradients
    if grads[1] === nothing || any_invalid(grads[1])
        return Inf, 0.0
    end

    # Compute gradient norm (before scaling)
    gradient_norm = compute_gradient_norm(grads[1])

    # Apply z_learning_rate_multiplier by scaling the log_Z gradient
    scaled_grads = if haskey(grads[1], :log_Z) && config.z_learning_rate_multiplier != 1.0
        scale_z_gradient(grads[1], config.z_learning_rate_multiplier)
    else
        grads[1]
    end

    # Apply gradient clipping
    scaled_norm = compute_gradient_norm(scaled_grads)
    if scaled_norm > config.gradient_clip_norm
        clip_scale = config.gradient_clip_norm / scaled_norm
        scaled_grads = clip_scale .* scaled_grads
    end

    # Update parameters using Optimisers.jl
    optimizer_state, parameters = Optimisers.update(model.optimizer, model.parameters, scaled_grads)

    # Update model state
    model.optimizer = optimizer_state
    model.parameters = parameters

    # Synchronize log_Z if using LEARNABLE_ESTIMATION
    if haskey(parameters, :log_Z)
        model.log_partition_function = parameters.log_Z
    end

    return loss_val, gradient_norm
end
