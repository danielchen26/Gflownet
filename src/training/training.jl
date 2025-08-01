# Main training loop and infrastructure for GFlowNet
# Consolidated from core/interface.jl for better organization

using Zygote
using Statistics
using Optimisers
using ComponentArrays
using Random

using ..GFlowNet: AbstractState, AbstractAction, GFlowNetModel, Trajectory
using ..GFlowNet: TrainingConfig, TrainingHistory, TrainingObjective
using ..GFlowNet: TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING
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

    if verbose
        println("🚀 Starting GFlowNet training...")
        println("   Configuration:")
        println("     - Objective: $(config.objective)")
        println("     - Iterations: $(config.n_iterations)")
        println("     - Batch size: $(config.batch_size)")
        println("     - Learning rate: $(config.learning_rate)")
    end

    for iteration in 1:config.n_iterations
        start_time = time()

        try
            # Sample trajectories
            trajectories = [GFlowNet.sample_trajectory(model) for _ in 1:config.batch_size]

            # Compute loss and gradients using official Lux pattern
            loss_val, gradient_norm = train_step!(model, trajectories, config)

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
                println("     - Trajectories: $(length(trajectories))")
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

    # Compute gradient norm
    gradient_norm = compute_gradient_norm(grads[1])

    # Update parameters using Optimisers.jl
    optimizer_state, parameters = Optimisers.update(model.optimizer, model.parameters, grads[1])

    # Update model state (mutation after gradient computation is safe)
    model.optimizer = optimizer_state
    model.parameters = parameters
    
    # Synchronize log_partition_function field with parameter if using LEARNABLE_ESTIMATION
    if haskey(parameters, :log_Z)
        model.log_partition_function = parameters.log_Z
    end

    return loss_val, gradient_norm
end