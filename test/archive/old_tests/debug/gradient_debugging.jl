# Gradient Debugging Script - Investigating Zero Gradient Issue
# This script systematically investigates why gradients are zero during training

using GFlowNet
using Zygote
using ComponentArrays
using Random
using Statistics

println("🔍 GRADIENT DEBUGGING ANALYSIS")
println("="^50)
println("Issue: Gradient norms are 0.0 while loss decreases - investigating root cause")
println()

Random.seed!(42)

# =============================================================================
# Setup Test Model
# =============================================================================

println("1️⃣ Creating Test Model...")
model = create_grid_world_gflownet(
    grid_size=3,
    hidden_dim=32,
    learning_rate=0.01,
    allow_all_moves=true
)

println("✅ Model created")
println("   - Parameters: $(length(model.parameters))")
println("   - Parameter types: $(typeof(model.parameters))")
println("   - Forward params keys: $(keys(model.parameters.forward))")

# =============================================================================
# Test 1: Basic Parameter Gradient Test
# =============================================================================

println("\n2️⃣ Testing Basic Parameter Gradients...")

# Simple test: can we get gradients from parameters?
test_fn = ps -> sum(ps.forward.layer_1.weight)

try
    grad_result = Zygote.gradient(test_fn, model.parameters)
    println("✅ Basic parameter gradient: SUCCESS")
    println("   - Gradient exists: $(grad_result[1] !== nothing)")
    if grad_result[1] !== nothing
        grad_norm = sqrt(sum(sum(abs2, grad) for grad in values(grad_result[1]) if grad isa AbstractArray))
        println("   - Gradient norm: $grad_norm")
    end
catch e
    println("❌ Basic parameter gradient: FAILED")
    println("   Error: $e")
end

# =============================================================================
# Test 2: Neural Network Forward Pass Gradients
# =============================================================================

println("\n3️⃣ Testing Neural Network Forward Pass Gradients...")

test_state = model.initial_state
test_features = state_to_features(test_state)

# Test if we can get gradients through the neural network
nn_test_fn = ps -> begin
    logits, _ = model.forward_policy.model(test_features, ps, model.states.forward)
    return sum(logits)
end

try
    grad_result = Zygote.gradient(nn_test_fn, model.parameters.forward)
    println("✅ Neural network gradient: SUCCESS")
    println("   - Gradient exists: $(grad_result[1] !== nothing)")
    if grad_result[1] !== nothing
        grad_norm = sqrt(sum(sum(abs2, grad) for grad in values(grad_result[1]) if grad isa AbstractArray))
        println("   - Gradient norm: $grad_norm")
    end
catch e
    println("❌ Neural network gradient: FAILED")
    println("   Error: $e")
end

# =============================================================================
# Test 3: Single Trajectory Loss Gradient
# =============================================================================

println("\n4️⃣ Testing Single Trajectory Loss Gradients...")

# Sample a trajectory
trajectory = sample_trajectory(model)
println("   - Trajectory length: $(length(trajectory.states))")
println("   - Terminal reward: $(reward(trajectory.states[end]))")

# Test gradients through single trajectory loss
single_loss_fn = ps -> compute_single_trajectory_loss(model, trajectory, ps)

try
    loss_val, grad_result = Zygote.withgradient(single_loss_fn, model.parameters)
    println("✅ Single trajectory loss gradient: SUCCESS")
    println("   - Loss value: $loss_val")
    println("   - Gradient exists: $(grad_result[1] !== nothing)")

    if grad_result[1] !== nothing
        grad_norm = sqrt(sum(sum(abs2, grad) for grad in values(grad_result[1]) if grad isa AbstractArray; init=0.0))
        println("   - Gradient norm: $grad_norm")

        # Check individual parameter gradients
        println("   - Forward policy gradients:")
        for (key, grad) in pairs(grad_result[1].forward)
            if grad isa AbstractArray
                key_norm = sqrt(sum(abs2, grad))
                println("     - $key: norm = $key_norm, size = $(size(grad))")
            end
        end
    end
catch e
    println("❌ Single trajectory loss gradient: FAILED")
    println("   Error: $e")
end

# =============================================================================
# Test 4: Batch Trajectory Loss Gradient
# =============================================================================

println("\n5️⃣ Testing Batch Trajectory Loss Gradients...")

# Sample multiple trajectories
trajectories = [sample_trajectory(model) for _ in 1:8]
config = TrainingConfig()

batch_loss_fn = ps -> compute_trajectory_loss(model, trajectories, ps, config)

try
    loss_val, grad_result = Zygote.withgradient(batch_loss_fn, model.parameters)
    println("✅ Batch trajectory loss gradient: SUCCESS")
    println("   - Loss value: $loss_val")
    println("   - Gradient exists: $(grad_result[1] !== nothing)")

    if grad_result[1] !== nothing
        grad_norm = sqrt(sum(sum(abs2, grad) for grad in values(grad_result[1]) if grad isa AbstractArray; init=0.0))
        println("   - Gradient norm: $grad_norm")

        if grad_norm == 0.0
            println("   🚨 FOUND THE ISSUE: Batch gradient norm is zero!")
            println("   - Investigating individual parameter gradients...")

            for (section_key, section_grads) in pairs(grad_result[1])
                println("     Section: $section_key")
                for (param_key, grad) in pairs(section_grads)
                    if grad isa AbstractArray
                        param_norm = sqrt(sum(abs2, grad))
                        has_nonzero = any(!iszero, grad)
                        println("       - $param_key: norm=$param_norm, has_nonzero=$has_nonzero, size=$(size(grad))")
                    end
                end
            end
        end
    end
catch e
    println("❌ Batch trajectory loss gradient: FAILED")
    println("   Error: $e")
end

# =============================================================================
# Test 5: Investigating Loss Function Components
# =============================================================================

println("\n6️⃣ Investigating Loss Function Components...")

# Check if the loss actually depends on parameters
println("   Testing parameter dependency...")

# Save original parameters
original_params = deepcopy(model.parameters)

# Modify parameters slightly
modified_params = deepcopy(model.parameters)
modified_params.forward.layer_1.weight .+= 0.1

# Compare losses
loss_original = compute_trajectory_loss(model, trajectories, original_params, config)
loss_modified = compute_trajectory_loss(model, trajectories, modified_params, config)

println("   - Loss with original params: $loss_original")
println("   - Loss with modified params: $loss_modified")
println("   - Loss difference: $(abs(loss_modified - loss_original))")

if abs(loss_modified - loss_original) < 1e-10
    println("   🚨 CRITICAL ISSUE: Loss doesn't depend on parameters!")
    println("   This means the loss function is not actually using the neural network parameters")
else
    println("   ✅ Loss properly depends on parameters")
end

# =============================================================================
# Test 6: Investigating Zygote.@ignore Impact
# =============================================================================

println("\n7️⃣ Testing Impact of Zygote.@ignore Wrapping...")

# Test the same trajectory loss but with less @ignore wrapping
unsafe_loss_fn = ps -> begin
    valid_trajectories = [traj for traj in trajectories if is_valid_trajectory(traj)]

    if isempty(valid_trajectories)
        return 0.0
    end

    losses = []
    for trajectory in valid_trajectories
        log_prob_sum = 0.0

        for i in 1:(length(trajectory.states)-1)
            state = trajectory.states[i]
            action = trajectory.actions[i]

            features = state_to_features(state)
            logits_vec, _ = model.forward_policy.model(features, ps.forward, model.states.forward)

            # Get applicable actions WITHOUT @ignore
            applicable_actions = get_applicable_actions(state, model.all_actions)

            if isempty(applicable_actions)
                push!(losses, Inf)
                break
            end

            # Find action index
            action_idx = findfirst(a -> a == action, model.all_actions)
            if isnothing(action_idx)
                push!(losses, Inf)
                break
            end

            # Find applicable indices
            applicable_indices = [i for (i, a) in enumerate(model.all_actions) if a in applicable_actions]

            if !(action_idx in applicable_indices)
                push!(losses, Inf)
                break
            end

            applicable_logits = logits_vec[applicable_indices]
            log_probs = applicable_logits .- logsumexp(applicable_logits)

            action_pos = findfirst(==(action_idx), applicable_indices)
            if isnothing(action_pos)
                push!(losses, Inf)
                break
            end

            log_prob_sum += log_probs[action_pos]
        end

        # Terminal reward
        terminal_state = trajectory.states[end]
        terminal_reward = reward(terminal_state)
        if terminal_reward <= 0
            terminal_reward = 1e-8
        end

        log_reward = log(terminal_reward)
        trajectory_balance_error = log_prob_sum - log_reward
        push!(losses, trajectory_balance_error^2)
    end

    finite_losses = filter(!isinf, losses)
    return isempty(finite_losses) ? Inf : mean(finite_losses)
end

try
    loss_val, grad_result = Zygote.withgradient(unsafe_loss_fn, model.parameters)
    println("✅ Unsafe loss function gradient: SUCCESS")
    println("   - Loss value: $loss_val")
    println("   - Gradient exists: $(grad_result[1] !== nothing)")

    if grad_result[1] !== nothing
        grad_norm = sqrt(sum(sum(abs2, grad) for grad in values(grad_result[1]) if grad isa AbstractArray; init=0.0))
        println("   - Gradient norm: $grad_norm")

        if grad_norm > 0.0
            println("   🎯 FOUND IT: Gradients work WITHOUT excessive @ignore wrapping!")
            println("   The issue is likely over-aggressive Zygote.@ignore usage")
        end
    end
catch e
    println("❌ Unsafe loss function gradient: FAILED")
    println("   Error: $e")
end

# =============================================================================
# Test 7: Training Step Simulation
# =============================================================================

println("\n8️⃣ Simulating Full Training Step...")

try
    # Use the built-in training step
    loss_val, gradient_norm = train_step!(model, trajectories, config)
    println("✅ Full training step: SUCCESS")
    println("   - Loss: $loss_val")
    println("   - Gradient norm: $gradient_norm")

    if gradient_norm == 0.0
        println("   🚨 CONFIRMED: Training step produces zero gradients")
        println("   This confirms the issue is in the main training pipeline")
    end
catch e
    println("❌ Full training step: FAILED")
    println("   Error: $e")
end

# =============================================================================
# Summary and Diagnosis
# =============================================================================

println("\n" * "="^50)
println("🔬 DIAGNOSIS SUMMARY")
println("="^50)

println("""
Based on the tests above, the zero gradient issue is likely caused by:

1. 🎯 MOST LIKELY: Over-aggressive Zygote.@ignore wrapping
   - Too many operations wrapped with @ignore
   - This prevents gradients from flowing through the computation graph
   - The loss function becomes non-differentiable w.r.t. parameters

2. 🔍 POSSIBLE: ComponentArray parameter handling issue
   - Parameters not properly connected to computation
   - Gradient computation succeeds but produces zeros

3. 🔍 LESS LIKELY: Lux model state issues
   - Model states not properly handled during AD

RECOMMENDED FIXES:
1. Remove excessive Zygote.@ignore wrapping from core computation
2. Only wrap true validation/debugging operations
3. Ensure neural network parameters flow through loss computation
4. Test gradient flow at each step of the pipeline

The fact that loss decreases while gradients are zero suggests:
- The optimizer is getting zero updates but loss varies due to stochasticity
- OR there's a bug in gradient norm computation
- OR gradients are computed but immediately zeroed out
""")

println("\n🔧 Next steps:")
println("1. Fix Zygote.@ignore wrapping in compute_trajectory_loss")
println("2. Verify gradient flow through each component")
println("3. Test with simpler loss functions first")
println("4. Add gradient debugging to main training loop")
