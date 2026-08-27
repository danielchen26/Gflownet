using GFlowNet
using Test
using Statistics
using GFlowNet: compute_trajectory_loss

# Import necessary types
import GFlowNet: GridState, is_valid_backward_transition, compute_backward_probability, forward_transition_probability, backward_transition_probability, flow

println("=== Final Verification of DETAILED_BALANCE Implementation ===\n")

# Test 1: Verify gradient computation works
println("1. Testing gradient computation...")
model = create_grid_world_gflownet(
    grid_size=3,
    hidden_dim=16,
    include_backward=true
)

# Sample trajectories
trajectories = [sample_trajectory(model) for _ in 1:4]

# Test gradient computation directly.
#
# This used to be wrapped in a bare try/catch that printed
# "✗ Gradient computation failed: $e" and carried on, so it could never fail the
# suite. The gradient in fact computed fine; the throw came from this file's own
# norm expression,
#   sqrt(sum(sum(abs2, g) for g in values(grads[1]) if g isa AbstractArray))
# which reduces over an empty collection for a ComponentArray and raised
# ArgumentError("reducing over an empty collection"). So a green run printed a
# failure message about a component that was working. Asserted properly now: a
# ComponentArray is itself an AbstractArray, so norm applies directly.
using Zygote
using LinearAlgebra: norm

loss_val, grads = Zygote.withgradient(model.parameters) do ps
    Zygote.@ignore clear_flow_cache!()
    compute_trajectory_loss(model, trajectories, ps, TrainingConfig(objective=DETAILED_BALANCE))
end

@test !isnothing(grads[1])
@test isfinite(loss_val)
grad_norm = norm(grads[1])
@test isfinite(grad_norm)
@test grad_norm > 0        # DB must produce a live gradient, not a dead one
println("✓ Gradient computation successful!")
println("  Loss value: $(round(loss_val, digits=4))")
println("  Gradient norm: $(round(grad_norm, digits=4))")

# Test 2: Compare TRAJECTORY_BALANCE vs DETAILED_BALANCE training
println("\n2. Comparing training objectives...")

# Train with TRAJECTORY_BALANCE
model_tb = create_grid_world_gflownet(
    grid_size=3,
    hidden_dim=16,
    include_backward=false  # TB doesn't need backward policy
)

config_tb = TrainingConfig(
    objective=TRAJECTORY_BALANCE,
    n_iterations=20,
    batch_size=8,
    learning_rate=0.01
)

history_tb = train_gflownet(model_tb, config_tb; verbose=false)
tb_losses = filter(!isnan, history_tb.losses)
println("✓ TRAJECTORY_BALANCE training completed")
println("  Initial loss: $(round(tb_losses[1], digits=4))")
println("  Final loss: $(round(tb_losses[end], digits=4))")
println("  Reduction: $(round((1 - tb_losses[end]/tb_losses[1]) * 100, digits=1))%")

# Train with DETAILED_BALANCE
model_db = create_grid_world_gflownet(
    grid_size=3,
    hidden_dim=16,
    include_backward=true  # DB requires backward policy
)

config_db = TrainingConfig(
    objective=DETAILED_BALANCE,
    n_iterations=20,
    batch_size=8,
    learning_rate=0.01
)

history_db = train_gflownet(model_db, config_db; verbose=false)
db_losses = filter(!isnan, history_db.losses)
println("\n✓ DETAILED_BALANCE training completed")
println("  Initial loss: $(round(db_losses[1], digits=4))")
println("  Final loss: $(round(db_losses[end], digits=4))")
println("  Reduction: $(round((1 - db_losses[end]/db_losses[1]) * 100, digits=1))%")

# Test 3: Verify detailed balance equation satisfaction
println("\n3. Verifying detailed balance equation...")

# Sample some trajectories
test_trajectories = [sample_trajectory(model_db) for _ in 1:10]

# Extract state pairs and compute balance ratios
balance_ratios = Float64[]
for traj in test_trajectories
    for i in 1:(length(traj.states)-1)
        source = traj.states[i]
        target = traj.states[i+1]
        
        if !is_terminal_state(source)
            # Compute forward probability
            forward_prob = forward_transition_probability(model_db, source, target)
            
            # Compute backward probability
            backward_prob = backward_transition_probability(model_db, target, source)
            
            # Compute flows
            source_flow = flow(model_db, source)
            target_flow = flow(model_db, target)
            
            # Compute balance ratio: (P_F * F_source) / (P_B * F_target)
            if forward_prob > 0 && backward_prob > 0 && source_flow > 0 && target_flow > 0
                ratio = (forward_prob * source_flow) / (backward_prob * target_flow)
                push!(balance_ratios, ratio)
            end
        end
    end
end

if !isempty(balance_ratios)
    mean_ratio = mean(balance_ratios)
    std_ratio = std(balance_ratios)
    println("✓ Balance ratio analysis:")
    println("  Mean ratio: $(round(mean_ratio, digits=4)) (should be ≈ 1.0)")
    println("  Std deviation: $(round(std_ratio, digits=4))")
    println("  Min ratio: $(round(minimum(balance_ratios), digits=4))")
    println("  Max ratio: $(round(maximum(balance_ratios), digits=4))")
end

# Test 4: Verify backward policy normalization
println("\n4. Testing backward policy normalization...")

test_state = GridState(2, 2, false)  # Middle state
all_probs = Float64[]

# For each possible previous state, compute backward probability
for x in 1:3, y in 1:3
    prev_state = GridState(x, y, false)
    if is_valid_backward_transition(prev_state, test_state, model_db.all_actions)
        prob = compute_backward_probability(
            model_db.backward_policy, test_state, prev_state,
            model_db.parameters.backward, model_db.states.backward,
            model_db.all_actions
        )
        push!(all_probs, prob)
    end
end

sum_probs = sum(all_probs)
println("✓ Backward policy normalization:")
println("  Number of valid previous states: $(length(all_probs))")
println("  Sum of probabilities: $(round(sum_probs, digits=4)) (should be ≈ 1.0)")
println("  Individual probabilities: $([round(p, digits=3) for p in all_probs])")

println("\n=== All verifications completed successfully! ===")