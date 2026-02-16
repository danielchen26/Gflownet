using GFlowNet
using Random

# Test single detailed balance computation
println("Testing single detailed balance computation...")

model = create_grid_world_gflownet(
    grid_size=3,
    hidden_dim=32,
    include_backward=true
)

source = GridState(1, 1, false)
target = GridState(2, 1, false)

# Test basic loss computation
println("\n1. Testing basic loss computation:")
try
    loss = detailed_balance_loss(model, source, target)
    println("✓ Loss computed successfully: $loss")
catch e
    println("✗ Error: $e")
end

# Test flow computation
println("\n2. Testing flow computation:")
try
    source_flow = flow(model, source)
    target_flow = flow(model, target)
    println("✓ Source flow: $source_flow")
    println("✓ Target flow: $target_flow")
catch e
    println("✗ Error in flow: $e")
end

# Test training with one iteration
println("\n3. Testing DETAILED_BALANCE training (1 iteration):")
config = TrainingConfig(
    objective=DETAILED_BALANCE,
    n_iterations=1,
    batch_size=2,
    learning_rate=0.01
)

try
    history = train_gflownet(model, config; verbose=true)
    println("✓ Training completed")
    println("  Loss: $(history.losses[1])")
catch e
    println("✗ Training error: $e")
    println("Stack trace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

# Test trajectory sampling and state pair extraction
println("\n4. Testing trajectory sampling and state pairs:")
trajectory = sample_trajectory(model)
println("✓ Sampled trajectory with $(length(trajectory.states)) states")

# Extract state pairs
state_pairs = [(trajectory.states[i], trajectory.states[i+1]) 
               for i in 1:(length(trajectory.states)-1)]
println("✓ Extracted $(length(state_pairs)) state pairs")

# Test loss computation for each pair
println("\n5. Testing loss for each state pair:")
for (i, (s, t)) in enumerate(state_pairs)
    try
        # Check if transition is valid
        applicable_actions = get_applicable_actions(s, model.all_actions)
        can_transition = any(apply_action(a, s) == t for a in applicable_actions)
        
        if can_transition && !is_terminal_state(s)
            loss = detailed_balance_loss(model, s, t)
            println("  Pair $i: Loss = $loss")
        else
            println("  Pair $i: Invalid transition")
        end
    catch e
        println("  Pair $i: Error - $e")
    end
end