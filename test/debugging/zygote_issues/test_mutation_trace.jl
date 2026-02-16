using GFlowNet
using Zygote

# Create model
model = create_grid_world_gflownet(
    grid_size=2,
    hidden_dim=16,
    include_backward=true
)

source = GridState(1, 1, false)
target = GridState(2, 1, false)

# Test each component separately
println("Testing components of detailed_balance_loss:")

# 1. Test forward probability
println("\n1. Forward probability gradient:")
try
    grad = Zygote.gradient(model.parameters) do params
        forward_transition_probability(model, source, target)
    end
    println("✓ Success")
catch e
    println("✗ Error: $e")
end

# 2. Test backward probability
println("\n2. Backward probability gradient:")
try
    grad = Zygote.gradient(model.parameters) do params
        backward_transition_probability(model, target, source)
    end
    println("✓ Success")
catch e
    println("✗ Error: $e")
end

# 3. Test flow computation
println("\n3. Flow computation gradient:")
try
    grad = Zygote.gradient(model.parameters) do params
        flow(model, source) + flow(model, target)
    end
    println("✓ Success")
catch e
    println("✗ Error: $e")
end

# 4. Test get_applicable_actions
println("\n4. Get applicable actions (should be non-differentiable):")
try
    actions = get_applicable_actions(source, model.all_actions)
    println("✓ Got $(length(actions)) actions")
catch e
    println("✗ Error: $e")
end

# 5. Now test the full function with detailed output
println("\n5. Full detailed_balance_loss with tracing:")
try
    # Clear any caches
    clear_flow_cache!()
    
    # Run with gradient
    grad = Zygote.gradient(model.parameters) do params
        # Manually inline the detailed_balance_loss to see where it fails
        
        # Check backward policy
        if isnothing(model.backward_policy)
            throw(ArgumentError("Detailed balance requires backward policy"))
        end
        
        # Get applicable actions
        applicable_actions = get_applicable_actions(source, model.all_actions)
        
        # Check transition validity
        can_transition = false
        for action in applicable_actions
            if apply_action(action, source) == target
                can_transition = true
                break
            end
        end
        
        if !can_transition
            throw(ArgumentError("Invalid transition"))
        end
        
        # Compute probabilities
        forward_prob = forward_transition_probability(model, source, target)
        backward_prob = backward_transition_probability(model, target, source)
        
        # Compute flows
        source_flow = flow(model, source)
        target_flow = flow(model, target)
        
        # Compute loss
        left = log(max(forward_prob, 1e-8)) + log(max(source_flow, 1e-8))
        right = log(max(backward_prob, 1e-8)) + log(max(target_flow, 1e-8))
        
        (left - right)^2
    end
    println("✓ Success! Gradient computed")
catch e
    println("✗ Error: $e")
    
    # Try to get more info about where the error occurs
    if isa(e, ErrorException) && contains(string(e), "Mutating arrays")
        println("\nTrying to locate mutation...")
        
        # Test without flow computation
        try
            grad = Zygote.gradient(model.parameters) do params
                forward_prob = forward_transition_probability(model, source, target)
                backward_prob = backward_transition_probability(model, target, source)
                log(forward_prob) - log(backward_prob)
            end
            println("  ✓ Works without flow computation")
        catch
            println("  ✗ Fails even without flow")
        end
    end
end