using GFlowNet
using Zygote

# Create a simple model
model = create_grid_world_gflownet(
    grid_size=2,
    hidden_dim=16,
    include_backward=true
)

# Test direct flow computation
println("1. Testing direct flow computation with Zygote:")
source = GridState(1, 1, false)
target = GridState(2, 1, false)

# This should work
loss1 = detailed_balance_loss(model, source, target)
println("✓ Direct loss: $loss1")

# Test gradient computation
println("\n2. Testing gradient of detailed_balance_loss:")
try
    grad = Zygote.gradient(model.parameters) do params
        # Temporarily update model parameters for gradient computation
        old_params = model.parameters
        model.parameters = params
        
        loss = detailed_balance_loss(model, source, target)
        
        # Restore original parameters
        model.parameters = old_params
        
        loss
    end
    println("✓ Gradient computed successfully")
catch e
    println("✗ Error: $e")
    # Print more details
    if isa(e, ErrorException) && contains(string(e), "Mutating arrays")
        println("  This is the mutation error we're trying to fix")
    end
end

# Test if the issue is in flow computation
println("\n3. Testing flow computation with Zygote:")
try
    grad = Zygote.gradient(model.parameters) do params
        # Just compute flow
        flow(model, source)
    end
    println("✓ Flow gradient computed successfully")
catch e
    println("✗ Error in flow gradient: $e")
end

# Test if memoization is the issue
println("\n4. Testing after clearing cache:")
clear_flow_cache!()
try
    grad = Zygote.gradient(model.parameters) do params
        old_params = model.parameters
        model.parameters = params
        loss = detailed_balance_loss(model, source, target)
        model.parameters = old_params
        loss
    end
    println("✓ Gradient computed after cache clear")
catch e
    println("✗ Error even after cache clear: $e")
end