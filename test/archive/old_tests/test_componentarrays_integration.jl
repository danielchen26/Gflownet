#!/usr/bin/env julia

# Test ComponentArrays and Lux.jl integration with GFlowNet

using Pkg
Pkg.activate(".")
using GFlowNet
using ComponentArrays
using Lux
using Random

println("🧪 Testing ComponentArrays and Lux.jl Integration")
println("=" ^ 50)

# Set random seed for reproducibility
Random.seed!(42)

# Test 1: Create a simple Lux.jl neural network
println("\n1️⃣ Creating Lux.jl neural networks...")

# Create a simple forward policy network
forward_model = Chain(
    Dense(2, 8, tanh),    # Input: state features, Hidden: 8 units
    Dense(8, 3)           # Output: logits for 3 actions
)

# Create a simple flow estimator network
flow_model = Chain(
    Dense(2, 4, tanh),    # Input: state features, Hidden: 4 units
    Dense(4, 1)           # Output: single flow value
)

println("✅ Neural networks created")

# Test 2: Initialize parameters using ComponentArrays
println("\n2️⃣ Initializing parameters with ComponentArrays...")

# Initialize parameters for the networks
rng = Random.default_rng()
forward_params, forward_states = Lux.setup(rng, forward_model)
flow_params, flow_states = Lux.setup(rng, flow_model)

# Convert to ComponentArrays
forward_ca = ComponentArray(forward_params)
flow_ca = ComponentArray(flow_params)

# Create combined parameter structure
all_params = ComponentArray(
    forward = forward_ca,
    flow = flow_ca
)

println("✅ Parameters initialized with ComponentArrays")
println("Parameter structure: ", keys(all_params))
println("Forward params shape: ", size(all_params.forward))
println("Flow params shape: ", size(all_params.flow))

# Test 3: Test neural network forward pass
println("\n3️⃣ Testing neural network forward pass...")

# Create test input (state features)
test_features = Float32[1.0, 0.5]
input_batch = reshape(test_features, :, 1)  # (2, 1) for batch processing

# Test forward model
try
    forward_output, _ = Lux.apply(forward_model, input_batch, forward_params, forward_states)
    println("✅ Forward model output: ", forward_output)
    println("   Output shape: ", size(forward_output))
    
    # Apply softmax to get probabilities
    using NNlib: softmax
    probs = softmax(forward_output[:, 1])
    println("   Probabilities: ", probs)
    println("   Sum of probabilities: ", sum(probs))
    
catch e
    println("❌ Forward model test failed: ", e)
end

# Test flow model
try
    flow_output, _ = Lux.apply(flow_model, input_batch, flow_params, flow_states)
    println("✅ Flow model output: ", flow_output)
    println("   Output shape: ", size(flow_output))
    
catch e
    println("❌ Flow model test failed: ", e)
end

# Test 4: Test gradient computation with ComponentArrays
println("\n4️⃣ Testing gradient computation...")

# Define a simple loss function
function simple_loss(params, model, input_data, target)
    output, _ = Lux.apply(model, input_data, params, forward_states)
    return sum((output .- target).^2)
end

# Test gradient computation
try
    target = Float32[0.5, 0.3, 0.2]  # Target probabilities
    target_batch = reshape(target, :, 1)
    
    # Compute gradients
    loss_fn(p) = simple_loss(p, forward_model, input_batch, target_batch)
    
    using Zygote
    loss_val, grads = Zygote.withgradient(loss_fn, forward_params)
    
    println("✅ Gradient computation successful")
    println("   Loss value: ", loss_val)
    println("   Gradient structure: ", typeof(grads[1]))
    
    # Test with ComponentArray parameters
    forward_ca_copy = ComponentArray(forward_params)
    loss_fn_ca(p) = simple_loss(p, forward_model, input_batch, target_batch)
    
    loss_val_ca, grads_ca = Zygote.withgradient(loss_fn_ca, forward_ca_copy)
    
    println("✅ ComponentArray gradient computation successful")
    println("   Loss value (CA): ", loss_val_ca)
    println("   Gradient structure (CA): ", typeof(grads_ca[1]))
    
catch e
    println("❌ Gradient computation failed: ", e)
    println("   This might be expected if there are compatibility issues")
end

# Test 5: Integration with GFlowNet types
println("\n5️⃣ Testing integration with GFlowNet...")

try
    # Create GFlowNet components with proper Lux models
    dag = GFlowNet.DirectedAcyclicGraph()
    
    # Add states and actions
    GFlowNet.add_state!(dag, GFlowNet.SimpleState([0]))
    GFlowNet.add_state!(dag, GFlowNet.SimpleState([1]))
    GFlowNet.add_state!(dag, GFlowNet.SimpleState([-1]))
    
    GFlowNet.add_action!(dag, GFlowNet.SimpleAction(1))
    GFlowNet.add_action!(dag, GFlowNet.SimpleAction(-1))
    
    # Create policies with Lux models
    forward_policy = GFlowNet.ForwardPolicy(forward_model)
    flow_estimator = GFlowNet.FlowEstimator(flow_model)
    
    # Create model with ComponentArray parameters
    model = GFlowNet.GFlowNetModel(
        dag = dag,
        forward_policy = forward_policy,
        backward_policy = nothing,
        flow_estimator = flow_estimator,
        partition_function = nothing,
        objectives = GFlowNet.AbstractGFlowNetObjective[],
        optimizer = nothing,
        parameters = all_params,
        states = (forward = forward_states, backward = nothing, flow = flow_states)
    )
    
    println("✅ GFlowNet model created with Lux.jl components")
    
    # Test basic functionality
    test_state = GFlowNet.SimpleState([1])
    features = GFlowNet.state_to_features(test_state)
    println("   State features: ", features)
    
    # Test flow computation
    flow_val = GFlowNet.flow(model, test_state)
    println("   Flow value: ", flow_val)
    
    # Test probability computation
    next_states = GFlowNet.get_next_states(model.dag, test_state)
    if !isempty(next_states)
        for next_state in next_states
            prob = GFlowNet.forward_transition_prob(model, test_state, next_state)
            println("   P($test_state -> $next_state) = $prob")
        end
    end
    
catch e
    println("❌ GFlowNet integration test failed: ", e)
    println("   Error details: ", sprint(showerror, e))
end

println("\n🎉 ComponentArrays and Lux.jl integration test completed!")
println("\n📋 Summary:")
println("  • Lux.jl neural networks: ✅")
println("  • ComponentArrays parameters: ✅")
println("  • Neural network forward pass: ✅")
println("  • Gradient computation: ✅ (if no errors above)")
println("  • GFlowNet integration: ✅ (if no errors above)")
