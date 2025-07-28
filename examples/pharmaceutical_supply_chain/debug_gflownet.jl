"""
Debug script to test GFlowNet components individually
"""

using GFlowNet
using Random

# Include the GFlowNet interface
include("src/gflownet_interface.jl")

println("🔍 GFlowNet Debug Test")
println("="^50)

# Create a simple scenario
println("1️⃣ Creating pharmaceutical scenario...")
initial_state, actions = GFlowNet.create_pharmaceutical_scenario()

println("   ✅ Scenario created:")
println("      • Actions: $(length(actions))")
println("      • Initial state terminal: $(GFlowNet.is_terminal_state(initial_state))")

# Test state_to_features
println("\n2️⃣ Testing state_to_features...")
try
    features = GFlowNet.state_to_features(initial_state)
    println("   ✅ Features computed: $(length(features)) dimensions")
    println("   📊 Feature range: [$(minimum(features)), $(maximum(features))]")
    println("   🔍 Any NaN/Inf: $(any(isnan, features) || any(isinf, features))")
catch e
    println("   ❌ state_to_features failed: $e")
end

# Test reward function
println("\n3️⃣ Testing reward function...")
try
    reward_val = GFlowNet.reward(initial_state)
    println("   ✅ Initial reward: $reward_val")
    println("   🔍 Is finite: $(isfinite(reward_val))")
catch e
    println("   ❌ Reward function failed: $e")
end

# Test action applicability
println("\n4️⃣ Testing action applicability...")
applicable_count = 0
try
    for action in actions[1:min(10, length(actions))]
        if GFlowNet.is_applicable(action, initial_state)
            applicable_count += 1
        end
    end
    println("   ✅ Applicable actions (first 10): $applicable_count")
catch e
    println("   ❌ Action applicability failed: $e")
end

# Test state transitions
println("\n5️⃣ Testing state transitions...")
try
    applicable_actions = filter(a -> GFlowNet.is_applicable(a, initial_state), actions)
    if !isempty(applicable_actions)
        test_action = applicable_actions[1]
        new_state = GFlowNet.apply_action(test_action, initial_state)
        new_reward = GFlowNet.reward(new_state)
        
        println("   ✅ State transition successful")
        println("   📊 New state terminal: $(GFlowNet.is_terminal_state(new_state))")
        println("   💰 New reward: $new_reward")
        println("   🔍 Reward finite: $(isfinite(new_reward))")
    else
        println("   ⚠️ No applicable actions found")
    end
catch e
    println("   ❌ State transition failed: $e")
end

# Test terminal state creation
println("\n6️⃣ Testing terminal state creation...")
try
    # Find termination action
    termination_actions = filter(a -> isa(a, GFlowNet.TerminatePharmaceuticalAction), actions)
    if !isempty(termination_actions)
        terminal_state = GFlowNet.apply_action(termination_actions[1], initial_state)
        terminal_reward = GFlowNet.reward(terminal_state)
        
        println("   ✅ Terminal state created")
        println("   📊 Is terminal: $(GFlowNet.is_terminal_state(terminal_state))")
        println("   💰 Terminal reward: $terminal_reward")
        println("   🔍 Reward finite: $(isfinite(terminal_reward))")
        
        # Test features of terminal state
        terminal_features = GFlowNet.state_to_features(terminal_state)
        println("   📊 Terminal features: $(length(terminal_features)) dims")
        println("   🔍 Features finite: $(all(isfinite, terminal_features))")
    else
        println("   ⚠️ No termination actions found")
    end
catch e
    println("   ❌ Terminal state test failed: $e")
end

println("\n✅ Debug test completed!")
