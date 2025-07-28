"""
Comprehensive Domain Compatibility Test for Supply Chain GFlowNet
"""

# Load GFlowNet from parent directory
push!(LOAD_PATH, joinpath(@__DIR__, "../../.."))
using GFlowNet
using Random

println("🔍 Comprehensive Domain Compatibility Test")
println("="^50)

# Create the same network as the main demo
drugs = [
    GFlowNet.Drug(1, "Oncology-A", GFlowNet.ONCOLOGY, GFlowNet.COLD, 6, 50.0, 2.0),
    GFlowNet.Drug(2, "Vaccine-B", GFlowNet.VACCINES, GFlowNet.COLD, 3, 30.0, 1.5),
    GFlowNet.Drug(3, "Generic-C", GFlowNet.GENERICS, GFlowNet.AMBIENT, 12, 15.0, 0.8),
    GFlowNet.Drug(4, "Biologic-D", GFlowNet.BIOLOGICS, GFlowNet.FROZEN, 1, 100.0, 5.0)
]

facilities = [
    GFlowNet.Facility(1, "Plant-US", GFlowNet.MANUFACTURING, (40.7, -74.0),
                     Dict(1=>200, 2=>500, 3=>1000, 4=>100), Dict(1=>300, 2=>600, 3=>2000, 4=>150), 5000.0, 2.0),
    GFlowNet.Facility(2, "Plant-EU", GFlowNet.MANUFACTURING, (52.5, 13.4),
                     Dict(1=>150, 2=>400, 3=>800, 4=>80), Dict(1=>200, 2=>500, 3=>1500, 4=>120), 4000.0, 1.8),
    GFlowNet.Facility(3, "DC-East", GFlowNet.DISTRIBUTION, (39.9, -75.2),
                     Dict{Int,Float64}(), Dict(1=>500, 2=>800, 3=>3000, 4=>200), 3000.0, 1.2),
    GFlowNet.Facility(4, "DC-West", GFlowNet.DISTRIBUTION, (34.1, -118.2),
                     Dict{Int,Float64}(), Dict(1=>400, 2=>700, 3=>2500, 4=>180), 2500.0, 1.1),
    GFlowNet.Facility(5, "Depot-Central", GFlowNet.DEPOT, (41.9, -87.6),
                     Dict{Int,Float64}(), Dict(1=>200, 2=>300, 3=>1000, 4=>100), 1500.0, 0.8)
]

regions = [
    GFlowNet.PatientRegion(1, "US-Northeast", (40.7, -74.0), Dict(1=>100, 2=>200, 3=>500, 4=>50), 0.98),
    GFlowNet.PatientRegion(2, "US-West", (34.1, -118.2), Dict(1=>80, 2=>150, 3=>400, 4=>40), 0.96),
    GFlowNet.PatientRegion(3, "EU-Central", (52.5, 13.4), Dict(1=>120, 2=>180, 3=>450, 4=>60), 0.97)
]

routes = [
    GFlowNet.TransportRoute(1, 3, 500.0, 2.0, 1, Dict(GFlowNet.COLD=>0.8, GFlowNet.AMBIENT=>1.0, GFlowNet.FROZEN=>0.6)),
    GFlowNet.TransportRoute(1, 4, 800.0, 3.0, 2, Dict(GFlowNet.COLD=>0.8, GFlowNet.AMBIENT=>1.0, GFlowNet.FROZEN=>0.6)),
    GFlowNet.TransportRoute(2, 3, 600.0, 2.5, 1, Dict(GFlowNet.COLD=>0.8, GFlowNet.AMBIENT=>1.0, GFlowNet.FROZEN=>0.6)),
    GFlowNet.TransportRoute(2, 5, 400.0, 1.5, 1, Dict(GFlowNet.COLD=>0.8, GFlowNet.AMBIENT=>1.0, GFlowNet.FROZEN=>0.6)),
    GFlowNet.TransportRoute(3, 5, 300.0, 1.0, 1, Dict(GFlowNet.COLD=>0.8, GFlowNet.AMBIENT=>1.0, GFlowNet.FROZEN=>0.6))
]

network = GFlowNet.SupplyChainNetwork(drugs, facilities, regions, routes)

# Create initial state
initial_state = GFlowNet.SupplyChainState(
    network,
    Dict{Tuple{Int,Int}, Float64}(),
    Dict{Tuple{Int,Int}, Float64}(),
    Dict{Tuple{Int,Int,Int}, Float64}(),
    Dict{Tuple{Int,Int}, Float64}(),
    1, 3, false, 0.0, 0.0
)

println("✅ Network created: $(length(drugs)) drugs, $(length(facilities)) facilities")

# Test 1: State Features Compatibility
println("\n1️⃣ Testing State Features...")
try
    features = GFlowNet.state_to_features(initial_state)
    println("✅ Features: $(length(features)) dimensions")
    println("   Range: [$(minimum(features)), $(maximum(features))]")
    println("   All finite: $(all(isfinite, features))")
    println("   No NaN: $(all(!isnan, features))")
    
    if length(features) != 13
        println("❌ ERROR: Expected 13 features, got $(length(features))")
    end
catch e
    println("❌ State features failed: $e")
end

# Test 2: Reward Function Stability
println("\n2️⃣ Testing Reward Function...")
try
    # Test non-terminal state
    non_terminal_reward = GFlowNet.reward(initial_state)
    println("✅ Non-terminal reward: $non_terminal_reward")
    
    # Test terminal state
    terminal_state = GFlowNet.SupplyChainState(
        network, Dict{Tuple{Int,Int}, Float64}(), Dict{Tuple{Int,Int}, Float64}(),
        Dict{Tuple{Int,Int,Int}, Float64}(), Dict{Tuple{Int,Int}, Float64}(),
        3, 3, true, 50000.0, 0.8
    )
    terminal_reward = GFlowNet.reward(terminal_state)
    println("✅ Terminal reward: $terminal_reward")
    
    # Test reward stability with different costs
    rewards = []
    for cost in [0.0, 10000.0, 50000.0, 100000.0, 500000.0]
        test_state = GFlowNet.SupplyChainState(
            network, Dict{Tuple{Int,Int}, Float64}(), Dict{Tuple{Int,Int}, Float64}(),
            Dict{Tuple{Int,Int,Int}, Float64}(), Dict{Tuple{Int,Int}, Float64}(),
            3, 3, true, cost, 0.5
        )
        reward = GFlowNet.reward(test_state)
        push!(rewards, reward)
    end
    
    reward_range = maximum(rewards) - minimum(rewards)
    println("✅ Reward stability test:")
    println("   Rewards: $(round.(rewards, digits=2))")
    println("   Range: $(round(reward_range, digits=2))")
    
    if reward_range > 20.0
        println("⚠️ WARNING: Large reward range may cause training instability")
    end
    
catch e
    println("❌ Reward function failed: $e")
end

# Test 3: Action Generation and Applicability
println("\n3️⃣ Testing Action Space...")

# Generate actions (same as demo)
actions = GFlowNet.SupplyChainAction[]

# Production actions
for facility in facilities
    if facility.type == GFlowNet.MANUFACTURING
        for (drug_id, capacity) in facility.production_capacity
            for pct in [0.5, 1.0]
                quantity = capacity * pct
                push!(actions, GFlowNet.ProduceAction(facility.id, drug_id, quantity))
            end
        end
    end
end

# Shipment actions
for route in routes
    for drug in drugs
        push!(actions, GFlowNet.ShipAction(route.from_facility, route.to_facility, drug.id, 500.0))
    end
end

# Serve actions
for facility in facilities
    if facility.type in [GFlowNet.DISTRIBUTION, GFlowNet.DEPOT]
        for region in regions
            for (drug_id, demand) in region.monthly_demand
                quantity = demand
                push!(actions, GFlowNet.ServeAction(facility.id, region.id, drug_id, quantity))
            end
        end
    end
end

push!(actions, GFlowNet.NextMonthAction())
push!(actions, GFlowNet.FinishPlanningAction())

println("✅ Generated $(length(actions)) actions")

# Test action applicability
applicable_count = 0
for action in actions
    if GFlowNet.is_applicable(action, initial_state)
        applicable_count += 1
    end
end

applicability_rate = applicable_count / length(actions)
println("✅ Applicability: $(applicable_count)/$(length(actions)) ($(round(applicability_rate*100, digits=1))%)")

if applicability_rate < 0.1
    println("❌ ERROR: Very low applicability rate - action space may be incompatible")
elseif applicability_rate < 0.3
    println("⚠️ WARNING: Low applicability rate - may cause trajectory failures")
end

# Test 4: State Transitions
println("\n4️⃣ Testing State Transitions...")
test_state = initial_state
transition_count = 0

for action in actions[1:min(10, length(actions))]
    if GFlowNet.is_applicable(action, test_state)
        try
            new_state = GFlowNet.apply_action(action, test_state)
            transition_count += 1
            println("✅ Transition $(transition_count): $(typeof(action)) successful")
            
            # Verify state integrity
            if new_state.total_cost < 0
                println("❌ ERROR: Negative cost in new state")
            end
            if new_state.service_level < 0 || new_state.service_level > 1
                println("❌ ERROR: Invalid service level: $(new_state.service_level)")
            end
            
            test_state = new_state
        catch e
            println("❌ Transition failed: $(typeof(action)) - $e")
        end
    end
end

println("✅ Successful transitions: $transition_count")

# Test 5: GFlowNet Model Creation
println("\n5️⃣ Testing GFlowNet Integration...")
try
    model = GFlowNet.create_gflownet(
        initial_state,
        actions;
        state_dim = 13,
        hidden_dim = 16,  # Very small for testing
        learning_rate = 0.001
    )
    println("✅ Model created successfully")
    
    # Test single trajectory
    try
        config = GFlowNet.create_default_sampling_config()
        traj = GFlowNet.sample_trajectory(model; config=config)
        println("✅ Trajectory sampling: $(length(traj.states)) states, reward $(GFlowNet.reward(traj.states[end]))")
        
        # Test minimal training
        try
            train_config = GFlowNet.TrainingConfig(
                objective=GFlowNet.TRAJECTORY_BALANCE,
                partition_function_method=GFlowNet.SIMPLE_ESTIMATION,
                n_iterations=2,
                batch_size=1,
                learning_rate=0.001,
                validation_frequency=1,
                early_stopping_patience=3
            )
            
            history = GFlowNet.train_gflownet(model, train_config; verbose=false)
            println("✅ Training test: $(length(history.losses)) iterations")
            
            if length(history.losses) >= 2
                loss_change = history.losses[1] - history.losses[end]
                println("   Loss change: $(round(loss_change, digits=4))")
                
                if abs(loss_change) > 1000
                    println("⚠️ WARNING: Large loss changes indicate instability")
                end
            end
            
        catch e
            println("❌ Training test failed: $e")
        end
        
    catch e
        println("❌ Trajectory sampling failed: $e")
    end
    
catch e
    println("❌ Model creation failed: $e")
end

println("\n" * "="^50)
println("🔍 Domain compatibility test completed")
println("✅ Check all results above for compatibility issues")
