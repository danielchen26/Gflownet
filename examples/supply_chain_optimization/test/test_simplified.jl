"""
Test Simplified Supply Chain Implementation
"""

# Load GFlowNet from parent directory
push!(LOAD_PATH, joinpath(@__DIR__, "../../.."))
using GFlowNet
using Random

println("🧪 Testing Simplified Supply Chain Implementation")
println("="^50)

# Create minimal network
drugs = [GFlowNet.Drug(1, "Drug-A", GFlowNet.ONCOLOGY, GFlowNet.AMBIENT, 12, 10.0, 1.0)]
facilities = [
    GFlowNet.Facility(1, "Plant", GFlowNet.MANUFACTURING, (0.0, 0.0),
                     Dict(1=>100), Dict(1=>50), 1000.0, 1.0),
    GFlowNet.Facility(2, "DC", GFlowNet.DISTRIBUTION, (1.0, 1.0),
                     Dict{Int,Float64}(), Dict(1=>100), 500.0, 0.5)
]
regions = [GFlowNet.PatientRegion(1, "Region-1", (2.0, 2.0), Dict(1=>50), 0.95)]
routes = [GFlowNet.TransportRoute(1, 2, 100.0, 1.0, 1, Dict(GFlowNet.AMBIENT=>1.0))]
network = GFlowNet.SupplyChainNetwork(drugs, facilities, regions, routes)

# Create simplified initial state
initial_state = GFlowNet.SupplyChainState(
    network,
    [0.0, 0.0],      # production_levels (2 facilities)
    [0.0, 0.0],      # inventory_levels (2 facilities)
    [0.0],           # demand_satisfaction (1 region)
    1, 5, false,     # current_step, max_steps, is_terminal
    0.0, 0.0         # total_cost, service_level
)

# Create simplified actions
actions = [
    GFlowNet.IncreaseProduction(1, 50.0),
    GFlowNet.IncreaseProduction(2, 30.0),
    GFlowNet.DistributeInventory(1, 25.0),
    GFlowNet.NextStep(),
    GFlowNet.TerminatePlanning()
]

println("✅ Simplified network and state created")

# Test basic functions
println("\n🔍 Testing GFlowNet interface functions...")

# Test state features
try
    features = GFlowNet.state_to_features(initial_state)
    println("✅ state_to_features: $(length(features)) dimensions")
    println("   Features: $(round.(features, digits=3))")
catch e
    println("❌ state_to_features failed: $e")
end

# Test reward
try
    reward_val = GFlowNet.reward(initial_state)
    println("✅ reward: $reward_val")
catch e
    println("❌ reward failed: $e")
end

# Test action applicability
try
    applicable = [GFlowNet.is_applicable(a, initial_state) for a in actions]
    println("✅ is_applicable: $(sum(applicable))/$(length(actions)) applicable")
    println("   Applicable actions: $applicable")
catch e
    println("❌ is_applicable failed: $e")
end

# Test state transitions
println("\n🔄 Testing state transitions...")
current_state = initial_state

for (i, action) in enumerate(actions)
    if GFlowNet.is_applicable(action, current_state)
        try
            new_state = GFlowNet.apply_action(action, current_state)
            println("✅ Action $i applied successfully")
            println("   Production: $(new_state.production_levels)")
            println("   Inventory: $(new_state.inventory_levels)")
            println("   Demand: $(new_state.demand_satisfaction)")
            println("   Step: $(new_state.current_step), Terminal: $(new_state.is_terminal)")
            current_state = new_state
        catch e
            println("❌ Action $i failed: $e")
        end
    else
        println("⚠️ Action $i not applicable")
    end
end

# Test GFlowNet model creation
println("\n🤖 Testing GFlowNet integration...")
try
    model = GFlowNet.create_gflownet(
        initial_state,
        actions;
        state_dim = 15,
        hidden_dim = 32,
        learning_rate = 0.01
    )
    println("✅ GFlowNet model created successfully")
    
    # Test trajectory sampling
    try
        config = GFlowNet.create_default_sampling_config()
        traj = GFlowNet.sample_trajectory(model; config=config)
        println("✅ Trajectory sampling successful")
        println("   States: $(length(traj.states))")
        println("   Actions: $(length(traj.actions))")
        println("   Final reward: $(GFlowNet.reward(traj.states[end]))")
        
        # Test minimal training
        println("\n🚀 Testing minimal training...")
        try
            train_config = GFlowNet.TrainingConfig(
                objective=GFlowNet.TRAJECTORY_BALANCE,
                partition_function_method=GFlowNet.SIMPLE_ESTIMATION,
                n_iterations=3,
                batch_size=2,
                learning_rate=0.01,
                validation_frequency=1,
                early_stopping_patience=5
            )
            
            history = GFlowNet.train_gflownet(model, train_config; verbose=true)
            println("✅ Training completed successfully!")
            println("   Final loss: $(history.losses[end])")
            println("   Training time: $(sum(history.iteration_times))s")
            
        catch e
            println("❌ Training failed: $e")
        end
        
    catch e
        println("❌ Trajectory sampling failed: $e")
    end
    
catch e
    println("❌ GFlowNet model creation failed: $e")
end

println("\n" * "="^50)
println("🎯 Simplified supply chain test completed!")
println("✅ Ready for full implementation testing")
