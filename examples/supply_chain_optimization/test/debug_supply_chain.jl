"""
Debug Supply Chain Implementation
"""

# Load GFlowNet from parent directory
push!(LOAD_PATH, joinpath(@__DIR__, "../.."))
using GFlowNet
using Random

println("🔍 Supply Chain Debug Test")
println("="^40)

# Create simple network
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

# Create initial state
initial_state = GFlowNet.SupplyChainState(
    network,
    Dict{Tuple{Int,Int}, Float64}(),
    Dict{Tuple{Int,Int}, Float64}(),
    Dict{Tuple{Int,Int,Int}, Float64}(),
    Dict{Tuple{Int,Int}, Float64}(),
    1, 2, false, 0.0, 0.0
)

# Create simple actions
actions = [
    GFlowNet.ProduceAction(1, 1, 50.0),
    GFlowNet.ShipAction(1, 2, 1, 25.0),
    GFlowNet.ServeAction(2, 1, 1, 25.0),
    GFlowNet.FinishPlanningAction()
]

println("✅ Simple network created")
println("   • Drugs: $(length(drugs))")
println("   • Facilities: $(length(facilities))")
println("   • Actions: $(length(actions))")

# Test basic functions
println("\n🧪 Testing basic functions...")

try
    features = GFlowNet.state_to_features(initial_state)
    println("✅ state_to_features: $(length(features)) dims")
catch e
    println("❌ state_to_features failed: $e")
end

try
    reward_val = GFlowNet.reward(initial_state)
    println("✅ reward: $reward_val")
catch e
    println("❌ reward failed: $e")
end

try
    applicable = [GFlowNet.is_applicable(a, initial_state) for a in actions]
    println("✅ is_applicable: $(sum(applicable))/$(length(actions)) applicable")
catch e
    println("❌ is_applicable failed: $e")
end

try
    # Test state transition
    produce_action = GFlowNet.ProduceAction(1, 1, 50.0)
    if GFlowNet.is_applicable(produce_action, initial_state)
        new_state = GFlowNet.apply_action(produce_action, initial_state)
        println("✅ apply_action: transition successful")
        
        # Test features of new state
        new_features = GFlowNet.state_to_features(new_state)
        println("✅ new state features: $(length(new_features)) dims")
    else
        println("⚠️ produce action not applicable")
    end
catch e
    println("❌ apply_action failed: $e")
end

# Test GFlowNet model creation
println("\n🤖 Testing GFlowNet model creation...")

try
    sample_features = GFlowNet.state_to_features(initial_state)
    state_dim = length(sample_features)
    
    model = GFlowNet.create_gflownet(
        initial_state,
        actions;
        state_dim = state_dim,
        hidden_dim = 32,
        learning_rate = 0.01
    )
    println("✅ GFlowNet model created successfully")
    
    # Test trajectory sampling
    try
        config = GFlowNet.create_default_sampling_config()
        traj = GFlowNet.sample_trajectory(model; config=config)
        println("✅ Trajectory sampling successful")
        println("   • States: $(length(traj.states))")
        println("   • Actions: $(length(traj.actions))")
        println("   • Final reward: $(GFlowNet.reward(traj.states[end]))")
    catch e
        println("❌ Trajectory sampling failed: $e")
    end
    
catch e
    println("❌ GFlowNet model creation failed: $e")
end

println("\n✅ Debug test completed!")
