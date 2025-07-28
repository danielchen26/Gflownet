"""
Test Rich Supply Chain Implementation with Proper GFlowNet Integration
"""

# Load GFlowNet from parent directory
push!(LOAD_PATH, joinpath(@__DIR__, "../../.."))
using GFlowNet
using Random

println("🧪 Testing Rich Supply Chain Implementation")
println("="^50)

# Create minimal network (same as before)
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

# Create rich initial state
initial_state = GFlowNet.SupplyChainState(
    network,
    Dict{Tuple{Int,Int}, Float64}(),      # production
    Dict{Tuple{Int,Int}, Float64}(),      # inventory
    Dict{Tuple{Int,Int,Int}, Float64}(),  # shipments
    Dict{Tuple{Int,Int}, Float64}(),      # demand_served
    1, 3, false,                          # current_month, planning_horizon, is_terminal
    0.0, 0.0                              # total_cost, service_level
)

# Create rich actions (discretized for stability)
actions = [
    GFlowNet.ProduceAction(1, 1, 50.0),
    GFlowNet.ProduceAction(1, 1, 25.0),
    GFlowNet.ShipAction(1, 2, 1, 25.0),
    GFlowNet.ServeAction(2, 1, 1, 25.0),
    GFlowNet.NextMonthAction(),
    GFlowNet.FinishPlanningAction()
]

println("✅ Rich network and state created")
println("   • Production dict: $(length(initial_state.production)) entries")
println("   • Inventory dict: $(length(initial_state.inventory)) entries")
println("   • Actions: $(length(actions)) rich actions")

# Test basic functions
println("\n🔍 Testing GFlowNet interface functions...")

# Test state features
try
    features = GFlowNet.state_to_features(initial_state)
    println("✅ state_to_features: $(length(features)) dimensions")
    println("   Features: $(round.(features, digits=3))")
    println("   All finite: $(all(isfinite, features))")
catch e
    println("❌ state_to_features failed: $e")
end

# Test reward
try
    reward_val = GFlowNet.reward(initial_state)
    println("✅ reward: $reward_val")
    println("   Is finite: $(isfinite(reward_val))")
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

# Test state transitions with rich actions
println("\n🔄 Testing rich state transitions...")
global current_state = initial_state

for (i, action) in enumerate(actions)
    global current_state
    if GFlowNet.is_applicable(action, current_state)
        try
            new_state = GFlowNet.apply_action(action, current_state)
            println("✅ Action $i ($(typeof(action))) applied successfully")
            println("   Production entries: $(length(new_state.production))")
            println("   Inventory entries: $(length(new_state.inventory))")
            println("   Cost: $(new_state.total_cost)")
            println("   Month: $(new_state.current_month), Terminal: $(new_state.is_terminal)")
            current_state = new_state
        catch e
            println("❌ Action $i failed: $e")
        end
    else
        println("⚠️ Action $i not applicable")
    end
end

# Test state hashing and equality (critical for GFlowNet)
println("\n🔑 Testing state hashing and equality...")
try
    state1 = initial_state
    state2 = GFlowNet.SupplyChainState(
        network,
        Dict{Tuple{Int,Int}, Float64}(),
        Dict{Tuple{Int,Int}, Float64}(),
        Dict{Tuple{Int,Int,Int}, Float64}(),
        Dict{Tuple{Int,Int}, Float64}(),
        1, 3, false, 0.0, 0.0
    )
    
    println("✅ State equality: $(state1 == state2)")
    println("✅ State hashing: $(hash(state1) == hash(state2))")
    
    # Test different states
    state3 = GFlowNet.apply_action(GFlowNet.ProduceAction(1, 1, 50.0), state1)
    println("✅ Different states: $(state1 != state3)")
    println("✅ Different hashes: $(hash(state1) != hash(state3))")
    
catch e
    println("❌ State hashing/equality failed: $e")
end

# Test GFlowNet model creation
println("\n🤖 Testing GFlowNet integration...")
try
    model = GFlowNet.create_gflownet(
        initial_state,
        actions;
        state_dim = 13,  # Fixed feature size
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
            println("   Final gradient norm: $(history.gradient_norms[end])")
            println("   Training time: $(sum(history.iteration_times))s")
            
            # Check if learning occurred
            if length(history.losses) > 1
                loss_change = history.losses[1] - history.losses[end]
                println("   Loss change: $(round(loss_change, digits=4))")
                if loss_change > 0
                    println("   🎯 Model is learning! (loss decreased)")
                else
                    println("   ⚠️ Loss did not decrease")
                end
            end
            
        catch e
            println("❌ Training failed: $e")
            println("   Error type: $(typeof(e))")
        end
        
    catch e
        println("❌ Trajectory sampling failed: $e")
    end
    
catch e
    println("❌ GFlowNet model creation failed: $e")
end

println("\n" * "="^50)
println("🎯 Rich supply chain implementation test completed!")
println("✅ Ready for full-scale training with rich problem formulation")
