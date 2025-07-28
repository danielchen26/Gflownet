"""
Test Systematic Fixes for Supply Chain GFlowNet Issues
"""

# Load GFlowNet from parent directory
push!(LOAD_PATH, joinpath(@__DIR__, "../../.."))
using GFlowNet
using Random

println("🔧 Testing Systematic Fixes")
println("="^40)

# Create minimal network
drugs = [GFlowNet.Drug(1, "Drug-A", GFlowNet.ONCOLOGY, GFlowNet.AMBIENT, 12, 10.0, 1.0)]
facilities = [
    GFlowNet.Facility(1, "Plant", GFlowNet.MANUFACTURING, (0.0, 0.0),
                     Dict(1=>100), Dict(1=>200), 1000.0, 1.0),
    GFlowNet.Facility(2, "DC", GFlowNet.DISTRIBUTION, (1.0, 1.0),
                     Dict{Int,Float64}(), Dict(1=>300), 500.0, 0.5)
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
    1, 3, false, 0.0, 0.0
)

# Create MINIMAL action set (like the fixed demo)
actions = GFlowNet.SupplyChainAction[]

# Only 2 production actions
push!(actions, GFlowNet.ProduceAction(1, 1, 50.0))  # 50% capacity
push!(actions, GFlowNet.ProduceAction(1, 1, 100.0)) # 100% capacity

# Only 1 shipment action
push!(actions, GFlowNet.ShipAction(1, 2, 1, 500.0))

# Only 1 serve action
push!(actions, GFlowNet.ServeAction(2, 1, 1, 50.0))

# Time actions
push!(actions, GFlowNet.NextMonthAction())
push!(actions, GFlowNet.FinishPlanningAction())

println("✅ Created minimal action set: $(length(actions)) actions")

# Test action applicability rates
println("\n🔍 Testing action applicability...")
current_state = initial_state

for step in 1:3
    applicable_count = 0
    for (i, action) in enumerate(actions)
        if GFlowNet.is_applicable(action, current_state)
            applicable_count += 1
            println("   Step $step: Action $i ($(typeof(action))) - ✅ Applicable")
        else
            println("   Step $step: Action $i ($(typeof(action))) - ❌ Not applicable")
        end
    end
    
    applicability_rate = applicable_count / length(actions)
    println("   📊 Step $step applicability: $(round(applicability_rate*100, digits=1))%")
    
    # Apply first applicable action to advance state
    for action in actions
        if GFlowNet.is_applicable(action, current_state)
            try
                current_state = GFlowNet.apply_action(action, current_state)
                println("   ✅ Applied $(typeof(action)) successfully")
                break
            catch e
                println("   ❌ Failed to apply $(typeof(action)): $e")
            end
        end
    end
    
    if current_state.is_terminal
        println("   🏁 Reached terminal state")
        break
    end
end

# Test trajectory sampling
println("\n🎯 Testing trajectory sampling...")
try
    model = GFlowNet.create_gflownet(
        initial_state,
        actions;
        state_dim = 13,
        hidden_dim = 32,
        learning_rate = 0.02
    )
    println("✅ Model created successfully")
    
    # Sample multiple trajectories
    successful_trajectories = 0
    total_attempts = 5
    
    for i in 1:total_attempts
        try
            config = GFlowNet.create_default_sampling_config()
            traj = GFlowNet.sample_trajectory(model; config=config)
            successful_trajectories += 1
            println("✅ Trajectory $i: $(length(traj.states)) states, reward $(GFlowNet.reward(traj.states[end]))")
        catch e
            println("❌ Trajectory $i failed: $e")
        end
    end
    
    success_rate = successful_trajectories / total_attempts
    println("📊 Trajectory success rate: $(round(success_rate*100, digits=1))%")
    
    if success_rate >= 0.8
        println("🎯 SUCCESS: High trajectory success rate!")
        
        # Test minimal training
        println("\n🚀 Testing minimal training...")
        try
            train_config = GFlowNet.TrainingConfig(
                objective=GFlowNet.TRAJECTORY_BALANCE,
                partition_function_method=GFlowNet.SIMPLE_ESTIMATION,
                n_iterations=3,
                batch_size=2,
                learning_rate=0.02,
                validation_frequency=1,
                early_stopping_patience=5
            )
            
            history = GFlowNet.train_gflownet(model, train_config; verbose=true)
            
            if length(history.losses) >= 2
                loss_change = history.losses[1] - history.losses[end]
                println("📈 Loss change: $(round(loss_change, digits=4))")
                
                if loss_change > 0
                    println("🎯 SUCCESS: Model is learning! (loss decreased)")
                    return true
                else
                    println("⚠️ WARNING: Loss did not decrease")
                end
            end
            
        catch e
            println("❌ Training failed: $e")
        end
        
    else
        println("❌ FAILURE: Low trajectory success rate")
    end
    
catch e
    println("❌ Model creation failed: $e")
end

println("\n" * "="^40)
println("🔧 Fix verification completed")
