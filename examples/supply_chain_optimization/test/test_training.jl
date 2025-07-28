"""
Minimal Training Test to Isolate GFlowNet Training Issues
"""

# Load GFlowNet from parent directory
push!(LOAD_PATH, joinpath(@__DIR__, "../../.."))
using GFlowNet
using Random

println("🔍 Minimal GFlowNet Training Test")
println("="^40)

# Create minimal network (same as debug test)
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

# Create minimal actions
actions = [
    GFlowNet.ProduceAction(1, 1, 50.0),
    GFlowNet.ShipAction(1, 2, 1, 25.0),
    GFlowNet.ServeAction(2, 1, 1, 25.0),
    GFlowNet.FinishPlanningAction()
]

println("✅ Network and actions created")

# Test model creation
println("\n🤖 Testing GFlowNet model creation...")
try
    model = GFlowNet.create_gflownet(
        initial_state,
        actions;
        state_dim = 13,
        hidden_dim = 32,
        learning_rate = 0.01
    )
    println("✅ Model created successfully")
    
    # Test single trajectory sampling
    println("\n🎯 Testing single trajectory sampling...")
    try
        config = GFlowNet.create_default_sampling_config()
        traj = GFlowNet.sample_trajectory(model; config=config)
        println("✅ Single trajectory sampling works")
        println("   • States: $(length(traj.states))")
        println("   • Actions: $(length(traj.actions))")
        println("   • Final reward: $(GFlowNet.reward(traj.states[end]))")
        
        # Test learning with multiple iterations
        println("\n🚀 Testing learning with multiple iterations...")
        try
            train_config = GFlowNet.TrainingConfig(
                objective=GFlowNet.TRAJECTORY_BALANCE,
                partition_function_method=GFlowNet.SIMPLE_ESTIMATION,
                n_iterations=5,  # Multiple iterations to see learning
                batch_size=4,    # Small batch
                learning_rate=0.01,
                validation_frequency=2,
                early_stopping_patience=10
            )
            
            println("   📋 Training config created")
            println("   🔧 Starting 5-iteration training to observe learning...")
            
            # This is where the error likely occurs
            history = GFlowNet.train_gflownet(model, train_config; verbose=true)
            
            println("✅ Training completed successfully!")
            println("   📊 Training history received")
            println("   📈 Final loss: $(history.losses[end])")
            println("   🔧 Final gradient norm: $(history.gradient_norms[end])")
            println("   ⏱️  Training time: $(sum(history.iteration_times))s")
            
        catch e
            println("❌ Training failed with error:")
            println("   Error type: $(typeof(e))")
            println("   Error message: $e")
            
            # Print stack trace for debugging
            println("\n📋 Stack trace:")
            for (i, frame) in enumerate(stacktrace(catch_backtrace()))
                println("   $i. $frame")
                if i > 10  # Limit stack trace length
                    break
                end
            end
        end
        
    catch e
        println("❌ Trajectory sampling failed: $e")
    end
    
catch e
    println("❌ Model creation failed: $e")
end

println("\n" * "="^40)
println("🔍 Training test completed")
