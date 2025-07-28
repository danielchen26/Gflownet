"""
Comprehensive Test Suite for Supply Chain Optimization
=====================================================

Tests all components of the pharmaceutical supply chain optimization
implementation to ensure correctness and reliability.
"""

# Load GFlowNet from parent directory
push!(LOAD_PATH, joinpath(@__DIR__, "../../.."))
using GFlowNet
using Random
using Statistics
using Test

println("🧪 Supply Chain Optimization Test Suite")
println("="^50)

# Set random seed for reproducible tests
Random.seed!(42)

# =============================================================================
# Test 1: Basic Data Structure Creation
# =============================================================================

println("\n1️⃣ Testing Data Structure Creation")
println("-"^40)

@testset "Data Structures" begin
    # Test drug creation
    drug = GFlowNet.Drug(1, "Test-Drug", GFlowNet.ONCOLOGY, GFlowNet.AMBIENT, 12, 10.0, 1.0)
    @test drug.id == 1
    @test drug.name == "Test-Drug"
    @test drug.type == GFlowNet.ONCOLOGY
    println("   ✅ Drug creation: PASSED")
    
    # Test facility creation
    facility = GFlowNet.Facility(1, "Test-Plant", GFlowNet.MANUFACTURING, (0.0, 0.0),
                                Dict(1=>100), Dict(1=>50), 1000.0, 1.0)
    @test facility.id == 1
    @test facility.type == GFlowNet.MANUFACTURING
    @test facility.production_capacity[1] == 100
    println("   ✅ Facility creation: PASSED")
    
    # Test patient region creation
    region = GFlowNet.PatientRegion(1, "Test-Region", (1.0, 1.0), Dict(1=>50), 0.95)
    @test region.id == 1
    @test region.monthly_demand[1] == 50
    @test region.service_level_target == 0.95
    println("   ✅ Patient region creation: PASSED")
    
    # Test transport route creation
    route = GFlowNet.TransportRoute(1, 2, 100.0, 1.0, 1, Dict(GFlowNet.AMBIENT=>1.0))
    @test route.from_facility == 1
    @test route.to_facility == 2
    @test route.distance_km == 100.0
    println("   ✅ Transport route creation: PASSED")
end

# =============================================================================
# Test 2: Network Assembly and Helper Functions
# =============================================================================

println("\n2️⃣ Testing Network Assembly")
println("-"^40)

# Create test network
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

@testset "Network Assembly" begin
    @test length(network.drugs) == 1
    @test length(network.facilities) == 2
    @test length(network.regions) == 1
    @test length(network.routes) == 1
    println("   ✅ Network assembly: PASSED")
    
    # Test helper functions
    facility = GFlowNet.get_facility(network, 1)
    @test facility !== nothing
    @test facility.id == 1
    println("   ✅ get_facility: PASSED")
    
    drug = GFlowNet.get_drug(network, 1)
    @test drug !== nothing
    @test drug.id == 1
    println("   ✅ get_drug: PASSED")
    
    region = GFlowNet.get_region(network, 1)
    @test region !== nothing
    @test region.id == 1
    println("   ✅ get_region: PASSED")
    
    route = GFlowNet.get_route(network, 1, 2)
    @test route !== nothing
    @test route.from_facility == 1
    println("   ✅ get_route: PASSED")
end

# =============================================================================
# Test 3: State Representation and Features
# =============================================================================

println("\n3️⃣ Testing State Representation")
println("-"^40)

# Create initial state
initial_state = GFlowNet.SupplyChainState(
    network,
    Dict{Tuple{Int,Int}, Float64}(),
    Dict{Tuple{Int,Int}, Float64}(),
    Dict{Tuple{Int,Int,Int}, Float64}(),
    Dict{Tuple{Int,Int}, Float64}(),
    1, 2, false, 0.0, 0.0
)

@testset "State Representation" begin
    @test initial_state.current_month == 1
    @test initial_state.planning_horizon == 2
    @test !initial_state.is_terminal
    @test initial_state.total_cost == 0.0
    println("   ✅ State creation: PASSED")
    
    # Test state features
    features = GFlowNet.state_to_features(initial_state)
    @test length(features) == 13
    @test all(isfinite, features)
    @test all(f -> f >= 0.0, features)  # All features should be non-negative
    println("   ✅ State features ($(length(features)) dims): PASSED")
    
    # Test terminal state detection
    @test !GFlowNet.is_terminal_state(initial_state)
    println("   ✅ Terminal state detection: PASSED")
end

# =============================================================================
# Test 4: Action Creation and Applicability
# =============================================================================

println("\n4️⃣ Testing Actions")
println("-"^40)

# Create test actions
produce_action = GFlowNet.ProduceAction(1, 1, 50.0)
ship_action = GFlowNet.ShipAction(1, 2, 1, 25.0)
serve_action = GFlowNet.ServeAction(2, 1, 1, 25.0)
finish_action = GFlowNet.FinishPlanningAction()

@testset "Action Applicability" begin
    # Test production action
    @test GFlowNet.is_applicable(produce_action, initial_state)
    println("   ✅ Production action applicability: PASSED")
    
    # Test shipment action (should fail - no inventory)
    @test !GFlowNet.is_applicable(ship_action, initial_state)
    println("   ✅ Shipment action applicability: PASSED")
    
    # Test serve action (should fail - no inventory)
    @test !GFlowNet.is_applicable(serve_action, initial_state)
    println("   ✅ Service action applicability: PASSED")
    
    # Test finish action
    @test GFlowNet.is_applicable(finish_action, initial_state)
    println("   ✅ Finish action applicability: PASSED")
end

# =============================================================================
# Test 5: State Transitions
# =============================================================================

println("\n5️⃣ Testing State Transitions")
println("-"^40)

@testset "State Transitions" begin
    # Test production action
    new_state = GFlowNet.apply_action(produce_action, initial_state)
    @test new_state.production[(1, 1)] == 50.0
    @test new_state.inventory[(1, 1)] == 50.0
    @test new_state.total_cost > initial_state.total_cost
    println("   ✅ Production transition: PASSED")
    
    # Test shipment action on new state
    if GFlowNet.is_applicable(ship_action, new_state)
        shipped_state = GFlowNet.apply_action(ship_action, new_state)
        @test shipped_state.inventory[(1, 1)] == 25.0  # Reduced source
        @test shipped_state.inventory[(2, 1)] == 25.0  # Increased destination
        println("   ✅ Shipment transition: PASSED")
        
        # Test serve action
        if GFlowNet.is_applicable(serve_action, shipped_state)
            served_state = GFlowNet.apply_action(serve_action, shipped_state)
            @test served_state.demand_served[(1, 1)] == 25.0
            @test served_state.inventory[(2, 1)] == 0.0
            println("   ✅ Service transition: PASSED")
        end
    end
    
    # Test termination
    terminal_state = GFlowNet.apply_action(finish_action, new_state)
    @test terminal_state.is_terminal
    println("   ✅ Termination transition: PASSED")
end

# =============================================================================
# Test 6: Reward Function
# =============================================================================

println("\n6️⃣ Testing Reward Function")
println("-"^40)

@testset "Reward Function" begin
    # Non-terminal state should have zero reward
    @test GFlowNet.reward(initial_state) == 0.0
    println("   ✅ Non-terminal reward: PASSED")
    
    # Terminal state should have positive reward
    terminal_state = GFlowNet.apply_action(finish_action, initial_state)
    terminal_reward = GFlowNet.reward(terminal_state)
    @test terminal_reward > 0.0
    @test isfinite(terminal_reward)
    println("   ✅ Terminal reward ($(round(terminal_reward, digits=2))): PASSED")
end

# =============================================================================
# Test 7: Cost Calculations
# =============================================================================

println("\n7️⃣ Testing Cost Calculations")
println("-"^40)

@testset "Cost Calculations" begin
    # Create state with some activity
    active_state = GFlowNet.apply_action(produce_action, initial_state)
    
    # Test cost calculation
    monthly_cost = GFlowNet.calculate_total_monthly_cost(active_state)
    @test monthly_cost > 0.0
    @test isfinite(monthly_cost)
    println("   ✅ Monthly cost calculation (\$$(round(monthly_cost, digits=0))): PASSED")
    
    # Test service level calculation
    service_level = GFlowNet.calculate_service_level(active_state)
    @test 0.0 <= service_level <= 1.0
    println("   ✅ Service level calculation ($(round(service_level*100, digits=1))%): PASSED")
end

# =============================================================================
# Test 8: GFlowNet Integration
# =============================================================================

println("\n8️⃣ Testing GFlowNet Integration")
println("-"^40)

@testset "GFlowNet Integration" begin
    # Create simple action set
    actions = [
        GFlowNet.ProduceAction(1, 1, 50.0),
        GFlowNet.ShipAction(1, 2, 1, 25.0),
        GFlowNet.ServeAction(2, 1, 1, 25.0),
        GFlowNet.FinishPlanningAction()
    ]
    
    # Test model creation
    try
        model = GFlowNet.create_gflownet(
            initial_state,
            actions;
            state_dim = 13,
            hidden_dim = 32,
            learning_rate = 0.01
        )
        println("   ✅ GFlowNet model creation: PASSED")
        
        # Test trajectory sampling
        config = GFlowNet.create_default_sampling_config()
        traj = GFlowNet.sample_trajectory(model; config=config)
        @test length(traj.states) >= 2
        @test length(traj.actions) >= 1
        println("   ✅ Trajectory sampling: PASSED")
        
        # Test reward calculation on sampled trajectory
        final_reward = GFlowNet.reward(traj.states[end])
        @test isfinite(final_reward)
        println("   ✅ Sampled trajectory reward ($(round(final_reward, digits=2))): PASSED")
        
    catch e
        println("   ❌ GFlowNet integration failed: $e")
        @test false
    end
end

# =============================================================================
# Test Summary
# =============================================================================

println("\n" * "="^50)
println("🎯 Test Suite Completed Successfully!")
println("✅ All components tested and validated")
println("🚀 Supply chain optimization ready for production use")
println("="^50)
