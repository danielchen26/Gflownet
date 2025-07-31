# Test Supply Chain Optimization Application
# Tests for the supply chain domain implementation

using Test
using GFlowNet
using Random

@testset "Supply Chain Tests" begin
    @testset "Supply Chain Types" begin
        # Test Drug type
        drug = Drug(
            1, "Aspirin", GENERICS, AMBIENT, 24,
            100.0, 10.0
        )
        @test drug.id == 1
        @test drug.name == "Aspirin"
        @test drug.type == GENERICS
        @test drug.storage == AMBIENT
        @test drug.production_cost == 100.0
        
        # Test Facility type
        facility = Facility(
            1, "Plant1", MANUFACTURING, (0.5, 0.5),
            Dict(1 => 1000.0),  # production capacity
            Dict(1 => 5000.0),  # storage capacity
            10000.0,  # fixed cost
            1.0       # variable cost
        )
        @test facility.id == 1
        @test facility.name == "Plant1"
        @test facility.type == MANUFACTURING
        @test facility.location == (0.5, 0.5)
        @test facility.production_capacity[1] == 1000.0
        
        # Test PatientRegion type
        region = PatientRegion(
            1, "Region1", (1.0, 1.0),
            Dict(1 => 500.0),  # demand
            0.95  # service level
        )
        @test region.id == 1
        @test region.name == "Region1"
        @test region.location == (1.0, 1.0)
        @test region.monthly_demand[1] == 500.0
        @test region.service_level_target == 0.95
        
        # Test TransportRoute type
        route = TransportRoute(
            1, 2, 100.0, 2.0, 2,
            Dict(AMBIENT => 1.0, COLD => 1.5, FROZEN => 2.0)
        )
        @test route.from_facility == 1
        @test route.to_facility == 2
        @test route.distance_km == 100.0
        @test route.cost_per_unit == 2.0
    end
    
    @testset "Supply Chain State" begin
        # Create a simple supply chain network
        drugs = [Drug(1, "DrugA", GENERICS, AMBIENT, 24, 100.0, 10.0)]
        
        facilities = [
            Facility(1, "Factory", MANUFACTURING, (0.0, 0.0),
                Dict(1 => 1000.0), Dict(1 => 5000.0), 10000.0, 1.0),
            Facility(2, "Warehouse", DISTRIBUTION, (0.5, 0.5),
                Dict(1 => 0.0), Dict(1 => 2000.0), 5000.0, 0.5)
        ]
        
        regions = [
            PatientRegion(1, "City", (1.0, 1.0), Dict(1 => 200.0), 0.95)
        ]
        
        routes = [
            TransportRoute(1, 2, 50.0, 1.0, 1, Dict(AMBIENT => 1.0))
        ]
        
        network = SupplyChainNetwork(drugs, facilities, regions, routes)
        
        # Create state
        state = SupplyChainState(
            network,
            Dict((1, 1) => 100.0),  # production
            Dict((1, 1) => 500.0),  # inventory
            Dict((1, 2, 1) => 50.0),  # shipments
            Dict((1, 1) => 150.0),  # demand served
            1,  # current month
            12, # planning horizon
            false,  # not terminal
            15000.0,  # total_cost
            0.75      # service_level
        )
        
        @test state.network == network
        @test state.current_month == 1
        @test !state.is_terminal
        @test haskey(state.production, (1, 1))
        @test state.production[(1, 1)] == 100.0
        
        # Test state features
        features = GFlowNet.state_to_features(state)
        @test features isa Vector{Float32}
        @test all(isfinite, features)
    end
    
    @testset "Supply Chain Actions" begin
        # Test action types exist
        @test isdefined(GFlowNet, :ProduceAction)
        @test isdefined(GFlowNet, :ShipAction)
        @test isdefined(GFlowNet, :NextMonthAction)
        @test isdefined(GFlowNet, :FinishPlanningAction)
        
        # Create sample actions
        produce_action = ProduceAction(1, 1, 50.0)  # facility, drug, quantity
        @test produce_action.facility_id == 1
        @test produce_action.drug_id == 1
        @test produce_action.quantity == 50.0
        
        ship_action = ShipAction(1, 2, 1, 25.0)  # from, to, drug, quantity
        @test ship_action.from_facility == 1
        @test ship_action.to_facility == 2
        @test ship_action.drug_id == 1
        @test ship_action.quantity == 25.0
        
        next_month = NextMonthAction()
        @test next_month isa NextMonthAction
        
        finish = FinishPlanningAction()
        @test finish isa FinishPlanningAction
    end
    
    @testset "Supply Chain Model Creation" begin
        # Create a simple supply chain network for testing
        drugs = [Drug(1, "DrugA", GENERICS, AMBIENT, 24, 100.0, 10.0)]
        facilities = [
            Facility(1, "Factory", MANUFACTURING, (0.0, 0.0),
                Dict(1 => 1000.0), Dict(1 => 5000.0), 10000.0, 1.0)
        ]
        regions = [PatientRegion(1, "City", (1.0, 1.0), Dict(1 => 200.0), 0.95)]
        routes = TransportRoute[]
        
        network = SupplyChainNetwork(drugs, facilities, regions, routes)
        
        # Test model creation
        model = create_supply_chain_gflownet(
            network=network,
            hidden_dim=16,
            learning_rate=0.01
        )
        
        @test model isa GFlowNet.GFlowNetModel
        @test model.initial_state isa SupplyChainState
        @test !isempty(model.all_actions)
        
        # Test that initial state has proper structure
        initial = model.initial_state
        @test hasfield(typeof(initial), :network)
        @test hasfield(typeof(initial), :production)
        @test hasfield(typeof(initial), :inventory)
        @test hasfield(typeof(initial), :shipments)
        @test hasfield(typeof(initial), :demand_served)
        
        # Test that we can sample from the model
        trajectory = GFlowNet.sample_trajectory(model)
        @test trajectory isa GFlowNet.Trajectory
        @test !isempty(trajectory.states)
        @test GFlowNet.is_terminal_state(trajectory.states[end])
    end
    
    @testset "Reward Calculation" begin
        # Create a simple network
        drugs = [Drug(1, "DrugA", GENERICS, AMBIENT, 24, 100.0, 10.0)]
        facilities = [Facility(1, "Factory", MANUFACTURING, (0.0, 0.0),
            Dict(1 => 1000.0), Dict(1 => 5000.0), 10000.0, 1.0)]
        regions = [PatientRegion(1, "City", (1.0, 1.0), Dict(1 => 200.0), 0.95)]
        routes = TransportRoute[]
        
        network = SupplyChainNetwork(drugs, facilities, regions, routes)
        
        # Non-terminal state should have zero reward
        non_terminal = SupplyChainState(
            network,
            Dict(), Dict(), Dict(), Dict(),
            1, 12, false,
            0.0, 0.0
        )
        @test GFlowNet.reward(non_terminal) == 0.0
        
        # Terminal state should have positive reward based on performance
        terminal = SupplyChainState(
            network,
            Dict(), Dict(), Dict(), Dict(),
            12, 12, true,  # terminal after 12 months
            100000.0,
            0.90
        )
        reward_value = GFlowNet.reward(terminal)
        @test reward_value > 0.0
        @test isfinite(reward_value)
    end
    
    @testset "Action Applicability" begin
        # Create a test network and state
        drugs = [Drug(1, "DrugA", GENERICS, AMBIENT, 24, 100.0, 10.0)]
        facilities = [
            Facility(1, "Factory", MANUFACTURING, (0.0, 0.0),
                Dict(1 => 1000.0), Dict(1 => 5000.0), 10000.0, 1.0)
        ]
        regions = [PatientRegion(1, "City", (1.0, 1.0), Dict(1 => 200.0), 0.95)]
        routes = TransportRoute[]
        
        network = SupplyChainNetwork(drugs, facilities, regions, routes)
        
        state = SupplyChainState(
            network,
            Dict((1, 1) => 100.0),
            Dict((1, 1) => 500.0),
            Dict(), Dict(),
            1, 12, false,
            0.0, 0.0
        )
        
        # Test production is applicable at manufacturing facility
        produce = ProduceAction(1, 1, 50.0)
        @test GFlowNet.is_applicable(produce, state)
        
        # Test finish is applicable for non-terminal
        @test GFlowNet.is_applicable(FinishPlanningAction(), state)
        
        # Test terminal state - no actions applicable
        terminal_state = SupplyChainState(
            network,
            Dict(), Dict(), Dict(), Dict(),
            1, 12, true,
            0.0, 0.0
        )
        @test !GFlowNet.is_applicable(produce, terminal_state)
        @test !GFlowNet.is_applicable(FinishPlanningAction(), terminal_state)
    end
end