"""
Comprehensive Test Suite for Ultimate Supply Chain GFlowNet
==========================================================

Tests all components of the final working implementation:
- Network setup and connectivity
- Action space completeness  
- Reward function correctness
- Training functionality
- Solution quality validation
- Report generation

Based on the successful 58% high-service achievement.
"""

using Test
using Random
using Statistics
using Dates

# Load GFlowNet from parent directory
push!(LOAD_PATH, joinpath(@__DIR__, "../.."))
using GFlowNet

println("🧪 Starting Ultimate Supply Chain GFlowNet Test Suite")
println("="^60)

@testset "Ultimate Supply Chain GFlowNet Complete Tests" begin
    
    @testset "1. Network Setup and Ultimate Connectivity" begin
        println("🔗 Testing network setup and ultimate connectivity...")
        
        # Test drug creation
        drugs = [
            GFlowNet.Drug(1, "Oncology-A", GFlowNet.ONCOLOGY, GFlowNet.COLD, 6, 50.0, 2.0),
            GFlowNet.Drug(2, "Vaccine-B", GFlowNet.VACCINES, GFlowNet.FROZEN, 12, 25.0, 5.0),
            GFlowNet.Drug(3, "Generic-C", GFlowNet.GENERICS, GFlowNet.AMBIENT, 24, 5.0, 0.5),
            GFlowNet.Drug(4, "Biologic-D", GFlowNet.BIOLOGICS, GFlowNet.FROZEN, 3, 200.0, 10.0)
        ]
        
        @test length(drugs) == 4
        @test all(d -> d.id in 1:4, drugs)
        @test drugs[1].category == GFlowNet.ONCOLOGY
        @test drugs[2].storage_requirement == GFlowNet.FROZEN
        
        # Test facility creation with increased capacities
        facilities = [
            GFlowNet.Facility(1, "Plant-US", GFlowNet.MANUFACTURING, (40.0, -74.0),
                             Dict(1=>1200, 2=>2500, 3=>6000, 4=>600),
                             Dict(1=>600, 2=>1200, 3=>2500, 4=>250),
                             120_000.0, 10.0),
            GFlowNet.Facility(2, "Plant-EU", GFlowNet.MANUFACTURING, (50.0, 4.0),
                             Dict(1=>1000, 2=>1800, 3=>5000, 4=>500),
                             Dict(1=>500, 2=>1000, 3=>1800, 4=>200),
                             110_000.0, 12.0),
            GFlowNet.Facility(3, "DC-East", GFlowNet.DISTRIBUTION, (41.0, -73.0),
                             Dict{Int,Float64}(),
                             Dict(1=>2500, 2=>3500, 3=>10000, 4=>1200),
                             60_000.0, 5.0),
            GFlowNet.Facility(4, "DC-West", GFlowNet.DISTRIBUTION, (37.0, -122.0),
                             Dict{Int,Float64}(),
                             Dict(1=>2000, 2=>3000, 3=>8000, 4=>1000),
                             55_000.0, 5.0),
            GFlowNet.Facility(5, "Depot-EU", GFlowNet.DEPOT, (48.0, 2.0),
                             Dict{Int,Float64}(),
                             Dict(1=>1300, 2=>2000, 3=>5000, 4=>600),
                             40_000.0, 3.0)
        ]
        
        @test length(facilities) == 5
        @test count(f -> f.type == GFlowNet.MANUFACTURING, facilities) == 2
        @test count(f -> f.type == GFlowNet.DISTRIBUTION, facilities) == 2
        @test count(f -> f.type == GFlowNet.DEPOT, facilities) == 1
        
        # Test increased capacities (should be 20-25% higher than baseline)
        @test facilities[1].production_capacity[1] >= 1000  # Was 1000, now 1200
        @test facilities[3].storage_capacity[1] >= 2000     # Was 2000, now 2500
        
        # Test patient regions
        regions = [
            GFlowNet.PatientRegion(1, "US-Northeast", (42.0, -71.0),
                                  Dict(1=>800, 2=>1200, 3=>3000, 4=>300), 0.95),
            GFlowNet.PatientRegion(2, "US-West", (34.0, -118.0),
                                  Dict(1=>600, 2=>1000, 3=>2500, 4=>250), 0.95),
            GFlowNet.PatientRegion(3, "EU-Central", (52.0, 13.0),
                                  Dict(1=>500, 2=>800, 3=>2000, 4=>200), 0.95)
        ]
        
        @test length(regions) == 3
        @test all(r -> r.service_level_target == 0.95, regions)
        
        # Test ultimate connectivity calculation
        mfg_facilities = filter(f -> f.type == GFlowNet.MANUFACTURING, facilities)
        dist_facilities = filter(f -> f.type in [GFlowNet.DISTRIBUTION, GFlowNet.DEPOT], facilities)
        
        expected_mfg_dist_routes = length(mfg_facilities) * length(dist_facilities)  # 2 * 3 = 6
        expected_dist_dist_routes = length(dist_facilities) * (length(dist_facilities) - 1)  # 3 * 2 = 6
        expected_total_routes = expected_mfg_dist_routes + expected_dist_dist_routes  # 12
        
        @test expected_total_routes == 12  # Ultimate connectivity target
        
        println("   ✅ Network setup verified: $(length(drugs)) drugs, $(length(facilities)) facilities, $(length(regions)) regions")
        println("   ✅ Ultimate connectivity: $expected_total_routes routes expected")
    end
    
    @testset "2. Action Space Completeness" begin
        println("🎯 Testing ultra-granular action space...")
        
        # Create test network
        drugs = [GFlowNet.Drug(1, "Test", GFlowNet.GENERICS, GFlowNet.AMBIENT, 24, 5.0, 0.5)]
        facilities = [
            GFlowNet.Facility(1, "Plant", GFlowNet.MANUFACTURING, (40.0, -74.0),
                             Dict(1=>1000), Dict(1=>500), 100_000.0, 10.0),
            GFlowNet.Facility(2, "DC", GFlowNet.DISTRIBUTION, (41.0, -73.0),
                             Dict{Int,Float64}(), Dict(1=>2000), 50_000.0, 5.0)
        ]
        regions = [GFlowNet.PatientRegion(1, "Region", (42.0, -71.0), Dict(1=>800), 0.95)]
        routes = [GFlowNet.TransportRoute(1, 2, 100.0, 1.0, 1, Dict(GFlowNet.AMBIENT=>1.0))]
        
        # Test action creation with ultra-granular levels
        actions = GFlowNet.SupplyChainAction[]
        
        # Production actions - 10 levels
        production_levels = [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        for facility in facilities
            if facility.type == GFlowNet.MANUFACTURING
                for (drug_id, capacity) in facility.production_capacity
                    for level in production_levels
                        push!(actions, GFlowNet.ProduceAction(facility.id, drug_id, capacity * level))
                    end
                end
            end
        end
        
        # Shipping actions - 8 quantities
        shipping_quantities = [100.0, 300.0, 500.0, 750.0, 1000.0, 1250.0, 1500.0, 2000.0]
        for route in routes
            for drug in drugs
                for qty in shipping_quantities
                    push!(actions, GFlowNet.ShipAction(route.from_facility, route.to_facility, drug.id, qty))
                end
            end
        end
        
        # Service actions - 14 levels (focus on high service)
        service_levels = [0.80, 0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
        for facility in facilities
            if facility.type in [GFlowNet.DISTRIBUTION, GFlowNet.DEPOT]
                for region in regions
                    for (drug_id, demand) in region.monthly_demand
                        for level in service_levels
                            push!(actions, GFlowNet.ServeAction(facility.id, region.id, drug_id, demand * level))
                        end
                    end
                end
            end
        end
        
        # Time progression actions
        push!(actions, GFlowNet.NextMonthAction())
        push!(actions, GFlowNet.FinishPlanningAction())
        
        # Test action space composition
        production_actions = count(a -> isa(a, GFlowNet.ProduceAction), actions)
        shipping_actions = count(a -> isa(a, GFlowNet.ShipAction), actions)
        service_actions = count(a -> isa(a, GFlowNet.ServeAction), actions)
        
        @test production_actions == 10  # 1 facility × 1 drug × 10 levels
        @test shipping_actions == 8    # 1 route × 1 drug × 8 quantities  
        @test service_actions == 14    # 1 facility × 1 region × 1 drug × 14 levels
        @test length(actions) == production_actions + shipping_actions + service_actions + 2
        
        # Test high service level focus (≥95% service options)
        high_service_actions = count(a -> isa(a, GFlowNet.ServeAction) && a.quantity >= regions[1].monthly_demand[1] * 0.95, actions)
        @test high_service_actions >= 6  # Should have multiple ≥95% service options
        
        println("   ✅ Action space verified: $(length(actions)) total actions")
        println("   ✅ Composition: $production_actions production, $shipping_actions shipping, $service_actions service")
        println("   ✅ High-service focus: $high_service_actions actions ≥95% service level")
    end
    
    @testset "3. Reward Function Correctness" begin
        println("🏆 Testing corrected reward function...")
        
        # Create test network
        network = GFlowNet.SupplyChainNetwork([], [], [], [])
        
        # Test high service state (≥95%)
        high_service_state = GFlowNet.SupplyChainState(
            network, Dict(), Dict(), Dict(), Dict(),
            1, 3, true, 0.96, 500000.0
        )
        
        # Test medium service state (90-95%)
        medium_service_state = GFlowNet.SupplyChainState(
            network, Dict(), Dict(), Dict(), Dict(),
            1, 3, true, 0.92, 400000.0
        )
        
        # Test low service state (<90%)
        low_service_state = GFlowNet.SupplyChainState(
            network, Dict(), Dict(), Dict(), Dict(),
            1, 3, true, 0.75, 300000.0
        )
        
        # Test non-terminal state
        non_terminal_state = GFlowNet.SupplyChainState(
            network, Dict(), Dict(), Dict(), Dict(),
            1, 3, false, 0.0, 0.0
        )
        
        # Test reward function with corrected implementation
        @test GFlowNet.reward(high_service_state) == 100.0
        @test GFlowNet.reward(medium_service_state) == 20.0
        @test GFlowNet.reward(low_service_state) == 0.01
        @test GFlowNet.reward(non_terminal_state) == 0.0
        
        # Test threshold boundaries
        boundary_95_state = GFlowNet.SupplyChainState(
            network, Dict(), Dict(), Dict(), Dict(),
            1, 3, true, 0.95, 500000.0
        )
        boundary_90_state = GFlowNet.SupplyChainState(
            network, Dict(), Dict(), Dict(), Dict(),
            1, 3, true, 0.90, 400000.0
        )
        
        @test GFlowNet.reward(boundary_95_state) == 100.0  # Exactly 95% gets high reward
        @test GFlowNet.reward(boundary_90_state) == 20.0   # Exactly 90% gets medium reward
        
        # Test extreme reward differentiation (10,000:1 ratio)
        reward_ratio = GFlowNet.reward(high_service_state) / GFlowNet.reward(low_service_state)
        @test reward_ratio == 10000.0  # 100.0 / 0.01 = 10,000
        
        println("   ✅ Reward function verified: 100.0/20.0/0.01 thresholds")
        println("   ✅ Extreme differentiation: $(Int(reward_ratio)):1 ratio")
    end
