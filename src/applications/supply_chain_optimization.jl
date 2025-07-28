# Real Supply Chain Flow Optimization for GFlowNet
# Models realistic pharmaceutical supply chain operations:
# - Production planning at manufacturing facilities
# - Inventory management at distribution centers  
# - Transportation optimization between facilities
# - Demand fulfillment for patient regions
# Objective: Minimize total cost while satisfying demand constraints

using ..GFlowNet: AbstractState, AbstractAction, state_to_features, is_applicable, apply_action, reward
using LinearAlgebra
using Statistics
using Random

# =============================================================================
# Supply Chain Domain Types
# =============================================================================

@enum DrugType ONCOLOGY VACCINES GENERICS BIOLOGICS
@enum FacilityType MANUFACTURING DISTRIBUTION DEPOT
@enum StorageType AMBIENT COLD FROZEN

"""
    Drug - Pharmaceutical product with supply chain properties
"""
struct Drug
    id::Int
    name::String
    type::DrugType
    storage::StorageType
    shelf_life_months::Int
    production_cost::Float64
    storage_cost_monthly::Float64
end

"""
    Facility - Manufacturing or distribution facility
"""
struct Facility
    id::Int
    name::String
    type::FacilityType
    location::Tuple{Float64, Float64}
    
    # Capacities per drug type
    production_capacity::Dict{Int, Float64}  # drug_id -> monthly capacity
    storage_capacity::Dict{Int, Float64}     # drug_id -> storage capacity
    
    # Operating costs
    fixed_monthly_cost::Float64
    variable_cost_per_unit::Float64
end

"""
    PatientRegion - Geographic region with drug demand
"""
struct PatientRegion
    id::Int
    name::String
    location::Tuple{Float64, Float64}
    monthly_demand::Dict{Int, Float64}  # drug_id -> required quantity
    service_level_target::Float64       # minimum % of demand to satisfy
end

"""
    TransportRoute - Transportation connection between facilities
"""
struct TransportRoute
    from_facility::Int
    to_facility::Int
    distance_km::Float64
    cost_per_unit::Float64
    lead_time_days::Int
    storage_multiplier::Dict{StorageType, Float64}  # cost multiplier by storage type
end

"""
    SupplyChainNetwork - Complete network definition
"""
struct SupplyChainNetwork
    drugs::Vector{Drug}
    facilities::Vector{Facility}
    regions::Vector{PatientRegion}
    routes::Vector{TransportRoute}
end

# =============================================================================
# Supply Chain State
# =============================================================================

"""
    SupplyChainState - Current state of supply chain operations
"""
struct SupplyChainState <: AbstractState
    network::SupplyChainNetwork
    
    # Decision variables (what GFlowNet optimizes)
    production::Dict{Tuple{Int,Int}, Float64}      # (facility_id, drug_id) -> quantity
    inventory::Dict{Tuple{Int,Int}, Float64}       # (facility_id, drug_id) -> quantity  
    shipments::Dict{Tuple{Int,Int,Int}, Float64}   # (from, to, drug_id) -> quantity
    demand_served::Dict{Tuple{Int,Int}, Float64}   # (region_id, drug_id) -> quantity
    
    # Time and state
    current_month::Int
    planning_horizon::Int
    is_terminal::Bool
    
    # Performance tracking
    total_cost::Float64
    service_level::Float64
end

# =============================================================================
# Supply Chain Actions
# =============================================================================

abstract type SupplyChainAction <: AbstractAction end

"""Production action: produce quantity of drug at facility"""
struct ProduceAction <: SupplyChainAction
    facility_id::Int
    drug_id::Int
    quantity::Float64
end

"""Shipment action: ship quantity between facilities"""
struct ShipAction <: SupplyChainAction
    from_facility::Int
    to_facility::Int
    drug_id::Int
    quantity::Float64
end

"""Demand action: serve patient demand from distribution facility"""
struct ServeAction <: SupplyChainAction
    facility_id::Int
    region_id::Int
    drug_id::Int
    quantity::Float64
end

"""Time action: advance to next planning period"""
struct NextMonthAction <: SupplyChainAction
end

"""Termination action: complete planning horizon"""
struct FinishPlanningAction <: SupplyChainAction
end

# =============================================================================
# Helper Functions
# =============================================================================

function get_facility(network::SupplyChainNetwork, id::Int)
    for facility in network.facilities
        if facility.id == id
            return facility
        end
    end
    return nothing
end

function get_drug(network::SupplyChainNetwork, id::Int)
    for drug in network.drugs
        if drug.id == id
            return drug
        end
    end
    return nothing
end

function get_region(network::SupplyChainNetwork, id::Int)
    for region in network.regions
        if region.id == id
            return region
        end
    end
    return nothing
end

function get_route(network::SupplyChainNetwork, from::Int, to::Int)
    for route in network.routes
        if route.from_facility == from && route.to_facility == to
            return route
        end
    end
    return nothing
end

function calculate_transportation_cost(network::SupplyChainNetwork, from::Int, to::Int, drug_id::Int, quantity::Float64)
    route = get_route(network, from, to)
    drug = get_drug(network, drug_id)
    
    if route === nothing || drug === nothing
        return Inf
    end
    
    base_cost = route.cost_per_unit * quantity
    storage_multiplier = get(route.storage_multiplier, drug.storage, 1.0)
    
    return base_cost * storage_multiplier
end

function calculate_total_monthly_cost(state::SupplyChainState)
    network = state.network
    total_cost = 0.0
    
    # Fixed facility costs
    for facility in network.facilities
        total_cost += facility.fixed_monthly_cost
    end
    
    # Production costs
    for ((facility_id, drug_id), quantity) in state.production
        facility = get_facility(network, facility_id)
        drug = get_drug(network, drug_id)
        if facility !== nothing && drug !== nothing
            total_cost += quantity * (drug.production_cost + facility.variable_cost_per_unit)
        end
    end
    
    # Storage costs
    for ((facility_id, drug_id), quantity) in state.inventory
        drug = get_drug(network, drug_id)
        if drug !== nothing
            total_cost += quantity * drug.storage_cost_monthly
        end
    end
    
    # Transportation costs
    for ((from, to, drug_id), quantity) in state.shipments
        total_cost += calculate_transportation_cost(network, from, to, drug_id, quantity)
    end
    
    return total_cost
end

function calculate_service_level(state::SupplyChainState)
    network = state.network
    total_demand = 0.0
    total_served = 0.0
    
    for region in network.regions
        for (drug_id, demand) in region.monthly_demand
            total_demand += demand
            served = get(state.demand_served, (region.id, drug_id), 0.0)
            total_served += served
        end
    end
    
    return total_demand > 0 ? total_served / total_demand : 1.0
end

# =============================================================================
# GFlowNet Interface Implementation
# =============================================================================

"""
    state_to_features(state::SupplyChainState)

Convert supply chain state to neural network features.
"""
function state_to_features(state::SupplyChainState)
    network = state.network

    # Basic network metrics
    n_facilities = length(network.facilities)
    n_drugs = length(network.drugs)
    n_regions = length(network.regions)

    # Time progress
    time_progress = state.current_month / state.planning_horizon

    # Production utilization
    total_production = sum(values(state.production))
    max_production = sum(sum(values(f.production_capacity)) for f in network.facilities)
    production_util = max_production > 0 ? total_production / max_production : 0.0

    # Inventory utilization
    total_inventory = sum(values(state.inventory))
    max_storage = sum(sum(values(f.storage_capacity)) for f in network.facilities)
    inventory_util = max_storage > 0 ? total_inventory / max_storage : 0.0

    # Service level
    service_level = calculate_service_level(state)

    # Cost metrics (normalized)
    normalized_cost = state.total_cost / 1_000_000.0  # Scale to millions

    # Drug type production distribution
    drug_type_production = Dict{DrugType, Float64}()
    for drug_type in instances(DrugType)
        drug_type_production[drug_type] = 0.0
    end

    for ((facility_id, drug_id), quantity) in state.production
        drug = get_drug(network, drug_id)
        if drug !== nothing
            drug_type_production[drug.type] += quantity
        end
    end

    total_prod = sum(values(drug_type_production))
    if total_prod > 0
        for drug_type in instances(DrugType)
            drug_type_production[drug_type] /= total_prod
        end
    end

    # Combine features
    features = Float32[
        # Network structure (3)
        n_facilities / 10.0,
        n_drugs / 10.0,
        n_regions / 10.0,

        # Progress (1)
        time_progress,

        # Utilization (3)
        production_util,
        inventory_util,
        service_level,

        # Cost (1)
        min(normalized_cost, 5.0) / 5.0,

        # Drug distribution (4)
        drug_type_production[ONCOLOGY],
        drug_type_production[VACCINES],
        drug_type_production[GENERICS],
        drug_type_production[BIOLOGICS],

        # Terminal indicator (1)
        Float32(state.is_terminal)
    ]

    return features
end

"""
    is_terminal_state(state::SupplyChainState)
"""
function is_terminal_state(state::SupplyChainState)
    return state.is_terminal
end

"""
    reward(state::SupplyChainState)

Calculate reward based on cost minimization and service level.
"""
function reward(state::SupplyChainState)
    if !state.is_terminal
        return 0.0
    end

    # Service level penalty (must meet minimum service)
    service_level = calculate_service_level(state)
    if service_level < 0.95  # Must serve 95% of demand
        return 1.0  # Low reward for poor service
    end

    # Cost-based reward (lower cost = higher reward)
    total_cost = calculate_total_monthly_cost(state) * state.current_month

    # Normalize cost (assume max reasonable cost is $10M per month)
    max_cost = 10_000_000.0 * state.planning_horizon
    normalized_cost = min(total_cost / max_cost, 1.0)

    # Reward = base + service bonus - cost penalty
    base_reward = 100.0
    service_bonus = (service_level - 0.95) * 200.0  # Bonus for exceeding 95%
    cost_penalty = normalized_cost * 50.0

    final_reward = base_reward + service_bonus - cost_penalty

    return max(final_reward, 1.0)  # Minimum reward of 1.0
end

# =============================================================================
# Action Applicability
# =============================================================================

"""
    is_applicable(action::ProduceAction, state::SupplyChainState)
"""
function is_applicable(action::ProduceAction, state::SupplyChainState)
    if state.is_terminal || action.quantity <= 0
        return false
    end

    facility = get_facility(state.network, action.facility_id)
    if facility === nothing || facility.type != MANUFACTURING
        return false
    end

    # Check production capacity
    capacity = get(facility.production_capacity, action.drug_id, 0.0)
    current_production = get(state.production, (action.facility_id, action.drug_id), 0.0)

    return current_production + action.quantity <= capacity
end

"""
    is_applicable(action::ShipAction, state::SupplyChainState)
"""
function is_applicable(action::ShipAction, state::SupplyChainState)
    if state.is_terminal || action.quantity <= 0
        return false
    end

    # Check if route exists
    route = get_route(state.network, action.from_facility, action.to_facility)
    if route === nothing
        return false
    end

    # Check source inventory
    current_inventory = get(state.inventory, (action.from_facility, action.drug_id), 0.0)
    if current_inventory < action.quantity
        return false
    end

    # Check destination capacity
    to_facility = get_facility(state.network, action.to_facility)
    if to_facility !== nothing
        capacity = get(to_facility.storage_capacity, action.drug_id, 0.0)
        current_dest = get(state.inventory, (action.to_facility, action.drug_id), 0.0)
        if current_dest + action.quantity > capacity
            return false
        end
    end

    return true
end

"""
    is_applicable(action::ServeAction, state::SupplyChainState)
"""
function is_applicable(action::ServeAction, state::SupplyChainState)
    if state.is_terminal || action.quantity <= 0
        return false
    end

    # Check if facility can serve (distribution type)
    facility = get_facility(state.network, action.facility_id)
    if facility === nothing || !(facility.type in [DISTRIBUTION, DEPOT])
        return false
    end

    # Check inventory availability
    current_inventory = get(state.inventory, (action.facility_id, action.drug_id), 0.0)
    if current_inventory < action.quantity
        return false
    end

    # Check remaining demand
    region = get_region(state.network, action.region_id)
    if region === nothing
        return false
    end

    demand = get(region.monthly_demand, action.drug_id, 0.0)
    served = get(state.demand_served, (action.region_id, action.drug_id), 0.0)

    return served + action.quantity <= demand
end

"""
    is_applicable(action::NextMonthAction, state::SupplyChainState)
"""
function is_applicable(action::NextMonthAction, state::SupplyChainState)
    return !state.is_terminal && state.current_month < state.planning_horizon
end

"""
    is_applicable(action::FinishPlanningAction, state::SupplyChainState)
"""
function is_applicable(action::FinishPlanningAction, state::SupplyChainState)
    return !state.is_terminal
end

# =============================================================================
# State Transitions
# =============================================================================

"""
    apply_action(action::ProduceAction, state::SupplyChainState)
"""
function apply_action(action::ProduceAction, state::SupplyChainState)
    # Update production levels
    new_production = copy(state.production)
    current = get(new_production, (action.facility_id, action.drug_id), 0.0)
    new_production[(action.facility_id, action.drug_id)] = current + action.quantity

    # Add to inventory at production facility
    new_inventory = copy(state.inventory)
    current_inv = get(new_inventory, (action.facility_id, action.drug_id), 0.0)
    new_inventory[(action.facility_id, action.drug_id)] = current_inv + action.quantity

    # Update cost
    drug = get_drug(state.network, action.drug_id)
    facility = get_facility(state.network, action.facility_id)
    production_cost = action.quantity * (drug.production_cost + facility.variable_cost_per_unit)
    new_cost = state.total_cost + production_cost

    return SupplyChainState(
        state.network,
        new_production,
        new_inventory,
        state.shipments,
        state.demand_served,
        state.current_month,
        state.planning_horizon,
        state.is_terminal,
        new_cost,
        state.service_level
    )
end

"""
    apply_action(action::ShipAction, state::SupplyChainState)
"""
function apply_action(action::ShipAction, state::SupplyChainState)
    # Update shipment flows
    new_shipments = copy(state.shipments)
    current = get(new_shipments, (action.from_facility, action.to_facility, action.drug_id), 0.0)
    new_shipments[(action.from_facility, action.to_facility, action.drug_id)] = current + action.quantity

    # Update inventories
    new_inventory = copy(state.inventory)

    # Remove from source
    source_current = get(new_inventory, (action.from_facility, action.drug_id), 0.0)
    new_inventory[(action.from_facility, action.drug_id)] = source_current - action.quantity

    # Add to destination
    dest_current = get(new_inventory, (action.to_facility, action.drug_id), 0.0)
    new_inventory[(action.to_facility, action.drug_id)] = dest_current + action.quantity

    # Update cost
    transport_cost = calculate_transportation_cost(state.network, action.from_facility, action.to_facility, action.drug_id, action.quantity)
    new_cost = state.total_cost + transport_cost

    return SupplyChainState(
        state.network,
        state.production,
        new_inventory,
        new_shipments,
        state.demand_served,
        state.current_month,
        state.planning_horizon,
        state.is_terminal,
        new_cost,
        state.service_level
    )
end

"""
    apply_action(action::ServeAction, state::SupplyChainState)
"""
function apply_action(action::ServeAction, state::SupplyChainState)
    # Update demand served
    new_demand_served = copy(state.demand_served)
    current = get(new_demand_served, (action.region_id, action.drug_id), 0.0)
    new_demand_served[(action.region_id, action.drug_id)] = current + action.quantity

    # Remove from facility inventory
    new_inventory = copy(state.inventory)
    current_inv = get(new_inventory, (action.facility_id, action.drug_id), 0.0)
    new_inventory[(action.facility_id, action.drug_id)] = current_inv - action.quantity

    # Update service level
    new_service_level = calculate_service_level(SupplyChainState(
        state.network, state.production, new_inventory, state.shipments,
        new_demand_served, state.current_month, state.planning_horizon,
        state.is_terminal, state.total_cost, state.service_level
    ))

    return SupplyChainState(
        state.network,
        state.production,
        new_inventory,
        state.shipments,
        new_demand_served,
        state.current_month,
        state.planning_horizon,
        state.is_terminal,
        state.total_cost,
        new_service_level
    )
end

"""
    apply_action(action::NextMonthAction, state::SupplyChainState)
"""
function apply_action(action::NextMonthAction, state::SupplyChainState)
    # Reset monthly decisions but keep inventory
    new_month = state.current_month + 1
    is_terminal = new_month >= state.planning_horizon

    return SupplyChainState(
        state.network,
        Dict{Tuple{Int,Int}, Float64}(),  # Reset production
        state.inventory,                   # Keep inventory
        Dict{Tuple{Int,Int,Int}, Float64}(), # Reset shipments
        Dict{Tuple{Int,Int}, Float64}(),   # Reset demand served
        new_month,
        state.planning_horizon,
        is_terminal,
        state.total_cost,
        state.service_level
    )
end

"""
    apply_action(action::FinishPlanningAction, state::SupplyChainState)
"""
function apply_action(action::FinishPlanningAction, state::SupplyChainState)
    return SupplyChainState(
        state.network,
        state.production,
        state.inventory,
        state.shipments,
        state.demand_served,
        state.current_month,
        state.planning_horizon,
        true,  # Terminal
        state.total_cost,
        state.service_level
    )
end
