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
    SupplyChainState - Rich state representation matching mathematical formulation
"""
struct SupplyChainState <: AbstractState
    network::SupplyChainNetwork

    # Rich decision variables (matching mathematical model)
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

# Essential for GFlowNet DAG construction - proper hashing and equality
function Base.:(==)(a::SupplyChainState, b::SupplyChainState)
    return a.production == b.production &&
           a.inventory == b.inventory &&
           a.shipments == b.shipments &&
           a.demand_served == b.demand_served &&
           a.current_month == b.current_month &&
           a.is_terminal == b.is_terminal
end

function Base.hash(state::SupplyChainState, h::UInt)
    return hash((state.production, state.inventory, state.shipments,
                state.demand_served, state.current_month, state.is_terminal), h)
end

# =============================================================================
# Supply Chain Actions
# =============================================================================

"""
    SupplyChainAction - Abstract base type for rich supply chain actions
"""
abstract type SupplyChainAction <: AbstractAction end

"""Production action: produce discrete quantity of drug at facility"""
struct ProduceAction <: SupplyChainAction
    facility_id::Int
    drug_id::Int
    quantity::Float64
end

"""Shipment action: ship discrete quantity between facilities"""
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
    GFlowNet.state_to_features(state::SupplyChainState)

Convert rich supply chain state to feature vector for neural network.
"""
function GFlowNet.state_to_features(state::SupplyChainState)::Vector{Float32}
    network = state.network

    # Basic network metrics
    n_facilities = length(network.facilities)
    n_drugs = length(network.drugs)
    n_regions = length(network.regions)

    # Time progress
    time_progress = Float32(state.current_month / state.planning_horizon)

    # Production utilization (aggregate across all drugs)
    total_production = sum(values(state.production))
    max_production_capacity = sum(sum(values(f.production_capacity)) for f in network.facilities)
    production_util = max_production_capacity > 0 ? Float32(total_production / max_production_capacity) : 0.0f0

    # Inventory utilization (aggregate across all drugs)
    total_inventory = sum(values(state.inventory))
    max_storage_capacity = sum(sum(values(f.storage_capacity)) for f in network.facilities)
    inventory_util = max_storage_capacity > 0 ? Float32(total_inventory / max_storage_capacity) : 0.0f0

    # Service level
    service_level = Float32(state.service_level)

    # Cost metrics (normalized)
    normalized_cost = Float32(min(state.total_cost / 1_000_000.0, 5.0) / 5.0)  # Cap at 5M

    # Shipment activity
    total_shipments = sum(values(state.shipments))
    shipment_activity = total_production > 0 ? Float32(min(total_shipments / total_production, 2.0) / 2.0) : 0.0f0

    # Drug type distribution in production
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

    total_drug_production = sum(values(drug_type_production))
    if total_drug_production > 0
        for drug_type in instances(DrugType)
            drug_type_production[drug_type] /= total_drug_production
        end
    end

    # Combine all features (fixed size: exactly 13 features)
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

        # Cost and activity (2)
        normalized_cost,
        shipment_activity,

        # Drug distribution (3) - reduced to fit 13 total
        Float32(drug_type_production[ONCOLOGY]),
        Float32(drug_type_production[VACCINES]),
        Float32(drug_type_production[GENERICS]),

        # Terminal indicator (1)
        state.is_terminal ? 1.0f0 : 0.0f0
    ]

    return features
end

"""
    GFlowNet.is_terminal_state(state::SupplyChainState)
"""
GFlowNet.is_terminal_state(state::SupplyChainState) = state.is_terminal

"""
    GFlowNet.reward(state::SupplyChainState)

Calculate reward that REQUIRES business activity (production + service).
"""
function GFlowNet.reward(state::SupplyChainState)::Float64
    !state.is_terminal && return 0.0

    # Calculate business activity metrics
    total_production = sum(values(state.production))
    total_demand_served = sum(values(state.demand_served))

    # PENALIZE doing nothing - require minimum business activity
    if total_production == 0.0 && total_demand_served == 0.0
        return 1.0  # Very low reward for doing nothing
    end

    # Base reward for having business activity
    base_reward = 5.0

    # Production bonus (0-3 points) - reward for producing
    production_bonus = min(total_production / 1000.0, 3.0)

    # Service bonus (0-5 points) - reward for serving demand
    service_bonus = state.service_level * 5.0

    # Cost penalty (0-2 points) - light penalty for high costs
    normalized_cost = min(state.total_cost / 200_000.0, 1.0)
    cost_penalty = normalized_cost * 2.0

    # Final reward (range: 1-13, but requires activity)
    reward_value = base_reward + production_bonus + service_bonus - cost_penalty

    return max(reward_value, 1.0)
end

# =============================================================================
# GFlowNet Interface Implementation (following grid world pattern)
# =============================================================================

"""
    GFlowNet.is_applicable(action::SupplyChainAction, state::SupplyChainState)

Check if rich supply chain actions are applicable.
"""
function GFlowNet.is_applicable(action::ProduceAction, state::SupplyChainState)::Bool
    state.is_terminal && return false

    facility = get_facility(state.network, action.facility_id)
    facility === nothing && return false
    facility.type != MANUFACTURING && return false

    # Check production capacity
    capacity = get(facility.production_capacity, action.drug_id, 0.0)
    current_production = get(state.production, (action.facility_id, action.drug_id), 0.0)

    return action.quantity > 0 && current_production + action.quantity <= capacity
end

function GFlowNet.is_applicable(action::ShipAction, state::SupplyChainState)::Bool
    state.is_terminal && return false
    action.quantity <= 0 && return false

    # Check if route exists
    route = get_route(state.network, action.from_facility, action.to_facility)
    route === nothing && return false

    # RELAXED: Check source inventory (allow if ANY inventory exists)
    current_inventory = get(state.inventory, (action.from_facility, action.drug_id), 0.0)
    # Allow shipping if we have at least 10% of requested quantity
    return current_inventory >= (action.quantity * 0.1)
end

function GFlowNet.is_applicable(action::ServeAction, state::SupplyChainState)::Bool
    state.is_terminal && return false
    action.quantity <= 0 && return false

    facility = get_facility(state.network, action.facility_id)
    facility === nothing && return false
    !(facility.type in [DISTRIBUTION, DEPOT]) && return false

    # RELAXED: Check inventory availability (allow if we have 10% of needed)
    current_inventory = get(state.inventory, (action.facility_id, action.drug_id), 0.0)
    return current_inventory >= (action.quantity * 0.1)
end

function GFlowNet.is_applicable(action::NextMonthAction, state::SupplyChainState)::Bool
    return !state.is_terminal && state.current_month < state.planning_horizon
end

function GFlowNet.is_applicable(action::FinishPlanningAction, state::SupplyChainState)::Bool
    return !state.is_terminal
end



"""
    GFlowNet.apply_action(action::SupplyChainAction, state::SupplyChainState)

Apply rich supply chain actions to states.
"""
function GFlowNet.apply_action(action::ProduceAction, state::SupplyChainState)::SupplyChainState
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
        state.network, new_production, new_inventory, state.shipments, state.demand_served,
        state.current_month, state.planning_horizon, state.is_terminal, new_cost, state.service_level
    )
end

function GFlowNet.apply_action(action::ShipAction, state::SupplyChainState)::SupplyChainState
    # Update shipment flows
    new_shipments = copy(state.shipments)
    current = get(new_shipments, (action.from_facility, action.to_facility, action.drug_id), 0.0)
    new_shipments[(action.from_facility, action.to_facility, action.drug_id)] = current + action.quantity

    # Update inventories ROBUSTLY
    new_inventory = copy(state.inventory)

    # Remove from source (but don't go negative)
    source_current = get(new_inventory, (action.from_facility, action.drug_id), 0.0)
    actual_shipped = min(action.quantity, source_current)  # Ship only what we have
    new_inventory[(action.from_facility, action.drug_id)] = max(0.0, source_current - actual_shipped)

    # Add to destination
    dest_current = get(new_inventory, (action.to_facility, action.drug_id), 0.0)
    new_inventory[(action.to_facility, action.drug_id)] = dest_current + actual_shipped

    # Update cost (based on actual shipped amount)
    transport_cost = calculate_transportation_cost(state.network, action.from_facility, action.to_facility, action.drug_id, actual_shipped)
    new_cost = state.total_cost + transport_cost

    return SupplyChainState(
        state.network, state.production, new_inventory, new_shipments, state.demand_served,
        state.current_month, state.planning_horizon, state.is_terminal, new_cost, state.service_level
    )
end

function GFlowNet.apply_action(action::ServeAction, state::SupplyChainState)::SupplyChainState
    # Update demand served ROBUSTLY
    new_demand_served = copy(state.demand_served)
    current = get(new_demand_served, (action.region_id, action.drug_id), 0.0)

    # Remove from facility inventory (but don't go negative)
    new_inventory = copy(state.inventory)
    current_inv = get(new_inventory, (action.facility_id, action.drug_id), 0.0)
    actual_served = min(action.quantity, current_inv)  # Serve only what we have

    new_demand_served[(action.region_id, action.drug_id)] = current + actual_served
    new_inventory[(action.facility_id, action.drug_id)] = max(0.0, current_inv - actual_served)

    # Update service level (simplified calculation)
    total_demand = sum(sum(values(r.monthly_demand)) for r in state.network.regions)
    total_served = sum(values(new_demand_served))
    new_service_level = total_demand > 0 ? min(1.0, total_served / total_demand) : 0.0

    return SupplyChainState(
        state.network, state.production, new_inventory, state.shipments, new_demand_served,
        state.current_month, state.planning_horizon, state.is_terminal, state.total_cost, new_service_level
    )
end

function GFlowNet.apply_action(action::NextMonthAction, state::SupplyChainState)::SupplyChainState
    # Reset monthly decisions but keep inventory
    new_month = state.current_month + 1
    is_terminal = new_month >= state.planning_horizon

    return SupplyChainState(
        state.network,
        Dict{Tuple{Int,Int}, Float64}(),  # Reset production
        state.inventory,                   # Keep inventory
        Dict{Tuple{Int,Int,Int}, Float64}(), # Reset shipments
        Dict{Tuple{Int,Int}, Float64}(),   # Reset demand served
        new_month, state.planning_horizon, is_terminal, state.total_cost, state.service_level
    )
end

function GFlowNet.apply_action(action::FinishPlanningAction, state::SupplyChainState)::SupplyChainState
    return SupplyChainState(
        state.network, state.production, state.inventory, state.shipments, state.demand_served,
        state.current_month, state.planning_horizon, true, state.total_cost, state.service_level
    )
end

# Supply chain optimization implementation complete
# All GFlowNet interface functions implemented following grid world pattern
