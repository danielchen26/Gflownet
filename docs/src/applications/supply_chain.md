# Supply Chain Optimization

## Overview

The supply chain optimization application demonstrates GFlowNet's ability to handle complex, real-world business optimization problems. It models pharmaceutical supply chain networks with multiple facilities, drugs, transport routes, and patient regions.

## Domain Description

### Components

1. **Facilities**
   - Manufacturing plants
   - Distribution centers
   - Regional depots
   - Each with capacity constraints and costs

2. **Drugs**
   - Different types (oncology, vaccines, generics, biologics)
   - Storage requirements (ambient, cold, frozen)
   - Shelf life constraints
   - Production costs and values

3. **Transport Routes**
   - Connect facilities
   - Have capacity and cost
   - Time delays
   - Temperature control capabilities

4. **Patient Regions**
   - Monthly demand for each drug
   - Service level requirements
   - Geographic locations

### State Representation

```julia
struct SupplyChainState <: AbstractState
    network::SupplyChainNetwork
    month::Int
    is_terminal::Bool
end

struct SupplyChainNetwork
    facilities::Vector{Facility}
    routes::Vector{TransportRoute}
    drugs::Vector{Drug}
    patients::Vector{PatientRegion}
    inventory::Dict{Tuple{Int,Int}, Int}  # (facility_id, drug_id) => quantity
    shipments::Vector{Shipment}
    served::Dict{Tuple{Int,Int}, Int}  # (region_id, drug_id) => quantity
end
```

## Actions

The domain supports multiple action types:

### 1. ProduceAction
Manufacture drugs at facilities:
```julia
struct ProduceAction <: SupplyChainAction
    facility_id::Int
    drug_id::Int
    quantity::Int
end
```

### 2. ShipAction
Transport drugs between facilities:
```julia
struct ShipAction <: SupplyChainAction
    route_id::Int
    drug_id::Int
    quantity::Int
end
```

### 3. ServeAction
Deliver drugs to patient regions:
```julia
struct ServeAction <: SupplyChainAction
    facility_id::Int
    region_id::Int
    drug_id::Int
    quantity::Int
end
```

### 4. NextMonthAction
Advance time (with inventory decay):
```julia
struct NextMonthAction <: SupplyChainAction end
```

### 5. FinishPlanningAction
Terminate planning:
```julia
struct FinishPlanningAction <: SupplyChainAction end
```

## Reward Function

The reward function balances multiple objectives:

```julia
function reward(state::SupplyChainState)
    if !state.is_terminal
        return 0.0
    end
    
    # Calculate metrics
    service_level = calculate_service_level(state.network)
    total_cost = calculate_total_cost(state.network)
    inventory_value = calculate_inventory_value(state.network)
    
    # Multi-objective reward
    reward = service_level * 100.0 - total_cost * 0.01 + inventory_value * 0.001
    
    # Ensure positive
    return max(reward, 1e-8)
end
```

## Key Features

### 1. Inventory Management
- Track inventory at each facility
- Handle expiration based on shelf life
- Balance holding costs vs stockouts

### 2. Multi-Period Planning
- Plan over 6-month horizon
- Sequential decision making
- Time-dependent constraints

### 3. Network Effects
- Route capacity constraints
- Multi-hop shipping
- Geographic considerations

### 4. Business Constraints
- Production capacity
- Storage limitations
- Budget constraints

## Implementation Highlights

### State Features
```julia
function state_to_features(state::SupplyChainState)
    features = Float32[]
    
    # Network structure
    append!(features, encode_network_topology(state.network))
    
    # Inventory levels
    append!(features, encode_inventory_state(state.network))
    
    # Demand satisfaction
    append!(features, encode_service_metrics(state.network))
    
    # Time features
    push!(features, Float32(state.month / 6.0))
    
    return features
end
```

### Action Applicability
Complex business rules determine valid actions:
```julia
function is_applicable(action::ProduceAction, state::SupplyChainState)
    facility = get_facility(state.network, action.facility_id)
    drug = get_drug(state.network, action.drug_id)
    
    # Check constraints
    has_capacity = facility.production_capacity[drug.id] >= action.quantity
    has_budget = facility.available_budget >= drug.production_cost * action.quantity
    not_terminated = !state.is_terminal
    
    return has_capacity && has_budget && not_terminated
end
```

## Results and Analysis

### Performance Metrics
- **Service Level**: Percentage of demand satisfied
- **Total Cost**: Production + shipping + holding costs
- **Network Efficiency**: Utilization of facilities and routes

### Discovered Strategies
The GFlowNet learns sophisticated strategies:
1. **Hub-and-Spoke**: Centralize distribution
2. **Just-in-Time**: Minimize inventory
3. **Risk Pooling**: Share inventory across regions
4. **Seasonal Planning**: Anticipate demand patterns

## Running the Example

```bash
cd examples/supply_chain_optimization
julia --project=. ultimate_connected_gflownet.jl
```

### Configuration Options
```julia
# Adjust network complexity
facilities = create_facilities(n_manufacturing=2, n_distribution=3)

# Modify planning horizon
max_months = 6

# Change reward weights
service_weight = 100.0
cost_weight = 0.01
```

## Extensions and Variations

### 1. Stochastic Demand
Add uncertainty to patient demand

### 2. Disruption Handling
Model facility or route failures

### 3. Competitive Dynamics
Multiple companies sharing infrastructure

### 4. Sustainability Metrics
Include carbon footprint in rewards

## Lessons Learned

1. **State Encoding**: Efficient representation of complex networks is crucial
2. **Action Space**: Hierarchical actions help manage complexity
3. **Reward Design**: Multi-objective rewards need careful balancing
4. **Computational Efficiency**: Smart caching and incremental updates

## See Also
- [Developer Guide](../manual/developer_guide.md) - Implementing complex domains
- [Grid World](grid_world.md) - Simpler starting example
- [Training Objectives](../manual/objectives.md) - Choosing objectives for complex domains