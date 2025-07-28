# Supply Chain Optimization Implementation Guide

## Overview

This guide explains how to use the pharmaceutical supply chain optimization example with GFlowNet. The implementation demonstrates real-world supply chain flow optimization including production planning, inventory management, and distribution decisions.

## Quick Start

### 1. Environment Setup

```bash
cd examples/supply_chain_optimization
julia --project=.
```

### 2. Run Complete Example

```bash
julia --project=. supply_chain_demo.jl
```

### 3. Run Tests

```bash
julia --project=. test/test_supply_chain.jl
```

## Code Structure

```
examples/supply_chain_optimization/
├── supply_chain_demo.jl          # Main demonstration script
├── docs/
│   ├── problem_formulation.md    # Mathematical formulation
│   ├── implementation_guide.md   # This guide
│   └── results_analysis.md       # Results interpretation
├── test/
│   ├── test_supply_chain.jl      # Unit tests
│   └── debug_supply_chain.jl     # Debug utilities
├── results/                      # Generated results
└── Project.toml                  # Dependencies
```

## Key Components

### 1. Network Definition

```julia
# Create drugs with supply chain properties
drugs = [
    Drug(1, "Oncology-A", ONCOLOGY, COLD, 6, 50.0, 2.0),
    Drug(2, "Vaccine-B", VACCINES, FROZEN, 12, 25.0, 5.0),
    # ...
]

# Create facilities (manufacturing, distribution, depots)
facilities = [
    Facility(1, "Plant-US", MANUFACTURING, (40.0, -74.0),
             Dict(1=>1000, 2=>2000), Dict(1=>500, 2=>1000),
             100_000.0, 10.0),
    # ...
]

# Create patient regions with demand
regions = [
    PatientRegion(1, "US-Northeast", (42.0, -71.0),
                  Dict(1=>800, 2=>1200), 0.95),
    # ...
]

# Create transportation network
routes = [
    TransportRoute(1, 3, 100.0, 0.5, 1, 
                   Dict(AMBIENT=>1.0, COLD=>1.2, FROZEN=>1.5)),
    # ...
]

network = SupplyChainNetwork(drugs, facilities, regions, routes)
```

### 2. State and Actions

```julia
# Initial state
initial_state = SupplyChainState(
    network,
    Dict{Tuple{Int,Int}, Float64}(),      # production
    Dict{Tuple{Int,Int}, Float64}(),      # inventory
    Dict{Tuple{Int,Int,Int}, Float64}(),  # shipments
    Dict{Tuple{Int,Int}, Float64}(),      # demand_served
    1, 3, false, 0.0, 0.0                # time, horizon, terminal, cost, service
)

# Action generation
actions = SupplyChainAction[]

# Production actions
for facility in facilities
    if facility.type == MANUFACTURING
        for (drug_id, capacity) in facility.production_capacity
            for pct in [0.25, 0.5, 0.75, 1.0]
                push!(actions, ProduceAction(facility.id, drug_id, capacity * pct))
            end
        end
    end
end

# Shipment and service actions...
```

### 3. GFlowNet Training

```julia
# Create model
model = create_gflownet(
    initial_state,
    actions;
    state_dim = 13,
    hidden_dim = 64,
    learning_rate = 0.005
)

# Configure training
config = TrainingConfig(
    objective=TRAJECTORY_BALANCE,
    n_iterations=30,
    batch_size=8,
    learning_rate=0.005
)

# Train model
history = train_gflownet(model, config; verbose=true)
```

### 4. Solution Analysis

```julia
# Sample diverse solutions
trajectories = [sample_trajectory(model) for _ in 1:50]

# Analyze results
rewards = [reward(traj.states[end]) for traj in trajectories]
costs = [traj.states[end].total_cost for traj in trajectories]
service_levels = [traj.states[end].service_level for traj in trajectories]

# Find best solutions
best_idx = argmax(rewards)
best_solution = trajectories[best_idx].states[end]
```

## Customization

### 1. Network Configuration

Modify the network setup in `supply_chain_demo.jl`:

```julia
# Add more drugs
drugs = [
    Drug(5, "Biologic-E", BIOLOGICS, FROZEN, 3, 300.0, 15.0),
    # ...
]

# Add facilities in new regions
facilities = [
    Facility(6, "Plant-Asia", MANUFACTURING, (35.0, 139.0), ...),
    # ...
]
```

### 2. Optimization Parameters

Adjust training configuration:

```julia
config = TrainingConfig(
    n_iterations=50,        # More iterations
    batch_size=16,          # Larger batches
    learning_rate=0.01,     # Higher learning rate
    validation_frequency=10  # Less frequent validation
)
```

### 3. Action Discretization

Modify action generation for finer control:

```julia
# Finer production quantities
for pct in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    push!(actions, ProduceAction(facility.id, drug_id, capacity * pct))
end
```

## Performance Tuning

### 1. State Representation

The 13-dimensional state features can be extended:

```julia
function state_to_features(state::SupplyChainState)
    # Add more features for better representation
    features = [
        # ... existing features ...
        facility_utilization_variance,  # Utilization balance
        regional_coverage_score,        # Geographic coverage
        drug_portfolio_diversity        # Product mix
    ]
    return features
end
```

### 2. Reward Engineering

Customize the reward function for specific objectives:

```julia
function reward(state::SupplyChainState)
    if !state.is_terminal
        return 0.0
    end
    
    # Custom reward components
    cost_efficiency = calculate_cost_efficiency(state)
    service_excellence = calculate_service_excellence(state)
    sustainability_score = calculate_sustainability(state)
    
    return 100.0 + 50.0 * service_excellence - 30.0 * cost_efficiency + 20.0 * sustainability_score
end
```

### 3. Constraint Handling

Add domain-specific constraints:

```julia
function is_applicable(action::ProduceAction, state::SupplyChainState)
    # Standard capacity check
    if !basic_capacity_check(action, state)
        return false
    end
    
    # Custom constraints
    if !regulatory_compliance_check(action, state)
        return false
    end
    
    if !quality_assurance_check(action, state)
        return false
    end
    
    return true
end
```

## Troubleshooting

### Common Issues

1. **Training Convergence**: Reduce learning rate, increase batch size
2. **Memory Usage**: Reduce action space size, use smaller networks
3. **Slow Sampling**: Optimize state transition functions
4. **Poor Solutions**: Adjust reward function, increase training iterations

### Debug Mode

Use the debug script for troubleshooting:

```bash
julia --project=. test/debug_supply_chain.jl
```

This will test individual components and identify issues.

## Results Interpretation

See `docs/results_analysis.md` for detailed guidance on interpreting optimization results and extracting business insights from GFlowNet solutions.

## Extensions

The framework can be extended for:
- **Stochastic Demand**: Add uncertainty in patient demand
- **Supply Disruptions**: Model facility outages and route closures
- **Regulatory Changes**: Dynamic approval requirements
- **Sustainability**: Carbon footprint optimization
- **Risk Management**: Supply chain resilience metrics
