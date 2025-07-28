# Supply Chain Network Optimization with GFlowNets

This example demonstrates how to use GFlowNets to optimize supply chain network structures that minimize cost while ensuring reliability.

## Problem Description

Supply chain network design involves making decisions about:

1. **Network Structure**: Which suppliers, warehouses, and customers to connect
2. **Cost Optimization**: Minimizing transportation and operational costs
3. **Reliability Assurance**: Ensuring robust supply paths and redundancy
4. **Capacity Planning**: Balancing network capacity with demand requirements

## Key Questions Addressed

- **What network structure minimizes cost while ensuring reliability?**
- **How do we balance direct connections vs. warehouse intermediaries?**
- **What level of redundancy is optimal for supply security?**
- **How do geographic factors affect network design?**

## Features

### Network Components
- **Suppliers**: Manufacturing facilities with production capacity
- **Warehouses**: Intermediate storage and distribution centers
- **Customers**: Retail locations with demand requirements
- **Connections**: Transportation links with capacity and cost

### Optimization Objectives
- **Cost Minimization**: Reduce fixed costs and transportation expenses
- **Reliability Maximization**: Ensure robust supply paths and redundancy
- **Capacity Utilization**: Optimize network throughput
- **Geographic Efficiency**: Minimize transportation distances

### GFlowNet Implementation
- **State Representation**: Current network configuration
- **Actions**: Add nodes, add connections, terminate construction
- **Reward Function**: Balances cost and reliability trade-offs
- **Sampling**: Generates diverse optimal network configurations

## Usage

### Basic Usage

```julia
using GFlowNet

# Create a supply chain optimization model
model = GFlowNet.create_supply_chain_gflownet(
    initial_suppliers=2,
    initial_customers=3,
    max_nodes=15,
    hidden_dim=128
)

# Configure training
config = GFlowNet.TrainingConfig(
    objective=GFlowNet.TRAJECTORY_BALANCE,
    n_iterations=500,
    batch_size=32
)

# Train the model
history = GFlowNet.train_gflownet(model, config; verbose=true)

# Sample optimal network configurations
trajectories = [GFlowNet.sample_trajectory(model) for _ in 1:100]
```

### Running the Example

```bash
cd examples/supply_chain
julia supply_chain_optimization.jl
```

## Output Files

The example generates several output files:

- `supply_chain_optimization_results.png`: Comprehensive visualization plots
- `supply_chain_training.csv`: Training progress and loss data
- `supply_chain_networks.csv`: Network configuration statistics

## Example Results

### Network Performance Metrics
- **Total Cost**: Sum of fixed costs and transportation costs
- **Reliability Score**: Network robustness and redundancy measure
- **Reward**: Combined optimization objective

### Typical Insights
- **Cost vs Reliability Trade-offs**: Different network configurations optimize different objectives
- **Redundancy Benefits**: Multiple supply paths improve reliability but increase costs
- **Geographic Clustering**: Nearby nodes tend to form efficient sub-networks
- **Warehouse Value**: Intermediate storage can reduce overall transportation costs

## Customization

### Modifying Network Parameters

```julia
# Custom suppliers
suppliers = [
    GFlowNet.SupplyChainNode(1, GFlowNet.SUPPLIER, (10.0, 20.0), 1000.0, 300.0, 2000.0)
]

# Custom customers  
customers = [
    GFlowNet.SupplyChainNode(2, GFlowNet.CUSTOMER, (70.0, 80.0), 400.0, 100.0, 500.0)
]

# Create initial state
initial_state = GFlowNet.create_initial_supply_chain_state(
    initial_suppliers=suppliers,
    initial_customers=customers
)
```

### Adjusting Reward Function

The reward function balances multiple objectives:

```julia
# Weights in the reward function
cost_weight = 0.4        # Emphasis on cost minimization
reliability_weight = 0.3  # Emphasis on network reliability
efficiency_weight = 0.2   # Emphasis on connection efficiency
structure_weight = 0.1    # Emphasis on network structure
```

## Advanced Features

### Multi-Scenario Analysis
The example includes functionality to compare different optimization scenarios:
- **Cost-Focused**: Prioritizes cost minimization
- **Reliability-Focused**: Prioritizes network robustness
- **Balanced**: Equal emphasis on cost and reliability

### Network Analysis
Comprehensive analysis of generated networks:
- Node type distribution (suppliers, warehouses, customers)
- Connection patterns and capacities
- Geographic spread and clustering
- Supply path redundancy

## Applications

This supply chain optimization approach can be applied to:

1. **Manufacturing Networks**: Optimizing production and distribution
2. **Retail Supply Chains**: Designing efficient delivery networks
3. **Emergency Response**: Planning resilient disaster response networks
4. **Telecommunications**: Designing robust communication networks
5. **Transportation**: Optimizing logistics and routing networks

## Dependencies

- GFlowNet.jl: Core GFlowNet implementation
- Plots.jl: Visualization and plotting
- CSV.jl, DataFrames.jl: Data export and analysis
- Statistics.jl: Statistical analysis
- LinearAlgebra.jl: Mathematical operations

## Next Steps

1. **Dynamic Scenarios**: Add time-varying demand and supply
2. **Capacity Constraints**: Include detailed capacity planning
3. **Risk Modeling**: Incorporate supply disruption scenarios
4. **Multi-Objective**: Extend to more complex objective functions
5. **Real Data**: Apply to actual supply chain datasets
