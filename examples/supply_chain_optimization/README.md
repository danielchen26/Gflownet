# Pharmaceutical Supply Chain Optimization with GFlowNet

A comprehensive implementation of real-world pharmaceutical supply chain flow optimization using Generative Flow Networks (GFlowNet). This example demonstrates production planning, inventory management, and distribution optimization to minimize costs while maintaining service level requirements.

## 🎯 Problem Overview

**Objective**: Optimize pharmaceutical supply chain operations by making production, inventory, and distribution decisions that minimize total cost while ensuring ≥95% service level for patient demand.

**Key Features**:
- **Multi-echelon network**: Manufacturing → Distribution → Regional → Patients
- **Multi-product optimization**: Different drugs with varying properties
- **Multi-objective balancing**: Cost minimization vs service level achievement
- **Realistic constraints**: Capacity limits, shelf life, regulatory requirements

## 🚀 Quick Start

### Prerequisites
- Julia 1.9+
- GFlowNet.jl package

### Installation
```bash
cd examples/supply_chain_optimization
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

### Run Complete Example
```bash
julia --project=. supply_chain_demo.jl
```

### Run Tests
```bash
julia --project=. test/test_supply_chain.jl
```

## 📁 Directory Structure

```
examples/supply_chain_optimization/
├── README.md                     # This file
├── supply_chain_demo.jl          # Main demonstration script
├── Project.toml                  # Julia dependencies
├── docs/                         # Comprehensive documentation
│   ├── problem_formulation.md    # Mathematical formulation
│   ├── implementation_guide.md   # Usage and customization guide
│   └── results_analysis.md       # Results interpretation guide
├── test/                         # Test suite and debugging
│   ├── test_supply_chain.jl      # Comprehensive unit tests
│   └── debug_supply_chain.jl     # Debug utilities
└── results/                      # Generated results and reports
```

## 🏗️ Architecture

### Network Components

**Drugs**: Pharmaceutical products with properties
- Storage requirements (ambient, cold, frozen)
- Shelf life constraints
- Production and storage costs

**Facilities**: Supply chain infrastructure
- Manufacturing plants (production capacity)
- Distribution centers (storage capacity)
- Regional depots (local distribution)

**Patient Regions**: Demand sources
- Monthly demand requirements
- Service level targets (≥95%)
- Geographic distribution

**Transportation**: Logistics network
- Routes between facilities
- Cost and lead time parameters
- Storage-specific multipliers

### Optimization Process

1. **State Representation**: 13-dimensional feature vector encoding network status
2. **Action Space**: Production, shipment, service, and time advancement actions
3. **GFlowNet Training**: Trajectory Balance objective with neural network policies
4. **Solution Generation**: Diverse high-quality operational strategies
5. **Analysis**: Cost-service trade-offs and business insights

## 📊 Key Results

The optimization generates diverse solutions with different strategic focuses:

**Cost-Efficient Solutions**:
- Minimize total operational costs
- Lean inventory management
- Concentrated production

**Service-Optimized Solutions**:
- Maximize patient service levels
- Distributed inventory buffers
- Redundant supply paths

**Balanced Strategies**:
- Moderate cost with excellent service
- Risk-balanced operations
- Flexible capacity utilization

## 🔧 Customization

### Network Configuration

Modify `supply_chain_demo.jl` to customize:

```julia
# Add new drugs
drugs = [
    Drug(5, "Biologic-E", BIOLOGICS, FROZEN, 3, 300.0, 15.0),
    # ...
]

# Add facilities in new regions
facilities = [
    Facility(6, "Plant-Asia", MANUFACTURING, (35.0, 139.0), ...),
    # ...
]

# Adjust patient demand
regions = [
    PatientRegion(4, "Asia-Pacific", (35.0, 139.0), 
                  Dict(1=>1200, 2=>800), 0.98),
    # ...
]
```

### Training Parameters

Adjust optimization settings:

```julia
config = TrainingConfig(
    n_iterations=50,        # More training iterations
    batch_size=16,          # Larger batch size
    learning_rate=0.01,     # Higher learning rate
    validation_frequency=10  # Validation frequency
)
```

### Reward Function

Customize objectives in `src/applications/supply_chain_optimization.jl`:

```julia
function reward(state::SupplyChainState)
    # Add custom objectives
    sustainability_score = calculate_carbon_footprint(state)
    resilience_score = calculate_supply_resilience(state)
    
    return base_reward + service_bonus - cost_penalty + 
           sustainability_bonus + resilience_bonus
end
```

## 📈 Performance Metrics

**Cost Metrics**:
- Total supply chain cost
- Cost breakdown by category
- Cost efficiency ratios

**Service Metrics**:
- Overall service level achievement
- Regional service performance
- Drug-specific service rates

**Operational Metrics**:
- Production utilization rates
- Inventory turnover ratios
- Transportation efficiency

## 🧪 Testing

The test suite validates all components:

```bash
# Run comprehensive tests
julia --project=. test/test_supply_chain.jl

# Debug specific issues
julia --project=. test/debug_supply_chain.jl
```

**Test Coverage**:
- Data structure creation and validation
- Network assembly and helper functions
- State representation and feature extraction
- Action applicability and state transitions
- Reward function correctness
- GFlowNet integration and sampling

## 📚 Documentation

**Mathematical Formulation** (`docs/problem_formulation.md`):
- Complete mathematical model
- Constraint definitions
- GFlowNet formulation
- Complexity analysis

**Implementation Guide** (`docs/implementation_guide.md`):
- Detailed usage instructions
- Customization examples
- Performance tuning tips
- Troubleshooting guide

**Results Analysis** (`docs/results_analysis.md`):
- Metrics interpretation
- Business insights extraction
- Comparative analysis framework
- Visualization guidelines

## 🔬 Research Applications

This implementation supports research in:

**Supply Chain Optimization**:
- Multi-objective optimization
- Stochastic demand modeling
- Risk management strategies

**GFlowNet Applications**:
- Combinatorial optimization
- Constraint satisfaction
- Multi-modal solution generation

**Operations Research**:
- Production planning
- Inventory management
- Distribution strategy

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

- **Stochastic Elements**: Demand uncertainty, supply disruptions
- **Sustainability**: Carbon footprint optimization
- **Advanced Constraints**: Regulatory compliance, quality requirements
- **Visualization**: Interactive dashboards, network visualization
- **Benchmarking**: Comparison with traditional optimization methods

## 📄 License

This example is part of the GFlowNet.jl package and follows the same license terms.

## 📞 Support

For questions and issues:
1. Check the documentation in `docs/`
2. Run the test suite to identify issues
3. Use the debug utilities in `test/`
4. Refer to the main GFlowNet.jl documentation

---

**🎯 This example demonstrates GFlowNet's power for real-world optimization problems, generating diverse, high-quality solutions for complex supply chain challenges.**
