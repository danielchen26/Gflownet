# 🏥 Pharmaceutical Supply Chain Optimization with GFlowNets

A comprehensive example demonstrating GFlowNet optimization for complex pharmaceutical supply chain problems, comparing against traditional optimization methods with professional reporting.

## 🎯 Overview

This example showcases the application of Generative Flow Networks (GFlowNets) to optimize pharmaceutical supply chains, addressing the complex multi-objective challenge of balancing:

- **💰 Cost Efficiency (30%)** - Minimize total supply chain costs
- **🏥 Patient Access (30%)** - Maximize medication availability across regions  
- **📋 Regulatory Compliance (20%)** - Ensure adherence to global regulations
- **🔗 Supply Chain Resilience (20%)** - Build robust, disruption-resistant networks

## 🚀 Quick Start

```bash
# Navigate to the example directory
cd examples/pharmaceutical_supply_chain

# Install dependencies
julia --project -e "using Pkg; Pkg.instantiate()"

# Run the comprehensive comparison
julia --project pharmaceutical_supply_chain.jl
```

## 📊 Methods Compared

1. **🧠 GFlowNets** - Generative Flow Networks for diverse, high-quality solutions
2. **📐 Nonlinear Programming** - Mathematical optimization with Ipopt solver
3. **⛰️ Hill Climbing** - Local search with multiple restarts
4. **🔍 Greedy** - Greedy optimization algorithm
5. **🌡️ Simulated Annealing** - Temperature-based probabilistic search
6. **🎲 Random Search** - Baseline random sampling method

## 📁 Project Structure

```
pharmaceutical_supply_chain/
├── pharmaceutical_supply_chain.jl    # Main example file
├── report_generation.jl              # Professional HTML report generation
├── src/
│   ├── gflownet_interface.jl         # Clean GFlowNet interface
│   └── baseline_methods.jl           # Traditional optimization methods
├── results/                          # Generated reports and data
├── resources/                        # Documentation and analysis
├── Project.toml                      # Julia project configuration
└── README.md                         # This file
```

## 🔬 Mathematical Problem

The optimization problem is formulated as:

**Maximize:** `R(x,p,t,s,u) = w₁·CE(x,p,u) + w₂·PA(x,p,s) + w₃·RC(x) + w₄·SR(x)`

Where:
- `x_f` = Binary facility activation variables
- `p_{f,d}` = Production quantities by facility and drug
- `t_{i,j,d}` = Transport flows between facilities
- `s_{f,d}` = Inventory levels
- `u_f` = Facility utilization rates

**Subject to:**
- Production capacity constraints
- Flow conservation equations
- Demand satisfaction requirements
- Regulatory compliance constraints
- Network viability requirements

## 📈 Results & Analysis

The example generates comprehensive results including:

### 📄 HTML Report
- Executive summary with key findings
- Detailed methodology and problem description
- Performance comparison tables
- High-quality visualizations
- Data export links

### 📊 Visualizations
- Performance comparison bar charts
- Diversity vs performance scatter plots
- Multi-criteria radar charts
- Solution quality distributions

### 💾 Data Export
- Summary statistics (CSV)
- Detailed results per method (CSV)
- Raw solution data for further analysis

## 🏆 Key Findings

Based on comprehensive testing, **GFlowNets consistently outperform traditional methods** by:

- **🎯 Superior Solution Quality** - Finding higher-reward solutions
- **🌈 Better Diversity** - Generating varied, non-dominated solutions
- **⚖️ Multi-Objective Balance** - Effectively balancing competing objectives
- **🔄 Robust Performance** - Consistent results across multiple runs

## 🛠️ Dependencies

### Required
- Julia 1.9+
- GFlowNet.jl (pharmaceutical supply chain module)
- JuMP.jl (mathematical optimization)
- Ipopt.jl (nonlinear programming solver)

### Optional (Enhanced Features)
- Plots.jl (high-quality visualizations)
- DataFrames.jl & CSV.jl (data export)

## 📚 Documentation

Detailed documentation is available in the `resources/docs/` directory:

- `MATHEMATICAL_PROBLEM_DEFINITION.md` - Complete mathematical formulation
- Generated HTML reports with comprehensive analysis

## 🤝 Contributing

This example demonstrates best practices for:
- Clean code organization and modular design
- Comprehensive method comparison and evaluation
- Professional report generation and visualization
- Mathematical problem formulation and validation

## 📄 License

This example is part of the GFlowNet.jl package and follows the same license terms.

---

*For questions or issues, please refer to the main GFlowNet.jl documentation or open an issue on the repository.*
