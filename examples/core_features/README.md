# Core Features Examples

This directory contains examples demonstrating core GFlowNet.jl features that apply across all domains.

## Available Examples

### 1. Learnable Partition Function (`learnable_partition_function/`)

Demonstrates the LEARNABLE_ESTIMATION method for learning the partition function Z as a parameter during training. This is essential for:
- Multi-start GFlowNets
- Theoretical completeness
- Flow conservation validation

## Structure

This directory uses a shared Project.toml for all core feature examples, making it easy to run any example after a single installation.

Each core feature example includes:
- A standalone Julia script demonstrating the feature
- A README explaining the concepts and usage
- Generated visualizations and results

## Running Examples

First, set up the environment with the local GFlowNet package and install dependencies:
```bash
cd examples/core_features
julia --project=. -e "using Pkg; Pkg.develop(path=\"../..\")"
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Then run any example:
```bash
julia --project=. <feature_name>/<example_script>.jl
```

For example:
```bash
julia --project=. learnable_partition_function/learnable_z_demo.jl
```

## Adding New Core Feature Examples

When adding a new core feature example:
1. Create a new subdirectory under `core_features/`
2. Include a Project.toml with necessary dependencies
3. Write a comprehensive example script
4. Add a README explaining the feature
5. Update this file to include your example

## Difference from Domain Examples

While the main `examples/` directory contains domain-specific implementations (grid world, molecules, etc.), this `core_features/` directory focuses on cross-cutting functionality that applies to all domains.