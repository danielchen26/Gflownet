# Causal Discovery Example

This example demonstrates how GFlowNets can be used for causal discovery, finding directed acyclic graphs (DAGs) that best explain observed data.

## Features
- Generates synthetic causal data with known ground truth
- Creates and trains a GFlowNet to discover causal structures
- Visualizes the discovered causal graphs
- Compares the discovered DAGs with the true underlying structure
- Calculates structural hamming distance as an evaluation metric

## Running the Example
From the project root directory:
```julia
julia examples/causal_discovery/causal_discovery.jl
```

## Output Files
The example generates several visualization files:
- `causal_discovery_loss.png`: Training loss curve
- `causal_discovery_best_dag.png`: Best discovered DAG
- `causal_discovery_true_dag.png`: True underlying DAG
- `causal_discovery_rewards.png`: Distribution of rewards 