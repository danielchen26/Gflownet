# Molecular Design Example

This example demonstrates how to use GFlowNets for molecular design tasks.

## Features
- Shows the composition-based approach for defining domain-specific types
- Implements molecular states and actions (adding atoms, bonds, etc.)
- Demonstrates molecule construction through sequential actions
- Visualizes molecule states and calculates rewards
- Computes feature vectors for neural network inputs

## Running the Example
From the project root directory:
```julia
julia examples/molecule_design/molecule_example.jl
```

## Implementation Details
The example uses:
- `MoleculeData`: Basic data structure for molecular information
- `MoleculeState`: State type that contains MoleculeData
- Various action types: `AddAtomAction`, `AddBondAction`, `TerminateMoleculeAction`

This design allows for clean separation between data and behavior, making it easy to extend with new functionality. 