#!/usr/bin/env julia

# Example script for molecular design using GFlowNets
# This demonstrates the composition-based approach to defining domain-specific types

# IMPORTANT: This script must be run from the project root directory
# Run with: julia examples/molecule_example.jl

using Pkg
Pkg.activate(".")  # Activate the project in the current directory (should be the project root)

using GFlowNet
using Random
using StatsBase  # Added for the sample function

"""
    run_molecule_example()

Run a simple example of molecular design using GFlowNets.

This example demonstrates:
1. Creating a model for molecular design using composition
2. Generating molecule states by applying sequences of actions
3. Visualizing molecule states and calculating rewards
4. Computing feature vectors for neural network inputs

The composition approach uses:
- MoleculeData: Basic data structure for molecular information
- MoleculeState: State type that contains MoleculeData
- Various action types: AddAtomAction, AddBondAction, TerminateMoleculeAction

This design allows for clean separation between data and behavior,
making it easy to extend with new functionality.
"""
function run_molecule_example()
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Create the GFlowNet model for molecular design
    # Using default parameters: C, H, O, N atoms, max 10 atoms, max 15 bonds
    model = create_molecular_design_model()
    
    println("Created molecular design model")
    println("Initial state:")
    initial_state = create_initial_molecule_state()
    visualize_molecule(initial_state)
    
    # Define available actions
    println("\nAvailable actions at initial state:")
    available_actions = filter(a -> is_applicable(a, initial_state), model.dag.actions)
    println("  $(length(available_actions)) actions available")
    
    # Select a few random actions
    selected_actions = sample(available_actions, min(5, length(available_actions)))
    
    # Apply each action to demonstrate state transitions
    current_state = initial_state
    println("\nSample molecule construction sequence:")
    
    for (i, action) in enumerate(selected_actions)
        # Apply the action
        new_state = apply_action(action, current_state)
        
        # Print the action and resulting state
        println("\nStep $i:")
        if action isa AddAtomAction
            println("  Added atom: $(action.atom_type) at position $(action.position)")
        elseif action isa AddBondAction
            println("  Added bond: $(action.atom1_idx)-$(action.atom2_idx) (type: $(action.bond_type))")
        elseif action isa TerminateMoleculeAction
            println("  Terminated molecule construction")
        end
        
        # Visualize the new state
        visualize_molecule(new_state)
        
        # Update current state
        current_state = new_state
    end
    
    # Demonstrate reward calculation
    if current_state.complete
        r = reward(current_state)
        println("\nReward for final molecule: $r")
    else
        # Create a terminated version for demonstration
        terminated_state = apply_action(TerminateMoleculeAction(), current_state)
        r = reward(terminated_state)
        println("\nReward if terminated: $r")
    end
    
    println("\nFeature vector for final state:")
    features = state_to_features(current_state)
    println(features)
    
    return model, current_state
end

# Run the example if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_molecule_example()
end 