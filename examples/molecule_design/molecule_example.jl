#!/usr/bin/env julia

# Example script for molecular design using GFlowNets
# This demonstrates a composition-based approach to defining domain-specific types

# IMPORTANT: This script must be run from the example directory
# Run with: julia molecule_example.jl

using Pkg
Pkg.activate(@__DIR__)  # Activate the project in the current directory (the example directory)

using GFlowNet
using Random
using StatsBase  # Added for the sample function
using Plots  # For visualization of training progress

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
    
    # Now train the model using modern interface
    println("\n" * "="^60)
    println("TRAINING MOLECULAR DESIGN GFLOWNET")
    println("="^60)
    
    # Create training configuration optimized for molecular design
    config = GFlowNet.TrainingConfig(
        objective=GFlowNet.HIERARCHICAL_SUB_TB,  # Multi-scale molecular structure
        partition_function_method=GFlowNet.SIMPLE_ESTIMATION,  # Molecular spaces often enumerable
        batch_size=32,
        learning_rate=0.001,
        n_iterations=1500,
        partition_update_frequency=50,
        validation_frequency=100,
        early_stopping_patience=200,
        sub_trajectory_config=Dict(
            :scales => [2, 4, 8, 16],  # Different molecular scales (bonds, rings, motifs, molecules)
            :n_subtrajectories => 5
        )
    )
    
    println("Training configuration for molecular design:")
    println("  Objective: $(config.objective) (hierarchical balance for multi-scale molecular structure)")
    println("  Partition function method: $(config.partition_function_method) (enumerable chemical spaces)")
    println("  Batch size: $(config.batch_size)")
    println("  Iterations: $(config.n_iterations)")
    println("  Molecular scales: $(config.sub_trajectory_config[:scales])")
    
    # Train the model
    training_history = GFlowNet.train_gflownet(model, config; verbose=true)
    
    println("\nTraining completed!")
    println("  Final loss: $(round(training_history[:losses][end], digits=6))")
    println("  Final Z estimate: $(round(training_history[:partition_function_estimates][end], digits=6))")
    println("  Total training iterations: $(length(training_history[:losses]))")
    
    # Sample diverse molecules after training
    println("\n" * "="^60)
    println("SAMPLING TRAINED MOLECULES")
    println("="^60)
    
    n_samples = 10
    sampled_trajectories = [GFlowNet.sample_trajectory(model) for _ in 1:n_samples]
    sampled_molecules = [traj.states[end] for traj in sampled_trajectories]
    
    println("Generated $(length(sampled_molecules)) molecules:")
    for (i, molecule) in enumerate(sampled_molecules)
        reward_val = GFlowNet.reward(molecule)
        println("\nMolecule $i (Reward: $(round(reward_val, digits=4))):")
        visualize_molecule(molecule)
    end
    
    # Find best molecule
    rewards = [GFlowNet.reward(mol) for mol in sampled_molecules]
    best_idx = argmax(rewards)
    best_molecule = sampled_molecules[best_idx]
    
    println("\n" * "="^60)
    println("BEST DISCOVERED MOLECULE")
    println("="^60)
    println("Reward: $(round(rewards[best_idx], digits=4))")
    visualize_molecule(best_molecule)
    
    # Plot training progress
    
    loss_plot = plot(
        1:length(training_history[:losses]),
        training_history[:losses],
        title="Molecular Design Training Loss",
        xlabel="Iteration",
        ylabel="Loss",
        lw=2,
        legend=false
    )
    savefig(loss_plot, "molecule_design_loss.png")
    
    # Plot reward distribution
    reward_plot = histogram(
        rewards,
        title="Generated Molecule Rewards",
        xlabel="Reward",
        ylabel="Frequency",
        bins=10,
        legend=false
    )
    savefig(reward_plot, "molecule_design_rewards.png")
    
    println("\nVisualization saved to molecule_design_*.png")
    
    return model, current_state, training_history, sampled_molecules
end

# Run the example if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_molecule_example()
end 