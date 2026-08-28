#!/usr/bin/env julia

# Example script for molecular design using GFlowNets
# This demonstrates a composition-based approach to defining domain-specific types

# IMPORTANT: This script must be run from the example directory
# Run with: julia molecule_example.jl

using Pkg
Pkg.activate(@__DIR__)  # Activate the project in the current directory (the example directory)

# `examples/**/Manifest.toml` is intentionally untracked (a manifest resolved on
# one Julia version cannot be instantiated on another), so a clean checkout has
# no manifest here and `using GFlowNet` would otherwise die with an opaque
# "required but does not seem to be installed". Project.toml carries a
# `[sources]` entry pointing GFlowNet at the repository root, so resolving and
# installing on first run needs no registry entry and no `Pkg.develop` step.
try
    Pkg.instantiate()
catch err
    @error """Could not instantiate the example environment at $(@__DIR__).
             Run `julia --project=. examples/setup_examples.jl` from the \
             repository root, then re-run this script.""" exception = err
    rethrow()
end

using GFlowNet
using Random
using StatsBase  # Added for the sample function
using Plots  # For visualization of training progress

# The model is built by `GFlowNet.create_molecular_design_model`. This example used to
# carry its own `build_molecular_design_model` because the library factory threw
# `ArgumentError("Molecular design model needs to be updated to new API…")`; that
# factory now works, so the local copy is gone.

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
    
    # Create the GFlowNet model for molecular design. `max_atoms`/`max_bonds` are kept
    # small on purpose: every sampling step evaluates `is_applicable` over the whole
    # action set, so the action count is the main driver of this example's runtime.
    model = create_molecular_design_model(
        max_atoms = 6,
        max_bonds = 6,
        learning_rate = 0.001,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,
        rng = Random.MersenneTwister(42)
    )
    
    println("Created molecular design model")
    println("Initial state:")
    initial_state = GFlowNet.create_initial_molecule_state()
    GFlowNet.visualize_molecule(initial_state)
    
    # Define available actions
    println("\nAvailable actions at initial state:")
    available_actions = filter(a -> is_applicable(a, initial_state), model.all_actions)
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
        GFlowNet.visualize_molecule(new_state)
        
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
    
    # Create training configuration for molecular design. This is a demo budget: a
    # molecule trajectory here runs up to max_atoms + max_bonds + 1 = 13 steps, and a
    # step costs one `is_applicable` sweep over all 50 actions, so even a handful of
    # iterations takes tens of seconds.
    #
    # LEARNABLE_ESTIMATION, not SIMPLE_ESTIMATION: this domain's terminal set is not
    # enumerable (500 samples from the trained model produced 489 distinct molecules),
    # so pinning Z = 1 makes the trajectory-balance residual irreducible. Measured
    # over 40 iterations, seed 1234: with SIMPLE_ESTIMATION the loss went 233.2 ->
    # 448.6 and sampling gave mean R = 0.589; with LEARNABLE_ESTIMATION it went
    # 233.2 -> 227.6, Z was learned as 4.3e4, and mean R rose to 0.619.
    #
    # SUB_TRAJECTORY_BALANCE is still not used here, but for a different reason than
    # before: it no longer dies on `state_to_features` (that function is mutation-free
    # now, so Zygote accepts it) and `sub_trajectory_balance_loss_batch` returns a
    # finite 174.7 on a fresh 4-atom model. Training it for two iterations
    # nevertheless reports NaN losses. The flow estimator is unconstrained and emits
    # negative F on this domain (-0.61 to +0.08 over one measured trajectory), which
    # `balance.jl`'s `log(max(F, 1e-8))` clamps to a constant with no gradient, so the
    # divergence is in the SubTB/flow path, not in the molecule domain.
    config = GFlowNet.TrainingConfig(
        objective=GFlowNet.TRAJECTORY_BALANCE,
        partition_function_method=GFlowNet.LEARNABLE_ESTIMATION,
        batch_size=4,
        learning_rate=0.001,
        n_iterations=5,
        validation_frequency=2,
        early_stopping_patience=5
    )

    println("Training configuration for molecular design:")
    println("  Objective: $(config.objective)")
    println("  Partition function method: $(config.partition_function_method) (Z is learned; the terminal set is not enumerable)")
    println("  Batch size: $(config.batch_size)")
    println("  Iterations: $(config.n_iterations)")
    
    # Train the model
    training_history = GFlowNet.train_gflownet(model, config; verbose=true)
    
    println("\nTraining completed!")
    println("  Final loss: $(round(training_history[:losses][end], digits=6))")
    println("  Mean gradient norm: $(round(sum(training_history[:gradient_norms]) / length(training_history[:gradient_norms]), digits=6))")
    println("  Total training iterations: $(length(training_history[:losses]))")
    
    # Sample diverse molecules after training
    println("\n" * "="^60)
    println("SAMPLING TRAINED MOLECULES")
    println("="^60)
    
    n_samples = 5
    sampled_trajectories = [GFlowNet.sample_trajectory(model) for _ in 1:n_samples]
    sampled_molecules = [traj.states[end] for traj in sampled_trajectories]
    
    println("Generated $(length(sampled_molecules)) molecules:")
    for (i, molecule) in enumerate(sampled_molecules)
        reward_val = GFlowNet.reward(molecule)
        println("\nMolecule $i (Reward: $(round(reward_val, digits=4))):")
        GFlowNet.visualize_molecule(molecule)
    end
    
    # Find best molecule
    rewards = [GFlowNet.reward(mol) for mol in sampled_molecules]
    best_idx = argmax(rewards)
    best_molecule = sampled_molecules[best_idx]
    
    println("\n" * "="^60)
    println("BEST DISCOVERED MOLECULE")
    println("="^60)
    println("Reward: $(round(rewards[best_idx], digits=4))")
    GFlowNet.visualize_molecule(best_molecule)
    
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
    # Write next to the script, not into whatever directory the user launched
    # from -- this example is normally run as `julia examples/molecule_design/…`
    # from the repository root.
    savefig(loss_plot, joinpath(@__DIR__, "molecule_design_loss.png"))
    
    # Plot reward distribution
    reward_plot = histogram(
        rewards,
        title="Generated Molecule Rewards",
        xlabel="Reward",
        ylabel="Frequency",
        bins=10,
        legend=false
    )
    savefig(reward_plot, joinpath(@__DIR__, "molecule_design_rewards.png"))
    
    println("\nVisualization saved to molecule_design_*.png")
    
    return model, current_state, training_history, sampled_molecules
end

# Run the example if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_molecule_example()
end 