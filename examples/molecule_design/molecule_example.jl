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

"""
    build_molecular_design_model(; atom_types, lattice_size, max_atoms, hidden_dim, learning_rate, rng)

Assemble the molecular-design GFlowNet with the current on-demand API.

`GFlowNet.create_molecular_design_model` still builds its action set the old way
and then throws
`ArgumentError("Molecular design model needs to be updated to new API. Use
create_grid_world_gflownet() as a reference.")`, so this example follows that
reference: hand the initial state plus the full action set to
`GFlowNet.create_gflownet`, which builds the policy network, the
`ComponentArray` parameters, the optimiser and the per-layer states itself.
"""
function build_molecular_design_model(;
    atom_types = [:C, :H, :O, :N],
    lattice_size::Int = 2,
    max_atoms::Int = 6,
    hidden_dim::Int = 64,
    learning_rate::Float64 = 0.001,
    rng = Random.default_rng()
)
    actions = GFlowNet.AbstractAction[]

    # One AddAtomAction per (element, lattice position) pair. `lattice_size` and
    # `max_atoms` are kept small on purpose: every sampling step evaluates
    # `is_applicable` for the whole action set, so the action count is the main
    # driver of this example's runtime.
    for atom_type in atom_types,
        x in 1:lattice_size, y in 1:lattice_size, z in 1:lattice_size

        push!(actions, AddAtomAction(atom_type, (Float64(x), Float64(y), Float64(z))))
    end

    # Bond actions are enumerated for every index pair; `is_applicable` filters
    # out the ones whose endpoints do not exist yet.
    for i in 1:max_atoms, j in (i + 1):max_atoms, bond_type in 1:3
        push!(actions, AddBondAction(i, j, bond_type))
    end

    push!(actions, TerminateMoleculeAction())

    initial_state = GFlowNet.create_initial_molecule_state()

    return GFlowNet.create_gflownet(
        initial_state,
        actions;
        state_dim = length(state_to_features(initial_state)),
        hidden_dim = hidden_dim,
        learning_rate = learning_rate,
        rng = rng
    )
end

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
    # Using default parameters: C, H, O, N atoms, max 10 atoms
    model = build_molecular_design_model(rng = Random.MersenneTwister(42))
    
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
    
    # Create training configuration for molecular design. This is a demo budget:
    # a molecule trajectory can run to `SamplingConfig`'s 100-step cap, so even a
    # handful of iterations takes tens of seconds.
    #
    # SUB_TRAJECTORY_BALANCE cannot be used here: it evaluates the flow estimator
    # on intermediate states inside the Zygote tape, and
    # `GFlowNet.state_to_features(::MoleculeState)` builds its vector with
    # `push!`, which Zygote rejects ("Mutating arrays is not supported").
    config = GFlowNet.TrainingConfig(
        objective=GFlowNet.TRAJECTORY_BALANCE,
        partition_function_method=GFlowNet.SIMPLE_ESTIMATION,  # Molecular spaces often enumerable
        batch_size=4,
        learning_rate=0.001,
        n_iterations=5,
        validation_frequency=2,
        early_stopping_patience=5
    )

    println("Training configuration for molecular design:")
    println("  Objective: $(config.objective)")
    println("  Partition function method: $(config.partition_function_method) (enumerable chemical spaces)")
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