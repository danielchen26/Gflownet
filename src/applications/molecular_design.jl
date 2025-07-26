using ..GFlowNet: AbstractState, AbstractAction, state_to_features, is_applicable, apply_action, reward

"""
    MoleculeData

Data structure for molecular information. Used with composition pattern
to create domain-specific states.

# Fields
- `atoms`: Vector of atom types
- `bonds`: Vector of bonds as tuples (atom1, atom2, bond_type)
"""
struct MoleculeData
    atoms::Vector{Symbol}
    bonds::Vector{Tuple{Int,Int,Int}}  # (atom1, atom2, bond_type)
end

"""
    MoleculeState <: AbstractState

A state representing a molecule in construction.
"""
struct MoleculeState <: AbstractState
    data::MoleculeData
    complete::Bool
end

"""
    AddAtomAction <: AbstractAction

An action that adds an atom to the molecule.
"""
struct AddAtomAction <: AbstractAction
    atom_type::Symbol
    position::Tuple{Float64,Float64,Float64}
end

"""
    AddBondAction <: AbstractAction

An action that adds a bond between two atoms in the molecule.
"""
struct AddBondAction <: AbstractAction
    atom1_idx::Int
    atom2_idx::Int
    bond_type::Int  # 1 = single, 2 = double, 3 = triple
end

"""
    TerminateMoleculeAction <: AbstractAction

An action that marks the molecule as complete.
"""
struct TerminateMoleculeAction <: AbstractAction end

# Implementation of required methods for the GFlowNet framework

"""
    state_to_features(state::MoleculeState)

Convert a molecule state to a feature vector for neural network inputs.
"""
function state_to_features(state::MoleculeState)
    # This is a simple implementation - in a real application,
    # you would use more sophisticated molecular featurization

    # Basic features: number of atoms, number of bonds, is_complete
    features = Float32[
        length(state.data.atoms),
        length(state.data.bonds),
        state.complete ? 1.0 : 0.0
    ]

    # Atom type counts
    atom_types = [:C, :H, :O, :N, :S, :P, :F, :Cl, :Br, :I]
    for atom_type in atom_types
        count_val = Base.count(a -> a == atom_type, state.data.atoms)
        push!(features, count_val)
    end

    return features
end

"""
    reward(state::MoleculeState)

Calculate the reward for a molecule state based on its properties.
"""
function reward(state::MoleculeState)
    if !state.complete
        return 0.0
    end

    # In a real application, you would compute molecular properties
    # like drug-likeness, binding affinity, etc.

    # Simple example reward: prefer molecules with 5-20 atoms
    n_atoms = length(state.data.atoms)
    if n_atoms < 5
        return 0.1
    elseif n_atoms > 20
        return 0.1
    else
        # Higher reward for medium-sized molecules
        return 1.0 - 0.05 * abs(n_atoms - 12.5)
    end
end

"""
    is_applicable(action::AddAtomAction, state::MoleculeState)

Check if adding an atom is valid for the current molecule state.
"""
function is_applicable(action::AddAtomAction, state::MoleculeState)
    # Can't add atoms to a completed molecule
    if state.complete
        return false
    end

    # Limit the total number of atoms
    if length(state.data.atoms) >= 20
        return false
    end

    # Additional checks could include:
    # - Spatial constraints
    # - Valence rules
    # - Chemical feasibility

    return true
end

"""
    is_applicable(action::AddBondAction, state::MoleculeState)

Check if adding a bond is valid for the current molecule state.
"""
function is_applicable(action::AddBondAction, state::MoleculeState)
    # Can't add bonds to a completed molecule
    if state.complete
        return false
    end

    # Check if the atom indices are valid
    if action.atom1_idx <= 0 || action.atom2_idx <= 0 ||
       action.atom1_idx > length(state.data.atoms) ||
       action.atom2_idx > length(state.data.atoms) ||
       action.atom1_idx == action.atom2_idx
        return false
    end

    # Check if the bond already exists
    for (a1, a2, _) in state.data.bonds
        if (a1 == action.atom1_idx && a2 == action.atom2_idx) ||
           (a1 == action.atom2_idx && a2 == action.atom1_idx)
            return false
        end
    end

    # Additional checks could include:
    # - Valence rules
    # - Distance between atoms
    # - Chemical feasibility

    return true
end

"""
    is_applicable(action::TerminateMoleculeAction, state::MoleculeState)

Check if terminating the molecule is valid for the current state.
"""
function is_applicable(action::TerminateMoleculeAction, state::MoleculeState)
    # Can't terminate an already completed molecule
    if state.complete
        return false
    end

    # Check if the molecule is valid for termination
    # In a real application, you would check for:
    # - Valid valence for all atoms
    # - Chemical stability
    # - Structural integrity

    # Simple check: must have at least one atom
    return length(state.data.atoms) > 0
end

"""
    apply_action(action::AddAtomAction, state::MoleculeState)

Apply an AddAtomAction to the current state, returning a new state.
"""
function apply_action(action::AddAtomAction, state::MoleculeState)
    # Create a new state with the atom added
    new_atoms = copy(state.data.atoms)
    push!(new_atoms, action.atom_type)

    new_data = MoleculeData(
        new_atoms,
        copy(state.data.bonds)
    )

    return MoleculeState(new_data, false)
end

"""
    apply_action(action::AddBondAction, state::MoleculeState)

Apply an AddBondAction to the current state, returning a new state.
"""
function apply_action(action::AddBondAction, state::MoleculeState)
    # Create a new state with the bond added
    new_bonds = copy(state.data.bonds)
    push!(new_bonds, (action.atom1_idx, action.atom2_idx, action.bond_type))

    new_data = MoleculeData(
        copy(state.data.atoms),
        new_bonds
    )

    return MoleculeState(new_data, false)
end

"""
    apply_action(action::TerminateMoleculeAction, state::MoleculeState)

Apply a TerminateMoleculeAction to the current state, returning a new state.
"""
function apply_action(action::TerminateMoleculeAction, state::MoleculeState)
    # Create a new state marked as complete
    return MoleculeState(
        MoleculeData(copy(state.data.atoms), copy(state.data.bonds)),
        true
    )
end

"""
    create_initial_molecule_state()

Create an empty initial molecule state.
"""
function create_initial_molecule_state()
    return MoleculeState(
        MoleculeData(Symbol[], Tuple{Int,Int,Int}[]),
        false
    )
end

"""
    create_molecular_design_model(atom_types, max_atoms, max_bonds)

Create a GFlowNet model for molecular design.
"""
function create_molecular_design_model(atom_types=[:C, :H, :O, :N], max_atoms=10, max_bonds=15)
    # Create the initial state
    initial_state = create_initial_molecule_state()

    # Create actions
    actions = AbstractAction[]

    # Add atom actions
    for atom_type in atom_types
        for x in 1:3, y in 1:3, z in 1:3
            push!(actions, AddAtomAction(atom_type, (Float64(x), Float64(y), Float64(z))))
        end
    end

    # Add bond actions (will be filtered by is_applicable)
    for i in 1:max_atoms
        for j in (i+1):max_atoms
            for bond_type in 1:3
                push!(actions, AddBondAction(i, j, bond_type))
            end
        end
    end

    # Add terminate action
    push!(actions, TerminateMoleculeAction())

    # The terminal states are empty at first - they'll be discovered during sampling
    terminal_states = MoleculeState[]

    # Create a special terminal sink state
    terminal_sink = MoleculeState(
        MoleculeData(Symbol[:SINK], Tuple{Int,Int,Int}[]),
        true
    )

    # Create the DAG
    dag = create_dag(initial_state, terminal_states, terminal_sink, actions)

    # Create a simple neural network for the forward policy
    # In a real application, you would use a more sophisticated model
    # that could handle molecular graphs

    # For demonstration purposes only:
    forward_policy = ForwardPolicy(identity) # Placeholder

    # Create the GFlowNet model
    return GFlowNetModel(
        dag,
        forward_policy,
        nothing,  # No backward policy
        nothing,  # No flow estimator
        nothing,  # No partition function
        [TrajectoryBalanceObjective(1.0)],
        nothing,  # No optimizer
        (forward=nothing, backward=nothing, flow=nothing),  # No parameters
        (forward=nothing, backward=nothing, flow=nothing)   # No states
    )
end

# Function to visualize molecules using the composition-based approach
function visualize_molecule(data::MoleculeData)
    println("Molecule Visualization:")
    println("  Atoms ($(length(data.atoms))): ", join(data.atoms, ", "))

    if !isempty(data.bonds)
        println("  Bonds ($(length(data.bonds))):")
        for (a1, a2, type) in data.bonds
            bond_symbol = type == 1 ? "-" : (type == 2 ? "=" : "≡")
            println("    $(data.atoms[a1]) $bond_symbol $(data.atoms[a2])")
        end
    else
        println("  No bonds")
    end
end

function visualize_molecule(state::MoleculeState)
    visualize_molecule(state.data)
    println("  Status: ", state.complete ? "Complete" : "In Progress")
end

# Add equality methods for proper comparison in DAG creation
import Base: ==

function ==(a::MoleculeState, b::MoleculeState)
    return a.complete == b.complete &&
           length(a.data.atoms) == length(b.data.atoms) &&
           all(a.data.atoms .== b.data.atoms) &&
           length(a.data.bonds) == length(b.data.bonds) &&
           all(a.data.bonds .== b.data.bonds)
end

function ==(a::MoleculeData, b::MoleculeData)
    return length(a.atoms) == length(b.atoms) &&
           all(a.atoms .== b.atoms) &&
           length(a.bonds) == length(b.bonds) &&
           all(a.bonds .== b.bonds)
end

function ==(a::AddAtomAction, b::AddAtomAction)
    return a.atom_type == b.atom_type && a.position == b.position
end

function ==(a::AddBondAction, b::AddBondAction)
    return a.atom1_idx == b.atom1_idx &&
           a.atom2_idx == b.atom2_idx &&
           a.bond_type == b.bond_type
end

function ==(a::TerminateMoleculeAction, b::TerminateMoleculeAction)
    return true  # All terminate actions are equivalent
end

# Add hash methods for our types to allow them to be used in dictionaries/sets
import Base: hash

function hash(state::MoleculeState, h::UInt)
    h = hash(:MoleculeState, h)
    h = hash(state.complete, h)
    for atom in state.data.atoms
        h = hash(atom, h)
    end
    for bond in state.data.bonds
        h = hash(bond, h)
    end
    return h
end

function hash(action::AddAtomAction, h::UInt)
    return hash((:AddAtomAction, action.atom_type, action.position), h)
end

function hash(action::AddBondAction, h::UInt)
    return hash((:AddBondAction, action.atom1_idx, action.atom2_idx, action.bond_type), h)
end

function hash(action::TerminateMoleculeAction, h::UInt)
    return hash(:TerminateMoleculeAction, h)
end

# Add display methods for prettier printing
import Base: show

function show(io::IO, state::MoleculeState)
    status = state.complete ? "complete" : "in progress"
    print(io, "MoleculeState($(length(state.data.atoms)) atoms, $(length(state.data.bonds)) bonds, $status)")
end

function show(io::IO, action::AddAtomAction)
    print(io, "AddAtomAction($(action.atom_type) at $(action.position))")
end

function show(io::IO, action::AddBondAction)
    print(io, "AddBondAction($(action.atom1_idx)-$(action.atom2_idx) type:$(action.bond_type))")
end

function show(io::IO, action::TerminateMoleculeAction)
    print(io, "TerminateMoleculeAction()")
end
