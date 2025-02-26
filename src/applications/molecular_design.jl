using ..GFlowNet: AbstractState, AbstractAction, state_to_features, is_applicable, apply_action, reward

"""
    MoleculeState <: AbstractState

State representation for molecular graphs in GFlowNets.
"""
struct MoleculeState <: AbstractState
    atoms::Vector{Symbol}  # Atom types
    bonds::Vector{Tuple{Int, Int, Int}}  # (atom1, atom2, bond_type)
    is_terminal::Bool
end

"""
    MoleculeAction <: AbstractAction

Action representation for molecular graph building.
"""
abstract type MoleculeAction <: AbstractAction end

"""
    AddAtomAction <: MoleculeAction

Action to add an atom to a molecule.
"""
struct AddAtomAction <: MoleculeAction
    atom_type::Symbol  # Type of atom to add (e.g., :C, :N, :O)
end

"""
    AddBondAction <: MoleculeAction

Action to add a bond between two atoms.
"""
struct AddBondAction <: MoleculeAction
    atom1::Int  # Index of first atom
    atom2::Int  # Index of second atom
    bond_type::Int  # Bond type (1: single, 2: double, 3: triple)
end

"""
    TerminateAction <: MoleculeAction

Action to terminate molecule construction.
"""
struct TerminateAction <: MoleculeAction end

# Implementation of required interface functions

"""
    is_applicable(action::AddAtomAction, state::MoleculeState)

Check if adding an atom is valid.
"""
function is_applicable(action::AddAtomAction, state::MoleculeState)
    # Can add an atom if the state is not terminal and molecule size is within limits
    return !state.is_terminal && length(state.atoms) < 50  # Arbitrary limit
end

"""
    is_applicable(action::AddBondAction, state::MoleculeState)

Check if adding a bond is valid.
"""
function is_applicable(action::AddBondAction, state::MoleculeState)
    # Cannot add bonds to a terminal state
    if state.is_terminal
        return false
    end
    
    # Check if atom indices are valid
    if action.atom1 > length(state.atoms) || action.atom2 > length(state.atoms)
        return false
    end
    
    # Check if atoms are different
    if action.atom1 == action.atom2
        return false
    end
    
    # Check if bond already exists
    for (a1, a2, _) in state.bonds
        if (a1 == action.atom1 && a2 == action.atom2) || 
           (a1 == action.atom2 && a2 == action.atom1)
            return false
        end
    end
    
    # Check valence constraints (simplified)
    # Count existing bonds for each atom
    atom1_bonds = count(b -> b[1] == action.atom1 || b[2] == action.atom1, state.bonds)
    atom2_bonds = count(b -> b[1] == action.atom2 || b[2] == action.atom2, state.bonds)
    
    # Get maximum valence based on atom type
    max_valence1 = get_max_valence(state.atoms[action.atom1])
    max_valence2 = get_max_valence(state.atoms[action.atom2])
    
    # Check if adding this bond would exceed valence limits
    return (atom1_bonds + action.bond_type <= max_valence1) && 
           (atom2_bonds + action.bond_type <= max_valence2)
end

"""
    is_applicable(action::TerminateAction, state::MoleculeState)

Check if termination is valid.
"""
function is_applicable(action::TerminateAction, state::MoleculeState)
    # Can terminate if there's at least one atom and not already terminated
    return !state.is_terminal && !isempty(state.atoms)
end

"""
    apply_action(action::AddAtomAction, state::MoleculeState)

Apply the action to add an atom.
"""
function apply_action(action::AddAtomAction, state::MoleculeState)
    # Create new state with added atom
    new_atoms = copy(state.atoms)
    push!(new_atoms, action.atom_type)
    
    return MoleculeState(new_atoms, copy(state.bonds), false)
end

"""
    apply_action(action::AddBondAction, state::MoleculeState)

Apply the action to add a bond.
"""
function apply_action(action::AddBondAction, state::MoleculeState)
    # Create new state with added bond
    new_bonds = copy(state.bonds)
    
    # Ensure atom1 < atom2 for consistency
    atom1, atom2 = minmax(action.atom1, action.atom2)
    
    push!(new_bonds, (atom1, atom2, action.bond_type))
    
    return MoleculeState(copy(state.atoms), new_bonds, false)
end

"""
    apply_action(action::TerminateAction, state::MoleculeState)

Apply the action to terminate molecule construction.
"""
function apply_action(action::TerminateAction, state::MoleculeState)
    # Create a new terminal state
    return MoleculeState(copy(state.atoms), copy(state.bonds), true)
end

"""
    state_to_features(state::MoleculeState)

Convert a molecule state to a feature vector.
"""
function state_to_features(state::MoleculeState)
    # This is a simplified representation
    # In practice, you would use more sophisticated molecular featurization
    
    # Count atom types
    atom_counts = Dict{Symbol, Int}()
    for atom in state.atoms
        atom_counts[atom] = get(atom_counts, atom, 0) + 1
    end
    
    # Count bond types
    bond_counts = Dict{Int, Int}()
    for (_, _, bond_type) in state.bonds
        bond_counts[bond_type] = get(bond_counts, bond_type, 0) + 1
    end
    
    # Create feature vector
    # [C count, N count, O count, H count, single bonds, double bonds, triple bonds, is_terminal]
    features = [
        get(atom_counts, :C, 0),
        get(atom_counts, :N, 0),
        get(atom_counts, :O, 0),
        get(atom_counts, :H, 0),
        get(bond_counts, 1, 0),  # single bonds
        get(bond_counts, 2, 0),  # double bonds
        get(bond_counts, 3, 0),  # triple bonds
        Int(state.is_terminal)
    ]
    
    return Float32.(features)
end

"""
    reward(state::MoleculeState)

Calculate the reward for a molecule state.
This would typically be based on properties like drug-likeness, 
synthetic accessibility, binding affinity, etc.
"""
function reward(state::MoleculeState)
    if !state.is_terminal
        return 0.0
    end
    
    # Mock reward function - in practice, this would involve predictions from 
    # property predictors, docking scores, etc.
    
    # Simple heuristic: reward based on molecule size, presence of key features
    # Number of atoms
    n_atoms = length(state.atoms)
    
    # Checks for specific elements
    has_nitrogen = :N in state.atoms
    has_oxygen = :O in state.atoms
    
    # Check for aromatic rings (very simplified)
    has_ring = false
    if length(state.bonds) >= 6
        # More sophisticated ring detection would be needed in practice
        has_ring = true
    end
    
    # Basic Lipinski's Rule of 5 check (extremely simplified)
    # In practice, you would compute actual molecular weight, logP, etc.
    rule_of_5_score = 0
    if 10 <= n_atoms <= 50  # Simple proxy for molecular weight
        rule_of_5_score += 1
    end
    
    # Combine factors
    base_score = 1.0
    size_factor = exp(-(n_atoms - 20)^2 / 100)  # Prefer ~20 atoms
    element_bonus = (has_nitrogen ? 1.2 : 1.0) * (has_oxygen ? 1.2 : 1.0)
    structure_bonus = has_ring ? 1.5 : 1.0
    
    return base_score * size_factor * element_bonus * structure_bonus * (1.0 + 0.2 * rule_of_5_score)
end

"""
    get_max_valence(atom_type::Symbol)

Get the maximum valence for an atom type.
"""
function get_max_valence(atom_type::Symbol)
    valences = Dict(
        :H => 1,
        :C => 4,
        :N => 3,
        :O => 2,
        :F => 1,
        :Cl => 1,
        :Br => 1,
        :I => 1,
        :S => 6,
        :P => 5
    )
    
    return get(valences, atom_type, 4)  # Default to 4 if unknown
end

"""
    create_molecule_actions()

Create a set of possible molecule building actions.
"""
function create_molecule_actions()
    actions = MoleculeAction[]
    
    # Add atom actions
    for atom_type in [:C, :N, :O, :H, :F, :Cl, :S]
        push!(actions, AddAtomAction(atom_type))
    end
    
    # Add bond actions - these will be filtered for validity at runtime
    # In a real implementation, we would generate these dynamically based on
    # the current molecule state
    for i in 1:10, j in i+1:10, bond_type in 1:3
        push!(actions, AddBondAction(i, j, bond_type))
    end
    
    # Add terminate action
    push!(actions, TerminateAction())
    
    return actions
end

"""
    create_initial_molecule_state()

Create the initial state for molecule building.
"""
function create_initial_molecule_state()
    return MoleculeState(Symbol[], Tuple{Int, Int, Int}[], false)
end 