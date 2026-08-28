using ..GFlowNet: AbstractState, AbstractAction, state_to_features, is_applicable, apply_action, reward
using ..GFlowNet: PartitionFunctionMethod, SIMPLE_ESTIMATION
using Random

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

# =============================================================================
# Environment configuration
# =============================================================================

# Size limits are properties of the environment, not of a state or an action, so
# `is_applicable` has nowhere else to read them from. Same mechanism as
# `GRID_CONFIG` in grid_world.jl:176.
#
# Before this existed, `create_molecular_design_model`'s `max_atoms` only sized the
# bond-action enumeration and `max_bonds` was accepted and then never read, while
# `is_applicable(::AddAtomAction, …)` hardcoded a cap of 20 atoms.
const MOLECULE_CONFIG = Ref{NamedTuple}()

const DEFAULT_MOLECULE_LIMITS = (max_atoms = 10, max_bonds = 15)

"""
    molecule_limits()

Active `(max_atoms, max_bonds)` limits, or `DEFAULT_MOLECULE_LIMITS` when
`create_molecular_design_model` has not been called in this session.
"""
molecule_limits() = isassigned(MOLECULE_CONFIG) ? MOLECULE_CONFIG[] : DEFAULT_MOLECULE_LIMITS

# Element channels of the feature vector. Fixed and ordered: it defines the input
# layout the policy network is built against.
const FEATURE_ATOM_TYPES = (:C, :H, :O, :N, :S, :P, :F, :Cl, :Br, :I)

# Implementation of required methods for the GFlowNet framework

"""
    GFlowNet.is_terminal_state(state::MoleculeState)

Whether the molecule is complete.

REQUIRED interface method, and it was missing, so any use of this domain failed --
examples/molecule_design/molecule_example.jl had to define it locally to run at all.
Qualified deliberately: this file does not import `is_terminal_state`, so an
unqualified definition would create a new module-local function that the package
never calls. See `grid_world.jl:141`.
"""
GFlowNet.is_terminal_state(state::MoleculeState)::Bool = state.complete

"""
    state_to_features(state::MoleculeState)

Convert a molecule state to a feature vector for neural network inputs.
"""
function state_to_features(state::MoleculeState)
    # Deliberately allocation-then-`vcat` rather than `push!`: Zygote rejects array
    # mutation ("Mutating arrays is not supported"), and SUB_TRAJECTORY_BALANCE
    # differentiates through the features of every intermediate state.
    counts = Float32[Base.count(isequal(t), state.data.atoms) for t in FEATURE_ATOM_TYPES]
    return vcat(
        Float32[
            length(state.data.atoms),
            length(state.data.bonds),
            state.complete ? 1 : 0
        ],
        counts
    )
end

"""
    reward(state::MoleculeState)

Reward of a molecule state: 0 while under construction, and for a completed molecule
a triangular preference for size, peaking at 12-13 atoms.

This is the sole reward method for the domain. A second, unreachable
`base_reward(::MoleculeState)` used to sit here returning `1.0 + bonds/atoms` — a
different scale and a different shape from this function. Nothing in `src/core`
calls `base_reward`, so it was dead weight that contradicted the live reward; it has
been removed rather than reconciled.
"""
function reward(state::MoleculeState)
    if !state.complete
        return 0.0
    end

    # In a real application, you would compute molecular properties
    # like drug-likeness, binding affinity, etc.

    # Simple example reward: prefer molecules with 5-20 atoms. Note that with the
    # default `max_atoms = 10` the peak at 12.5 is out of reach, so the reward is
    # monotone in size over the reachable states and the optimum is "fill the cap".
    n_atoms = length(state.data.atoms)
    if n_atoms < 5 || n_atoms > 20
        return 0.1
    end
    return 1.0 - 0.05 * abs(n_atoms - 12.5)
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
    if length(state.data.atoms) >= molecule_limits().max_atoms
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

    # Limit the total number of bonds
    if length(state.data.bonds) >= molecule_limits().max_bonds
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
    create_molecular_design_model(; atom_types, lattice_size, max_atoms, max_bonds,
                                  hidden_dim, learning_rate, include_flow_estimator,
                                  partition_function_method, rng)

Create a GFlowNet model for lattice-based molecular design.

Builds the action set — one `AddAtomAction` per (element, lattice site), one
`AddBondAction` per (atom-index pair, bond order), one `TerminateMoleculeAction` —
installs the size limits that `is_applicable` enforces, and hands the whole thing to
[`create_gflownet`](@ref), which owns the policy network, the `ComponentArray`
parameters, the optimiser and the per-layer states.

# Arguments
- `atom_types::Vector{Symbol}=[:C, :H, :O, :N]`: elements that may be added
- `lattice_size::Int=1`: sites per axis, so `lattice_size^3` positions per element.
  Defaults to 1 on purpose: `apply_action(::AddAtomAction, …)` records only
  `action.atom_type`, never `action.position`, so two `AddAtomAction`s that differ
  only in position produce *identical* child states. Raising `lattice_size`
  therefore adds parallel edges, not reachable states — it multiplies the per-step
  `is_applicable` sweep by `lattice_size^3` and shifts probability mass from
  terminating towards growing the molecule, without enlarging the state space.
- `max_atoms::Int=10`: hard cap on atoms; also the largest bond index enumerated
- `max_bonds::Int=15`: hard cap on bonds
- `hidden_dim::Int=64`: hidden layer size
- `learning_rate::Float64=0.01`: optimiser learning rate
- `include_flow_estimator::Bool=false`: needed by DIRECT_FLOW / FLOW_MATCHING
- `partition_function_method::PartitionFunctionMethod=SIMPLE_ESTIMATION`
- `rng::AbstractRNG`: random number generator

There is no `include_backward`: a backward policy is only usable together with
`get_previous_states` (core/graphs.jl:216), which recovers parents by brute-force
enumeration of the reachable state space. That is fine for a 5x5 grid and hopeless
here, so DETAILED_BALANCE is not offered for this domain.

# Returns
- `GFlowNetModel`: ready for `train_gflownet` / `sample_trajectory`

# Example
```julia
model = create_molecular_design_model(max_atoms = 6, max_bonds = 6)
history = train_gflownet(model, TrainingConfig(n_iterations = 20, batch_size = 4))
molecules = [sample_trajectory(model).states[end] for _ in 1:20]
```
"""
function create_molecular_design_model(;
    atom_types::Vector{Symbol} = [:C, :H, :O, :N],
    lattice_size::Int = 1,
    max_atoms::Int = 10,
    max_bonds::Int = 15,
    hidden_dim::Int = 64,
    learning_rate::Float64 = 0.01,
    include_flow_estimator::Bool = false,
    partition_function_method::PartitionFunctionMethod = SIMPLE_ESTIMATION,
    rng::AbstractRNG = Random.default_rng()
)
    # Validate inputs
    isempty(atom_types) && throw(ArgumentError("atom_types must not be empty"))
    allunique(atom_types) || throw(ArgumentError("atom_types must not contain duplicates"))
    lattice_size >= 1 || throw(ArgumentError("lattice_size must be at least 1"))
    max_atoms >= 1 || throw(ArgumentError("max_atoms must be at least 1"))
    max_bonds >= 0 || throw(ArgumentError("max_bonds must be non-negative"))
    hidden_dim > 0 || throw(ArgumentError("hidden_dim must be positive"))
    0 < learning_rate < 1 || throw(ArgumentError("learning_rate must be in (0,1)"))

    # Set global configuration for domain functions
    MOLECULE_CONFIG[] = (atom_types = atom_types, max_atoms = max_atoms, max_bonds = max_bonds)

    initial_state = create_initial_molecule_state()

    actions = AbstractAction[]

    for atom_type in atom_types,
        x in 1:lattice_size, y in 1:lattice_size, z in 1:lattice_size

        push!(actions, AddAtomAction(atom_type, (Float64(x), Float64(y), Float64(z))))
    end

    # Bond actions are enumerated for every index pair; `is_applicable` filters out
    # the ones whose endpoints do not exist yet.
    for i in 1:max_atoms, j in (i + 1):max_atoms, bond_type in 1:3
        push!(actions, AddBondAction(i, j, bond_type))
    end

    push!(actions, TerminateMoleculeAction())

    # Create model using on-demand approach
    return GFlowNet.create_gflownet(
        initial_state,
        actions;
        state_dim = length(state_to_features(initial_state)),
        hidden_dim = hidden_dim,
        learning_rate = learning_rate,
        include_flow_estimator = include_flow_estimator,
        partition_function_method = partition_function_method,
        rng = rng
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
