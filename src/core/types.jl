using Graphs
using ComponentArrays

"""
    AbstractState

Abstract type representing a state in a GFlowNet.
Concrete implementations should define specific states using composition
rather than direct inheritance where possible.
"""
abstract type AbstractState end

"""
    AbstractAction 

Abstract type representing an action in a GFlowNet.
Concrete implementations should define specific actions for domain-specific operations.
"""
abstract type AbstractAction end

"""
    Trajectory

Represents a sequence of states through the GFlowNet.
Used for recording paths through the state space during sampling and training.
"""
struct Trajectory
    states::Vector{AbstractState}
end

"""
    DirectedAcyclicGraph{S<:AbstractState, A<:AbstractAction}

Represents the state transition graph for a GFlowNet.
This is a parametric type that specializes on concrete state and action types.

# Fields
- `graph`: The underlying graph structure
- `states`: Vector of all states in the graph
- `actions`: Vector of all possible actions
- `state_to_idx`: Mapping from states to graph indices
- `initial_state`: The starting state for sampling
- `terminal_states`: States that represent complete objects
- `terminal_sink`: Special sink state that connects to all terminal states
"""
struct DirectedAcyclicGraph{S<:AbstractState, A<:AbstractAction}
    graph::SimpleGraph
    states::Vector{S}
    actions::Vector{A}
    state_to_idx::Dict{S, Int}
    initial_state::S
    terminal_states::Vector{S}
    terminal_sink::S
end

# Define a simple state type for testing
"""
    SimpleState

A simple state type for testing and basic usage.
"""
struct SimpleState <: AbstractState
    data::Vector{Int}
end

# Implement equality comparison for SimpleState
Base.:(==)(s1::SimpleState, s2::SimpleState) = s1.data == s2.data
Base.hash(s::SimpleState, h::UInt) = hash(s.data, h)

# Define a simple action type for testing
"""
    SimpleAction

A simple action type for testing and basic usage.
"""
struct SimpleAction <: AbstractAction
    value::Int
end

# Implement equality comparison for SimpleAction
Base.:(==)(a1::SimpleAction, a2::SimpleAction) = a1.value == a2.value
Base.hash(a::SimpleAction, h::UInt) = hash(a.value, h)

# Add default constructor for simple cases
"""
    DirectedAcyclicGraph()

Create an empty DirectedAcyclicGraph with simple state and action types.
This is useful for testing and simple cases where specific types aren't needed.
"""
function DirectedAcyclicGraph()
    # Use simple concrete state and action types
    S = SimpleState
    A = SimpleAction

    # Create empty components
    graph = SimpleGraph(0)
    states = S[]
    actions = A[]
    state_to_idx = Dict{S, Int}()

    # Create placeholder initial and terminal states
    initial_state = SimpleState([0])  # Simple initial state
    terminal_states = S[]
    terminal_sink = SimpleState([-1])  # Special sink state

    return DirectedAcyclicGraph{S, A}(
        graph, states, actions, state_to_idx,
        initial_state, terminal_states, terminal_sink
    )
end

# Define domain-specific data structures for composition
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
    bonds::Vector{Tuple{Int, Int, Int}}  # (atom1, atom2, bond_type)
end

"""
    DAGData

Data structure for causal graph information. Used with composition pattern
to create domain-specific states.

# Fields
- `adjacency_matrix`: Binary adjacency matrix
- `node_names`: Names of nodes/variables
"""
struct DAGData
    adjacency_matrix::Matrix{Int}
    node_names::Vector{String}
end

"""
    ExperimentData

Data structure for experimental design information. Used with composition pattern
to create domain-specific states.

# Fields
- `experiments`: Vector of experiment indices
- `features`: Feature matrix for experiments
"""
struct ExperimentData
    experiments::Vector{Int}
    features::Matrix{Float64}
end

"""
    AbstractGFlowNetObjective

Abstract type for GFlowNet training objectives.
Concrete implementations include FlowMatchingObjective, DetailedBalanceObjective,
and TrajectoryBalanceObjective.
"""
abstract type AbstractGFlowNetObjective end

"""
    FlowMatchingObjective <: AbstractGFlowNetObjective

Flow Matching objective for training GFlowNets.
Enforces flow conservation at each non-terminal state.

# Fields
- `weight`: Weight of this objective in the combined loss
"""
struct FlowMatchingObjective <: AbstractGFlowNetObjective
    weight::Float64
end

"""
    DetailedBalanceObjective <: AbstractGFlowNetObjective

Detailed Balance objective for training GFlowNets.
Enforces consistency between forward and backward transition probabilities.

# Fields
- `weight`: Weight of this objective in the combined loss
"""
struct DetailedBalanceObjective <: AbstractGFlowNetObjective
    weight::Float64
end

"""
    TrajectoryBalanceObjective <: AbstractGFlowNetObjective

Trajectory Balance objective for training GFlowNets.
Enforces consistency across entire trajectories.

# Fields
- `weight`: Weight of this objective in the combined loss
"""
struct TrajectoryBalanceObjective <: AbstractGFlowNetObjective
    weight::Float64
end

"""
    AbstractPolicy

Abstract type for GFlowNet policies.
Concrete implementations are ForwardPolicy and BackwardPolicy.
"""
abstract type AbstractPolicy end

"""
    ForwardPolicy <: AbstractPolicy

Forward policy for GFlowNets, mapping states to distributions over next states.
Can be composed with any model that transforms state features to action probabilities.

# Fields
- `model`: The model used to compute forward transition probabilities
"""
struct ForwardPolicy{M} <: AbstractPolicy
    model::M
end

"""
    BackwardPolicy <: AbstractPolicy

Backward policy for GFlowNets, mapping states to distributions over previous states.
Can be composed with any model that transforms state features to previous state probabilities.

# Fields
- `model`: The model used to compute backward transition probabilities
"""
struct BackwardPolicy{M} <: AbstractPolicy
    model::M
end

"""
    FlowEstimator{M}

Neural network model to directly estimate flow values for states or edges.

# Fields
- `model`: The model used to compute flow estimates
"""
struct FlowEstimator{M}
    model::M
end

"""
    GFlowNetModel

Complete GFlowNet model integrating all components needed for sampling and training.

# Fields
- `dag`: The directed acyclic graph representing the state space
- `forward_policy`: Policy for forward sampling
- `backward_policy`: Optional policy for backward sampling
- `flow_estimator`: Optional direct flow estimator
- `partition_function`: Optional estimate of the partition function
- `objectives`: Training objectives
- `optimizer`: Optimizer for training
- `parameters`: NamedTuple containing model parameters for forward, backward, and flow
- `states`: NamedTuple containing model states for forward, backward, and flow

# Constructor
```julia
GFlowNetModel(;
    dag::DirectedAcyclicGraph,
    forward_policy::ForwardPolicy,
    backward_policy::Union{Nothing, BackwardPolicy} = nothing,
    flow_estimator::Union{Nothing, FlowEstimator} = nothing,
    partition_function::Union{Nothing, Float64} = nothing,
    objectives::Vector{AbstractGFlowNetObjective} = AbstractGFlowNetObjective[],
    optimizer = nothing,
    parameters::NamedTuple,
    states::NamedTuple
)
```
"""
mutable struct GFlowNetModel
    dag::DirectedAcyclicGraph
    forward_policy::ForwardPolicy
    backward_policy::Union{Nothing, BackwardPolicy}
    flow_estimator::Union{Nothing, FlowEstimator}
    partition_function::Union{Nothing, Float64}
    objectives::Vector{AbstractGFlowNetObjective}
    optimizer
    parameters::Union{NamedTuple, ComponentArray}  # Support both NamedTuple and ComponentArray
    states::NamedTuple
    
    # Keyword constructor
    function GFlowNetModel(;
        dag::DirectedAcyclicGraph,
        forward_policy::ForwardPolicy,
        backward_policy::Union{Nothing, BackwardPolicy} = nothing,
        flow_estimator::Union{Nothing, FlowEstimator} = nothing,
        partition_function::Union{Nothing, Float64} = nothing,
        objectives::Vector{<:AbstractGFlowNetObjective} = AbstractGFlowNetObjective[],
        optimizer = nothing,
        parameters::Union{NamedTuple, ComponentArray},
        states::NamedTuple
    )
        new(dag, forward_policy, backward_policy, flow_estimator, 
            partition_function, objectives, optimizer, parameters, states)
    end
    
    # Positional constructor for backward compatibility
    function GFlowNetModel(
        dag::DirectedAcyclicGraph,
        forward_policy::ForwardPolicy,
        backward_policy::Union{Nothing, BackwardPolicy},
        flow_estimator::Union{Nothing, FlowEstimator},
        partition_function::Union{Nothing, Float64},
        objectives::Vector{AbstractGFlowNetObjective},
        optimizer,
        parameters::NamedTuple,
        states::NamedTuple
    )
        new(dag, forward_policy, backward_policy, flow_estimator, 
            partition_function, objectives, optimizer, parameters, states)
    end
end 