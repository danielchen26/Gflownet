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
- `graph`: The underlying directed graph structure (CORRECTED: Now uses SimpleDiGraph)
- `states`: Vector of all states in the graph
- `actions`: Vector of all possible actions
- `state_to_idx`: Mapping from states to graph indices
- `initial_state`: The starting state for sampling
- `terminal_states`: States that represent complete objects
- `terminal_sink`: Special sink state that connects to all terminal states
- `action_cache`: OPTIMIZED: Cache of applicable actions per state for performance
"""
struct DirectedAcyclicGraph{S<:AbstractState,A<:AbstractAction}
    graph::SimpleDiGraph
    states::Vector{S}
    actions::Vector{A}
    state_to_idx::Dict{S,Int}
    initial_state::S
    terminal_states::Vector{S}
    terminal_sink::S
    action_cache::Dict{S,Vector{A}}  # Cache applicable actions per state
end

# Simple test types moved to test/test_utilities.jl (for testing only)
# For production implementations, see:
# - src/applications/molecular_design.jl (MoleculeState, MoleculeData)
# - src/applications/causal_discovery.jl (DAGState, DAGData)
# - src/applications/active_learning.jl (ExperimentState, ExperimentData)

# Default constructor removed - use domain-specific constructors instead
# For DAG construction examples, see the applications/ directory

# Domain-specific data structures moved to src/applications/
# - MoleculeData → src/applications/molecular_design.jl
# - DAGData → src/applications/causal_discovery.jl
# - ExperimentData → src/applications/active_learning.jl

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
mutable struct GFlowNetModel{P<:ComponentArray}
    dag::DirectedAcyclicGraph
    forward_policy::ForwardPolicy
    backward_policy::Union{Nothing,BackwardPolicy}
    flow_estimator::Union{Nothing,FlowEstimator}
    partition_function::Union{Nothing,Float64}
    objectives::Vector{AbstractGFlowNetObjective}
    optimizer
    parameters::P  # Strictly typed as ComponentArray
    states::NamedTuple

    # Keyword constructor
    function GFlowNetModel(;
        dag::DirectedAcyclicGraph,
        forward_policy::ForwardPolicy,
        backward_policy::Union{Nothing,BackwardPolicy}=nothing,
        flow_estimator::Union{Nothing,FlowEstimator}=nothing,
        partition_function::Union{Nothing,Float64}=nothing,
        objectives::Vector{<:AbstractGFlowNetObjective}=AbstractGFlowNetObjective[],
        optimizer=nothing,
        parameters::ComponentArray,
        states::NamedTuple
    )
        # Validate GFlowNetModel construction
        validate_gflownet_model_construction(dag, forward_policy, backward_policy,
            flow_estimator, partition_function,
            parameters, states)

        new{typeof(parameters)}(dag, forward_policy, backward_policy, flow_estimator,
            partition_function, objectives, optimizer, parameters, states)
    end

    # Positional constructor for backward compatibility
    function GFlowNetModel(
        dag::DirectedAcyclicGraph,
        forward_policy::ForwardPolicy,
        backward_policy::Union{Nothing,BackwardPolicy},
        flow_estimator::Union{Nothing,FlowEstimator},
        partition_function::Union{Nothing,Float64},
        objectives::Vector{AbstractGFlowNetObjective},
        optimizer,
        parameters::ComponentArray,
        states::NamedTuple
    )
        new{typeof(parameters)}(dag, forward_policy, backward_policy, flow_estimator,
            partition_function, objectives, optimizer, parameters, states)
    end
end

# =============================================================================
# Utility Functions for Parameter Type Conversion
# =============================================================================

"""
    to_component_array(params::NamedTuple)

Convert NamedTuple parameters to ComponentArray for gradient compatibility.
This utility function provides backward compatibility for code using NamedTuple parameters.

# Arguments
- `params::NamedTuple`: Parameters as NamedTuple

# Returns
- `ComponentArray`: Parameters converted to ComponentArray format

# Example
```julia
# Convert existing NamedTuple parameters
old_params = (forward = θ_f, backward = θ_b, flow = θ_flow)
new_params = to_component_array(old_params)
```
"""
function to_component_array(params::NamedTuple)
    @warn "Converting NamedTuple to ComponentArray. Consider using ComponentArray directly for better performance."
    return ComponentArray(params)
end

"""
    to_component_array(params::ComponentArray)

Identity function for ComponentArray parameters (no conversion needed).
"""
function to_component_array(params::ComponentArray)
    return params
end

"""
    to_component_array(params::AbstractArray)

Convert any array-like parameter structure to ComponentArray.
"""
function to_component_array(params::AbstractArray)
    return ComponentArray(params)
end

"""
    create_gflownet_model_safe(; parameters, kwargs...)

Safe constructor for GFlowNetModel that automatically converts parameters to ComponentArray.
This provides backward compatibility while enforcing the standardized parameter type.

# Arguments
- `parameters`: Model parameters (NamedTuple or ComponentArray)
- `kwargs...`: Other arguments for GFlowNetModel constructor

# Returns
- `GFlowNetModel` with ComponentArray parameters
"""
function create_gflownet_model_safe(; parameters, kwargs...)
    # Ensure parameters are ComponentArray
    if !isa(parameters, ComponentArray)
        @warn "Parameters should be ComponentArray. Converting automatically."
        standardized_params = to_component_array(parameters)
    else
        standardized_params = parameters
    end

    return GFlowNetModel(; parameters=standardized_params, kwargs...)
end

# =============================================================================
# GFlowNet Model Validation
# =============================================================================

"""
    validate_gflownet_model_construction(dag, forward_policy, backward_policy,
                                        flow_estimator, partition_function,
                                        parameters, states)

Comprehensive validation for GFlowNetModel construction.
Ensures all components are compatible and properly configured.

# Arguments
- `dag`: DirectedAcyclicGraph
- `forward_policy`: ForwardPolicy
- `backward_policy`: Optional BackwardPolicy
- `flow_estimator`: Optional FlowEstimator
- `partition_function`: Optional partition function value
- `parameters`: Model parameters (ComponentArray)
- `states`: Model states (NamedTuple)

# Throws
- `ArgumentError` if model configuration is invalid
"""
function validate_gflownet_model_construction(dag, forward_policy, backward_policy,
    flow_estimator, partition_function,
    parameters, states)
    # Validate DAG
    if isempty(dag.states)
        throw(ArgumentError("GFlowNetModel requires a DAG with states"))
    end

    # Validate forward policy is required
    if isnothing(forward_policy)
        throw(ArgumentError("GFlowNetModel requires a forward policy"))
    end

    # Validate parameters structure
    if !haskey(parameters, :forward)
        throw(ArgumentError("Model parameters must include :forward policy parameters"))
    end

    if !isnothing(backward_policy) && !haskey(parameters, :backward)
        throw(ArgumentError("Backward policy provided but :backward parameters missing"))
    end

    if !isnothing(flow_estimator) && !haskey(parameters, :flow)
        throw(ArgumentError("Flow estimator provided but :flow parameters missing"))
    end

    # Validate states structure
    if !haskey(states, :forward)
        throw(ArgumentError("Model states must include :forward policy states"))
    end

    if !isnothing(backward_policy) && !haskey(states, :backward)
        throw(ArgumentError("Backward policy provided but :backward states missing"))
    end

    if !isnothing(flow_estimator) && !haskey(states, :flow)
        throw(ArgumentError("Flow estimator provided but :flow states missing"))
    end

    # Validate partition function
    if !isnothing(partition_function)
        if isnan(partition_function) || isinf(partition_function)
            throw(ArgumentError("Partition function must be finite"))
        end

        if partition_function <= 0.0
            throw(ArgumentError("Partition function must be positive"))
        end
    end

    # Validate parameter consistency
    validate_model_parameters(parameters, "GFlowNetModel parameters")

    @debug "GFlowNetModel validation passed"
end

# =============================================================================
# Partition Function Estimator Types (moved from partition.jl to avoid redefinition)
# =============================================================================

"""
    AbstractPartitionFunctionEstimator

Abstract type for different partition function estimation strategies.
"""
abstract type AbstractPartitionFunctionEstimator end

"""
    SimplePartitionFunctionEstimator <: AbstractPartitionFunctionEstimator

Simple estimator that sums all terminal state rewards.
This is the current default implementation.
"""
struct SimplePartitionFunctionEstimator <: AbstractPartitionFunctionEstimator end

"""
    LearnablePartitionFunctionEstimator <: AbstractPartitionFunctionEstimator

Learnable partition function as a parameter that's updated via gradient descent.

# Fields
- `log_Z`: Learnable log partition function parameter
- `optimizer`: Optimizer for the log_Z parameter
"""
mutable struct LearnablePartitionFunctionEstimator <: AbstractPartitionFunctionEstimator
    log_Z::Float64
    optimizer::Any
end

"""
    SamplingPartitionFunctionEstimator <: AbstractPartitionFunctionEstimator

Sampling-based partition function estimator with exponential smoothing.

# Fields
- `n_samples`: Number of samples to use for estimation
- `history_length`: Number of past estimates to keep for smoothing
- `smoothing_factor`: Exponential smoothing factor
"""
mutable struct SamplingPartitionFunctionEstimator <: AbstractPartitionFunctionEstimator
    n_samples::Int
    history_length::Int
    smoothing_factor::Float64
    estimate_history::Vector{Float64}
end

"""
    AdaptivePartitionFunctionEstimator <: AbstractPartitionFunctionEstimator

Adaptive estimator that switches between different methods based on training progress.

# Fields
- `simple_estimator`: Simple estimation method
- `sampling_estimator`: Sampling-based estimation method
- `learnable_estimator`: Learnable parameter method
- `method`: Current estimation method (:simple, :sampling, :learnable)
- `switch_thresholds`: Thresholds for switching methods
"""
mutable struct AdaptivePartitionFunctionEstimator <: AbstractPartitionFunctionEstimator
    simple_estimator::SimplePartitionFunctionEstimator
    sampling_estimator::SamplingPartitionFunctionEstimator
    learnable_estimator::Union{Nothing,LearnablePartitionFunctionEstimator}
    method::Symbol
    switch_thresholds::Dict{Symbol,Float64}
    training_iteration::Int
end

# =============================================================================
# Training Configuration Types (moved from config.jl to avoid redefinition)
# =============================================================================

"""
    TrainingObjective

Enumeration of available training objectives for GFlowNet.
"""
@enum TrainingObjective begin
    TRAJECTORY_BALANCE
    DETAILED_BALANCE
    SUB_TRAJECTORY_BALANCE
    HIERARCHICAL_SUB_TB
    ADAPTIVE_SUB_TB
    FLOW_CONSISTENCY
end

"""
    PartitionFunctionMethod

Enumeration of partition function estimation methods.
"""
@enum PartitionFunctionMethod begin
    SIMPLE_ESTIMATION
    SAMPLING_ESTIMATION
    LEARNABLE_ESTIMATION
    ADAPTIVE_ESTIMATION
end

"""
    TrainingConfig

Configuration for GFlowNet training.

# Fields
- `objective`: Training objective to use
- `partition_function_method`: Method for estimating partition function
- `batch_size`: Number of trajectories per training batch
- `learning_rate`: Learning rate for optimization
- `n_iterations`: Total number of training iterations
- `partition_update_frequency`: How often to update partition function estimate
- `validation_frequency`: How often to run validation
- `early_stopping_patience`: Number of iterations without improvement before stopping
- `sub_trajectory_config`: Configuration for sub-trajectory methods
"""
struct TrainingConfig
    objective::TrainingObjective
    partition_function_method::PartitionFunctionMethod
    batch_size::Int
    learning_rate::Float64
    n_iterations::Int
    partition_update_frequency::Int
    validation_frequency::Int
    early_stopping_patience::Int
    sub_trajectory_config::Dict{Symbol,Any}

    # Keyword constructor with defaults
    function TrainingConfig(; objective=TRAJECTORY_BALANCE,
        partition_function_method=SIMPLE_ESTIMATION,
        batch_size=32, learning_rate=0.001, n_iterations=1000,
        partition_update_frequency=10, validation_frequency=50,
        early_stopping_patience=100, sub_trajectory_config=Dict())

        # Default sub-trajectory configuration
        default_sub_config = Dict(
            :min_length => 2,
            :max_length => nothing,
            :n_subtrajectories => 5,
            :scales => [2, 4, 8],
            :difficulty_threshold => 0.1,
            :flow_consistency_mode => :STATE_LEVEL,  # Default flow consistency mode
            :max_grad_norm => 1.0  # Default gradient clipping norm
        )

        merged_sub_config = merge(default_sub_config, sub_trajectory_config)

        return new(objective, partition_function_method, batch_size, learning_rate,
            n_iterations, partition_update_frequency, validation_frequency,
            early_stopping_patience, merged_sub_config)
    end
end
