using Graphs

"""
    AbstractState

Abstract type representing a state in a GFlowNet.
Concrete implementations should specify the structure of states.
"""
abstract type AbstractState end

"""
    AbstractAction 

Abstract type representing an action in a GFlowNet.
"""
abstract type AbstractAction end

"""
    DirectedAcyclicGraph{S<:AbstractState, A<:AbstractAction}

Type representing a directed acyclic graph with states of type S
and actions of type A.
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

"""
    AbstractGFlowNetObjective

Abstract type for GFlowNet training objectives.
"""
abstract type AbstractGFlowNetObjective end

"""
    FlowMatchingObjective <: AbstractGFlowNetObjective

Flow Matching objective for training GFlowNets.
"""
struct FlowMatchingObjective <: AbstractGFlowNetObjective
    weight::Float64
end

"""
    DetailedBalanceObjective <: AbstractGFlowNetObjective

Detailed Balance objective for training GFlowNets.
"""
struct DetailedBalanceObjective <: AbstractGFlowNetObjective
    weight::Float64
end

"""
    TrajectoryBalanceObjective <: AbstractGFlowNetObjective

Trajectory Balance objective for training GFlowNets.
"""
struct TrajectoryBalanceObjective <: AbstractGFlowNetObjective
    weight::Float64
end

"""
    AbstractPolicy

Abstract type for GFlowNet policies.
"""
abstract type AbstractPolicy end

"""
    ForwardPolicy <: AbstractPolicy

Forward policy for GFlowNets, mapping states to distributions over next states.
"""
struct ForwardPolicy{M} <: AbstractPolicy
    model::M
end

"""
    BackwardPolicy <: AbstractPolicy

Backward policy for GFlowNets, mapping states to distributions over previous states.
"""
struct BackwardPolicy{M} <: AbstractPolicy
    model::M
end

"""
    FlowEstimator

Neural network model to directly estimate flow values for states or edges.
"""
struct FlowEstimator{M}
    model::M
end

"""
    Trajectory{S<:AbstractState}

A trajectory in a GFlowNet, represented as a sequence of states.
"""
struct Trajectory{S<:AbstractState}
    states::Vector{S}
end

"""
    GFlowNetModel{S<:AbstractState, A<:AbstractAction}

Complete GFlowNet model.
"""
mutable struct GFlowNetModel{S<:AbstractState, A<:AbstractAction, 
                            FP<:ForwardPolicy, 
                            BP<:Union{Nothing, BackwardPolicy},
                            FE<:Union{Nothing, FlowEstimator}}
    dag::DirectedAcyclicGraph{S, A}
    forward_policy::FP
    backward_policy::BP
    flow_estimator::FE
    partition_function::Union{Nothing, Float64}
    objectives::Vector{AbstractGFlowNetObjective}
    optimizer
end 