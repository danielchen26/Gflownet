# Flow Networks

This page documents the core functions and concepts related to flow networks in GFlowNet.jl.

## Core Flow Functions

The following functions form the foundation of GFlowNet operations:

* `flow(model, state)`: Computes the flow (unnormalized probability) through a state
* `edge_flow(model, parent, child)`: Computes the flow along an edge between states
* `state_to_features(state)`: Converts a state to a feature representation for neural networks
* `reward(state)`: Calculates the reward associated with a terminal state

## Transition Probabilities

GFlowNets define probability distributions over transitions:

* `forward_transition_prob(policy, state, action)`: Probability of taking an action from a state
* `backward_transition_prob(policy, state, parent_state)`: Probability of transitioning to a parent state

## Exploration and Sampling

* `sample_trajectory(model, [params])`: Samples a complete trajectory through the state space
* `sample_action(policy, state, [params])`: Samples the next action to take from a state

## Flow Estimation

* `estimate_flow(flow_estimator, state, ...)`: Estimates the flow through a state
* `estimate_edge_flow(flow_estimator, parent, child, ...)`: Estimates the flow along an edge

## State and Edge Operations

Functions for managing states and edges in the flow network:

* `get_next_states(state, actions)`: Gets possible next states from the current state
* `get_previous_states(state)`: Gets possible parent states of the current state
* `get_incoming_edges(state)`: Gets edges leading to the current state
* `get_outgoing_edges(state)`: Gets edges leading from the current state
* `is_applicable(action, state)`: Checks if an action can be applied to a state
* `apply_action(action, state)`: Applies an action to a state, returning a new state

## Flow Network Concepts

The GFlowNet framework is based on several key concepts:

1. **Flow Conservation**: For non-terminal states, the sum of incoming flows equals the sum of outgoing flows
2. **Edge Flow**: The flow along an edge from state s to s' is determined by the product of the flow through s and the probability of the transition from s to s'
3. **Transition Probabilities**: The forward policy defines probabilities of transitioning from a state to its children
4. **Reward Function**: The reward function defines the target distribution that the GFlowNet will learn to sample from

## Code Example

```julia
# Computing flows and sampling trajectories
function example_flow_network()
    # Create a GFlowNet model
    model = create_model(...)
    
    # Compute flow through a state
    state = create_initial_state()
    state_flow = flow(model, state)
    
    # Sample a trajectory
    trajectory = sample_trajectory(model)
    
    # Verify flow conservation (for non-terminal states)
    for state in trajectory.states[1:end-1]
        incoming_flow = sum(edge_flow(model, p, state) for p in get_previous_states(state))
        outgoing_flow = sum(edge_flow(model, state, n) for n in get_next_states(state))
        @assert abs(incoming_flow - outgoing_flow) < 1e-5
    end
    
    return trajectory
end
```
