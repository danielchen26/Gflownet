# Flow Networks

This page documents the current flow-related concepts in GFlowNet.jl.

## Current Implementation

GFlowNet.jl uses an implicit flow approach through the Trajectory Balance objective:

* `state_to_features(state)`: Converts a state to a feature representation for neural networks
* `reward(state)`: Calculates the reward associated with a terminal state
* Flow computations are handled implicitly during training

**Note**: Explicit flow functions like `flow(model, state)` and `edge_flow(model, parent, child)` are not currently implemented.

## Transition Probabilities

GFlowNets define probability distributions over transitions:

* `forward_transition_prob(policy, state, action)`: Probability of taking an action from a state
* `backward_transition_prob(policy, state, parent_state)`: Probability of transitioning to a parent state

## Exploration and Sampling

* `sample_trajectory(model, [params])`: Samples a complete trajectory through the state space
* `sample_action(policy, state, [params])`: Samples the next action to take from a state

## Flow Estimation

**Note**: Explicit flow estimation functions are not currently implemented. Flow values are computed implicitly through the Trajectory Balance objective during training.

## State and Action Operations

Functions for managing states and actions in the current implementation:

* `get_applicable_actions(state, all_actions)`: Gets actions that can be applied to the current state
* `is_applicable(action, state)`: Checks if an action can be applied to a state
* `apply_action(action, state)`: Applies an action to a state, returning a new state

**Note**: Functions like `get_next_states()` and `get_previous_states()` are not implemented. Use the action-based approach instead.

## Flow Network Concepts

The GFlowNet framework is based on several key concepts:

1. **Flow Conservation**: For non-terminal states, the sum of incoming flows equals the sum of outgoing flows
2. **Edge Flow**: The flow along an edge from state s to s' is determined by the product of the flow through s and the probability of the transition from s to s'
3. **Transition Probabilities**: The forward policy defines probabilities of transitioning from a state to its children
4. **Reward Function**: The reward function defines the target distribution that the GFlowNet will learn to sample from

## Code Example

```julia
# Sampling trajectories with current implementation
function example_sampling()
    # Create a GFlowNet model using high-level interface
    model = create_grid_world_gflownet(grid_size=5)
    
    # Sample a trajectory
    trajectory = sample_trajectory(model)
    
    # Get the terminal state and reward
    terminal_state = trajectory.states[end]
    final_reward = reward(terminal_state)
    
    # Example of action-based state exploration
    current_state = trajectory.states[1]  # Initial state
    applicable_actions = get_applicable_actions(current_state, model.all_actions)
    
    # Apply an action
    if !isempty(applicable_actions)
        next_state = apply_action(applicable_actions[1], current_state)
    end
    
    return trajectory
end
```
