# Core Concepts

This page introduces the core concepts behind GFlowNets.

## Directed Acyclic Graphs (DAGs)

At the heart of GFlowNets is a directed acyclic graph (DAG) that represents the construction process:

- **States**: Nodes in the graph, representing partial or complete objects
- **Actions**: Edges in the graph, representing transitions between states
- **Terminal states**: Special states that represent fully constructed objects

## Flow Networks

GFlowNets are based on the concept of flow networks:

- **Flow**: A measure assigned to each edge in the graph
- **Flow conservation**: At each non-terminal state, the incoming flow equals the outgoing flow
- **Source flow**: The total flow entering the initial state
- **Terminal flow**: The flow at each terminal state, proportional to the reward

## Policies

GFlowNets use two types of policies:

- **Forward policy**: Defines the probability of taking an action from a state during sampling
- **Backward policy**: Defines the probability of taking a reverse action from a state during training

## Flow Estimators

Flow estimators are neural networks that learn to predict:

- **State flows**: The total flow through a state
- **Edge flows**: The flow along specific edges

## Environment

The environment defines:

- **State space**: The set of possible states
- **Action space**: The set of possible actions from each state
- **Transition function**: Mapping from state-action pairs to new states
- **Reward function**: Function that assigns rewards to terminal states

## Training Objectives

GFlowNets can be trained using different objectives:

- **Flow matching**: Ensures that the parameterized flows satisfy flow conservation
- **Detailed balance**: Ensures that the forward and backward flows between states are consistent
- **Trajectory balance**: Ensures that the product of forward probabilities along a trajectory is proportional to the reward

## Sampling

After training, GFlowNets can generate samples by:

1. Starting from the initial state
2. Repeatedly taking actions according to the forward policy
3. Stopping when a terminal state is reached
