# Policies

This page documents the policy components in GFlowNet.jl.

## Policy Types

GFlowNet.jl defines several policy types:

* `AbstractPolicy`: Base type for all policies
* `ForwardPolicy`: Policy for forward sampling and traversing the state space
* `BackwardPolicy`: Policy for backward sampling, used in training

## Creating Policies

* `create_forward_policy(input_dim, hidden_dim, output_dim, [activation])`: Creates a forward policy
* `create_backward_policy(input_dim, hidden_dim, output_dim, [activation])`: Creates a backward policy

## Forward Policy

The forward policy determines the probability distribution over actions at each state. It is typically implemented as a neural network that maps state features to action logits.

Key functions:
* `forward_transition_logits(policy, state, ...)`: Computes unnormalized log probabilities
* `forward_transition_prob(policy, state, action, ...)`: Computes normalized probabilities
* `sample_action(policy, state, ...)`: Samples an action from the policy

## Backward Policy

The backward policy defines the probability of transitioning backward from a state to one of its parents. It's essential for some training objectives like detailed balance.

Key functions:
* `backward_transition_logits(policy, state, ...)`: Computes unnormalized log probabilities
* `backward_transition_prob(policy, state, parent, ...)`: Computes normalized probabilities
* `sample_prev_state(policy, state, ...)`: Samples a parent state

## Policy Networks

GFlowNet.jl uses the Lux.jl library for implementing policy networks:

```julia
function create_policy_network(input_dim, hidden_dim, output_dim, activation=relu)
    return Lux.Chain(
        Lux.Dense(input_dim => hidden_dim, activation),
        Lux.Dense(hidden_dim => hidden_dim, activation),
        Lux.Dense(hidden_dim => output_dim)
    )
end
```

## Exploration Strategies

GFlowNet.jl supports various exploration strategies:

* Temperature scaling: Controls the exploration-exploitation trade-off
* ε-greedy exploration: Takes random actions with probability ε
* Boltzmann exploration: Samples actions based on softmax of logits

## Code Example

```julia
function create_and_use_policy()
    # Create a forward policy
    input_dim = 10  # State feature dimension
    hidden_dim = 64
    output_dim = 5  # Number of possible actions
    
    forward_policy = create_forward_policy(input_dim, hidden_dim, output_dim)
    
    # Use the policy to sample actions
    state = create_initial_state()
    action = sample_action(forward_policy, state)
    
    # Compute transition probabilities
    probs = forward_transition_prob(forward_policy, state, actions)
    
    return action, probs
end
```
