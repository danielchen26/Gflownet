# GFlowNet Developer Guide

This guide explains how to extend the GFlowNet.jl framework with new domains and customizations using the composition-based design pattern.

## Table of Contents

- [Implementing a New Domain](#implementing-a-new-domain)
- [Required Interface](#required-interface)
- [Best Practices](#best-practices)
- [Testing Your Implementation](#testing-your-implementation)
- [Advanced Customizations](#advanced-customizations)

## Implementing a New Domain

To implement a new domain for GFlowNets, follow these steps:

1. **Create domain-specific data structure**:
   Start by defining a data structure that holds the specific information for your domain.

   ```julia
   struct MyDomainData
       # Fields specific to your domain
       field1::Type1
       field2::Type2
       # ...
   end
   ```

2. **Define state and action types**:
   Create concrete implementations of `AbstractState` and `AbstractAction` using composition.

   ```julia
   struct MyDomainState <: AbstractState
       data::MyDomainData
       complete::Bool
       # Other state-specific fields
   end
   
   struct ActionType1 <: AbstractAction
       # Action-specific fields
   end
   
   struct ActionType2 <: AbstractAction
       # Action-specific fields
   end
   ```

3. **Implement the required interface**:
   Implement the necessary methods for your domain (see [Required Interface](#required-interface) below).

4. **Create helper functions**:
   Add utility functions specific to your domain.

   ```julia
   function create_initial_state()
       return MyDomainState(MyDomainData(...), false)
   end
   
   function create_model(params...)
       # Create DAG, policies, etc.
   end
   ```

5. **Add visualization**:
   Implement visualization functions for your domain.

   ```julia
   function visualize(state::MyDomainState)
       # Domain-specific visualization
   end
   ```

## Required Interface

For each new domain, you must implement the following methods:

### Core Methods

1. **`is_applicable`** - Check if an action can be applied to a state:

   ```julia
   function is_applicable(action::ActionType1, state::MyDomainState)
       # Return true if action is applicable to state, false otherwise
   end
   ```

2. **`apply_action`** - Apply an action to a state:

   ```julia
   function apply_action(action::ActionType1, state::MyDomainState)
       # Create and return a new state
   end
   ```

3. **`state_to_features`** - Convert a state to features for neural networks:

   ```julia
   function state_to_features(state::MyDomainState)
       # Return a feature vector
   end
   ```

4. **`reward`** - Calculate reward for a state:

   ```julia
   function reward(state::MyDomainState)
       # Return a non-negative reward value
   end
   ```

### Additional Required Methods

5. **Equality method** - For proper DAG construction:

   ```julia
   function ==(a::MyDomainState, b::MyDomainState)
       # Return true if states are equivalent
   end
   
   function ==(a::ActionType1, b::ActionType1)
       # Return true if actions are equivalent
   end
   ```

6. **Hash method** - For using states in dictionaries:

   ```julia
   function hash(state::MyDomainState, h::UInt)
       # Return a hash value
   end
   
   function hash(action::ActionType1, h::UInt)
       # Return a hash value
   end
   ```

## Best Practices

1. **Immutable States**: Make states immutable to avoid bugs from shared references.

2. **Copy Data**: When applying actions, create new copies of data structures.

3. **Type Stability**: Ensure all methods maintain type stability.

4. **Avoid Cycles**: The GFlowNet state space must be a directed acyclic graph.

5. **Terminal States**: Clearly define the criteria for terminal states.

6. **Meaningful Rewards**: Design reward functions that capture the desired properties.

7. **Efficient Feature Extraction**: Keep `state_to_features` efficient as it will be called frequently.

## Testing Your Implementation

1. **Create a Basic Test**:
   Start with a simple test script that creates states and applies actions.

   ```julia
   # Create an initial state
   initial_state = create_initial_state()
   
   # Create some actions
   actions = [ActionType1(...), ActionType2(...)]
   
   # Test if actions are applicable
   for action in actions
       if is_applicable(action, initial_state)
           new_state = apply_action(action, initial_state)
           println("Applied $(typeof(action))")
           println("New state: $new_state")
       end
   end
   ```

2. **Test DAG Construction**:
   Test that the DAG can be constructed with your states and actions.

   ```julia
   # Create initial and terminal states
   initial_state = create_initial_state()
   terminal_states = [MyDomainState(...)]
   terminal_sink = MyDomainState(...)
   
   # Create actions
   actions = [ActionType1(...), ActionType2(...)]
   
   # Create DAG
   dag = create_dag(initial_state, terminal_states, terminal_sink, actions)
   ```

3. **Test Reward Function**:
   Verify that your reward function behaves as expected.

4. **Test Feature Extraction**:
   Check that the feature extraction produces sensible vectors.

## Advanced Customizations

### Custom Policy Networks

You can implement custom policy networks by:

1. Define a model structure:
   ```julia
   struct MyModel
       # Model parameters
   end
   ```

2. Define forward pass:
   ```julia
   function (model::MyModel)(features)
       # Compute and return policy distribution
   end
   ```

3. Use with ForwardPolicy:
   ```julia
   policy = ForwardPolicy(MyModel(...))
   ```

### Custom Training Objectives

To implement a custom training objective:

1. Define a new objective type:
   ```julia
   struct MyObjective <: AbstractGFlowNetObjective
       weight::Float64
   end
   ```

2. Implement the training logic:
   ```julia
   function compute_loss(obj::MyObjective, model, batch)
       # Compute and return loss
   end
   ```

### Multiple Reward Components

For rewards with multiple components:

```julia
function reward(state::MyDomainState)
    if !state.complete
        return 0.0
    end
    
    # Component 1
    reward1 = calculate_component1(state)
    
    # Component 2
    reward2 = calculate_component2(state)
    
    # Combine rewards
    return reward1 * reward2
end
```

## Additional Resources

- See the `examples/` directory for complete examples
- Refer to the source code for the existing domains:
  - `src/applications/molecular_design.jl`
  - `src/applications/causal_discovery.jl` 
  - `src/applications/active_learning.jl` 