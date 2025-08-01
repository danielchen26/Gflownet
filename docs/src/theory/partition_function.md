# Understanding the Partition Function Z in GFlowNets

## Table of Contents
1. [Mathematical Foundation](#mathematical-foundation)
2. [When Z = 1 is Valid](#when-z--1-is-valid)
3. [When Z Must Be Learned](#when-z-must-be-learned)
4. [Why Flow Functions Are Needed](#why-flow-functions-are-needed)
5. [Implementation Strategies](#implementation-strategies)
6. [Current State in GFlowNet.jl](#current-state-in-gflownetjl)

## Mathematical Foundation

### What is the Partition Function Z?

In GFlowNets, the partition function $Z$ is the total flow through the initial state $s_0$:

$$Z = F(s_0) = \sum_{\tau: s_0 \to s_T} P_F(\tau) R(s_T)$$

Where:
- $\tau$ represents all possible trajectories from $s_0$ to any terminal state $s_T$
- $P_F(\tau)$ is the forward probability of trajectory $\tau$
- $R(s_T)$ is the reward at terminal state $s_T$

### Trajectory Balance Condition

The fundamental trajectory balance equation is:

$$Z \cdot P_F(\tau) = R(s_T) \cdot P_B(\tau)$$

Taking logarithms:
$$\log Z + \sum \log P_F(s_{i+1}|s_i) = \log R(s_T) + \sum \log P_B(s_i|s_{i+1})$$

## When Z = 1 is Valid

### Fixed Initial State Scenario

$Z = 1$ is mathematically valid when:

1. **The initial state $s_0$ is fixed and known**
2. **We define our probability distribution starting from $s_0$**

In this case, we're learning $P(\tau|s_0)$ - the distribution over trajectories *given* we start at $s_0$.

### Examples Where Z = 1 Works

1. **Grid World**: Always start at position (1,1)
   ```julia
   initial_state = GridState(1, 1, false)  # Fixed starting position
   ```

2. **Molecule Generation from Scratch**: Always start with empty molecule
   ```julia
   initial_state = MoleculeState(empty_graph, false)  # Fixed empty start
   ```

3. **Supply Chain**: Start with empty network
   ```julia
   initial_state = SupplyChainState(empty_network, 0, false)  # Fixed empty start
   ```

### Mathematical Justification

When $s_0$ is fixed, we can define:
- $P(s_T|s_0) \propto R(s_T)$ (distribution over terminal states given $s_0$)
- The normalization constant for this conditional distribution can be absorbed into the learning process
- Setting $Z = 1$ is equivalent to learning unnormalized probabilities

## When Z Must Be Learned

### Multiple Initial States Scenario

$Z \neq 1$ and must be learned when:

1. **Multiple possible starting states**
2. **The distribution over initial states matters**
3. **We want $P(\tau)$ not $P(\tau|s_0)$**

### Examples Where Z Must Be Learned

1. **Molecule Editing**: Start from different existing molecules
   ```julia
   # Different initial molecules require different Z values
   initial_states = [
       MoleculeState(benzene, false),    # Z₁
       MoleculeState(methane, false),    # Z₂
       MoleculeState(ethanol, false)     # Z₃
   ]
   ```

2. **Causal Discovery with Prior Knowledge**: Start from partially known graphs
   ```julia
   # Different prior structures need different normalizations
   initial_states = [
       CausalState(partial_dag_1, false),  # Z₁
       CausalState(partial_dag_2, false)   # Z₂
   ]
   ```

3. **Supply Chain Expansion**: Improve existing networks
   ```julia
   # Existing networks have different improvement potentials
   initial_states = [
       SupplyChainState(existing_network_1, month, false),  # Z₁
       SupplyChainState(existing_network_2, month, false)   # Z₂
   ]
   ```

### Why Different Z Values?

Each initial state has different:
- Number of reachable terminal states
- Path lengths to high-reward states  
- Total reward mass in its reachable set

Therefore, each needs its own normalization constant $Z(s_0)$.

## Why Flow Functions Are Needed

### The Recursive Nature of Z

The partition function is defined recursively:

$$Z = F(s_0) = \sum_{s': s_0 \to s'} P_F(s'|s_0) \cdot F(s')$$

Where $F(s)$ is the flow through state $s$:
$$F(s) = \begin{cases}
R(s) & \text{if } s \text{ is terminal} \\
\sum_{s': s \to s'} P_F(s'|s) \cdot F(s') & \text{otherwise}
\end{cases}$$

### Why We Can't Learn Z Directly

1. **$Z$ depends on the policy $P_F$**: When $P_F$ changes during training, $Z$ changes
2. **$Z$ is a sum over exponentially many paths**: Direct computation is intractable
3. **$Z$ requires knowing $F(s)$ for all reachable states**: This is the flow function

### Connection to Trajectory Balance

Without flow functions, we can only use trajectory balance with:
- Fixed $Z = 1$ (assumes fixed $s_0$)
- Or learnable scalar $Z$ (assumes single initial state)

With flow functions, we can:
- Handle multiple initial states with $Z(s_0) = F(s_0)$
- Use detailed balance: $P_F(s'|s)F(s) = P_B(s|s')F(s')$
- Implement flow matching objectives

## Implementation Strategies

### Strategy 1: Fixed Initial State (Current Approach)
```julia
# Simple and effective for single s₀
log_Z = 0.0  # log(1) = 0
loss = (log_Z + log_P_F - log_R - log_P_B)²
```

**Pros**: Simple, stable, sufficient for many applications
**Cons**: Limited to single initial state

### Strategy 2: Learnable Scalar Z
```julia
# Learn a single Z value
log_Z = model.parameters.log_partition_function
loss = (log_Z + log_P_F - log_R - log_P_B)²
```

**Pros**: Can adapt to data, still simple
**Cons**: Still assumes single initial state

### Strategy 3: Flow Network (Full Solution)
```julia
# Learn F(s) for all states
log_flow_s0 = log(flow_network(s₀))
loss = (log_flow_s0 + log_P_F - log_R - log_P_B)²
```

**Pros**: Handles multiple initial states, enables all objectives
**Cons**: Complex, requires more computation

### Strategy 4: Conditional GFlowNet
```julia
# Condition on initial state explicitly
features = [state_features; initial_state_features]
# Use Z = 1 for conditional distribution P(τ|s₀)
```

**Pros**: Handles multiple $s_0$ while keeping $Z = 1$
**Cons**: Larger networks, need $s_0$ at inference time

## Current State in GFlowNet.jl

### What We Have
```julia
# In balance.jl
log_initial_flow = 0.0  # Assumes Z = 1

# In flows.jl  
function partition_function(model::GFlowNetModel)::Float64
    return 1.0  # Assumes Z = 1
end
```

### Why This Works
- All examples use fixed initial states
- The math is valid for $P(\tau|s_0)$
- Training successfully learns good policies

### What's Missing
1. **Flow network implementation** for $F(s)$ computation
2. **Multi-initial-state support** with different $Z$ values
3. **Detailed balance** and **flow matching** objectives

### When to Implement Full Z

You need proper $Z$ computation when:
1. **Multiple initial states**: Different starting configurations
2. **Transfer learning**: Adapt trained model to new initial states
3. **Theoretical guarantees**: Need exact flow conservation
4. **Advanced objectives**: Want detailed balance or flow matching

## Recommendations

### For Most Applications
- **Keep $Z = 1$**: It's simple and works well
- **Use fixed initial state**: Design your problem with single $s_0$
- **Focus on trajectory balance**: It's sufficient for good results

### When to Extend
Implement proper $Z$ computation only when:
- Your application has multiple natural starting points
- You need to compare probabilities across different $s_0$
- You want to use detailed balance or flow matching
- You're doing research on GFlowNet theory

### Implementation Priority
1. **Now**: Document clearly that $Z = 1$ assumes fixed $s_0$
2. **Later**: Add learnable scalar $Z$ option
3. **Future**: Implement full flow networks when needed
4. **Research**: Explore conditional GFlowNets as alternative

## Conclusion

The current $Z = 1$ assumption in GFlowNet.jl is:
- **Mathematically valid** for fixed initial states
- **Practically sufficient** for most applications  
- **Theoretically limited** but not problematic

The partition function parameter in TrainingConfig remains unused because:
- Implementing proper $Z$ requires flow functions
- Flow functions require explicit state enumeration or function approximation
- This complexity isn't needed for current applications

Future work could add $Z$ learning, but it's not a priority for the working examples.