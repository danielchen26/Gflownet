# Flow Consistency and Conservation

## Mathematical Foundation

Flow consistency is a fundamental property of GFlowNets that ensures the learned distribution matches the target distribution proportional to rewards.

## Flow Conservation Equation

The core flow conservation principle states:

```
F(s) = Σ_{s': s→s'} P_F(s'|s) · F(s')
```

Where:
- `F(s)` is the flow through state s
- `P_F(s'|s)` is the forward transition probability
- The sum is over all states s' reachable from s

## Boundary Conditions

### Terminal States
For terminal states, flow equals reward:
```
F(s) = R(s) for s ∈ S_T
```

### Initial State
The flow through the initial state is the partition function:
```
F(s₀) = Z = Σ_{τ: s₀→s_T} P_F(τ) · R(s_T)
```

## Balance Conditions

### Detailed Balance
At the edge level, detailed balance requires:
```
F(s) · P_F(s'|s) = F(s') · P_B(s|s')
```

This ensures flow is conserved along each edge.

### Flow Matching
At the state level, flow matching requires:
```
F(s) = Σ_{s': s'→s} F(s') · P_F(s|s')
```

This ensures incoming flow equals outgoing flow.

### Trajectory Balance
At the trajectory level:
```
Z · P_F(τ) = R(s_T) · P_B(τ)
```

This is what we actually optimize in practice.

## Relationship Between Balance Conditions

### Theoretical Equivalence
Under certain conditions, all three balance conditions are equivalent:
1. **Detailed Balance ⟹ Flow Matching**: If detailed balance holds for all edges, flow matching holds for all states
2. **Flow Matching ⟹ Trajectory Balance**: If flow matching holds everywhere, trajectory balance holds for all trajectories
3. **Trajectory Balance ⟹ Flow Conservation**: If trajectory balance holds for sufficient trajectories, flow conservation emerges

### Practical Differences
- **Trajectory Balance**: Global constraint, easier to optimize
- **Detailed Balance**: Local constraint, requires backward policy
- **Flow Matching**: State-level constraint, requires flow network

## Flow Consistency Objective

The flow consistency objective unifies different balance conditions:

```julia
L_FC = α₁ · L_edge + α₂ · L_state + α₃ · L_trajectory
```

Where:
- `L_edge`: Edge-level detailed balance loss
- `L_state`: State-level flow matching loss
- `L_trajectory`: Trajectory-level balance loss

## Mathematical Properties

### Conservation Laws
1. **Flow Conservation**: Total flow is conserved through the network
2. **Probability Conservation**: Transition probabilities sum to 1
3. **Reward Conservation**: Terminal flows equal rewards

### Convergence Properties
- **Fixed Point**: Optimal policies satisfy all balance conditions
- **Unique Solution**: Given rewards, there's a unique flow satisfying conservation
- **Stability**: Small perturbations in rewards lead to small changes in flows

## Implementation Challenges

### Without Explicit DAG
- Cannot enumerate all states for flow matching
- Cannot compute exact partition function
- Must rely on trajectory-based approximations

### Current Approach
GFlowNet.jl uses trajectory balance with Z=1 assumption:
- Valid for fixed initial state
- Avoids need for flow computation
- Sufficient for most applications

### Future Extensions
To implement full flow consistency:
1. **Flow Network**: Learn F(s) for all states
2. **Backward Policy**: Enable detailed balance
3. **State Enumeration**: For exact flow matching

## Theoretical Insights

### Why Trajectory Balance Works
- Samples from P_F provide unbiased gradient estimates
- Convergence guaranteed under mild conditions
- No need for explicit flow computation

### When Flow Consistency Matters
- Multiple initial states
- Need for exact probability estimates
- Theoretical analysis of convergence

### Approximations and Their Impact
- Z=1 assumption: Valid for conditional distributions
- Missing backward policy: Limits to deterministic environments
- No flow network: Cannot verify conservation directly

## Practical Implications

### For Simple Domains
- Trajectory balance is sufficient
- No need for flow networks
- Fast and stable training

### For Complex Domains
- May benefit from flow consistency
- Better credit assignment with detailed balance
- More accurate probability estimates

### Computational Trade-offs
- Flow networks add memory overhead
- Detailed balance requires backward policy
- Full consistency is computationally expensive

## Future Research Directions

### Implicit Flow Networks
- Learn flows without explicit representation
- Use function approximation
- Maintain conservation approximately

### Hybrid Objectives
- Combine trajectory and detailed balance
- Adaptive weighting during training
- Best of both approaches

### Theoretical Analysis
- Convergence rates for different objectives
- Sample complexity bounds
- Approximation quality guarantees

## See Also
- [Mathematical Background](mathematical_background.md) - Core GFlowNet theory
- [Training Objectives](../manual/objectives.md) - Practical implementation
- [Partition Function](partition_function.md) - Related concepts