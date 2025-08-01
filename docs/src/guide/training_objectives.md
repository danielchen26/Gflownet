# Training Objectives

This page provides a comprehensive overview of the different training objectives used in GFlowNets, their mathematical derivations, practical considerations, and relative strengths.

## Introduction to Training Objectives

GFlowNets can be trained using several different objectives, each with unique properties and trade-offs. The choice of training objective can significantly impact the convergence rate, sample quality, and overall performance of the model.

At a high level, all GFlowNet training objectives aim to ensure that the network satisfies the flow consistency conditions:

1. **Flow Conservation**: For all non-terminal states, the incoming flow equals the outgoing flow
2. **Terminal State Flow**: The flow at terminal states is proportional to their rewards
3. **Total Flow**: The total flow through the network equals the sum of all terminal rewards

The training objectives differ in how they enforce these constraints and in their computational and statistical properties.

## Current Implementation Overview

**This implementation uses the Trajectory Balance objective with periodic Z estimation** for stability and simplicity. Key characteristics:

- **Simplified Trajectory Balance**: Uses the form without backward probabilities `P_B(τ)`
- **Periodic Z Estimation**: Z is re-estimated every 10 iterations by summing terminal rewards
- **Stability Focus**: Chosen for robust training rather than theoretical generality

## Flow Matching (FM)

Flow Matching is one of the most direct training objectives for GFlowNets. It explicitly enforces flow conservation at each state by training a neural network to estimate flows.

### Mathematical Formulation

The Flow Matching objective minimizes the squared difference between the neural network flow estimate and the true flow computed recursively:

$$\mathcal{L}_{FM}(s) = (Z(s) - F(s))^2$$

where:
- $Z(s)$ is the neural network's flow estimate for state $s$
- $F(s)$ is the true flow computed as:
  - $F(s) = R(s)$ for terminal states
  - $F(s) = \sum_{s'} P_F(s'|s) \cdot F(s')$ for non-terminal states

### Practical Implementation

The current implementation in GFlowNet.jl:

```julia
# Neural network estimates flow
estimated_flow = flow_estimate(model.flow_estimator, state, params, states)

# Compute expected flow recursively
expected_flow = 0.0
for action in applicable_actions
    next_state = apply_action(action, state)
    transition_prob = P_F(next_state|state)
    next_flow = flow(model, next_state)  # Recursive computation
    expected_flow += transition_prob * next_flow
end

# Loss is squared difference
loss = (estimated_flow - expected_flow)^2
```

Key features:
1. Uses a dedicated neural network (`flow_estimator`) to predict F(s)
2. Computes true flows recursively with memoization for efficiency
3. Only requires forward policy (no backward policy needed)
4. Flows are treated as fixed during gradient computation

### Advantages and Limitations

**Advantages:**
- Direct enforcement of flow conservation
- Provides explicit flow estimates for analysis
- No backward policy required
- Can be combined with other objectives
- Efficient with memoization

**Limitations:**
- Requires recursive flow computation (can be expensive)
- May converge slower than trajectory-based methods
- Flow estimates may be less accurate early in training

### When to Use Flow Matching

Flow Matching is particularly useful when:
- You need explicit flow estimates for analysis
- Backward policy is not available or hard to define
- You want to visualize flow distribution
- Combined with other objectives for better convergence

## Detailed Balance (DB)

Detailed Balance focuses on enforcing consistency between forward and backward policies at the edge level.

### Mathematical Formulation

The Detailed Balance objective enforces the following constraint for all edges $(s, s')$:

$$F(s) \cdot P_F(s' | s) = F(s') \cdot P_B(s | s')$$

where:
- $F(s)$ is the flow through state $s$
- $P_F(s' | s)$ is the forward policy probability of transitioning from $s$ to $s'$
- $P_B(s | s')$ is the backward policy probability of transitioning from $s'$ to $s$

This constraint can be transformed into a loss function:

$$\mathcal{L}_{DB}(F) = \sum_{(s, s') \in \mathcal{A}} \left( \log F(s) + \log P_F(s' | s) - \log F(s') - \log P_B(s | s') \right)^2$$

where $\mathcal{A}$ is the set of all edges.

### Practical Implementation

The Detailed Balance objective requires both forward and backward policies. It can be implemented by:

1. Sampling edges $(s, s')$ from trajectories
2. Computing the forward and backward policy probabilities
3. Computing the state flows $F(s)$ and $F(s')$
4. Minimizing the squared log-ratio

### Advantages and Limitations

**Advantages:**
- Provides a local consistency constraint at the edge level
- Can lead to more stable training compared to Flow Matching
- Works well with parameterizations that share parameters between forward and backward policies

**Limitations:**
- Requires a backward policy, which adds parameters to learn
- Can be sensitive to the flow parametrization
- May still struggle with long trajectories

## Trajectory Balance (TB) - Current Implementation

Trajectory Balance, introduced by Malkin et al. (2022), provides a more efficient credit assignment mechanism by considering entire trajectories. **This is the primary objective used in the current implementation.**

### Mathematical Formulation

#### General Form (Not Used Here)
The complete Trajectory Balance equation is:
$$Z \cdot P_F(\tau) = R(s_{|\tau|-1}) \cdot P_B(\tau)$$

#### Simplified Form (Used in This Implementation)
For problems with **deterministic backward paths** (where each state has exactly one parent), $P_B(\tau) = 1$, simplifying to:

$$Z \cdot P_F(\tau) = R(s_{|\tau|-1})$$

where:
- $\tau = (s_0, s_1, \ldots, s_{|\tau|-1})$ is a trajectory from the initial state to a terminal state
- $Z$ is the partition function (total flow)
- $P_F(\tau) = \prod_{t=0}^{|\tau|-1} P_F(s_{t+1} | s_t)$ is the product of forward transition probabilities
- $R(s_{|\tau|-1})$ is the reward of the terminal state

This leads to the loss function:

$$\mathcal{L}_{TB}(F) = \mathbb{E}_{\tau \sim \rho(\tau)} \left[ \left( \log Z + \sum_{t=0}^{|\tau|-1} \log P_F(s_{t+1} | s_t) - \log R(s_{|\tau|-1}) \right)^2 \right]$$

### Implementation Details

```julia
# Compute the trajectory probability
forward_prob_product = 1.0
for i in 1:(length(trajectory.states)-1)
    source = trajectory.states[i]
    target = trajectory.states[i+1]
    prob = forward_transition_prob(working_model, source, target)
    forward_prob_product *= prob
end

# Get Z (partition function) - estimated periodically
Z = model.partition_function

# Compute the loss
ratio = (Z * forward_prob_product) / final_reward
log_ratio = log(ratio)
total_loss += log_ratio^2
```

### Z Estimation Strategy

**This implementation uses periodic estimation rather than joint learning:**

```julia
# In the training loop (every 10 iterations)
if iter % 10 == 0
    model.partition_function = estimate_partition_function(model)
end

# Simple estimation: sum of all terminal rewards
function estimate_partition_function(model::GFlowNetModel)
    total = 0.0
    for state in model.dag.terminal_states
        total += reward(state)
    end
    return total
end
```

**Why This Approach?**
1. **Stability**: Avoids instability from joint Z-policy optimization
2. **Simplicity**: No additional hyperparameters or learning rates for Z
3. **Predictability**: Deterministic Z updates based on known terminal states

### When the Simplified Form is Valid

The simplified Trajectory Balance (without $P_B(\tau)$) is appropriate when:

1. **Sequential Construction**: Objects are built step-by-step in a deterministic manner
2. **Unique Parents**: Each state (except initial) has exactly one possible parent
3. **Examples**: 
   - Building molecules atom-by-atom with specific attachment rules
   - Grid world navigation (each position reached from one previous position)
   - **Not suitable**: Set construction where order doesn't matter (like feature acquisition)

### Advantages and Limitations

**Advantages:**
- Better credit assignment for long trajectories
- Often leads to faster convergence and better sample quality
- More robust to parametrization choices
- Works well with off-policy learning
- **No backward policy needed** in simplified form

**Limitations:**
- Requires sampling complete trajectories
- May have higher variance for very long trajectories
- **Simplified form only works for deterministic backward paths**
- Periodic Z estimation may lag behind policy learning

## Comparing Training Objectives

The choice of training objective depends on the specific application and constraints:

| Objective | Credit Assignment | Backward Policy Required | Computational Complexity | Sample Efficiency |
|-----------|-------------------|--------------------------|--------------------------|-------------------|
| Flow Matching | Local | No | Moderate | Moderate |
| Detailed Balance | Edge-level | Yes | Low | Moderate |
| **Trajectory Balance (Current)** | **Trajectory-level** | **No** | **Low** | **High** |

## Implementation Recommendations

### For New Domains

1. **Start with Trajectory Balance** (current implementation) if:
   - Sequential construction process
   - Each state has a unique parent
   - Want simple, stable training

2. **Consider the general TB form** if:
   - States can have multiple parents
   - Order of actions doesn't matter
   - Need theoretical completeness

3. **Use Detailed Balance** if:
   - Want local, edge-level control
   - Have specific backward policy requirements
   - Dealing with complex, non-tree-like state graphs

### Practical Considerations

When implementing GFlowNet training objectives, consider the following:

1. **Off-policy learning**: All objectives can be used with off-policy learning, but they differ in their sensitivity to the behavior policy.

2. **Parametrization**: The choice of parametrization for flows and policies can significantly impact performance.

3. **Learning rate scheduling**: Different objectives may benefit from different learning rate schedules.

4. **Reward scaling**: Some objectives (particularly TB) can be sensitive to reward scaling. Consider log-transforming rewards if they have a wide dynamic range.

5. **Exploration strategies**: All objectives require adequate exploration of the state space.

## Conclusion

**The current implementation uses Trajectory Balance with periodic Z estimation**, chosen for its excellent balance of:
- **Stability**: Robust training without hyperparameter sensitivity
- **Efficiency**: Good credit assignment for sequential tasks
- **Simplicity**: Minimal implementation complexity

This approach works well for the domains implemented (grid world, molecular design, causal discovery, active learning) where the backward path is typically deterministic.

For new domains with more complex state relationships, consider whether the simplified TB form is appropriate, or if the general form (with $P_B(\tau)$) would be more suitable.

## References

1. Bengio, E., Jain, M., Korablyov, M., Precup, D., & Bengio, Y. (2021). Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation. Advances in Neural Information Processing Systems, 34.

2. Malkin, N., Jain, M., Bengio, E., Sun, C., & Bengio, Y. (2022). Trajectory balance: Improved credit assignment in GFlowNets. NeurIPS 2022.

3. Madan, R., Charpagne, M.A., Shen, C., Bengio, Y., & Bengio, E. (2023). Sub-Trajectory Balance: Improving Credit Assignment in GFlowNets. arXiv preprint.

4. Zhang, L., Madan, R., Zhang, C.L., Bengio, Y., Garg, A., & Bengio, E. (2023). Amortized Flow Transport Monte Carlo. ICML 2023.
