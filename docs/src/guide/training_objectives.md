# Training Objectives

This page provides a comprehensive overview of the different training objectives used in GFlowNets, their mathematical derivations, practical considerations, and relative strengths.

## Introduction to Training Objectives

GFlowNets can be trained using several different objectives, each with unique properties and trade-offs. The choice of training objective can significantly impact the convergence rate, sample quality, and overall performance of the model.

At a high level, all GFlowNet training objectives aim to ensure that the network satisfies the flow consistency conditions:

1. **Flow Conservation**: For all non-terminal states, the incoming flow equals the outgoing flow
2. **Terminal State Flow**: The flow at terminal states is proportional to their rewards
3. **Total Flow**: The total flow through the network equals the sum of all terminal rewards

The training objectives differ in how they enforce these constraints and in their computational and statistical properties.

## Flow Matching (FM)

Flow Matching is one of the most direct training objectives for GFlowNets. It explicitly enforces flow conservation at each state.

### Mathematical Formulation

The Flow Matching objective minimizes the squared difference between incoming and outgoing flows at each non-terminal state:

$$\mathcal{L}_{FM}(F) = \sum_{s \in \mathcal{S} \setminus \{s_0, \mathcal{S}_T\}} \left( \sum_{s' \in \text{Parent}(s)} F(s' \rightarrow s) - \sum_{s' \in \text{Child}(s)} F(s \rightarrow s') \right)^2$$

where:
- $\mathcal{S}$ is the set of all states
- $s_0$ is the initial state
- $\mathcal{S}_T$ is the set of terminal states
- $F(s' \rightarrow s)$ is the flow along the edge from $s'$ to $s$

Additionally, for terminal states $s_T \in \mathcal{S}_T$, we enforce:

$$F(s_T \rightarrow s_f) = R(s_T)$$

where $s_f$ is a virtual sink state and $R(s_T)$ is the reward of terminal state $s_T$.

### Practical Implementation

In practice, the Flow Matching objective can be implemented by:

1. Sampling states from trajectories
2. Computing the incoming and outgoing flows for each sampled state
3. Minimizing the squared difference

A key challenge is that we need to compute flows for all parents and children of each sampled state, which can be computationally expensive if the branching factor is large.

### Advantages and Limitations

**Advantages:**
- Direct enforcement of flow conservation
- Conceptually simple and aligned with the theoretical foundation
- Can be used with off-policy learning

**Limitations:**
- May suffer from credit assignment issues for long trajectories
- Requires computing flows for all parents and children of sampled states
- Can be sensitive to parametrization choices

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

## Trajectory Balance (TB)

Trajectory Balance, introduced by Malkin et al. (2022), provides a more efficient credit assignment mechanism by considering entire trajectories.

### Mathematical Formulation

The Trajectory Balance objective enforces consistency across complete trajectories:

$$\frac{Z \cdot \prod_{t=0}^{|\tau|-1} P_F(s_{t+1} | s_t)}{R(s_{|\tau|-1})} = 1$$

where:
- $\tau = (s_0, s_1, \ldots, s_{|\tau|-1})$ is a trajectory from the initial state to a terminal state
- $Z$ is the partition function (total flow)
- $P_F(s_{t+1} | s_t)$ is the forward policy probability
- $R(s_{|\tau|-1})$ is the reward of the terminal state

This constraint leads to the loss function:

$$\mathcal{L}_{TB}(F) = \mathbb{E}_{\tau \sim \rho(\tau)} \left[ \left( \log Z + \sum_{t=0}^{|\tau|-1} \log P_F(s_{t+1} | s_t) - \log R(s_{|\tau|-1}) \right)^2 \right]$$

where $\rho(\tau)$ is a sampling distribution over trajectories.

### Practical Implementation

Trajectory Balance can be implemented by:

1. Sampling complete trajectories using a behavior policy
2. Computing the product of forward probabilities along each trajectory
3. Minimizing the squared difference between this product (scaled by $Z$) and the terminal reward

The partition function $Z$ can be parameterized directly or computed as the flow at the initial state.

### Advantages and Limitations

**Advantages:**
- Better credit assignment for long trajectories
- Often leads to faster convergence and better sample quality
- More robust to parametrization choices
- Works well with off-policy learning

**Limitations:**
- Requires sampling complete trajectories
- May have higher variance for very long trajectories
- Sensitive to the estimation of the partition function $Z$

## Sub-Trajectory Balance (SubTB)

Sub-Trajectory Balance is an extension of Trajectory Balance that applies the same principle to sub-trajectories.

### Mathematical Formulation

For any sub-trajectory $\tau_{i:j} = (s_i, s_{i+1}, \ldots, s_j)$:

$$\frac{F(s_i) \cdot \prod_{t=i}^{j-1} P_F(s_{t+1} | s_t)}{F(s_j)} = 1$$

This leads to the loss function:

$$\mathcal{L}_{SubTB}(F) = \mathbb{E}_{\tau \sim \rho(\tau)} \mathbb{E}_{(i,j)} \left[ \left( \log F(s_i) + \sum_{t=i}^{j-1} \log P_F(s_{t+1} | s_t) - \log F(s_j) \right)^2 \right]$$

### Practical Implementation

Sub-Trajectory Balance can be implemented by:

1. Sampling complete trajectories
2. Randomly selecting sub-trajectories
3. Computing the balance condition for each sub-trajectory
4. Minimizing the squared difference

### Advantages and Limitations

**Advantages:**
- Combines benefits of Trajectory Balance with more local credit assignment
- Can lead to more sample-efficient learning
- Works well for very long trajectories

**Limitations:**
- More complex to implement
- Requires estimating state flows at intermediate states
- May be sensitive to the selection of sub-trajectories

## Amortized Flow Transport (AFT)

Amortized Flow Transport is a recent training objective that combines GFlowNets with ideas from optimal transport.

### Mathematical Formulation

AFT minimizes a transport cost between the flow distribution and the target distribution:

$$\mathcal{L}_{AFT}(F) = \mathbb{E}_{s_T \sim \pi_F} [c(s_T, R)] - \lambda \cdot \mathbb{H}[\pi_F]$$

where:
- $\pi_F$ is the distribution over terminal states induced by the GFlowNet
- $c(s_T, R)$ is a cost function measuring the discrepancy between the sampled state and the target reward
- $\mathbb{H}[\pi_F]$ is the entropy of the induced distribution
- $\lambda$ is a temperature parameter

### Practical Implementation

AFT can be implemented by:

1. Sampling terminal states using the current GFlowNet
2. Computing the transport cost and entropy terms
3. Updating the parameters to minimize the objective

### Advantages and Limitations

**Advantages:**
- Can handle continuous state spaces more naturally
- Provides explicit control over the exploration-exploitation trade-off
- Often more robust to reward scaling

**Limitations:**
- Requires careful choice of the cost function and temperature
- May be more complex to implement
- Less direct connection to the flow interpretation

## Comparing Training Objectives

The choice of training objective depends on the specific application and constraints:

| Objective | Credit Assignment | Parametrization Flexibility | Computational Complexity | Sample Efficiency |
|-----------|-------------------|----------------------------|--------------------------|-------------------|
| Flow Matching | Local | Moderate | Moderate | Moderate |
| Detailed Balance | Edge-level | Low | Low | Moderate |
| Trajectory Balance | Trajectory-level | High | Low | High |
| Sub-Trajectory Balance | Mixed | High | Moderate | Very High |
| Amortized Flow Transport | Global | Very High | High | High |

For most applications, Trajectory Balance offers a good balance of simplicity, efficiency, and performance, particularly for problems with long action sequences.

## Practical Considerations

When implementing GFlowNet training objectives, consider the following:

1. **Off-policy learning**: All objectives can be used with off-policy learning, but they differ in their sensitivity to the behavior policy.

2. **Parametrization**: The choice of parametrization for flows and policies can significantly impact performance. Options include:
   - Direct parametrization of state flows $F(s)$
   - Direct parametrization of edge flows $F(s \rightarrow s')$
   - Parametrization of forward policy $P_F(s' | s)$ and backward policy $P_B(s | s')$
   - Parametrization of forward policy and state flows

3. **Learning rate scheduling**: Different objectives may benefit from different learning rate schedules.

4. **Reward scaling**: Some objectives (particularly TB) can be sensitive to reward scaling. Consider log-transforming rewards if they have a wide dynamic range.

5. **Exploration strategies**: All objectives require adequate exploration of the state space. Consider using techniques like entropy regularization or epsilon-greedy exploration.

## Conclusion

The choice of training objective is a critical design decision when implementing GFlowNets. The best choice depends on the specific problem characteristics, such as trajectory length, state space size, and reward structure.

For beginners, Trajectory Balance is often a good starting point due to its robust performance across a wide range of problems. As you gain experience, you may want to experiment with other objectives or even combinations of objectives to optimize performance for your specific application.

## References

1. Bengio, E., Jain, M., Korablyov, M., Precup, D., & Bengio, Y. (2021). Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation. Advances in Neural Information Processing Systems, 34.

2. Malkin, N., Jain, M., Bengio, E., Sun, C., & Bengio, Y. (2022). Trajectory balance: Improved credit assignment in GFlowNets. NeurIPS 2022.

3. Madan, R., Charpagne, M.A., Shen, C., Bengio, Y., & Bengio, E. (2023). Sub-Trajectory Balance: Improving Credit Assignment in GFlowNets. arXiv preprint.

4. Zhang, L., Madan, R., Zhang, C.L., Bengio, Y., Garg, A., & Bengio, E. (2023). Amortized Flow Transport Monte Carlo. ICML 2023.
