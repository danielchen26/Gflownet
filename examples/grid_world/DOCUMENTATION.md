# Grid World Navigation with GFlowNets

This document provides a comprehensive explanation of the Grid World navigation example implemented using Generative Flow Networks (GFlowNets).

## Conceptual Overview

Grid World navigation is a classic problem in reinforcement learning where an agent must navigate through a 2D grid to find high-reward states. This simple environment provides an excellent introduction to GFlowNets because:

1. The state space is discrete and easy to visualize
2. The action space is simple (move in four directions)
3. The sequential decision-making process is intuitive

In our implementation, we define a 5×5 grid with rewards at specific locations. The agent starts at position (1,1) and can move right, left, up, or down, or choose to terminate at any position.

## Mathematical Framework

### GFlowNet Fundamentals

GFlowNets are generative models that learn a stochastic policy for constructing objects sequentially. The key insight is to model the generation process as a flow network where:

- Each state (grid position in our case) has an associated flow value
- Flow is conserved throughout the network
- Terminal states have flows proportional to their rewards

Formally, for a state $s$, the flow $F(s)$ must satisfy:

$$F(s) = \sum_{s' \in \text{Parents}(s)} F(s') \cdot P_F(s | s')$$

where $P_F(s | s')$ is the forward transition probability from state $s'$ to state $s$.

For terminal states, the flow is directly proportional to the reward:

$$F(s_{\text{terminal}}) = R(s_{\text{terminal}})$$

The goal of training is to learn the parameters of the forward policy that satisfy these flow constraints.

### Trajectory Balance

Our implementation uses the Trajectory Balance objective, which aligns the product of forward probabilities along a trajectory with the reward of the terminal state:

$$\prod_{t=0}^{T-1} P_F(s_{t+1} | s_t) \propto R(s_T)$$

Taking the logarithm, we aim to minimize:

$$\mathcal{L}_{\text{TB}} = \left( \sum_{t=0}^{T-1} \log P_F(s_{t+1} | s_t) - \log R(s_T) + \log Z \right)^2$$

where $Z$ is the partition function (sum of all rewards).

### Neural Network Architecture

Our implementation uses a neural network to parameterize the forward policy. The network takes the state features as input and outputs logits for all possible states in the DAG. We use:

- Input layer: Grid position (one-hot encoded) + terminal flag (5×5+1 = 26 dimensions)
- Hidden layers: 64 neurons with ReLU activation
- Output layer: Linear activation with dimension equal to the number of states

The forward probability distribution is obtained by taking the softmax of the relevant logits:

$$P_F(s_{\text{next}} | s_{\text{current}}) = \frac{\exp(f_\theta(s_{\text{current}})_{s_{\text{next}}})}{\sum_{s' \in \text{Next}(s_{\text{current}})} \exp(f_\theta(s_{\text{current}})_{s'})}$$

where $f_\theta(s_{\text{current}})_{s_{\text{next}}}$ is the logit corresponding to state $s_{\text{next}}$.

## Implementation Details

### State and Action Representation

We represent grid positions using a `GridState` struct:

```julia
struct GridState <: GFlowNet.AbstractState
    x::Int  # x-coordinate
    y::Int  # y-coordinate
    is_terminal::Bool
end
```

The possible actions are:
- `MoveRightAction`
- `MoveLeftAction`
- `MoveUpAction`
- `MoveDownAction`
- `TerminateAction`

### Feature Representation

States are converted to feature vectors using one-hot encoding for the position in the grid, plus a terminal flag:

```julia
function GFlowNet.state_to_features(state::GridState)
    pos_idx = (state.x - 1) * GRID_SIZE + state.y
    grid_size_sq = GRID_SIZE * GRID_SIZE
    
    features = vcat(
        # Position feature (one hot)
        Float32.([(i == pos_idx) for i in 1:grid_size_sq]),
        # Terminal state feature
        Float32[state.is_terminal]
    )
    
    return features
end
```

### Reward Function

Rewards are assigned to specific positions in the grid:

```julia
const REWARD_POSITIONS = [(5, 5) => 10.0, (3, 4) => 5.0, (2, 2) => 2.0]
```

The reward function returns the corresponding reward for terminal states at these positions, and 0 otherwise.

### Training Process

The training loop:
1. Samples trajectories using the current policy
2. Computes the trajectory balance loss
3. Updates the policy parameters using gradient descent
4. Periodically re-estimates the partition function

## Visualization and Analysis

After training, we visualize:
1. The training loss curve
2. Sampled trajectories in the grid world
3. A distribution of rewards obtained from sampled trajectories

## Intuitive Explanation

To understand GFlowNets intuitively in the grid world context:

1. Imagine water flowing through a network of pipes, where the source is the starting position (1,1) and the sinks are all possible terminal states.
2. The width of each pipe represents the probability of taking that path.
3. The amount of water flowing to each sink (terminal state) is proportional to the reward at that state.

During training, the network adjusts the pipe widths (policy parameters) to ensure that the flow of water (probability mass) to each terminal state is proportional to its reward. This allows the trained model to sample high-reward trajectories more frequently, while still maintaining diversity.

The key insight is that GFlowNets learn not just to find the single highest reward state, but to generate a diverse set of states with probability proportional to their rewards.

## Relation to Other Methods

GFlowNets can be compared to several other approaches:

1. **Reinforcement Learning**: Unlike traditional RL which focuses on finding the optimal policy to maximize cumulative reward, GFlowNets aim to sample from the reward distribution.

2. **MCMC Methods**: Both sample according to a target distribution, but GFlowNets construct samples sequentially rather than modifying existing ones.

3. **Variational Autoencoders**: Both are generative models, but GFlowNets directly model the generation process as a flow network.

## Mathematical Derivation of Flow Consistency

For a more rigorous derivation, consider the GFlowNet's flow consistency condition:

For any non-terminal state $s$, the incoming flow equals the outgoing flow:

$$\sum_{s' \in \text{Parents}(s)} F(s', s) = \sum_{s'' \in \text{Children}(s)} F(s, s'')$$

where $F(s', s)$ is the flow on the edge from $s'$ to $s$.

The edge flow can be expressed as:

$$F(s', s) = F(s') \cdot P_F(s | s')$$

For terminal states, the flow is proportional to the reward:

$$F(s_{\text{terminal}}) = R(s_{\text{terminal}})$$

The sum of all terminal flows equals the partition function $Z$:

$$Z = \sum_{s \in \text{Terminal}} R(s)$$

Under these conditions, the probability of sampling a complete trajectory $\tau = (s_0, s_1, \ldots, s_T)$ is:

$$P(\tau) = \prod_{t=0}^{T-1} P_F(s_{t+1} | s_t) = \frac{R(s_T)}{Z}$$

This ensures that terminal states are sampled with probability proportional to their rewards, which is the key objective of GFlowNets.

## Conclusion

The Grid World example provides an intuitive introduction to GFlowNets by demonstrating how they can learn to navigate to high-reward states in a simple environment. The concepts introduced here—such as flow consistency, trajectory balance, and sampling according to rewards—are fundamental to understanding more complex applications of GFlowNets in domains like molecular design, causal discovery, and active learning. 