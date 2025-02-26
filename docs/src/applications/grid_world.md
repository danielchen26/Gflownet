# Grid World Navigation with GFlowNets

Grid World navigation is a classic problem in reinforcement learning where an agent must navigate through a 2D grid to find high-reward states. This simple environment provides an excellent introduction to GFlowNets because of its discrete, easy-to-visualize state space and intuitive sequential decision-making process.

## Why Use GFlowNets for Grid World?

The Grid World example serves as an ideal introduction to GFlowNets because:

1. The state space is discrete and easy to visualize
2. The action space is simple (move in four directions)
3. The sequential decision-making process is intuitive
4. It demonstrates key GFlowNet concepts without domain-specific complexity

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

### Trajectory Balance

Our implementation uses the Trajectory Balance objective, which aligns the product of forward probabilities along a trajectory with the reward of the terminal state:

$$\prod_{t=0}^{T-1} P_F(s_{t+1} | s_t) \propto R(s_T)$$

Taking the logarithm, we aim to minimize:

$$\mathcal{L}_{\text{TB}} = \left( \sum_{t=0}^{T-1} \log P_F(s_{t+1} | s_t) - \log R(s_T) + \log Z \right)^2$$

where $Z$ is the partition function (sum of all rewards).

## Implementation Details

```julia
struct GridState <: GFlowNet.AbstractState
    x::Int  # x-coordinate
    y::Int  # y-coordinate
    is_terminal::Bool
end

# Action types
struct MoveRightAction <: GFlowNet.AbstractAction end
struct MoveLeftAction <: GFlowNet.AbstractAction end
struct MoveUpAction <: GFlowNet.AbstractAction end
struct MoveDownAction <: GFlowNet.AbstractAction end
struct TerminateAction <: GFlowNet.AbstractAction end
```

### Feature Representation

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

```julia
const REWARD_POSITIONS = [(5, 5) => 10.0, (3, 4) => 5.0, (2, 2) => 2.0]

function GFlowNet.reward(state::GridState)
    if !state.is_terminal
        return 0.0
    end
    
    # Check if the position has a reward
    for ((x, y), r) in REWARD_POSITIONS
        if state.x == x && state.y == y
            return r
        end
    end
    
    # Default reward for terminal states with no special reward
    return 0.1
end
```

## Using the Grid World Example

To run the grid world example:

```julia
julia examples/grid_world/grid_world.jl
```

The example will:
1. Define a 5×5 grid with rewards at specific positions
2. Train a GFlowNet using the trajectory balance objective
3. Visualize the training process and resulting policy
4. Generate and plot sample trajectories

## Intuitive Explanation

To understand GFlowNets intuitively in the grid world context:

1. Imagine water flowing through a network of pipes, where the source is the starting position (1,1) and the sinks are all possible terminal states.
2. The width of each pipe represents the probability of taking that path.
3. The amount of water flowing to each sink (terminal state) is proportional to the reward at that state.

During training, the network adjusts the pipe widths (policy parameters) to ensure that the flow of water (probability mass) to each terminal state is proportional to its reward. This allows the trained model to sample high-reward trajectories more frequently, while still maintaining diversity.

## Relation to Other Methods

GFlowNets can be compared to several other approaches:

1. **Reinforcement Learning**: Unlike traditional RL which focuses on finding the optimal policy to maximize cumulative reward, GFlowNets aim to sample from the reward distribution.

2. **MCMC Methods**: Both sample according to a target distribution, but GFlowNets construct samples sequentially rather than modifying existing ones.

3. **Variational Autoencoders**: Both are generative models, but GFlowNets directly model the generation process as a flow network.

## Further Reading

For a more detailed explanation of grid world navigation with GFlowNets, see the comprehensive documentation in the examples directory. 