# GFlowNet: A Comprehensive Framework for Generative Flow Networks

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundations](#mathematical-foundations)
   - [Flow Networks](#flow-networks)
   - [Directed Acyclic Graphs (DAGs)](#directed-acyclic-graphs)
   - [Probability Flow](#probability-flow)
3. [GFlowNet Architecture](#gflownet-architecture)
   - [Core Components](#core-components)
   - [State and Action Spaces](#state-and-action-spaces)
   - [Flow Functions](#flow-functions)
4. [Training Objectives](#training-objectives)
   - [Flow Matching (FM)](#flow-matching)
   - [Detailed Balance (DB)](#detailed-balance)
   - [Trajectory Balance (TB)](#trajectory-balance)
5. [Mathematical Derivations](#mathematical-derivations)
   - [Flow Consistency Equations](#flow-consistency-equations)
   - [Training Objectives Derivation](#training-objectives-derivation)
   - [Relationship to Variational Inference](#relationship-to-variational-inference)
6. [Extensions and Variants](#extensions-and-variants)
   - [Continuous State Spaces](#continuous-state-spaces)
   - [Non-Acyclic GFlowNets](#non-acyclic-gflownets)
   - [Entropy and Mutual Information Estimation](#entropy-and-mutual-information-estimation)
7. [Applications](#applications)
   - [Molecular Design](#molecular-design)
   - [Causal Discovery](#causal-discovery)
   - [Active Learning](#active-learning)
8. [Implementation Considerations](#implementation-considerations)
   - [Neural Network Architectures](#neural-network-architectures)
   - [Hyperparameter Selection](#hyperparameter-selection)
9. [Future Directions](#future-directions)
10. [References](#references)

## Introduction

Generative Flow Networks (GFlowNets) are a novel class of generative models introduced by Bengio et al. (2021) that lie at the intersection of reinforcement learning, deep generative models, and energy-based probabilistic modeling. GFlowNets offer a powerful framework for sampling from complex, high-dimensional distributions over discrete compositional objects such as graphs, molecules, or sets, where many traditional sampling methods like Markov Chain Monte Carlo (MCMC) struggle to efficiently explore multimodal distributions.

The key innovation of GFlowNets is their ability to model flows of probabilities through a directed acyclic graph (DAG) representing the sequential construction of complex objects. By learning these flows, GFlowNets can generate samples with probabilities proportional to a given reward or energy function, effectively amortizing the cost of sampling from complex distributions into a single trained generative pass.

## Mathematical Foundations

### Flow Networks

At the core of GFlowNets is the concept of a flow network, which is a mathematical structure used to model the distribution of flow through a directed graph. In the context of GFlowNets, we define:

**Definition 1**: A flow network is a tuple $(G, F)$ where:
- $G = (𝒮, 𝔸)$ is a directed acyclic graph with states $𝒮$ and actions $𝔸$
- $F$ is a non-negative flow function that assigns values to edges or states in the graph

The flow function $F$ must satisfy certain conservation properties, namely:

$$\forall s \in 𝒮 \setminus \{s_0, s_f\}: \sum_{s' \in \text{Parent}(s)} F(s' \rightarrow s) = \sum_{s' \in \text{Child}(s)} F(s \rightarrow s')$$

Where $s_0$ is the initial state and $s_f$ is the terminal sink state.

### Directed Acyclic Graphs

In GFlowNets, the directed acyclic graph $G$ represents the set of all possible trajectories for constructing complex objects. Each state $s \in 𝒮$ represents a partially constructed object, and actions $a \in 𝔸$ represent transitions that construct the object step by step.

**Definition 2**: A pointed DAG is a directed acyclic graph with a distinguished initial state $s_0$ and a terminal sink state $s_f$. All complete trajectories in the graph start at $s_0$ and end at $s_f$.

For a graph generation task, a state might represent a partially constructed graph, and actions might involve adding nodes or edges. The terminal states $𝒮^f \subset 𝒮$ represent the complete objects.

### Probability Flow

GFlowNets learn to model probability flows on this graph structure:

**Definition 3**: Given a reward function $R: 𝒮^f \rightarrow \mathbb{R}^+$ that assigns a non-negative value to each terminal state, a probability flow in a GFlowNet satisfies:

$$\forall s \in 𝒮^f: F(s \rightarrow s_f) = R(s)$$

Where $F(s \rightarrow s_f)$ is the flow from the terminal state $s$ to the sink $s_f$.

The total flow entering the network equals the total flow exiting:

$$F(s_0) = \sum_{s \in 𝒮^f} R(s) = Z$$

Where $Z$ is analogous to a partition function in statistical physics.

## GFlowNet Architecture

### Core Components

A GFlowNet consists of the following core components:

1. **State Space ($𝒮$)**: The set of all possible partially constructed objects
2. **Action Space ($𝔸$)**: The set of all possible actions that transform states
3. **Transition Function $T$**: A function $T: 𝒮 \times 𝔸 \rightarrow 𝒮$ that maps a state-action pair to the next state
4. **Flow Function $F$**: A function that assigns non-negative values to states or edges
5. **Policy Network**: A neural network that parametrizes the forward transition probabilities $P_F(s' | s)$

**Formal Definition**: A GFlowNet is a tuple $(G, R, 𝒪, \Pi, \mathcal{H})$ where:
- $G = (𝒮, 𝔸)$ is a pointed DAG with initial state $s_0$ and sink state $s_f$
- $R: 𝒮^f \rightarrow \mathbb{R}^+$ is a target reward function
- $(𝒪, \Pi, \mathcal{H})$ is a flow parametrization of $(G, R)$ where:
  - $𝒪$ is a non-empty set of GFlowNet configurations
  - $\Pi$ maps each configuration to a probability distribution over trajectories
  - $\mathcal{H}$ is an injective mapping from flows to configurations

### State and Action Spaces

The design of state and action spaces depends on the specific application:

- For **molecular generation**, states are partial molecules and actions add atoms or bonds
- For **graph generation**, states are partial graphs and actions add nodes or edges
- For **set generation**, states are partial sets and actions add elements

GFlowNets are particularly well-suited for objects with combinatorial structure, where the number of possible objects grows exponentially with their size.

### Flow Functions

There are multiple ways to parametrize flows in a GFlowNet:

1. **State Flow Function $F(s)$**: Assigns a flow value to each state
2. **Edge Flow Function $F(s \rightarrow s')$**: Assigns a flow value to each edge
3. **Forward Transition Probabilities $P_F(s' | s)$**: The probability of transitioning from state $s$ to $s'$
4. **Backward Transition Probabilities $P_B(s | s')$**: The probability of reaching $s$ given that we're at $s'$

These parametrizations are related through the following equations:

$$F(s \rightarrow s') = F(s) \cdot P_F(s' | s)$$
$$F(s \rightarrow s') = F(s') \cdot P_B(s | s')$$

## Training Objectives

GFlowNets can be trained using several objectives, each with different properties:

### Flow Matching

The flow matching (FM) objective enforces flow conservation at each state:

$$\mathcal{L}_{FM}(F) = \sum_{s \in 𝒮 \setminus \{s_0, s_f\}} \left( \sum_{s' \in \text{Parent}(s)} F(s' \rightarrow s) - \sum_{s' \in \text{Child}(s)} F(s \rightarrow s') \right)^2$$

This ensures that for each non-terminal state, the sum of incoming flows equals the sum of outgoing flows.

### Detailed Balance

The detailed balance (DB) objective enforces consistency between forward and backward transition probabilities:

$$\mathcal{L}_{DB}(F) = \sum_{(s, s') \in 𝔸} \left( F(s) \cdot P_F(s' | s) - F(s') \cdot P_B(s | s') \right)^2$$

This objective is analogous to the detailed balance condition in MCMC methods.

### Trajectory Balance

The trajectory balance (TB) objective, introduced by Malkin et al. (2022), provides more efficient credit assignment:

$$\mathcal{L}_{TB}(F) = \mathbb{E}_{\tau \sim \rho(\tau)} \left[ \left( \frac{Z \cdot \prod_{t=0}^{|\tau|-1} P_F(s_{t+1} | s_t)}{R(s_{|\tau|-1})} - 1 \right)^2 \right]$$

Where $\tau = (s_0, s_1, \ldots, s_{|\tau|-1})$ is a trajectory and $\rho(\tau)$ is a sampling distribution.

Trajectory balance directly enforces consistency across entire trajectories, leading to faster convergence and better performance for long action sequences.

## Mathematical Derivations

### Flow Consistency Equations

For a properly trained GFlowNet, the flow function $F$ must satisfy the following consistency equations:

1. **Flow Conservation**: For all non-terminal states $s \in 𝒮 \setminus \{s_0, s_f\}$:
   $$\sum_{s' \in \text{Parent}(s)} F(s' \rightarrow s) = \sum_{s' \in \text{Child}(s)} F(s \rightarrow s')$$

2. **Terminal State Flow**: For all terminal states $s \in 𝒮^f$:
   $$F(s \rightarrow s_f) = R(s)$$

3. **Total Flow**: The total flow entering the network equals the partition function:
   $$F(s_0) = \sum_{s \in 𝒮^f} R(s) = Z$$

These consistency equations ensure that the GFlowNet learns to sample objects with probability proportional to their reward:

$$P(s) = \frac{R(s)}{Z}$$

### Training Objectives Derivation

The different training objectives can be derived from these consistency equations:

**Flow Matching Objective**: The flow matching objective directly minimizes the violation of the flow conservation equation:

$$\mathcal{L}_{FM}(F) = \sum_{s \in 𝒮 \setminus \{s_0, s_f\}} \left( \sum_{s' \in \text{Parent}(s)} F(s' \rightarrow s) - \sum_{s' \in \text{Child}(s)} F(s \rightarrow s') \right)^2$$

**Detailed Balance Objective**: The detailed balance objective enforces consistency between forward and backward flows:

$$\mathcal{L}_{DB}(F) = \sum_{(s, s') \in 𝔸} \left( F(s) \cdot P_F(s' | s) - F(s') \cdot P_B(s | s') \right)^2$$

**Trajectory Balance Objective**: The trajectory balance objective can be derived by considering the ratio of the probability of a trajectory under the GFlowNet policy to the target probability:

$$\mathcal{L}_{TB}(F) = \mathbb{E}_{\tau \sim \rho(\tau)} \left[ \left( \frac{Z \cdot \prod_{t=0}^{|\tau|-1} P_F(s_{t+1} | s_t)}{R(s_{|\tau|-1})} - 1 \right)^2 \right]$$

### Relationship to Variational Inference

GFlowNets are closely related to variational inference (VI) methods:

1. Both aim to approximate complex probability distributions.
2. GFlowNets with trajectory balance are similar to hierarchical VI.
3. However, GFlowNets offer better trade-offs between modeling the best solution (like reverse KL divergence in VI) and modeling a blend of solutions (like forward KL).
4. GFlowNets can be more easily trained off-policy compared to hierarchical VI.

The key difference is that GFlowNets optimize objectives like flow matching or trajectory balance, which enforce internal consistency, while VI methods typically minimize divergence metrics directly.

## Extensions and Variants

### Continuous State Spaces

GFlowNets can be extended to continuous state spaces by:

1. Defining flows and policies as density functions rather than discrete probabilities
2. Replacing sums with integrals in the flow conservation equations
3. Using parametrized continuous distributions for transition probabilities

For a state with both discrete and continuous components $s = (s_i, s_x)$ where $s_i$ is discrete and $s_x$ is continuous, the flow becomes:

$$F((s_i, s_x) \rightarrow (s_i', s_x')) = F(s_i, s_x) \cdot P_F((s_i', s_x') | (s_i, s_x))$$

### Non-Acyclic GFlowNets

Recent work by Brunswic et al. (2023) has extended GFlowNets to non-acyclic graphs, allowing for:

1. Cycles in the state transition graph
2. More flexible modeling of complex systems
3. Applications to continuous state spaces without cycle restrictions

Standard GFlowNet losses can push flows to get stuck in cycles, so specialized loss functions are needed for the non-acyclic case.

### Entropy and Mutual Information Estimation

GFlowNets can be used to estimate information-theoretic quantities:

**Entropy**: For a probability distribution $P(s) = \frac{R(s)}{Z}$, the entropy can be estimated as:

$$H(P) = \log Z - \mathbb{E}_{s \sim P}[\log R(s)]$$

**Mutual Information**: For two variables $X$ and $Y$, the mutual information can be estimated as:

$$I(X; Y) = \log Z - \mathbb{E}_{x \sim P(X)}[\log Z_x]$$

Where $Z_x$ is the conditional partition function for a fixed $x$.

## Applications

### Molecular Design

GFlowNets have been successfully applied to molecular design:

1. States represent partial molecules
2. Actions include adding atoms, bonds, or functional groups
3. Rewards are based on molecular properties (e.g., binding affinity, solubility)

GFlowNets can generate diverse molecules with high reward, exploring multiple modes of the distribution rather than converging to a single "best" molecule.

### Causal Discovery

For learning causal graphs:

1. States represent partial causal graphs
2. Actions add edges between variables
3. Rewards are based on how well the graph explains observed data

GFlowNets can model the full distribution over plausible causal graphs rather than committing to a single one, which is crucial when multiple causal explanations are consistent with limited observational data.

### Active Learning

GFlowNets can be used for active learning and exploration:

1. Learn a distribution over candidate solutions
2. Sample diverse, high-reward candidates for evaluation
3. Update the reward function based on evaluations
4. Retrain the GFlowNet with the updated rewards

This creates a feedback loop that efficiently explores the solution space.

## Implementation Considerations

### Neural Network Architectures

GFlowNets can be implemented using various neural network architectures:

1. **Forward Policy Network**: Maps states to a distribution over next actions
2. **Backward Policy Network**: Maps states to a distribution over previous actions
3. **Flow Network**: Directly estimates state or edge flows

The architecture choice depends on the specific application and state/action space structure.

### Hyperparameter Selection

Important hyperparameters include:

1. Learning rates for the forward policy, backward policy, and flow networks
2. Replay buffer size and sampling strategy
3. Choice of training objective (FM, DB, or TB)
4. Off-policy learning parameters
5. Neural network architecture and capacity

Trajectory balance often performs better than flow matching or detailed balance for complex problems with long action sequences.

## Future Directions

Promising future directions for GFlowNet research include:

1. **Theoretical understanding**: Further exploring the connections to variational inference, reinforcement learning, and energy-based models
2. **Algorithmic improvements**: Developing more efficient training objectives and architectures
3. **Application domains**: Expanding to new areas such as program synthesis, protein design, and scientific discovery
4. **Hybrid models**: Combining GFlowNets with other generative approaches
5. **Distributed training**: Scaling up GFlowNets to larger and more complex problems

## References

1. Bengio, E., Jain, M., Korablyov, M., Precup, D., & Bengio, Y. (2021). Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation. Advances in Neural Information Processing Systems, 34.

2. Bengio, Y., Deleu, T., Lahlou, S., Hu, E.J., Tiwari, M., & Bengio, E. (2021). GFlowNet Foundations. arXiv:2111.09266.

3. Malkin, N., Jain, M., Bengio, E., Sun, C., & Bengio, Y. (2022). Trajectory balance: Improved credit assignment in GFlowNets. NeurIPS 2022.

4. Hu, E., Malkin, N., & Everett, K. (2023). What do GFlowNets and Variational Inference Have in Common? Mila Blog.

5. Brunswic, L.M., Li, Y., Xu, Y., Jui, S., & Ma, L. (2023). A Theory of Non-Acyclic Generative Flow Networks. AAAI 2024.

6. Zhang, M., Bengio, S., & Bengio, Y. (2022). The Mutual Information Between GFlowNet Trajectories and Rewards. arXiv preprint.

# Additional Examples

The package includes several example applications to demonstrate GFlowNet usage:

## Grid World Navigation

A simple grid world example demonstrating the basic concepts of GFlowNets. The agent learns to navigate
a 2D grid world to find high-reward locations.

Run the example:
```bash
julia examples/grid_world.jl
```

## Causal Discovery

An example showing how GFlowNets can be used for causal discovery, finding directed acyclic graphs
that best explain observed data.

Run the example:
```bash
julia examples/causal_discovery.jl
```

## Active Learning

An example demonstrating how GFlowNets can be used for active learning and experimental design,
selecting a diverse and informative set of experiments.

Run the example:
```bash
julia examples/active_learning.jl
```

## Applications

The package includes several more specialized applications:

1. **Molecular Design**: Generate molecular structures with desired properties
2. **Causal Discovery**: Discover causal structures from data
3. **Active Learning**: Select informative experiments

## Extensions

Additional extensions to the base GFlowNet framework:

1. **Continuous State Spaces**: Support for continuous state and action spaces
2. **Non-Acyclic GFlowNets**: Support for flow networks with cycles
3. **Information-Based Methods**: Entropy estimation and mutual information

