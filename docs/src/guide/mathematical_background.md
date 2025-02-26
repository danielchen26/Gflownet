# Mathematical Background

This page provides the mathematical foundation for understanding GFlowNets.

## Flow Networks

A flow network is defined as a directed acyclic graph (DAG) $G = (S, A)$, where:
- $S$ is the set of states
- $A$ is the set of actions/edges where $(s, s') \in A$ means there is an edge from state $s$ to state $s'$

We designate a special initial state $s_0$ and a set of terminal states $S_T$.

## Flow Conservation

The key property of a flow network is flow conservation. For any state $s$ that is not a terminal state:

$$F(s) = \sum_{s': (s', s) \in A} F(s', s) = \sum_{s': (s, s') \in A} F(s, s')$$

where:
- $F(s)$ is the flow through state $s$
- $F(s, s')$ is the flow along the edge from $s$ to $s'$

## Reward and Terminal Flow

GFlowNets are designed to sample from a distribution defined by a reward function $R: S_T \rightarrow \mathbb{R}^+$. The terminal flow at each terminal state $s_T$ is proportional to the reward:

$$F(s_T) = R(s_T)$$

## Forward Policy

The forward policy defines the probability of taking an action $a$ from state $s$:

$$P_F(s' | s) = \frac{F(s, s')}{F(s)}$$

This policy allows sampling trajectories according to the flow.

## Backward Policy

The backward policy defines the probability of taking a reverse action from state $s'$ to state $s$:

$$P_B(s | s') = \frac{F(s, s')}{F(s')}$$

## Flow Parameterization

In practice, the flow is parameterized by a neural network $F_\theta$ with parameters $\theta$. This network learns to approximate the true flow.

## Connection to Probabilistic Models

GFlowNets have connections to several probabilistic models:
- The probability of a terminal state $s_T$ under a GFlowNet is $P(s_T) = \frac{R(s_T)}{Z}$
- $Z = \sum_{s_T \in S_T} R(s_T)$ is the normalization constant
- This is equivalent to a Boltzmann distribution $P(s_T) = \frac{e^{-E(s_T)}}{Z}$ with energy $E(s_T) = -\log R(s_T)$
