# Molecular Design with GFlowNets

This document provides a comprehensive explanation of the Molecular Design example implemented using Generative Flow Networks (GFlowNets).

## Conceptual Overview

Molecular design is a challenging task in cheminformatics and drug discovery where the goal is to generate molecules with desired properties. This presents an ideal application for GFlowNets because:

1. The chemical space is vast and discrete
2. Molecules can be constructed sequentially (atom by atom, bond by bond)
3. Rewards are often only meaningful for complete structures
4. We want to discover diverse molecules with desired properties, not just a single "best" molecule

In our implementation, molecules are constructed sequentially by adding atoms and bonds, eventually creating a chemical graph structure that can be evaluated for properties of interest.

## Mathematical Framework

### Molecular State Space

A molecule can be represented as a graph $G = (V, E)$ where:
- $V$ is the set of atoms, each with properties like element type and position
- $E$ is the set of bonds, each with properties like bond type and connecting atoms

The state space consists of all valid molecular graphs that can be constructed sequentially, starting from an empty molecule. Each state $s$ in this space represents a partial or complete molecule.

### Action Space

The action space for molecular design includes:
- $\mathcal{A}_{\text{add\_atom}}$: Add an atom of a specific type at a specific position
- $\mathcal{A}_{\text{add\_bond}}$: Add a bond of a specific type between two existing atoms
- $\mathcal{A}_{\text{terminate}}$: Declare the molecule as complete

Each action $a \in \mathcal{A}$ transforms a state $s$ to a new state $s'$ according to the transition function $s' = T(s, a)$.

### GFlowNet Framework

As in the Grid World example, the GFlowNet for molecular design learns a policy that samples trajectories (sequences of molecular construction steps) with probability proportional to their reward:

$$P(\tau) \propto R(s_T)$$

where $\tau = (s_0, s_1, \ldots, s_T)$ is a trajectory and $s_T$ is the terminal state representing a complete molecule.

### Reward Function

The reward function $R(s)$ for molecular design typically incorporates:
- Chemical validity: Ensuring the molecule follows chemical rules
- Desired properties: Such as drug-likeness, synthesizability, or binding affinity
- Novelty: Encouraging the discovery of previously unexplored structures

Mathematically, a reward function might take the form:

$$R(s) = w_1 \cdot \text{validity}(s) + w_2 \cdot \text{property}(s) + w_3 \cdot \text{novelty}(s)$$

where $w_i$ are weighting coefficients.

## Implementation Details

### Molecular Data Representation

We use a composition-based approach for representing molecules:

```julia
struct MoleculeData
    atoms::Vector{Symbol}  # Atom types (e.g., :C, :H, :O)
    bonds::Vector{Tuple{Int, Int, Int}}  # (atom1_idx, atom2_idx, bond_type)
end

struct MoleculeState <: GFlowNet.AbstractState
    data::MoleculeData
    complete::Bool  # Whether molecule construction is complete
end
```

### Action Types

```julia
struct AddAtomAction <: GFlowNet.AbstractAction
    atom_type::Symbol
    position::Tuple{Float64, Float64, Float64}
end

struct AddBondAction <: GFlowNet.AbstractAction
    atom1_idx::Int
    atom2_idx::Int
    bond_type::Int
end

struct TerminateMoleculeAction <: GFlowNet.AbstractAction end
```

### Feature Representation

For neural network input, we convert molecular states to feature vectors that capture:
1. Atom types and counts
2. Bond types and connectivity
3. Molecular graph features (e.g., cycles, functional groups)
4. Completion status

### Reward Calculation

Rewards are calculated based on molecular properties. In a real application, this might involve:
- Checking for chemical validity
- Computing drug-likeness scores (e.g., QED, SA score)
- Predicting bioactivity using machine learning models
- Evaluating synthetic accessibility

## Mathematical Derivation: Molecular Feature Embeddings

For effective learning, we need to convert molecular structures into meaningful feature vectors. A common approach is to use graph neural networks (GNNs) to compute atom embeddings through message passing:

1. Initialize atom features: $h_v^{(0)} = \text{featurize}(v)$ for each atom $v$

2. Message passing iterations:
   $$h_v^{(t+1)} = \text{UPDATE}\left(h_v^{(t)}, \text{AGGREGATE}\left(\{h_u^{(t)} : u \in \mathcal{N}(v)\}\right)\right)$$
   
   where $\mathcal{N}(v)$ represents the neighbors of atom $v$.

3. Molecular embedding: $h_G = \text{READOUT}(\{h_v^{(T)} : v \in V\})$

The embeddings capture both local (atom and bond) and global (molecular) properties, enabling effective policy learning.

## Intuitive Explanation

To understand molecular design with GFlowNets intuitively:

1. Imagine building a molecule piece by piece, like assembling a LEGO structure.
2. At each step, you decide whether to add an atom, add a bond, or declare the structure complete.
3. Different construction sequences can lead to the same final molecule.
4. The GFlowNet learns which construction paths are more likely to lead to molecules with desired properties.

The beauty of GFlowNets for molecular design is that they:
- Sample from the distribution of molecules with high rewards
- Provide diversity in the generated structures
- Learn the sequential construction process, which mirrors how chemists think about synthesis

## Relation to Other Molecular Generation Methods

GFlowNets offer several advantages compared to other molecular generation approaches:

1. **SMILES-based methods (e.g., RNNs, Transformers):**
   - GFlowNets operate directly on molecular graphs rather than string representations
   - They explicitly model the construction process, potentially yielding more chemically valid structures

2. **Variational Autoencoders (VAEs):**
   - GFlowNets don't require finding a continuous latent space representation
   - They directly model the probability of each construction step

3. **Reinforcement Learning:**
   - GFlowNets aim to sample proportionally to rewards rather than maximizing a single reward
   - This promotes diversity in the generated molecules

4. **Fragment-based methods:**
   - GFlowNets can incorporate both atom-level and fragment-level actions
   - The sequential nature matches how chemists think about building molecules

## Applications of Molecular GFlowNets

Molecular GFlowNets have potential applications in:

1. **Drug Discovery:** Generating candidate molecules with desired pharmacological properties

2. **Materials Science:** Designing molecules with specific physical or chemical characteristics

3. **Reaction Prediction:** Learning to generate likely products of chemical reactions

4. **Retrosynthesis Planning:** Suggesting plausible synthetic routes for target molecules

## Advanced Concepts: Multi-Objective Molecular Design

In practice, molecular design often involves multiple competing objectives. GFlowNets can be extended to handle this by:

1. Defining a composite reward function that balances different objectives:
   $$R(s) = \prod_{i=1}^k R_i(s)^{w_i}$$
   where $R_i(s)$ are individual reward components and $w_i$ are importance weights.

2. Using conditional GFlowNets to generate molecules conditioned on desired property values:
   $$P(\tau | c) \propto R(s_T, c)$$
   where $c$ represents the condition (e.g., desired properties).

This allows for targeted exploration of the molecular space based on specific design criteria.

## Conclusion

The Molecular Design example demonstrates the power of GFlowNets for generating complex, structured objects like molecules. By learning to navigate the vast chemical space efficiently, GFlowNets can discover diverse molecules with desired properties, potentially accelerating drug discovery and materials design processes.

The sequential construction approach aligns with how chemists think about building molecules, and the ability to sample proportionally to rewards enables balanced exploration of the chemical space, avoiding the pitfalls of methods that focus solely on optimization. 