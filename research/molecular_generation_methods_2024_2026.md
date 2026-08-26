# Deep Research: Molecular Generation Methods (2024-2026)
## Why Certain Methods Succeed — Architecture, Training, and Performance Analysis

*Research date: 2026-03-02*

---

## Table of Contents
1. [PMO Benchmark Leaderboard Analysis](#1-pmo-benchmark-leaderboard-analysis)
2. [SMILES-Based Generation Approaches](#2-smiles-based-generation-approaches)
3. [S3-GFN: Soft-Constrained Synthesizable GFlowNets](#3-s3-gfn-soft-constrained-synthesizable-gflownets)
4. [SynFlowNet, RGFN, and RxnFlow](#4-synflownet-rgfn-and-rxnflow)
5. [Key Bottlenecks in Molecular GFlowNets](#5-key-bottlenecks-in-molecular-gflownets)
6. [Synthesis and Recommendations](#6-synthesis-and-recommendations)

---

## 1. PMO Benchmark Leaderboard Analysis

### 1.1 Benchmark Overview

The PMO (Practical Molecular Optimization) benchmark evaluates molecular design algorithms on **23 single-objective optimization tasks** with a strict **10,000 oracle call budget**. The primary metric is **AUC Top-10**: area under the curve of the top-10 average property score versus oracle calls. This measures both optimization quality AND sample efficiency.

**Oracle tasks include**: QED, DRD2, GSK3beta, JNK3, and 19 GuacaMol oracles (albuterol_similarity, amlodipine_MPO, celecoxib_rediscovery, deco_hop, fexofenadine_MPO, isomers_C7H8N2O2, isomers_C9H10N2O2PF2Cl, median_1, median_2, mestranol_similarity, osimertinib_MPO, perindopril_MPO, ranolazine_MPO, scaffold_hop, sitagliptin_MPO, thiothixene_rediscovery, troglitazone_rediscovery, valsartan_SMARTS, zaleplon_MPO).

### 1.2 Current Leaderboard (Compiled from Multiple Sources)

**AUC Top-10 (Sum over 23 tasks, 10K oracle budget):**

| Rank | Method | AUC Top-10 | Diversity | Type | Year |
|------|--------|------------|-----------|------|------|
| 1 | **SEISMO** | ~21.17* | N/A | LLM agent (inference-time) | 2025 |
| 2 | **MolLIBRA-L** | Best on 14/22* | N/A | GA + GP surrogates + CLAMP | 2026 |
| 3 | **ExLLM** | SOTA (PMO)* | N/A | LLM-as-optimizer | 2025 |
| 4 | **Chemma-2B** | **17.534** | N/A | LLM fine-tuning + GA | 2024 |
| 5 | **Chemlactica-1.3B** | **17.284** | N/A | LLM fine-tuning + GA | 2024 |
| 6 | **Genetic GFN** | **16.213** | 0.432 | SMILES GFN + GA | 2024 |
| 7 | **Mol GA** | **15.686** | N/A | SELFIES GA | 2023 |
| 8 | **SMILES REINVENT** | **15.185** | N/A | SMILES RL | 2022 |
| 9 | **Augmented Memory** | **15.002** | 0.801 | SMILES RL + replay | 2024 |
| 10 | **GEGL** | **14.354** | N/A | Expert-guided GA | 2022 |
| 11 | **GP BO** | **14.264** | N/A | Bayesian optimization | 2022 |
| 12 | **REINVENT** | **14.196** | N/A | SMILES RL | 2022 |
| 13 | **Fragment GFN** | **~10.957** | 0.816 | Fragment GFlowNet | 2022 |
| 14 | **Fragment GFN-AL** | **~9.917** | N/A | Fragment GFN + active learning | 2022 |

*Note: SEISMO, MolLIBRA, and ExLLM use different budgets or evaluation protocols that make direct AUC comparison nuanced. SEISMO achieves near-maximal scores within 50 oracle calls but the AUC numbers are computed differently (Top-1 AUC rather than Top-10 in some tables).*

### 1.3 Task-Specific Performance Patterns

**Where different methods excel:**

| Task Category | Best Method Type | Why |
|---------------|-----------------|-----|
| **Binding affinity** (DRD2, GSK3beta, JNK3) | Genetic GFN, Chemlactica | Dense reward landscape, SMILES mutations effective |
| **Rediscovery tasks** (celecoxib, thiothixene, troglitazone) | LLM-based | LLMs have memorized molecular structures |
| **Similarity-constrained** (albuterol, mestranol) | REINVENT, AugMem | RL with prior anchoring stays near targets |
| **Multi-property MPO** (sitagliptin, fexofenadine) | Chemlactica, Genetic GFN | Need to balance multiple objectives |
| **Isomer generation** | All methods struggle | Highly constrained combinatorial problem |
| **Scaffold hop** | Genetic GFN, Chemlactica | Requires structural creativity |

**Key observation**: No single method dominates ALL tasks. LLM-based methods excel at rediscovery (memorization advantage) and MPO tasks. GA+GFN hybrids are strong on binding affinity. Fragment-based GFlowNets consistently underperform due to action space limitations.

### 1.4 The Diversity-Performance Trade-off

This is the central tension in molecular optimization:

- **Fragment GFN**: High diversity (0.816) but low AUC (10.957) -- explores too broadly
- **Genetic GFN at beta=50**: Low diversity (0.432) but high AUC (16.213) -- exploits efficiently
- **Genetic GFN at beta=1**: High diversity (0.812) but low AUC (11.083) -- matches Fragment GFN
- **Augmented Memory**: Moderate diversity (0.801) with decent AUC (15.002) -- best balance

The inverse temperature beta in GFlowNets directly controls this trade-off. Practical drug discovery needs BOTH high scores AND diversity (to provide backup candidates), making this trade-off critical.

---

## 2. SMILES-Based Generation Approaches

### 2.1 REINVENT4 — Architecture and Training

**Architecture:**
- **Generator types**: De novo (Reinvent), Scaffold decoration (Libinvent), Linker design (Linkinvent), Molecule-to-molecule (Mol2Mol)
- **Neural network**: Autoregressive RNN (LSTM/GRU) or Transformer over SMILES tokens
- **Tokenization**: Character-level with multi-character exceptions (e.g., "[nH]" as single token)
- **Prior model**: Pre-trained on large SMILES corpus (ZINC, ChEMBL) using teacher forcing
- **Agent model**: Initialized from prior, fine-tuned via RL

**Training — The DAP (Difference of Augmented Posterior) Loss:**
```
L(T) = [log P_aug(T) - log P_agent(T)]^2

where: log P_aug(T) = log P_prior(T) + sigma * S(T)
```
- `P_prior(T)`: Prior model likelihood (regularization anchor)
- `P_agent(T)`: Current agent likelihood
- `S(T)`: Scoring function (multi-component)
- `sigma`: Controls reward vs. regularization balance

**Key mechanisms:**
1. **Experience Replay (Inception)**: Buffer of top-100 scoring SMILES; 10 sampled per batch
2. **Staged Learning**: Sequential RL stages with increasing complexity (cheap filters -> expensive docking)
3. **Diversity Filters**: Bemis-Murcko scaffold buckets; zero-score when bucket fills
4. **Multi-component Scoring**: Weighted arithmetic/geometric mean of QED, docking, SA, QSAR, etc.

**Why REINVENT works**: The prior model provides a strong inductive bias toward valid chemistry. The DAP loss prevents the agent from straying too far from the prior while still optimizing the reward. Experience replay maintains memory of good solutions.

### 2.2 Chemlactica/Chemma — LLM-Based Molecular Optimization

**Models:**
| Model | Base | Parameters | Training Data | Training Cost |
|-------|------|-----------|---------------|---------------|
| Chemlactica-125M | Galactica | 125M | 2.1B tokens | 306 A100 hrs |
| Chemlactica-1.3B | Galactica | 1.3B | 40B tokens | 288 H100 hrs |
| Chemma-2B | Gemma | 2B | 39B tokens | 488 H100 hrs |

**Training corpus**: 110 million molecules from PubChem with computed properties (SAS, QED, MW, TPSA, CLogP, H-bond donors/acceptors, ring counts). 4 billion similar molecule pairs sampled from 200 billion pairs using ECFC4 fingerprints. JSONL format with XML-style tags for properties and SMILES.

**PMO Optimization Algorithm (3 components):**

1. **LLM-enhanced genetic algorithm**: Generates similar molecules via prompts with `[SIMILAR]<smiles> <similarity>[/SIMILAR]...[START_SMILES]` tags
2. **Explicit oracle modeling**: Fine-tunes the LLM on high-performing molecules using property scores when improvement stalls for K iterations
3. **Population-based approach**: Maintains top-P molecules, iteratively generates N new candidates

**Generation details:**
- Chain-of-thought approach (omitting START_SMILES initially to let the model reason)
- Dynamic temperature scheduling: starts at 1.0, linearly increases to 1.5
- Repetition penalty and token suppression for diversity

**Per-task scores (selected):**
- JNK3: 0.891 +/- 0.032
- Median1: 0.382 +/- 0.022
- Scaffold_hop: 0.669 +/- 0.110
- Sitagliptin_MPO: 0.613 +/- 0.018
- **Sum-23: 17.534 +/- 0.214** (Chemma-2B, best)

**Why Chemlactica succeeds:**
1. **Massive pre-training**: 110M molecules with computed properties create deep chemical knowledge
2. **Structured prompts**: XML-tagged format enables property conditioning and similarity-guided generation
3. **Adaptive fine-tuning**: Oracle feedback is directly incorporated through fine-tuning cycles
4. **Scale matters**: Performance improved significantly with larger models (125M -> 1.3B -> 2B)
5. **Genetic diversity**: Population-based approach prevents mode collapse

### 2.3 Augmented Memory (AugMem)

**Core Insight**: SMILES strings are non-injective -- the same molecule has MANY valid SMILES representations (via different atom orderings/DFS traversals). This means a single oracle call can be reused for multiple gradient updates by augmenting the SMILES representation.

**Architecture**: Built on REINVENT (LSTM-based autoregressive SMILES generator)

**Training Algorithm:**
1. Generate SMILES batch from current policy
2. Compute rewards using oracle
3. Update replay buffer (keep top-k molecules)
4. **Augment entire replay buffer** with randomized SMILES (different atom orderings via RDKit DFS)
5. Update agent N times using augmented data (multiple updates per oracle call!)

**Loss Function**: Augmented Likelihood = Prior Likelihood + sigma * Score. Minimize squared difference between augmented and agent likelihoods.

**Selective Memory Purge**: Removes Bemis-Murcko scaffolds from replay buffer when penalized by diversity filter. Prevents mode collapse while maintaining accelerated learning.

**Performance:**
- AUC Top-10 Sum-23: **15.002** (vs. REINVENT's 14.016)
- Outperforms REINVENT on 14/23 tasks (95% confidence)
- DRD2 case: 2000+ more molecules with better docking scores under 9,600 oracle calls
- IntDiv1 diversity: 0.801

**Why AugMem works**: The key insight is that experience replay + SMILES augmentation = extreme sample efficiency. One oracle call produces information that can update the model many times through different SMILES representations of the same molecule.

### 2.4 Why SMILES-Level Generation is So Effective

**Five fundamental reasons:**

1. **Action space compactness**: A SMILES string of length L has a vocabulary of ~50-70 tokens. The total action space per step is O(|V|), compared to O(|fragments| * |attachment_points|) for fragment-based methods.

2. **Genetic operations are powerful in string space**: Crossover and mutation on SMILES strings can produce molecules with LARGE structural differences even when the edit distance is small. "Offspring can have large distance from parents in 1D string space, even if molecular distances are small" (Genetic GFN paper). This enables escaping local optima.

3. **Pre-training is natural**: Language models (RNN, Transformer, LLM) can be pre-trained on millions of SMILES strings, learning the grammar of valid chemistry. This provides a strong prior that guides optimization.

4. **Credit assignment is simpler**: In a SMILES generator, each token is generated left-to-right with direct gradient flow. In a graph/fragment generator, credit must be assigned across a more complex DAG structure.

5. **Non-injectivity enables data augmentation**: The same molecule has many valid SMILES representations, enabling experience replay amplification (the AugMem insight).

### 2.5 SEISMO — The New Frontier (LLM Agent, 2025)

**Paradigm shift**: Pure inference-time optimization using an LLM agent.

**Architecture:**
- **LLM backbone**: Claude Opus (claude-opus-4-5-20251101-v1:0)
- **Implementation**: LangGraph stateful workflow with 5 nodes (generation, parsing, validation, prediction, finalization)
- **No training/fine-tuning**: Purely in-context learning from optimization trajectory

**How it works:**
1. Start with task description in natural language
2. At each step, propose molecule based on FULL history of prior trials (molecules, scores, feedback)
3. Receive oracle score + optional explanatory feedback
4. Repeat -- the conversation history IS the learning signal

**Performance (50 oracle calls):**
- Top-1 AUC: **19.85** (vs. REINVENT 7.06, Graph-GA 7.04)
- Achieves scores within 10% of task maximum on **16/23 tasks** with just 50 calls
- Best on ALL 23 tasks at 50-call budget
- Remains best on 20/23 tasks even at 10,000-call budget

**Why SEISMO is revolutionary:**
1. **Ultimate sample efficiency**: Near-maximal scores in 50 calls vs. 10,000 for traditional methods
2. **Leverages pre-trained chemical knowledge**: Decades of medicinal chemistry encoded in LLM weights
3. **Trajectory reasoning**: Each proposal conditions on full optimization history
4. **Explanatory feedback helps**: Richer feedback (subscores, explanations) consistently improves performance
5. **No training infrastructure**: No GPUs needed for model training, just API calls

**Limitations**: Relies on frontier LLM (expensive per call), diversity not explicitly measured, requires structured feedback for best results.

---

## 3. S3-GFN: Soft-Constrained Synthesizable GFlowNets

### 3.1 Core Innovation

S3-GFN bridges the gap between SMILES-level generation (high optimization power) and reaction-based generation (synthesizability guarantees) through **soft constraints** rather than hard MDP constraints.

**Key idea**: Instead of restricting the action space to only valid reactions (like SynFlowNet/RGFN), S3-GFN generates freely in SMILES space but uses contrastive learning to push probability mass AWAY from unsynthesizable regions.

### 3.2 Architecture

**Base model**: GP-MolFormer (pre-trained SMILES language model trained on large-scale molecular datasets)

**MDP design**: Sequence-based, where states are partial SMILES token sequences (prefixes). Actions append tokens from a fixed vocabulary. Initial state = empty string, terminal state = complete valid SMILES.

**Training uses Relative Trajectory Balance (RTB) loss** -- a GFlowNet objective that operates on relative rewards between trajectories.

### 3.3 Contrastive Learning Signal

**Two replay buffers:**
- **Positive buffer (B+)**: Molecules where a valid synthetic pathway exists (verified via retrosynthesis planning with predefined reaction templates and building blocks)
- **Negative buffer (B-)**: Molecules where no valid synthesis pathway exists

**Auxiliary contrastive loss:**
```
L_aux = -sum_{tau in B+} log[ exp(s(tau)) / (exp(s(tau)) + sum_{tau' in B-} exp(s(tau'))) ]
```
where `s(tau) = log P_F(tau)` (sequence-level forward policy log-probability).

This loss explicitly **suppresses probability mass on unsynthesizable molecules** while maintaining likelihoods on synthesizable ones.

**Negative sample generation**: Genetic mutations applied to positive samples using graph-based operations to generate "hard negatives" that are structurally similar but unsynthesizable.

### 3.4 Training Procedure

**Two-phase training per iteration:**

**Phase 1 — On-Policy (positive samples only):**
```
L_RTB+ = (1/|B_t+|) * sum_{tau in B_t+} L_RTB(tau)
```

**Phase 2 — Replay (combined objective):**
```
L_replay = (1/|B+|) * sum_{tau in B+} L_RTB(tau) + alpha * L_aux
```
with alpha = 10^-3.

**Synthesizability determination**: AiZynthFinder retrosynthesis planning using USPTO reaction templates. A molecule qualifies as "positive" if a valid synthetic pathway exists.

### 3.5 Performance Numbers

**sEH Target (proxy-based):**
| Method | Top-100 sEH | Positive Ratio | Diversity |
|--------|-------------|----------------|-----------|
| SynFlowNet | 0.989 +/- 0.005 | 1.000 | High |
| S3-GFN | **1.043 +/- 0.001** | 0.945 +/- 0.009 | High |
| RTB + RS | 1.040 +/- 0.001 | 0.987 +/- 0.003 | High |

**Docking Tasks (LIT-PCBA, Vina scores in kcal/mol):**
| Target | S3-GFN Vina | AiZynthFinder Rate |
|--------|-------------|-------------------|
| ADRB2 | -12.32 +/- 0.026 | **100.0%** |
| ALDH1 | -11.63 +/- 0.015 | **97.0%** |
| ESR_ago | -11.41 +/- 0.035 | **96.67%** |
| ESR_antago | -11.24 +/- 0.011 | **96.33%** |
| FEN1 | -7.70 +/- 0.011 | **99.0%** |

S3-GFN outperformed all reaction-based methods (SynFlowNet, RGFN, RxnFlow) on average Vina scores across all five targets while achieving >95% synthesizability.

### 3.6 Why S3-GFN Succeeds

1. **Best of both worlds**: SMILES-level generation (powerful optimization) + synthesizability awareness (contrastive learning)
2. **Rich pre-trained prior**: GP-MolFormer initialization provides strong inductive bias toward valid, synthesizable chemistry
3. **Soft constraints > Hard constraints**: Hard MDP constraints (SynFlowNet, RGFN) limit chemical space exploration. Soft constraints allow the model to explore broadly but learn to avoid unsynthesizable regions.
4. **Contrastive hard negatives**: Mutated negatives that are structurally similar to positives provide the strongest learning signal
5. **No action space limitation**: Unlike reaction-based methods constrained to 17-105 templates, S3-GFN can generate ANY molecule that SMILES can represent

### 3.7 Limitations
- Synthesizability is not **guaranteed** (95-100%, not 100%)
- Requires retrosynthesis oracle (AiZynthFinder) during training
- Computational cost of retrosynthesis verification per molecule
- Positive ratio can drop below 95% on some targets

---

## 4. SynFlowNet, RGFN, and RxnFlow

### 4.1 SynFlowNet (ICLR 2025 Spotlight)

**Architecture:**
- **Policy network**: Graph Transformer with shared embedding passed to 8 separate MLPs for different action types
- **Building blocks**: Up to 221,181 Enamine purchasable compounds (experiments use 10K subsets)
- **Reaction templates**: 105 templates (13 unimolecular + 92 bimolecular)
- **Building block encoding**: Fixed Morgan fingerprints (2048-bit) rather than learnable embeddings (key for scaling)

**Action space (5 forward, 3 backward):**
- Forward: Stop, AddFirstReactant, ReactUni, ReactBi, AddReactant
- Backward: BckReactUni, BckReactBi, BckRemoveFirstReactant

**Synthesizability guarantee**: Hard MDP constraint -- all molecules MUST be constructible through documented reaction sequences from purchasable building blocks. Template masking ensures only compatible reactions/reactants can be selected.

**Backward policy challenge**: Not every backward action leads to a state reachable from forward trajectories. Solutions:
- Maximum Likelihood training: 99.3% solved (train), 32.3% (test)
- REINFORCE with entropy: 100% solved (train), 44.3% (test)
- Uniform baseline: only 11% success

**Performance:**
- Comparable binding scores to REINVENT on sEH
- Superior diversity vs. entropy-regularized RL
- AiZynthFinder success: **62%** (L=3 trajectories) vs. **0%** for FragGFN
- State space ~10 orders of magnitude smaller than fragment-based

**Key limitation**: The backward policy problem is fundamental -- reaction-based MDPs cannot guarantee that backward-constructed trajectories can return to the initial state.

### 4.2 RGFN (NeurIPS 2024)

**Architecture:**
- **Backbone**: Graph Transformer (5 layers, 4 attention heads, 64 hidden dims)
- **Separate MLPs**: MLPM (building blocks), MLPR (reactions), MLPB (backward policy)

**Reaction templates**: **17 high-yield reactions** encoded as 132 SMARTS templates (amide bond formation, Suzuki-Miyaura, Buchwald-Hartwig, Sonogashira, SuFEx, cycloadditions, etc.)

**Building blocks**: **350 affordable building blocks** (mean cost $22.52/gram)

**Synthesizability guarantee**: Design constraint -- operates exclusively within predefined chemical reaction space. All top modes manually validated by expert chemists.

**Performance:**
| Metric | RGFN | FGFN | GraphGA |
|--------|------|------|---------|
| SA Score (lower=better) | 2.79-3.24 | 2.94-3.55 | N/A |
| AiZynthFinder rate | 0.56-0.87 | 0.02-0.76 | N/A |
| Senolytic modes found | Yes | **Zero** | Yes |

**Limitations:**
1. Only 17 reactions + 350 building blocks = tiny fraction of drug-like space
2. Generates linear, flat-shaped molecules (lacks cyclization reactions)
3. No reaction conditions, catalysts, or protection group strategies
4. 400,000 oracle calls needed (vs. 10K for PMO benchmark methods)
5. Docking scores correlate strongly with molecular weight (known issue)

### 4.3 RxnFlow (ICLR 2025)

**Key innovation**: Action space subsampling to handle **1.2 million building blocks** with **71 reaction templates**.

**Architecture innovations:**
1. **Non-hierarchical MDP**: Jointly selects (reaction, building_block) pairs rather than sequential selection
2. **Action embedding**: Neural network phi_block embeds building blocks into continuous space (generalizes to unseen blocks)
3. **Subsampling policy**: Samples subset of action space with importance weighting, reducing complexity from O(|B|*|R2|) to O(|B*|*|R2|)
4. **Fixed backward policy**: Non-parameterized to preserve molecule isomorphism invariance

**Performance (LIT-PCBA):**
| Metric | RxnFlow | SynFlowNet | RGFN |
|--------|---------|------------|------|
| Building Blocks | **1.2M** | 6K | 350 |
| Avg Vina (ADRB2) | **-11.45** | -10.85 | -9.84 |
| Hit Ratio (ADRB2) | **60.25%** | 52.75% | 46.75% |
| Synth Steps | **2.42** | 2.64 | 2.88 |

**CrossDocked2020 zero-shot**: Avg Vina -8.85, QED 0.67, Synthesizability 34.8%, Diversity 0.81.

**Key advance**: Demonstrated that increasing building block library size improves BOTH optimization power AND diversity (more unique Bemis-Murcko scaffolds).

### 4.4 Reaction-Based Methods: Summary Comparison

| Feature | SynFlowNet | RGFN | RxnFlow | S3-GFN |
|---------|-----------|------|---------|--------|
| **Approach** | Hard MDP constraint | Hard MDP constraint | Hard MDP constraint | Soft constraint |
| **Representation** | Graph | Graph | Graph | SMILES |
| **Building blocks** | 10K-221K | 350 | **1.2M** | N/A (free gen) |
| **Reactions** | 105 | 17 | 71 | N/A |
| **Synth guarantee** | By construction | By construction | By construction | ~95% (learned) |
| **Optimization power** | Moderate | Low-Moderate | High | **Highest** |
| **Diversity** | Good | Moderate | Good | Good |
| **Backward policy** | Learned (problematic) | Learned | Fixed | N/A (SMILES) |
| **Scalability** | Moderate | Low | **High** | High |

---

## 5. Key Bottlenecks in Molecular GFlowNets

### 5.1 Why Fragment GFN Scores Only ~10 on PMO

**Root causes (in order of impact):**

1. **Action space exploration problem**: Fragment-based generation uses a vocabulary of molecular fragments (e.g., 50-100 BRICS fragments). Each step attaches a fragment to an attachment point. The state space is ~10 orders of magnitude larger than reaction-based MDPs, but "graph-based generating policies search broader chemical space, including undesired ones" -- they waste oracle calls on bad regions.

2. **No pre-trained prior**: Unlike SMILES-based methods (REINVENT, Chemlactica) that start from a pre-trained model encoding valid chemistry, fragment GFlowNets typically train from scratch. This means early samples are mostly invalid/low-quality, wasting precious oracle budget.

3. **Credit assignment over complex DAG**: Fragment attachment creates a tree/DAG structure where credit must propagate through multiple branching decisions. Trajectory balance helps but is less efficient than direct autoregressive gradient flow in SMILES models.

4. **Diversity bias over exploitation**: GFlowNets are designed to sample proportionally to reward R(x), producing diverse samples. But PMO measures TOP-10 performance, which rewards exploitation. At the default temperature, GFlowNets over-explore.

5. **Limited genetic search**: Fragment-based mutations (swap a fragment, add/remove) produce SMALL structural changes. SMILES-based genetic operations can produce LARGE structural jumps with small edit distances.

**Evidence**: Genetic GFN achieves 16.213 (SMILES) vs. Fragment GFN 10.957 (fragments) on the SAME benchmark with the SAME GFlowNet training objective. The only difference is the action space.

### 5.2 The Credit Assignment Problem

**The fundamental issue**: In a trajectory of length T that builds a molecule token-by-token (SMILES) or fragment-by-fragment, the reward is only observed at the terminal state. Credit must be assigned to each of the T intermediate decisions.

**How different objectives handle this:**

| Objective | Credit Assignment | Efficiency | Molecular GFN Impact |
|-----------|------------------|------------|---------------------|
| **Flow Matching (FM)** | Per-edge (local) | O(1) per edge | Poor for long trajectories -- "prone to inefficient credit propagation" |
| **Detailed Balance (DB)** | Per-transition (local) | O(1) per transition | Better than FM, still local |
| **Trajectory Balance (TB)** | Per-trajectory (global) | O(T) per trajectory | **Much better** -- "benefits for convergence, diversity, robustness to long sequences" |
| **Relative TB (RTB)** | Per-trajectory relative | O(T) | Best -- relative rewards between trajectories |

**For molecular generation:**
- Fragment GFN with TB: ~10 on PMO (25-50 step trajectories)
- SMILES GFN with TB: ~16 on PMO (20-40 token trajectories)
- The trajectory lengths are comparable, but SMILES tokens have more direct semantic meaning per step

**The atom-based challenge**: Using individual atoms as actions would give the most expressive space, but trajectories become 20-80+ steps. "The vastness of this accessible state space makes training atom-based GFlowNets susceptible to collapse." No one has made atom-based GFlowNets work competitively.

### 5.3 How Action Space Design Affects Performance

**The fundamental trade-off:**

```
Fine-grained actions (atoms) -----> Coarse-grained actions (fragments/reactions)
   More expressive                        More constrained
   Longer trajectories                    Shorter trajectories
   Harder credit assignment               Easier credit assignment
   Larger state space                     Smaller state space
   More exploration needed                Less exploration needed
   Lower sample efficiency                Higher sample efficiency per step
   BUT lower quality per oracle call      BUT limited chemical space
```

**SMILES tokens sit in a sweet spot**: Each token encodes meaningful chemical information (atoms, bonds, branches, rings) in a compact vocabulary of ~50-70 symbols. Trajectories are 20-40 steps (manageable for TB). The autoregressive left-to-right generation provides natural credit flow.

**The Genetic GFN insight**: Instead of relying solely on GFlowNet training for exploration, use genetic algorithms (crossover/mutation on SMILES strings) to propose diverse candidates, then train the GFlowNet on the high-reward ones. This solves the exploration problem without sacrificing GFlowNet's reward-proportional sampling.

**Quantified impact of genetic search** (from ablation):
- GFlowNet only: AUC ~14.7
- GFlowNet + Genetic search: AUC **16.2** (+1.5)
- KL regularization: +0.112 additional
- Combined: **16.213** (highest on 14/23 oracles)

### 5.4 The Sample Efficiency Bottleneck

**The core problem**: With 10K oracle calls, a method must learn to optimize across 23 diverse tasks. Each oracle call costs compute time (especially docking). Methods that waste calls on poor molecules fall behind.

**Sample efficiency ranking:**
1. **SEISMO**: 50 calls to near-optimal (but uses frontier LLM API)
2. **Chemlactica**: Fine-tunes on feedback, reuses oracle information through population maintenance
3. **Genetic GFN**: Genetic operations amplify each oracle call, rank-based reweighting prioritizes best
4. **Augmented Memory**: SMILES augmentation enables N updates per oracle call
5. **REINVENT**: Experience replay helps, but single-SMILES representation limits reuse
6. **Fragment GFN**: Explores too broadly, many oracle calls wasted on low-quality molecules

---

## 6. Synthesis and Recommendations

### 6.1 The State of the Art (2026)

The field has converged on several key insights:

1. **SMILES-based generation dominates optimization benchmarks** because of compact action spaces, strong pre-trainable priors, and natural genetic operations.

2. **LLMs are game-changers**: Chemlactica showed that pre-trained chemical LLMs with population-based optimization are extremely effective. SEISMO showed that frontier LLMs can optimize in-context with 50 calls.

3. **Synthesizability remains the Achilles' heel of SMILES methods**: High PMO scores don't help if molecules can't be made. S3-GFN's soft-constraint approach is the most promising bridge.

4. **Reaction-based methods guarantee synthesizability but sacrifice optimization power**: RxnFlow partially closes this gap with 1.2M building blocks, but still cannot match SMILES-level optimization.

5. **The diversity-optimization trade-off is controllable**: GFlowNets with temperature/beta control can navigate this Pareto frontier explicitly.

### 6.2 Architecture Recommendations for GFlowNet Development

**For maximum PMO performance:**
- Use SMILES-based generation with pre-trained prior (GP-MolFormer or similar)
- Combine with genetic search (Genetic GFN approach)
- Apply trajectory balance with rank-based reweighting
- Use KL regularization to anchor to prior

**For synthesizable generation:**
- S3-GFN approach (SMILES + contrastive soft constraints) is strongest
- If hard guarantees needed, RxnFlow with 1.2M building blocks
- Consider hybrid: generate SMILES candidates, filter by retrosynthesis, fine-tune

**For diversity-aware drug discovery:**
- GFlowNet with moderate temperature (beta=10-30)
- Multi-objective Pareto optimization (MOGFN-PC)
- Augmented Memory's selective memory purge for controlled diversity

### 6.3 Key Open Problems

1. **Closing the gap between SMILES optimization and synthesizability**: S3-GFN achieves 95%+, but 100% would require either hard constraints or perfect retrosynthesis models
2. **Scaling reaction-based methods**: RxnFlow's 1.2M blocks is a start, but real combinatorial chemistry has billions of possibilities
3. **3D-aware generation**: Current methods optimize 2D molecular graphs; 3D conformation matters for binding
4. **Multi-step synthesis planning**: Current methods generate 2-4 step routes; real synthesis often needs 5-15 steps
5. **LLM cost**: SEISMO uses frontier LLMs that cost ~$15-75/molecule for 50 calls; not practical for high-throughput screening

---

## Sources

### PMO Benchmark
- [PMO Benchmark Paper (NeurIPS 2022)](https://arxiv.org/abs/2206.12411)
- [mol_opt GitHub Repository](https://github.com/wenhao-gao/mol_opt)

### SMILES-Based Methods
- [REINVENT4: Modern AI-Driven Generative Molecule Design](https://link.springer.com/article/10.1186/s13321-024-00812-5)
- [REINVENT4 GitHub](https://github.com/MolecularAI/REINVENT4)
- [Small Molecule Optimization with Large Language Models (Chemlactica)](https://arxiv.org/html/2407.18897v1)
- [Augmented Memory: Sample-Efficient Generative Molecular Design](https://pmc.ncbi.nlm.nih.gov/articles/PMC11200228/)
- [SEISMO: Trajectory-Aware LLM Agent](https://arxiv.org/html/2602.00663)
- [ExLLM: Experience-Enhanced LLM Optimization](https://arxiv.org/abs/2502.12845)
- [MolLIBRA: Genetic Optimization with Multi-Fingerprint Surrogates](https://arxiv.org/abs/2602.07002)

### GFlowNet Methods
- [Genetic-guided GFlowNets (NeurIPS 2024)](https://arxiv.org/html/2402.05961v4)
- [S3-GFN: Soft-Constrained Synthesizable GFlowNets](https://arxiv.org/html/2602.04119)
- [SynFlowNet (ICLR 2025)](https://arxiv.org/html/2405.01155v2)
- [RGFN (NeurIPS 2024)](https://arxiv.org/html/2406.08506v2)
- [RxnFlow (ICLR 2025)](https://arxiv.org/html/2410.04542v1)
- [Trajectory Balance: Improved Credit Assignment](https://arxiv.org/abs/2201.13259)

### Genetic Algorithms
- [Mol GA GitHub](https://github.com/AustinT/mol_ga)
- [Genetic Algorithms are Strong Baselines](https://arxiv.org/pdf/2310.09267)
