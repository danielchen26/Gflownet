# Direction Portfolio Sprint Closeout — Structural Core Screening

**Authoritative user time:** Saturday, 2026-06-20 01:45 EDT  
**Workspace:** `/Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint`  
**Plan v2:** `/Users/tianchichen/Documents/GitHub/Gflownet/sessions/260314-misty-eddy/plans/remaining_direction_portfolio_sprint_plan_v2_reaudited_2026-06-20.md`  
**Runner:** `test/smiles_gflownet/run_direction_portfolio_sprint.jl`  
**Tier-0 bundle:** `checkpoints/direction_portfolio/portfolio_tier0_results.jls`  
**Tier-1 bundle:** `checkpoints/direction_portfolio/portfolio_tier1_results.jls`  
**Schema:** `direction_portfolio_v2_reaudited_core_screening`

---

## 1. Objective

The user requested a broader first-principles search for a genuinely new algorithmic fusion direction, analogous in spirit to:

```text
Genetic Algorithm + GFlowNet
```

but not another selector/resampling/exchange variant.

This sprint tested remaining structural primitives at synthetic level only:

```text
FEBB  = Free-Energy Branch-and-Bound
EFR   = Exact Fragment Recombination
SGC   = Spin-Glass Cluster Moves
RG    = Renormalization Coarse-to-Fine
AIA   = Active-Information Acquisition
MCTS  = Lookahead Planning
NCI   = Noisy-Channel Inverse Decoding
```

Important:

```text
These were Stage-0 core primitive tests, not validated GFlowNet algorithms.
No PMO micro was run.
No SOTA claim is allowed.
```

---

## 2. Commands

### Compile

```bash
cd /Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint
JULIA_CONDAPKG_BACKEND=Null \
JULIA_PYTHONCALL_EXE=$PWD/.CondaPkg/.pixi/envs/default/bin/python \
PORT_MODE=compile \
julia --project=. test/smiles_gflownet/run_direction_portfolio_sprint.jl
```

### Tier-0

```bash
cd /Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint
mkdir -p checkpoints/direction_portfolio
JULIA_CONDAPKG_BACKEND=Null \
JULIA_PYTHONCALL_EXE=$PWD/.CondaPkg/.pixi/envs/default/bin/python \
PORT_MODE=tier0 \
PORT_BUDGET=128 \
PORT_SEEDS=17,23 \
julia --project=. test/smiles_gflownet/run_direction_portfolio_sprint.jl \
2>&1 | tee checkpoints/direction_portfolio/run_tier0.log
```

### Tier-1

```bash
cd /Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint
mkdir -p checkpoints/direction_portfolio
JULIA_CONDAPKG_BACKEND=Null \
JULIA_PYTHONCALL_EXE=$PWD/.CondaPkg/.pixi/envs/default/bin/python \
PORT_MODE=tier1 \
PORT_BUDGETS=256,1024 \
PORT_SEEDS=17,23,31 \
julia --project=. test/smiles_gflownet/run_direction_portfolio_sprint.jl \
2>&1 | tee checkpoints/direction_portfolio/run_tier1.log
```

### Regression tests

```bash
JULIA_CONDAPKG_BACKEND=Null \
JULIA_PYTHONCALL_EXE=$PWD/.CondaPkg/.pixi/envs/default/bin/python \
julia --project=. test/smiles_gflownet/test_option_flow_poc.jl

JULIA_CONDAPKG_BACKEND=Null \
JULIA_PYTHONCALL_EXE=$PWD/.CondaPkg/.pixi/envs/default/bin/python \
julia --project=. test/smiles_gflownet/test_option_flow_real_catalog.jl
```

---

## 3. Engineering fixes during execution

| Issue | Symptom | Fix |
|---|---|---|
| FEBB empty prefix bound | `reducing over an empty collection` | explicit `k == 0` and terminal remaining handling |
| SGC/RG/GA/SA duplicate loops | repeated candidates caused `calls` not to increase | added attempt caps and random unseen fallback |
| AIA/MCTS duplicate loops | acquisition/MCTS could repeatedly select same candidate | added attempt caps and fallback |
| MCTS variable shadowing | local `values` shadowed `Base.values()` | renamed to `value_sums` and used `Base.values` |
| Beam terminal frontier | infinite loop when frontier terminal and `newfront` empty | changed to `break` |
| Ranking NaN | `maximum(...; init=NaN)` propagated NaN | added `max_or_nan` helper |

After these fixes, Tier-0 and Tier-1 completed and saved valid bundles.

---

## 4. Tier-0 result — all seven screened

Tier-0 used:

```text
budget = 128
seeds = 17, 23
```

Summary:

| Candidate | Tier-0 label | Intended AUC | Ablation AUC | Best baseline AUC | Key verdict |
|---|---|---:|---:|---:|---|
| EFR | `SYNTHETIC_ONLY_INTERESTING` | 0.679575 | 0.523475 | 0.523475 | strong if grammar complete; anti-bias failed |
| AIA | `STOP_OR_REDESIGN` | 0.741638 | 0.689057 | 0.742323 | did not beat greedy surrogate/best base |
| FEBB | `SYNTHETIC_ONLY_INTERESTING` | 0.996094 | 0.968786 | 0.968786 | strong pruning signal; anti-bias failed |
| RG | `SYNTHETIC_ONLY_INTERESTING` | 0.925108 | 0.799276 | 0.883391 | coarse-to-fine helps but anti-bias failed |
| MCTS | `STOP_OR_REDESIGN` | 0.870341 | 0.644448 | 0.916957 | beam search stronger |
| SGC | `REBRANDED_BASELINE` | 0.512323 | 0.519519 | 0.609750 | cluster moves lost local/GA |
| NCI | `REBRANDED_BASELINE` | 0.973675 | 0.993150 | 0.993150 | forward rerank beat inverse decoder |

Tier-0 survivors for Tier-1:

```text
EFR, FEBB, RG
```

Notably:

- `SGC` failed its own cluster ablation.
- `NCI` failed its own inverse-vs-forward-rerank ablation.
- `MCTS` was dominated by beam search.
- `AIA` was close but did not beat the best surrogate baseline.

---

## 5. Tier-1 result — survivors only

Tier-1 used:

```text
budgets = 256, 1024
seeds = 17, 23, 31
survivors = EFR, FEBB, RG
```

Final Tier-1 ranking:

| Candidate | Tier-1 label | Structural score | Intended AUC | Ablation AUC | Best baseline AUC | Anti-bias AUC | Anti-bias best baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| FEBB | `SYNTHETIC_ONLY_INTERESTING` | 0.750000 | **0.998779** | 0.970848 | 0.971411 | 0.649073 | 0.919027 |
| EFR | `SYNTHETIC_ONLY_INTERESTING` | 0.597913 | 0.686484 | 0.638575 | 0.668974 | 0.474035 | 0.966107 |
| RG | `STOP_OR_REDESIGN` | -0.361956 | 0.968532 | 0.896448 | **0.970454** | 0.627937 | 0.873216 |

No candidate earned:

```text
PROMOTE_TO_PMO_PLAN
```

Therefore:

```text
No PMO micro should be run from this sprint.
```

---

## 6. Candidate-specific interpretation

### 6.1 FEBB — Free-Energy Branch-and-Bound

Physical object:

```text
A partial state defines a basin; an admissible upper envelope U(B) prunes whole regions whose best possible reward cannot beat the incumbent.
```

Theoretical support:

```text
If U(B) >= max_{x in B} R(x), pruning when U(B) <= incumbent is safe.
```

Tier-1 intended-family result:

```text
FEBB AUC:              0.998779
nearest ablation AUC:  0.970848
best baseline AUC:     0.971411
```

Why it is interesting:

- massive hidden-cost advantage on bounded basin:
  ```text
  FEBB hidden cost ≈ 33
  beam/no-prune hidden cost ≈ 712–2444
  ```
- shows real structural primitive: certified pruning can avoid many regions.

Why it is **not promoted**:

- It did not beat best baseline by the required +5% relative, because the benchmark was already near ceiling.
- It failed anti-bias `loose_bound_tree`:
  ```text
  FEBB anti-bias AUC: 0.649073
  best anti-bias baseline: 0.919027
  ```
- The top10 metric is imperfect for exact-pruning search: FEBB can certify/stop after very few terminal evaluations, so top10 denominator can be <10. This is a valid “optimization/certification” signal but not directly PMO top10 behavior.

Correct verdict:

```text
FEBB is the strongest remaining structural idea, but only as synthetic/certified-search signal.
It needs a new v2 plan for top-k enumeration / PMO-compatible candidate generation before any PMO claim.
```

### 6.2 EFR — Exact Fragment Recombination

Physical object:

```text
Constrained fragment hypergraph / grammar; exact recombination searches feasible assemblies instead of blind crossover.
```

Tier-1 intended-family result:

```text
EFR AUC:              0.686484
nearest ablation AUC: 0.638575
best baseline AUC:    0.668974
```

Why it is interesting:

- Clear operator-level novelty: exact compositional recombination beats blind fragment GA on complete grammar.
- Stronger PMO analogy than FEBB in one sense: molecules really are compositional/fragmented.

Why it is **not promoted**:

- Anti-bias failure was catastrophic when grammar was incomplete:
  ```text
  EFR anti-bias AUC: 0.474035
  best anti-bias baseline: 0.966107
  ```
- This exposes the central risk: exact fragment recombination is only as good as grammar completeness.

Correct verdict:

```text
EFR is promising only if we can build a high-recall molecular fragment grammar.
Without grammar coverage guarantees, it is brittle and not PMO-ready.
```

### 6.3 RG — Renormalization Coarse-to-Fine

Physical object:

```text
Effective coarse Hamiltonian over relevant variables, followed by fine refinement.
```

Tier-1 intended-family result:

```text
RG AUC:             0.968532
fine-only AUC:      0.896448
best baseline AUC:  0.970454
```

Interpretation:

- RG beat fine-only search, so the coarse primitive worked.
- But strong GA matched/exceeded it at higher budget.
- Anti-bias failure was large:
  ```text
  RG anti-bias AUC: 0.627937
  best anti-bias baseline: 0.873216
  ```

Correct verdict:

```text
RG remains too fragile, especially given earlier Shape-then-TB failure.
Stop or redesign.
```

### 6.4 AIA, MCTS, SGC, NCI

| Candidate | Verdict | Reason |
|---|---|---|
| AIA | `STOP_OR_REDESIGN` | close to greedy/UCB but did not beat best surrogate baseline; rebranding risk remains |
| MCTS | `STOP_OR_REDESIGN` | beam search beat MCTS-lookahead on intended family |
| SGC | `REBRANDED_BASELINE` | cluster moves lost local/no-cluster and GA baselines |
| NCI | `REBRANDED_BASELINE` | forward prior + rerank beat inverse posterior decoding |

---

## 7. Theory verdict

The only directions with real structural evidence are:

```text
1. FEBB — certified pruning / upper-bound search
2. EFR  — exact recombination over a complete grammar
```

But neither passes promotion gates.

Theoretical verdict:

```text
No remaining direction is currently validated enough for PMO micro.
FEBB is the strongest first-principles candidate, but only under admissible/strong bounds and with a top-k enumeration fix.
EFR is the strongest molecule-like compositional candidate, but only under high-recall fragment grammar assumptions.
```

The clean theoretical condition for possible superiority is now clearer:

| Direction | Condition under which it can beat GA/GFN |
|---|---|
| FEBB | admissible/tight upper bounds allow pruning/certifying whole low-free-energy basins |
| EFR | grammar is complete/high-recall and local constraints factorize composition efficiently |

Without these conditions, both fail hard.

---

## 8. Novelty verdict

Safe claims:

```text
We tested all remaining structural core directions in a re-audited synthetic portfolio.
FEBB and EFR are the only synthetic-interesting survivors.
FEBB is currently the highest-ranked structural core, but not PMO-ready.
EFR is molecule-like but grammar-brittle.
```

Unsafe claims:

```text
FEBB/EFR beat PMO SOTA.
FEBB/EFR are validated GFlowNet algorithms.
Any candidate is proven better than Genetic GFN.
Any candidate should enter PMO micro without a separate plan.
```

---

## 9. Testing / automation verdict

Regression tests passed:

| Test | Result |
|---|---:|
| `test/smiles_gflownet/test_option_flow_poc.jl` | 25 / 25 |
| `test/smiles_gflownet/test_option_flow_real_catalog.jl` | 25 / 25 |

No existing Option-Flow tests were broken by the portfolio runner.

---

## 10. Done vs remaining

| Done | Remaining |
|---|---|
| Re-audited portfolio v1 and wrote v2 | Do not run PMO micro from current results |
| Implemented synthetic core screening runner | Need decide whether to write separate FEBB-top-k or EFR-grammar plan |
| Tested all seven remaining core directions in Tier-0 | Need avoid overclaiming FEBB due top10-denominator/certification issue |
| Ran Tier-1 for survivors EFR/FEBB/RG | Need possibly return to GA_GFN app integration if no new theory direction passes |
| Identified SGC/NCI as rebranded baselines | Need preserve negative results |
| Regression tests passed |  |

---

## 11. Bottom line

```text
No remaining direction is PMO-ready.
```

Best current ranking:

```text
1. FEBB — strongest structural signal, but synthetic-only and bound-dependent.
2. EFR  — molecule-like compositional signal, but grammar-completeness brittle.
3. RG   — useful vs fine-only but loses to strong GA and anti-bias.
4. AIA  — close but not better than surrogate baselines.
5. MCTS — beam search stronger.
6. SGC  — rebranded/weak cluster move.
7. NCI  — rebranded reranking/inverse decoding.
```

Recommended next theory action, if continuing theory search:

```text
Write a separate v2-audited plan for FEBB-top-k enumeration or EFR high-recall molecular grammar.
```

Recommended practical action, if stopping theory search for now:

```text
Return to GA_GFN app integration.
```
