# Option-Flow Online-Lite PMO Closeout

**Authoritative time:** Thursday, 2026-06-18 12:28 PM EDT  
**Workspace:** `/Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint`  
**Plan:** `option_flow_online_lite_pmo_plan_v2_reaudited_2026-06-18.md`  
**Runner:** `test/smiles_gflownet/run_option_flow_online_lite_pmo.jl`

---

## 1. Executive verdict

The E3 object **does transfer online**, but not yet as a clean global heuristic-beater.

Headline result:

```text
O1 warm-start verdict: ONLINE_SELECTOR_USEFUL_BUT_FAIRNESS_OPEN
O2 total-budget-fair verdict: ONLINE_FAIR_BUDGET_SIGNAL_PRESENT
```

Meaning:

- Option-Flow sampling beat uniform option selection on **3/3 tasks** in both O1 and O2.
- It beat heuristic HE on the structural task `celecoxib_rediscovery` in both O1 and O2.
- It did **not** beat heuristic HE on all-task mean AUC; heuristic remains slightly better on QED/DRD2 and overall AUC.
- Therefore the current setup is **useful and worth integrating further**, but it is **not yet a SOTA claim**.

The honest strongest claim is:

> Option-Flow is now supported by offline strict-object evidence and first online-lite deployment evidence. It has fair-budget directional signal, especially for structural search, but still needs integrated PMO-lite/O3 and then 10K/23-task benchmarking before any SOTA-facing claim.

---

## 2. What was implemented

Added:

```text
test/smiles_gflownet/run_option_flow_online_lite_pmo.jl
```

The runner supports:

- O1 warm-started online selector proof;
- O2 total-budget-fair Variant A;
- deployable arms only for headline comparisons;
- optional `oracle_upper` diagnostic, excluded from headline;
- online-lite top10 AUC tracking;
- budget-separated serialized result bundles.

Artifacts:

```text
checkpoints/option_flow_online_lite_pmo/option_flow_online_lite_pmo_smoke_results.jls
checkpoints/option_flow_online_lite_pmo/option_flow_online_lite_pmo_o1_core_results.jls
checkpoints/option_flow_online_lite_pmo/option_flow_online_lite_pmo_o2_total_budget_results.jls
checkpoints/option_flow_online_lite_pmo/run_o1_core.log
checkpoints/option_flow_online_lite_pmo/run_o2_total_budget.log
```

---

## 3. Commands executed

### Smoke

```bash
JULIA_CONDAPKG_BACKEND=Null \
JULIA_PYTHONCALL_EXE=$PWD/.CondaPkg/.pixi/envs/default/bin/python \
OPTION_FLOW_ONLINE_MODE=smoke \
OPTION_FLOW_ONLINE_TASKS=qed \
OPTION_FLOW_ONLINE_ARMS=uniform_schema,option_flow_sample \
OPTION_FLOW_ONLINE_BUDGET=120 \
OPTION_FLOW_ONLINE_TRAIN_BUDGET=180 \
OPTION_FLOW_ONLINE_TRAIN_SNAPSHOTS=2 \
OPTION_FLOW_ONLINE_EPOCHS=40 \
OPTION_FLOW_ONLINE_MAX_STEPS=16 \
julia --project=. test/smiles_gflownet/run_option_flow_online_lite_pmo.jl
```

Smoke passed. `option_flow_sample` beat uniform on QED, but the smoke verdict remained weak because the heuristic arm was not included.

### O1 core

```bash
JULIA_CONDAPKG_BACKEND=Null \
JULIA_PYTHONCALL_EXE=$PWD/.CondaPkg/.pixi/envs/default/bin/python \
OPTION_FLOW_ONLINE_MODE=o1_core \
OPTION_FLOW_ONLINE_TASKS=qed,drd2,celecoxib_rediscovery \
OPTION_FLOW_ONLINE_ARMS=uniform_schema,heuristic_mixed_h2,prior_best_schema,option_flow_sample,option_flow_greedy \
OPTION_FLOW_ONLINE_BUDGET=400 \
OPTION_FLOW_ONLINE_TRAIN_BUDGET=400 \
OPTION_FLOW_ONLINE_TRAIN_SNAPSHOTS=3 \
OPTION_FLOW_ONLINE_EPOCHS=140 \
OPTION_FLOW_ONLINE_SEEDS=17,23 \
OPTION_FLOW_ONLINE_MAX_STEPS=64 \
julia --project=. test/smiles_gflownet/run_option_flow_online_lite_pmo.jl
```

### O2 total-budget-fair

```bash
JULIA_CONDAPKG_BACKEND=Null \
JULIA_PYTHONCALL_EXE=$PWD/.CondaPkg/.pixi/envs/default/bin/python \
OPTION_FLOW_ONLINE_MODE=o2_total_budget \
OPTION_FLOW_ONLINE_TASKS=qed,drd2,celecoxib_rediscovery \
OPTION_FLOW_ONLINE_ARMS=uniform_schema,heuristic_mixed_h2,prior_best_schema,option_flow_sample,option_flow_greedy \
OPTION_FLOW_ONLINE_BUDGET=600 \
OPTION_FLOW_ONLINE_TRAIN_FRACTION=0.30 \
OPTION_FLOW_ONLINE_TRAIN_SNAPSHOTS=2 \
OPTION_FLOW_ONLINE_EPOCHS=90 \
OPTION_FLOW_ONLINE_SEEDS=17,23 \
OPTION_FLOW_ONLINE_MAX_STEPS=80 \
julia --project=. test/smiles_gflownet/run_option_flow_online_lite_pmo.jl
```

---

## 4. Budget accounting

| Protocol | Selector training budget | Online budget | Headline fairness |
|---|---:|---:|---|
| Smoke | nominal 180 on QED; actual 65 calls | 120 per arm | smoke only |
| O1 | nominal 400/task; actual 290 calls total across 9 catalogs | 400 per task/arm/seed | warm-start; fairness open |
| O2 | nominal 180/task/seed charged to selector arms | 420 for selector arms, 600 for baselines | total-budget-fair, conservative |

O2 was conservative: selector arms lost the full nominal 180 calls of online budget, even though actual training calls were only 55–78 per task/seed.

O1 pretraining catalogs:

| Task | Catalogs | Actual training calls | Frontier advanced by |
|---|---:|---:|---|
| `qed` | 3 | 92 | `mutate_h2`, `mixed_h2`, `mutate_h2` |
| `drd2` | 3 | 107 | `mutate_h2`, `mutate_h2`, `mixed_h1` |
| `celecoxib_rediscovery` | 3 | 91 | `mutate_h2`, `mutate_h1`, `mutate_h2` |

O1 training total:

```text
catalogs = 9
candidates = 54
actual training calls = 290
all 9 catalogs informative
```

---

## 5. O1 results — warm-started online selector

### O1 overall deployable leaderboard

| Arm | Mean AUC top10 | Mean final top10 |
|---|---:|---:|
| `heuristic_mixed_h2` | 0.282966 | 0.304946 |
| `option_flow_sample` | 0.281349 | 0.300976 |
| `uniform_schema` | 0.270882 | 0.298574 |
| `option_flow_greedy` | 0.257446 | 0.269343 |
| `prior_best_schema` | 0.257446 | 0.269343 |

### O1 task-level sample comparison

| Task | Uniform AUC | Heuristic AUC | Option-Flow sample AUC | Sample verdict |
|---|---:|---:|---:|---|
| `qed` | 0.617309 | 0.641386 | 0.640833 | beats uniform, ties/slightly trails heuristic |
| `drd2` | 0.016633 | 0.022973 | 0.016660 | barely beats uniform, trails heuristic |
| `celecoxib_rediscovery` | 0.178703 | 0.184538 | 0.186554 | beats both |

O1 gate:

```text
sample_wins_uniform = 3/3
sample_wins_heuristic = 1/3
structural_positive = true
verdict = ONLINE_SELECTOR_USEFUL_BUT_FAIRNESS_OPEN
```

Interpretation:

- Strong evidence that Option-Flow improves over random option selection.
- Structural-task signal is real.
- Not yet a global heuristic replacement.

---

## 6. O2 results — total-budget-fair Variant A

O2 charges selector-dependent arms a nominal 30% training budget, leaving less online budget than baselines.

### O2 overall deployable leaderboard

| Arm | Mean AUC top10 | Mean final top10 |
|---|---:|---:|
| `heuristic_mixed_h2` | 0.286048 | 0.306559 |
| `option_flow_sample` | 0.281988 | 0.308457 |
| `uniform_schema` | 0.271610 | 0.298779 |
| `option_flow_greedy` | 0.270982 | 0.288659 |
| `prior_best_schema` | 0.269979 | 0.286522 |

### O2 task-level sample comparison

| Task | Uniform AUC | Heuristic AUC | Option-Flow sample AUC | Sample verdict |
|---|---:|---:|---:|---|
| `qed` | 0.618968 | 0.646083 | 0.635480 | beats uniform, trails heuristic |
| `drd2` | 0.017924 | 0.025851 | 0.023011 | beats uniform, trails heuristic |
| `celecoxib_rediscovery` | 0.177938 | 0.186210 | 0.187473 | beats both |

O2 gate:

```text
sample_wins_uniform = 3/3
sample_wins_heuristic = 1/3
structural_positive = true
verdict = ONLINE_FAIR_BUDGET_SIGNAL_PRESENT
```

Important nuance:

- O2 is positive because Option-Flow sample stayed above uniform on every task and beat heuristic on the structural task **despite charged training budget**.
- O2 is not a clean all-task heuristic victory; heuristic still has slightly higher overall AUC.
- Option-Flow sample had the best O2 overall final top10 mean, but not the best AUC.

---

## 7. Sampling vs greedy interpretation

Greedy did not dominate.

In O1:

- `option_flow_greedy` and `prior_best_schema` were identical at overall mean AUC 0.257446.
- Both were below sampling and below heuristic.

In O2:

- `option_flow_sample` AUC: 0.281988
- `option_flow_greedy` AUC: 0.270982
- `prior_best_schema` AUC: 0.269979

This supports the GFlowNet-style non-collapse thesis more than a pure argmax/ranker thesis:

> proportional option sampling is currently more useful than always taking the model's greedy schema.

However, the small task/seed count means this is directional, not final.

---

## 8. Done vs remaining

| Done | Remaining |
|---|---|
| Implemented online-lite Option-Flow runner | Integrate Option-Flow policy into `run_smiles_pmo_task` / `pmo_benchmark.jl` HE scheduling |
| Ran smoke | Add direct TB-only / Genetic-GFN / AugMem-comparable O3 PMO-lite baselines |
| Ran O1 warm-start core on 3 tasks × 5 arms × 2 seeds | Run broader 6-task and then 23-task protocols |
| Ran O2 total-budget-fair protocol | Run larger budgets, especially 1K–2K and 10K |
| Verified deployable arms do not evaluate all schemas before choosing | Add leave-task-out amortized selector protocol |
| Preserved oracle-upper as diagnostic-only and did not use it in headline | Add official PMO metric compatibility layer |
| Existing Option-Flow unit/real-catalog tests still pass | Build automated unit tests for the new online runner internals |

---

## 9. Theory verdict

**Positive but narrower than SOTA.**

The theory survived its first online deployment check:

```math
P_\theta(\omega \mid S_t) \propto U(\omega; S_t)
```

Evidence now supports three levels:

1. **E3 object-level proof:** same-state option utilities are real and learnable.
2. **O1 online transfer:** learned sampling improves over uniform online and helps structural search.
3. **O2 fair-budget signal:** the same pattern survives conservative total-budget charging.

But the theory is not yet a complete PMO algorithm because it has not been integrated with the full token-level GFlowNet/TB/GA stack.

The current best interpretation is:

> Option-Flow is a promising frontier-conditioned option-routing layer, especially for structural search and non-greedy exploration, but it is not yet a full standalone replacement for strong heuristic HE or full PMO baselines.

---

## 10. Testing / automation verdict

**Improved and reproducible.**

New automated runner:

```text
test/smiles_gflownet/run_option_flow_online_lite_pmo.jl
```

Validation commands:

```bash
julia --project=. test/smiles_gflownet/test_option_flow_poc.jl
julia --project=. test/smiles_gflownet/test_option_flow_real_catalog.jl
```

Results:

```text
Option-Flow v0 POC | 25 pass / 25 total
Option-Flow real artifact catalog | 25 pass / 25 total
```

The online runner itself is an experiment runner, not a unit test, because it invokes RDKit/TDC and real HE rollouts.

---

## 11. SOTA viability verdict

**SOTA pursuit is justified, but SOTA is not proven.**

The result is strong enough to justify the next integrated PMO step because:

- Option-Flow sample consistently beats uniform online.
- It wins the structural task against heuristic in both O1 and O2.
- It remains positive under conservative total-budget charging.
- Greedy does not dominate sampling, so the GFlowNet-style distributional object still matters.

The result is not strong enough to claim SOTA because:

- heuristic HE still slightly wins mean AUC;
- there is no O3 integrated PMO comparison yet;
- no TB-only / Genetic-GFN / AugMem direct benchmark in this protocol;
- budgets are 400–600, not 10K;
- tasks are 3, not 23;
- seeds are 2, not enough for final statistical claims.

### Next decisive step

Implement **O3 integrated PMO-lite**:

```text
TB-only / heuristic HE / Option-Flow HE under matched PMO budget
```

The clean next gate should be:

> Does Option-Flow improve PMO AUC/top10 over heuristic HE and TB-only when plugged into the actual `run_smiles_pmo_task` loop?

If O3 passes, then a 6-task/1K–2K run is warranted. If that passes, then a 23-task/10K SOTA-facing run is warranted.
