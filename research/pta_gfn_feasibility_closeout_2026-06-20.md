# PTA-GFN Feasibility Validation Closeout

**Authoritative time:** Saturday, 2026-06-20 00:02 EDT  
**Execution workspace:** `/Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint`  
**Plan:** `pta_gfn_feasibility_validation_plan_v2_reaudited_2026-06-19.md`  
**Runner:** `test/smiles_gflownet/run_pta_gfn_feasibility.jl`  
**Final verdict:** `PTA0_STOP_OR_REDESIGN`

---

## 1. Executive summary

PTA-GFN v2 was implemented as a **PTA-0 frozen-state exact-pool feasibility test**.

The test directly asked:

```text
On held-out frozen PMO states, with the exact same candidate pool and exact same batch-option catalog,
does a learned batch-flow selector beat uniform and no-oracle proxy batch selection?
```

Answer:

```text
No.
```

Main held-out PTA-0 micro result:

| Arm | Mean utility | Mean ΔTop10 |
|---|---:|---:|
| `oracle_upper` | 0.00090018 | 0.00720144 |
| `pta_batch_flow_greedy` | 0.00071207 | 0.00569653 |
| `proxy_parent_reward` | 0.00069308 | 0.00554466 |
| `proxy_diverse` | 0.00063475 | 0.00507798 |
| `uniform_batch` | 0.00060408 | 0.00483260 |
| `pta_batch_flow_sample` | 0.00057721 | 0.00461772 |
| `candidate_ranker_greedy` | 0.00055029 | 0.00440233 |
| `candidate_ranker_sample` | 0.00042787 | 0.00342298 |

The pre-registered continue gate failed because:

```text
pta_batch_flow_sample <= uniform_batch
pta_batch_flow_sample < best no-oracle proxy baseline
pta_batch_flow_greedy > pta_batch_flow_sample
```

Therefore PTA-GFN should **not** proceed to PTA-1 online PMO-lite in its current form.

---

## 2. What was implemented

Added:

```text
test/smiles_gflownet/run_pta_gfn_feasibility.jl
```

The runner implements:

- frozen-state generation from fixed bootstrap seeds;
- candidate pool generation from existing replay/top parents via mutation/crossover/token mutation;
- canonicalization, duplicate filtering, and already-evaluated filtering before selection;
- batch-option catalogs where each option is an entire batch `B_t`;
- strict held-out evaluation on frozen states;
- separate batch-flow model and candidate-ranker ablation;
- exact pool hash / candidate ID logging;
- evidence oracle-call accounting separated from online PMO budget;
- stop/continue gate.

No `run_smiles_pmo_task` integration was performed, by design.

---

## 3. Commands run

### PTA-0 smoke

```bash
cd /Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint
JULIA_CONDAPKG_BACKEND=Null \
JULIA_PYTHONCALL_EXE=$PWD/.CondaPkg/.pixi/envs/default/bin/python \
PTA_MODE=smoke \
PTA_TASKS=qed \
PTA_SEEDS=17 \
PTA_TRAIN_STATES=2 \
PTA_HELDOUT_STATES=1 \
PTA_POOL_SIZE=24 \
PTA_BATCH_SIZE=6 \
PTA_BATCH_OPTIONS=8 \
PTA_EPOCHS=30 \
PTA_RESUME=false \
julia --project=. test/smiles_gflownet/run_pta_gfn_feasibility.jl
```

Smoke engineering result:

```text
frozen states generated: yes
candidate pool generated: yes
batch catalogs generated: yes
batch-flow selector trained: yes
candidate-ranker ablation trained: yes
held-out evaluation completed: yes
```

Smoke scientific signal was already negative, but one held-out catalog was treated as engineering smoke, not a final scientific gate.

### PTA-0 micro

```bash
cd /Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint
mkdir -p checkpoints/pta_gfn_feasibility
JULIA_CONDAPKG_BACKEND=Null \
JULIA_PYTHONCALL_EXE=$PWD/.CondaPkg/.pixi/envs/default/bin/python \
PTA_MODE=pta0_micro \
PTA_TASKS=qed,drd2,celecoxib_rediscovery \
PTA_SEEDS=17,23 \
PTA_TRAIN_STATES=4 \
PTA_HELDOUT_STATES=2 \
PTA_POOL_SIZE=40 \
PTA_BATCH_SIZE=8 \
PTA_BATCH_OPTIONS=16 \
PTA_EPOCHS=120 \
PTA_RESUME=false \
julia --project=. test/smiles_gflownet/run_pta_gfn_feasibility.jl \
  2>&1 | tee checkpoints/pta_gfn_feasibility/run_pta0_micro.log
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

## 4. Artifacts

```text
checkpoints/pta_gfn_feasibility/pta_smoke_results.jls
checkpoints/pta_gfn_feasibility/pta_smoke_selector_bundle.jls
checkpoints/pta_gfn_feasibility/pta_pta0_micro_results.jls
checkpoints/pta_gfn_feasibility/pta_pta0_micro_selector_bundle.jls
checkpoints/pta_gfn_feasibility/pta_latest_results.jls
checkpoints/pta_gfn_feasibility/run_pta0_micro.log
research/pta_gfn_feasibility_closeout_2026-06-20.md
```

---

## 5. PTA-0 micro setup

```text
tasks = qed, drd2, celecoxib_rediscovery
seeds = 17, 23
train states per task/seed = 4
held-out states per task/seed = 2
candidate pool size = 40
batch size = 8
batch options per state = 16
epochs = 120
held-out catalogs = 12
train batch catalogs = 24
```

Training/validation split inside the batch-flow trainer:

```text
batch-flow train catalogs = 19
batch-flow validation catalogs = 5
candidate-ranker train catalogs = 19
candidate-ranker validation catalogs = 5
```

Batch-flow training metrics:

```text
train CE gain vs uniform = +0.0187
val CE gain vs uniform = +0.0082
train rank correlation = 0.2198
val rank correlation = 0.2312
```

Candidate-ranker training metrics:

```text
train CE gain vs uniform = +0.0759
val CE gain vs uniform = +0.0427
```

Interpretation: the batch-flow model barely learned useful ranking signal; candidate-ranker had more offline CE signal, but did not dominate held-out utility either.

---

## 6. Evidence oracle budget

These are PTA-0 evidence-generation calls, not online PMO deployment calls.

| Task/Seed | Total unique oracle calls | Catalog evidence calls | Bootstrap seeds |
|---|---:|---:|---:|
| `qed::17` | 195 | 138 | 9 |
| `qed::23` | 216 | 159 | 9 |
| `drd2::17` | 210 | 150 | 12 |
| `drd2::23` | 198 | 138 | 12 |
| `celecoxib_rediscovery::17` | 193 | 133 | 13 |
| `celecoxib_rediscovery::23` | 201 | 141 | 13 |

Total unique evidence calls across all task/seeds: approximately `1213`.

This is acceptable for an object/signal test, but it is not a fair online PMO budget.

---

## 7. Per-task held-out results

Mean utility over held-out states.

| Task | Arm | Mean utility |
|---|---|---:|
| `qed` | `oracle_upper` | 0.001964195 |
| `qed` | `proxy_diverse` | 0.001704386 |
| `qed` | `uniform_batch` | 0.001624823 |
| `qed` | `pta_batch_flow_greedy` | 0.001586652 |
| `qed` | `pta_batch_flow_sample` | 0.001567553 |
| `qed` | `proxy_parent_reward` | 0.001538233 |
| `qed` | `candidate_ranker_greedy` | 0.001478138 |
| `qed` | `candidate_ranker_sample` | 0.001046869 |
| `drd2` | `oracle_upper` | 0.000164143 |
| `drd2` | `candidate_ranker_sample` | 0.000108998 |
| `drd2` | `uniform_batch` | 0.000088270 |
| `drd2` | `pta_batch_flow_greedy` | 0.000079575 |
| `drd2` | `proxy_diverse` | 0.000072105 |
| `drd2` | `proxy_parent_reward` | 0.000071045 |
| `drd2` | `pta_batch_flow_sample` | 0.000030171 |
| `drd2` | `candidate_ranker_greedy` | 0.000001060 |
| `celecoxib_rediscovery` | `oracle_upper` | 0.000572203 |
| `celecoxib_rediscovery` | `proxy_parent_reward` | 0.000469971 |
| `celecoxib_rediscovery` | `pta_batch_flow_greedy` | 0.000469971 |
| `celecoxib_rediscovery` | `candidate_ranker_greedy` | 0.000171676 |
| `celecoxib_rediscovery` | `pta_batch_flow_sample` | 0.000133920 |
| `celecoxib_rediscovery` | `candidate_ranker_sample` | 0.000127750 |
| `celecoxib_rediscovery` | `proxy_diverse` | 0.000127750 |
| `celecoxib_rediscovery` | `uniform_batch` | 0.000099133 |

Task pattern:

- QED: `pta_batch_flow_sample` loses uniform and proxy-diverse.
- DRD2: `pta_batch_flow_sample` loses uniform and candidate-ranker-sample.
- Celecoxib: `pta_batch_flow_sample` beats uniform but loses proxy-parent and greedy.

So there is no robust cross-task signal.

---

## 8. Gate result

Serialized gate:

```text
verdict = PTA0_STOP_OR_REDESIGN
continue_gate = false
sample_vs_uniform_relative = -4.45%
sample_vs_proxy_best_relative = -16.72%
paired_wins_vs_uniform = 4 / 12
```

Reasons:

```text
pta_batch_flow_sample <= uniform_batch on held-out frozen states.
pta_batch_flow_sample loses best no-oracle proxy baseline.
pta_batch_flow_greedy beats sample; check temperature/calibration or reinterpret as value ranker.
```

This directly triggers the v2 stop condition.

---

## 9. Interpretation

The result does **not** prove that all possible population/top-k acquisition algorithms are bad.

It does prove that this simplest PTA-GFN v0 implementation does **not** yet show the required core signal:

```text
learned batch-flow sample > uniform same-pool batch selection
```

The strongest non-oracle signal in this v0 is closer to a simple proxy/greedy rule:

```text
pta_batch_flow_greedy ≈ proxy_parent_reward
```

That suggests the model may be learning a weak value/ranking tendency, not a useful proportional batch-flow distribution.

---

## 10. Done vs remaining

| Area | Done | Remaining |
|---|---|---|
| PTA v2 re-audit | Completed and corrected plan. | None. |
| PTA-0 runner | Implemented. | Could be cleaned into tests only if direction continues. |
| Smoke | Passed engineering path. | No need to rerun. |
| PTA-0 micro | Completed 12 held-out catalogs. | PTA-1 should not run under current gate. |
| Same-pool fairness | Implemented by frozen-state batch catalogs and pool hashes. | Could add stronger cryptographic proof table if needed. |
| Candidate-ranker ablation | Implemented. | More sophisticated candidate-ranker could be tested only in redesign. |
| Regression tests | Option-Flow POC and real catalog tests passed. | Add dedicated PTA unit tests only if keeping code. |

---

## 11. Theory verdict

```text
PTA object: conceptually still plausible.
PTA-GFN v0 evidence: negative.
Batch-flow sample signal: not established.
Uniform same-pool beating: failed.
Proxy baseline beating: failed.
Online PTA-1: not warranted.
```

This is analogous to the Option-Flow lesson but caught earlier and more cheaply:

```text
A learned flow-like selector that cannot beat uniform same-pool selection should not be promoted.
```

---

## 12. Novelty verdict

Safe:

```text
Population/top-k acquisition flow remains a potentially novel formulation.
```

Blocked:

```text
No claim that PTA-GFN works.
No claim that it is better than Genetic GFN.
No claim that it is a SOTA-facing algorithm yet.
No claim that batch-flow is better than ranker/proxy selection.
```

---

## 13. Testing / automation verdict

Passed:

```text
PTA-0 smoke engineering path
PTA-0 micro execution
Option-Flow v0 POC: 25 / 25
Option-Flow real artifact catalog: 25 / 25
```

Automation caveat:

```text
The PTA runner is an experimental feasibility runner, not a production test.
It should not be integrated into the main PMO loop unless a redesigned PTA-0 passes the continue gate.
```

---

## 14. Recommended next decision

Do **not** run PTA-1 online-lite for the current PTA-GFN v0.

Reasonable choices:

1. Stop PTA-GFN mainline for now and return to GA_GFN/app integration.
2. If still interested in the acquisition idea, redesign around a stronger value/ranker objective first, not proportional batch-flow sampling.
3. Consider a different first-principles direction: distributional/quantile GFN or learned genetic operator model, but require the same uniform/proxy comparator discipline.

Bottom line:

```text
PTA-GFN was the highest-prior next idea, but the first strict PTA-0 test failed.
Do not promote it.
```
