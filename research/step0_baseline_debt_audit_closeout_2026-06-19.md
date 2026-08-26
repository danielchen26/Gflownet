# Step 0 Baseline-Debt Audit Closeout

**Authoritative time:** Friday, 2026-06-19 01:14 EDT  
**Execution workspace:** `/Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint`  
**Plan:** `step0_baseline_debt_audit_plan_2026-06-19.md`  
**Runner:** `test/smiles_gflownet/run_step0_baseline_debt_audit.jl`

---

## 1. Executive verdict

Step 0 was executed as a smoke + micro audit after the 600-call reduced core proved too slow for a single run window.

Canonical Step 0 result:

```text
QGFN / Boosting do not close the gap.
Heuristic HE remains strongest by mean AUC in the fast audit.
TB-only has the best mean final top10.
Option-Flow is not rescued by baseline debt, but also not yet validated.
Integrated O3 remains the decisive test.
```

This means the critique-driven worry — “maybe Option-Flow only looks new because QGFN/Boosting were never run” — is **not supported by this micro audit**. QGFN and Boosting did not beat heuristic HE overall.

But the result does **not** restore optimistic SOTA language. Instead:

> Option-Flow remains conditionally plausible only if integrated O3 can directly beat heuristic HE or provide a structural-task specialist advantage.

---

## 2. What was implemented

Added:

```text
test/smiles_gflownet/run_step0_baseline_debt_audit.jl
```

The runner:

- loads `checkpoints/pretrain/final.jls`;
- runs PMO baselines through `run_smiles_pmo_task`;
- supports `tb_only`, `tb_qgfn`, `tb_boosting`, `tb_qgfn_boosting`, `heuristic_he`;
- records per-run status instead of crashing the entire audit;
- computes aggregate rows, paired deltas, and go/no-go decisions;
- now saves incremental partial bundles after every arm.

---

## 3. Runs executed

### 3.1 Smoke gate

```bash
STEP0_MODE=smoke
STEP0_TASKS=qed
STEP0_ARMS=tb_only,tb_qgfn,tb_boosting,tb_qgfn_boosting,heuristic_he
STEP0_BUDGET=192
STEP0_SEEDS=17
STEP0_ITERS=3
STEP0_BATCH=8
STEP0_BOOST_ROUNDS=3
```

Smoke result:

- all confirmatory arms ran successfully;
- checkpoint/runtime OK;
- QGFN/Boosting symbols available;
- no PMO arm crashed;
- log `tee` initially failed only because the output directory did not exist before tee opened the file; the experiment itself completed and saved `step0_smoke_results.jls`.

Smoke overall AUC on QED:

| Arm | AUC |
|---|---:|
| `tb_qgfn_boosting` | 0.898374 |
| `tb_qgfn` | 0.896303 |
| `tb_only` | 0.884203 |
| `tb_boosting` | 0.873535 |
| `heuristic_he` | 0.854522 |

Smoke already suggested that on smooth QED, TB/QGFN can beat heuristic HE.

### 3.2 Reduced core attempt

```bash
STEP0_MODE=core_reduced
STEP0_TASKS=qed,drd2,celecoxib_rediscovery
STEP0_ARMS=tb_only,tb_qgfn,tb_boosting,tb_qgfn_boosting,heuristic_he
STEP0_BUDGET=600
STEP0_SEEDS=17,23
STEP0_ITERS=5
STEP0_BATCH=12
STEP0_BOOST_ROUNDS=3
```

Outcome:

- The run timed out after 3 hours.
- This established that the 600-call full matrix is too heavy for this workflow without sharding/resume.
- After this, the runner was improved to save partial results after every arm.

Because the pre-partial-save core was interrupted, it is **not** used as the headline result.

### 3.3 Micro audit

```bash
STEP0_MODE=micro
STEP0_TASKS=qed,drd2,celecoxib_rediscovery
STEP0_ARMS=tb_only,tb_qgfn,tb_boosting,tb_qgfn_boosting,heuristic_he
STEP0_BUDGET=300
STEP0_SEEDS=17,23
STEP0_ITERS=3
STEP0_BATCH=8
STEP0_BOOST_ROUNDS=1
```

Outcome:

- all runs completed;
- but Boosting arms only used ~192/300 calls because `BOOST_ROUNDS=1` ended early;
- therefore Boosting in this micro run was under-budget and not fair as a final Boosting verdict.

### 3.4 Boosting fairness correction

```bash
STEP0_MODE=micro_boost3
STEP0_TASKS=qed,drd2,celecoxib_rediscovery
STEP0_ARMS=tb_boosting,tb_qgfn_boosting
STEP0_BUDGET=300
STEP0_SEEDS=17,23
STEP0_ITERS=3
STEP0_BATCH=8
STEP0_BOOST_ROUNDS=3
```

Outcome:

- all boosting runs completed;
- all boosting runs used 300 calls;
- this corrected the under-budget Boosting issue.

The final interpretation uses:

- `tb_only`, `tb_qgfn`, `heuristic_he` from `micro`;
- `tb_boosting`, `tb_qgfn_boosting` from `micro_boost3`.

---

## 4. Combined Step 0 results

### Overall by arm

| Arm | Mean AUC | Mean final top10 | Interpretation |
|---|---:|---:|---|
| `heuristic_he` | **0.646938** | 0.690663 | best mean AUC |
| `tb_only` | 0.605886 | **0.776086** | best final top10 |
| `tb_qgfn` | 0.497922 | 0.591509 | helps QED, hurts DRD2/celecoxib |
| `tb_qgfn_boosting` | 0.420661 | 0.433082 | weak overall |
| `tb_boosting` | 0.415551 | 0.439400 | weak overall |

### Per-task combined results

| Task | Arm | AUC mean | AUC std | Top10 | Calls |
|---|---|---:|---:|---:|---:|
| `qed` | `tb_qgfn` | **0.915053** | 0.001379 | 0.924500 | 300 |
| `qed` | `tb_only` | 0.909723 | 0.021219 | **0.930990** | 300 |
| `qed` | `tb_qgfn_boosting` | 0.897890 | 0.003358 | 0.917627 | 300 |
| `qed` | `heuristic_he` | 0.877243 | 0.043457 | 0.902446 | 300 |
| `qed` | `tb_boosting` | 0.869682 | 0.010755 | 0.900097 | 300 |
| `drd2` | `heuristic_he` | **0.421961** | 0.419807 | 0.495620 | 300 |
| `drd2` | `tb_only` | 0.353112 | 0.053026 | **0.577545** | 300 |
| `drd2` | `tb_boosting` | 0.091287 | 0.101576 | 0.107770 | 300 |
| `drd2` | `tb_qgfn` | 0.067827 | 0.012515 | 0.102762 | 300 |
| `drd2` | `tb_qgfn_boosting` | 0.046626 | 0.004609 | 0.055048 | 300 |
| `celecoxib_rediscovery` | `heuristic_he` | **0.641611** | 0.163643 | 0.673923 | 300 |
| `celecoxib_rediscovery` | `tb_only` | 0.554824 | 0.010091 | **0.819724** | 300 |
| `celecoxib_rediscovery` | `tb_qgfn` | 0.510885 | 0.052618 | 0.747264 | 300 |
| `celecoxib_rediscovery` | `tb_qgfn_boosting` | 0.317466 | 0.011751 | 0.326571 | 300 |
| `celecoxib_rediscovery` | `tb_boosting` | 0.285683 | 0.006520 | 0.310333 | 300 |

---

## 5. What the audit says

### 5.1 QGFN

QGFN is useful on QED:

```text
qed tb_qgfn AUC = 0.915053
qed heuristic AUC = 0.877243
```

But QGFN is weak on DRD2 and celecoxib in this micro setting:

```text
drd2 tb_qgfn AUC = 0.067827 vs heuristic 0.421961
celecoxib tb_qgfn AUC = 0.510885 vs heuristic 0.641611
```

So QGFN does **not** close the baseline debt globally.

### 5.2 Boosting

After rerunning with `BOOST_ROUNDS=3`, Boosting still did not close the gap:

```text
tb_boosting mean AUC = 0.415551
tb_qgfn_boosting mean AUC = 0.420661
heuristic_he mean AUC = 0.646938
```

Boosting is therefore not a hidden easy answer in this small audit.

### 5.3 Heuristic HE

Heuristic HE remains the strongest AUC arm overall and on the two non-smooth tasks:

```text
drd2 heuristic_he AUC = 0.421961
celecoxib heuristic_he AUC = 0.641611
```

But it has high variance, especially on DRD2:

```text
drd2 heuristic_he std = 0.419807
```

So the result is strong enough to guide next work, but not statistically final.

### 5.4 TB-only

TB-only has the best final top10 mean overall:

```text
tb_only mean final top10 = 0.776086
heuristic_he mean final top10 = 0.690663
```

This suggests a recurring pattern:

- heuristic HE improves AUC/sample-efficiency on harder/structural tasks;
- TB-only can still produce strong final top-k by the end of the small budget.

This matters for Option-Flow: if Option-Flow is built only over HE, it may miss TB-only's final top-k strength.

---

## 6. Go / No-Go verdict

Pre-registered rule triggered:

```text
Heuristic HE remains strongest among audited baselines:
Option-Flow must beat heuristic in integrated O3 or be downgraded.
```

Rules not triggered:

```text
QGFN or Boosting >= heuristic HE globally
```

Therefore:

> QGFN/Boosting do not invalidate Option-Flow by closing the baseline gap. But they also do not rescue the project. The decisive test remains integrated O3 against heuristic HE.

---

## 7. Option-Flow theory verdict

**Still inconclusive with negative pressure.**

Step 0 changes the diagnosis in a specific way:

Before Step 0, a serious critique was:

> Maybe Option-Flow only looks useful because existing QGFN/Boosting baselines were not audited.

After Step 0 micro:

> That critique is weakened. QGFN/Boosting did not beat heuristic overall. There remains a real search-control gap.

But Option-Flow still has not filled that gap.

Therefore the theory status becomes:

```text
Not falsified by baseline debt.
Not validated as main algorithm.
Integrated O3 is mandatory.
```

---

## 8. Novelty verdict

Step 0 does not prove academic novelty, but it clarifies practical novelty pressure.

Current safe novelty claim remains:

> A novel molecular-PMO formulation/layer: frontier-conditioned finite-horizon Option-Flow GFlowNet over bounded search options with frontier/search-state improvement utility.

But current practical status is:

> It is a plausible new direction only if integrated Option-Flow beats heuristic HE or specializes strongly on structural/scaffold-sensitive tasks.

If O3 fails, the method should be downgraded to:

```text
auxiliary router / structural-task specialist / research idea
```

not a main SOTA algorithm.

---

## 9. Testing / automation verdict

**Improved, but core audit needs sharding/resume.**

What works:

- Step 0 runner works;
- checkpoint loads;
- QGFN / Boosting / heuristic HE arms run;
- all smoke and micro runs completed successfully;
- Boosting fairness correction completed with full 300 calls;
- partial-save support was added after the 600-call timeout.

What needs improvement:

- 600-call × 30-run matrix is too slow for a single synchronous run;
- future audits need sharding or resume from partials;
- diversity is often 0.0 in these small runs, so diversity diagnostics are not informative yet.

---

## 10. Done vs remaining

| Done | Remaining |
|---|---|
| Audited and re-audited Step 0 plan | Run sharded 600–1000 budget core if needed |
| Implemented Step 0 runner | Add proper resume/skip-existing logic |
| Ran smoke gate | Build integrated O3 Option-Flow PMO hook |
| Ran micro audit across 3 tasks × 5 arms × 2 seeds | Compare integrated Option-Flow vs heuristic HE |
| Corrected Boosting under-budget confound with boost_rounds=3 rerun | Test structural-task specialist hypothesis |
| Established QGFN/Boosting do not close gap in micro audit | Scale only if O3 beats heuristic or has structural specialty |

---

## 11. Next required step

Do **not** jump to 23-task or SOTA claim.

Next step is:

```text
O3 integrated PMO-lite:
TB-only / heuristic HE / Option-Flow HE under matched PMO budget
```

But O3 should be scoped around the actual Step 0 insight:

1. Option-Flow must beat heuristic HE on AUC, or it is downgraded.
2. If it loses smooth tasks but wins structural tasks, preserve it only as structural/scaffold-sensitive auxiliary router.
3. If it loses heuristic again overall and structurally, Direction C/Option-Flow mainline should be paused.
