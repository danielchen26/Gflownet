# Option-Flow Real Evidence POC Closeout

**Authoritative time:** Thursday, 2026-06-11 16:06 EDT  
**Workspace:** `/Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint`  
**Plan executed:** `sessions/260314-misty-eddy/plans/option_flow_real_evidence_poc_plan_v2_reaudited.md`  
**Evidence level reached:** E1 summary proxy + E2 typed-path proxy  

---

## 1. Plain-English executive summary

The synthetic Option-Flow v0 result was strong. The next question was whether the same philosophy has real signal in actual HE/PMO artifacts.

This run says:

> **There is real signal, but it is not yet decisive.**

More precisely:

1. **Summary-only leak-free evidence is weak / fragile.**
   - With the primary `frontier_gain_sum` utility and the stricter `task_phase_budget100` proxy grouping, leak-free summary features did **not** robustly beat uniform.
   - Leaky outcome features were strongly positive, proving the artifacts contain outcome signal, but that is **not deployable evidence**.

2. **Typed raw decision-log features improve the picture.**
   - A lightweight raw-artifact stub successfully loaded typed HE diagnostics for **214/214** episodes without full GFlowNet/RDKit.
   - `typed_path` features from basin/parent/operator decision logs produced mixed-positive signal.
   - Coarse proxy grouping passed all seeds and structural tasks showed positive signal, especially `celecoxib_rediscovery`.

3. **This still does not prove the final Option-Flow object.**
   - Existing historical artifacts have **zero repeated `task::snapshot_id` groups**, so they cannot directly test exact same-state catalogs.
   - Therefore this is not yet strict proof of:

```math
P_\theta(\omega \mid S_t) \propto U(\omega; S_t)
```

It is realistic evidence that real HE decision traces contain some utility-allocating signal, and it justifies moving to **E3 strict same-snapshot catalog generation**, not claiming SOTA yet.

---

## 2. What was implemented

| Area | Files |
|---|---|
| Real artifact loader / catalog builder | `src/training/option_flow_real_catalog.jl` |
| Raw diagnostic lightweight deserialization stub | `test/smiles_gflownet/support/option_flow_raw_stub.jl` |
| Real evidence runner | `test/smiles_gflownet/run_option_flow_real_evidence_poc.jl` |
| Unit tests | `test/smiles_gflownet/test_option_flow_real_catalog.jl` |
| Package exports | `src/GFlowNet.jl` |
| v0 metric boundary fix | `src/training/option_flow_loss.jl` |

The implementation deliberately avoids loading full `GFlowNet.jl` / PythonCall / RDKit for E1/E2 evidence.

---

## 3. Artifact inventory

Default roots:

- `checkpoints/final_theory_direct_test/artifacts/final_theory_v1`
- `checkpoints/level3_shape_then_tb/artifacts/heuristic_shape_then_tb`
- `checkpoints/level3_shape_then_tb/artifacts/learned_shape_then_tb`
- `/Users/tianchichen/Documents/GitHub/Gflownet/checkpoints/truth_sprint_stage_b_f015_truth_tasksharded`

Audit:

| Item | Count / Status |
|---|---:|
| HE summary files | 36 |
| Real episodes | 214 |
| Tasks | 6 |
| Repeated `task::snapshot_id` groups | 0 |
| `he_episode_summary.jls` lightweight deserialize | yes |
| `he_capacity_summary.jls` lightweight deserialize | yes |
| `he_raw_diagnostics.jls` without stub | no |
| `he_raw_trajectory.jls` without stub | no |
| typed raw diagnostics with stub | 214 / 214 loaded |

Task counts:

| Task | Episodes |
|---|---:|
| `qed` | 40 |
| `drd2` | 40 |
| `celecoxib_rediscovery` | 41 |
| `gsk3b` | 32 |
| `albuterol_similarity` | 31 |
| `jnk3` | 30 |

---

## 4. Anti-leakage firewall

Headline models exclude post-outcome fields such as:

- `frontier_gain_sum`
- `frontier_gain_max`
- `delta_top1_max`
- `delta_top10_mean_max`
- `best_reward`
- `commits_applied`
- `step_count`
- `calls_used`
- `child_reward`
- `reward_delta`
- `frontier_utility_delta`

A `leaky_upper` mode was run only as a diagnostic upper bound. It is explicitly **not deployable** and is not used for the theory verdict.

---

## 5. Main result bundles and logs

| Run | Result bundle | Log |
|---|---|---|
| Primary full E1 | `checkpoints/option_flow_real_evidence_poc/option_flow_real_evidence_poc_full_e1_results.jls` | `checkpoints/option_flow_real_evidence_poc/run_full_e1.log` |
| Secondary `delta_top10` | `checkpoints/option_flow_real_evidence_poc/option_flow_real_evidence_poc_delta_top10_leak_free_results.jls` | `checkpoints/option_flow_real_evidence_poc/run_delta_top10_leak_free.log` |
| Secondary `best_delta` | `checkpoints/option_flow_real_evidence_poc/option_flow_real_evidence_poc_best_delta_leak_free_results.jls` | `checkpoints/option_flow_real_evidence_poc/run_best_delta_leak_free.log` |
| Typed-path primary | `checkpoints/option_flow_real_evidence_poc/option_flow_real_evidence_poc_typed_path_primary_results.jls` | `checkpoints/option_flow_real_evidence_poc/run_typed_path_primary.log` |

---

## 6. Primary utility result: `frontier_gain_sum`

### 6.1 Summary-only E1

| Grouping | Feature mode | Pass seeds | CE gain | Utility lift | Top-Q lift | Rank corr | Catalogs | Candidates |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `task_phase_budget100` | leak-free | 1/3 | -0.00155 | -0.00216 | -0.00236 | 0.0217 | 23 | 200 |
| `task_phase_b100_t05` | leak-free | 2/3 | +0.00094 | +0.00181 | +0.00105 | 0.1458 | 31 | 141 |
| `task_phase` | leak-free | 2/3 | +0.00469 | +0.02832 | +0.01085 | 0.2020 | 22 | 214 |
| `task_phase_budget100` | leaky upper | 3/3 | +0.06147 | +0.31341 | +0.11188 | 0.9667 | 23 | 200 |

Interpretation:

- Strict-ish leak-free summary grouping is **not robust**.
- Coarser grouping has some positive signal, but it is weaker evidence because state matching is looser.
- Leaky upper bound is strongly positive, confirming the artifacts contain outcome signal, but this cannot support deployable Option-Flow claims.

### 6.2 E2 typed-path proxy

Typed-path mode uses raw basin/parent/operator decision logs, excluding child reward / utility outcomes as features.

| Grouping | Feature mode | Pass seeds | CE gain | Utility lift | Top-Q lift | Rank corr | Catalogs | Candidates |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `task_phase_budget100` | typed-path | 2/3 | -0.00144 | +0.00064 | -0.00027 | 0.0620 | 23 | 200 |
| `task_phase_b100_t05` | typed-path | 1/3 | -0.00165 | -0.00270 | +0.00117 | 0.0738 | 31 | 141 |
| `task_phase` | typed-path | 3/3 | +0.00920 | +0.05392 | +0.02456 | 0.2913 | 22 | 214 |

Structural task signal under typed-path:

| Grouping | Structural signal |
|---|---|
| `task_phase_budget100` | `celecoxib_rediscovery` positive; `albuterol_similarity` negative |
| `task_phase_b100_t05` | weak `celecoxib_rediscovery` positive; `albuterol_similarity` negative |
| `task_phase` | `celecoxib_rediscovery` and `albuterol_similarity` positive |

Interpretation:

- Typed decision-path features are more promising than summary-only features.
- The effect is still grouping-sensitive.
- This supports continuing to exact same-snapshot catalogs, but does not yet prove the object.

---

## 7. Secondary utility sensitivity

### 7.1 `delta_top10_mean_max`

| Grouping | Feature mode | Pass seeds | CE gain | Utility lift | Top-Q lift | Rank corr |
|---|---|---:|---:|---:|---:|---:|
| `task_phase_budget100` | leak-free | 0/3 | -0.00056 | -0.00004 | +0.00058 | 0.0150 |
| `task_phase_b100_t05` | leak-free | 0/3 | -0.00095 | -0.00001 | +0.00183 | -0.0026 |
| `task_phase` | leak-free | 3/3 | +0.10967 | +0.00395 | +0.09563 | 0.1994 |

Interpretation: top10-delta signal appears only under very coarse grouping. It is not strong strict-proxy evidence.

### 7.2 `best_delta_vs_initial_top1`

| Grouping | Feature mode | Pass seeds | CE gain | Utility lift | Top-Q lift | Rank corr |
|---|---|---:|---:|---:|---:|---:|
| `task_phase_budget100` | leak-free | 2/3 | +0.01733 | +0.00086 | +0.00666 | 0.2046 |
| `task_phase_b100_t05` | leak-free | 1/3 | -0.00112 | +0.00047 | -0.00676 | 0.1248 |
| `task_phase` | leak-free | 0/3 | -0.01746 | +0.00147 | -0.00406 | 0.1185 |

Interpretation: best-delta has a clearer strict-proxy CE signal than frontier-gain, but the absolute utility lift is tiny.

---

## 8. Pass/fail gates

| Gate | Result | Reason |
|---|---|---|
| E1 summary-only primary | **Fail / weak** | Headline strict-ish `task_phase_budget100` leak-free primary utility passed only 1/3 seeds and had negative mean CE/utility lift. |
| E1 summary-only secondary | **Mixed** | `best_delta` strict-ish passed 2/3 seeds; `delta_top10` only worked in coarse grouping. |
| E2 typed-path proxy | **Mixed-positive** | raw diagnostics loaded 214/214; coarse typed-path passed 3/3 and structural tasks positive, but stricter groupings remain fragile. |
| E3 strict same-snapshot | **Not executed** | Existing artifacts have zero repeated exact same-state groups; fresh generation requires full HE runtime, currently blocked/risky due full package / CondaPkg-RDKit load path. |
| SOTA / PMO claim | **Not supported** | No online-lite PMO and no 23-task benchmark. |

---

## 9. Done vs remaining

| Done | Remaining |
|---|---|
| Built leak-free real artifact loader from summary Dicts | Exact same-snapshot `S_t -> {ω_i}` catalogs |
| Added summary proxy catalog construction | Online-lite execution of Option-Flow-selected options |
| Added source/config prior baselines | Direct comparison vs TB-only / HE / greedy / Genetic GFN online |
| Added leaky upper-bound diagnostic with explicit label | Full PMO AUC evidence |
| Added raw diagnostic stub and typed-path features | Formal stochastic edit-TB semantics |
| Loaded typed raw diagnostics for 214/214 episodes | Resolve full HE runtime / CondaPkg-RDKit load blocker safely |
| Added unit tests: 25/25 pass | Larger multi-seed strict generated catalogs |
| Produced result bundles/logs for primary + sensitivities | SOTA-scale 10K / 23-task evidence |

---

## 10. Theory verdict

**Mixed-positive, not decisive.**

The real artifacts do **not** yet prove that the full Option-Flow philosophy will beat current GFlowNet SOTA. The summary-only strict-proxy evidence is too weak for that.

However, the typed-path result is important:

> Real HE decision logs contain some learnable frontier-utility allocation signal under leak-free decision-time features.

So the theory is **not falsified**. It remains plausible and now has real-artifact support beyond synthetic catalogs, but the next required proof is exact same-snapshot catalog generation.

The honest conclusion is:

> Option-Flow is promising enough to continue to E3, but not strong enough yet to claim method correctness or SOTA potential as established fact.

---

## 11. Testing / automation verdict

**Automation improved materially.**

Passing test:

```text
julia --project=. test/smiles_gflownet/test_option_flow_real_catalog.jl
Option-Flow real artifact catalog | 25 pass / 25 total
```

Automated coverage now includes:

- summary artifact discovery;
- schema normalization;
- proxy catalog construction;
- leak-free / typed-path / leaky feature modes;
- metadata prior baseline;
- small training/evaluation path;
- E1 gate helper.

Caveats:

- full SMILES test suite was not rerun;
- raw stub produces Julia world-age warnings under Julia 1.12 but successfully deserializes current artifacts;
- E3 generation is not automated yet.

---

## 12. Recommended next step

Do **not** jump to SOTA claims.

The next realistic proof step is:

> **Option-Flow E3: strict same-snapshot generated catalogs.**

Required experiment:

```text
same frozen frontier S_t
  -> generate K bounded HE options from cloned frontier
  -> score each with real frontier utility U_i
  -> train/evaluate Pθ(ω_i | S_t) ∝ U_i
  -> compare uniform / heuristic / greedy / Option-Flow
```

This requires resolving the full HE runtime path safely. The earlier full `GFlowNet` load timed out on CondaPkg/RDKit locking; do not silently delete lock files. Either:

1. ask permission to clean the stale `.CondaPkg/lock`, then run E3 in the isolated clone; or
2. create a dedicated lightweight strict-catalog generator that avoids full package loading.

Only if E3 passes should we proceed to online-lite PMO and SOTA-facing comparisons.
