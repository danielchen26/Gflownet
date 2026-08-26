# O3 Integrated Option-Flow PMO-Lite Closeout

**Authoritative time:** Friday, 2026-06-19 12:25 EDT  
**Execution workspace:** `/Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint`  
**Mode:** O3a warm-start integrated PMO-lite  
**Final verdict:** `O3_MAINLINE_DOWNGRADE_WITH_STRUCTURAL_AUXILIARY_SIGNAL`

---

## 1. Executive summary

O3 v2 was implemented and executed inside the real `run_smiles_pmo_task` PMO loop.

The key fairness correction was applied: Option-Flow did **not** deploy an h1/h2-only menu against a heuristic h8 baseline. All option-menu arms used the same locked menu:

```text
o3_schema_menu_v1:
  mutate_h2
  crossover_h2
  mixed_h2
  mixed_h4
  heuristic_default_h8
```

The `heuristic_default_h8` equivalence check passed.

### Final O3a result

Option-Flow sample beat heuristic HE on mean AUC, but failed the stronger pre-registered rule because it did **not** beat uniform option selection over the same menu.

```text
uniform_option_he      mean AUC = 0.712451
option_flow_sample_he  mean AUC = 0.658979
heuristic_he_default   mean AUC = 0.631866
option_flow_greedy_he  mean AUC = 0.574180
tb_only                mean AUC = 0.502968
```

Therefore:

```text
Mainline Option-Flow PMO direction: downgrade / pause.
Structural/scaffold-sensitive auxiliary signal: preserve as a possible specialist only.
No SOTA or mainline restoration claim.
```

---

## 2. What changed

### Modified

```text
src/utils/visualization/core/pmo_benchmark.jl
```

Added:

- `he_episode_selector` keyword to `run_smiles_pmo_task`;
- selector invocation before HE warmup and interleaved HE episodes;
- `operator_override` passthrough to `run_hierarchical_edit_episode!`;
- explicit versioned `he_selector_metadata` in every HE episode summary.

### Added

```text
test/smiles_gflownet/run_o3_integrated_option_flow_pmo.jl
```

Runner features:

- exact O3 schema menu;
- strict same-snapshot catalog training over the exact deployed menu;
- integrated PMO arms:
  - `tb_only`
  - `heuristic_he_default`
  - `uniform_option_he`
  - `option_flow_sample_he`
  - `option_flow_greedy_he`
- partial-save/resume;
- selector metadata round-trip check;
- heuristic h8 equivalence check;
- final O3 gate with uniform-option downgrade priority;
- selector artifact persistence for future deterministic resume.

---

## 3. Commands run

### Compile smoke

```bash
cd /Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint
JULIA_CONDAPKG_BACKEND=Null \
JULIA_PYTHONCALL_EXE=$PWD/.CondaPkg/.pixi/envs/default/bin/python \
O3_MODE=compile_smoke \
O3_TASKS=qed \
O3_ARMS=heuristic_he_default \
O3_BUDGET=48 \
O3_SEEDS=17 \
O3_ITERS=1 \
O3_BATCH=4 \
O3_VERBOSE=false \
O3_RESUME=false \
julia --project=. test/smiles_gflownet/run_o3_integrated_option_flow_pmo.jl
```

### O3 smoke

```bash
cd /Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint
JULIA_CONDAPKG_BACKEND=Null \
JULIA_PYTHONCALL_EXE=$PWD/.CondaPkg/.pixi/envs/default/bin/python \
O3_MODE=smoke \
O3_TASKS=qed \
O3_ARMS=heuristic_he_default,uniform_option_he,option_flow_sample_he \
O3_BUDGET=192 \
O3_SEEDS=17 \
O3_ITERS=3 \
O3_BATCH=8 \
O3_VERBOSE=true \
O3_RESUME=true \
julia --project=. test/smiles_gflownet/run_o3_integrated_option_flow_pmo.jl \
  2>&1 | tee checkpoints/o3_integrated_option_flow_pmo/run_smoke.log
```

### O3a micro

```bash
cd /Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint
JULIA_CONDAPKG_BACKEND=Null \
JULIA_PYTHONCALL_EXE=$PWD/.CondaPkg/.pixi/envs/default/bin/python \
O3_MODE=o3a_micro \
O3_TASKS=qed,drd2,celecoxib_rediscovery \
O3_ARMS=tb_only,heuristic_he_default,uniform_option_he,option_flow_sample_he,option_flow_greedy_he \
O3_BUDGET=300 \
O3_SEEDS=17,23 \
O3_ITERS=3 \
O3_BATCH=8 \
O3_VERBOSE=false \
O3_RESUME=true \
julia --project=. test/smiles_gflownet/run_o3_integrated_option_flow_pmo.jl \
  2>&1 | tee checkpoints/o3_integrated_option_flow_pmo/run_micro.log
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
checkpoints/o3_integrated_option_flow_pmo/o3_smoke_results.jls
checkpoints/o3_integrated_option_flow_pmo/o3_o3a_micro_results.jls
checkpoints/o3_integrated_option_flow_pmo/o3_latest_results.jls
checkpoints/o3_integrated_option_flow_pmo/o3_o3a_micro_partial_results.jls
checkpoints/o3_integrated_option_flow_pmo/run_smoke.log
checkpoints/o3_integrated_option_flow_pmo/run_micro.log
research/o3_integrated_option_flow_pmo_closeout_2026-06-19.md
```

Important integrity note:

- During post-run gate regeneration, resume mode retrained selector catalogs without rerunning rows.
- The final bundle was patched so `selector_training` is restored from the original partial bundle generated during the completed rows.
- Runner code was then fixed to save/load selector artifacts for future deterministic resume.

Final bundle integrity check:

```text
selector_training total_calls = 396
metadata_roundtrip_check = ok, 18 / 18 active selector/menu rows
```

---

## 5. Smoke result

Task: `qed`, budget `192`, seed `17`.

| Arm | AUC | Top10 | Calls | HE episodes |
|---|---:|---:|---:|---:|
| `heuristic_he_default` | 0.926098 | 0.929080 | 192 | 2 |
| `option_flow_sample_he` | 0.878464 | 0.915171 | 192 | 3 |
| `uniform_option_he` | 0.862878 | 0.880197 | 192 | 3 |

Smoke engineering checks:

```text
heuristic_default_h8 equivalence: pass
selector arms execute: pass
metadata round-trip: pass, 2 / 2 active selector/menu rows
```

Smoke scientific signal:

```text
sample > uniform, but sample < heuristic on qed.
Negative pressure already visible, but not sufficient alone for final O3a verdict.
```

---

## 6. O3a micro aggregate results

Tasks: `qed`, `drd2`, `celecoxib_rediscovery`  
Seeds: `17`, `23`  
Budget: `300` online calls per task/arm/seed  
Selector training: warm-start, reported separately, not counted in headline online budget.

### Overall by arm

| Arm | Mean AUC | Mean final top10 |
|---|---:|---:|
| `uniform_option_he` | 0.712451 | 0.748276 |
| `option_flow_sample_he` | 0.658979 | 0.727208 |
| `heuristic_he_default` | 0.631866 | 0.700307 |
| `option_flow_greedy_he` | 0.574180 | 0.626599 |
| `tb_only` | 0.502968 | 0.621920 |

### Per-task AUC

| Task | TB-only | Heuristic HE | Uniform option | Option-Flow sample | Option-Flow greedy |
|---|---:|---:|---:|---:|---:|
| `celecoxib_rediscovery` | 0.521293 | 0.738672 | 0.840469 | 0.862483 | 0.677000 |
| `drd2` | 0.092785 | 0.253692 | 0.390784 | 0.186971 | 0.150745 |
| `qed` | 0.894824 | 0.903235 | 0.906100 | 0.927482 | 0.894794 |

### Selector training summary

```text
catalogs = 9
candidates = 45
actual selector-training calls = 396
train CE gain vs uniform = +0.1676985
val CE gain vs uniform = +0.12670231
training budget counted in headline = false
```

### Selector distribution

For `option_flow_sample_he` only:

```text
mutate_h2              4
crossover_h2           2
mixed_h2               2
mixed_h4              10
heuristic_default_h8  10
```

No total option collapse:

```text
unique selected schemas = 5
max schema fraction = 0.3571
collapse check = pass
```

---

## 7. Gate analysis

Final serialized gate:

```text
verdict = O3_MAINLINE_DOWNGRADE_WITH_STRUCTURAL_AUXILIARY_SIGNAL
failed_runs = 0
metadata_roundtrip_check = ok, 18 / 18
sample_vs_heuristic_relative = +4.29%
sample_vs_heuristic_top10_relative = +3.84%
sample_vs_uniform = negative overall
eligible_for_o3b_raw_before_uniform_gate = true
eligible_for_o3b = false
learned_not_better_than_uniform = true
```

Interpretation:

1. **Against heuristic HE:** Option-Flow sample looks better on mean AUC and top10.
2. **Against uniform option HE:** Option-Flow sample loses clearly.
3. Therefore, the learned selector is not adding value beyond random selection over the same fair menu.
4. The pre-registered rule says this is a downgrade condition.

---

## 8. Structural auxiliary signal

`celecoxib_rediscovery` is the one clear structural/scaffold-like positive:

```text
option_flow_sample_he AUC = 0.862483
heuristic_he_default AUC = 0.738672
uniform_option_he AUC = 0.840469
```

Relative to heuristic HE:

```text
AUC gain ≈ +16.8%
final top10 gain ≈ +9.9%
```

Relative to uniform option HE:

```text
AUC gain ≈ +2.6%
final top10 gain ≈ +1.7%
```

Because the selector did not collapse to a single option, this is a legitimate auxiliary signal. But because the learned selector loses uniform overall, it is **not** enough to preserve Option-Flow as a mainline PMO algorithm direction.

Safe interpretation:

```text
Possible structural/scaffold-sensitive option router.
Not validated as a generally useful learned PMO selector.
```

---

## 9. Done vs remaining

| Area | Done | Remaining |
|---|---|---|
| PMO integration | Added `he_episode_selector` hook inside real `run_smiles_pmo_task`. | None for O3a. |
| Fair schema menu | Locked exact menu with `heuristic_default_h8`. | If continuing, design a structural-only menu or richer state features. |
| Metadata | Versioned selector metadata stored in HE episode summaries. | None immediate. |
| Resume | Partial rows saved; future selector artifact save/load added. | Existing O3a final bundle patched manually for training-summary integrity. |
| Smoke | Passed. | None. |
| O3a micro | Completed 30 / 30 rows. | No O3b mainline run warranted under current gate. |
| Tests | Existing Option-Flow POC and real catalog tests passed. | Add a small dedicated unit test for `he_episode_selector` hook if this code is kept. |
| Scientific decision | Mainline downgrade with structural auxiliary signal. | Decide whether to pursue targeted structural specialist or pause Option-Flow. |

---

## 10. Theory verdict

```text
Object signal: still present.
Integrated online PMO signal: mixed-to-negative.
Heuristic beating: sample beats heuristic in O3a micro, but this is not sufficient.
Uniform-menu beating: failed.
Mainline Option-Flow PMO status: downgraded / paused.
Structural specialist status: plausible auxiliary signal only.
```

The canonical theory object remains meaningful:

```math
P_\theta(\omega \mid S_t) \propto U(\omega; S_t)
```

But O3a shows that the learned selector is not yet reliably better than a uniform sampler over a fair option menu inside integrated PMO. That weakens the claim from “new useful PMO direction” to “interesting option-flow object with possible structural routing use.”

---

## 11. Novelty verdict

Safe claim remains:

```text
Novel formulation/layer: frontier-conditioned bounded option-flow over finite-horizon search options.
```

Unsafe claims remain blocked:

```text
Not SOTA.
Not a generally validated PMO algorithm.
Not shown to beat a simple uniform option scheduler over the same fair menu.
```

---

## 12. Testing / automation verdict

```text
O3 smoke: passed.
O3a micro: completed 30 / 30 rows.
Metadata round-trip: passed, 18 / 18 active selector/menu rows.
Heuristic h8 equivalence: passed.
Option-Flow v0 POC test: 25 / 25 passed.
Option-Flow real artifact catalog test: 25 / 25 passed.
```

Automation caveat fixed:

```text
Initial resume-only gate regeneration retrained selector metadata without rerunning rows.
Final bundle was patched from the original partial artifact.
Runner now persists selector artifacts for future deterministic resume.
```

---

## 13. Recommended next decision

Do **not** run mainline O3b as if O3a were positive. The uniform-option failure blocks mainline restoration.

Reasonable choices now:

1. **Pause/downgrade Option-Flow mainline** and return to GA_GFN/app integration.
2. **Run a targeted structural-specialist follow-up** only on structural/scaffold-sensitive tasks, explicitly framed as an auxiliary router test.
3. **Reformulate selector learning** with stronger state features or leave-task-out transfer, but only after accepting current O3a as a negative mainline result.
