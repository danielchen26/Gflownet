# PCCWM Core Sprint Closeout — Proof-Carrying Counterfactual Core Did Not Pass

**Authoritative user time:** Saturday, 2026-06-20 11:43 EDT  
**Plan executed:** `/Users/tianchichen/Documents/GitHub/Gflownet/sessions/260314-misty-eddy/plans/proof_carrying_counterfactual_world_model_plan_v2_reaudited_2026-06-20.md`  
**Execution workspace:** `/Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint`  
**Runner:** `test/smiles_gflownet/run_pccwm_core_sprint.jl`  
**Result bundle:** `checkpoints/pccwm_core/pccwm_micro_results.jls`  
**Verdict:** `PCCWM_STOP_OR_REDESIGN`

---

## 1. What was tested

This sprint tested the v2 re-audited idea:

```text
PCCWM core = Proof-Carrying Counterfactual World Model core
```

The goal was not to optimize molecules directly. It was to test whether a model over **intervention programs** could produce a machine-checkable certificate of counterfactual improvement that remains useful under synthetic OOD/confounding stress.

Important scope guardrails:

```text
Stage 0 only.
No PMO benchmark.
No learned GFlowNet component.
No SOTA claim.
No molecular causal-mechanism claim.
```

---

## 2. Implementation summary

Added one isolated runner:

```text
test/smiles_gflownet/run_pccwm_core_sprint.jl
```

Implemented:

- synthetic causal worlds with motif, nuisance, scaffold, and spurious confounder bits;
- intervention operators over motif bits;
- a CWM effect model estimated from paired interventions;
- conformal-style lower-bound certificate:

```text
L(u,c) = predicted_delta(u,c) - calibration_quantile
Verify = support_count >= threshold AND L(u,c) > 0
```

- required arms:

```text
pccwm_full
no_certificate_cwm
correlational_surrogate
certificate_only_filter
random_intervention
ga_candidate_search
rank_weighted_ga
ucb_surrogate
thompson_surrogate
```

- required synthetic tasks:

```text
T1 confounded_motif_ood
T2 scaffold_shift_transfer
T3 nonidentifiable_antibias
```

- required metrics/gates:

```text
counterfactual MAE
certified hit rate
false certificate rate
OOD top10 AUC
coverage / abstention
paired wins vs best strong baseline
```

---

## 3. Runs

### Compile

```text
PCCWM_MODE=compile
```

Passed.

### Smoke

```text
PCCWM_MODE=smoke
PCCWM_TASKS=confounded_motif_ood
PCCWM_SEEDS=17
PCCWM_BUDGET=80
```

Engineering passed. Scientific signal was already negative:

```text
PCCWM_STOP_OR_REDESIGN
```

### Micro

```text
PCCWM_MODE=micro
PCCWM_TASKS=confounded_motif_ood,scaffold_shift_transfer,nonidentifiable_antibias
PCCWM_SEEDS=17,23
PCCWM_BUDGET=200
```

Formal result:

```text
PCCWM_STOP_OR_REDESIGN
```

---

## 4. Micro overall ranking

| Arm | Mean AUC | Mean Top10 | Certified hit | False cert | Coverage | Proposal hit |
|---|---:|---:|---:|---:|---:|---:|
| `ucb_surrogate` | 0.948390168 | 0.993646667 | NaN | 0.0000 | 0.0000 | 0.1966 |
| `thompson_surrogate` | 0.929581658 | 0.987603333 | NaN | 0.0000 | 0.0000 | 0.2083 |
| `rank_weighted_ga` | 0.924841595 | 0.997376667 | NaN | 0.0000 | 0.0000 | 0.1670 |
| `certificate_only_filter` | 0.914858161 | 0.998926667 | 1.0000 | 0.0000 | 0.8723 | 0.9357 |
| `ga_candidate_search` | 0.910437772 | 0.997800000 | NaN | 0.0000 | 0.0000 | 0.2509 |
| `correlational_surrogate` | 0.904367550 | 0.997600000 | NaN | 0.0000 | 0.0000 | 0.9339 |
| `no_certificate_cwm` | 0.888878510 | 0.983950000 | NaN | 0.0000 | 0.0000 | 0.7717 |
| `random_intervention` | 0.887598518 | 0.967686667 | NaN | 0.0000 | 0.0000 | 0.1812 |
| `pccwm_full` | 0.880536102 | 0.971316667 | 1.0000 | 0.0000 | 0.6368 | 0.7998 |

Key observation:

```text
PCCWM had safe certificates, but safety did not translate into optimization advantage.
```

Even worse for the novelty claim:

```text
certificate_only_filter > pccwm_full
ucb_surrogate > pccwm_full
thompson_surrogate > pccwm_full
rank_weighted_ga > pccwm_full
```

So the proof/certificate layer was useful only as a filter, not as evidence that the counterfactual world-model mechanism itself is a strong new optimization primitive.

---

## 5. Gate results

From `checkpoints/pccwm_core/pccwm_micro_results.jls`:

| Gate | Result | Metric |
|---|---:|---:|
| Mechanism gate | FAIL | CWM/correlational MAE ratio = 0.943178 |
| Certificate gate | PASS | false cert = 0.0; cert hit = 1.0; no-cert proposal hit = 0.771739 |
| Optimization gate | FAIL | PCCWM mean AUC = 0.880536 vs best strong baseline = 0.948390 |
| Paired wins vs best strong baseline | FAIL | 0 / 6 |
| Continue gate | FAIL | final verdict `PCCWM_STOP_OR_REDESIGN` |

Required threshold for mechanism gate was:

```text
CWM MAE <= 0.80 × correlational MAE
```

Observed:

```text
CWM MAE / correlational MAE = 0.943178
```

That is only a small error reduction and not enough to claim a causal/counterfactual mechanism advantage.

Required optimization threshold was:

```text
PCCWM AUC >= 1.05 × best strong baseline AUC
paired wins > half of task-seed cases
```

Observed:

```text
PCCWM vs best strong baseline relative = -7.15%
paired wins = 0 / 6
```

---

## 6. Per-task diagnosis

### T1 — confounded motif OOD

| Arm | AUC |
|---|---:|
| `ucb_surrogate` | 0.950791079 |
| `thompson_surrogate` | 0.935609688 |
| `rank_weighted_ga` | 0.927870163 |
| `correlational_surrogate` | 0.921695529 |
| `no_certificate_cwm` | 0.914419431 |
| `pccwm_full` | 0.907174529 |

PCCWM certificate was safe:

```text
certified_hit = 1.0
false_cert = 0.0
coverage = 0.9620
```

But the search/active-learning baselines found high reward faster. The ordinary correlational surrogate also did not collapse badly enough under this OOD split to expose a decisive mechanism advantage.

### T2 — scaffold shift transfer

| Arm | AUC |
|---|---:|
| `ucb_surrogate` | 0.964816018 |
| `thompson_surrogate` | 0.942867252 |
| `rank_weighted_ga` | 0.933884504 |
| `no_certificate_cwm` | 0.922334901 |
| `pccwm_full` | 0.921910266 |

PCCWM again produced safe certificates:

```text
certified_hit = 1.0
false_cert = 0.0
coverage = 0.9484
```

But it did not beat standard active-learning/search baselines. This blocks the claim that the intervention-program object is already stronger than conventional acquisition/search.

### T3 — non-identifiable anti-bias

| Arm | AUC |
|---|---:|
| `ucb_surrogate` | 0.929563406 |
| `certificate_only_filter` | 0.926570124 |
| `rank_weighted_ga` | 0.912770118 |
| `thompson_surrogate` | 0.910268036 |
| `ga_candidate_search` | 0.904754172 |
| `pccwm_full` | 0.812523511 |

PCCWM correctly avoided false certificates:

```text
certified_count = 0
false_cert = 0.0
coverage = 0.0
```

This is the one good behavior: under non-identifiability, the mechanism certificate abstained instead of hallucinating a proof.

But abstention also made optimization poor, and the v2 stop gate explicitly disallows a direction that wins only by safety/abstention. Here it did not even win.

---

## 7. Theoretical verdict

### Final theory verdict

```text
PCCWM_CORE_NOT_PROMOTED
```

More detailed verdict:

```text
Proof-carrying certification is useful as a safety/filtering layer,
but this synthetic test did not establish a new counterfactual world-model optimization primitive.
```

Why not promoted:

1. **Mechanism advantage too weak.**
   The CWM counterfactual MAE was only slightly better than a correlational surrogate on identifiable OOD tasks.

2. **Optimization advantage absent.**
   PCCWM lost to UCB, Thompson, rank-weighted GA, certificate-only filter, GA, and correlational surrogate in mean AUC.

3. **Certificate layer is not unique to PCCWM.**
   `certificate_only_filter` outperformed `pccwm_full`, implying the observed certificate usefulness may come from conformal filtering rather than the proposed causal/counterfactual object.

4. **Anti-bias behavior was safe but not useful.**
   PCCWM correctly abstained in the non-identifiable task, but this produced poor optimization and cannot justify a PMO/GFN direction.

---

## 8. What remains scientifically true

Allowed statement:

```text
PCCWM Stage 0 showed that certificates can be calibrated conservatively in the synthetic setup, including abstention under non-identifiability.
```

Not allowed:

```text
PCCWM is a new validated PMO/GFlowNet algorithm.
PCCWM beats strong optimization baselines.
PCCWM establishes molecular causal mechanisms.
PCCWM warrants PMO microbenchmarking now.
```

---

## 9. Completed vs remaining

| Category | Completed | Remaining / blocked |
|---|---|---|
| Plan execution | Implemented and ran approved v2 synthetic POC. | No PMO/GFN continuation authorized. |
| Engineering | Runner compiles and runs smoke/micro. | None required unless preserving/cleaning artifacts later. |
| Mechanism test | Measured OOD counterfactual MAE. | Failed required >=20% improvement. |
| Certificate test | Measured certified hit, false cert, coverage. | Certificate passed, but not enough for promotion. |
| Optimization test | Compared against GA/rank-GA/UCB/TS/correlational/filter baselines. | Failed: PCCWM lost to strong baselines. |
| Anti-bias test | Non-identifiable task caused PCCWM abstention and no false certificates. | Optimization poor; safety-only result insufficient. |
| Regression | Option-Flow POC and real catalog tests passed. | Broader package tests not run in this sprint. |

---

## 10. Regression tests

Both requested regressions passed after adding the runner.

```text
test/smiles_gflownet/test_option_flow_poc.jl
Option-Flow v0 POC | 25 pass / 25 total
```

```text
test/smiles_gflownet/test_option_flow_real_catalog.jl
Option-Flow real artifact catalog | 25 pass / 25 total
```

---

## 11. Artifacts

```text
test/smiles_gflownet/run_pccwm_core_sprint.jl
checkpoints/pccwm_core/run_smoke.log
checkpoints/pccwm_core/run_micro.log
checkpoints/pccwm_core/pccwm_smoke_results.jls
checkpoints/pccwm_core/pccwm_micro_results.jls
checkpoints/pccwm_core/pccwm_latest_results.jls
research/pccwm_core_closeout_2026-06-20.md
```

---

## 12. Recommendation

Do not proceed to PCCWM-PMO or PCCWM-GFN now.

The idea remains conceptually interesting only as a possible **certificate/safety layer** for some other future model, but the proposed core does not currently justify a new SOTA-facing algorithm family.

Current post-sprint state:

```text
Option-Flow: structural object signal, PMO mainline downgraded.
PTA-GFN: stopped.
AFK-SMC: stopped.
TREX-HE: stopped.
Portfolio: FEBB/EFR synthetic-only interesting.
PCCWM: certificate layer safe, core not promoted.
SOTA claim: blocked.
```

---

## 13. Bottom line

PCCWM did find a meaningful safety behavior:

```text
when evidence is insufficient, it abstains rather than issuing false certificates.
```

But the larger hypothesis failed:

```text
proof-carrying counterfactual world modeling did not outperform strong acquisition/search baselines and did not show a decisive mechanism advantage.
```

Therefore the correct decision is:

```text
PCCWM_STOP_OR_REDESIGN
```
