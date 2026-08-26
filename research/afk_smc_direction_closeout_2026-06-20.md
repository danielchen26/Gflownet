# AFK-SMC Direction Sprint Closeout — Core Failed Gate

**Authoritative user time:** Saturday, 2026-06-20 01:07 EDT  
**Workspace:** `/Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint`  
**Runner:** `test/smiles_gflownet/run_afk_smc_direction_sprint.jl`  
**Plan:** `/Users/tianchichen/Documents/GitHub/Gflownet/sessions/260314-misty-eddy/plans/statphys_afk_smc_direction_validation_plan_v2_reaudited_2026-06-20.md`  
**Result bundle:** `checkpoints/afk_smc_direction/afk_smc_micro_results.jls`

---

## 1. Objective

Validate the corrected Stage-0 object:

```text
AFK-SMC core validation
Adaptive Feynman-Kac Sequential Monte Carlo for PMO rare-event search
```

This was deliberately **not** called AFK-GFN, because no learned GFlowNet twisted proposal was used.

The scientific question was:

```text
Does adaptive Feynman-Kac SMC + genealogy/scaffold diversity guard beat strong GA/FK baselines under identical oracle budget?
```

---

## 2. Commands

### Smoke

```bash
cd /Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint
mkdir -p checkpoints/afk_smc_direction
JULIA_CONDAPKG_BACKEND=Null \
JULIA_PYTHONCALL_EXE=$PWD/.CondaPkg/.pixi/envs/default/bin/python \
AFK_MODE=smoke \
AFK_TASKS=qed \
AFK_SEEDS=17 \
AFK_BUDGET=128 \
AFK_POPULATION=32 \
AFK_CHILDREN=16 \
AFK_ARMS=uniform_population,elite_ga,rank_weighted_ga,fk_fixed_beta,fk_adaptive_no_diversity,afk_smc \
julia --project=. test/smiles_gflownet/run_afk_smc_direction_sprint.jl \
2>&1 | tee checkpoints/afk_smc_direction/run_smoke.log
```

### Micro

```bash
cd /Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint
mkdir -p checkpoints/afk_smc_direction
JULIA_CONDAPKG_BACKEND=Null \
JULIA_PYTHONCALL_EXE=$PWD/.CondaPkg/.pixi/envs/default/bin/python \
AFK_MODE=micro \
AFK_TASKS=qed,drd2,celecoxib_rediscovery \
AFK_SEEDS=17,23 \
AFK_BUDGET=300 \
AFK_POPULATION=48 \
AFK_CHILDREN=16 \
AFK_ARMS=uniform_population,elite_ga,rank_weighted_ga,fk_fixed_beta,fk_adaptive_no_diversity,afk_smc \
julia --project=. test/smiles_gflownet/run_afk_smc_direction_sprint.jl \
2>&1 | tee checkpoints/afk_smc_direction/run_micro.log
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

## 3. Arms

| Arm | Role |
|---|---|
| `uniform_population` | weak sanity baseline |
| `elite_ga` | strong greedy high-reward GA baseline |
| `rank_weighted_ga` | Genetic-GFN-like rank-weighted parent selection |
| `fk_fixed_beta` | FK resampling with fixed inverse-temperature increment |
| `fk_adaptive_no_diversity` | adaptive ESS FK without diversity guard |
| `afk_smc` | adaptive ESS FK + scaffold/genealogy diversity guard |

All arms used the same unique-valid-oracle budget and the same mutation/crossover proposal family.

---

## 4. Smoke result

Smoke ran all 6 arms on QED with budget 128 and seed 17.

| Arm | AUC top10 | Final top10 |
|---|---:|---:|
| `rank_weighted_ga` | 0.601147 | 0.675372 |
| `fk_adaptive_no_diversity` | 0.590071 | 0.631773 |
| `fk_fixed_beta` | 0.585458 | 0.634580 |
| `elite_ga` | 0.577758 | 0.631184 |
| `uniform_population` | 0.538691 | 0.601778 |
| `afk_smc` | 0.527683 | 0.624886 |

Smoke verdict:

```text
engineering passed; scientific signal already negative
```

---

## 5. Micro overall result

Tasks:

```text
qed, drd2, celecoxib_rediscovery
```

Seeds:

```text
17, 23
```

Budget:

```text
300 unique canonical valid oracle calls per task/seed/arm
```

Overall mean across tasks:

| Arm | Mean AUC top10 | Mean final top10 |
|---|---:|---:|
| `rank_weighted_ga` | **0.287177139** | 0.310399300 |
| `elite_ga` | 0.283997979 | **0.318480967** |
| `fk_fixed_beta` | 0.283744609 | 0.307634345 |
| `fk_adaptive_no_diversity` | 0.278649430 | 0.297904773 |
| `afk_smc` | 0.265726149 | 0.294685625 |
| `uniform_population` | 0.260994021 | 0.278381831 |

Gate output:

```text
AFK_SMC_STOP_OR_INCONCLUSIVE
```

Key gate deltas:

| Comparison | Relative AUC delta for `afk_smc` |
|---|---:|
| vs `rank_weighted_ga` | -7.47% |
| vs `elite_ga` | -6.43% |
| vs `fk_fixed_beta` | -6.35% |

Paired wins vs `rank_weighted_ga`:

```text
1 / 6
```

Severe collapse:

```text
false
```

Interpretation: diversity guard prevented catastrophic collapse, but did not improve optimization enough.

---

## 6. Per-task aggregate

| Task | Arm | AUC | Top10 | Max genealogy frac |
|---|---|---:|---:|---:|
| `qed` | `rank_weighted_ga` | **0.648493** | 0.701580 | 1.0000 |
| `qed` | `elite_ga` | 0.634736 | **0.719469** | 1.0000 |
| `qed` | `fk_fixed_beta` | 0.631038 | 0.680299 | 1.0000 |
| `qed` | `afk_smc` | 0.591456 | 0.654083 | 0.6875 |
| `drd2` | `elite_ga` | **0.021830** | **0.029448** | 0.8229 |
| `drd2` | `fk_adaptive_no_diversity` | 0.021401 | 0.028310 | 1.0000 |
| `drd2` | `fk_fixed_beta` | 0.020876 | 0.024940 | 1.0000 |
| `drd2` | `rank_weighted_ga` | 0.020310 | 0.025311 | 1.0000 |
| `drd2` | `afk_smc` | 0.015731 | 0.021196 | 0.5833 |
| `celecoxib_rediscovery` | `fk_fixed_beta` | **0.199320** | **0.217664** | 1.0000 |
| `celecoxib_rediscovery` | `elite_ga` | 0.195427 | 0.206526 | 0.6250 |
| `celecoxib_rediscovery` | `fk_adaptive_no_diversity` | 0.194685 | 0.210655 | 1.0000 |
| `celecoxib_rediscovery` | `rank_weighted_ga` | 0.192729 | 0.204306 | 1.0000 |
| `celecoxib_rediscovery` | `afk_smc` | 0.189992 | 0.208778 | 0.5417 |

AFK-SMC was more diverse than many exploitative baselines, but the extra diversity did not buy enough top-k AUC.

---

## 7. Theory verdict

```text
AFK-SMC core: failed current gate.
AFK-GFN Sprint 2: not warranted.
```

Reason:

1. `afk_smc` lost to `rank_weighted_ga`.
2. `afk_smc` lost to `elite_ga`.
3. `afk_smc` lost to `fk_fixed_beta`.
4. Adaptive ESS + diversity guard was not validated as a useful physics component.
5. The result is not simply collapse-related; the method preserved diversity but under-optimized.

Corrected interpretation:

```text
The first-principles Feynman-Kac/rare-event framing remains conceptually clean, but the tested AFK-SMC mechanism does not yet provide a stronger PMO optimizer than simple strong GA/FK baselines.
```

---

## 8. Novelty verdict

Safe claim:

```text
We implemented and fairly tested an AFK-SMC core hypothesis for PMO.
```

Unsafe claims:

```text
AFK-GFN works.
AFK-SMC is SOTA-facing.
Adaptive FK improves PMO.
The physics core passed.
```

The GFlowNet extension must remain blocked until a different core or ablation passes.

---

## 9. Testing / automation verdict

Regression tests passed:

| Test | Result |
|---|---:|
| `test/smiles_gflownet/test_option_flow_poc.jl` | 25 / 25 |
| `test/smiles_gflownet/test_option_flow_real_catalog.jl` | 25 / 25 |

No existing Option-Flow catalog tests were broken by the AFK runner.

---

## 10. Done vs remaining

| Done | Remaining |
|---|---|
| Re-audited original AFK-GFN plan and found inconsistencies | Do not promote AFK-SMC to AFK-GFN |
| Renamed Stage 0 to AFK-SMC core validation | Do not claim SOTA or GFN novelty |
| Implemented strict budget runner | Need broader direction search beyond selector/FK family |
| Ran smoke and micro | Need identify a more structurally different candidate |
| Compared against elite/rank/fixed/adaptive FK baselines | Need simulation POC for next candidate only after re-audited plan |
| Logged negative scientific verdict | Eventually return to GA_GFN app integration if theory search stalls |

---

## 11. Bottom line

```text
AFK-SMC is currently stopped or inconclusive-negative.
It preserved diversity but did not beat the strong exploitation baselines that matter for PMO AUC.
No AFK-GFN/TREGFN Sprint 2 should be run from this result.
```
