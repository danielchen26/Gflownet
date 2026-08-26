# TREX-HE Temperature-Exchange Direction Closeout — Synthetic Gate Failed

**Authoritative user time:** Saturday, 2026-06-20 01:18 EDT  
**Workspace:** `/Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint`  
**Runner:** `test/smiles_gflownet/run_trex_temperature_exchange_pmo.jl`  
**Plan:** `/Users/tianchichen/Documents/GitHub/Gflownet/sessions/260314-misty-eddy/plans/statphys_ai_direction_rethink_trex_pmo_plan_v2_reaudited_2026-06-20.md`  
**Result bundle:** `checkpoints/trex_temperature_exchange/trex_synthetic_results.jls`  
**Schema:** `trex_he_v2_pre_result_exchange_sign_amended`

---

## 1. Objective

本轮目标不是证明 SOTA，也不是实现 GFlowNet，而是做一个 first-principles POC：

```text
TREX-HE core validation
= heuristic temperature-exchange population search for PMO-like rugged landscapes.
```

核心问题：

```text
温度分层 + reward-informed cross-temperature exchange 是否能比 single hot β、no-exchange ladder、elite/rank GA 更好？
```

---

## 2. Pre-result amendment

执行前发现 v2 plan 里的 exchange sign 有方向错误。已在任何 TREX 结果运行前修正为：

```text
log_alpha = (β_cold - β_hot) * (z_hot - z_cold)
```

这使得：

```text
热/低β副本发现高分样本 -> 更容易交换进冷/高β副本
```

该 runner 明确记录：

```text
heuristic score exchange, not exact MCMC
```

因此安全 claim 只能是 heuristic temperature-exchange search core，不是 exact replica-exchange MCMC。

---

## 3. Commands

### Compile smoke

```bash
cd /Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint
JULIA_CONDAPKG_BACKEND=Null \
JULIA_PYTHONCALL_EXE=$PWD/.CondaPkg/.pixi/envs/default/bin/python \
TREX_MODE=compile \
julia --project=. test/smiles_gflownet/run_trex_temperature_exchange_pmo.jl
```

### Synthetic mechanism diagnostic

```bash
cd /Users/tianchichen/Documents/GitHub/Gflownet-theory-sprint
mkdir -p checkpoints/trex_temperature_exchange
JULIA_CONDAPKG_BACKEND=Null \
JULIA_PYTHONCALL_EXE=$PWD/.CondaPkg/.pixi/envs/default/bin/python \
TREX_MODE=synthetic \
TREX_SYNTH_FAMILIES=funnel,deceptive_trap,multi_peak \
TREX_SEEDS=17,23 \
TREX_BUDGET=256 \
TREX_POPULATION=48 \
TREX_CHILDREN=24 \
TREX_ARMS=uniform_population,elite_ga,rank_weighted_ga,single_hot_beta,temperature_ladder_no_exchange,random_exchange_control,trex_exchange \
julia --project=. test/smiles_gflownet/run_trex_temperature_exchange_pmo.jl \
2>&1 | tee checkpoints/trex_temperature_exchange/run_synthetic.log
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

## 4. Implementation notes

Added isolated runner only:

```text
test/smiles_gflownet/run_trex_temperature_exchange_pmo.jl
```

Required arms:

| Arm | Role |
|---|---|
| `uniform_population` | weak sanity baseline |
| `elite_ga` | greedy GA baseline |
| `rank_weighted_ga` | Genetic-GFN-like rank baseline |
| `single_hot_beta` | one β=8 exploitative population |
| `temperature_ladder_no_exchange` | ladder without exchanges |
| `random_exchange_control` | ladder with random exchanges |
| `trex_exchange` | full heuristic temperature exchange |

Locked β ladder:

```text
[0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
```

Synthetic families:

```text
funnel
deceptive_trap
multi_peak
```

---

## 5. Bug fixes during execution

These were engineering bugs, not scientific failures:

| Bug | Symptom | Fix |
|---|---|---|
| `SubString{String}` mode | `run_suite(::SubString{String})` MethodError | Convert mode to `String(strip(...))` |
| `div` resolution | `UndefVarError: div not defined in local scope` | Use Julia integer division operator `÷` |
| aggregate type | `mean_or_nan(::Vector{Any})` MethodError | Relax helper functions to convert `Float64.(xs)` |

After these fixes, synthetic ran to completion and saved a valid bundle.

---

## 6. Synthetic overall result

Synthetic diagnostic pass:

```text
false
```

Pass families:

```text
[]
```

Fail families:

```text
funnel
deceptive_trap
multi_peak
```

Overall mean AUC/top10:

| Arm | Mean AUC | Mean top10 |
|---|---:|---:|
| `single_hot_beta` | **0.574376451** | **0.679842778** |
| `rank_weighted_ga` | 0.566484020 | 0.656982220 |
| `elite_ga` | 0.534456225 | 0.599495209 |
| `random_exchange_control` | 0.530040425 | 0.588124760 |
| `trex_exchange` | 0.528302204 | 0.584693050 |
| `temperature_ladder_no_exchange` | 0.515439028 | 0.552472889 |
| `uniform_population` | 0.476644886 | 0.492983024 |

Gate:

```text
TREX_HE_STOP_OR_INCONCLUSIVE
```

Primary relative result:

```text
trex_exchange vs best baseline = -8.02% relative AUC
paired wins vs best baseline = 0 / 6
```

Stop reasons from bundle:

```text
trex_exchange <= elite_ga
trex_exchange <= rank_weighted_ga
trex_exchange <= single_hot_beta
trex_exchange <= random_exchange_control
trex_exchange has severe scaffold/genealogy collapse in most rows
```

---

## 7. Per-family aggregate

| Family | Best arm | Best AUC | `trex_exchange` AUC | `trex_exchange` exchange acceptance |
|---|---|---:|---:|---:|
| `deceptive_trap` | `single_hot_beta` | 0.788829 | 0.785409 | 0.7273 |
| `funnel` | `rank_weighted_ga` | 0.569000 | 0.558074 | 0.7000 |
| `multi_peak` | `single_hot_beta` | 0.366126 | 0.241424 | 0.7636 |

The exchange acceptance rates were high, especially for `multi_peak`, but this did not translate into better AUC. In fact, random exchange was slightly better overall than reward-informed TREX:

```text
random_exchange_control AUC = 0.530040
trex_exchange AUC          = 0.528302
```

This directly fails the reward-informed exchange ablation.

---

## 8. Cost / hidden-effort diagnosis

The gate did **not** fail because of hidden proposal cost:

```text
hidden_cost_bad = false
```

Representative duplicate/filter rates:

| Family | Arm | Duplicate/filter rate |
|---|---|---:|
| `deceptive_trap` | `trex_exchange` | 0.1289 |
| `deceptive_trap` | `single_hot_beta` | 0.4060 |
| `funnel` | `trex_exchange` | 0.1211 |
| `funnel` | `single_hot_beta` | 0.4046 |
| `multi_peak` | `trex_exchange` | 0.0700 |
| `multi_peak` | `single_hot_beta` | 0.1199 |

So the negative result is not explained by TREX secretly spending more proposal effort. It simply did not optimize better.

---

## 9. Theory verdict

```text
TREX-HE core failed the synthetic mechanism gate.
```

Interpretation:

1. Temperature ladder alone was not enough.
2. Reward-informed exchange did not beat random exchange.
3. Single hot β and rank-weighted GA were stronger than TREX on the synthetic diagnostic.
4. The mechanism did not pass even before real PMO.

Therefore:

```text
Do not run PMO micro.
Do not promote to TREX-GFN.
```

---

## 10. Novelty verdict

Safe claim:

```text
We implemented and falsified a heuristic temperature-exchange core hypothesis under strict ablations.
```

Unsafe claims:

```text
TREX-GFN works.
Temperature exchange improves PMO.
Replica exchange beats GA.
Exact replica-exchange MCMC applies.
SOTA is plausible from this evidence.
```

The idea is currently stopped at Stage 0.

---

## 11. Testing / automation verdict

| Check | Result |
|---|---:|
| `TREX_MODE=compile` | pass |
| synthetic run after fixes | completed |
| `test/smiles_gflownet/test_option_flow_poc.jl` | 25 / 25 |
| `test/smiles_gflownet/test_option_flow_real_catalog.jl` | 25 / 25 |

No existing Option-Flow tests were broken.

---

## 12. Done vs remaining

| Done | Remaining |
|---|---|
| Re-audited TREX v1 and found overclaim/math issues | Do not run PMO micro for TREX-HE from this result |
| Wrote v2 with downgraded heuristic claim | Do not claim TREX-GFN or SOTA |
| Corrected exchange sign before results | Consider next structurally different candidate |
| Implemented isolated TREX runner | Likely next: free-energy branch-and-bound or active-information acquisition |
| Ran compile + synthetic | Preserve negative result and avoid retuning ladder post hoc |
| Ran regression tests | Eventually return to GA_GFN app integration if theory search stalls |

---

## 13. Bottom line

```text
TREX-HE failed early.
```

It failed not because of implementation/runtime blockage, but because the core ablation story did not hold:

```text
single_hot_beta > rank_weighted_ga > random_exchange_control > trex_exchange > no_exchange > uniform
```

Thus the temperature-exchange philosophical idea is not currently promising enough to justify PMO micro or TREX-GFN Sprint 2.
