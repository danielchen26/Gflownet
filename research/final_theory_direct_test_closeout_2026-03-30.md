> **Status: Historical closeout.** This closeout records the v1 direct test result. It is preserved for evidence; current target is Option-Flow.

# Final Theory Direct Test Closeout — 2026-03-30

## What was tested
A minimal direct comparison between:
- **Baseline:** pure TB
- **Candidate:** TB + task-aware learned edit control + conservative gate

### Test configuration
- **Budget:** 256
- **Tasks:** `qed`, `drd2`, `celecoxib_rediscovery`
- **Repeats:** 2
- **Offline candidate training:** RWMLE on HE artifacts with task conditioning enabled
- **Conservative gate:** heuristic interpolation `λ=0.5`, entropy floor `0.05`
- **Lean runtime harness:** per-step GA disabled to reduce confounds; replay retained; HE warmup/interleaving kept minimal

## Summary results

### Arm means
| Arm | Task | AUC top-10 | Top-10 | Top-1 |
|---|---|---:|---:|---:|
| TB | qed | 0.8960 | 0.9015 | 0.9371 |
| FinalTheory-v1 | qed | 0.8943 | 0.9077 | 0.9358 |
| TB | drd2 | 0.0438 | 0.0490 | 0.2595 |
| FinalTheory-v1 | drd2 | 0.0686 | 0.0720 | 0.3346 |
| TB | celecoxib_rediscovery | 0.2979 | 0.3057 | 1.0000 |
| FinalTheory-v1 | celecoxib_rediscovery | 0.6753 | 0.6753 | 1.0000 |

### Deltas: FinalTheory-v1 − TB
| Task | ΔAUC | ΔTop10 | ΔTop1 |
|---|---:|---:|---:|
| qed | -0.0017 | +0.0062 | -0.0013 |
| drd2 | +0.0249 | +0.0231 | +0.0750 |
| celecoxib_rediscovery | +0.3774 | +0.3696 | +0.0000 |
| **Mean** | **+0.1335** | **+0.1329** | — |

## Observability findings
- **HE top-10 provenance stayed at 0% on all tasks**
- Mean **parent override rate**: ~0.75
- Mean **operator override rate**: ~0.33
- Mean **override frontier-delta**: **+0.2667**
- HE produced positive frontier gain and active overrides, but the final top-k molecules were still attributed to the TB path after replay/frontier shaping

## Interpretation
This direct test produced a very important split result:

1. **System-level quality improved** on 2/3 tasks, especially `celecoxib_rediscovery`
2. **Direct HE provenance did not improve** — top molecules still did not come from the HE path itself

That means the candidate seems to be helping the search **indirectly**:
- by shaping frontier state
- by improving replay contents
- by changing what TB sees next

But it is **not yet validating** the stricter theory that the learned edit path itself becomes a direct top-k contributor.

## Theory verdict
### Strict Final Theory v1 verdict
**Currently falsified / not yet working under the pre-registered strict criterion.**

Why:
- the pre-registered provenance condition failed
- learned edit did not directly author top-k molecules

### Softer system-level verdict
**Promising but mis-specified.**

Why:
- whole-system AUC/Top10 improved materially
- learned control was active and locally productive
- but the gains appear to come through **frontier/replay shaping**, not direct HE terminal success

## What was done vs what remains
| Done | Remains |
|---|---|
| Wrote `Final Theory v1` note | Reformulate the object around frontier/replay shaping vs direct top-k edit authorship |
| Added task conditioning to the edit policy | Decide whether the next candidate should optimize direct edit terminal quality or memory-state shaping |
| Added runtime override/entropy observability | Add explicit attribution for replay-mediated gains so indirect edit credit is visible |
| Ran budget-256 direct test on 3 tasks × 2 repeats | Re-run after theory reformulation with a provenance-aware metric/credit path |
| Produced a hard pass/fail verdict | Scale only if the revised object survives another direct test |

## Automatic testing verdict
### Does automatic theory-conformance testing exist now?
**Partially yes.**

### Automatically tested in this milestone
- unit tests for task-conditioned edit-policy code
- end-to-end direct test runner:
  - budget 256
  - 3 tasks
  - 2 repeats
  - strict automatic verdict calculation
- automatic logging of:
  - AUC / top10 / top1
  - HE calls
  - parent/operator override rates
  - frontier-gain summary
  - top-k provenance

### Not yet automatically resolved
- whether replay-mediated gains should count as satisfying the final theory
- whether zero direct HE provenance means the object is wrong or the attribution path is wrong
- which revised object definition is the right next theory candidate

## Updated working hypothesis
The next theory candidate should probably test one of these two more explicit objects:

1. **Direct edit authorship theory**
   - learned edit should directly create top-k molecules
2. **Frontier/replay shaping theory**
   - learned edit's value is to improve the search state that TB learns from, even if final top-k attribution lands on TB

Right now, the evidence points more strongly toward **(2)** than **(1)**.