> **Status: Historical closeout.** This records the Level 3 shape-then-TB test. It is preserved as evidence that weak handoff shaping was insufficient, not as a final theory target.

# Level 3 Shape-then-TB Closeout — 2026-03-30

## What was tested
A causal Level 3 experiment designed to test the **search-state-shaping** theory directly.

### Theory object
A task-aware shaping policy is valuable if it creates a frontier/replay state that makes later **pure TB** perform better under the same total budget.

### Arms
1. **TB only**
2. **Heuristic shaping -> TB**
3. **Learned shaping -> TB**

### Budget split
- **Total budget:** 256
- **Shaping budget:** 64
- **Downstream TB budget:** 192
- **Tasks:** `qed`, `drd2`, `celecoxib_rediscovery`
- **Repeats:** 2

### Primary metric
- **Downstream-only AUC** during the post-shaping TB phase

### Supporting metrics
- final top-10 / top-1 after the full budget
- handoff-to-final frontier gain
- top-10 overlap with handoff frontier/replay state (mediated credit proxy)

## Summary results

### Downstream-only AUC
| Arm | qed | drd2 | celecoxib_rediscovery | Mean |
|---|---:|---:|---:|---:|
| TB only | 0.8826 | 0.0403 | 0.3174 | 0.4134 |
| Heuristic shape -> TB | 0.8703 | 0.0386 | 0.2347 | 0.3812 |
| Learned shape -> TB | 0.8593 | 0.0193 | 0.2430 | 0.3739 |

### Final top-10 after full budget
| Arm | qed | drd2 | celecoxib_rediscovery |
|---|---:|---:|---:|
| TB only | 0.9020 | 0.0541 | 0.3317 |
| Heuristic shape -> TB | 0.9264 | 0.4294 | 0.8385 |
| Learned shape -> TB | 0.8908 | 0.0525 | 0.8187 |

### Key deltas
#### Learned shape -> TB minus TB only
- `qed`: ΔDownstreamAUC = **-0.0233**
- `drd2`: ΔDownstreamAUC = **-0.0211**
- `celecoxib_rediscovery`: ΔDownstreamAUC = **-0.0744**
- **Mean ΔDownstreamAUC = -0.0396**

#### Learned shape -> TB minus heuristic shape -> TB
- `qed`: **-0.0109**
- `drd2`: **-0.0193**
- `celecoxib_rediscovery`: **+0.0082**
- **Mean ΔDownstreamAUC = -0.0073**

## Interpretation
This experiment gave a clean and useful answer.

### What succeeded
- The Level 3 runner worked as intended.
- The shaping phases created strong **handoff-state retention**:
  - learned frontier overlap ≈ **0.75**
  - learned replay overlap ≈ **0.75**
- Structural full-budget top-10 quality after shaping was often much higher than TB-only.

### What failed
The actual Level 3 headline claim failed:

> **the shaped state did not make downstream pure-TB better**

The downstream-only AUC was worse than TB-only on all three tasks for the learned shaping arm.

### What this means
The learned/heuristic shaping phases are good at **injecting strong molecules into handoff state**, but the current downstream TB phase does **not exploit that state better** than simply running TB end-to-end.

So the current Level 3 hypothesis, as written, is **not working**.

## Theory verdict
### Strict Level 3 verdict
**CURRENTLY FALSIFIED**

Reason:
- mean learned downstream AUC vs TB-only = **-0.0396**
- learned shaping does not beat heuristic shaping on average either

### Softer interpretation
This does **not** mean shaping is useless.
It means:
- shaping can create a high-value handoff state
- but the current downstream TB exploiter does not convert that handoff state into better downstream sample efficiency

So the likely mismatch is now between:
- **state construction**, and
- **state exploitation**

rather than simply between:
- good theory, and
- bad shaping policy

## What was done vs what remains
| Done | Remains |
|---|---|
| Defined Final Theory v2 around search-state shaping | Decide whether to abandon Level 3 or reformulate the downstream exploiter / handoff contract |
| Built a Shape-then-TB causal runner | Determine whether the failure is in shaping, exploitation, or the metric interface |
| Added downstream-only AUC metric | Add sharper exploitation-aware diagnostics if pursuing this line |
| Added mediated-credit proxy via handoff overlap | Decide whether preserving strong handoff state without downstream AUC gain has any scientific value |
| Ran 3 tasks × 2 repeats | Choose whether to pivot away from Level 3 or redesign the exploitation phase |

## Automatic testing verdict
### Does automatic Level 3 testing now exist?
**Yes, partially.**

### Automatically tested here
- three-arm Shape-then-TB experiment
- downstream-only AUC
- final top-10 / top-1
- handoff-to-final frontier gain
- handoff overlap proxies for frontier/replay
- automatic verdict calculation

### Still not automatically resolved
- whether the right failure interpretation is:
  - shaping is wrong,
  - downstream TB is the wrong exploiter,
  - or the handoff contract is wrong
- whether a stronger replay-conditioned exploiter would recover the shaped-state value
- full ancestry-based mediated credit beyond overlap proxies

## Best current interpretation
The Level 3 abstraction was worth testing, and the test was scientifically useful.

But the answer is:

> **state shaping alone, followed by current pure TB, is not yet an actionable winning theory**

The experiment suggests a new bottleneck:
- the system can **construct** useful state,
- but the current downstream learner may not **use** that state properly.