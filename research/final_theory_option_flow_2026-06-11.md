# Final Theory: Frontier-Conditioned Option-Flow GFlowNet

**Date:** 2026-06-11  
**Status:** Canonical theory target after holistic reset  

---

## 1. One-sentence hypothesis

Under fixed oracle budget, molecular PMO should be modeled as a GFlowNet over **frontier-conditioned finite-horizon bounded options**, where options are sampled proportionally to their frontier-improvement utility.

```math
P_\theta(\omega \mid S_t) \propto U(\omega; S_t)
```

---

## 2. Final philosophical target

The goal is not only to learn a molecule generator. The goal is to learn a search process that improves the current search state under an expensive oracle budget.

The correct separation is:

| Layer | Role |
|---|---|
| Outer objective | improve search state under oracle budget |
| Utility | frontier/search-state improvement caused by an option |
| GFlowNet object | bounded option / short operational schema |
| Substrate | TB, HE, GA, replay, frontier, QGFN, Boosting |

---

## 3. What is the object?

The learned object is a bounded option `ω` conditioned on search state `S_t`.

A practical option may contain:

```text
ω = routing context + basin + parent + schema/operator + primitive realization + commit/writeback
```

The option is finite-horizon. It is long enough to capture continuation-sensitive value, but short enough to preserve interpretable local credit assignment.

---

## 4. What is not the object?

The final object is not:

- a primitive local operator policy;
- a separate basin / parent / operator head by itself;
- a heuristic HE controller;
- shape-then-blind-TB;
- a greedy value function;
- a Bayesian optimization acquisition function;
- app integration / GA_GFN product wiring;
- molecule-level terminal distribution alone.

---

## 5. Search state definition

`S_t` is the current search state, not just a partial molecule.

It may include:

- frontier;
- replay;
- oracle history;
- top-k set;
- scaffold / family coverage;
- budget remaining;
- provenance / lineage;
- optional surrogate uncertainty or belief.

---

## 6. Frontier utility definition

The outer utility should measure search-state improvement:

```math
\Delta V(S_t, \omega) = V(S_{t+1}) - V(S_t)
```

`V` may include:

- top-1 improvement;
- top-10 mean improvement;
- top-k entry contribution;
- diversity / scaffold novelty;
- family coverage;
- budgeted future-search value.

The exact `U(ω; S_t)` should be pre-registered for each experiment.

---

## 7. GFlowNet objective

The defining objective is proportional option sampling:

```math
P_\theta(\omega \mid S_t) \propto U(\omega; S_t)
```

The key point is not to greedily choose one best option. The policy should sample a diverse set of high-utility options.

A greedy ablation may be useful as a baseline, but it is not the final GFlowNet thesis.

---

## 8. Relationship to Direction C v2/v3

This theory continues the Direction C evolution:

- Direction C v2 introduced frontier-conditioned hierarchical edit-flow.
- Direction C v3 sharpened the object to finite-horizon option / subtrajectory flow.
- This reset makes the final distinction explicit: search-state improvement is the utility target, while bounded option-flow is the GFlowNet object.

---

## 9. Why previous tests were not final tests

### Level 2
Level 2 tested local controllers and edit heads. Those are possible internal factors of an option, not the final option-flow object.

### Level 3
Level 3 tested shaped frontier/replay state handed to blind downstream pure TB. That weak handoff failed, but it did not test state-conditioned option sampling.

---

## 10. Relationship to QGFN / Boosting evidence debt

QGFN, Boosting, 10K multi-task PMO, and full 23-task PMO remain important evidence debts for the current TB pipeline.

They should be closed later, but they do not change the final object-level conclusion:

> the strongest framework target is frontier-conditioned bounded option flow, not molecule-level terminal distribution alone.

---

## 11. Open theoretical issue: stochastic edit transitions

A true Option-Flow v0 must formalize stochastic edit generation before overclaiming classical TB.

Open questions:

| Question | Why it matters |
|---|---|
| Is the action only an operator, or operator plus selected child? | Defines the trajectory |
| Does proposal set enter state/action? | Determines whether stochastic proposals can be flow-modeled |
| What is the backward policy? | Required for TB-style objectives |
| Is commit/writeback terminal or transitional? | Determines reward ownership |
| Is frontier utility stable enough? | Determines trainability |

---

## 12. Minimal future direct test

Suggested arms:

1. TB-only baseline
2. heuristic HE substrate
3. Option-Flow v0 trained on logged bounded options
4. greedy value/ranker ablation
5. optional QGFN / Boosting baseline branch

Primary metrics:

- fixed-budget PMO AUC;
- final top-10;
- top-k provenance;
- diversity / scaffold coverage.

Theory diagnostics:

- sampled option frequency vs empirical utility bucket;
- high-utility option diversity;
- frontier utility delta;
- option family/scaffold coverage;
- option-level credit assignment;
- flow proportionality diagnostic.

---

## 13. Failure conditions

The theory is weakened if a properly formalized Option-Flow v0:

- cannot beat heuristic HE or TB-only under fixed budget;
- collapses into greedy / single-mode behavior;
- cannot preserve diversity of high-utility options;
- cannot show option-level utility calibration;
- cannot define stable rewards or legal trajectories under stochastic edit transitions.

---

## 14. Relation to GA_GFN / app integration

GA_GFN app integration remains useful product and workflow work. It is not the final theory object.

The theory target is not “expose GA_GFN in the app.” The theory target is:

> flow over frontier-conditioned bounded options that improve search state under oracle budget.
