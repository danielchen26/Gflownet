# Current Context - GFlowNet.jl

**Last Updated:** 2026-06-11 EDT  
**Current canonical theory target:** Frontier-Conditioned Finite-Horizon Option-Flow GFlowNet  
**Current branch in main checkout:** `core-development`  
**Important status:** main checkout is dirty; do not assume a clean working tree.

---

## Current Final Theory Target

The project is now centered on:

> **Search-state improvement is the utility target; frontier-conditioned finite-horizon option-flow is the GFlowNet learned object.**

Core equation:

```math
P_\theta(\omega \mid S_t) \propto U(\omega; S_t)
```

Definitions:

- `S_t` = search state: frontier, replay, oracle history, top-k set, scaffold/family coverage, budget state, optional uncertainty/belief.
- `ω` = bounded option / short operational schema: routing context + basin + parent + schema/operator + primitive realization + commit/writeback.
- `U(ω; S_t)` = frontier/search-state improvement utility under fixed oracle budget.

The project should not drift into greedy `argmax U`, BO-style acquisition, or generic Active Belief-State RL. Proportional sampling over high-utility bounded options is the GFlowNet point.

---

## Canonical Documents

| File | Role |
|---|---|
| `research/final_theory_option_flow_2026-06-11.md` | Current short canonical theory spec |
| `research/cafe_gfn_novel_directions.md` | Long evolutionary theory record; latest authority is Part 21 |
| `research/DEVELOPMENT_LOG.md` | Canonical research ledger; latest relevant entry is Analysis #52 |
| `research/README.md` | Research document index |
| `MEMORY.md` | Compressed cross-session memory; Option-Flow section near top |

---

## Historical Tracks Now Superseded as Final Targets

These remain useful evidence, but are not the current final theory:

- local basin / parent / operator learned heads as final object;
- Level 2 learned edit controller as final object;
- Level 3 shape-then-TB as final object;
- sparse intervention gates as the primary final target;
- generic Active Belief-State GFlowNet naming;
- GA_GFN app integration as theory success.

---

## Evidence Debt

QGFN, Boosting, 10K multi-task PMO, and full 23-task PMO remain evidence-debt closures for the existing TB pipeline.

They should be tracked separately from the Option-Flow final theory target:

| Evidence debt | Status |
|---|---|
| QGFN + TB v10 | implemented, not fully end-to-end validated |
| Boosting | implemented, not fully benchmark validated |
| 10K multi-task PMO | QED done; broader task coverage incomplete |
| Full 23-task PMO | pending |

---

## Current Cleanup Status

- `research/README.md` created as canonical index.
- Current session old Option-Flow drafts and historical theory plans archived under `sessions/260314-misty-eddy/plans/archive/`.
- Cleanup manifest: `sessions/260314-misty-eddy/data/option_flow_repository_cleanup_manifest_2026-06-11.md`.
- Old `.claude/sessions/current_context.md` was preserved in `.claude/sessions/archive/current_context_pre_option_flow_2026-06-11.md`.

---

## Active Plans

| File | Status |
|---|---|
| `sessions/260314-misty-eddy/plans/option_flow_holistic_reposition_plan_zh_v4_final.md` | Approved/current Option-Flow reposition plan |
| `sessions/260314-misty-eddy/plans/ga_gfn_branch_plan.md` | Separate engineering/integration track |

---

## Next Recommended Work

1. Do not continue local patch loops unless they explicitly support option-level flow.
2. If doing theory work next, formalize Option-Flow v0:
   - action semantics: operator vs operator + selected child;
   - proposal-set semantics;
   - backward policy;
   - commit/writeback semantics;
   - frontier utility stability.
3. If doing evidence-debt closure next, treat QGFN / Boosting / PMO as a separate validation branch, not as the final theory.
4. If doing app/product work next, keep GA_GFN integration separate from theory success.

---

## Safety Constraints

- Do not delete checkpoints, benchmark artifacts, old session records, or research evidence without explicit confirmation.
- Do not switch branches in the shared main checkout while another session may be using it.
- Preserve negative results and historical theory shifts as evidence.
