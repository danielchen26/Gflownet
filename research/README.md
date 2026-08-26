# Research Document Index

**Current as of:** 2026-06-11 EDT  
**Canonical theory target:** Frontier-Conditioned Finite-Horizon Option-Flow GFlowNet

## Current canonical files

| File | Status | Purpose |
|---|---|---|
| `final_theory_option_flow_2026-06-11.md` | **Current canonical theory spec** | Short target document for future sessions |
| `cafe_gfn_novel_directions.md` | **Long-form evolutionary theory record** | Historical theory evolution; latest authority is **Part 21** |
| `DEVELOPMENT_LOG.md` | **Canonical research ledger** | Durable chronological record; latest relevant entry is **Analysis #52** |
| `molecular_generation_methods_2024_2026.md` | Background survey | Literature / benchmark context, not the current theory target |

## Current final target

The project is now centered on:

> **Search-state improvement is the utility target; frontier-conditioned finite-horizon option-flow is the GFlowNet learned object.**

Core equation:

```math
P_\theta(\omega \mid S_t) \propto U(\omega; S_t)
```

where:

- `S_t` = current search state: frontier, replay, oracle history, top-k set, scaffold coverage, budget state;
- `ω` = bounded option / short operational schema;
- `U(ω; S_t)` = frontier/search-state improvement utility.

## Historical / superseded theory tracks

Older plans and session notes about local basin/parent/operator heads, Level 2 learned edit, Level 3 shape-then-TB, sparse intervention gates, and static shape-then-TB are preserved as historical evidence. They should **not** be used as the current theory target unless explicitly reframed under Option-Flow.

## Evidence debt, not final theory

QGFN, Boosting, 10K multi-task PMO, and full 23-task PMO remain useful evidence-closure work for the existing TB pipeline. They do **not** replace the Option-Flow final target.

## Cleanup rule

Do not delete checkpoints, benchmark artifacts, or historical session records without explicit confirmation. Prefer adding status notices and indexes over deletion.
