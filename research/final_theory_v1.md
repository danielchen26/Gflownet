> **Status: Historical predecessor.** Final Theory v1 tested a learned generate+edit policy object. It is superseded as the current canonical target by `final_theory_option_flow_2026-06-11.md`. Preserve this file as evidence, not as the active theory.

# Final Theory v1

## One-sentence hypothesis
Under a fixed oracle budget, a **task-aware, end-to-end learned, conservatively gated generate+edit policy** can improve system-level molecular search quality over pure TB **without collapsing**, rather than merely increasing frontier diversity.

## The object we are trying to learn
Not a better standalone edit heuristic, and not a better local operator score.

The target object is a **joint search policy**:
- TB remains the backbone generator
- learned edit control decides **which basin, which parent, and which operator** to invoke
- the learned edit control is **task-aware**
- the learned edit control is **conservatively gated** by interpolation with the heuristic policy
- success is judged by **whole-system outcomes under the same total budget**

## What is explicitly *not* the theory
These are out of scope for the headline claim:
- heuristic HE as the mainline theory
- static interpolation variants such as `interpolated_050`
- diversity-only wins without top-quality gains
- rescue patches for a single failing task without system-level justification

## Direct comparison
- **Baseline:** pure TB under the matched PMO budget/protocol
- **Candidate:** TB + task-conditioned learned edit controller + conservative gate

## Minimal decisive test
- **Budget:** 256
- **Tasks:** `qed`, `drd2`, `celecoxib_rediscovery`
- **Repeats:** 2
- **Primary metrics:** AUC top-10, top-10 quality, top-10 provenance, edit-call usage, override efficiency, collapse indicators

## Pre-committed pass criteria
The theory stays alive if all of the following hold in the direct test:
1. Mean performance is **non-negative vs TB** on the headline metric
2. The stress task does **not** show catastrophic collapse
3. Learned edit path contributes non-trivially to top-k provenance
4. The candidate’s gains are not only diversity gains; they show up in quality metrics too
5. Learned overrides are at least directionally sensible under the new observability summary

## Pre-committed failure criteria
The candidate is weakened or rejected if one or more of the following dominate:
1. Mean performance is clearly negative vs TB
2. The stress task collapses materially
3. Learned edit contribution to top-k provenance remains effectively zero
4. Override activity mostly adds noise without frontier-quality gains
5. Results are unstable and uninterpretable across seeds/tasks

## Interpretation rule
A positive result here does **not** prove the whole final theory.
It only proves the theory is still **alive enough to scale up**.

A negative result here does **not** kill every future edit-based idea.
It does mean this specific **task-aware learned+gated generate/edit object** is not yet working under the current formulation.