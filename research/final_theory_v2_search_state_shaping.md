> **Status: Historical predecessor.** Final Theory v2 / search-state shaping tested a weak shape-then-TB handoff bridge. The current canonical target is Option-Flow: `final_theory_option_flow_2026-06-11.md`.

# Final Theory v2 — Search-State Shaping

## One-sentence hypothesis
Under a fixed oracle budget, the right learned object is a **task-aware policy for constructing high-value search states**—mainly frontier and replay states—such that downstream pure-TB search performs better, even if final top-k attribution still lands on TB.

## The abstract objective
We are no longer optimizing only for the best edit decision now.
We are optimizing for the **future value of the search state** created by current interventions.

Let:
- $$B$$ = total oracle budget
- $$b_s$$ = shaping budget
- $$E$$ = downstream exploiter, initially pure TB
- $$S_{b_s}^{\pi}$$ = search state after applying shaping policy $$\pi$$ for the first $$b_s$$ calls

Then the objective is:

$$
J(\pi_{shape}) = \mathbb{E}\left[V_E\left(S_{b_s}^{\pi}, \tau, B-b_s\right)\right]
$$

where $$V_E(S, \tau, b)$$ is downstream task performance achieved by exploiter $$E$$ from state $$S$$ on task $$\tau$$ with remaining budget $$b$$.

## What counts as state
At Level 3, state includes:
- frontier contents
- replay contents
- scaffold/diversity coverage
- budget remaining
- task identity
- what TB will train on next

## What Level 3 is *not*
This theory does **not** require:
- direct HE top-k authorship
- heuristic rescue as the mainline story
- static interpolation as the key scientific object

## Causal test
### Shape-then-TB
Three arms:
1. **TB only**
2. **Heuristic shaping -> TB**
3. **Learned shaping -> TB**

The shaping phase spends a small fixed budget constructing frontier/replay state.
Then shaping is turned off and the same pure TB exploits the resulting state.

## Headline metric
**Downstream-only AUC** after the shaping phase.

## Required supporting metrics
- handoff frontier top-10 quality
- handoff replay top-10 quality
- downstream final top-10 / top-1
- frontier/replay overlap with handoff state (mediated credit proxy)
- handoff-to-final uplift

## Pass criteria
The theory stays alive if:
1. learned shaping beats TB-only on downstream-only AUC
2. learned shaping beats or matches heuristic shaping
3. handoff state quality is meaningfully better
4. no catastrophic collapse appears on the stress task
5. mediated credit is non-zero and interpretable

## Failure criteria
The current Level 3 formulation is weakened or rejected if:
1. downstream-only AUC does not improve over TB-only
2. learned shaping is not better than heuristic shaping
3. handoff frontier/replay state is not better
4. gains happen only during shaping, not downstream
5. mediated credit remains invisible

## Interpretation rule
A positive result means the Level 3 theory is actionable and worth scaling.
A negative result means we should revise the Level 3 object—not fall back to local patch rescue.