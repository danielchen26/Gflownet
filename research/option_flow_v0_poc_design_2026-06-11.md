# Option-Flow v0 POC Design

**Date:** 2026-06-11  
**Status:** Prototype design for finite-catalog Option-Flow validation

---

## Goal

Validate the new philosophical object in the smallest clean setting:

```math
P_\theta(\omega \mid S_t) \propto U(\omega; S_t)
```

where:

- `S_t` is the current search state;
- `ω` is a bounded finite-horizon option / operational schema;
- `U(ω; S_t)` is frontier/search-state improvement utility.

---

## v0 simplification

Full online Option-Flow has a hard stochastic edit-transition issue:

```text
operator -> stochastic proposals -> selected child -> commit/writeback
```

Therefore v0 uses a finite realized option catalog:

```text
S_t -> choose one realized option ω_i -> terminal option outcome
```

For each frozen snapshot / catalog:

```math
\pi^*(\omega_i \mid S_t) = \frac{U_i}{\sum_j U_j}
```

This is a one-step finite GFlowNet / exact normalized catalog-flow problem. It does not claim to solve full stochastic edit-TB, but it directly tests whether the final object has learnable signal.

---

## Definitions

### Search state `S_t`

A compact feature vector for the frozen search state, including top reward, top-10 mean, frontier size, scaffold/family coverage, budget context, and task context.

### Option `ω`

A realized bounded episode or episode fragment containing routing context, basin/parent/operator/schema choices, realized children, commit/writeback outcomes, and utility deltas.

### Utility `U(ω; S_t)`

Primary v0 utility:

```text
U_primary = epsilon + max(0, sum(frontier_utility_delta over option steps))
```

All utility values are positive after the epsilon floor.

### Target distribution

For every catalog with candidates `{ω_i}`:

```math
\pi_i^* = U_i / \sum_j U_j
```

If all utilities are zero before epsilon, the catalog is marked low-information and can be excluded from headline metrics.

---

## Model

A small MLP scores each `(S_t, ω_i)` pair:

```text
score_i = fθ([state_features, option_features_i])
Pθ(ω_i | S_t) = softmax(score_i over candidates)
```

---

## Objective

Catalog cross-entropy:

```math
L = -\sum_i \pi_i^* \log P_\theta(\omega_i \mid S_t)
```

Diagnostics:

```math
r_i = \log P_\theta(\omega_i \mid S_t) - \log U_i + \log \sum_j U_j
```

At the target distribution, residuals approach zero.

---

## Success criteria

POC is worth continuing if held-out catalogs show:

- lower CE/KL than uniform;
- higher top-utility-quartile mass than uniform;
- non-collapsed entropy;
- positive rank correlation between predicted scores and option utility;
- signal on at least one structural task catalog.

---

## Non-goals

This POC does not:

- prove full classical TB over stochastic edit environments;
- run full PMO;
- claim SOTA;
- replace QGFN / Boosting evidence closure;
- modify app/UI.
