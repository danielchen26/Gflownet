# GFlowNet Core Mathematics Repair Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the implementation actually satisfy the defining GFlowNet theorem — terminal states sampled with `p(x) = R(x)/Z` — and make every remaining objective either mathematically faithful or explicitly disabled, never silently wrong.

**Architecture:** Build the analytic oracle first, because every fix below is accepted or rejected by it. Then repair in dependency order: `P_B` must become a probability distribution before the TB loss can legitimately reference it; the TB loss must be correct before flow semantics matter; flow semantics must be correct before DB/FM can be repaired.

**Tech Stack:** Julia 1.11, Lux + Zygote + ComponentArrays + Optimisers, exact enumeration on a 3×3 grid world as ground truth.

## Global Constraints

- The defining theorem is `p(x) = R(x) / Z` with `Z = Σ_x R(x)`. Any change is judged against it, not against loss curves. A falling loss on a wrong objective is worthless.
- **Measured baseline, established by adversarial verification (this is what we are fixing):**
  - The coded TB loss reaches machine zero (`4.109e-33`) at a policy whose terminal law is `p(x) = n(x)R(x)/Σ n(y)R(y)`, verified to `1.11e-16`, where `n(x)` is the number of paths to `x`.
  - On the 3×3 grid with peak `(3,3)=10.0`: `Z_true = 19.0`, `Z_coded = Σ n(x)R(x) = 78.0`, ratio `4.105`. Per-state `p_opt/target` ranges `0.2436` (n=1) to `1.4615` (n=6), exactly `0.2436·n(x)`.
  - Component gradient norms (0 = that component can never learn):
    | objective | forward | backward | flow | log_Z |
    |---|---|---|---|---|
    | TRAJECTORY_BALANCE | 14.42 | 0 | 0 | 5.399 |
    | DETAILED_BALANCE | 1.672 | 0.5954 | **0** | **0** |
    | FLOW_MATCHING | **0.004535** | 0 | 14.39 | **0** |
    | SUB_TRAJECTORY_BALANCE | 6.233 | 0 | 312.3 | **0** |
    | DIRECT_FLOW_OBJECTIVE | throws during compilation | | | |
  - `Σ_{parents} P_B` measured `1.1967 … 1.2922` on 4 multi-parent states (must be `1.0`); single-parent states give `P_B ∈ [0.51, 0.68]` (must be `1.0`).
  - `flow(model, s0) = 2.218362209547` equals `E_{P_F}[R]` to `0.000e+00`, versus `Z_true = 19.0`.
  - SubTB loss is `9.990423551228066` for both `R(3,3)=10` and `R(3,3)=1000` — difference exactly `0.0`.
  - Autodiff is SOUND: Zygote vs central differences, 16 coordinates, max relative error `7.2e-09`. Do not go looking for an autodiff bug; there isn't one.
- Every task must leave `julia --project=. -e 'using GFlowNet'` working.
- Never weaken a test to make it pass. If a fix cannot make the oracle green, report the number and stop.
- Commit after every task, then push. Conventional Commits.
- Do not touch `src/utils/visualization/` — the app is working and out of scope here.

## Decisions already made

| ID | Decision | Rationale |
|----|----------|-----------|
| E1 | `P_B` becomes a **softmax over the enumerated parent set**, not a per-edge sigmoid | A GFlowNet requires `Σ_{s∈parents(s')} P_B(s|s') = 1`. Per-edge sigmoids cannot satisfy that except by coincidence. |
| E2 | Fix TB properly; for DB/FM/SubTB either make faithful or **hard-error** | They are currently silently wrong, which is worse than unavailable. An explicit `error("not implemented faithfully")` is a strict improvement. |
| E3 | `DIRECT_FLOW_OBJECTIVE` is **removed** from the enum | It is not a published objective, it throws during compilation, and it duplicates TB with a learned `Z(s0)`. |
| E4 | `log_Z` gets its **own optimiser state** with a plain higher learning rate | Adam is invariant to gradient rescaling, so `z_learning_rate_multiplier` is provably a no-op (verified: scales 1×/10×/100× all give `log_Z = 0.2` after 200 steps). |
| E5 | The analytic oracle is the acceptance gate for the whole plan | Without it we are back to `validate_flow_consistency` returning `1.0`. |

---

## File Structure

Created:
- `test/theory/test_reward_proportionality.jl` — the oracle. Enumerates the 3×3 grid exactly, computes the analytic zero of the *current* TB loss, and asserts the resulting terminal law is reward-proportional. Red now, green when Task 5 lands.
- `test/theory/test_gradient_coverage.jl` — asserts no objective has a dead gradient component it claims to use.
- `test/theory/test_backward_normalization.jl` — asserts `Σ_parents P_B = 1`.
- `test/theory/enumerate.jl` — shared exact-enumeration helpers (parents, path counts `n(x)`, `Z_true`, terminal law of a given policy). One implementation, used by all three tests.

Modified:
- `src/core/policies.jl` — `P_B` becomes a parent-set softmax; `validate_backward_policy_normalization` fixed to actually compute the sum.
- `src/core/flows.jl` — `compute_recursive_flow` becomes a sum; `validate_flow_consistency` gets a real implementation.
- `src/training/losses.jl` — TB gains `Σ log P_B`; FM moves to log-space with gradients attached; DB un-detaches flow; SubTB gains the `F(x)=R(x)` boundary; `DIRECT_FLOW` branch deleted.
- `src/training/training.jl` — `log_Z` gets its own optimiser; gradient clipping actually applied.
- `src/core/balance.jl` — `validate_balance_conditions` implements the DB and FM checks instead of returning `nothing`.
- `src/training/configuration.jl` — `DIRECT_FLOW_OBJECTIVE` removed from the enum.

---

## Phase A — Build the oracle (must be RED before anything is fixed)

### Task 1: Exact enumeration helpers

**Files:**
- Create: `test/theory/enumerate.jl`

**Interfaces:**
- Produces, all for `GridState` on an `n×n` grid: `enumerate_states(n)`, `parents_of(state)`, `path_counts(n) -> Dict{Tuple,Int}` (number of distinct paths from `(1,1)`), `exact_Z(n, rewards)`, `terminal_law(model)` (exact terminal distribution of the model's current policy by full trajectory enumeration, not sampling). Consumed by Tasks 2, 3, 5.

- [ ] **Step 1: Write the helpers**

Grid moves are `MoveRight` (`x+1`) and `MoveUp` (`y+1`) plus `Terminate`, so the DAG is a lattice: `parents_of((x,y))` is `{(x-1,y), (x,y-1)}` intersected with the grid, and `n(x,y) = binomial(x+y-2, x-1)`. Verify the closed form against a dynamic-programming count rather than trusting it.

```julia
# test/theory/enumerate.jl
using GFlowNet
const GS = GFlowNet.GridState

"""Number of distinct (1,1) -> (x,y) paths, by DP. The lattice closed form is
binomial(x+y-2, x-1); we compute it by DP and assert agreement."""
function path_counts(n::Int)
    u = Dict{Tuple{Int,Int},Int}()
    for x in 1:n, y in 1:n
        u[(x,y)] = (x == 1 && y == 1) ? 1 :
            get(u, (x-1,y), 0) + get(u, (x,y-1), 0)
    end
    for x in 1:n, y in 1:n
        @assert u[(x,y)] == binomial(x + y - 2, x - 1) "path count mismatch at ($x,$y)"
    end
    return u
end

"""Exact Z = sum over terminal states of R(x). Requires GRID_CONFIG to be set."""
exact_Z(n::Int) = sum(GFlowNet.reward(GS(x, y, true)) for x in 1:n, y in 1:n)

parents_of(s) = [p for p in (GS(s.x-1, s.y, false), GS(s.x, s.y-1, false))
                 if p.x >= 1 && p.y >= 1]
```

`terminal_law(model)` must enumerate every complete trajectory and accumulate `∏ P_F` into its terminal cell. Assert the returned probabilities sum to `1.0` within `1e-10`; that self-check is what makes the oracle trustworthy.

- [ ] **Step 2: Verify the helpers against the numbers already measured**

Run a script asserting, for `n=3` with `reward_positions = Dict((3,3)=>10.0)`:
`exact_Z(3) == 19.0`, `path_counts(3)[(3,3)] == 6`, `sum_x n(x)*R(x) == 78.0`.
Expected: all three hold. These are the figures the adversarial verification produced, so disagreement means the helper is wrong, not the baseline.

- [ ] **Step 3: Commit**

```bash
git add test/theory/enumerate.jl
git commit -m "test: exact enumeration helpers for the 3x3 grid ground truth"
```

### Task 2: The reward-proportionality oracle — must FAIL now

**Files:**
- Create: `test/theory/test_reward_proportionality.jl`

- [ ] **Step 1: Write the failing test**

The test does not train. Training is slow and noisy; instead it constructs the *analytic optimum of the currently-coded loss* and asks whether that optimum is reward-proportional. That is the sharpest possible statement and it runs in seconds.

The coded TB loss `(log Z + Σ log P_F − log R)²` is zeroed by the policy
`P_F(s→s') ∝ u(s')` where `u` is the per-path mass `u(s) = Σ_{paths to s} 1` weighted by downstream reward. Concretely, define `w(s) = R(s)` if terminal else `0`, and the backward-recursive mass `m(s) = w(s) + Σ_{s' children} m(s')`; then the zero-loss policy is `P_F(s→s') = m(s')/m(s)` and `P_T(s) = w(s)/m(s)`. Build it, confirm the coded loss is ~0 there, then measure the terminal law.

```julia
using Test
include(joinpath(@__DIR__, "enumerate.jl"))

@testset "GFlowNet defining theorem: p(x) proportional to R(x)" begin
    n = 3
    GFlowNet.GRID_CONFIG[] = (grid_size = n, reward_positions = Dict((3,3) => 10.0))
    Z_true = exact_Z(n)
    @test Z_true ≈ 19.0

    law = analytic_optimum_terminal_law(n)      # from enumerate.jl
    @test sum(values(law)) ≈ 1.0 atol=1e-10

    ratios = Dict(k => law[k] / (GFlowNet.reward(GS(k[1], k[2], true)) / Z_true)
                  for k in keys(law))
    worst = maximum(abs(r - 1) for r in values(ratios))
    @info "reward-proportionality" worst_deviation=worst ratios=ratios
    # The defining theorem. Currently violated: ratios span 0.2436 to 1.4615.
    @test worst < 0.05
end
```

- [ ] **Step 2: Run it and confirm it FAILS with the expected numbers**

Run: `julia --project=. -e 'using Test; using GFlowNet; include("test/theory/test_reward_proportionality.jl")'`
Expected: FAIL, and the `@info` must report ratios spanning approximately `0.2436` to `1.4615` with `worst ≈ 0.46`. If the numbers differ materially from the recorded baseline, the test is wrong — fix the test before touching any source.

- [ ] **Step 3: Add the path-count correlation assertion**

Strengthen it: assert the *current* (broken) behaviour is exactly the path-count bias, so the test documents the mechanism and not just the symptom.

```julia
    u = path_counts(n)
    Z_coded = sum(u[k] * GFlowNet.reward(GS(k[1], k[2], true)) for k in keys(law))
    @test Z_coded ≈ 78.0
    # Mechanism: the coded optimum is n(x)R(x)/Z_coded, not R(x)/Z_true.
    # This assertion is expected to HOLD now and to BREAK once P_B is added --
    # at which point delete it, since the bias it documents will be gone.
    for k in keys(law)
        @test law[k] ≈ u[k] * GFlowNet.reward(GS(k[1], k[2], true)) / Z_coded atol=1e-9
    end
```

- [ ] **Step 4: Commit the red test**

```bash
git add test/theory/test_reward_proportionality.jl
git commit -m "test: add failing oracle for the GFlowNet defining theorem

Asserts the analytic optimum of the coded TB loss is reward-proportional.
It is not: ratios span 0.2436 (n=1) to 1.4615 (n=6), exactly 0.2436*n(x).
This test is the acceptance gate for the core repair."
```

### Task 3: Gradient-coverage and P_B-normalisation tests — both must FAIL now

**Files:**
- Create: `test/theory/test_gradient_coverage.jl`, `test/theory/test_backward_normalization.jl`

- [ ] **Step 1: Write the gradient-coverage test**

For each objective, assert the components it *claims* to use have non-zero gradient. Use the measured table in Global Constraints as the expected-failure record.

```julia
# expected usage per objective
REQUIRED = Dict(
  GFlowNet.TRAJECTORY_BALANCE      => (:forward, :log_Z),
  GFlowNet.DETAILED_BALANCE        => (:forward, :backward, :flow),
  GFlowNet.FLOW_MATCHING           => (:forward, :flow),
  GFlowNet.SUB_TRAJECTORY_BALANCE  => (:forward, :flow),
)
```
For each, compute `Zygote.gradient` of `compute_trajectory_loss` and assert every required component has norm `> 1e-6`. Extract `log_Z` with a scalar branch — `getproperty(g, :log_Z)` is a `Number`, and calling `vec` on it throws.

Expected failures now: `DETAILED_BALANCE`/`:flow` = 0, `FLOW_MATCHING`/`:forward` = 0.004535 (below any sane threshold; set the threshold to `1e-2` so this registers as the failure it is), `SUB_TRAJECTORY_BALANCE` will pass this test but is caught by Task 4's reward test.

- [ ] **Step 2: Write the SubTB reward-sensitivity test**

```julia
@testset "every objective responds to reward" begin
    for obj in (GFlowNet.TRAJECTORY_BALANCE, GFlowNet.DETAILED_BALANCE,
                GFlowNet.FLOW_MATCHING, GFlowNet.SUB_TRAJECTORY_BALANCE)
        cfg = TrainingConfig(objective = obj, n_iterations = 1, batch_size = 8)
        GFlowNet.GRID_CONFIG[] = (grid_size = 3, reward_positions = Dict((3,3) => 10.0))
        l1 = GFlowNet.compute_trajectory_loss(model, trajs, model.parameters, cfg)
        GFlowNet.GRID_CONFIG[] = (grid_size = 3, reward_positions = Dict((3,3) => 1000.0))
        l2 = GFlowNet.compute_trajectory_loss(model, trajs, model.parameters, cfg)
        @info "reward sensitivity" objective=obj l1=l1 l2=l2 delta=abs(l1-l2)
        @test abs(l1 - l2) > 1e-6   # a task-blind objective cannot be a GFlowNet objective
    end
end
```
Expected: `SUB_TRAJECTORY_BALANCE` fails with `delta` exactly `0.0` (baseline: both `9.990423551228066`).

- [ ] **Step 3: Write the P_B normalisation test**

For every state with ≥2 parents on a 3×3 grid, assert `Σ_{parents} P_B = 1` within `1e-6`; also assert single-parent states give `P_B = 1`.
Expected failure now: sums `1.1967 … 1.2922`, single-parent `0.51 … 0.68`.

- [ ] **Step 4: Run all three, record the failures, commit**

```bash
git add test/theory/
git commit -m "test: add failing gradient-coverage, reward-sensitivity and P_B tests"
```

---

## Phase B — Make `P_B` a probability distribution

### Task 4: `P_B` becomes a softmax over the enumerated parent set

`src/core/policies.jl:485-514` returns `clamp(sigmoid(logit), 1e-8, 1-1e-8)` for one `(parent, child)` edge. Independent sigmoids cannot sum to 1 over a parent set. This must be fixed *before* TB references `P_B`, otherwise Task 5 would add a term that is not a log-probability.

**Files:**
- Modify: `src/core/policies.jl` (`compute_backward_probability`, `validate_backward_policy_normalization`)
- Modify: `src/core/interface.jl` if the backward network's output shape must change

**Interfaces:**
- Produces: `compute_backward_probability(policy, parent, child, params, states, all_actions) -> Float64` satisfying `Σ_{p ∈ parents(child)} P_B(p|child) = 1`. Consumed by Task 5 (TB), the DB branch, and TLM.

- [ ] **Step 1: Confirm the failure**

Run the Task 3 P_B test. Expected: FAIL with sums in `[1.1967, 1.2922]`.

- [ ] **Step 2: Add a parent enumerator to the domain interface**

The core cannot softmax over parents without knowing the parent set. There is already `is_valid_backward_transition` (`policies.jl:517+`) and `find_parent_for_action`. Add a generic `get_parent_states(model, child)` that returns all valid parents, with the grid-world method in `src/applications/grid_world.jl`. Read `is_valid_backward_transition` first and reuse its validity rule so the two cannot disagree.

- [ ] **Step 3: Normalise over parents**

Rewrite so the scalar logit is computed for every parent of the child and then softmaxed:

```julia
function compute_backward_probability(policy::BackwardPolicy, target_state, source_state,
                                      parameters, states, all_actions)
    is_valid_backward_transition(source_state, target_state, all_actions) || return 0.0
    parents = get_parent_states(source_state, all_actions)
    isempty(parents) && return 0.0
    length(parents) == 1 && return 1.0        # unique parent: P_B must be exactly 1
    logits = map(parents) do p
        joint = vcat(state_to_features(source_state), state_to_features(p))
        first(safe_model_call(policy.model, joint, parameters, states)[1])
    end
    idx = findfirst(p -> p == target_state, parents)
    idx === nothing && return 0.0
    mx = maximum(logits)
    ex = exp.(logits .- mx)
    return Float64(ex[idx] / sum(ex))
end
```

This must stay Zygote-differentiable — no mutation, no `Zygote.@ignore` around the logits. Note `length(parents) == 1 -> 1.0` is not a shortcut, it is the correct value, and it fixes the measured `P_B ∈ [0.51, 0.68]` on single-parent states.

- [ ] **Step 4: Fix the broken validator**

`validate_backward_policy_normalization` currently reports `total_prob = 0.0` while simultaneously finding 2 parents — it is itself broken and would have masked this bug. Make it sum the actual probabilities and return the deviation.

- [ ] **Step 5: Verify**

Run the Task 3 P_B test. Expected: PASS, all multi-parent sums `1.0 ± 1e-6`, all single-parent `P_B = 1.0`.
Also re-run the gradient-coverage test: `DETAILED_BALANCE`/`:backward` must still be non-zero, i.e. the softmax kept differentiability.

- [ ] **Step 6: Commit**

```bash
git add src/core/policies.jl src/applications/grid_world.jl src/core/interface.jl
git commit -m "fix: P_B is now a distribution over parents, not a per-edge sigmoid

Measured before: sum over parents 1.1967-1.2922 on 4 multi-parent states,
and 0.51-0.68 on single-parent states where it must be exactly 1.
After: 1.0 within 1e-6 everywhere."
```

---

## Phase C — Make Trajectory Balance correct

### Task 5: Add `Σ log P_B` to the TB loss — the oracle must turn GREEN

This is the root cause. `src/training/losses.jl:484-486` computes `(log_Z + Σ log P_F − log R)²`, omitting `− Σ log P_B`, which makes the optimum `p(x) ∝ n(x)R(x)`.

**Files:**
- Modify: `src/training/losses.jl` (`compute_single_trajectory_loss`, and the MOGFN variant at `:884`)

- [ ] **Step 1: Confirm the oracle is red**

Run Task 2's test. Expected: FAIL, worst deviation ≈ `0.46`, ratios `0.2436 … 1.4615`.

- [ ] **Step 2: Add the backward term**

In `compute_single_trajectory_loss`, accumulate `log P_B` alongside `log P_F` over the same transitions and subtract it:

```julia
    loss = (log_Z + log_prob_sum - log_backward_sum - log(terminal_reward))^2
```

Where `log_backward_sum` sums `log P_B(s_{i-1} | s_i)` over the trajectory. When the model has no backward policy, `P_B ≡ 1` and the term is `0` — that is the current behaviour, so it must remain the explicit fallback rather than an accident. Emit a one-time `@warn` when TB runs without a backward policy on a non-tree DAG, because that is exactly the silent-bias case.

- [ ] **Step 3: Verify against the oracle**

Run Task 2's test. Expected: PASS, worst deviation `< 0.05`.
The path-count assertion added in Task 2 Step 3 must now FAIL — delete it and note in the commit that its removal is the proof the bias is gone.

- [ ] **Step 4: Verify the analytic optimum now uses the true Z**

Extend the check: at the corrected optimum, the `log_Z` that zeroes the loss must be `log(Z_true) = log(19.0) = 2.9497`, not `log(78.0) = 4.3567`. Assert within `1e-6`.

- [ ] **Step 5: Verify TB still trains end to end**

Train 500 iterations on the 3×3 grid with `LEARNABLE_ESTIMATION`, sample 4000 trajectories, and report TV distance to `R(x)/Z`. Record the number even if it is not yet small — Task 6 addresses the `log_Z` rate.

- [ ] **Step 6: Commit**

```bash
git add src/training/losses.jl test/theory/test_reward_proportionality.jl
git commit -m "fix: add the missing sum log P_B term to the TB loss

The coded loss omitted it, making the optimum p(x) proportional to
n(x)R(x) instead of R(x) -- verified to 1.11e-16 on the 3x3 grid, with
per-state ratios exactly 0.2436*n(x) and Z_coded=78 versus Z_true=19.
Oracle now green: worst deviation from reward-proportionality < 0.05."
```

### Task 6: Give `log_Z` its own optimiser

`z_learning_rate_multiplier` scales the gradient and then Adam normalises the scale away — verified: gradient scales 1×, 10× and 100× all yield `log_Z = 0.2` after 200 Adam steps, i.e. exactly `lr` per step. Reaching `log Z ≈ 20` therefore needs ~20,000 iterations.

**Files:**
- Modify: `src/training/training.jl` (optimiser setup and `train_step!`)

- [ ] **Step 1: Reproduce the no-op**

Run the three-way comparison inside the real training loop with the multiplier at 1.0, 10.0, 100.0 for 200 iterations and report final `log_Z`. Expected: all three approximately equal.

- [ ] **Step 2: Split the optimiser**

Set up a separate optimiser state for `log_Z` with its own learning rate (`learning_rate * z_learning_rate_multiplier`) rather than scaling its gradient, and drop `scale_z_gradient` from the update path. Keep the function for now but stop calling it, and note in its docstring that gradient scaling does not work under Adam.

- [ ] **Step 3: Initialise `log_Z` sensibly**

At construction under `LEARNABLE_ESTIMATION`, `log_Z = 0.0` starts ~20 nats from where it must be. Initialise it from a short unrolled estimate: sample a handful of trajectories and set `log_Z ≈ mean(log R) − mean(Σ log P_F)`. Guard against empty samples.

- [ ] **Step 4: Verify**

Repeat Step 1. Expected: the three multipliers now give visibly different `log_Z`, and with the multiplier at 10.0 the 500-iteration `log_Z` is within a few nats of `log(19) = 2.9497`. Report the TV distance from Task 5 Step 5 again — it should now drop materially.

- [ ] **Step 5: Commit**

---

## Phase D — Repair flow semantics and the remaining objectives

### Task 7: `flow()` must be unnormalised mass, not an expectation

`compute_recursive_flow` (`src/core/flows.jl:143-192`) computes `F(s) = Σ_{s'} P_F(s'|s) F(s')`. Since `Σ P_F = 1` this is a convex combination, so `F(s0) ∈ [min R, max R]` and can never equal `Z` once more than one terminal is rewarded. Measured: `flow(model,s0) = 2.2184` equals `E_{P_F}[R]` to `0.000e+00`, versus `Z_true = 19.0`.

**Files:**
- Modify: `src/core/flows.jl` (`compute_recursive_flow`, `partition_function`, `validate_flow_consistency`)

- [ ] **Step 1: Write the failing assertion**

Add to `test/theory/`: on the 3×3 grid, `partition_function(model)` must equal `exact_Z(3) = 19.0` within `1e-6`. Expected: FAIL at `2.2184`.

- [ ] **Step 2: Decide the semantics explicitly and document it**

Two coherent options; pick and write it in the docstring:
(a) `F(s) = Σ_{s' children} F(s') + R(s)·[s terminal-capable]` — unnormalised mass, `F(s0) = Σ_x n(x)R(x)` on a DAG with multiple paths;
(b) the true `Z = Σ_x R(x)` requires dividing by path multiplicity, i.e. the `P_B`-weighted recursion `F(s) = Σ_{s'} F(s')·P_B(s|s')`.
Option (b) is the one consistent with the corrected TB loss. Read Task 5's outcome before choosing, and state the choice in the commit.

- [ ] **Step 3: Implement, verify against `exact_Z`, commit**

Expected: `partition_function(model) ≈ 19.0`. Note this changes `flow()` for every consumer, so re-run the whole `test/theory/` suite plus `test/core/` afterwards and report any movement.

### Task 8: Detailed Balance — attach the flow gradient

Measured: DB's `:flow` and `:log_Z` gradient norms are exactly `0`. `losses.jl:152-153` wraps both flow terms in `Zygote.@ignore`, so the DB residual can only be closed by moving the policy against a frozen target.

- [ ] **Step 1** Confirm `:flow` norm is 0 via the Task 3 test.
- [ ] **Step 2** Replace the `Zygote.@ignore`'d `flow(model, ·)` calls with the differentiable learned flow estimator (`flow_estimate(model.flow_estimator, s, params.flow, ...)`), which is what DB's `F` is supposed to be. Require `include_flow_estimator`; error with a clear message if absent, rather than silently using a detached value.
- [ ] **Step 3** Verify `:flow` norm `> 1e-6` and that the DB reward-sensitivity test passes.
- [ ] **Step 4** Commit.

### Task 9: Flow Matching — log-space, edge flows, terminal term, attached policy

Measured: FM's `:forward` gradient norm is `0.004535` versus TB's `14.42` — that residual is only the entropy term, because `losses.jl:215-223` wraps the policy in `Zygote.@ignore`. The loss also compares raw flows rather than log-sum-exp, omits `R(s)` on the out-flow side, and uses state rather than edge flows.

- [ ] **Step 1** Confirm `:forward` ≈ `0.0045` and `:log_Z` = 0.
- [ ] **Step 2** Reimplement per Bengio et al. 2021: for each interior state,
  `(log Σ_{(s'',a'')→s} F(s'',a'') − log[R(s) + Σ_{a'} F(s,a')])²`, computed with `logsumexp`, with the policy inside the gradient. This needs edge flows; if the flow estimator only produces state flows, say so explicitly and either extend it to take `(state, action)` or hard-error the objective per decision E2 rather than shipping the state-flow approximation.
- [ ] **Step 3** Verify `:forward` norm is now comparable to TB's, and reward-sensitivity passes.
- [ ] **Step 4** Commit.

### Task 10: Sub-Trajectory Balance — apply the terminal boundary

Measured: the SubTB loss is `9.990423551228066` for `R(3,3)=10` and for `R(3,3)=1000` — difference exactly `0.0`. Reward never enters, because `balance.jl:265-273` takes the raw flow-net output even when `s_j` is terminal instead of `R(s_j)`.

- [ ] **Step 1** Confirm `delta == 0.0` via the Task 3 reward-sensitivity test.
- [ ] **Step 2** Apply `F(x) = R(x)` when the sub-trajectory endpoint is terminal, and add the missing `Σ log P_B` over the sub-trajectory now that Task 4 makes `P_B` a distribution. Also reconcile the parameterisation clash: `flow_estimate` exponentiates the flow head (`policies.jl:356`) while SubTB applies `log(max(·, 1e-8))` to it (`balance.jl:268`) — the same network cannot be both a flow and a log-flow. Pick one and fix the other call site.
- [ ] **Step 3** Verify `delta > 1e-6` and that a 100× reward change moves the loss.
- [ ] **Step 4** Commit.

### Task 11: Remove `DIRECT_FLOW_OBJECTIVE`

It throws during compilation, `losses.jl:293` calls `direct_flow_loss_batch(model, trajectories)` without `params`, and internally it reads `model.parameters` and `Zygote.@ignore`s `log Z`, so it is a constant with respect to the differentiation variable. It is not a published objective.

- [ ] **Step 1** Confirm the exception.
- [ ] **Step 2** Remove it from the `@enum TrainingObjective`, delete the dispatcher branch and `direct_flow_loss*`, and delete or unwire `test/objectives/direct_flow/test_direct_flow.jl` (currently unwired anyway, 17 pass / 2 fail). Also remove the dead `COMBINED_OBJECTIVES` enum entry, which has no dispatcher branch and throws `ArgumentError` if selected.
- [ ] **Step 3** Verify `using GFlowNet` loads and the export test passes.
- [ ] **Step 4** Commit.

---

## Phase E — Close the verification holes

### Task 12: Make the validators actually validate, and apply gradient clipping

Every bug above survived because the checks return passing values: `validate_flow_consistency` (`flows.jl:502-506`) `@warn`s and returns `1.0`; `validate_balance_conditions` (`balance.jl:885-891`) leaves DB and FM as `nothing`; `validate_backward_policy_normalization` reported `total_prob = 0.0` while finding 2 parents. Separately, `gradient_clip_norm` is configured (`configuration.jl:222`) but `train_step!` never calls `clip_gradients!` (`objectives.jl:521`) — observed gradient norm 43.9.

- [ ] **Step 1** Implement `validate_flow_consistency` as a real in-flow/out-flow residual, returning the max relative residual, and make it error or warn loudly above a threshold. Implement the DB and FM branches of `validate_balance_conditions`.
- [ ] **Step 2** Call `clip_gradients!` in `train_step!` when `gradient_clip_norm` is finite. Note the `option-flow-development` branch fixed this too but duplicated the block; do it once.
- [ ] **Step 3** Wire the whole `test/theory/` directory into `test/runtests.jl` as a new group so the oracle runs in CI.
- [ ] **Step 4** Verify all of `test/theory/` is green, then run the full suite once and compare against the recorded baseline (1076 pass / 6 fail / 19 error before this plan; the flow-estimator and segfault fixes have since changed it — re-measure).
- [ ] **Step 5** Commit, and update `CHANGELOG.md` Known issues: the flow-matching convergence and `log Z` entries are resolved by Tasks 5-9 and must be moved out of Known issues.

---

## Self-Review

**Coverage.** Every proven defect maps to a task: missing `Σ log P_B` → Task 5; `P_B` unnormalised → Task 4; `flow()` = `E[R]` → Task 7; DB flow detached → Task 8; FM policy detached + raw space → Task 9; SubTB reward-blind → Task 10; DIRECT_FLOW inert → Task 11; `log_Z` Adam no-op → Task 6; validators returning passes → Task 12; gradient clipping dead → Task 12.

**Ordering.** Task 4 strictly precedes Task 5 (a `Σ log P_B` term is only meaningful once `P_B` is a distribution). Task 5 precedes Task 7 (the corrected TB determines which flow semantics is consistent). Tasks 8-10 depend on Task 7. Task 1 precedes everything because it is the acceptance gate.

**Falsifiability.** Each fix has a numeric before/after drawn from the adversarial verification, not an opinion. Task 2's path-count assertion is deliberately designed to *break* when Task 5 lands — a test whose failure is the proof of a fix.

**Known risk.** Task 9 may reveal that the flow estimator produces only state flows, making a faithful edge-flow FM impossible without extending the network. Decision E2 covers that: hard-error rather than ship an approximation labelled as Bengio et al. 2021.
