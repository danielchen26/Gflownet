# Quick Reference - GFlowNet.jl Development

## 🔥 Critical Rules

1. **NO MUTATIONS in differentiable code**
   ```julia
   # ❌ WRONG
   push!(array, item)
   state.value = new_value
   
   # ✅ CORRECT
   array = [array..., item]
   new_state = StateType(new_value, state.other_fields...)
   ```

2. **Wrap non-differentiable operations**
   ```julia
   # Discrete operations
   indices = Zygote.@ignore findfirst(...)
   
   # Flow computations during gradients
   flow_val = Zygote.@ignore flow(model, state)
   ```

3. **Always use Float32 for NN**
   ```julia
   return Float32[x, y, z]  # Not Float64!
   ```

## 📦 Current State

- **Branch**: `main` — the single integration branch. `core-development` was fully
  merged into `main` and then deleted; do not branch from it.
- **Training objectives**: `TRAJECTORY_BALANCE` and `MULTI_OBJECTIVE_TB` (MOGFN-PC)
  are the two that actually train. `FLOW_MATCHING`, `DETAILED_BALANCE`,
  `SUB_TRAJECTORY_BALANCE` and `DIRECT_FLOW_OBJECTIVE` exist as enum branches but
  their coded losses deviate from the published equations and do not converge —
  see the "Known issues" section of `CHANGELOG.md`.
- **Tests**: `Pkg.test()` runs without Python; chemistry assertions are opt-in via
  `GFLOWNET_TEST_RDKIT=true`. The suite is NOT fully green — see `CHANGELOG.md`.
- **Training reorganized**: all training code lives in `src/training/`.

## 🚀 Quick Commands

```bash
# Run specific test
julia --project=. test/objectives/flow_matching/test_flow_matching.jl

# Run example
cd examples/core_features/multi_start && julia --project=. multi_start_demo.jl

# Check what changed
git status
git diff --name-only
```

## 🎯 Key Locations

- **Model Creation**: `src/core/interface.jl` (create_gflownet, sampling)
- **Training Logic**: `src/training/` (training.jl, losses.jl, objectives.jl)
- **Loss Functions**: `src/core/balance.jl` (mathematical definitions)
- **Type Definitions**: `src/core/types.jl`
- **Flow Computation**: `src/core/flows.jl` (recursive & direct methods)
- **Examples**: `examples/core_features/`
- **Tests**: `test/` (hierarchical by objective/feature)

## ⚠️ Common Pitfalls

1. **GridState** not GridWorldState
2. **logsumexp** already defined - don't redefine
3. **time()** is in Base, not Dates
4. **Each example** needs its own Project.toml
5. **Test in proper folders** not root test/