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

- **Branch**: core-development
- **Just Completed**: DIRECT_FLOW_OBJECTIVE implementation
- **All Training Objectives**: TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING, SUB_TRAJECTORY_BALANCE, DIRECT_FLOW_OBJECTIVE
- **Training Reorganized**: All training code now in src/training/
- **Tests Pass**: All core functionality working
- **Next Up**: GPU acceleration, debugging tools, or variance reduction

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