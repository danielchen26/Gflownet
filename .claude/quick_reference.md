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
- **Just Completed**: FLOW_MATCHING, Multi-Start GFlowNets
- **Tests Pass**: All core functionality working
- **Next Up**: Training reorganization or SUB_TRAJECTORY_BALANCE

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

- **Training Logic**: `src/core/interface.jl` (needs move to training/)
- **Loss Functions**: `src/core/balance.jl`
- **Type Definitions**: `src/core/types.jl`
- **Examples**: `examples/core_features/`
- **Tests**: `test/` (hierarchical structure)

## ⚠️ Common Pitfalls

1. **GridState** not GridWorldState
2. **logsumexp** already defined - don't redefine
3. **time()** is in Base, not Dates
4. **Each example** needs its own Project.toml
5. **Test in proper folders** not root test/