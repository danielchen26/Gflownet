# GFlowNet.jl Development Session State

Last Updated: January 2025

## Current Development Status

### Recently Completed Features

1. **DETAILED_BALANCE Implementation** ✅
   - Fixed Zygote mutation issues (push! → array comprehension)
   - Resolved flow caching during gradient computation
   - Created comprehensive tests and examples
   - Location: `src/core/balance.jl`, `src/core/interface.jl`

2. **FLOW_MATCHING Objective** ✅
   - Implemented flow_matching_loss minimizing (Z(s) - F(s))²
   - Integrated with training system
   - Flows treated as constants during backprop (Zygote.@ignore)
   - Location: `src/core/balance.jl`, tests in `test/objectives/flow_matching/`

3. **Multi-Start GFlowNets** ✅
   - `MultiStartGFlowNetModel` with per-initial-state Z values
   - P(s₀ⁱ) = Z(s₀ⁱ) / Σⱼ Z(s₀ʲ) for initial state selection
   - Full training integration
   - Location: `src/core/multi_start.jl`, `src/training/multi_start_training.jl`

### Current Architecture Understanding

#### Key Design Patterns
1. **On-demand computation**: No explicit DAG, compute as needed
2. **Zygote compatibility**: No mutations in differentiable code
3. **Type safety**: Consistent Float32 for neural networks
4. **Modular objectives**: TB, DB, FM all follow same pattern

#### Important Files
- `src/core/interface.jl` - Main training loop (needs reorganization)
- `src/core/balance.jl` - Loss functions for all objectives
- `src/training/configuration.jl` - Training config types
- `src/core/flows.jl` - Flow computation with memoization

### Known Issues and Solutions

1. **Zygote Mutations**
   - Problem: push! breaks gradients
   - Solution: Use array comprehensions
   ```julia
   # Bad: push!(arr, item)
   # Good: arr = [arr..., item] or comprehensions
   ```

2. **Flow Caching During Gradients**
   - Problem: Dict operations cause gradient errors
   - Solution: Wrap flow calls in Zygote.@ignore
   ```julia
   flow_value = Zygote.@ignore flow(model, state)
   ```

3. **Method Overwriting**
   - Problem: Duplicate logsumexp definitions
   - Solution: Import from one location

### Development Workflow Established

1. **Testing Pattern**
   ```bash
   julia --project=. test/category/subcategory/test_file.jl
   ```

2. **Git Commit Pattern**
   - Comprehensive commit messages
   - Technical details section
   - Test results confirmation

3. **File Organization**
   - Tests: Hierarchical in `test/`
   - Examples: Domain-specific in `examples/`
   - Core math in `src/core/`
   - Training in `src/training/` (needs consolidation)

## TODO List Status

### High Priority ✅
1. ✅ Update documentation for completed features
2. ✅ Create DETAILED_BALANCE vs TRAJECTORY_BALANCE example
3. ✅ Implement FLOW_MATCHING objective

### Medium Priority
4. ✅ Multi-start GFlowNets with per-initial-state Z
5. ⏳ **Reorganize training code** (move from core/interface.jl to training/)
6. ⏳ Add backward policy probability normalization validation
7. ⏳ Implement SUB_TRAJECTORY_BALANCE objective
8. ⏳ Add flow estimator network for DIRECT_FLOW method

### Low Priority
9. ⏳ GPU acceleration for trajectory sampling
10. ⏳ Debugging/visualization tools
11. ⏳ Variance reduction techniques

## Key Commands and Patterns

### Creating New Objectives
1. Add loss function to `src/core/balance.jl`
2. Add case to `compute_trajectory_loss` in `src/core/interface.jl`
3. Add enum value to `TrainingObjective` in `src/training/configuration.jl`
4. Create tests in `test/objectives/[name]/`
5. Create example in `examples/core_features/[name]/`

### Running Tests
```bash
# Single test file
julia --project=. test/objectives/flow_matching/test_flow_matching.jl

# All tests
julia --project=. test/runtests.jl
```

## Next Session Recommendations

1. **Training Code Reorganization** (Medium complexity)
   - Move train_gflownet, compute_trajectory_loss to training/
   - Clean separation of concerns
   - Update all imports

2. **SUB_TRAJECTORY_BALANCE** (High complexity)
   - New mathematical objective
   - Requires trajectory segment handling
   - Good next challenge after reorganization

3. **Backward Policy Validation** (Low complexity)
   - Add normalization checks
   - Ensure Σ P_B(s|s') = 1
   - Quick win for robustness

## Important Context

- **Grid World Types**: Use `GridState` not `GridWorldState`
- **Imports**: Many functions need explicit imports when creating new files
- **GPU**: Currently CPU-only, GPU acceleration is future work
- **Examples**: Each needs its own Project.toml
- **Documentation**: Update docs/src/internals/ when adding features

## Session Recovery Instructions

If starting a new session:
1. Read this file first
2. Check git status and recent commits
3. Run tests to ensure everything works
4. Continue with recommended next tasks
5. Update this file with new progress