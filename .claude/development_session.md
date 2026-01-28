# GFlowNet.jl Development Session State

Last Updated: January 28, 2026

## Update Timeline
- **January 2026**: Documentation audit and real training visualization plan revision
- **August 2025**: Master agent orchestrator system designed
- **January 2025**: Core training objectives completed (TB, DB, FM, STB, DIRECT_FLOW)

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

4. **Training Code Reorganization** ✅
   - Moved all training functions from `core/interface.jl` to `training/` folder
   - Created modular structure: training.jl, losses.jl, utils.jl
   - Moved objectives.jl from core/ to training/
   - Clean separation between core API and training implementation
   - All tests passing after reorganization

5. **Backward Policy Validation** ✅
   - `validate_backward_policy_normalization` - checks Σ P_B(s|s') = 1
   - `validate_backward_policy_consistency` - validates across trajectories
   - `monitor_backward_policy_learning` - tracks learning progress
   - Location: `src/core/policies.jl`

6. **Comprehensive Test Fixes After Reorganization** ✅
   - Fixed all import issues (compute_trajectory_loss, etc.)
   - Fixed type name issues (GridWorldState → GridState)
   - Fixed ForwardPolicy callable issue in validation.jl
   - Fixed optimizer name conflicts (GFlowNet.ADAM)
   - All critical tests now passing

7. **Documentation Updates** ✅
   - Created comprehensive module structure reference
   - Updated architecture documentation
   - Added import troubleshooting guide
   - Updated CLAUDE.md with fixes

8. **SUB_TRAJECTORY_BALANCE Implementation** ✅
   - Implemented sub-trajectory balance loss in `src/core/balance.jl`
   - Added support in training system (`src/training/losses.jl`)
   - Created domain-agnostic tests and examples
   - Key insight: Provides O(T²) learning signals vs O(T) for regular TB
   - Benefits: Better credit assignment, lower variance, faster convergence

9. **DIRECT_FLOW_OBJECTIVE Implementation** ✅
   - Added flow estimator network support to `create_gflownet` with `include_flow_estimator` parameter
   - Implemented `direct_flow_loss` in `src/core/balance.jl`
   - Added `compute_log_forward_probability` helper function
   - Integrated with training system in `src/training/losses.jl`
   - Renamed from DIRECT_FLOW to DIRECT_FLOW_OBJECTIVE to avoid conflict with FlowComputationMethod enum
   - Created tests in `test/objectives/direct_flow/`
   - Created example in `examples/core_features/direct_flow/`
   - Key insight: Uses neural network Z(s) to directly estimate flows instead of recursive computation

### Current Architecture Understanding

#### Key Design Patterns
1. **On-demand computation**: No explicit DAG, compute as needed
2. **Zygote compatibility**: No mutations in differentiable code
3. **Type safety**: Consistent Float32 for neural networks
4. **Modular objectives**: TB, DB, FM, STB all follow same pattern
5. **Domain-agnostic implementations**: Core algorithms work with any domain

#### Important Files
- `src/core/interface.jl` - Model creation and sampling only (reorganized)
- `src/core/balance.jl` - Mathematical loss definitions
- `src/training/training.jl` - Main training loop
- `src/training/losses.jl` - Loss computation functions
- `src/training/objectives.jl` - Objective configurations
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
5. ✅ **Reorganize training code** (moved to training/ folder)
6. ✅ Add backward policy probability normalization validation
7. ✅ Implement SUB_TRAJECTORY_BALANCE objective
8. ✅ Update all agent instruction files
9. ⏳ Add flow estimator network for DIRECT_FLOW method

### Low Priority
9. ⏳ GPU acceleration for trajectory sampling
10. ⏳ Debugging/visualization tools
11. ⏳ Variance reduction techniques

## Key Commands and Patterns

### Creating New Objectives
1. Add loss function to `src/core/balance.jl`
2. Add case to `compute_trajectory_loss` in `src/training/losses.jl`
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

1. **SUB_TRAJECTORY_BALANCE** (High complexity)
   - New mathematical objective
   - Requires trajectory segment handling
   - Next major feature to implement
   - All infrastructure now ready

2. **Flow Estimator Network for DIRECT_FLOW** (Medium complexity)
   - Alternative to recursive flow computation
   - Neural network directly estimates F(s)
   - Good performance optimization

3. **GPU Acceleration** (High complexity)
   - Accelerate trajectory sampling
   - Parallel batch processing
   - Significant performance gains

## Summary of Training Reorganization

The training code has been successfully reorganized with:
- Clean separation between core/ and training/
- All tests passing after comprehensive fixes
- Better modularity and maintainability
- Full backward compatibility maintained

Key functions moved:
- `train_gflownet()` → `src/training/training.jl`
- `compute_trajectory_loss()` → `src/training/losses.jl`
- `objectives.jl` → `src/training/objectives.jl`

### Recent Updates (August 2025 - January 2026)

10. **Master Agent Orchestrator System** ✅ (August 2025)
    - Designed comprehensive agent coordination system
    - Created master orchestrator agent template
    - System design documents in `.claude/system_design/`
    - **Status**: Template exists, coordination logic pending implementation

11. **Real Training Visualization Plan Revision** ✅ (January 2026, Commit 39697005)
    - Comprehensive plan validated against actual codebase
    - Fixed all API signature mismatches
    - Added missing utility functions (parse_objective, error tracking)
    - Corrected TrainingConfig construction
    - Added frontend integration details and test plan
    - Document location: `docs/src/internals/development_guides/real_training_visualization_plan.md`
    - **Status**: Plan complete, implementation pending

12. **Documentation Audit and Fixes** ✅ (January 2026)
    - Comprehensive audit of `.claude/` and `.cursor/rules/` folders
    - 39 issues identified (6 critical, 8 high priority, 25 medium/low)
    - Fixed outdated timestamps, wrong API references, missing feature docs
    - Updated all agent instruction files for consistency
    - **Status**: In progress

## Important Context

- **Grid World Types**: Use `GridState` not `GridWorldState`
- **Imports**: Many functions need explicit imports when creating new files
- **GPU**: Currently CPU-only, GPU acceleration is future work (Phase 1 roadmap)
- **Examples**: Each needs its own Project.toml
- **Documentation**: Update docs/src/internals/ when adding features
- **Visualization**: UI production-ready, backend integration planned (see real_training_visualization_plan.md)
- **Agent System**: Individual agents usable, master coordination logic not yet implemented

## Session Recovery Instructions

If starting a new session:
1. Read this file first
2. Check git status and recent commits
3. Run tests to ensure everything works
4. Continue with recommended next tasks
5. Update this file with new progress