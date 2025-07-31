# GFlowNet.jl Test Suite

This directory contains the test suite for GFlowNet.jl.

## Test Suite Structure

### Active Test Files
- `runtests.jl` - Main test runner
- `test_utilities.jl` - Basic test utilities and helper types
- `test_neural_networks.jl` - Tests for Lux.jl and ComponentArrays integration
- `test_core_interface.jl` - Tests for the high-level GFlowNet interface
- `test_grid_world.jl` - Tests for the grid world application
- `test_training.jl` - Tests for training infrastructure
- `test_supply_chain.jl` - Tests for supply chain optimization
- `test_core_functions.jl` - Comprehensive tests for all exported functions (NEW)

### Running Tests

```bash
# Run all tests
julia --project=. -e "using Pkg; Pkg.test()"

# Run a specific test file
julia --project=. test/test_grid_world.jl

# Run with detailed output
julia --project=. test/runtests.jl
```

### Test Statistics (January 2025)
- **Total tests**: 311
- **Passing**: 304 (97.7%)
- **Failing**: 7 (2.3%)
- **Success rate improvement**: From 52% to 97.7%

## Critical Findings from Comprehensive Testing

### 1. Architecture Transition Issue
The codebase is in transition between two architectural approaches:
- **Old approach**: Explicit `DirectedAcyclicGraph` with `get_next_states()`
- **New approach**: On-demand computation with `get_applicable_actions()` + `compute_next_state()`

**Problem**: Many core functions still use the old approach while infrastructure uses the new approach.

### Why Examples Still Work Despite Missing Functions

**Key Insight**: The grid world and other examples work perfectly because they use `TRAJECTORY_BALANCE` objective, which doesn't need the broken functions. This explains why you can successfully run examples without encountering any of the missing function errors.

#### The Working Path (What Examples Use)

##### 1. Training Configuration
```julia
# Grid world example configuration
config = TrainingConfig(
    objective=TRAJECTORY_BALANCE,  # ✅ This is the key!
    partition_function_method=SIMPLE_ESTIMATION,  # Note: Not actually used
    n_iterations=50,
    batch_size=16
)
```

##### 2. What Trajectory Balance Actually Computes
Looking at `compute_single_trajectory_loss()` in `src/core/interface.jl`:
```julia
# Step 1: Sum log probabilities along trajectory
log_prob_sum = Σ log P_F(action_i | state_i)

# Step 2: Get terminal reward
log_reward = log(reward(terminal_state))

# Step 3: Compute loss (enforces P_F(τ) ∝ R(s_T))
loss = (log_prob_sum - log_reward)²
```

**Critical**: This computation is completely self-contained and only requires:
- `forward_probability()` - Computing P_F(a|s) ✅ Works
- `get_applicable_actions()` - Getting valid actions ✅ Works
- `apply_action()` - State transitions ✅ Works
- `reward()` - Terminal rewards ✅ Works

##### 3. What It Doesn't Need (The Broken Parts)
The trajectory balance loss does NOT need:
- `flow()` or `compute_recursive_flow()` - No flow computation required
- `get_next_states()` - Uses `get_applicable_actions()` instead
- `backward_policy` - Only forward policy needed
- `partition_function()` - Not used despite being in config
- DAG traversal - Everything computed on-demand

##### 4. How Sampling Works
`sample_trajectory()` is also self-contained:
```julia
function sample_trajectory(model)
    state = model.initial_state
    trajectory = [state]
    
    while !is_terminal_state(state)
        # Get applicable actions on-demand
        actions = get_applicable_actions(state, model.all_actions)
        
        # Sample from forward policy
        action = sample_forward_action(model.forward_policy, state, actions, ...)
        
        # Apply action to get next state
        state = apply_action(action, state)
        trajectory.append(state)
    end
    
    return trajectory
end
```

No DAG needed, no flow computation, just forward sampling!

#### The Broken Path (What Would Fail)

##### 1. Detailed Balance Objective
```julia
config = TrainingConfig(objective=DETAILED_BALANCE)  # ❌ This would fail
```
Why it fails:
- Calls `detailed_balance_loss()` in `src/core/balance.jl`
- Requires `model.backward_policy` - but model doesn't have this field
- Needs to compute P_B(s|s') - backward transition probabilities
- Would throw: `type GFlowNetModel has no field backward_policy`

##### 2. Flow Matching Objective
```julia
config = TrainingConfig(objective=FLOW_MATCHING)  # ❌ This would fail
```
Why it fails:
- Calls `flow_matching_loss()` in `src/core/balance.jl`
- Line 350: `next_states = get_next_states(model.dag, state)`
- Would throw: `UndefVarError: get_next_states not defined`

##### 3. Direct Flow Computation
```julia
Z = partition_function(model)  # ❌ Would fail
flow_val = flow(model, state)  # ❌ Would fail
```
Why they fail:
- `partition_function()` calls `get_root_state(model.dag)` - function doesn't exist
- `flow()` calls `get_next_states(model.dag, state)` - function doesn't exist
- Both would throw `UndefVarError`

#### Architecture Insight: Two Parallel Implementations

The codebase has two parallel implementations:

##### Old Architecture (Broken)
- Explicit DAG construction: `DirectedAcyclicGraph` objects
- Functions: `get_next_states()`, `get_previous_states()`, `get_root_state()`
- Used by: Flow computation, detailed balance, flow matching
- Status: Functions called but never implemented

##### New Architecture (Working)
- On-demand computation: No explicit DAG
- Functions: `get_applicable_actions()`, `apply_action()`, `compute_next_state()`
- Used by: Trajectory balance training, sampling
- Status: Fully implemented and working

#### Why This Matters

1. **For Users**: Stick to `TRAJECTORY_BALANCE` and everything works perfectly
2. **For Developers**: The core training loop has been modernized, but mathematical features haven't
3. **For the Future**: Need to migrate flow computation to on-demand architecture

#### Summary
Examples work because they use the subset of GFlowNet that has been successfully migrated to the new on-demand architecture. The trajectory balance objective is mathematically complete without needing flow computation, making it independent of all the broken DAG-based functions.

### 2. Critical Missing Functions
These functions are called throughout the codebase but **never defined**:
- `get_next_states()` - Used in 10+ locations
- `get_previous_states()` - Referenced in backward policy
- `get_root_state()` - Used in partition function

**Impact**: Core mathematical functions like `flow()`, `compute_recursive_flow()`, and balance losses fail at runtime.

### 3. Non-Existent Exported Functions
These are exported but don't exist:
- `trajectory_probability`, `log_trajectory_probability`
- `validate_trajectory` (replaced by `is_valid_trajectory`)
- `get_trajectory_summary`
- `validate_state_features`, `validate_training_config`

### 4. Broken Core Functions
Due to missing dependencies:
- `flow()` - Calls non-existent `get_next_states()`
- `compute_recursive_flow()` - Same issue
- `partition_function()` - Calls non-existent `get_root_state()`
- `detailed_balance_loss()` - Requires missing `backward_policy` field
- `flow_matching_loss()` - Uses non-existent DAG functions
- `validate_flow_conservation()` - Same issue

### 5. API Inconsistencies
- Model doesn't have `state_dim` or `backward_policy` fields
- Validation functions return `nothing` instead of `Bool`
- Optimizer name: `RMSprop` vs `RMSProp` mismatch

## Safe vs Unsafe Features Guide

### ✅ Safe to Use (Working Features)
These features work correctly and are used in examples:
- **Training Objectives**: `TRAJECTORY_BALANCE` only
- **Model Creation**: All `create_*_gflownet()` functions
- **Training**: `train_gflownet()` with trajectory balance
- **Sampling**: `sample_trajectory()`, `sample_trajectory_batch()`
- **Core Interface**: State/action interface methods
- **Analysis**: Domain-specific analysis functions

### ❌ Unsafe to Use (Broken Features)
These will cause runtime errors:
- **Training Objectives**: `DETAILED_BALANCE`, `FLOW_MATCHING`
- **Flow Functions**: `flow()`, `compute_recursive_flow()`, `partition_function()`
- **Validation**: `validate_flow_conservation()`
- **Backward Policy**: Any function expecting `model.backward_policy`
- **Missing Exports**: Functions listed in exports but not implemented

### 🔧 Usage Example (Safe Path)
```julia
# This works perfectly:
model = create_grid_world_gflownet(grid_size=5, hidden_dim=64)
config = TrainingConfig(
    objective=TRAJECTORY_BALANCE,  # ✅ Safe objective
    n_iterations=100,
    batch_size=32
)
history = train_gflownet(model, config)
trajectories = [sample_trajectory(model) for _ in 1:100]
```

## Legacy Code Recommendations

### Immediate Actions Needed

1. **Fix Critical Missing Functions** - Two options:
   ```julia
   # Option A: Implement missing functions using on-demand approach
   function get_next_states(state, all_actions)
       applicable_actions = get_applicable_actions(state, all_actions)
       return [compute_next_state(action, state) for action in applicable_actions]
   end
   
   # Option B: Replace all calls with on-demand computation directly
   ```

2. **Clean Up Exports**
   - Remove non-existent function exports from `src/GFlowNet.jl`
   - Or implement the missing functions if they're actually needed

3. **Fix Known Bugs**
   - Change `RMSprop` to `RMSProp` in configuration.jl:383
   - Add missing model fields or update functions that expect them

4. **Complete Architecture Migration**
   - Choose one approach (recommend on-demand) and stick to it
   - Update all core mathematical functions consistently

### Functions to Remove/Replace

Based on testing, these should be removed or completely rewritten:
- All functions expecting explicit DAG objects
- Functions calling non-existent `get_next_states()`
- Exports without implementations
- Legacy validation functions with wrong signatures

## Test Coverage Status

### Working Components ✅
- Basic state/action interface
- Forward policy functions
- Neural network integration (Lux.jl)
- Grid world implementation
- Supply chain implementation
- Training infrastructure (mostly)
- Model creation functions
- Sampling functions

### Broken Components ❌
- Flow computation (all variants)
- Balance loss computation (all variants)
- Backward policy (not implemented)
- Flow conservation validation
- Several exported validation functions

## Archived Tests

The `archive/old_tests/` directory contains tests from previous versions that reference outdated APIs:
- Tests using `DirectedAcyclicGraph` (replaced by on-demand computation)
- Tests using `SimpleState` (moved to test_utilities.jl)
- Tests with old training configurations
- Performance benchmarks that need updating

These archived tests demonstrate the old API and should not be used as examples.

## Adding New Tests

When adding tests:
1. **Verify functions exist** before writing tests
2. **Check actual signatures** in source code
3. **Use proper imports** to avoid ambiguity
4. **Test the actual API**, not expected behavior
5. **Skip broken functionality** with clear comments

Example structure:
```julia
@testset "Domain Tests" begin
    @testset "State Interface" begin
        # Test state_to_features, is_terminal_state, reward
    end
    
    @testset "Action Interface" begin
        # Test is_applicable, apply_action
    end
    
    @testset "Model Creation" begin
        # Test create_<domain>_gflownet function
    end
end
```

## Known Issues

1. **Zygote warnings**: `@ignore` deprecated, use `ChainRulesCore.ignore_derivatives`
2. **Supply chain**: 1 test fails consistently
3. **Training infrastructure**: 5 tests fail
4. **Optimizer ambiguity**: Name conflicts between modules

## Future Work

1. Complete migration to on-demand computation
2. Implement missing critical functions
3. Clean up exports to match implementation
4. Update documentation to reflect current architecture
5. Achieve 100% test pass rate
6. Remove all legacy code references

---

*Last updated: January 2025*
*Major update: Added comprehensive testing analysis and legacy code findings*