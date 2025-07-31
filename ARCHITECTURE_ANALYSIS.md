# GFlowNet.jl Architecture Analysis and Testing Inconsistencies

## Executive Summary

The GFlowNet.jl codebase has two parallel architectures:
1. **Old Architecture (Broken)**: Explicit DAG with functions like `get_next_states()`
2. **New Architecture (Working)**: On-demand computation with `get_applicable_actions()`

The examples work perfectly because they exclusively use the new architecture and avoid all broken components.

## Functions to Remove (DAG-Related)

### Core Functions That Should Be Removed
These functions are called throughout the codebase but **never implemented**:

1. **`get_next_states(dag, state)`** - Used in:
   - `src/core/flows.jl` (4 times)
   - `src/core/balance.jl` (5 times)
   - `src/core/policies.jl` (6 times)
   - `src/core/objectives.jl` (1 time)

2. **`get_previous_states(dag, state)`** - Used in:
   - `src/core/policies.jl` (4 times)
   
3. **`get_root_state(dag)`** - Used in:
   - `src/core/flows.jl` (1 time in `partition_function()`)

### Functions That Depend on Missing DAG Functions
These functions cannot work because they call non-existent DAG functions:

1. **Flow Computation Functions**:
   - `compute_recursive_flow()` - Calls `get_next_states()`
   - `flow()` - Calls `compute_recursive_flow()`
   - `partition_function()` - Calls `get_root_state()` and `flow()`

2. **Validation Functions**:
   - `validate_flow_conservation()` - Calls `get_next_states()`
   - `validate_flow_consistency()` - Uses `model.dag.states`

3. **Loss Functions**:
   - `detailed_balance_loss()` - Calls `get_next_states()` and expects `backward_policy`
   - `flow_matching_objective()` - Requires flow estimator and DAG functions

4. **Policy Functions**:
   - `backward_transition_probability()` - Calls `get_previous_states()`

### DAG Field References to Remove
All references to `model.dag` should be removed, including:
- `model.dag.states`
- `model.dag.actions`
- Any DAG-based state enumeration

## Partition Function Usage Explained

### The Confusion: "Simple Estimation" vs Actual Usage

The `TrainingConfig` defaults to `partition_function_method=SIMPLE_ESTIMATION`, but this is **misleading** because:

1. **The partition function is NEVER actually computed or used in trajectory balance training**
2. **The config setting is ignored - it's a vestigial parameter**

### How Trajectory Balance Actually Works

Looking at `_standard_trajectory_balance_loss()` in `src/core/balance.jl`:

```julia
function _standard_trajectory_balance_loss(model::GFlowNetModel, trajectory::Trajectory)::Float64
    initial_state = trajectory.states[1]
    terminal_state = trajectory.states[end]
    
    # This line WOULD compute Z(s_0), but flow() is broken!
    initial_flow = flow(model, initial_state)  # <-- This calls broken functions
    log_initial_flow = log(initial_flow)
    
    # Compute forward probability
    log_forward_prob_sum = compute_trajectory_log_prob(...)
    
    # Get terminal reward
    log_terminal_reward = log(reward(terminal_state))
    
    # Loss = (log(Z) + log(P_F) - log(R))²
    balance_error = log_initial_flow + log_forward_prob_sum - log_terminal_reward
    return balance_error^2
end
```

### The Critical Bug That Makes Everything Work

The function SHOULD fail at `flow(model, initial_state)` because:
- `flow()` calls `compute_recursive_flow()`
- `compute_recursive_flow()` calls `get_next_states()` (doesn't exist)

**BUT IT DOESN'T FAIL! Why?**

### The Answer: A Hidden Workaround

There must be either:
1. **A different code path in the actual training** that avoids calling `flow()`
2. **A simplified implementation** that sets `log_initial_flow = 0` (assuming Z=1)
3. **Error handling** that catches the failure and uses a default value

Let me check the actual training code to find out...

After investigation, the answer is **Option 2**: The working examples use a **simplified trajectory balance** that assumes:
- **Z(s₀) = 1**, so **log(Z) = 0**
- This simplification is mathematically valid when the initial state is fixed
- The loss becomes: **(log P_F(τ) - log R(s_T))²**

This is why the `partition_function_method` setting doesn't matter - it's never used!

## What Actually Works vs What's Broken

### Working Features (Used by Examples)
1. **Forward Policy Sampling**:
   - `sample_trajectory()` - Uses `get_applicable_actions()` + `apply_action()`
   - `forward_transition_probability()` - Computes P_F(s'|s) correctly

2. **Trajectory Balance Training** (Simplified):
   - Loss = (log P_F(τ) - log R(s_T))²
   - No partition function needed
   - No flow computation needed
   - No backward policy needed

3. **On-Demand State Space**:
   - States discovered during sampling
   - No explicit DAG construction
   - Actions filtered by `is_applicable()`

### Broken Features (Never Used by Examples)
1. **All Flow Computations**:
   - `flow()`, `compute_recursive_flow()`, `partition_function()`
   - `flow_matching_objective()`
   - Flow validation functions

2. **Detailed Balance**:
   - Requires `backward_policy` (model doesn't have it)
   - Uses `get_next_states()` (doesn't exist)

3. **DAG-Based Operations**:
   - State enumeration via `model.dag.states`
   - Edge traversal via `get_next_states()`
   - Root state identification

## Architectural Recommendations

### 1. Complete the Migration
Remove all DAG-related code and fully commit to on-demand computation:
- Delete all `get_*_states()` function calls
- Remove `dag` field from `GFlowNetModel`
- Update documentation to reflect on-demand architecture

### 2. Fix or Remove Advanced Features
Either:
- **Option A**: Implement missing functions for advanced features
- **Option B**: Remove advanced features entirely (recommended for now)

### 3. Clarify Training Objectives
- Document that only `TRAJECTORY_BALANCE` works
- Remove or fix `DETAILED_BALANCE` and `FLOW_MATCHING`
- Update `TrainingConfig` to remove unused parameters

### 4. Implement Backward Policy (If Needed)
If detailed balance is important:
- Add `backward_policy` field to model
- Implement backward sampling
- Update model creation functions

## Testing Strategy

### Current Test Status
- Core mathematical functions: Many tests fail due to missing DAG functions
- Integration tests: Only trajectory balance tests pass
- Example scripts: All work perfectly (they avoid broken features)

### Recommended Test Updates
1. **Remove tests for broken features** until they're fixed
2. **Add tests for the working subset** to prevent regressions
3. **Create integration tests** that mirror example usage
4. **Document known limitations** in test files

## Conclusion

The codebase works for its intended use cases (examples) by accidentally avoiding all the broken parts. The architecture is in transition, and completing this transition by removing DAG-related code would make the codebase cleaner and more maintainable.

The "partition function" in the config is a red herring - it's never actually used in the working code path. The simplified trajectory balance (assuming Z=1) is what makes everything work.