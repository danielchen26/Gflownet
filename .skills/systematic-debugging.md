---
name: systematic-debugging
description: Evidence-based debugging for GFlowNet - check previous success before making changes. Use when encountering bugs, training failures, or Zygote errors.
---

<EXTREMELY-IMPORTANT>
Before making ANY debugging changes, you MUST follow this evidence-first methodology.

**The Golden Rule**: If it worked before, what changed?

Never jump to symptom-based fixes. Always gather evidence first.
</EXTREMELY-IMPORTANT>

## When to Use This Skill

Invoke this skill when encountering:
- ❌ Training failures or convergence issues
- ❌ Zygote mutation errors ("Mutating arrays is not supported")
- ❌ Type errors or method ambiguities
- ❌ NaN/Inf values in loss or rewards
- ❌ Unexpected behavior in examples
- ❌ Test failures after changes

## Systematic Debugging Workflow

### Phase 1: Gather Success Evidence

**BEFORE touching any code**, create a TodoWrite checklist with these tasks:

1. **Check for previous successful runs**
   ```bash
   # Look for results from successful executions
   ls -lt results/
   ls -lt examples/*/results/
   ```

   Evidence to find:
   - Timestamped output files that completed
   - HTML reports that were generated successfully
   - CSV files with actual training data
   - Log files showing completed training epochs

2. **Search git history for working versions**
   ```bash
   git log --oneline --all
   git log --oneline --grep="working" --grep="success"
   ```

   Questions to answer:
   - When did this last work?
   - What was the last successful commit?
   - What commits happened since then?

3. **Identify what changed**
   ```bash
   git diff <working_commit> HEAD -- path/to/failing/file.jl
   ```

   Look for:
   - Function signature changes
   - New dependencies or imports
   - Configuration changes
   - Mathematical formula modifications

4. **Document the exact error**
   - Copy the full error message
   - Capture the stack trace
   - Note which line fails
   - Record input values that trigger the error

### Phase 2: Root Cause Analysis

Use this decision tree to identify the root cause:

#### If Error Contains "Mutating arrays is not supported"
**Root Cause**: Zygote mutation violation

**Common Patterns to Find**:
```julia
# ❌ MUTATIONS (cause Zygote errors)
x += 1
y -= delta
array[i] = value
push!(array, item)
state.field = new_value
```

**How to Fix**:
```julia
# ✅ PURE FUNCTIONAL (Zygote-compatible)
x = x + 1
y = y - delta
new_array = [array[1:i-1]..., value, array[i+1:end]...]
new_array = [array..., item]
new_state = @set state.field = new_value
```

**Search Command**:
```bash
# Find mutations in differentiable code
grep -n "+=\|-=\|\*=" path/to/file.jl
grep -n "push!\|append!\|setindex!" path/to/file.jl
```

#### If Error Contains "MethodError" or "type" issues
**Root Cause**: Type instability or interface mismatch

**Check**:
1. Are you using `Vector{<:Trajectory}` not `Vector{Any}`?
2. Are states/actions concrete types (not abstract)?
3. Are Float32/Float64 types consistent?
4. Did interface requirements change?

**Validation**:
```julia
# Add type assertions
@assert isa(trajectories, Vector{<:Trajectory})
@assert typeof(state) <: AbstractState
@assert eltype(features) == Float32
```

#### If Error Contains "NaN" or "Inf"
**Root Cause**: Numerical instability

**Check**:
1. Are rewards positive? (`reward > 0` for terminal states)
2. Are probabilities normalized? (`sum(probs) ≈ 1.0`)
3. Are log arguments positive? (`log(max(x, 1e-10))`)
4. Are gradients exploding? (check gradient norms)

**Fix Pattern**:
```julia
# Safe numerical operations
reward = max(reward_value, 1e-8)
log_reward = log(reward)

prob = max(prob, 1e-10)
log_prob = log(prob)
```

### Phase 3: Minimal Reproduction

Create a minimal failing example to isolate the issue:

```julia
# Minimal reproduction template
using GFlowNet
using Test

@testset "Minimal Bug Reproduction" begin
    # Set up minimal state
    state = GridState(1, 1, false)
    action = MoveUp()

    # Test the specific failing operation
    # If Zygote error, test with gradient
    using Zygote
    @test_throws "expected error" gradient(x -> failing_function(x), state)

    # If type error, test type assertions
    @test typeof(result) <: ExpectedType

    # If numerical error, test bounds
    @test all(isfinite, result)
end
```

### Phase 4: Fix Root Cause Only

**CRITICAL RULES**:

1. ✅ **Fix ONE thing at a time**
   - Change only the identified root cause
   - Don't refactor unrelated code
   - Don't change learning rates, batch sizes, etc.

2. ✅ **Keep everything else intact**
   - Preserve working functionality
   - Maintain same configuration
   - Don't "improve" other parts

3. ✅ **Test the fix in isolation**
   - Run the minimal reproduction
   - Verify it passes
   - Check that nothing else broke

4. ✅ **Verify previous success is restored**
   - Run the full example
   - Compare results with previous successful run
   - Ensure metrics are similar

## Common Anti-Patterns to Avoid

### ❌ Shotgun Debugging
```julia
# DON'T change multiple things at once:
# - Lower learning rate AND
# - Reduce batch size AND
# - Change model architecture AND
# - Modify reward function
```

### ❌ Assumption-Based Fixes
```julia
# DON'T assume without evidence:
# "Training is unstable, must be learning rate"
# "Loss is NaN, must be gradient explosion"
# "Let's just try a different objective"
```

### ❌ Ignoring Success Evidence
```julia
# DON'T ignore that it worked before:
# "Let's rewrite everything from scratch"
# "The old approach was probably wrong"
# "I'll try a completely different method"
```

## GFlowNet-Specific Root Causes

### 1. Zygote Mutations (Most Common)
**Symptoms**: "Mutating arrays is not supported" error
**Location**: Usually in `apply_action()`, `state_to_features()`, or custom reward functions
**Fix**: Replace all `+=`, `-=`, `push!` with pure functional equivalents

### 2. Flow Caching Issues
**Symptoms**: Stale flows, incorrect gradients
**Solution**: Ensure `Zygote.@ignore` wraps cache operations
**Check**: `clear_flow_cache!()` called when parameters update

### 3. Backward Policy Errors
**Symptoms**: NaN in DETAILED_BALANCE loss
**Check**: Is `include_backward=true` when using DETAILED_BALANCE?
**Validate**: Use `validate_backward_policy_normalization()`

### 4. Multi-Start Configuration
**Symptoms**: Type errors with initial states
**Check**: All initial states are same type?
**Validate**: `initial_states::Vector{S} where S <: AbstractState`

### 5. Interface Requirements
**Symptoms**: MethodError for required functions
**Missing**: One of `state_to_features`, `is_applicable`, `apply_action`, `reward`, `is_terminal_state`
**Fix**: Implement all required interface functions

## Success Criteria

Before considering the bug fixed:

- [ ] Minimal reproduction passes
- [ ] Full example completes without errors
- [ ] Results match previous successful run quality
- [ ] No new warnings or deprecations introduced
- [ ] Tests pass (if test suite exists)
- [ ] Git diff shows minimal, focused changes

## Documentation

After fixing, update:
1. Add comment explaining the root cause
2. Document why the fix works (not just what changed)
3. Add regression test if appropriate
4. Update examples if interface changed

## Example: Grid World Zygote Fix

**Problem**: Grid world example failing with Zygote mutation error

**Evidence Gathering**:
- Found previous successful run: `results/2025-07-25_18-59-32/`
- Last working commit: `a1b2c3d`
- Change since then: Modified `apply_action()` to use `+=`

**Root Cause**: Mutation in `apply_action()`
```julia
# ❌ WRONG (found via grep "+=" grid_world.jl)
function apply_action(action::GridAction, state::GridState)
    x, y = state.x, state.y
    isa(action, MoveUp) && (y += 1)  # ← MUTATION!
    ...
end
```

**Fix** (minimal, targeted):
```julia
# ✅ CORRECT (pure functional)
function apply_action(action::GridAction, state::GridState)
    y = isa(action, MoveUp) ? state.y + 1 : state.y
    y = isa(action, MoveDown) ? state.y - 1 : y
    ...
end
```

**Result**: Training completes successfully, results match previous run

## Checklist Template

Use TodoWrite to create this checklist at the start of debugging:

```markdown
Debugging Checklist:
- [ ] Phase 1: Gather evidence (previous runs, git history, changes)
- [ ] Phase 2: Root cause analysis (Zygote/type/numerical)
- [ ] Phase 3: Create minimal reproduction
- [ ] Phase 4: Fix root cause only (no shotgun debugging)
- [ ] Verify: Minimal test passes
- [ ] Verify: Full example works
- [ ] Verify: Results match previous success
- [ ] Document the fix and root cause
```

