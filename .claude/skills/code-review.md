---
name: code-review
description: Automated code quality checklist for GFlowNet - verify Zygote compatibility, clean comments, proper structure. Use before commits or after implementation.
---

<EXTREMELY-IMPORTANT>
Code review is NOT optional. Use this skill proactively:
- ✅ After implementing any feature
- ✅ Before creating commits or pull requests
- ✅ When reviewing generated or modified code
- ✅ After major refactoring

This skill creates a comprehensive TodoWrite checklist to ensure code quality.
</EXTREMELY-IMPORTANT>

## When to Use This Skill

Invoke this skill:
- 📝 After completing a feature implementation
- 📝 Before committing code to git
- 📝 When reviewing AI-generated code
- 📝 After refactoring existing code
- 📝 When setting up new examples or domains
- 📝 Before creating pull requests

## Code Review Workflow

### Phase 1: Zygote Compatibility Verification (CRITICAL)

**This is the #1 cause of difficult bugs in GFlowNet!**

Create TodoWrite checklist:

**Task 1.1: Search for Mutations**
```bash
# Find all potential mutations in Julia files
grep -rn "+=\|-=\|\*=\|/=" src/ examples/
grep -rn "push!\|append!\|pop!\|popfirst!" src/ examples/
grep -rn "setindex!\|\.\[\].*=" src/ examples/
```

**Task 1.2: Review Differentiable Functions**

Functions that might be called during gradient computation MUST be mutation-free:
- `apply_action()`
- `state_to_features()`
- `reward()`
- Custom loss functions
- Policy networks

**Checklist for each function**:
- [ ] No `+=`, `-=`, `*=`, `/=` operators
- [ ] No `push!`, `append!`, `pop!`, `popfirst!`
- [ ] No array index assignment (`arr[i] = value`)
- [ ] No struct field mutation (`obj.field = value`)
- [ ] Uses pure functional patterns instead

**Good vs Bad Examples**:
```julia
# ❌ BAD (will break Zygote)
function apply_action(action::MoveUp, state::GridState)
    y = state.y
    y += 1  # MUTATION!
    return GridState(state.x, y, false)
end

# ✅ GOOD (Zygote-compatible)
function apply_action(action::MoveUp, state::GridState)
    y = state.y + 1  # Pure calculation
    return GridState(state.x, y, false)
end

# ❌ BAD (array mutation)
function build_trajectory(actions, state)
    states = [state]
    for action in actions
        push!(states, apply_action(action, states[end]))  # MUTATION!
    end
    return states
end

# ✅ GOOD (functional array construction)
function build_trajectory(actions, state)
    states = [state]
    for action in actions
        new_state = apply_action(action, states[end])
        states = [states..., new_state]  # New array each time
    end
    return states
end
```

### Phase 2: Comment Style and Documentation

**Task 2.1: Remove Development Tags**

NO development tags allowed in production code:
```bash
# Find and remove these patterns
grep -rn "CORRECTED\|OPTIMIZED\|ENHANCED\|FIXED\|ADDED\|REMOVED" src/ examples/
grep -rn "TODO\|FIXME\|HACK\|XXX" src/ examples/
```

**Bad Examples (remove these)**:
```julia
# ❌ REMOVED: old function moved to flows.jl
# ❌ FIXED: Now uses SimpleDiGraph instead of Graph
# ❌ TODO: optimize this later
# ❌ CORRECTED: Type annotation added
```

**Good Examples (keep these)**:
```julia
# ✅ Uses SimpleDiGraph for proper directed graph representation
# ✅ Zygote requires pure functions; mutations would break AD
# ✅ Comprehensive validation prevents type issues in gradient computation
```

**Task 2.2: Verify Comment Quality**

Comments should explain **WHY**, not **WHAT**:
- [ ] No obvious comments describing what code does
- [ ] Include mathematical context for algorithms
- [ ] Explain non-obvious design decisions
- [ ] Document Zygote compatibility rationale
- [ ] Reference papers or theory when applicable

**Example**:
```julia
# ❌ BAD: Obvious what the code does
# Create a new GridState with x and y coordinates
state = GridState(x, y, false)

# ✅ GOOD: Explains why
# Terminal flag must be false to allow further exploration
# Ensures DAG construction doesn't terminate prematurely
state = GridState(x, y, false)

# ✅ GOOD: Mathematical context
# Flow conservation: ∑ P_F(s'|s) * F(s) = F(s')
# This ensures detailed balance property holds
```

### Phase 3: Type System Validation

**Task 3.1: Check Type Annotations**
```bash
# Find Vector{Any} or other type instabilities
grep -rn "Vector{Any}\|Dict{Any" src/ examples/
grep -rn "::Any" src/ examples/
```

**Requirements**:
- [ ] All struct fields have concrete types
- [ ] Use `Vector{<:Trajectory}` not `Vector{Any}`
- [ ] Use `Vector{Float32}` not `Vector{Real}`
- [ ] Parametric types where needed: `Dict{S, Vector{A}}`

**Good Patterns**:
```julia
# ✅ GOOD: Concrete types
struct GridState <: AbstractState
    x::Int
    y::Int
    is_terminal::Bool
end

# ✅ GOOD: Parametric with constraints
struct DirectedAcyclicGraph{S<:AbstractState, A<:AbstractAction}
    states::Vector{S}
    actions::Vector{A}
end

# ❌ BAD: Type instability
struct BadState
    data::Vector  # Missing element type!
    metadata::Any  # Too generic!
end
```

**Task 3.2: Validate Interface Implementations**

For any new domain, verify all required interfaces:
- [ ] `state_to_features` returns `Vector{Float32}`
- [ ] `is_applicable` returns `Bool`
- [ ] `apply_action` returns correct state type
- [ ] `is_terminal_state` returns `Bool`
- [ ] `reward` returns `Float32`, positive for terminals

### Phase 4: Numerical Stability

**Task 4.1: Check Reward Validation**
```bash
# Find reward calculations
grep -rn "reward(" src/ examples/
```

**Requirements**:
- [ ] Terminal rewards are always positive (`> 0`)
- [ ] Non-terminal rewards are exactly zero
- [ ] Rewards use Float32 type
- [ ] Safe numerical operations (no log(0), divide by zero)

**Good Pattern**:
```julia
# ✅ GOOD: Safe reward calculation
function reward(state::MyState)
    !state.is_terminal && return 0.0f0

    raw_reward = calculate_reward(state)
    # Ensure positivity (GFlowNet mathematical requirement)
    return Float32(max(raw_reward, 1e-8))
end

# ❌ BAD: Can return negative or zero
function reward(state::MyState)
    return Float32(state.score)  # What if score ≤ 0?
end
```

**Task 4.2: Check Probability Handling**
```bash
# Find log operations that could fail
grep -rn "log(" src/ examples/
```

**Safe Operations**:
```julia
# ✅ GOOD: Safe log with lower bound
log_prob = log(max(prob, 1e-10))

# ✅ GOOD: Safe normalization
probs = softmax(logits)  # Numerically stable
@assert sum(probs) ≈ 1.0  # Validate normalization

# ❌ BAD: Could fail if prob == 0
log_prob = log(prob)
```

### Phase 5: Example Directory Structure

**Task 5.1: Validate Example Organization**

For any example in `examples/`, check:
```
examples/domain_name/
├── domain_name.jl     ← Single main file (REQUIRED)
├── Project.toml       ← Dependencies (REQUIRED)
├── Manifest.toml      ← Version lock
├── README.md          ← Documentation (REQUIRED)
└── results/           ← Auto-created outputs (optional)
```

**Checklist**:
- [ ] Single main `.jl` file with clear name
- [ ] `Project.toml` exists with dependencies
- [ ] `README.md` documents purpose and usage
- [ ] No scattered test files (`test_*.jl`, `demo_*.jl`)
- [ ] No debug logs (`.log` files)
- [ ] No system files (`.DS_Store`)
- [ ] Results directory (if exists) contains only latest outputs

**Task 5.2: Check for Cleanup**
```bash
# Find files that should be removed
find examples/ -name "test_*.jl" -o -name "demo_*.jl" -o -name "simple_*.jl"
find examples/ -name "*.log" -o -name ".DS_Store"
find examples/ -name "*_backup.jl" -o -name "*_old.jl"
```

**Archive Exception**:
If `archive/` directory exists, verify it's properly documented:
- [ ] Has `ARCHIVE_README.md` explaining purpose
- [ ] Clear versioning structure (`v1/`, `v2/`, etc.)
- [ ] Documented research or educational value
- [ ] Not just random temporary files

### Phase 6: Function Design Quality

**Task 6.1: Single Responsibility Principle**

Each function should have one clear purpose:
```julia
# ✅ GOOD: Focused, single responsibility
function compute_trajectory_probability(trajectory, policy, parameters)
    # Only computes probability, nothing else
end

function validate_trajectory_structure(trajectory)
    # Only validates structure, doesn't compute anything
end

# ❌ BAD: Too many responsibilities
function process_trajectory(trajectory, policy, params, should_validate, should_cache)
    # Does too many things: validate, compute, cache, maybe more
end
```

**Task 6.2: Error Handling Quality**
```bash
# Check error messages
grep -rn "error(" src/ examples/
```

**Good Error Messages**:
```julia
# ✅ GOOD: Specific, actionable
if !haskey(parameters, :forward)
    error("Model parameters must include :forward policy parameters. Got keys: $(keys(parameters))")
end

if length(trajectory.states) < 2
    error("Trajectory too short: need at least 2 states (initial + one action), got $(length(trajectory.states))")
end

# ❌ BAD: Vague, unhelpful
if !haskey(parameters, :forward)
    error("Missing parameters")
end
```

### Phase 7: High-Level API Usage

**Task 7.1: Verify No Manual Networks**
```bash
# Should NOT find these patterns
grep -rn "Chain(\|Dense(" src/ examples/
grep -rn "Flux\|Lux" examples/  # Unless in special cases
```

**Requirements**:
- [ ] No manual `Chain()` or `Dense()` definitions
- [ ] Use `create_gflownet()` for model creation
- [ ] Use `create_forward_policy()` if custom policy needed
- [ ] Use `TrainingConfig` and `train_gflownet()` for training

**Good vs Bad**:
```julia
# ❌ BAD: Manual network definition
policy_network = Chain(
    Dense(state_dim, 64, relu),
    Dense(64, n_actions),
    softmax
)

# ✅ GOOD: High-level API
model = create_gflownet(
    initial_state,
    all_actions;
    state_dim = 10,
    hidden_dim = 64
)
```

### Phase 8: Performance and Memory

**Task 8.1: Check for Inefficiencies**
```bash
# Find potential performance issues
grep -rn "global " src/ examples/
grep -rn "eval(" src/ examples/
```

**Checklist**:
- [ ] No global variables in hot paths
- [ ] No `eval()` usage
- [ ] Use `@views` for array slicing if appropriate
- [ ] Cache computations when safe (with `Zygote.@ignore`)

**Task 8.2: Memory Management**
```julia
# ✅ GOOD: Clear caches when parameters change
function update_parameters!(model, new_params)
    model.parameters = new_params
    clear_flow_cache!()  # Prevent stale cached values
end

# ✅ GOOD: Pre-allocate in training loops
losses = Vector{Float64}(undef, n_iterations)
for i in 1:n_iterations
    losses[i] = compute_loss(...)
end

# ❌ BAD: Repeated allocations
losses = []
for i in 1:n_iterations
    push!(losses, compute_loss(...))  # Allocates each time
end
```

### Phase 9: Exception Handling and Feature Completeness

**Task 9.1: Check for Silent Exception Handling**
```bash
# Find try-catch blocks that might swallow errors
grep -rn "catch\|return nothing" --include="*.jl" src/ examples/
```

**Checklist**:
- [ ] All catch blocks log warnings/errors (use `@warn` or `@error`)
- [ ] No silent `return nothing` after exceptions
- [ ] Critical errors are re-thrown after logging
- [ ] Failures are visible for debugging

**Good vs Bad**:
```julia
# ❌ BAD: Silent failure
catch e
    return nothing
end

# ✅ GOOD: Visible failure
catch e
    @warn "Operation failed" exception=e
    return nothing
end
```

**Task 9.2: Verify Feature Implementation Completeness**

Check that exposed parameters have working implementations:
```bash
# Find TrainingConfig or SamplingConfig definitions
grep -rn "struct.*Config" --include="*.jl" src/
```

**Checklist**:
- [ ] Every config parameter has implementation code using it
- [ ] Experimental/unimplemented features are clearly marked
- [ ] Parameters without implementation throw warnings when enabled
- [ ] No "TODO: implement" without corresponding timeline

**Task 9.3: Gradient Type Consistency**

Ensure functions handle `ComponentVector` gradients correctly:
```bash
# Find gradient processing functions
grep -rn "grads\|gradient" --include="*.jl" src/training/
```

**Checklist**:
- [ ] Functions don't assume `NamedTuple` for gradients
- [ ] Use `haskey()` or duck-typing for gradient access
- [ ] Test gradient handling with actual Zygote output

## Complete Code Review Checklist

Use TodoWrite to create this comprehensive checklist:

```markdown
Code Review Checklist:

CRITICAL - Zygote Compatibility:
- [ ] Search for mutations (+=, push!, etc.)
- [ ] Verify apply_action is pure functional
- [ ] Verify state_to_features is pure functional
- [ ] Test gradient computation doesn't error

Code Quality:
- [ ] Remove development tags (FIXED, TODO, CORRECTED)
- [ ] Comments explain WHY not WHAT
- [ ] Include mathematical context where relevant
- [ ] Professional comment style

Type System:
- [ ] No Vector{Any} or Dict{Any,Any}
- [ ] Concrete struct field types
- [ ] Proper parametric types where needed
- [ ] All interface functions have correct return types

Numerical Stability:
- [ ] Terminal rewards always positive
- [ ] Safe log operations (max with lower bound)
- [ ] Probability normalization validated
- [ ] No divide-by-zero possibilities

Example Structure:
- [ ] Single main .jl file
- [ ] Project.toml with dependencies
- [ ] README.md with documentation
- [ ] No test_*.jl or demo_*.jl clutter
- [ ] No .log or .DS_Store files
- [ ] Archive (if exists) properly documented

Function Design:
- [ ] Single responsibility per function
- [ ] Descriptive function names
- [ ] Informative error messages with context
- [ ] No functions > 50 lines (consider splitting)

High-Level API:
- [ ] No manual Chain() or Dense() definitions
- [ ] Uses create_gflownet() for model creation
- [ ] Uses TrainingConfig and train_gflownet()

Performance:
- [ ] No global variables
- [ ] Caches cleared when parameters update
- [ ] Reasonable memory allocation patterns

Exception Handling:
- [ ] No silent catch blocks (all log warnings)
- [ ] Return nothing cases are logged
- [ ] Critical errors visible for debugging

Feature Completeness:
- [ ] All config parameters have implementations
- [ ] Experimental features clearly marked
- [ ] Gradient functions handle ComponentVector
```

## Automated Checks

Run these commands to automate parts of the review:

```bash
# 1. Find all mutations
echo "=== Checking for Zygote-breaking mutations ==="
grep -rn "+=\|-=\|\*=" --include="*.jl" src/ examples/ || echo "✅ No mutations found"

# 2. Find development tags
echo "=== Checking for development comment tags ==="
grep -rn "FIXED\|TODO\|CORRECTED\|REMOVED" --include="*.jl" src/ examples/ || echo "✅ No tags found"

# 3. Find type instabilities
echo "=== Checking for type instabilities ==="
grep -rn "Vector{Any}\|::Any" --include="*.jl" src/ examples/ || echo "✅ No obvious type issues"

# 4. Find manual networks
echo "=== Checking for manual network definitions ==="
grep -rn "Chain(\|Dense(" --include="*.jl" examples/ || echo "✅ Using high-level API"

# 5. Find cleanup targets
echo "=== Checking for files to clean up ==="
find examples/ -name "test_*.jl" -o -name "*.log" -o -name ".DS_Store" || echo "✅ Examples are clean"
```

## Review Result Template

After completing the review, document findings:

```markdown
## Code Review Summary

**Reviewed**: [file/module name]
**Date**: [date]

### Zygote Compatibility: ✅/⚠️/❌
- Found X potential mutations
- [List each issue with file:line]

### Code Quality: ✅/⚠️/❌
- Found X development tags to remove
- Comments: [assessment]

### Type System: ✅/⚠️/❌
- Type instabilities: [count]
- Interface compliance: [yes/no]

### Required Changes:
1. [Priority 1 change]
2. [Priority 2 change]

### Recommendations:
- [Improvement suggestion 1]
- [Improvement suggestion 2]

### Approved for Commit: YES/NO
```

## Common Issues and Fixes

### Issue 1: Zygote Mutation in apply_action
**Fix**: Replace with conditional expressions
```julia
# Before
y += 1

# After
y = y + 1
```

### Issue 2: Development Comment Tags
**Fix**: Rewrite to explain why
```julia
# Before
# FIXED: Now uses correct type

# After
# Uses Vector{<:Trajectory} for type stability during gradient computation
```

### Issue 3: Manual Network Definition
**Fix**: Use high-level API
```julia
# Before
policy = Chain(Dense(state_dim, 64, relu), Dense(64, n_actions))

# After
model = create_gflownet(initial_state, actions; state_dim=state_dim, hidden_dim=64)
```

### Issue 4: Example Directory Clutter
**Fix**: Clean up and organize
```bash
# Remove temporary files
rm examples/my_domain/test_*.jl
rm examples/my_domain/debug.log

# Ensure clean structure
examples/my_domain/
├── my_domain.jl
├── Project.toml
└── README.md
```

This comprehensive review process ensures all GFlowNet code meets quality standards and avoids common pitfalls.
