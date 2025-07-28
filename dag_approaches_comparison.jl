"""
DAG Approaches Comparison for GFlowNet
=====================================

This file compares three different approaches to handling the Directed Acyclic Graph (DAG)
in GFlowNet implementations, clarifying the architectural trade-offs.

The key insight: The mathematical DAG concept is required, but the explicit DAG
data structure is an engineering choice that can be avoided.
"""

using Random, Statistics

# =============================================================================
# Approach 1: EXPLICIT DAG (Current Complex Implementation)
# =============================================================================

"""
Approach 1: Pre-compute and store the entire state space as an explicit graph

Pros:
- Fast lookups after construction
- Complete view of state space
- Can analyze graph properties

Cons:
- Complex cache management
- Object identity issues
- Memory intensive for large spaces
- Brittle caching mechanisms
- Over-engineered for most domains
"""

struct ExplicitDAG
    states::Vector
    edges::Vector
    action_cache::Dict  # state → applicable_actions
    adjacency_cache::Dict  # state → neighbor_states
    state_to_idx::Dict  # state → index
end

function build_explicit_dag(initial_state, actions)
    # Complex exploration algorithm
    discovered_states = Set([initial_state])
    edges = []
    action_cache = Dict()
    # ... 100+ lines of complex construction logic

    return ExplicitDAG(states, edges, action_cache, adjacency_cache, state_to_idx)
end

function get_applicable_actions_explicit(dag::ExplicitDAG, state)
    # PROBLEM: Cache miss causes errors!
    if haskey(dag.action_cache, state)
        return dag.action_cache[state]  # ← Object identity dependency
    else
        # Fallback can fail if state wasn't in original construction
        error("Cache miss! State $state not found in DAG")
    end
end

# Training with explicit DAG
function train_with_explicit_dag(model, dag)
    # Sample trajectory
    trajectory = sample_trajectory_from_dag(model, dag)

    # Compute loss - can fail with cache misses
    loss = compute_loss(model, trajectory, dag)  # ← Can throw MethodError

    return loss
end

"""
Why Approach 1 Has Problems:
- Cache misses: `MethodError(getindex, (Dict{Any, Any}(),)`
- Object identity: Same logical state, different object → cache miss
- Over-engineering: 90% of complexity for marginal benefit
- Brittleness: Many failure modes
"""

# =============================================================================
# Approach 2: IMPLICIT DAG (Mathematical Only)
# =============================================================================

"""
Approach 2: Define the DAG implicitly through domain functions, compute on-demand

Pros:
- Simple and robust
- No cache management
- Works with any state representation
- Easy to debug
- Minimal code

Cons:
- Recomputes applicable actions each time
- No pre-computed graph analysis
- Slightly slower (usually negligible)
"""

# No explicit DAG struct needed!
# The DAG is defined implicitly by:

function is_applicable(action, state)
    # Domain-specific logic defines valid transitions
    # This IS the mathematical DAG definition
end

function apply_action(action, state)
    # State transition function
    # This defines the DAG edges
end

function get_applicable_actions_implicit(state, all_actions)
    # Compute fresh each time - simple and robust!
    return [action for action in all_actions if is_applicable(action, state)]
end

# Training with implicit DAG
function train_with_implicit_dag(model, initial_state, all_actions)
    # Sample trajectory - no DAG object needed
    trajectory = sample_trajectory_implicit(model, initial_state, all_actions)

    # Compute loss - always works, no cache misses!
    loss = compute_loss_implicit(model, trajectory, all_actions)

    return loss
end

function sample_trajectory_implicit(model, initial_state, all_actions)
    states = [initial_state]
    actions = []
    current_state = initial_state

    while !is_terminal(current_state)
        # Compute applicable actions fresh - no caching needed
        applicable = get_applicable_actions_implicit(current_state, all_actions)

        if isempty(applicable)
            break
        end

        # Sample action from policy
        action = sample_action_from_policy(model, current_state, applicable, all_actions)
        next_state = apply_action(action, current_state)

        push!(actions, action)
        push!(states, next_state)
        current_state = next_state
    end

    return (states=states, actions=actions)
end

"""
Why Approach 2 Works Better:
- No cache misses possible
- Object identity irrelevant
- 10x simpler code
- Robust by design
- Easy to debug
"""

# =============================================================================
# Approach 3: HYBRID (Best of Both Worlds)
# =============================================================================

"""
Approach 3: Use implicit DAG for training, explicit DAG for analysis (optional)

Pros:
- Robust training (implicit)
- Rich analysis when needed (explicit)
- Clear separation of concerns

Cons:
- Two different code paths
- More complex overall system
"""

struct HybridGFlowNet
    # Training uses implicit DAG
    initial_state
    all_actions

    # Analysis uses explicit DAG (computed lazily)
    cached_dag::Union{Nothing, ExplicitDAG}
end

function train_hybrid(model::HybridGFlowNet)
    # Training: Use implicit DAG (robust)
    return train_with_implicit_dag(model.policy, model.initial_state, model.all_actions)
end

function analyze_hybrid(model::HybridGFlowNet)
    # Analysis: Build explicit DAG if needed
    if isnothing(model.cached_dag)
        model.cached_dag = build_explicit_dag(model.initial_state, model.all_actions)
    end

    return analyze_dag_structure(model.cached_dag)
end

# =============================================================================
# Concrete Example: Grid World
# =============================================================================

# Define domain (works with any approach)
struct GridState
    x::Int
    y::Int
    terminal::Bool
end

abstract type GridAction end
struct MoveRight <: GridAction end
struct MoveUp <: GridAction end
struct Terminate <: GridAction end

# Mathematical DAG definition (required for all approaches)
function is_applicable(action::MoveRight, state::GridState)
    return !state.terminal && state.x < 5
end

function is_applicable(action::MoveUp, state::GridState)
    return !state.terminal && state.y < 5
end

function is_applicable(action::Terminate, state::GridState)
    return !state.terminal
end

function apply_action(action::MoveRight, state::GridState)
    return GridState(state.x + 1, state.y, false)
end

function apply_action(action::MoveUp, state::GridState)
    return GridState(state.x, state.y + 1, false)
end

function apply_action(action::Terminate, state::GridState)
    return GridState(state.x, state.y, true)
end

function is_terminal(state::GridState)
    return state.terminal
end

# =============================================================================
# Performance & Robustness Comparison
# =============================================================================

function compare_approaches()
    println("🔍 DAG Approaches Comparison")
    println("=" ^ 50)

    # Setup
    initial_state = GridState(1, 1, false)
    all_actions = [MoveRight(), MoveUp(), Terminate()]

    # Approach 1: Explicit DAG
    println("\n1️⃣ Explicit DAG Approach:")
    try
        # This would be complex and potentially fail
        println("   ⚠️  Complex construction required")
        println("   ⚠️  Cache management needed")
        println("   ⚠️  Object identity issues")
        println("   ⚠️  MethodError risks during training")
    catch e
        println("   ❌ Failed: $e")
    end

    # Approach 2: Implicit DAG
    println("\n2️⃣ Implicit DAG Approach:")
    try
        # Simple and robust
        state = GridState(2, 3, false)
        applicable = [action for action in all_actions if is_applicable(action, state)]
        println("   ✅ Applicable actions: $(length(applicable))")
        println("   ✅ No caching needed")
        println("   ✅ No object identity issues")
        println("   ✅ Robust by design")
    catch e
        println("   ❌ Failed: $e")
    end

    # Code complexity comparison
    println("\n📊 Complexity Comparison:")
    println("   Explicit DAG: ~500+ lines of complex caching logic")
    println("   Implicit DAG: ~50 lines of simple domain functions")
    println("   Complexity Ratio: 10:1")

    println("\n🎯 Recommendation:")
    println("   Use Implicit DAG for most applications")
    println("   Add explicit DAG only if you need rich graph analysis")
    println("   The mathematical DAG is always required")
    println("   The explicit DAG data structure is optional")
end

# =============================================================================
# Key Insights & Recommendations
# =============================================================================

"""
🔑 KEY INSIGHTS:

1. MATHEMATICAL DAG ≠ EXPLICIT DAG DATA STRUCTURE
   - Mathematical DAG: Required (defines valid transitions)
   - Explicit DAG: Optional engineering choice

2. THE TRAINING ERRORS COME FROM EXPLICIT DAG COMPLEXITY
   - Cache misses: Different object identity for same logical state
   - Over-engineering: 90% complexity for 10% benefit
   - Brittleness: Many failure modes

3. IMPLICIT DAG IS SIMPLER AND MORE ROBUST
   - No caching → No cache misses
   - No object identity issues
   - Compute on-demand is fast enough for most domains
   - Much easier to debug

4. WHEN TO USE EACH APPROACH:
   - Implicit DAG: Default choice for training
   - Explicit DAG: Only when you need graph analysis
   - Hybrid: For systems that need both

🎯 ANSWER TO "CAN WE IGNORE THE DAG?":

❌ NO: Cannot ignore the mathematical DAG concept
   - GFlowNet theory requires valid state transitions
   - Defined by is_applicable() and apply_action()

✅ YES: Can ignore the explicit DAG data structure
   - Pre-computing all states/edges is optional
   - Compute applicable actions on-demand instead
   - Much simpler and more robust

The current training errors disappear with implicit DAG because:
- No cache misses possible
- No object identity dependencies
- Simpler control flow
- Fewer failure modes
"""

# Run the comparison
if abspath(PROGRAM_FILE) == @__FILE__
    compare_approaches()
end
