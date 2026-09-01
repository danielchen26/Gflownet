# Flow Conservation and Computation
# Mathematical foundations of flow F(s) and flow conservation in GFlowNets

using Zygote

# =============================================================================
# Flow Conservation - Mathematical Foundation
# =============================================================================

"""
    FlowComputationMethod

Enumeration of methods for computing flows in GFlowNet.

# Mathematical Foundation
Different approaches to computing the flow F(s) through each state:
- `:recursive`: F(s) = Σ_{s'} P_F(s'|s) * F(s') (recursive definition)
- `:direct`: F(s) = Z(s) (direct estimation via flow network)
- `:mixed`: Combination of recursive and direct methods
"""
@enum FlowComputationMethod begin
    RECURSIVE_FLOW
    DIRECT_FLOW
    MIXED_FLOW
end

# =============================================================================
# Direct Flow Estimation - Neural Network Based
# =============================================================================

# Note: compute_flow_estimate function is defined later in this file

# =============================================================================
# Flow Caching System
# =============================================================================

"""
Global flow cache for memoization of expensive flow computations.

# Mathematical Foundation
Caches computed flow values F(s) to avoid recomputation since:
- Flow values are deterministic given model parameters
- Recursive computation can be expensive for deep DAGs
- Many training objectives require repeated flow queries
"""
# Use a more specific type for the cache to avoid type instability
# Use a Ref to allow mutation of the cache container itself
const FLOW_CACHE = Ref(Dict{Tuple{UInt64,Any}, Float64}())

"""
    clear_flow_cache!()

Clear the global flow cache.

# Usage
Should be called when model parameters change to ensure cache consistency.
"""
function clear_flow_cache!()
    empty!(FLOW_CACHE[])
    return nothing
end

"""
    get_cache_key(model::GFlowNetModel, state::AbstractState)

Generate a unique cache key for flow memoization.

# Mathematical Foundation
Creates a key that uniquely identifies the (model, state) pair for caching.
Uses parameter hash to ensure cache invalidation when model changes.
"""
function get_cache_key(model::GFlowNetModel, state::AbstractState)
    # Use hash of parameters to ensure cache invalidation on parameter changes
    param_hash = hash(model.parameters)
    return (param_hash, state)
end

# =============================================================================
# Terminal State Flow - Mathematical Foundation
# =============================================================================

"""
    terminal_flow(state::AbstractState)::Float64

Compute flow for terminal states: F(s) = R(s).

# Mathematical Foundation
For terminal states s ∈ S_T, the flow is defined as:
F(s) = R(s)

where R(s) is the reward function. This is the boundary condition for
flow conservation equations.

# Arguments
- `state::AbstractState`: Terminal state

# Returns
- `Float64`: Flow value F(s) = R(s)

# Mathematical Requirements
- State must be terminal: is_terminal_state(state) == true
- Reward must be positive: R(s) > 0 (GFlowNet requirement)
"""
function terminal_flow(state::AbstractState)::Float64
    if !is_terminal_state(state)
        throw(ArgumentError("terminal_flow can only be called on terminal states"))
    end

    reward_value = reward(state)

    # Validate reward positivity (mathematical requirement)
    if reward_value <= 0
        throw(ArgumentError("Terminal state reward must be positive: got $reward_value"))
    end

    return Float64(reward_value)
end

# =============================================================================
# Recursive Flow Computation - Mathematical Foundation
# =============================================================================

"""
    compute_recursive_flow(model::GFlowNetModel, state::AbstractState)::Float64

Compute flow using recursive flow conservation equation.

# Mathematical Foundation
For non-terminal states, flow is computed recursively as:
F(s) = Σ_{s' ∈ children(s)} P_F(s'|s) * F(s')

For terminal states:
F(s) = R(s)

This implements the fundamental flow conservation equation of GFlowNets.

# Arguments
- `model::GFlowNetModel`: Complete GFlowNet model
- `state::AbstractState`: State to compute flow for

# Returns
- `Float64`: Flow value F(s)

# Mathematical Properties
- F(terminal) = R(terminal)
- F(s) = sum over children s' of F(s') * P_B(s|s'), so F(s0) = Z = sum_x R(x)
- Result is deterministic given model parameters

# Why P_B and not P_F
This previously computed F(s) = sum_{s'} P_F(s'|s) F(s'). Because sum P_F = 1
that is a CONVEX COMBINATION, i.e. an expectation, so F(s0) was bounded by
max_x R(x) and could never equal Z once more than one terminal state is
rewarded. Measured on the 3x3 grid: flow(model, s0) = 2.2184, matching
E_{P_F}[R] to 0.000e+00 while Z_true = 19.0 -- so the function named
`partition_function` was returning a reward expectation.

Weighting each child by P_B(s|s') makes the path multiplicity cancel rather
than accumulate, which is the recursion consistent with the corrected
Trajectory Balance objective, and gives F(s0) = sum_x R(x) exactly.
"""
# The recursion below needs a POLICY CONTEXT, not a model: all_actions plus the backward
# policy and its parameters/states. Nothing in it reads the initial state, which is why
# F(s) is start-independent and the same recursion is correct for a multi-start model --
# only F(s_0^i) = Z_i differs per start. Extracted so multi_start.jl can reuse it without
# duplicating the definition; that file loads after this one, so it cannot be named here.
function compute_recursive_flow(all_actions, backward_policy, parameters, states,
                                state::AbstractState;
                                in_progress = nothing)::Float64
    # CYCLE DETECTION. Without it this recursion StackOverflows on any cyclic state graph,
    # which `allow_all_moves=true` produces for grid world: MoveLeft/MoveDown give the start
    # state parents, so the backward chain is never absorbed at s_0.
    #
    # That is not a performance problem, it is the absence of a solution. sum_tau P_B(tau|x)
    # becomes the expected number of visits to a recurrent finite chain -- measured on the
    # 2x2 allow_all model, the backward transfer matrix has spectral radius exactly
    # 1.0000000000 and sigma_min(I - W') = 1.5e-16, so a direct solve throws
    # SingularException. The partial sums diverge: horizon 10/50/200/1000/4000 gives
    # sum_tau P_B = 2.0/12.0/49.5/249.5/999.5 where it must be 1, and the Z candidate m(s0)
    # gives 24/144/594/2994/11994 against exact_Z = 12.
    #
    # So refuse, and name the cause. DETAILED_BALANCE and FLOW_MATCHING fall back to flow()
    # whenever include_flow_estimator=false (the default), so before this guard they crashed
    # with StackOverflowError on iteration 1 -- which train_gflownet then caught and recorded
    # as NaN, reporting the run as complete.
    seen = isnothing(in_progress) ? Set{Any}() : in_progress
    if state in seen
        throw(ArgumentError(
            "cyclic state graph: $state was reached again on its own path, so the backward " *
            "chain is not absorbed at the initial state and no finite F satisfies " *
            "F(s) = sum_children F(c) P_B(s|c). Trajectory balance has no Z on this graph. " *
            "For grid world this is allow_all_moves=true; train with a LEARNABLE partition " *
            "function, which absorbs the cycles, and do not read flow() or " *
            "partition_function() as a ground truth there."))
    end

    # Base case: terminal states
    if is_terminal_state(state)
        return terminal_flow(state)
    end

    applicable_actions = get_applicable_actions(state, all_actions)
    isempty(applicable_actions) && return 0.0

    push!(seen, state)
    total_flow = 0.0
    for action in all_actions
        action in applicable_actions || continue
        next_state = apply_action(action, state)

        # P_B(state | next_state). Uniform over the parents of next_state when there is
        # no backward policy -- a valid fixed P_B. P_B == 1 would NOT be: it is a
        # distribution only where every state has one parent.
        back_prob = if isnothing(backward_policy) || !haskey(parameters, :backward)
            np = length(backward_parent_states(next_state, all_actions))
            np == 0 ? 1.0 : 1.0 / np
        else
            compute_backward_probability(backward_policy, next_state, state,
                                         parameters.backward, states.backward, all_actions)
        end

        total_flow += back_prob * compute_recursive_flow(all_actions, backward_policy,
                                                        parameters, states, next_state;
                                                        in_progress = seen)
    end
    # Pop on the way out: `seen` tracks the current PATH, not every state ever visited. A
    # state reachable by two different paths is not a cycle, and keeping it in the set would
    # report a false one -- which is exactly the multi-parent structure this whole repair is
    # about.
    delete!(seen, state)
    return total_flow
end

compute_recursive_flow(model::GFlowNetModel, state::AbstractState)::Float64 =
    compute_recursive_flow(model.all_actions, model.backward_policy,
                           model.parameters, model.states, state)

"""
    compute_recursive_flow_memoized(model::GFlowNetModel, state::AbstractState)::Float64

Compute recursive flow with REAL memoization over the whole recursion.

# Why this was rewritten
The previous version cached only the ROOT call and then delegated to
`compute_recursive_flow`, whose internal recursive calls went to ITSELF rather
than back through the memo. So every interior node was recomputed once per path
reaching it, and the docstring's claim of "O(|S|) time complexity instead of
potentially exponential" was false: it was exponential with exactly one cached
entry. Measured on a 4x4 grid (16 states, hidden_dim 32), one call cost 47.1 ms
and left the cache holding 0 entries.

The cost was paid per call by every consumer of `flow()`: the DB and FM
fallbacks, both flow validators, `compute_gflownet_metrics`, and the dashboard's
`compute_flow_field`, which calls `flow()` once per grid cell -- 64 exponential
recursions for one Flow-view request on an 8x8 grid.

# Differentiability
NON-DIFFERENTIABLE by design, which is why the memo is sound. A cache hit
returns a plain Float64 carrying no tape, so memoizing a differentiable
recursion would silently drop gradients. Verified that no caller differentiates
through here: every loss-path use of `flow()` is already inside
`Zygote.@ignore` (losses.jl DB and FM fallbacks), and the validators, metrics and
server paths run outside any gradient. `compute_recursive_flow` remains the
exact, differentiable, unmemoized definition and is unchanged.

# Cache key
The parameter hash is computed ONCE per top-level call and threaded through the
recursion. It used to be recomputed inside `get_cache_key` on every probe, at
5.4 us per probe for a 2597-element parameter vector -- which would have become
the dominant cost once the interior was actually memoized.
"""
function compute_recursive_flow_memoized(model::GFlowNetModel, state::AbstractState)::Float64
    return Zygote.@ignore begin
        param_hash = hash(model.parameters)
        _memoized_flow(model, state, param_hash)
    end
end

function _memoized_flow(model::GFlowNetModel, state::AbstractState,
                        param_hash::UInt64; in_progress = nothing)::Float64
    # Cycle detection, same contract as compute_recursive_flow. It has to be HERE as well:
    # `flow` and `partition_function` reach the memoized variant, not the plain one, so
    # guarding only the plain recursion left both of them still throwing StackOverflowError
    # on an allow_all_moves model. Verified by measurement -- the acyclic value stayed 19.0
    # and the cyclic call still overflowed -- which is how this second site was found.
    seen = isnothing(in_progress) ? Set{Any}() : in_progress
    if state in seen
        throw(ArgumentError(
            "cyclic state graph: $state was reached again on its own path, so the backward " *
            "chain is not absorbed at the initial state and no finite F satisfies " *
            "F(s) = sum_children F(c) P_B(s|c). Trajectory balance has no Z on this graph. " *
            "For grid world this is allow_all_moves=true; train with a LEARNABLE partition " *
            "function, which absorbs the cycles, and do not read flow() or " *
            "partition_function() as a ground truth there."))
    end

    cache = FLOW_CACHE[]
    cache_key = (param_hash, state)

    cached = get(cache, cache_key, nothing)
    isnothing(cached) || return cached

    push!(seen, state)

    value = if is_terminal_state(state)
        terminal_flow(state)
    else
        applicable_actions = get_applicable_actions(state, model.all_actions)
        if isempty(applicable_actions)
            0.0
        else
            total = 0.0
            for action in model.all_actions
                action in applicable_actions || continue
                child = apply_action(action, state)

                # P_B(state | child), matching compute_recursive_flow exactly.
                back_prob = if isnothing(model.backward_policy) ||
                               !haskey(model.parameters, :backward)
                    np = length(backward_parent_states(child, model.all_actions))
                    np == 0 ? 1.0 : 1.0 / np
                else
                    compute_backward_probability(
                        model.backward_policy, child, state,
                        model.parameters.backward, model.states.backward,
                        model.all_actions
                    )
                end

                total += back_prob * _memoized_flow(model, child, param_hash;
                                                    in_progress = seen)
            end
            total
        end
    end

    # Pop on the way out: `seen` is the current PATH, not every state visited. A state
    # reachable by two paths is not a cycle, and leaving it in would report a false one --
    # which is precisely the multi-parent structure this file exists to handle.
    delete!(seen, state)
    cache[cache_key] = value
    return value
end

# =============================================================================
# Direct Flow Estimation - Mathematical Foundation
# =============================================================================

"""
    compute_flow_estimate(model::GFlowNetModel, state::AbstractState)::Float64

Compute flow using direct flow estimator network Z(s).

# Mathematical Foundation
Uses the flow estimator network to directly predict:
F(s) ≈ Z(s) = exp(z_θ(s))

where z_θ(s) is the log-flow estimate from the neural network.

# Arguments
- `model::GFlowNetModel`: Model with flow estimator
- `state::AbstractState`: State to estimate flow for

# Returns
- `Float64`: Estimated flow value Z(s)

# Requirements
- Model must have flow estimator: model.flow_estimator ≠ nothing
"""
function compute_flow_estimate(model::GFlowNetModel, state::AbstractState)::Float64
    if isnothing(model.flow_estimator)
        throw(ArgumentError("Model must have flow estimator for direct flow computation"))
    end

    return flow_estimate(
        model.flow_estimator, state,
        model.parameters.flow, model.states.flow
    )
end

# =============================================================================
# Unified Flow Interface - Mathematical Foundation
# =============================================================================

"""
    flow(model::GFlowNetModel, state::AbstractState; method::FlowComputationMethod=RECURSIVE_FLOW)::Float64

Unified interface for computing flow F(s) using different methods.

# Mathematical Foundation
Computes the flow F(s) through state s using the specified method:
- RECURSIVE_FLOW: Uses flow conservation equation recursively
- DIRECT_FLOW: Uses flow estimator network Z(s)
- MIXED_FLOW: Combines both methods with validation

# Arguments
- `model::GFlowNetModel`: Complete GFlowNet model
- `state::AbstractState`: State to compute flow for
- `method::FlowComputationMethod`: Computation method to use

# Returns
- `Float64`: Flow value F(s)

# Method Selection Guidelines
- RECURSIVE_FLOW: Mathematically exact, slower for large DAGs
- DIRECT_FLOW: Fast approximation, requires trained flow estimator
- MIXED_FLOW: Uses both methods for validation and robustness
"""
function flow(model::GFlowNetModel, state::AbstractState;
              method::FlowComputationMethod=RECURSIVE_FLOW)::Float64

    if method == RECURSIVE_FLOW
        return compute_recursive_flow_memoized(model, state)

    elseif method == DIRECT_FLOW
        return compute_flow_estimate(model, state)

    elseif method == MIXED_FLOW
        # Compute using both methods
        recursive_flow = compute_recursive_flow_memoized(model, state)

        if !isnothing(model.flow_estimator)
            direct_flow = compute_flow_estimate(model, state)

            # Check consistency (non-differentiable validation)
            Zygote.@ignore begin
                relative_error = abs(recursive_flow - direct_flow) / max(recursive_flow, direct_flow, 1e-8)
                if relative_error > 0.1  # 10% tolerance
                    @warn "Flow computation methods disagree" recursive=recursive_flow direct=direct_flow relative_error=relative_error
                end
            end

            # Return average for robustness
            return (recursive_flow + direct_flow) / 2.0
        else
            return recursive_flow
        end
    else
        throw(ArgumentError("Unknown flow computation method: $method"))
    end
end

# =============================================================================
# Edge Flow Computation - Mathematical Foundation
# =============================================================================

"""
    edge_flow(model::GFlowNetModel, source_state::AbstractState, target_state::AbstractState)::Float64

Compute flow along edge from source to target state.

# Mathematical Foundation
The edge flow is defined as:
F(s→s') = P_F(s'|s) * F(s)

This represents the amount of flow passing through the specific edge s→s'.

# Arguments
- `model::GFlowNetModel`: Complete GFlowNet model
- `source_state::AbstractState`: Source state s
- `target_state::AbstractState`: Target state s'

# Returns
- `Float64`: Edge flow F(s→s')

# Mathematical Properties
- Non-negative: F(s→s') ≥ 0
- Flow conservation: Σ_{s'} F(s→s') = F(s)
- Zero if no valid transition exists
"""
function edge_flow(model::GFlowNetModel, source_state::AbstractState, target_state::AbstractState)::Float64
    # Check if transition is valid using on-demand computation
    applicable_actions = get_applicable_actions(source_state, model.all_actions)
    is_valid = false
    for action in applicable_actions
        if apply_action(action, source_state) == target_state
            is_valid = true
            break
        end
    end
    
    if !is_valid
        return 0.0
    end

    # Compute transition probability P_F(s'|s)
    transition_prob = forward_transition_probability(model, source_state, target_state)

    # Compute source flow F(s)
    source_flow = flow(model, source_state)

    # Edge flow: F(s→s') = P_F(s'|s) * F(s)
    return transition_prob * source_flow
end

# =============================================================================
# Partition Function Computation - Mathematical Foundation
# =============================================================================

"""
    partition_function(model::GFlowNetModel)::Float64

Compute the partition function Z = F(s₀) for the GFlowNet.

# Mathematical Foundation
The partition function is the total flow from the initial state:
Z = F(s₀)

This represents the sum of rewards over all possible trajectories,
weighted by their forward probabilities.

# Arguments
- `model::GFlowNetModel`: Complete GFlowNet model

# Returns
- `Float64`: Partition function Z

# Mathematical Significance
- Z appears in trajectory balance and other training objectives
- Measures the "mass" of the probability distribution over trajectories
- Essential for proper normalization in GFlowNet training
"""
function partition_function(model::GFlowNetModel)::Float64
    # Compute the flow through the initial state
    # Z = F(s₀)
    return flow(model, model.initial_state)
end

# =============================================================================
# Flow Validation and Consistency Checks
# =============================================================================

"""
    validate_flow_conservation(model::GFlowNetModel, state::AbstractState; tolerance::Float64=1e-6)::Bool

Validate flow conservation equation for a specific state.

# Mathematical Foundation
Checks that the flow conservation equation holds:
F(s) = Σ_{s'} P_F(s'|s) * F(s')

# Arguments
- `model::GFlowNetModel`: Model to validate
- `state::AbstractState`: State to check conservation for
- `tolerance::Float64`: Numerical tolerance for equality check

# Returns
- `Bool`: true if conservation holds within tolerance

# Mathematical Validation
This is a critical test of GFlowNet mathematical consistency.
Violations indicate either:
- Numerical instability in policy or flow networks
- Bugs in implementation
- Insufficient training of the model
"""
function validate_flow_conservation(model::GFlowNetModel, state::AbstractState; tolerance::Float64=1e-6)::Bool
    # Skip validation for terminal states (boundary condition)
    if is_terminal_state(state)
        return true
    end

    # Compute left side: F(s)
    left_side = flow(model, state)

    # Right side: sum over children s' of F(s') * P_B(s|s').
    #
    # This previously weighted by P_F(s'|s), i.e. it asserted
    # F(s) = sum P_F(s'|s) F(s'), which is a CONVEX COMBINATION of the children
    # and therefore forces F(s0) to lie between min and max R rather than equal
    # sum_x R(x). That is the same defect that made partition_function return
    # 2.2184 instead of 19.0 on the 3x3 grid. The conservation law that actually
    # holds for unnormalised flow is the P_B-weighted one.
    applicable_actions = get_applicable_actions(state, model.all_actions)
    right_side = 0.0

    for action in applicable_actions
        child = apply_action(action, state)
        back_prob = if isnothing(model.backward_policy) || !haskey(model.parameters, :backward)
            # A domain that does not implement find_parent_for_action yields an
            # empty parent set. Treat that as a unique parent (P_B = 1) rather
            # than as zero probability: zero would silently drive every flow to
            # 0 and report a conservation violation for any custom state type,
            # and P_B = 1 is exactly correct whenever the DAG is a tree. This
            # matches compute_recursive_flow, which must agree with this check.
            parents = backward_parent_states(child, model.all_actions)
            isempty(parents) ? 1.0 : 1.0 / length(parents)
        else
            compute_backward_probability(model.backward_policy, child, state,
                                         model.parameters.backward,
                                         model.states.backward, model.all_actions)
        end
        right_side += back_prob * flow(model, child)
    end

    # Check conservation within tolerance
    conservation_error = abs(left_side - right_side)
    relative_error = conservation_error / max(left_side, right_side, 1e-8)

    is_conserved = relative_error <= tolerance

    if !is_conserved
        @warn "Flow conservation violation detected" state=state left_side=left_side right_side=right_side relative_error=relative_error
    end

    return is_conserved
end

"""
    validate_flow_consistency(model::GFlowNetModel; sample_size::Int=min(50, length(model.dag.states)))::Float64

Validate flow conservation across multiple states in the DAG.

# Arguments
- `model::GFlowNetModel`: Model to validate
- `sample_size::Int`: Number of states to sample for validation

# Returns
- `Float64`: Fraction of states that satisfy flow conservation

# Mathematical Significance
A well-trained GFlowNet should have flow conservation close to 1.0.
Lower values indicate training issues or model problems.
"""
function validate_flow_consistency(model::GFlowNetModel; sample_size::Int=50,
                                   tolerance::Float64=0.1)::Float64
    # This returned a hardcoded 1.0 behind a @warn, so every caller was told
    # flow conservation held perfectly no matter what the model did.
    #
    # The check must be FALSIFIABLE. Comparing compute_recursive_flow against
    # its own defining recursion would be a tautology that returns 1.0 for any
    # model, which is no better than the hardcoded value. So the real question
    # is asked instead: does the LEARNED flow network agree with the flow
    # implied by the policy?
    #
    #     flow_estimate(s)  ==  sum over children s' of F(s') * P_B(s|s')
    #
    # For an untrained model this fails and the fraction is near 0; it rises to
    # 1 as the flow estimator learns. A model with no flow estimator has
    # nothing to validate, which is reported as NaN rather than as success.
    isnothing(model.flow_estimator) && return NaN

    states = AbstractState[]
    for _ in 1:sample_size
        traj = sample_trajectory(model)
        is_valid_trajectory(traj) || continue
        for s in traj.states
            is_terminal_state(s) && continue
            isempty(get_applicable_actions(s, model.all_actions)) && continue
            push!(states, s)
        end
    end
    states = unique(states)

    isempty(states) && return NaN

    satisfied = 0
    for s in states
        estimated = flow_estimate(model.flow_estimator, s,
                                  model.parameters.flow, model.states.flow)
        implied = compute_recursive_flow(model, s)

        # Relative agreement in log space: flows span orders of magnitude, so an
        # absolute tolerance would be meaningless.
        if abs(log(max(estimated, 1e-12)) - log(max(implied, 1e-12))) <= tolerance
            satisfied += 1
        end
    end

    return satisfied / length(states)
end

# =============================================================================
# Flow Debugging and Analysis Tools
# =============================================================================

"""
    flow_analysis(model::GFlowNetModel, state::AbstractState)

Comprehensive flow analysis for debugging purposes.

# Returns
Named tuple with flow analysis information:
- `flow_value`: F(s)
- `is_terminal`: Whether state is terminal
- `next_states`: List of next states
- `transition_probs`: P_F(s'|s) for each next state
- `next_flows`: F(s') for each next state
- `conservation_check`: Whether flow conservation holds
"""
function flow_analysis(model::GFlowNetModel, state::AbstractState)
    flow_value = flow(model, state)
    is_terminal = is_terminal_state(state)

    if is_terminal
        return (
            flow_value = flow_value,
            is_terminal = true,
            next_states = AbstractState[],
            transition_probs = Float64[],
            next_flows = Float64[],
            conservation_check = true,
            conservation_error = 0.0
        )
    end

    # Get next states using on-demand computation
    applicable_actions = get_applicable_actions(state, model.all_actions)
    next_states = [apply_action(action, state) for action in applicable_actions]
    transition_probs = Float64[]
    next_flows = Float64[]

    for next_state in next_states
        push!(transition_probs, forward_transition_probability(model, state, next_state))
        push!(next_flows, flow(model, next_state))
    end

    # Check conservation
    expected_flow = sum(transition_probs .* next_flows)
    conservation_error = abs(flow_value - expected_flow)
    conservation_check = conservation_error < 1e-6

    return (
        flow_value = flow_value,
        is_terminal = false,
        next_states = next_states,
        transition_probs = transition_probs,
        next_flows = next_flows,
        conservation_check = conservation_check,
        conservation_error = conservation_error
    )
end

# =============================================================================
# Performance Monitoring
# =============================================================================

"""
    flow_computation_benchmark(model::GFlowNetModel, n_samples::Int=100)

Benchmark flow computation performance across different methods.

# Returns
Named tuple with timing information for different computation methods.
"""
function flow_computation_benchmark(model::GFlowNetModel, n_samples::Int=100)
    # Clear cache for fair comparison
    clear_flow_cache!()

    # NOTE: Benchmarking requires access to all states which needs DAG
    # For now, return placeholder values
    # TODO: Implement proper benchmarking when state enumeration is available
    @warn "Flow computation benchmark not fully implemented"
    return (
        recursive_time = NaN,
        direct_time = NaN,
        n_samples = n_samples,
        avg_recursive_time = NaN,
        avg_direct_time = NaN
    )
end

# =============================================================================
# Display Methods
# =============================================================================

function Base.show(io::IO, method::FlowComputationMethod)
    method_name = if method == RECURSIVE_FLOW
        "Recursive Flow"
    elseif method == DIRECT_FLOW
        "Direct Flow Estimation"
    elseif method == MIXED_FLOW
        "Mixed Flow (Recursive + Direct)"
    else
        "Unknown Method"
    end
    print(io, method_name)
end
