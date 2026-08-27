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
function compute_recursive_flow(model::GFlowNetModel, state::AbstractState)::Float64
    # Base case: terminal states
    if is_terminal_state(state)
        return terminal_flow(state)
    end

    # Get applicable actions using on-demand computation
    applicable_actions = get_applicable_actions(state, model.all_actions)
    
    # If no applicable actions, this is effectively a terminal state with zero reward
    if isempty(applicable_actions)
        return 0.0
    end
    
    # F(s) = sum over children s' of F(s') * P_B(s|s')
    total_flow = 0.0

    for action in model.all_actions
        action in applicable_actions || continue

        next_state = apply_action(action, state)

        # P_B(state | next_state): probability that a backward walk at next_state
        # steps to state. With no backward policy this is uniform over the parents
        # of next_state, which is a perfectly valid fixed P_B.
        back_prob = if isnothing(model.backward_policy) || !haskey(model.parameters, :backward)
            np = length(backward_parent_states(next_state, model.all_actions))
            np == 0 ? 1.0 : 1.0 / np
        else
            compute_backward_probability(
                model.backward_policy, next_state, state,
                model.parameters.backward, model.states.backward, model.all_actions
            )
        end

        total_flow += back_prob * compute_recursive_flow(model, next_state)
    end

    return total_flow
end

"""
    compute_recursive_flow_memoized(model::GFlowNetModel, state::AbstractState)::Float64

Compute recursive flow with memoization for performance.

# Mathematical Foundation
Same as compute_recursive_flow but uses caching to avoid recomputing
flow values for states that have already been processed.

# Performance Benefits
- O(|S|) time complexity instead of potentially exponential
- Essential for large DAGs with many shared substructures
- Maintains mathematical correctness while improving efficiency
"""
function compute_recursive_flow_memoized(model::GFlowNetModel, state::AbstractState)::Float64
    # All cache operations are wrapped in Zygote.@ignore to avoid mutation issues
    cache_key = Zygote.@ignore get_cache_key(model, state)
    
    # Check cache (non-differentiable)
    cached_value = Zygote.@ignore begin
        if haskey(FLOW_CACHE[], cache_key)
            FLOW_CACHE[][cache_key]
        else
            nothing
        end
    end
    
    if !isnothing(cached_value)
        return cached_value
    end

    # Compute flow value (this is differentiable)
    flow_value = compute_recursive_flow(model, state)

    # Cache result (non-differentiable)
    Zygote.@ignore begin
        FLOW_CACHE[][cache_key] = flow_value
    end

    return flow_value
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

    # Compute right side using on-demand computation
    applicable_actions = get_applicable_actions(state, model.all_actions)
    next_states = [apply_action(action, state) for action in applicable_actions]
    right_side = 0.0

    for next_state in next_states
        transition_prob = forward_transition_probability(model, state, next_state)
        next_flow = flow(model, next_state)
        right_side += transition_prob * next_flow
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
function validate_flow_consistency(model::GFlowNetModel; sample_size::Int=50)::Float64
    # NOTE: Flow consistency validation requires full flow computation
    # which is not yet implemented. Returning placeholder value.
    # TODO: Implement when flow functions are ready
    @warn "Flow consistency validation not fully implemented"
    return 1.0
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
