using ..GFlowNet: DirectedAcyclicGraph, AbstractState, AbstractAction
using Graphs

"""
    CyclicFlowNetwork

Type representing a flow network that may contain cycles.
This generalizes the standard DAG-based GFlowNets.
"""
struct CyclicFlowNetwork{S<:AbstractState, A<:AbstractAction}
    graph::SimpleDiGraph
    states::Vector{S}
    actions::Vector{A}
    state_to_idx::Dict{S, Int}
    initial_state::S
    terminal_states::Vector{S}
    terminal_sink::S
    max_steps::Int
end

"""
    create_cyclic_network(initial_state::S, terminal_states::Vector{S}, 
                        terminal_sink::S, actions::Vector{A},
                        max_steps::Int=100) where {S<:AbstractState, A<:AbstractAction}

Create a CyclicFlowNetwork that allows for cycles in state transitions.
The max_steps parameter limits the maximum trajectory length to prevent infinite loops.
"""
function create_cyclic_network(initial_state::S, terminal_states::Vector{S}, 
                             terminal_sink::S, actions::Vector{A},
                             max_steps::Int=100) where {S<:AbstractState, A<:AbstractAction}
    
    # Create an initial network with just the states
    all_states = [initial_state; terminal_states; [terminal_sink]]
    unique_states = unique(all_states)
    
    state_to_idx = Dict(state => i for (i, state) in enumerate(unique_states))
    
    # Initialize an empty graph
    graph = SimpleDiGraph(length(unique_states))
    
    # Generate all possible transitions (potentially including cycles)
    for action in actions
        for state in unique_states
            if is_applicable(action, state)
                next_state = apply_action(action, state)
                if next_state in unique_states
                    add_edge!(graph, state_to_idx[state], state_to_idx[next_state])
                end
            end
        end
    end
    
    # Add edges from terminal states to sink
    for term_state in terminal_states
        add_edge!(graph, state_to_idx[term_state], state_to_idx[terminal_sink])
    end
    
    return CyclicFlowNetwork{S, A}(
        graph,
        unique_states,
        actions,
        state_to_idx,
        initial_state,
        terminal_states,
        terminal_sink,
        max_steps
    )
end

"""
    cyclic_trajectory_balance_loss(model, trajectories)

Compute the trajectory balance loss modified for cyclic GFlowNets.
"""
function cyclic_trajectory_balance_loss(model, trajectories)
    total_loss = 0.0
    n_trajectories = length(trajectories)
    
    for trajectory in trajectories
        # Last state in trajectory (before sink)
        final_state = trajectory.states[end]
        
        # Product of forward probabilities along the trajectory
        forward_prob_product = 1.0
        for i in 1:(length(trajectory.states)-1)
            source = trajectory.states[i]
            target = trajectory.states[i+1]
            prob = forward_transition_prob(model, source, target)
            forward_prob_product *= prob
        end
        
        # For cyclic networks, we adjust the loss to account for cycles
        # by introducing a termination probability at each step
        
        # Compute the reward of the final state
        final_reward = reward(final_state)
        
        # Compute Z (partition function)
        Z = isnothing(model.partition_function) ? 
            estimate_partition_function(model) : model.partition_function
        
        # Compute the ratio (should be 1 for perfect balance)
        ratio = (Z * forward_prob_product) / final_reward
        
        # Account for trajectory length in the loss (penalize very long trajectories)
        trajectory_length = length(trajectory.states)
        length_factor = exp(-trajectory_length / model.dag.max_steps)
        
        # Squared log error with length penalty
        log_ratio = log(ratio)
        total_loss += log_ratio^2 * length_factor
    end
    
    return total_loss / n_trajectories
end

"""
    sample_cyclic_trajectory(model, rng=nothing)

Sample a trajectory from a cyclic GFlowNet, with a maximum number of steps
to prevent infinite loops.
"""
function sample_cyclic_trajectory(model, rng=nothing)
    if isnothing(rng)
        rng = Random.default_rng()
    end
    
    trajectory = [model.dag.initial_state]
    current_state = model.dag.initial_state
    
    # Track visited states and their counts to detect cycles
    visited_counts = Dict(current_state => 1)
    
    # Limit trajectory length to prevent infinite loops
    for _ in 1:model.dag.max_steps
        if current_state ∈ model.dag.terminal_states
            break
        end
        
        # Get next state probabilities
        features = state_to_features(current_state)
        logits = model.forward_policy.model(features)
        
        # Get all possible next states
        next_states = get_next_states(model.dag, current_state)
        if isempty(next_states)
            break
        end
        
        next_state_indices = [model.dag.state_to_idx[s] for s in next_states]
        relevant_logits = logits[next_state_indices]
        probs = softmax(relevant_logits)
        
        # Sample next state
        next_state_idx = sample(1:length(next_states), Weights(probs))
        next_state = next_states[next_state_idx]
        
        # Update visited counts
        visited_counts[next_state] = get(visited_counts, next_state, 0) + 1
        
        # Check for excessive cycling
        if visited_counts[next_state] > 5
            # If we're in a tight cycle, increase probability of termination
            if rand(rng) < 0.5
                # Find a terminal state to force exit
                for term_state in model.dag.terminal_states
                    if term_state ∈ next_states
                        next_state = term_state
                        break
                    end
                end
            end
        end
        
        push!(trajectory, next_state)
        current_state = next_state
    end
    
    # If we hit the max steps without reaching a terminal state,
    # we need to ensure we end at a terminal state for the trajectory balance to work
    if current_state ∉ model.dag.terminal_states
        # Find closest terminal state
        closest_terminal = find_closest_terminal(model.dag, current_state)
        push!(trajectory, closest_terminal)
    end
    
    return Trajectory(trajectory)
end

"""
    find_closest_terminal(dag, state)

Find the closest terminal state to the given state.
"""
function find_closest_terminal(dag, state)
    # Simple implementation - return the first terminal state
    # A more sophisticated implementation would use graph distances
    return first(dag.terminal_states)
end 