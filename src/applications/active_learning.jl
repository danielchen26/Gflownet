using ..GFlowNet: AbstractState, AbstractAction, state_to_features, is_applicable, apply_action, reward
using LinearAlgebra
using Statistics

"""
    ExperimentData

Data structure for experimental information. Used with composition pattern
to create domain-specific states.

# Fields
- `experiments`: Vector of experiment indices
- `features`: Matrix of experimental features
"""
struct ExperimentData
    experiments::Vector{Int}
    features::Matrix{Float64}
end

"""
    ExperimentState <: AbstractState

State representation for experiment design in active learning.
"""
struct ExperimentState <: AbstractState
    experiments::Vector{Int}  # Indices of experiments performed
    max_experiments::Int      # Maximum number of experiments allowed
    is_terminal::Bool
end

"""
    ExperimentAction <: AbstractAction

Action representation for experiment selection.
"""
abstract type ExperimentAction <: AbstractAction end

"""
    SelectExperimentAction <: ExperimentAction

Action to select a specific experiment.
"""
struct SelectExperimentAction <: ExperimentAction
    experiment_idx::Int
end

"""
    TerminateExperimentAction <: ExperimentAction

Action to terminate the experiment selection process.
"""
struct TerminateExperimentAction <: ExperimentAction end

# Implementation of required interface functions

"""
    is_applicable(action::SelectExperimentAction, state::ExperimentState)

Check if selecting an experiment is valid.
"""
function is_applicable(action::SelectExperimentAction, state::ExperimentState)
    # Cannot modify terminal states
    if state.is_terminal
        return false
    end

    # Cannot exceed maximum number of experiments
    if length(state.experiments) >= state.max_experiments
        return false
    end

    # Cannot select an experiment already selected
    if action.experiment_idx in state.experiments
        return false
    end

    return true
end

"""
    is_applicable(action::TerminateExperimentAction, state::ExperimentState)

Check if termination is valid.
"""
function is_applicable(action::TerminateExperimentAction, state::ExperimentState)
    # Can terminate if not already terminated and at least one experiment is selected
    return !state.is_terminal && !isempty(state.experiments)
end

"""
    apply_action(action::SelectExperimentAction, state::ExperimentState)

Apply the action to select an experiment.
"""
function apply_action(action::SelectExperimentAction, state::ExperimentState)
    new_experiments = copy(state.experiments)
    push!(new_experiments, action.experiment_idx)

    return ExperimentState(new_experiments, state.max_experiments, false)
end

"""
    apply_action(action::TerminateExperimentAction, state::ExperimentState)

Apply the action to terminate experiment selection.
"""
function apply_action(action::TerminateExperimentAction, state::ExperimentState)
    return ExperimentState(copy(state.experiments), state.max_experiments, true)
end

"""
    state_to_features(state::ExperimentState, experiment_features::Matrix{Float64})

Convert an experiment state to a feature vector, using experiment features.
"""
function state_to_features(state::ExperimentState, experiment_features::Matrix{Float64})
    # Number of possible experiments
    n_experiments = size(experiment_features, 1)

    # Create a binary vector indicating which experiments have been selected
    selection_vector = zeros(Float32, n_experiments)
    selection_vector[state.experiments] .= 1.0

    # Add summary features of selected experiments
    if isempty(state.experiments)
        mean_features = zeros(Float32, size(experiment_features, 2))
        diversity = 0.0
    else
        # Mean features across selected experiments
        selected_features = experiment_features[state.experiments, :]
        mean_features = Float32.(vec(mean(selected_features, dims=1)))

        # Diversity of selected experiments (average pairwise distance)
        diversity = 0.0
        if length(state.experiments) > 1
            n_pairs = 0
            for i in 1:length(state.experiments)
                for j in i+1:length(state.experiments)
                    diversity += norm(selected_features[i, :] - selected_features[j, :])
                    n_pairs += 1
                end
            end
            diversity /= n_pairs
        end
    end

    # Create feature vector
    features = [
        selection_vector;
        mean_features;
        Float32(diversity);
        Float32(length(state.experiments));
        Float32(state.max_experiments);
        Float32(state.is_terminal)
    ]

    return features
end

"""
    reward(state::ExperimentState, experiment_features::Matrix{Float64}, experiment_values::Vector{Float64})

Calculate the reward based on the information gain from selected experiments.
"""
function reward(state::ExperimentState, experiment_features::Matrix{Float64}, experiment_values::Vector{Float64})
    if !state.is_terminal
        return 0.0
    end

    # This is a simplified reward for active learning
    # In practice, this would depend on the specific application and model

    # If no experiments selected, return minimal reward
    if isempty(state.experiments)
        return 0.1
    end

    # Reward components:
    # 1. Information gain (simplified as variance of selected experiments)
    selected_values = experiment_values[state.experiments]
    value_variance = var(selected_values)

    # 2. Diversity of selected experiments (average pairwise distance)
    selected_features = experiment_features[state.experiments, :]
    diversity = 0.0
    if length(state.experiments) > 1
        n_pairs = 0
        for i in 1:length(state.experiments)
            for j in i+1:length(state.experiments)
                diversity += norm(selected_features[i, :] - selected_features[j, :])
                n_pairs += 1
            end
        end
        diversity /= n_pairs
    end

    # 3. Reward efficiency (value per experiment)
    efficiency = sum(selected_values) / length(state.experiments)

    # Combine components
    reward_value = value_variance * (1.0 + diversity) * efficiency

    # Ensure positive reward
    return max(0.1, reward_value)
end

"""
    create_experiment_actions(n_experiments::Int)

Create a set of possible experiment selection actions.
"""
function create_experiment_actions(n_experiments::Int)
    actions = ExperimentAction[]

    # Add experiment selection actions
    for i in 1:n_experiments
        push!(actions, SelectExperimentAction(i))
    end

    # Add terminate action
    push!(actions, TerminateExperimentAction())

    return actions
end

"""
    create_initial_experiment_state(max_experiments::Int)

Create the initial state for experiment selection.
"""
function create_initial_experiment_state(max_experiments::Int)
    return ExperimentState(Int[], max_experiments, false)
end

"""
    simulate_experiment_data(n_experiments::Int, feature_dim::Int)

Simulate experiment features and values for testing.
"""
function simulate_experiment_data(n_experiments::Int, feature_dim::Int)
    # Generate random experiment features
    experiment_features = randn(n_experiments, feature_dim)

    # Generate experiment values
    # In this simulation, we'll make the values relate to the features
    # in a nonlinear way to simulate complex dependencies

    # Random weight matrix
    weights = randn(feature_dim, 1)

    # Linear component
    linear_values = experiment_features * weights

    # Nonlinear component (interactions between features)
    nonlinear_values = zeros(n_experiments)
    for i in 1:n_experiments
        for j in 1:feature_dim
            for k in j:feature_dim
                nonlinear_values[i] += 0.1 * experiment_features[i, j] * experiment_features[i, k]
            end
        end
    end

    # Combine components and add noise
    experiment_values = vec(linear_values) + nonlinear_values + 0.2 * randn(n_experiments)

    return experiment_features, experiment_values
end
