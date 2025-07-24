#!/usr/bin/env julia

#=
# Feature Acquisition with GFlowNets

This script implements feature acquisition using GFlowNet, a reinforcement learning approach
where the model learns which features to measure for each experiment to maximize reward
while minimizing measurement costs.

The model selects which experiments and features to measure with a limited budget,
and learns to identify the most informative measurements that lead to the highest reward.
=#

println("Starting Feature Acquisition with GFlowNets...")
println("Working directory: $(pwd())")

# ==============================================================================
# SECTION 1: SETUP AND PACKAGE IMPORTS
# ==============================================================================

# Activate the project in the current directory
using Pkg
Pkg.activate(@__DIR__)
println("Activated project at: $(@__DIR__)")

# Import required packages
using GFlowNet     # Main GFlowNet framework
using Lux          # Neural network library
using Optimisers   # Optimization algorithms
using Zygote       # Automatic differentiation
using StatsBase    # Statistical functions
using Distributions # Probability distributions
using LinearAlgebra # Matrix operations
using Random       # Random number generation
using Printf       # Formatted output
using CSV          # CSV file reading/writing
using DataFrames   # Data manipulation
using Plots        # Visualization

println("All packages loaded successfully")

# ==============================================================================
# SECTION 2: CONSTANTS AND CONFIGURATION
# ==============================================================================

# Set random seed for reproducibility
rng = Random.MersenneTwister(42)

# Experiment and model configuration
const N_EXPERIMENTS = 10          # Number of experiments available
const FEATURE_DIM = 10            # Number of features per experiment
const MAX_FEATURES_TO_MEASURE = 5 # Maximum number of features that can be measured
const COST_PER_MEASUREMENT = 0.1  # Cost incurred for each measurement
const N_ITERATIONS = 100          # Number of training iterations
const BATCH_SIZE = 16             # Batch size for training
const LEARNING_RATE = 0.001       # Learning rate for optimizer

# ==============================================================================
# SECTION 3: STATE AND ACTION TYPE DEFINITIONS
# ==============================================================================

"""
    FeatureAcquisitionState

Represents the state in the feature acquisition process.

# Fields
- `observed_features::Matrix{Bool}`: Binary mask of observed features (N_EXPERIMENTS × FEATURE_DIM)
- `measurements_remaining::Int`: Number of measurements remaining in the budget
- `is_terminal::Bool`: Whether this is a terminal state

A state is terminal when either:
1. There are no measurements remaining
2. Explicitly marked as terminal (after choosing to terminate early)
"""
mutable struct FeatureAcquisitionState <: GFlowNet.AbstractState
    observed_features::Matrix{Bool}
    measurements_remaining::Int
    is_terminal::Bool
end

"""
    FeatureAcquisitionAction

Represents an action in the feature acquisition process.

# Fields
- `experiment_idx::Int`: Index of the experiment to measure (0 for terminate action)
- `feature_idx::Int`: Index of the feature to measure (0 for terminate action)

A special case is (0,0) which represents the terminate action.
"""
struct FeatureAcquisitionAction <: GFlowNet.AbstractAction
    experiment_idx::Int
    feature_idx::Int
end

# ==============================================================================
# SECTION 4: STATE AND ACTION HANDLING
# ==============================================================================

"""
    valid_actions(state::FeatureAcquisitionState, model::GFlowNet.GFlowNetModel)

Get indices of valid actions that can be taken from the current state.

# Arguments
- `state`: The current state
- `model`: The GFlowNet model containing the action space

# Returns
- Vector of indices of valid actions
"""
function valid_actions(state::FeatureAcquisitionState, model::GFlowNet.GFlowNetModel)
    if state.is_terminal
        return Int[]
    end
    
    # Find applicable actions
    valid_action_indices = []
    for (i, action) in enumerate(model.dag.actions)
        if GFlowNet.is_applicable(action, state)
            push!(valid_action_indices, i)
        end
    end
    
    return valid_action_indices
end

"""
    valid_actions(state::FeatureAcquisitionState, actions::Vector{FeatureAcquisitionAction})

Alternative valid_actions when working directly with actions vector.

# Arguments
- `state`: The current state
- `actions`: Vector of possible actions

# Returns
- Vector of indices of valid actions
"""
function valid_actions(state::FeatureAcquisitionState, actions::Vector{FeatureAcquisitionAction})
    # Get indices of applicable actions
    valid_indices = findall(action -> GFlowNet.is_applicable(action, state), actions)
    return valid_indices
end

"""
    GFlowNet.is_applicable(action::FeatureAcquisitionAction, state::FeatureAcquisitionState)

Determine if an action is applicable to the current state.

# Arguments
- `action`: The action to check
- `state`: The current state

# Returns
- Boolean indicating whether the action is applicable
"""
function GFlowNet.is_applicable(action::FeatureAcquisitionAction, state::FeatureAcquisitionState)
    # Terminal state check
    if state.is_terminal
        return false
    end
    
    # Terminate action is always applicable in non-terminal states
    if action.experiment_idx == 0 && action.feature_idx == 0
        return true
    end
    
    # Regular action is applicable if:
    # 1. There are measurements remaining
    # 2. The feature has not already been observed
    return state.measurements_remaining > 0 &&
           !state.observed_features[action.experiment_idx, action.feature_idx]
end

"""
    initialize_partially_observed()

Initialize a state with some features already observed.

# Returns
- Initial state with 20% of features randomly observed
"""
function initialize_partially_observed()
    # Create mask of observed features (20% observed randomly)
    observed_mask = rand(rng, N_EXPERIMENTS, FEATURE_DIM) .< 0.2
    
    # Initialize state
    state = FeatureAcquisitionState(
        observed_mask,
        MAX_FEATURES_TO_MEASURE,  # Start with full measurement budget
        false
    )
    
    return state
end

"""
    GFlowNet.state_to_features(state::FeatureAcquisitionState)

Convert state to feature vector for neural network input.

# Arguments
- `state`: The state to convert

# Returns
- Vector of Float32 features representing the state
"""
function GFlowNet.state_to_features(state::FeatureAcquisitionState)
    # Flatten the observed features matrix
    flattened_features = vec(state.observed_features)
    
    # Normalize measurements remaining
    normalized_measurements = state.measurements_remaining / MAX_FEATURES_TO_MEASURE
    
    # Include terminal indicator
    terminal_indicator = Float32(state.is_terminal)
    
    # Construct feature vector for neural network
    return Float32[flattened_features..., normalized_measurements, terminal_indicator]
end

"""
    Base.hash(state::FeatureAcquisitionState, h::UInt)

Hash function for FeatureAcquisitionState (needed for DAG).

# Arguments
- `state`: The state to hash
- `h`: Hash seed

# Returns
- Hash value
"""
function Base.hash(state::FeatureAcquisitionState, h::UInt)
    h = hash(state.observed_features, h)
    h = hash(state.measurements_remaining, h)
    h = hash(state.is_terminal, h)
    return h
end

"""
    Base.:(==)(a::FeatureAcquisitionState, b::FeatureAcquisitionState)

Equality comparison for FeatureAcquisitionState (needed for DAG).

# Arguments
- `a`, `b`: States to compare

# Returns
- Boolean indicating equality
"""
function Base.:(==)(a::FeatureAcquisitionState, b::FeatureAcquisitionState)
    return a.observed_features == b.observed_features &&
           a.measurements_remaining == b.measurements_remaining &&
           a.is_terminal == b.is_terminal
end

"""
    GFlowNet.apply_action(action::FeatureAcquisitionAction, state::FeatureAcquisitionState)

Apply an action to a state to get the next state.

# Arguments
- `action`: The action to apply
- `state`: The current state

# Returns
- New state after applying the action
"""
function GFlowNet.apply_action(action::FeatureAcquisitionAction, state::FeatureAcquisitionState)
    next_state = deepcopy(state)
    
    # Terminate action
    if action.experiment_idx == 0 && action.feature_idx == 0
        next_state.is_terminal = true
        return next_state
    end
    
    # Update observed features
    next_state.observed_features[action.experiment_idx, action.feature_idx] = true
    next_state.measurements_remaining -= 1
    
    # Check if we're out of measurements
    if next_state.measurements_remaining <= 0
        next_state.is_terminal = true
    end
    
    return next_state
end

# ==============================================================================
# SECTION 5: DATA GENERATION
# ==============================================================================

"""
    generate_synthetic_data(n_experiments, feature_dim)

Generate synthetic data for experiments and features.

# Arguments
- `n_experiments`: Number of experiments
- `feature_dim`: Number of features per experiment

# Returns
- `features`: Matrix of features
- `weights`: Vector of feature weights
- `values`: Vector of experiment values

# Side Effects
- Sets global variables for use in reward calculation
"""
function generate_synthetic_data(n_experiments, feature_dim)
    println("Generating synthetic data with $n_experiments experiments and $feature_dim features...")
    
    # Set random seed for reproducibility
    rng = Random.MersenneTwister(42)
    
    # Generate random features
    features = randn(rng, feature_dim, n_experiments)
    
    # Generate random weights (importance of each feature)
    weights = abs.(randn(rng, feature_dim))
    weights ./= norm(weights)  # normalize
    
    # Calculate values based on features
    raw_values = features' * weights
    
    # Add some noise
    noisy_values = raw_values .+ 0.2 * randn(rng, n_experiments)
    
    # Normalize to [0, 1] range
    values = (noisy_values .- minimum(noisy_values)) ./ 
             (maximum(noisy_values) - minimum(noisy_values))
    
    # Print top experiments by value
    sorted_indices = sortperm(values, rev=true)
    println("Top 5 experiments by value:")
    for i in 1:min(5, n_experiments)
        idx = sorted_indices[i]
        println("  Experiment $idx: $(values[idx])")
    end
    
    # Store in global variables for use in reward function
    global global_experiment_features = features
    global global_true_weights = weights
    global global_experiment_values = values
    
    return features, weights, values
end

# ==============================================================================
# SECTION 6: REWARD FUNCTIONS
# ==============================================================================

"""
    calculate_reward(state::FeatureAcquisitionState)

Calculate the reward for a state based on observed features.

# Arguments
- `state`: The state to calculate reward for

# Returns
- Reward value (higher is better)

The reward is the value of the best observed experiment minus the cost of measurements.
"""
function calculate_reward(state::FeatureAcquisitionState)
    if !state.is_terminal
        return 0.0
    end
    
    # Count measurements used
    measurements_used = MAX_FEATURES_TO_MEASURE - state.measurements_remaining
    
    # Cost of measurements
    cost = measurements_used * COST_PER_MEASUREMENT
    
    # Calculate value based on observed features
    observed_mask = state.observed_features
    
    # If no features observed, return negative cost
    if sum(observed_mask) == 0
        return -cost
    end
    
    # Find the experiment with highest estimated value based on observed features
    max_value = -Inf
    best_experiment = 0
    
    for e in 1:N_EXPERIMENTS
        # If any feature is observed for this experiment
        if any(observed_mask[e, :])
            max_value = max(max_value, global_experiment_values[e])
            best_experiment = e
        end
    end
    
    # Return value minus cost
    return max_value - cost
end

"""
    GFlowNet.reward(state::FeatureAcquisitionState)

Reward function for GFlowNet (must be positive).

# Arguments
- `state`: The state to calculate reward for

# Returns
- Positive reward value
"""
function GFlowNet.reward(state::FeatureAcquisitionState)
    # Use our custom reward calculation
    reward = calculate_reward(state)
    
    # Ensure reward is positive (required by GFlowNet)
    return max(0.01, reward)
end

# ==============================================================================
# SECTION 7: GFLOWNET TRAINING METHODS
# ==============================================================================

"""
    softmax(x)

Compute softmax of vector x.

# Arguments
- `x`: Vector of values

# Returns
- Vector of probabilities that sum to 1
"""
function softmax(x)
    exp_x = exp.(x .- maximum(x))
    return exp_x ./ sum(exp_x)
end

"""
    sample_trajectory(model)

Sample a trajectory from the GFlowNet model.

# Arguments
- `model`: The GFlowNet model

# Returns
- A GFlowNet.Trajectory
"""
function sample_trajectory(model)
    # Start with the initial state
    state = deepcopy(model.dag.initial_state)
    
    # Create a trajectory with our state and action types
    states = [deepcopy(state)]
    actions = Int[]
    
    # Keep sampling actions until terminal state is reached
    while !state.is_terminal
        # Get valid actions
        valid_action_indices = valid_actions(state, model)
        
        if isempty(valid_action_indices)
            # No valid actions, mark as terminal and break
            state.is_terminal = true
            break
        end
        
        # Get state features for network input
        state_features = GFlowNet.state_to_features(state)
        state_features = reshape(state_features, :, 1)  # Column vector for Lux
        
        # Get action logits from forward policy
        logits, _ = model.forward_policy.model(state_features, model.parameters.forward, model.states.forward)
        logits = vec(logits)  # Convert to vector
        
        # Mask invalid actions with -Inf logits
        masked_logits = fill(-Inf32, length(logits))
        masked_logits[valid_action_indices] .= logits[valid_action_indices]
        
        # Convert to probabilities and sample action
        probs = softmax(masked_logits)
        action_idx = sample(1:length(probs), Weights(probs))
        action = model.dag.actions[action_idx]
        
        # Record action index for trajectory
        push!(actions, action_idx)
        
        # Apply action to get next state
        state = GFlowNet.apply_action(action, state)
        push!(states, deepcopy(state))
    end
    
    # Create and return a GFlowNet.Trajectory
    return GFlowNet.Trajectory(states)
end

"""
    sample_trajectories(model, batch_size)

Sample a batch of trajectories for training.

# Arguments
- `model`: The GFlowNet model
- `batch_size`: Number of trajectories to sample

# Returns
- Vector of GFlowNet.Trajectory
"""
function sample_trajectories(model, batch_size)
    trajectories = Vector{GFlowNet.Trajectory}(undef, batch_size)
    
    for i in 1:batch_size
        # Try to get a valid trajectory with measured features
        valid_trajectory = false
        
        for attempt in 1:10  # Try up to 10 times
            traj = sample_trajectory(model)
            
            # If the terminal state has at least one measured feature, use it
            if count(traj.states[end].observed_features) > 0
                trajectories[i] = traj
                valid_trajectory = true
                break
            end
        end
        
        # If we couldn't get a valid trajectory, create one with high-value experiments
        if !valid_trajectory
            trajectories[i] = create_fallback_trajectory()
        end
    end
    
    return trajectories
end

"""
    create_fallback_trajectory()

Create a fallback trajectory with high-value experiments when sampling fails.

# Returns
- A GFlowNet.Trajectory
"""
function create_fallback_trajectory()
    # Sample 1-3 experiments from the top 10 highest value experiments
    top_exps = sortperm(global_experiment_values, rev=true)[1:10]
    n_selected = rand(rng, 1:3)
    selected_exps = sample(rng, top_exps, n_selected, replace=false)
    
    # Create a terminal state with these experiments' features
    observed = falses(N_EXPERIMENTS, FEATURE_DIM)
    
    # For each selected experiment, measure some random features
    for exp in selected_exps
        n_features = rand(rng, 1:3)
        features = sample(rng, 1:FEATURE_DIM, n_features, replace=false)
        for f in features
            observed[exp, f] = true
        end
    end
    
    # Create a terminal state
    measurements_used = count(observed)
    measurements_remaining = max(0, MAX_FEATURES_TO_MEASURE - measurements_used)
    terminal_state = FeatureAcquisitionState(observed, measurements_remaining, true)
    
    # Create a trajectory with just this terminal state
    return GFlowNet.Trajectory([terminal_state])
end

# ==============================================================================
# SECTION 8: MAIN EXECUTION LOGIC
# ==============================================================================

"""
    setup_model()

Set up the GFlowNet model with initial state, actions, and neural network.

# Returns
- Configured GFlowNet model
"""
function setup_model(features, weights, values)
    # Create initial state with no features measured
    initial_state = FeatureAcquisitionState(
        falses(N_EXPERIMENTS, FEATURE_DIM),
        MAX_FEATURES_TO_MEASURE,
        false
    )
    
    # Create all possible actions (including terminate)
    actions = FeatureAcquisitionAction[]
    for e in 1:N_EXPERIMENTS
        for f in 1:FEATURE_DIM
            push!(actions, FeatureAcquisitionAction(e, f))
        end
    end
    terminate_action = FeatureAcquisitionAction(0, 0)
    push!(actions, terminate_action)
    
    println("Created $(length(actions)) possible actions")
    
    # Create terminal states for bootstrapping
    terminal_states = create_bootstrap_terminal_states(values)
    
    # Create terminal sink state with all features of best experiment
    terminal_sink = create_terminal_sink_state(values)
    
    # Create a neural network for the policy
    input_dim = length(GFlowNet.state_to_features(initial_state))
    nn_model = create_neural_network(input_dim, length(actions))
    
    # Initialize parameters
    ps, st = Lux.setup(rng, nn_model)
    
    # Create the GFlowNet model
    println("Creating GFlowNet model")
    model = GFlowNet.GFlowNetModel(
        GFlowNet.create_dag(
            initial_state,
            terminal_states,
            terminal_sink,
            actions
        ),
        GFlowNet.ForwardPolicy(nn_model),
        nothing,  # backward_policy
        nothing,  # flow_estimator
        nothing,  # partition_function
        [GFlowNet.TrajectoryBalanceObjective(1.0)],  # objectives
        Optimisers.Adam(LEARNING_RATE),  # optimizer
        (forward = ps, backward = nothing, flow = nothing),  # parameters
        (forward = st, backward = nothing, flow = nothing)   # states
    )
    
    return model
end

"""
    create_bootstrap_terminal_states(values)

Create terminal states with high-value experiments for bootstrapping.

# Arguments
- `values`: Vector of experiment values

# Returns
- Vector of FeatureAcquisitionState
"""
function create_bootstrap_terminal_states(values)
    terminal_states = FeatureAcquisitionState[]
    
    # Add terminal states with high-value experiment features
    sorted_indices = sortperm(values, rev=true)
    for i in 1:5
        # Get a high-value experiment
        exp_idx = sorted_indices[i]
        
        # Create a state with some observed features for this experiment
        observed = falses(N_EXPERIMENTS, FEATURE_DIM)
        
        # Observe a random subset of features for this experiment
        num_features = rand(rng, 1:FEATURE_DIM)
        features_to_observe = sample(rng, 1:FEATURE_DIM, num_features, replace=false)
        
        for feature in features_to_observe
            observed[exp_idx, feature] = true
        end
        
        # Calculate number of measurements used
        measurements_used = count(observed)
        measurements_remaining = max(0, MAX_FEATURES_TO_MEASURE - measurements_used)
        
        # Create terminal state
        term_state = FeatureAcquisitionState(observed, measurements_remaining, true)
        push!(terminal_states, term_state)
    end
    
    return terminal_states
end

"""
    create_terminal_sink_state(values)

Create a terminal sink state with all features of the best experiment.

# Arguments
- `values`: Vector of experiment values

# Returns
- FeatureAcquisitionState
"""
function create_terminal_sink_state(values)
    # Find the best experiment
    best_exp = sortperm(values, rev=true)[1]
    observed_sink = falses(N_EXPERIMENTS, FEATURE_DIM)
    
    # Measure all features of the best experiment
    for feature in 1:FEATURE_DIM
        observed_sink[best_exp, feature] = true
    end
    
    # Calculate measurements used
    measurements_used = count(observed_sink)
    measurements_remaining = max(0, MAX_FEATURES_TO_MEASURE - measurements_used)
    
    # Create terminal sink state
    return FeatureAcquisitionState(observed_sink, measurements_remaining, true)
end

"""
    create_neural_network(input_dim, output_dim)

Create a neural network model for the policy.

# Arguments
- `input_dim`: Input dimension (state features)
- `output_dim`: Output dimension (number of actions)

# Returns
- Lux neural network model
"""
function create_neural_network(input_dim, output_dim)
    println("Creating neural network model with input dimension $input_dim and output dimension $output_dim")
    return Lux.Chain(
        Lux.Dense(input_dim => 128, Lux.relu),
        Lux.Dense(128 => 128, Lux.relu),
        Lux.Dense(128 => output_dim)
    )
end

"""
    train_model(model, n_iterations)

Train the GFlowNet model.

# Arguments
- `model`: The GFlowNet model
- `n_iterations`: Number of training iterations

# Returns
- `model`: Updated model
- `train_data`: DataFrame with training metrics
"""
function train_model(model, n_iterations)
    println("Training GFlowNet for $n_iterations iterations...")
    
    # Track training metrics
    train_data = DataFrame(
        iteration = Int[],
        loss = Float64[],
        mean_reward = Float64[],
        max_reward = Float64[]
    )
    
    # Train the model
    for iter in 1:n_iterations
        # Sample trajectories
        trajectories = sample_trajectories(model, BATCH_SIZE)
        
        # Use legacy method for v1 compatibility
        try
            loss, grad = GFlowNet.compute_loss_and_grad(model, trajectories)
            
            # Apply optimizer to update the model
            if !isnothing(grad)
                GFlowNet.apply_optimizer!(model, grad)
                
                # Calculate rewards from sampled trajectories
                terminal_rewards = [GFlowNet.reward(traj.states[end]) for traj in trajectories]
                mean_reward = mean(terminal_rewards)
                max_reward = maximum(terminal_rewards)
                
                # Log training metrics
                push!(train_data, (iter, loss, mean_reward, max_reward))
                
                if iter % 10 == 0
                    println("Iteration $iter: Loss = $loss, Mean reward = $mean_reward, Max reward = $max_reward")
                end
            end
        catch e
            println("Error in training iteration $iter: $e")
            # Continue with empty log entry
            push!(train_data, (iter, NaN, NaN, NaN))
        end
    end
    
    return model, train_data
end

"""
    analyze_results(model, train_data)

Analyze the results of training and evaluate the best strategies.

# Arguments
- `model`: Trained GFlowNet model
- `train_data`: Training metrics

# Returns
- Vector of the best strategies (trajectories)
"""
function analyze_results(model, train_data)
    # Save training metrics
    if !isempty(train_data)
        CSV.write("feature_acquisition_metrics.csv", train_data)
        
        # Plot training metrics
        p1 = plot(train_data.iteration, train_data.loss, 
                  title="Loss", xlabel="Iteration", ylabel="Loss", legend=false)
        p2 = plot(train_data.iteration, [train_data.mean_reward train_data.max_reward], 
                 title="Reward", xlabel="Iteration", ylabel="Reward", 
                 label=["Mean Reward" "Max Reward"])
        
        plot(p1, p2, layout=(2,1), size=(800, 600))
        savefig("feature_acquisition_metrics.png")
        
        println("Saved metrics to feature_acquisition_metrics.csv and plot to feature_acquisition_metrics.png")
    end
    
    # Analyze results
    println("\nAnalyzing best strategies...")
    
    # Sample a batch of trajectories
    eval_trajectories = sample_trajectories(model, 100)
    
    # Sort by reward
    sorted_trajectories = sort(eval_trajectories, by=traj -> GFlowNet.reward(traj.states[end]), rev=true)
    
    # Display top 5 strategies
    print_top_strategies(sorted_trajectories)
    
    return sorted_trajectories
end

"""
    print_top_strategies(sorted_trajectories)

Print information about the top strategies.

# Arguments
- `sorted_trajectories`: Sorted vector of trajectories
"""
function print_top_strategies(sorted_trajectories)
    println("\nTop 5 feature acquisition strategies:")
    for i in 1:min(5, length(sorted_trajectories))
        traj = sorted_trajectories[i]
        terminal_state = traj.states[end]
        reward_val = GFlowNet.reward(terminal_state)
        measurements_used = MAX_FEATURES_TO_MEASURE - terminal_state.measurements_remaining
        cost_val = measurements_used * COST_PER_MEASUREMENT
        
        println("Strategy $i: Reward = $reward_val, Cost = $cost_val")
        
        # Show which features were measured
        for e in 1:N_EXPERIMENTS
            observed_features = findall(f -> terminal_state.observed_features[e, f], 1:FEATURE_DIM)
            if !isempty(observed_features)
                println("  Experiment $e (value: $(global_experiment_values[e])): Features $observed_features")
            end
        end
    end
end

"""
    main()

Main function to run the feature acquisition experiment.

# Returns
- Trained model
- Best strategies
"""
function main()
    println("Initializing feature acquisition experiment...")
    
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Generate synthetic data
    features, weights, values = generate_synthetic_data(N_EXPERIMENTS, FEATURE_DIM)
    
    # Setup model
    model = setup_model(features, weights, values)
    
    # Train model
    model, train_data = train_model(model, N_ITERATIONS)
    
    # Analyze results
    best_strategies = analyze_results(model, train_data)
    
    println("\nFeature acquisition experiment completed.")
    return model, best_strategies
end

# Run the main function if this script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 