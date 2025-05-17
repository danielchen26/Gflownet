using GFlowNet
using Random
using Statistics
using Plots
using CSV
using DataFrames
using LinearAlgebra  # Added for normalize function
using Lux
using Optimisers
using NNlib
using StatsBase  # Added for sample function
using Lux: Chain, Dense  # Added for neural network layers
using Zygote  # Added for gradient computation
using Distributions  # Added for Weights
using Base.Iterators  # Added for product

# Constants for feature acquisition
const FEATURE_DIM = 10  # Number of features per experiment
const MAX_FEATURES_TO_MEASURE = 15  # Maximum number of features that can be measured
const rng = Random.MersenneTwister(42)  # Random number generator for reproducibility

"""
    PartialFeatureState

State representation for feature acquisition with partial observations.
"""
struct PartialFeatureState <: GFlowNet.AbstractState
    observed_features::Vector{Bool}  # Which features have been measured
    feature_values::Vector{Float64}  # Values of measured features
    initial_features::Vector{Bool}   # Which features were initially observed
    measurements_remaining::Int      # Number of measurements remaining
    is_terminal::Bool               # Whether this is a terminal state
end

"""
    MeasureFeatureAction

Action to measure a specific feature.
"""
struct MeasureFeatureAction <: GFlowNet.AbstractAction
    feature_idx::Int
end

"""
    TerminateAction

Action to terminate the feature acquisition process.
"""
struct TerminateAction <: GFlowNet.AbstractAction end

# Enhanced reward type for partial observations
struct PartialFeatureReward <: GFlowNet.RewardFunction
    experiment_values::Vector{Float64}
    cost_per_measurement::Float64
    value_weight::Float64
    cost_weight::Float64
    initial_observations::Vector{Bool}  # Changed from Matrix{Bool} to Vector{Bool}
end

"""
    create_initial_state(num_features::Int, max_steps::Int, observation_ratio::Float64, experiment_values::Vector{Float64})

Create an initial state for the feature acquisition task.
In v3, this initializes with observation_ratio of features already measured.
"""
function create_initial_state(num_features::Int, max_steps::Int, observation_ratio::Float64, experiment_values::Vector{Float64})
    # Initialize all features as unobserved
    observed_features = falses(num_features)
    feature_values = zeros(Float64, num_features)
    
    # Randomly observe some features based on observation_ratio
    num_initial = round(Int, num_features * observation_ratio)
    if num_initial > 0
        initial_indices = randperm(num_features)[1:num_initial]
        initial_features = falses(num_features)
        initial_features[initial_indices] .= true
        
        # Set initial feature values - using feature weights as values
        # This is more consistent with the structured data generation approach
        if !isempty(experiment_values) && length(experiment_values) >= num_initial
            # Use experiment values directly for the selected features
            feature_values[initial_indices] .= experiment_values[1:num_initial]
        elseif @isdefined(global_true_weights) && length(global_true_weights) >= num_features
            # Fallback to using the true feature weights (more meaningful than random)
            feature_values[initial_indices] .= global_true_weights[initial_indices]
            # Normalize to [0,1]
            if maximum(feature_values) > 0
                feature_values ./= maximum(feature_values)
            end
        else
            # Last resort fallback
            feature_values[initial_indices] .= rand(num_initial)
        end
        
        observed_features[initial_indices] .= true
    else
        initial_features = falses(num_features)
    end
    
    return PartialFeatureState(
        observed_features,
        feature_values,
        initial_features,
        max_steps,
        false
    )
end

"""
    get_valid_actions(state::PartialFeatureState)

Get all valid actions that can be taken from the current state.
"""
function get_valid_actions(state::PartialFeatureState)
    valid_actions = Vector{GFlowNet.AbstractAction}()
    
    # Can't remeasure already observed features
    for i in 1:size(state.observed_features, 1)
        if !state.observed_features[i] && state.measurements_remaining > 0
            push!(valid_actions, MeasureFeatureAction(i))
        end
    end
    
    # Add terminate action if we've made at least one new measurement
    # or if we have no measurements left
    if sum(state.observed_features .& .!state.initial_features) > 0 || 
       state.measurements_remaining <= 0
        push!(valid_actions, TerminateAction())
    end
    
    return valid_actions
end

"""
    apply_action(state::PartialFeatureState, action_idx::Int)

Apply an action to a state and return the resulting state.
"""
function apply_action(state::PartialFeatureState, action_idx::Int)
    if action_idx == length(state.feature_values) + 1
        # Terminate action
        return PartialFeatureState(
            state.observed_features,
            state.feature_values,
            state.initial_features,
            state.measurements_remaining,
            true
        )
    else
        # Measure feature action
        new_observed = copy(state.observed_features)
        new_observed[action_idx] = true
        
        # Update feature values if needed
        new_values = copy(state.feature_values)
        
        # Create new state with one less measurement remaining
        return PartialFeatureState(
            new_observed,
            new_values,
            state.initial_features,
            state.measurements_remaining - 1,
            state.measurements_remaining <= 1  # Terminal if no measurements left after this one
        )
    end
end

"""
    calculate_reward(final_state::PartialFeatureState, reward_fn::PartialFeatureReward)

Calculate the reward for a terminal state based on the value gained from new measurements
and the cost of those measurements. Updated to use actual feature values.
"""
function calculate_reward(final_state::PartialFeatureState, reward_fn::PartialFeatureReward)
    if !final_state.is_terminal
        return 0.0f0
    end
    
    # Calculate value gained from new measurements only
    initial_value = calculate_best_value(final_state.initial_features, reward_fn.experiment_values)
    final_value = calculate_best_value(final_state.observed_features, reward_fn.experiment_values)
    value_gained = final_value - initial_value
    
    # Count only new measurements for cost
    new_measurements = sum(final_state.observed_features .& .!final_state.initial_features)
    cost_penalty = convert(Float32, new_measurements * reward_fn.cost_per_measurement)
    
    # Combine value and cost with weights
    reward = convert(Float32, reward_fn.value_weight) * value_gained - 
             convert(Float32, reward_fn.cost_weight) * cost_penalty
             
    return max(0.0f0, reward)  # Ensure non-negative reward
end

"""
    calculate_best_value(features::Vector{Bool}, experiment_values::Vector{Float64})

Calculate the best value achievable given the observed features.
Updated to use actual feature values instead of binary observations.
"""
function calculate_best_value(features::Vector{Bool}, experiment_values::Vector{Float64})
    if !any(features)
        return 0.0f0
    end
    return convert(Float32, maximum(experiment_values[features]))
end

"""
    GFlowNet.state_to_features(state::PartialFeatureState)

Convert a PartialFeatureState to a feature vector for the neural network.
"""
function GFlowNet.state_to_features(state::PartialFeatureState)
    state_to_vector(state)  # Use our consistent vector representation
end

"""
    state_to_vector(state::PartialFeatureState)

Convert a PartialFeatureState to a vector representation for the neural network.
This is an alias for GFlowNet.state_to_features for consistency.
"""
function state_to_vector(state::PartialFeatureState)
    # Convert boolean vectors to Float32 with explicit typing
    observed = convert(Vector{Float32}, state.observed_features)
    values = convert(Vector{Float32}, state.feature_values)
    initial = convert(Vector{Float32}, state.initial_features)
    
    # Add metadata features with explicit typing
    metadata = Vector{Float32}([
        state.measurements_remaining / MAX_FEATURES_TO_MEASURE,  # Normalized remaining budget
        state.is_terminal ? 1.0f0 : 0.0f0  # Terminal state indicator
    ])
    
    # Concatenate all features into a single vector with explicit type
    return vcat(observed, values, initial, metadata)
end

"""
    logsumexp(x)

Compute log(sum(exp(x))) in a numerically stable way.
"""
function logsumexp(x)
    max_x = maximum(x)
    max_x + log(sum(exp.(x .- max_x)))
end

"""
    compute_loss_and_gradient(model, ps, st, trajectories, rewards, output_dim)

Compute the GFlowNet loss and gradients for a batch of trajectories.
Updated to handle continuous feature values and improved gradient computation.
"""
function compute_loss_and_gradient(model, ps, st, trajectories, rewards, output_dim)
    # Ensure all inputs are properly typed
    ps = convert(NamedTuple, ps)
    rewards = convert(Vector{Float32}, rewards)
    
    # Define loss function for Zygote
    function loss_fn(p)
        batch_losses = map(zip(trajectories, rewards)) do (traj, reward)
            # Process each state-action pair in the trajectory
            traj_loss = 0.0f0
            for i in 1:(length(traj.states)-1)
                state = traj.states[i]
                action = traj.actions[i]
                next_state = traj.states[i+1]
                
                # Convert state to feature vector
                features = state_to_vector(state)
                features = reshape(features, :, 1)
                
                # Get model output
                logits, _ = model(features, p, st)
                logits = vec(logits)
                
                # Compute log probability
                chosen_idx = action  # action is already the index
                
                # Compute log probability using logsumexp for numerical stability
                log_prob = logits[chosen_idx] - logsumexp(logits)
                
                traj_loss += -log_prob * Float32(reward)  # Negative log likelihood weighted by reward
            end
            return traj_loss
        end
        
        # Average loss over batch
        return sum(batch_losses) / length(trajectories)
    end
    
    # Compute gradients using Zygote with explicit typing
    val_and_grad = Zygote.gradient(loss_fn, ps)
    avg_loss = loss_fn(ps)
    grads = val_and_grad[1]
    
    # Ensure gradients are not nothing
    if isnothing(grads)
        @warn "Gradient computation returned nothing"
        return avg_loss, ps  # Return original parameters if gradient is nothing
    end
    
    return convert(Float32, avg_loss), grads
end

# Helper function for numerical stability
function logsumexp(x::AbstractVector{T}) where T
    max_x = maximum(x)
    max_x + log(sum(exp.(x .- max_x)))
end

"""
    apply_gradients(ps, grads)

Apply gradients to model parameters using the optimizer.
Updated with improved learning rate scheduling and gradient clipping.
"""
function apply_gradients(ps, grads)
    # Initialize optimizer if not already done
    if !@isdefined(opt_state)
        global opt_state = Optimisers.setup(Optimisers.Adam(0.001f0), ps)
    end
    
    # Calculate gradient norm for clipping
    grad_norm = sqrt(sum(sum(abs2.(g)) for g in Iterators.flatten(values.(values(grads)))))
    
    # Clip gradients if norm is too large
    if grad_norm > 1.0f0
        scale = 1.0f0 / grad_norm
        grads = map(layer -> map(g -> g .* scale, layer), grads)
    end
    
    # Update parameters using optimizer
    global opt_state, new_ps = Optimisers.update(opt_state, ps, grads)
    
    return new_ps
end

# Global step counter for learning rate scheduling
global_step = 0
get_global_step() = global_step
increment_global_step!() = (global global_step += 1)

"""
    run_feature_acquisition_v3(;
        num_features::Int=FEATURE_DIM,
        num_experiments::Int=100,
        max_steps::Int=15,
        cost_per_measurement::Float64=0.1,
        n_iterations::Int=1000,
        batch_size::Int=32,
        observation_ratio::Float64=0.5,
        output_prefix::String="v3_results"
    )

Runs the feature acquisition process using GFlowNets with partial observations.
Updated to handle continuous feature values and improved training process.
"""
function run_feature_acquisition_v3(;
    num_features::Int=FEATURE_DIM,
    num_experiments::Int=100,
    max_steps::Int=15,
    cost_per_measurement::Float64=0.05,
    n_iterations::Int=1000,
    batch_size::Int=32,
    observation_ratio::Float64=0.5,
    output_prefix::String="v3_results"
)
    println("Starting run_feature_acquisition_v3 with:")
    println("  num_features: $num_features")
    println("  num_experiments: $num_experiments")
    println("  max_steps: $max_steps")
    println("  cost_per_measurement: $cost_per_measurement")
    println("  n_iterations: $n_iterations")
    println("  batch_size: $batch_size")
    println("  observation_ratio: $observation_ratio")
    
    # Initialize metrics storage
    losses = Float64[]
    mean_rewards = Float64[]
    max_rewards = Float64[]
    value_discoveries = Float32[]
    measurement_efficiencies = Float32[]
    
    # Temporary arrays for each iteration
    iter_value_discoveries = Float32[]
    iter_measurement_efficiencies = Float32[]

    println("\nGenerating experiment values...")
    # Generate experiment values and store them globally
    global global_experiment_values = generate_experiment_values(num_features, num_experiments)
    global global_cost_per_measurement = cost_per_measurement
    
    println("Setting up reward parameters...")
    # Create reward function
    reward_fn = PartialFeatureReward(
        global_experiment_values,
        cost_per_measurement,
        1.0,  # value_weight
        1.0,  # cost_weight
        falses(num_features)  # initial_observations - now a vector
    )

    println("\nCreating initial state...")
    # Create initial state with experiment values
    initial_state = create_initial_state(
        num_features,
        max_steps,
        observation_ratio,
        global_experiment_values
    )

    println("\nCreating neural network model...")
    # Create neural network model
    input_dim = length(state_to_vector(initial_state))
    output_dim = num_features + 1  # +1 for terminate action
    println("  input_dim: $input_dim")
    println("  output_dim: $output_dim")
    model = create_model(input_dim, output_dim)
    
    println("\nInitializing model parameters...")
    # Initialize model parameters
    ps, st = Lux.setup(rng, model)
    
    println("\nStarting training loop...")
    for iter in 1:n_iterations
        println("\nIteration $iter:")
        println("  Sampling trajectories...")
        
        # Sample trajectories
        trajectories = []
        rewards = Float32[]
        empty!(iter_value_discoveries)
        empty!(iter_measurement_efficiencies)
        
        for i in 1:batch_size
            traj = sample_trajectory(model, ps, st, initial_state)
            push!(trajectories, traj)
            
            # Get final state
            final_state = traj.states[end]
            
            # Calculate reward
            reward = calculate_reward(final_state, reward_fn)
            push!(rewards, reward)
            
            # Calculate value discovery (best value found vs true best value)
            true_best_value = maximum(global_experiment_values)
            found_best_value = maximum(final_state.feature_values[final_state.observed_features])
            value_discovery = found_best_value / true_best_value
            push!(iter_value_discoveries, value_discovery)
            
            # Calculate measurement efficiency
            new_measurements = sum([obs && !init for (obs, init) in zip(final_state.observed_features, final_state.initial_features)])
            measurement_efficiency = if new_measurements > 0
                value_discovery / (new_measurements * cost_per_measurement)
            else
                10.0  # High efficiency if no new measurements (found good value with initial features)
            end
            push!(iter_measurement_efficiencies, measurement_efficiency)
            
            println("    Trajectory $i:")
            println("      Reward: $reward")
            println("      Value Discovery: $value_discovery")
            println("      Measurement Efficiency: $measurement_efficiency")
        end
        
        println("  Computing loss and gradients...")
        # Compute loss and gradients
        loss, grads = compute_loss_and_gradient(model, ps, st, trajectories, rewards, output_dim)
        
        # Apply gradients
        ps = apply_gradients(ps, grads)
        
        # Record metrics
        push!(losses, loss)
        push!(mean_rewards, mean(rewards))
        push!(max_rewards, maximum(rewards))
        
        # Calculate mean value discovery and measurement efficiency for this iteration
        mean_value_discovery = mean(iter_value_discoveries)
        mean_measurement_efficiency = mean(iter_measurement_efficiencies)
        
        # Add to the metrics arrays
        push!(value_discoveries, mean_value_discovery)
        push!(measurement_efficiencies, mean_measurement_efficiency)
        
        println("  Loss: $loss")
        println("  Mean reward: $(mean(rewards))")
        println("  Max reward: $(maximum(rewards))")
        println("  Mean value discovery: $mean_value_discovery")
        println("  Mean measurement efficiency: $mean_measurement_efficiency")
        
        # Update learning rate schedule
        increment_global_step!()
    end
    
    # Create results dictionary with extended metrics
    results = Dict(
        "model" => model,
        "parameters" => ps,
        "states" => st,
        "metrics" => Dict(
            "losses" => losses,
            "mean_rewards" => mean_rewards,
            "max_rewards" => max_rewards,
            "value_discoveries" => value_discoveries,
            "measurement_efficiencies" => measurement_efficiencies
        )
    )
    
    return results
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    model, analysis = run_feature_acquisition_v3(
        num_features=10,
        num_experiments=10,
        max_steps=5,
        cost_per_measurement=0.1,
        n_iterations=1000,
        batch_size=32,
        observation_ratio=0.2  # 20% of features initially observed
    )
end

"""
    setup_model(features, weights, values, reward_fn)

Create and initialize the GFlowNet model for feature acquisition.

# Arguments
- `features`: Matrix of features
- `weights`: Vector of feature weights
- `values`: Vector of experiment values
- `reward_fn`: Function to compute rewards

# Returns
- `GFlowNetModel`: The initialized model
"""
function setup_model(features, weights, values, reward_fn)
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
    
    # Create DAG first
    dag = GFlowNet.create_dag(
        initial_state,
        terminal_states,
        terminal_sink,
        actions
    )
    
    # Create neural network for the policy
    input_dim = length(GFlowNet.state_to_features(initial_state))
    nn_model = Lux.Chain(
        Lux.Dense(input_dim => 128, Lux.relu),
        Lux.Dense(128 => 128, Lux.relu),
        Lux.Dense(128 => length(actions))
    )
    
    # Initialize parameters
    ps, st = Lux.setup(rng, nn_model)
    
    # Create the GFlowNet model
    println("Creating GFlowNet model")
    model = GFlowNet.GFlowNetModel(
        dag,                    # DirectedAcyclicGraph
        GFlowNet.ForwardPolicy(nn_model),  # ForwardPolicy
        nothing,               # BackwardPolicy
        nothing,               # FlowEstimator
        nothing,               # partition_function
        [GFlowNet.TrajectoryBalanceObjective(1.0)],  # objectives
        Optimisers.Adam(LEARNING_RATE),  # optimizer
        (forward = ps, backward = nothing, flow = nothing),  # parameters
        (forward = st, backward = nothing, flow = nothing)   # states
    )
    
    return model
end

"""
    create_neural_network(; input_dim::Int, output_dim::Int)

Creates a neural network model for the policy with the specified input and output dimensions.
"""
function create_neural_network(; input_dim::Int, output_dim::Int)
    println("Creating neural network model with input_dim=$input_dim, output_dim=$output_dim")
    
    # Create a feedforward neural network with two hidden layers using Lux
    Lux.Chain(
        Lux.Dense(input_dim => 128, Lux.relu),
        Lux.Dense(128 => 128, Lux.relu),
        Lux.Dense(128 => output_dim)
    )
end

"""
    is_applicable(action::MeasureFeatureAction, state::PartialFeatureState)

Determines whether a MeasureFeatureAction can be applied to a PartialFeatureState.
An action is applicable if:
1. The feature hasn't been measured yet
2. The state is not terminal
3. There are remaining measurements available
"""
function GFlowNet.is_applicable(action::MeasureFeatureAction, state::PartialFeatureState)
    # Check if the state is terminal
    if state.is_terminal
        return false
    end
    
    # Check if we have remaining measurements
    if state.measurements_remaining <= 0
        return false
    end
    
    # Check if the feature hasn't been measured yet
    !state.observed_features[action.feature_idx]
end

"""
    is_applicable(action::TerminateAction, state::PartialFeatureState)

Determines whether a TerminateAction can be applied to a PartialFeatureState.
A terminate action is applicable if:
1. The state is not already terminal
2. Either all measurements have been used or we have enough information
"""
function GFlowNet.is_applicable(action::TerminateAction, state::PartialFeatureState)
    # Cannot terminate an already terminal state
    if state.is_terminal
        return false
    end
    
    # Can terminate if we have no measurements left
    if state.measurements_remaining <= 0
        return true
    end
    
    # Can terminate if we have made at least one new measurement
    sum(state.observed_features .& .!state.initial_features) > 0
end

"""
    train_model(model::GFlowNetModel, n_iterations::Int)

Train the GFlowNet model for the specified number of iterations.
"""
function train_model(model::GFlowNetModel, n_iterations::Int)
    opt_state = Optimisers.setup(Optimisers.Adam(0.001), model.forward_policy)
    metrics = Dict("loss" => Float64[])
    
    for iter in 1:n_iterations
        # Sample trajectories
        trajectories = [sample_trajectory(model, create_initial_state()) for _ in 1:32]
        
        # Skip empty trajectories
        trajectories = filter(!isempty, trajectories)
        if isempty(trajectories)
            println("Warning: No valid trajectories sampled in iteration $iter")
            continue
        end
        
        # Compute loss and gradients
        try
            loss, grads = GFlowNet.compute_loss_and_grad(model, trajectories)
            
            # Check for null gradients
            if any(isnothing.(values(grads)))
                println("Warning: Null gradient detected in iteration $iter")
                continue
            end
            
            # Update model parameters
            Optimisers.update!(opt_state, model.forward_policy, grads)
            
            # Log metrics
            push!(metrics["loss"], loss)
            
            # Print progress
            if iter % 100 == 0
                println("Iteration $iter: Loss = $(loss)")
                
                # Print example trajectory
                if !isempty(trajectories)
                    traj = first(trajectories)
                    println("Example trajectory:")
                    for (s, next_s) in traj
                        reward = GFlowNet.reward(next_s)
                        n_measurements = sum(next_s.observed_features .& .!next_s.initial_features)
                        println("  State transition: $(n_measurements) new measurements, reward = $(reward)")
                    end
                end
            end
            
        catch e
            println("Error in iteration $iter: $e")
            continue
        end
    end
    
    return metrics
end

"""
    GFlowNet.reward(state::PartialFeatureState)

Compute the reward for a terminal state in the GFlowNet framework.
The reward is based on:
1. The improvement in value from initial observations
2. The cost of additional measurements made
"""
function GFlowNet.reward(state::PartialFeatureState)
    if !state.is_terminal
        return 0.01  # Small positive reward for non-terminal states
    end
    
    # Calculate value gained from new measurements only
    initial_value = calculate_best_value(state.initial_features, global_experiment_values)
    final_value = calculate_best_value(state.observed_features, global_experiment_values)
    value_gained = final_value - initial_value
    
    # Count only new measurements for cost
    new_measurements = sum(state.observed_features .& .!state.initial_features)
    cost = new_measurements * global_cost_per_measurement
    
    # Combine value and cost
    reward = value_gained - cost
    
    # Ensure positive reward
    return max(0.01, reward)
end

# Add iteration interface for Trajectory type
import Base: iterate, length, isempty, lastindex, last

function Base.iterate(traj::GFlowNet.Trajectory)
    if length(traj.states) < 2
        return nothing
    end
    return (traj.states[1], traj.states[2]), 2
end

function Base.iterate(traj::GFlowNet.Trajectory, state::Int)
    if state >= length(traj.states)
        return nothing
    end
    return (traj.states[state-1], traj.states[state]), state + 1
end

function Base.length(traj::GFlowNet.Trajectory)
    return length(traj.states)
end

function Base.isempty(traj::GFlowNet.Trajectory)
    return isempty(traj.states)
end

function Base.lastindex(traj::GFlowNet.Trajectory)
    return length(traj.states)
end

function Base.last(traj::GFlowNet.Trajectory)
    return traj.states[end]
end

# Add indexing interface for PartialFeatureState
import Base: getindex

function Base.getindex(state::PartialFeatureState, i::Int)
    # For vector-based state, just return the i-th element
    return state.observed_features[i]
end

"""
    create_bootstrap_terminal_states(values)

Create terminal states with high-value features for bootstrapping.
"""
function create_bootstrap_terminal_states(values)
    # Sort values and select top 5 features
    sorted_indices = sortperm(values, rev=true)
    top_indices = sorted_indices[1:min(5, length(values))]
    num_features = length(values)
    
    terminal_states = PartialFeatureState[]
    
    # For each top feature, create a terminal state with it observed
    for idx in top_indices
        observed = falses(num_features)
        feature_values = zeros(Float64, num_features)
        
        # Observe the selected feature
        observed[idx] = true
        feature_values[idx] = rand()  # Fallback to random value
        
        state = PartialFeatureState(
            observed,
            feature_values,
            falses(num_features),
            0,
            true
        )
        push!(terminal_states, state)
    end
    
    return terminal_states
end

"""
    create_terminal_sink_state(values)

Create a terminal sink state with all features observed.
"""
function create_terminal_sink_state(values)
    num_features = length(values)
    
    # Create state with all features observed
    observed = trues(num_features)
    feature_values = rand(num_features)
    
    return PartialFeatureState(
        observed,
        feature_values,
        falses(num_features),
        0,
        true
    )
end

"""
    generate_experiment_values(num_features::Int, num_experiments::Int=5)

Generate synthetic experiment values normalized to [0,1] using a structured approach.
This matches the v2 approach for better consistency between versions.
"""
function generate_experiment_values(num_features::Int, num_experiments::Int=5)
    println("Generating synthetic data with more structure for visualization...")
    
    # Set random seed for reproducibility
    local_rng = Random.MersenneTwister(42)
    
    # Generate random features
    features = randn(local_rng, num_features, num_experiments)
    
    # Generate random weights (importance of each feature)
    weights = abs.(randn(local_rng, num_features))
    weights ./= norm(weights)  # normalize
    
    # Calculate values based on features
    raw_values = features' * weights
    
    # Add some noise
    noisy_values = raw_values .+ 0.2 * randn(local_rng, num_experiments)
    
    # Normalize to [0, 1] range
    values = (noisy_values .- minimum(noisy_values)) ./ 
             (maximum(noisy_values) - minimum(noisy_values))
    
    # Print top experiments by value
    sorted_indices = sortperm(values, rev=true)
    println("Top 5 experiments by value:")
    for i in 1:min(5, num_experiments)
        idx = sorted_indices[i]
        println("  Experiment $idx: $(values[idx])")
    end
    
    # Store feature data for visualization
    global global_experiment_features = features
    global global_true_weights = weights
    
    return values
end

"""
    create_model(input_dim::Int, output_dim::Int)

Create a neural network model for the GFlowNet policy.
The input dimension is 3 * num_features + 2 because we have:
- num_features for observed features (Bool)
- num_features for feature values (Float64)
- num_features for initial features (Bool)
- 2 for metadata (measurements_remaining and is_terminal)
"""
function create_model(input_dim::Int, output_dim::Int)
    return Chain(
        Dense(input_dim, 64, relu),
        Dense(64, 32, relu),
        Dense(32, output_dim)
    )
end

"""
    sample_trajectory(model, ps, st, initial_state)

Sample a trajectory through the state space using the current policy model.
Returns a list of (state, action, next_state) tuples.
Updated to handle continuous feature values.
"""
function sample_trajectory(model, ps, st, initial_state)
    # Extract raw parameters if they are wrapped in optimizer state
    raw_ps = map(layer -> map(p -> p isa Optimisers.Leaf ? p.val : p, layer), ps)
    
    current_state = deepcopy(initial_state)
    states = [current_state]
    actions = Int[]
    
    while !current_state.is_terminal
        # Convert state to input vector
        state_vec = state_to_vector(current_state)
        state_batch = reshape(state_vec, :, 1)
        
        # Forward pass through the model
        logits = model(state_batch, raw_ps, st)[1]
        
        # Sample action from logits
        probs = softmax(vec(logits))
        action_idx = sample(1:length(probs), Weights(probs))
        
        # Take action
        next_state = apply_action(current_state, action_idx)
        
        # Update trajectory
        push!(states, next_state)
        push!(actions, action_idx)
        
        current_state = next_state
    end
    
    return (states=states, actions=actions)
end

"""
    compute_reward(trajectory, values, cost_per_measurement)

Compute the reward for a trajectory based on experiment values and measurement costs.
"""
function compute_reward(trajectory, values, cost_per_measurement)
    final_state = trajectory[end][3]  # Get the final state from the last transition
    
    # Calculate value gained from new measurements only
    initial_value = calculate_best_value(final_state.initial_features, values)
    final_value = calculate_best_value(final_state.observed_features, values)
    value_gained = final_value - initial_value
    
    # Count only new measurements for cost
    new_measurements = sum(final_state.observed_features .& .!final_state.initial_features)
    cost_penalty = cost_per_measurement * new_measurements
    
    return value_gained - cost_penalty
end