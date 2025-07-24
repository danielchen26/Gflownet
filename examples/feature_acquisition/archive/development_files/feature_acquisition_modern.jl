#!/usr/bin/env julia

"""
Feature Acquisition with GFlowNet - Modern Implementation

This example demonstrates strategic feature acquisition using the modern GFlowNet.jl
training interface with ADAPTIVE_SUB_TB + ADAPTIVE_ESTIMATION configuration.

Following the training system rules with TrainingConfig and train_gflownet().
"""

using GFlowNet
using Random
using Statistics
using Plots
using CSV
using DataFrames
using LinearAlgebra
using Lux
using Optimisers
using NNlib
using StatsBase
using Zygote
using Distributions

# Set random seed for reproducibility
Random.seed!(42)

# Environment constants
const FEATURE_DIM = 8
const MAX_MEASUREMENTS = 5
const rng = Random.default_rng()

"""
    FeatureAcquisitionState <: AbstractState

State representing partial feature measurements in an experiment.
Tracks which features have been measured and their observed values.
"""
mutable struct FeatureAcquisitionState <: GFlowNet.AbstractState
    observed_features::Vector{Bool}      # Which features have been measured
    feature_values::Vector{Float64}      # True values (unknown until measured)
    measured_values::Vector{Float64}     # Values we've actually observed
    step_count::Int                      # Number of measurements made
    max_steps::Int                       # Budget constraint
    experiment_id::Int                   # Which experiment this is

    function FeatureAcquisitionState(n_features::Int, max_steps::Int, exp_id::Int, true_values::Vector{Float64})
        observed = falses(n_features)
        measured = zeros(n_features)
        return new(observed, true_values, measured, 0, max_steps, exp_id)
    end
end

"""
    MeasureFeatureAction <: AbstractAction

Action to measure a specific feature (incurs cost, reveals information).
"""
struct MeasureFeatureAction <: GFlowNet.AbstractAction
    feature_idx::Int
end

"""
    TerminateAction <: AbstractAction

Action to terminate the experiment and use current measurements for decision.
"""
struct TerminateAction <: GFlowNet.AbstractAction end

"""
    FeatureAcquisitionEnvironment

Environment for strategic feature acquisition with cost-benefit optimization.
"""
struct FeatureAcquisitionEnvironment
    n_features::Int
    max_steps::Int
    cost_per_measurement::Float64
    experiment_values::Matrix{Float64}  # [n_features x n_experiments]
    n_experiments::Int

    function FeatureAcquisitionEnvironment(n_features::Int, max_steps::Int, cost::Float64, n_exp::Int)
        # Generate diverse experiment values with different optimal features
        values = generate_experiment_values(n_features, n_exp)
        new(n_features, max_steps, cost, values, n_exp)
    end
end

"""
    generate_experiment_values(n_features::Int, n_experiments::Int)

Generate realistic experiment values where different features are optimal for different experiments.
"""
function generate_experiment_values(n_features::Int, n_experiments::Int)
    values = zeros(n_features, n_experiments)

    for exp in 1:n_experiments
        # Create base values with some correlation structure
        base_values = 0.5 .+ 0.3 * randn(rng, n_features)

        # Make 1-2 features clearly superior for this experiment
        superior_features = sample(rng, 1:n_features, min(2, n_features), replace=false)
        for feat in superior_features
            base_values[feat] += 1.0 + 0.5 * rand(rng)
        end

        # Ensure positive values
        values[:, exp] = max.(base_values, 0.1)
    end

    return values
end

"""
    get_valid_actions(env::FeatureAcquisitionEnvironment, state::FeatureAcquisitionState)

Get all valid actions from current state.
"""
function get_valid_actions(env::FeatureAcquisitionEnvironment, state::FeatureAcquisitionState)
    actions = GFlowNet.AbstractAction[]

    # Can always terminate
    push!(actions, TerminateAction())

    # Can measure unobserved features if budget allows
    if state.step_count < state.max_steps
        for i in 1:env.n_features
            if !state.observed_features[i]
                push!(actions, MeasureFeatureAction(i))
            end
        end
    end

    return actions
end

"""
    apply_action(env::FeatureAcquisitionEnvironment, state::FeatureAcquisitionState, action::AbstractAction)

Apply action to state and return new state.
"""
function apply_action(env::FeatureAcquisitionEnvironment, state::FeatureAcquisitionState, action::AbstractAction)
    new_state = deepcopy(state)

    if action isa MeasureFeatureAction
        # Measure the feature (reveal its true value)
        feat_idx = action.feature_idx
        new_state.observed_features[feat_idx] = true
        new_state.measured_values[feat_idx] = state.feature_values[feat_idx]
        new_state.step_count += 1

    elseif action isa TerminateAction
        # No change to state, just signals termination
        nothing
    end

    return new_state
end

"""
    calculate_reward(env::FeatureAcquisitionEnvironment, state::FeatureAcquisitionState)

Calculate reward balancing information gain vs measurement costs.
"""
function calculate_reward(env::FeatureAcquisitionEnvironment, state::FeatureAcquisitionState)
    # Information gain: best value among measured features
    measured_indices = findall(state.observed_features)

    if isempty(measured_indices)
        # No measurements made - very low reward
        return 0.1
    end

    # Best value found among measured features
    best_measured_value = maximum(state.measured_values[measured_indices])

    # Cost: number of measurements made
    cost = state.step_count * env.cost_per_measurement

    # Reward = information gain - cost, with positive baseline
    reward = best_measured_value - cost + 0.5

    return max(reward, 0.1)  # Ensure positive rewards
end

"""
    is_terminal(env::FeatureAcquisitionEnvironment, state::FeatureAcquisitionState)

Check if state is terminal (experiment ended).
"""
function is_terminal(env::FeatureAcquisitionEnvironment, state::FeatureAcquisitionState)
    # Terminal if we've hit budget limit or if this state was reached by TerminateAction
    return state.step_count >= state.max_steps
end

"""
    state_to_features(env::FeatureAcquisitionEnvironment, state::FeatureAcquisitionState)

Convert state to feature vector for neural networks.
"""
function state_to_features(env::FeatureAcquisitionEnvironment, state::FeatureAcquisitionState)
    features = Float32[]

    # Observed features (binary indicators)
    append!(features, Float32.(state.observed_features))

    # Measured values (0 if not measured)
    append!(features, Float32.(state.measured_values))

    # Budget information
    push!(features, Float32(state.step_count))
    push!(features, Float32(state.max_steps))
    push!(features, Float32(state.step_count / state.max_steps))  # Budget utilization

    # Experiment context
    push!(features, Float32(state.experiment_id))

    return features
end

"""
    create_initial_states(env::FeatureAcquisitionEnvironment)

Create initial states for all experiments.
"""
function create_initial_states(env::FeatureAcquisitionEnvironment)
    states = FeatureAcquisitionState[]

    for exp_id in 1:env.n_experiments
        true_values = env.experiment_values[:, exp_id]
        initial_state = FeatureAcquisitionState(env.n_features, env.max_steps, exp_id, true_values)
        push!(states, initial_state)
    end

    return states
end

"""
    setup_feature_acquisition_model(env::FeatureAcquisitionEnvironment)

Set up the GFlowNet model for feature acquisition.
"""
function setup_feature_acquisition_model(env::FeatureAcquisitionEnvironment)
    # Create initial states
    initial_states = create_initial_states(env)

    # Set up reward function
    reward_fn = state -> calculate_reward(env, state)

    # Create DAG (this will be dynamically expanded during training)
    dag = GFlowNet.create_dag(initial_states[1], FeatureAcquisitionState[])

    # Neural network dimensions
    sample_state = initial_states[1]
    input_dim = length(state_to_features(env, sample_state))
    output_dim = env.n_features + 1  # +1 for terminate action

    # Create neural network for policy
    policy_network = Chain(
        Dense(input_dim, 64, tanh),
        Dense(64, 32, tanh),
        Dense(32, output_dim)
    )

    # Initialize model
    ps, st = Lux.setup(rng, policy_network)

    # Create forward policy
    forward_policy = GFlowNet.ForwardPolicy(policy_network, ps, st)

    # Create GFlowNet model
    model = GFlowNet.GFlowNetModel(
        dag=dag,
        forward_policy=forward_policy,
        backward_policy=nothing,  # Not needed for this configuration
        flow_estimator=nothing,   # Not needed for this configuration
        reward_function=reward_fn
    )

    return model, initial_states
end

"""
    run_feature_acquisition_modern(; kwargs...)

Run feature acquisition using modern GFlowNet training interface.

# Arguments
- `n_features::Int=8`: Number of features per experiment
- `n_experiments::Int=50`: Number of different experiments
- `max_measurements::Int=5`: Budget limit for measurements per experiment
- `cost_per_measurement::Float64=0.1`: Cost for each measurement
- `n_iterations::Int=500`: Training iterations
- `batch_size::Int=16`: Batch size for training
"""
function run_feature_acquisition_modern(;
    n_features::Int=8,
    n_experiments::Int=50,
    max_measurements::Int=5,
    cost_per_measurement::Float64=0.1,
    n_iterations::Int=500,
    batch_size::Int=16,
    output_prefix::String="modern_results"
)

    println("🔬 Feature Acquisition with Modern GFlowNet Training")
    println("="^60)
    println("Configuration:")
    println("  Features: $n_features")
    println("  Experiments: $n_experiments")
    println("  Max measurements: $max_measurements")
    println("  Cost per measurement: $cost_per_measurement")
    println("  Training iterations: $n_iterations")
    println("  Batch size: $batch_size")
    println()

    # Create environment
    println("Setting up environment...")
    env = FeatureAcquisitionEnvironment(n_features, max_measurements, cost_per_measurement, n_experiments)

    # Set up model
    println("Setting up GFlowNet model...")
    model, initial_states = setup_feature_acquisition_model(env)

    # Create modern training configuration
    # Following rules: ADAPTIVE_SUB_TB + ADAPTIVE_ESTIMATION for feature acquisition
    println("Creating training configuration...")
    config = GFlowNet.TrainingConfig(
        objective=GFlowNet.ADAPTIVE_SUB_TB,
        partition_function_method=GFlowNet.ADAPTIVE_ESTIMATION,
        batch_size=batch_size,
        learning_rate=0.001,
        n_iterations=n_iterations,
        partition_update_frequency=10,
        validation_frequency=50,
        early_stopping_patience=100,
        sub_trajectory_config=Dict(
            :difficulty_threshold => 0.05,  # Focus on difficult sub-trajectories
            :min_length => 2,
            :n_subtrajectories => 3
        )
    )

    # Run modern training
    println("Starting training with ADAPTIVE_SUB_TB + ADAPTIVE_ESTIMATION...")
    println()

    history = GFlowNet.train_gflownet(model, config; verbose=true)

    # Analyze results
    println("\n📊 Training Results Analysis")
    println("="^40)

    final_loss = history[:losses][end]
    mean_final_losses = mean(history[:losses][max(1, end - 9):end])

    println("Final loss: $(round(final_loss, digits=4))")
    println("Mean last 10 losses: $(round(mean_final_losses, digits=4))")

    # Test the trained model
    println("\n🧪 Testing Trained Model")
    println("="^30)

    test_results = test_trained_model(env, model, 10)

    println("Average measurements used: $(round(test_results[:avg_measurements], digits=2))")
    println("Average value discovered: $(round(test_results[:avg_value_found], digits=4))")
    println("Average efficiency: $(round(test_results[:avg_efficiency], digits=4))")

    # Save results
    println("\nSaving results...")
    results_dict = Dict(
        "training_history" => history,
        "test_results" => test_results,
        "config" => config,
        "environment" => Dict(
            "n_features" => n_features,
            "n_experiments" => n_experiments,
            "max_measurements" => max_measurements,
            "cost_per_measurement" => cost_per_measurement
        )
    )

    # Save training metrics
    metrics_df = DataFrame(
        iteration=1:length(history[:losses]),
        loss=history[:losses]
    )

    if haskey(history, :partition_function_estimates) && !isempty(history[:partition_function_estimates])
        # Add partition function estimates (may be sparse)
        z_estimates = Float64[]
        for i in 1:length(history[:losses])
            if i <= length(history[:partition_function_estimates])
                push!(z_estimates, history[:partition_function_estimates][i])
            else
                push!(z_estimates, z_estimates[end])  # Use last known value
            end
        end
        metrics_df.partition_function = z_estimates
    end

    CSV.write("$(output_prefix)_training_metrics.csv", metrics_df)

    # Create visualization
    create_results_visualization(history, test_results, output_prefix)

    println("✅ Modern feature acquisition training completed!")
    println("Results saved with prefix: $output_prefix")

    return results_dict
end

"""
    test_trained_model(env::FeatureAcquisitionEnvironment, model::GFlowNetModel, n_tests::Int)

Test the trained model on new experiments.
"""
function test_trained_model(env::FeatureAcquisitionEnvironment, model::GFlowNetModel, n_tests::Int)
    measurements_used = Int[]
    values_found = Float64[]
    efficiencies = Float64[]

    for test_id in 1:n_tests
        # Create a new test experiment
        test_values = env.experiment_values[:, min(test_id, env.n_experiments)]
        test_state = FeatureAcquisitionState(env.n_features, env.max_steps, test_id, test_values)

        # Sample trajectory using trained model
        trajectory = GFlowNet.sample_trajectory(model, test_state)

        if !isempty(trajectory.states)
            final_state = trajectory.states[end]

            # Calculate metrics
            n_measurements = sum(final_state.observed_features)
            push!(measurements_used, n_measurements)

            if n_measurements > 0
                best_found = maximum(final_state.measured_values[final_state.observed_features])
                push!(values_found, best_found)

                # Efficiency = value found per unit cost
                efficiency = best_found / (n_measurements * env.cost_per_measurement)
                push!(efficiencies, efficiency)
            else
                push!(values_found, 0.0)
                push!(efficiencies, 0.0)
            end
        end
    end

    return Dict(
        :avg_measurements => mean(measurements_used),
        :avg_value_found => mean(values_found),
        :avg_efficiency => mean(efficiencies),
        :measurements_used => measurements_used,
        :values_found => values_found,
        :efficiencies => efficiencies
    )
end

"""
    create_results_visualization(history::Dict, test_results::Dict, output_prefix::String)

Create comprehensive visualization of training and testing results.
"""
function create_results_visualization(history::Dict, test_results::Dict, output_prefix::String)
    # Training progress plot
    p1 = plot(history[:losses],
        title="Training Loss Progress",
        xlabel="Iteration",
        ylabel="Loss",
        linewidth=2,
        color=:blue)

    # Partition function estimates (if available)
    p2 = if haskey(history, :partition_function_estimates) && !isempty(history[:partition_function_estimates])
        plot(history[:partition_function_estimates],
            title="Partition Function Estimates",
            xlabel="Update Step",
            ylabel="Z Estimate",
            linewidth=2,
            color=:red,
            marker=:circle,
            markersize=3)
    else
        plot(title="Partition Function",
            xlabel="No estimates available",
            ylabel="",
            grid=false,
            showaxis=false)
    end

    # Test results histograms
    p3 = histogram(test_results[:measurements_used],
        title="Measurements Used",
        xlabel="Number of Measurements",
        ylabel="Frequency",
        bins=5,
        color=:green,
        alpha=0.7)

    p4 = histogram(test_results[:efficiencies],
        title="Measurement Efficiency",
        xlabel="Value per Unit Cost",
        ylabel="Frequency",
        bins=8,
        color=:orange,
        alpha=0.7)

    # Combine plots
    combined_plot = plot(p1, p2, p3, p4,
        layout=(2, 2),
        size=(800, 600),
        margin=5Plots.mm)

    # Save plot
    savefig(combined_plot, "$(output_prefix)_results.png")

    println("Visualization saved as: $(output_prefix)_results.png")
end

"""
    main()

Main function to run the modern feature acquisition example.
"""
function main()
    println("🚀 Running Modern Feature Acquisition Example")
    println("Using ADAPTIVE_SUB_TB + ADAPTIVE_ESTIMATION (recommended configuration)")
    println()

    # Run with default parameters
    results = run_feature_acquisition_modern(
        n_features=8,
        n_experiments=30,
        max_measurements=4,
        cost_per_measurement=0.1,
        n_iterations=200,
        batch_size=16,
        output_prefix="feature_acquisition_modern"
    )

    println("\n🎉 Example completed successfully!")
    return results
end

# Run example if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end
