"""
Comprehensive GFlowNet Training Example

This example demonstrates every training objective and partition-function
estimation method that GFlowNet.jl actually implements, plus the advanced
training knobs (ε-uniform exploration, entropy regularization, experience
replay, TLM backward-policy learning) and the built-in plotting helpers.

It runs on the built-in acyclic grid world, which already implements the full
domain interface (`state_to_features`, `is_terminal_state`, `reward`,
`is_applicable`, `apply_action`), so no domain code is needed here. See
`examples/core_features/sub_trajectory_balance/domain_agnostic_stb_demo.jl`
for a worked example of implementing the interface for a custom domain.
"""

using GFlowNet
using Plots
using Random, Statistics

# Set random seed for reproducibility
Random.seed!(42)

# Demo scale. This script performs 16 independent training runs, so the
# schedules are deliberately tiny in order to finish quickly; real runs use
# n_iterations in the hundreds or thousands.
const DEMO_ITERATIONS = 15
const DEMO_BATCH = 8
const HIDDEN_DIM = 32

# The grid world stores its size and reward layout in a package-level
# configuration cell, so every model built in one script run must share the same
# grid size -- otherwise a model trained earlier is scored against a different
# reward function.
const GRID_SIZE = 4
const REWARD_POSITIONS = Dict{Tuple{Int,Int},Float64}(
    (GRID_SIZE, GRID_SIZE) => 10.0,   # main mode
    (1, GRID_SIZE) => 5.0,            # secondary mode
    (GRID_SIZE, 1) => 5.0             # secondary mode
)

"""
    build_model(; include_backward=false, include_flow_estimator=false,
                  partition_function_method=SIMPLE_ESTIMATION)

Build a grid world GFlowNet with exactly the components a given objective needs.

`create_grid_world_gflownet` forwards to `create_gflownet`, which only allocates
a backward policy or a flow estimator when asked. Objectives that read those
components (`DETAILED_BALANCE`, `FLOW_MATCHING`, `SUB_TRAJECTORY_BALANCE`,
`TRAJECTORY_LIKELIHOOD_MAXIMIZATION`) therefore have to be paired with the right
flags, and `LEARNABLE_ESTIMATION` has to be requested at construction time so
that `log_Z` exists in the parameter vector.
"""
function build_model(;
    include_backward::Bool = false,
    include_flow_estimator::Bool = false,
    partition_function_method::PartitionFunctionMethod = SIMPLE_ESTIMATION
)
    return create_grid_world_gflownet(
        grid_size = GRID_SIZE,
        reward_positions = REWARD_POSITIONS,
        hidden_dim = HIDDEN_DIM,
        learning_rate = 0.01,
        include_backward = include_backward,
        include_flow_estimator = include_flow_estimator,
        partition_function_method = partition_function_method
    )
end

"""
    summarize_history(history)

Reduce a `TrainingHistory` to `(final_loss, successful, total)`.

`train_gflownet` records `NaN` for any iteration whose loss or gradient step
threw, so a run can "complete" having learned nothing at all. Always report the
number of successful iterations alongside the loss.
"""
function summarize_history(history::TrainingHistory)
    finite = filter(isfinite, history.losses)
    final_loss = isempty(finite) ? NaN : finite[end]
    return (final_loss, length(finite), length(history.losses))
end

"""
    report(name, history)

Print a one-line summary of a training run and return the final loss.
"""
function report(name::String, history::TrainingHistory)
    final_loss, successful, total = summarize_history(history)
    if successful == 0
        println("  ⚠️  $name: 0/$total iterations produced a usable gradient")
    else
        println("  ✅ $name: final loss $(round(final_loss, digits=4)) " *
                "($successful/$total successful iterations)")
    end
    return final_loss
end

"""
    mean_sampled_reward(model; n=200)

Mean terminal reward of `n` trajectories sampled from the trained policy.
This is the quantity a GFlowNet is supposed to improve: it should sample
terminal states in proportion to their reward.
"""
function mean_sampled_reward(model::GFlowNetModel; n::Int = 100)
    trajectories = [sample_trajectory(model) for _ in 1:n]
    return mean(reward(traj.states[end]) for traj in trajectories)
end

"""
    demonstrate_training_objectives()

Train with every objective that has a working loss implementation.
"""
function demonstrate_training_objectives()
    println("🎯 Demonstrating All Implemented Training Objectives")
    println("=" ^ 60)

    # (objective, label, model flags). `get_objective_requirements` reports the
    # declared requirements, but SUB_TRAJECTORY_BALANCE's loss additionally
    # demands a flow estimator, so the flags below are what the losses actually
    # read, not a restatement of that function's output.
    objectives = [
        (TRAJECTORY_BALANCE, "Trajectory Balance", (;)),
        (DETAILED_BALANCE, "Detailed Balance", (; include_backward = true)),
        (FLOW_MATCHING, "Flow Matching", (; include_flow_estimator = true)),
        (SUB_TRAJECTORY_BALANCE, "Sub-Trajectory Balance", (; include_flow_estimator = true)),
        (TRAJECTORY_LIKELIHOOD_MAXIMIZATION, "Trajectory Likelihood Maximization (TLM)",
         (; include_backward = true))
    ]

    results = Dict{String,TrainingHistory}()

    for (objective, name, flags) in objectives
        println("\n📈 $name")
        println("   Declared requirements: " *
                "$(join(get_objective_requirements(objective), ", "))")
        println("   Model built with: " *
                (isempty(flags) ? "forward policy only" :
                 join(["$k=$v" for (k, v) in pairs(flags)], ", ")))

        model = build_model(; flags...)
        config = TrainingConfig(
            objective = objective,
            partition_function_method = SIMPLE_ESTIMATION,
            n_iterations = DEMO_ITERATIONS,
            batch_size = DEMO_BATCH,
            learning_rate = 0.01,
            validation_frequency = 10
        )

        history = train_gflownet(model, config; verbose = false)
        report(name, history)
        println("     mean sampled reward: $(round(mean_sampled_reward(model), digits=3))")
        results[name] = history
    end

    # Objectives that exist as enum values but cannot be trained as-is.
    println("\nℹ️  Objectives declared but not trainable through train_gflownet:")
    println("   • $DIRECT_FLOW_OBJECTIVE — disabled in src: its loss was constant")
    println("     w.r.t. the parameters, so training under it was a silent no-op.")
    println("   • $COMBINED_OBJECTIVES — no loss branch implemented.")
    println("   • $MULTI_OBJECTIVE_TB — needs a preference encoder and Z(w) network;")
    println("     build it with create_mogfn_gflownet, not create_grid_world_gflownet.")

    return results
end

"""
    demonstrate_partition_function_methods()

Train with each partition-function estimation method under trajectory balance.
"""
function demonstrate_partition_function_methods()
    println("\n🔢 Demonstrating Partition Function Methods")
    println("=" ^ 60)

    methods = [
        (SIMPLE_ESTIMATION, "Simple Estimation (Z fixed at 1)"),
        (LEARNABLE_ESTIMATION, "Learnable Z (gradient descent on log Z)"),
        (SAMPLING_ESTIMATION, "Sampling-Based Estimation"),
        (ADAPTIVE_ESTIMATION, "Adaptive Method Switching")
    ]

    results = Dict{String,TrainingHistory}()

    for (method, name) in methods
        println("\n🧮 $name")

        # The method has to be given to BOTH the constructor (so that log_Z is
        # allocated in the parameter vector) and the training config.
        model = build_model(; partition_function_method = method)
        config = TrainingConfig(
            objective = TRAJECTORY_BALANCE,
            partition_function_method = method,
            n_iterations = DEMO_ITERATIONS,
            batch_size = DEMO_BATCH,
            learning_rate = 0.01,
            validation_frequency = 10
        )

        history = train_gflownet(model, config; verbose = false)
        report(name, history)

        if method == LEARNABLE_ESTIMATION
            learned_Z = exp(model.parameters.log_Z)
            println("     learned Z = $(round(learned_Z, digits=4)) " *
                    "(sum of rewards = $(round(sum(values(REWARD_POSITIONS)), digits=1)))")
        end

        results[name] = history
    end

    return results
end

"""
    demonstrate_advanced_configurations()

Show the advanced training controls: exploration, entropy regularization,
experience replay and TLM backward-policy learning.
"""
function demonstrate_advanced_configurations()
    println("\n⚙️  Demonstrating Advanced Configurations")
    println("=" ^ 60)

    results = Dict{String,TrainingHistory}()

    # 1. Exploration: ε-uniform sampling annealed to zero, plus a temperature
    #    above 1 to flatten the forward policy early in training.
    println("\n🔍 ε-uniform exploration + temperature + entropy regularization")
    model_explore = build_model(; partition_function_method = LEARNABLE_ESTIMATION)
    config_explore = TrainingConfig(
        objective = TRAJECTORY_BALANCE,
        partition_function_method = LEARNABLE_ESTIMATION,
        n_iterations = DEMO_ITERATIONS,
        batch_size = DEMO_BATCH,
        learning_rate = 0.01,
        temperature = 1.5,
        epsilon = 0.2,
        epsilon_decay = true,
        entropy_weight = 0.05,
        gradient_clip_norm = 0.5,
        z_learning_rate_multiplier = 10.0,
        validation_frequency = 10
    )
    results["Exploration"] = train_gflownet(model_explore, config_explore; verbose = false)
    report("Exploration", results["Exploration"])

    # 2. Experience replay: reuse high-reward trajectories off-policy.
    println("\n🧺 Experience replay (off-policy, reward-prioritized)")
    model_replay = build_model()
    config_replay = TrainingConfig(
        objective = TRAJECTORY_BALANCE,
        n_iterations = DEMO_ITERATIONS,
        batch_size = DEMO_BATCH,
        learning_rate = 0.01,
        use_replay_buffer = true,
        replay_buffer_size = 500,
        replay_ratio = 0.5,
        replay_priority_alpha = 0.6,
        validation_frequency = 10
    )
    results["Replay buffer"] = train_gflownet(model_replay, config_replay; verbose = false)
    report("Replay buffer", results["Replay buffer"])

    # 3. Sub-trajectory balance with a shorter sub-path window: more, smaller
    #    balance constraints per trajectory.
    println("\n🪜 Sub-trajectory balance with a short sub-path window")
    model_subtb = build_model(; include_flow_estimator = true)
    config_subtb = TrainingConfig(
        objective = SUB_TRAJECTORY_BALANCE,
        n_iterations = DEMO_ITERATIONS,
        batch_size = DEMO_BATCH,
        learning_rate = 0.01,
        sub_trajectory_length = 3,
        validation_frequency = 10
    )
    results["Short sub-trajectories"] = train_gflownet(model_subtb, config_subtb; verbose = false)
    report("Short sub-trajectories", results["Short sub-trajectories"])

    return results
end

"""
    compare_methods_performance()

Compare objective/Z-method pairs side by side on identical schedules.
"""
function compare_methods_performance()
    println("\n📊 Performance Comparison")
    println("=" ^ 60)

    scenarios = [
        ("TB + fixed Z", TRAJECTORY_BALANCE, SIMPLE_ESTIMATION, (;)),
        ("TB + learnable Z", TRAJECTORY_BALANCE, LEARNABLE_ESTIMATION, (;)),
        ("Sub-TB + adaptive Z", SUB_TRAJECTORY_BALANCE, ADAPTIVE_ESTIMATION,
         (; include_flow_estimator = true)),
        ("DB + sampling Z", DETAILED_BALANCE, SAMPLING_ESTIMATION,
         (; include_backward = true))
    ]

    results = Dict{String,TrainingHistory}()

    for (name, objective, z_method, flags) in scenarios
        println("\n🔬 $name")

        model = build_model(; partition_function_method = z_method, flags...)
        config = TrainingConfig(
            objective = objective,
            partition_function_method = z_method,
            n_iterations = DEMO_ITERATIONS,
            batch_size = DEMO_BATCH,
            learning_rate = 0.012,
            validation_frequency = 10
        )

        history = train_gflownet(model, config; verbose = false)
        report(name, history)

        finite = filter(isfinite, history.losses)
        if !isempty(finite)
            tail = finite[max(1, end - 4):end]
            println("     mean of last $(length(tail)) losses: $(round(mean(tail), digits=4))")
        end

        trajectories = [sample_trajectory(model) for _ in 1:60]
        analyze_grid_world_results(trajectories, GRID_SIZE)

        results[name] = history
    end

    return results
end

"""
    create_performance_plots(results, output_path)

Plot the loss curves of every scenario next to a bar chart of final losses,
and save the figure. Also demonstrates the built-in `plot_training_progress`.
"""
function create_performance_plots(results, output_path::String)
    println("\n📈 Creating Performance Plots")
    println("-" ^ 30)

    names = sort(collect(keys(results)))

    # Non-finite entries mark failed iterations; NaN leaves a gap in the curve
    # instead of collapsing the y-axis the way Inf would.
    plottable(losses) = [isfinite(v) ? v : NaN for v in losses]

    p1 = plot(title = "Training Loss", xlabel = "Iteration", ylabel = "Loss",
              legend = :topright)
    for name in names
        plot!(p1, plottable(results[name].losses), label = name, linewidth = 2)
    end

    final_losses = [summarize_history(results[name])[1] for name in names]
    p2 = bar(names, [isfinite(v) ? v : 0.0 for v in final_losses],
             title = "Final Loss", ylabel = "Loss", legend = false,
             xrotation = 30)

    combined = plot(p1, p2, layout = (2, 1), size = (900, 700))
    savefig(combined, output_path)
    println("  ✅ Saved comparison figure to $output_path")

    # The package also ships a ready-made progress plot for a single history.
    progress_path = replace(output_path, ".png" => "_progress.png")
    savefig(plot_training_progress(results[names[1]]), progress_path)
    println("  ✅ Saved $(names[1]) progress plot to $progress_path")

    return combined
end

"""
    main()

Run all demonstrations.
"""
function main()
    println("🚀 GFlowNet.jl Comprehensive Training Demonstration")
    println("=" ^ 60)
    println("Domain: $(GRID_SIZE)x$(GRID_SIZE) acyclic grid world " *
            "(moves: right, up, terminate)")
    println("Reward modes: $(REWARD_POSITIONS)")
    println("Schedule: $DEMO_ITERATIONS iterations x batch $DEMO_BATCH per run")
    println()

    objective_results = demonstrate_training_objectives()
    z_method_results = demonstrate_partition_function_methods()
    advanced_results = demonstrate_advanced_configurations()
    comparison_results = compare_methods_performance()

    output_path = joinpath(@__DIR__, "gflownet_training_comparison.png")
    plots = try
        create_performance_plots(comparison_results, output_path)
    catch e
        println("  ⚠️  Could not create plots: $e")
        nothing
    end

    println("\n🎉 Demonstration Complete!")
    println("=" ^ 30)
    println("Exercised:")
    println("  • $(length(objective_results)) training objectives")
    println("  • $(length(z_method_results)) partition function methods")
    println("  • $(length(advanced_results)) advanced configurations")
    println("  • $(length(comparison_results)) head-to-head comparisons")

    return Dict(
        "objectives" => objective_results,
        "z_methods" => z_method_results,
        "advanced" => advanced_results,
        "comparison" => comparison_results,
        "plots" => plots
    )
end

# Run the demonstration if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end
