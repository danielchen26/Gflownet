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
    flags_for(objective)

The `build_model` / `TrainingConfig` knobs that supply everything `objective`
requires, DERIVED from `get_objective_requirements` rather than hand-listed.

Every site in this file that builds a model for a given objective goes through here,
so no site can drift from the library's own declaration of what that objective needs.
"""
function flags_for(objective::TrainingObjective)
    reqs = get_objective_requirements(objective)
    return (;
        include_backward = "backward_policy" in reqs,
        include_flow_estimator = "flow_estimator" in reqs,
        partition_function_method = "learnable_log_Z" in reqs ?
            LEARNABLE_ESTIMATION : SIMPLE_ESTIMATION
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

    # Flags are DERIVED from get_objective_requirements, never hand-listed.
    #
    # This table used to be maintained by hand, with a comment explaining that it
    # deliberately disagreed with get_objective_requirements because that function
    # under-reported SUB_TRAJECTORY_BALANCE. The function is correct now -- it and
    # validate_training_config read one declarative table -- so duplicating it here
    # would only be a second copy free to drift again. Deriving instead means this
    # demo also proves the reported requirements are sufficient to actually train.
    objectives = [
        (TRAJECTORY_BALANCE, "Trajectory Balance"),
        (DETAILED_BALANCE, "Detailed Balance"),
        (FLOW_MATCHING, "Flow Matching"),
        (SUB_TRAJECTORY_BALANCE, "Sub-Trajectory Balance"),
        (TRAJECTORY_LIKELIHOOD_MAXIMIZATION, "Trajectory Likelihood Maximization (TLM)")
    ]

    results = Dict{String,TrainingHistory}()

    for (objective, name) in objectives
        flags = flags_for(objective)
        println("\n📈 $name")
        println("   Declared requirements: " *
                "$(join(get_objective_requirements(objective), ", "))")
        println("   Model built with: " *
                join(["$k=$v" for (k, v) in pairs(flags)], ", "))

        model = build_model(; flags...)
        config = TrainingConfig(
            objective = objective,
            # Must MATCH the model: train_step! reads the method from the config
            # when deciding whether to update log_Z, so a config saying SIMPLE
            # against a model that has a learnable Z silently freezes it.
            partition_function_method = flags.partition_function_method,
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

    # Only the two methods that EXIST. SAMPLING_ESTIMATION and ADAPTIVE_ESTIMATION are
    # exported enum values documented "Not implemented"; nothing allocates or updates a
    # log_Z for them, so they silently pin Z = 1. This demo used to train all four and
    # report four sets of numbers, three of which were the same fixed-Z run under
    # different labels. They are demonstrated below as REFUSED instead.
    methods = [
        (SIMPLE_ESTIMATION, "Simple Estimation (Z fixed at 1)"),
        (LEARNABLE_ESTIMATION, "Learnable Z (gradient descent on log Z)")
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

    # The two unimplemented methods, shown being refused. This is the useful thing to
    # demonstrate about them: previously a caller got numbers back and no indication
    # that Z had been pinned at 1.
    println("\n🚫 Methods that are refused rather than silently pinning Z = 1")
    for method in (SAMPLING_ESTIMATION, ADAPTIVE_ESTIMATION)
        try
            train_gflownet(build_model(; partition_function_method = method),
                           TrainingConfig(objective = TRAJECTORY_BALANCE,
                                          partition_function_method = method,
                                          n_iterations = 1, batch_size = 2);
                           verbose = false)
            error("$method was accepted -- the guard in validate_training_config is gone")
        catch e
            e isa ArgumentError || rethrow()
            println("   $method -> refused: $(first(split(e.msg, " -- ")))")
        end
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
    # flags_for, not a hand-picked flag: SubTB needs a backward policy and a learnable
    # Z as well as the flow estimator. With only the estimator every iteration threw
    # and train_gflownet recorded NaN, so this block reported a finished run that had
    # trained on nothing.
    subtb_flags = flags_for(SUB_TRAJECTORY_BALANCE)
    model_subtb = build_model(; subtb_flags...)
    config_subtb = TrainingConfig(
        objective = SUB_TRAJECTORY_BALANCE,
        partition_function_method = subtb_flags.partition_function_method,
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

    # (label, objective, Z method). Components come from flags_for, so only the Z
    # method varies per row -- that is the axis this comparison is about.
    #
    # Two rows used to read ADAPTIVE_ESTIMATION and SAMPLING_ESTIMATION. Neither is
    # implemented, so both were really running fixed Z = 1 under a label claiming
    # otherwise, and the "Sub-TB + adaptive Z" row was the degenerate configuration
    # that collapses onto one terminal state. Only the two methods that exist appear
    # now, and SUB_TRAJECTORY_BALANCE takes LEARNABLE because it has no valid
    # fixed-Z form.
    scenarios = [
        ("TB + fixed Z", TRAJECTORY_BALANCE, SIMPLE_ESTIMATION),
        ("TB + learnable Z", TRAJECTORY_BALANCE, LEARNABLE_ESTIMATION),
        ("Sub-TB + learnable Z", SUB_TRAJECTORY_BALANCE, LEARNABLE_ESTIMATION),
        ("DB + fixed Z", DETAILED_BALANCE, SIMPLE_ESTIMATION)
    ]

    results = Dict{String,TrainingHistory}()

    for (name, objective, z_method) in scenarios
        println("\n🔬 $name")

        # Components from flags_for; the row's z_method overrides only the Z axis.
        flags = merge(flags_for(objective), (; partition_function_method = z_method))
        model = build_model(; flags...)
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
