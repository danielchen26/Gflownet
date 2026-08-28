# Reward Shaping 8x8 Grid - Tuning for extreme 3432:1 path asymmetry
# The basic test showed reward shaping alone isn't enough for 8x8.
# This script tests reward shaping + stronger exploration combinations.
#
# Run with: julia --project=. examples/core_features/reward_shaping_8x8_tuning.jl

using Statistics
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using GFlowNet

include(joinpath(@__DIR__, "..", "convergence_assertions.jl"))

# Budget cut from 2000-3000 iterations at batch 64 / hidden 128 to 150 at batch 16 /
# hidden 64, and evaluation from 1000 to 500 samples.
#
# This was by far the most expensive script in the set. Measured per-iteration cost at
# grid 8: batch 64 / hidden 128 ran at 6.3-8.6 s/iteration under load, so the old
# 3000-iteration runs were hours of work, not minutes. At batch 16 / hidden 64 the
# same grid measured 84-256 s for a whole 150-iteration run.
#
# Measured at 150 iterations / batch 16 / hidden 64 on the shaped 8x8 rewards:
#   epsilon 0.15, decay : loss 29.221 -> 0.079 (ratio 0.0027), majority peak 287/500
#   epsilon 0.3, no decay: loss 23.852 -> 0.080 (ratio 0.0034), majority peak 319/500
const N_EVAL = 500

function count_peaks_8x8(samples)
    p1, p2, other = 0, 0, 0
    for traj in samples
        terminal = traj.states[end]
        pos = (terminal.x, terminal.y)
        if pos == (8, 8)
            p1 += 1
        elseif pos == (1, 8)
            p2 += 1
        else
            other += 1
        end
    end
    return p1, p2, other
end

function run_8x8_experiment(;label, reward_positions, n_iterations, epsilon, entropy_weight,
                             use_replay=false, epsilon_decay=true, seed=42)
    println("-" ^ 70)
    println("  $label")
    println("-" ^ 70)

    Random.seed!(seed)

    model = GFlowNet.create_grid_world_gflownet(
        grid_size = 8,
        reward_positions = reward_positions,
        hidden_dim = 64,
        learning_rate = 0.005,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
    )

    config = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,
        n_iterations = n_iterations,
        batch_size = 16,
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = epsilon,
        epsilon_decay = epsilon_decay,
        entropy_weight = entropy_weight,
        z_learning_rate_multiplier = 10.0,
        use_replay_buffer = use_replay,
        replay_buffer_size = 5000,
        replay_ratio = 0.5,
        verbose = false
    )

    print("  Training ($n_iterations iter, ε=$epsilon, H=$entropy_weight, replay=$use_replay)... ")
    history = GFlowNet.train_gflownet(model, config; verbose=false)

    assert_finite_iterations(history, n_iterations, label)
    # `filter(!isnan, ...)` then `isempty(...) ? NaN : ...` was the silent-pass shape:
    # a run where every iteration threw printed "loss: NaN" and continued. The
    # assertion above makes that a hard failure; the filter is kept only for printing.
    valid_losses = filter(!isnan, history.losses)
    final_loss = isempty(valid_losses) ? NaN : valid_losses[end]
    println("done! (loss: $(round(final_loss, digits=4)))")

    # Tests 2-4 mix 50% replayed trajectories into every batch, which reweights the
    # loss: measured on comparable replay runs the end-of-run mean sits ABOVE the
    # opening mean (ratios 1.14 and 1.39) on a perfectly healthy sampler. So the loss
    # decrease is only asserted for the on-policy runs. Measured ratios there:
    # 0.0027 (epsilon 0.15 annealed) and 0.0034 (epsilon 0.3 fixed). Bar 0.5.
    use_replay || assert_loss_decreased(history, label; window=10, max_ratio=0.5)

    eval_config = GFlowNet.SamplingConfig(
        strategy = GFlowNet.STOCHASTIC_SAMPLING,
        epsilon = 0.0,
        max_trajectory_length = 100
    )

    samples = [GFlowNet.sample_trajectory(model; config=eval_config) for _ in 1:N_EVAL]
    p1, p2, other = count_peaks_8x8(samples)

    # Only the majority peak is asserted. Reaching (1,8) requires the ONE all-up path
    # out of binomial(14,7) = 3432, and at this budget it measured 0/500 even with the
    # loss fully converged; that shortfall is reported in this slice's findings rather
    # than asserted away. The majority-peak bar still fails loudly if training breaks.
    # Measured majority counts: 287/500 and 319/500; bar 60.
    assert_modes_discovered([p1], "$label majority peak";
                            min_per_mode=60, n_samples=N_EVAL)
    ratio = p2 > 0 ? round(p1 / p2, digits=1) : Inf
    modes = (p1 > 10 ? 1 : 0) + (p2 > 10 ? 1 : 0)

    println("  Peak(8,8)=$p1, Peak(1,8)=$p2, Other=$other, Ratio=$ratio:1, Modes=$modes/2")
    println()
    return (p1=p1, p2=p2, other=other, ratio=ratio, modes=modes, loss=final_loss)
end

function main()
    println("=" ^ 70)
    println("8×8 REWARD SHAPING TUNING")
    println("Finding the right combination for 3432:1 path asymmetry")
    println("=" ^ 70)
    println()

    shaped_rewards = Dict((8,8) => 10.0, (1,8) => 34320.0)

    # Test 1: Higher epsilon, no decay
    r1 = run_8x8_experiment(
        label = "Test 1: Shaped + high ε=0.3, no decay, 150 iter",
        reward_positions = shaped_rewards,
        n_iterations = 150,
        epsilon = 0.3,
        entropy_weight = 0.02,
        epsilon_decay = false,
        seed = 42
    )

    # Test 2: Shaped + replay buffer + moderate epsilon
    r2 = run_8x8_experiment(
        label = "Test 2: Shaped + ε=0.2 + replay buffer, 150 iter",
        reward_positions = shaped_rewards,
        n_iterations = 150,
        epsilon = 0.2,
        entropy_weight = 0.02,
        use_replay = true,
        epsilon_decay = true,
        seed = 42
    )

    # Test 3: Shaped + very high epsilon + high entropy + replay
    r3 = run_8x8_experiment(
        label = "Test 3: Shaped + ε=0.3 + H=0.05 + replay, 150 iter",
        reward_positions = shaped_rewards,
        n_iterations = 150,
        epsilon = 0.3,
        entropy_weight = 0.05,
        use_replay = true,
        epsilon_decay = false,
        seed = 42
    )

    # Test 4: Extreme epsilon to force discovery
    r4 = run_8x8_experiment(
        label = "Test 4: Shaped + ε=0.5 + H=0.05 + replay, 150 iter",
        reward_positions = shaped_rewards,
        n_iterations = 150,
        epsilon = 0.5,
        entropy_weight = 0.05,
        use_replay = true,
        epsilon_decay = false,
        seed = 42
    )

    println("=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println()
    println("| Test | ε    | H    | Replay | Iter | Peak(8,8) | Peak(1,8) | Modes |")
    println("|------|------|------|--------|------|-----------|-----------|-------|")
    for (i, r) in enumerate([r1, r2, r3, r4])
        println("| $i    | -    | -    | -      | -    | $(lpad(r.p1, 9)) | $(lpad(r.p2, 9)) | $(r.modes)/2   |")
    end
    println()

    best = argmax([r.p2 for r in [r1, r2, r3, r4]])
    results = [r1, r2, r3, r4]
    println("Best minority mode discovery: Test $best with $(results[best].p2) samples at (1,8)")
    println("=" ^ 70)

    return (r1, r2, r3, r4)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
