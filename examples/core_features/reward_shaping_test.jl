# Reward Shaping Test - Compensating for 70:1 Path Asymmetry
# Run with: julia --project=. examples/core_features/reward_shaping_test.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using GFlowNet
using Random

include(joinpath(@__DIR__, "..", "convergence_assertions.jl"))

Random.seed!(42)

function test_reward_shaping()
    println("=" ^ 70)
    println("REWARD SHAPING TEST - Compensating for 70:1 Path Asymmetry")
    println("=" ^ 70)
    println()

    # 5×5 grid - compensate for path asymmetry with reward
    # Peak (5,5): 70 paths × R=10 = 700 weighted
    # Peak (1,5): 1 path × R=700 = 700 weighted (balanced!)
    grid_size = 5

    println("Theory: P(x) ∝ R(x) × #paths(x)")
    println("  Peak (5,5): 70 paths × R=10 = 700")
    println("  Peak (1,5): 1 path × R=700 = 700")
    println("  → Both modes should be discovered with ~1:1 ratio")
    println()

    # Test with compensated rewards
    reward_positions = Dict(
        (5, 5) => 10.0,   # 70 paths
        (1, 5) => 700.0   # 1 path, but 70x higher reward
    )

    model = GFlowNet.create_grid_world_gflownet(
        grid_size = grid_size,
        reward_positions = reward_positions,
        hidden_dim = 64,
        learning_rate = 0.005,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION
    )

    # Three changes from the original config, ALL of them forced by measurement, and
    # together they are what finally makes this script's own claim come true.
    #
    # Sweep 1 -- budget and replay buffer (500 eval samples, epsilon 0.1 annealed):
    #   iters  replay  peak(5,5)  peak(1,5)   loss ratio
    #     200  yes           139         10       2.84
    #     400  yes           212         30       2.21
    #     800  yes           310         29       1.06
    #     200  no            345          1       0.0012
    #     400  no            205        148       0.063
    #     800  no            238        206       0.035
    # The replay buffer was PREVENTING the result: at 800 iterations with replay the
    # minority mode is still 29/500, so 4x the budget does not rescue it.
    #
    # Sweep 2 -- the no-replay result above turned out to be initialisation-dependent
    # rather than reliable: the same 400-iteration setup gave (205,148) on one RNG
    # state and (315,0) on another. Annealing epsilon to 0 is why. The minority peak
    # is reachable by exactly ONE path out of binomial(8,4) = 70, so it only stays in
    # the training distribution while epsilon-uniform exploration is still on.
    # Measured at 400 iterations across seeds [42,1,2]:
    #   epsilon 0.1, no decay              -> (310,3)  (337,0)  (337,18)
    #   epsilon 0.3, no decay              -> (319,7)  (239,9)  (274,0)
    #   epsilon 0.3, no decay, Z lr x10    -> (302,27) (278,13) (173,189)
    #
    # Sweep 3 -- the last recipe at 800 iterations, across seeds [42,1,2,3,7]:
    #   (204,203) (204,209) (205,186) (192,195) (210,190)
    # Every seed lands on the 1:1 split that theory predicts -- (5,5) is R=10 over 70
    # paths = 700 weighted, (1,5) is R=700 over 1 path = 700 weighted -- and the worst
    # minority count over five seeds is 186/500. That is the configuration used here.
    config = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
        n_iterations = 800,
        batch_size = 16,
        learning_rate = 0.005,
        temperature = 1.0,
        epsilon = 0.3,
        epsilon_decay = false,          # MUST stay on: the minority mode has 1 path
        entropy_weight = 0.01,
        z_learning_rate_multiplier = 10.0,
        use_replay_buffer = false,      # measurably prevents minority-mode discovery
        verbose = false
    )

    print("Training with reward shaping (R(1,5)=700)... ")
    history = GFlowNet.train_gflownet(model, config; verbose=false)
    println("done!")

    assert_finite_iterations(history, config.n_iterations, "reward shaping")
    # On-policy batches, so the loss is a genuine progress statistic. Measured ratio
    # 0.035 at 800 iterations without replay; bar 0.5.
    assert_loss_decreased(history, "reward shaping"; window=25, max_ratio=0.5)

    eval_config = GFlowNet.SamplingConfig(
        strategy = GFlowNet.STOCHASTIC_SAMPLING,
        epsilon = 0.0,
        max_trajectory_length = 100
    )

    n_samples = 500
    samples = [GFlowNet.sample_trajectory(model; config=eval_config) for _ in 1:n_samples]

    p1 = 0
    p2 = 0
    for traj in samples
        terminal = traj.states[end]
        pos = (terminal.x, terminal.y)
        if pos == (5, 5)
            p1 += 1
        elseif pos == (1, 5)
            p2 += 1
        end
    end

    println()
    println("Results WITH reward shaping:")
    println("  Peak (5,5) R=10:  $p1 / $n_samples ($(round(p1/n_samples*100, digits=1))%)")
    println("  Peak (1,5) R=700: $p2 / $n_samples ($(round(p2/n_samples*100, digits=1))%)")
    modes = (p1 > 10 ? 1 : 0) + (p2 > 10 ? 1 : 0)
    println("  Modes discovered: $modes / 2")

    # THE claim of this script: shaping the reward by the path ratio makes the 1-path
    # mode as likely as the 70-path mode. Measured across five seeds at this exact
    # configuration: (204,203) (204,209) (205,186) (192,195) (210,190), so the worst
    # minority count observed is 186/500. Bar 80 is ~2.3x below that and an order of
    # magnitude above a collapsed run, which measures 0-3.
    assert_modes_discovered([p1, p2], "reward shaping mode balance";
                            min_per_mode=80, n_samples=n_samples)

    if p2 > 10
        actual_ratio = p1 / max(p2, 1)
        println("  Actual sampling ratio: $(round(actual_ratio, digits=2)):1")
        println()
        println("✅ SUCCESS: Reward shaping SOLVES the extreme mode collapse!")
        println("   By compensating the 70:1 path asymmetry with reward,")
        println("   both modes are now discovered.")
    else
        println()
        println("⚠️  Still collapsed even with reward shaping")
    end
    println()

    return (p1, p2)
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_reward_shaping()
end
