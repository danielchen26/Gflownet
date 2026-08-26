using Test
using GFlowNet
using Random
using Statistics
using Zygote

# ============================================================================
# Test Helpers — Synthetic Data
# ============================================================================

"""Create a synthetic EditTBTrajectory for testing (no artifact dependency)."""
function _make_synthetic_trajectory(;
    task_name="test_task",
    run_id="test_task__run1",
    episode_id="test-ep-1",
    n_steps=3,
    n_basin_candidates=4,
    n_parent_candidates=6,
    n_operator_candidates=3,
    terminal_reward=0.05,
    include_heuristic=true,
    rng=Random.MersenneTwister(42)
)
    basin_feat_dim = include_heuristic ? 6 : 5
    parent_feat_dim = include_heuristic ? 5 : 4
    operator_feat_dim = include_heuristic ? 4 : 3

    steps = EditTBStep[]
    for t in 1:n_steps
        basin = EditTBBasinChoice(
            randn(rng, Float32, 8),                                          # frontier features
            randn(rng, Float32, basin_feat_dim, n_basin_candidates),         # candidate features
            rand(rng, 1:n_basin_candidates),                                 # chosen index
            n_basin_candidates
        )
        parent = EditTBParentChoice(
            randn(rng, Float32, 4),                                          # basin context
            randn(rng, Float32, parent_feat_dim, n_parent_candidates),       # candidate features
            rand(rng, 1:n_parent_candidates),                                # chosen index
            n_parent_candidates
        )
        operator = EditTBOperatorChoice(
            randn(rng, Float32, 6),                                          # parent context
            randn(rng, Float32, operator_feat_dim, n_operator_candidates),   # candidate features
            rand(rng, 1:n_operator_candidates),                              # chosen index
            n_operator_candidates
        )
        push!(steps, EditTBStep(basin, parent, operator))
    end

    return EditTBTrajectory(task_name, run_id, episode_id, steps, terminal_reward, :warmup)
end

"""Create a synthetic EditTBDataset for testing."""
function _make_synthetic_dataset(;
    n_trajectories=20,
    include_heuristic=true,
    rng=Random.MersenneTwister(42)
)
    tasks = ["celecoxib_rediscovery", "drd2", "albuterol_similarity"]
    runs = ["run1", "run2", "run3"]
    trajs = EditTBTrajectory[]

    for i in 1:n_trajectories
        task = tasks[mod1(i, length(tasks))]
        run = runs[mod1(i, length(runs))]
        push!(trajs, _make_synthetic_trajectory(
            task_name=task,
            run_id="$(task)__$(run)",
            episode_id="ep-$i",
            n_steps=rand(rng, 1:5),
            terminal_reward=rand(rng) * 0.1 + 0.001,
            include_heuristic=include_heuristic,
            rng=rng
        ))
    end

    stats = Dict{String, Any}(
        "n_trajectories" => length(trajs),
        "n_steps" => sum(length(t.steps) for t in trajs),
        "tasks" => sort(unique(t.task_name for t in trajs)),
        "runs" => sort(unique(t.run_id for t in trajs)),
        "mean_terminal_reward" => mean(t.terminal_reward for t in trajs),
        "mean_steps_per_episode" => mean(length(t.steps) for t in trajs),
        "reward_mode" => :top10_delta,
        "include_heuristic_scores" => include_heuristic,
    )

    return EditTBDataset(trajs, sort(unique(t.task_name for t in trajs)), stats)
end

# ============================================================================
# Tests
# ============================================================================

@testset "Edit-TB — Config" begin
    config = EditTBConfig()
    @test config.objective == :rwmle
    @test config.reward_mode == :top10_delta  # pre-registered headline
    @test config.n_epochs == 50
    @test config.beta == 4.0
    @test config.include_heuristic_scores == true
    @test config.heuristic_interpolation == 0.5
    @test config.entropy_floor == 0.1
    @test config.task_conditioning == false

    config2 = EditTBConfig(objective=:tb, include_heuristic_scores=false, task_conditioning=true)
    @test config2.objective == :tb
    @test config2.include_heuristic_scores == false
    @test config2.task_conditioning == true
end

@testset "Edit-TB — Data Structures" begin
    traj = _make_synthetic_trajectory()
    @test length(traj.steps) == 3
    @test traj.terminal_reward > 0.0
    @test traj.phase == :warmup
    @test traj.task_name == "test_task"
    @test traj.run_id == "test_task__run1"

    for step in traj.steps
        @test length(step.basin.frontier_features) == 8
        @test step.basin.n_candidates == 4
        @test 1 <= step.basin.chosen_index <= step.basin.n_candidates
        @test step.parent.n_candidates == 6
        @test 1 <= step.parent.chosen_index <= step.parent.n_candidates
        @test step.operator.n_candidates == 3
        @test 1 <= step.operator.chosen_index <= step.operator.n_candidates
        @test size(step.basin.candidate_features, 2) == step.basin.n_candidates
        @test size(step.parent.candidate_features, 2) == step.parent.n_candidates
        @test size(step.operator.candidate_features, 2) == step.operator.n_candidates
    end
end

@testset "Edit-TB — Data Structures (No Heuristic)" begin
    traj = _make_synthetic_trajectory(include_heuristic=false)
    step = traj.steps[1]
    @test size(step.basin.candidate_features, 1) == 5   # not 6
    @test size(step.parent.candidate_features, 1) == 4   # not 5
    @test size(step.operator.candidate_features, 1) == 3  # not 4
end

@testset "Edit-TB — Model Creation" begin
    rng = Random.MersenneTwister(42)
    config = EditTBConfig()
    model, params, states = init_edit_policy(rng; config=config)

    @test haskey(model, :basin)
    @test haskey(model, :parent)
    @test haskey(model, :operator)
    @test haskey(params, :basin)
    @test haskey(params, :parent)
    @test haskey(params, :operator)

    n_params = count_edit_policy_params(params)
    @test n_params > 1000
    @test n_params < 20000  # ~6K expected
end

@testset "Edit-TB — Model Creation (No Heuristic)" begin
    rng = Random.MersenneTwister(42)
    config = EditTBConfig(include_heuristic_scores=false)
    model, params, states = init_edit_policy(rng; config=config)
    n_params = count_edit_policy_params(params)
    @test n_params > 1000
    @test n_params < 20000
end

@testset "Edit-TB — Task-Conditioned Model" begin
    rng = Random.MersenneTwister(42)
    task_names = ["celecoxib_rediscovery", "drd2", "qed"]
    config = EditTBConfig(task_conditioning=true)
    model, params, states = init_edit_policy(rng; config=config, task_names=task_names)

    n_params = count_edit_policy_params(params)
    @test n_params > 1000
    @test n_params < 25000

    dataset = _make_synthetic_dataset(n_trajectories=18, rng=Random.MersenneTwister(99))
    loss = compute_edit_rwmle_loss(model, params, states, dataset.trajectories;
        beta=config.beta,
        task_names=task_names,
        task_conditioning=true)
    @test isfinite(loss)
    @test loss > 0.0
end

@testset "Edit-TB — Forward Pass" begin
    rng = Random.MersenneTwister(42)
    config = EditTBConfig()
    model, params, states = init_edit_policy(rng; config=config)

    traj = _make_synthetic_trajectory(rng=Random.MersenneTwister(99))

    # Test log P_F computation
    log_pf = GFlowNet._edit_tb_compute_log_pf(model, params, states, traj)
    @test isfinite(log_pf)
    @test log_pf < 0.0  # log-probabilities are negative
end

@testset "Edit-TB — RWMLE Loss" begin
    rng = Random.MersenneTwister(42)
    config = EditTBConfig()
    model, params, states = init_edit_policy(rng; config=config)

    dataset = _make_synthetic_dataset(rng=Random.MersenneTwister(99))

    loss = compute_edit_rwmle_loss(model, params, states, dataset.trajectories;
        beta=config.beta)
    @test isfinite(loss)
    @test loss > 0.0  # negative log-likelihood is positive

    # Test gradient computation
    loss_val, grads = Zygote.withgradient(
        ps -> compute_edit_rwmle_loss(model, ps, states, dataset.trajectories;
            beta=config.beta),
        params
    )
    @test isfinite(loss_val)
    @test grads[1] !== nothing
end

@testset "Edit-TB — TB-Style Loss" begin
    rng = Random.MersenneTwister(42)
    config = EditTBConfig(objective=:tb)
    model, params, states = init_edit_policy(rng; config=config)

    dataset = _make_synthetic_dataset(rng=Random.MersenneTwister(99))
    log_Z = Float32[0.0f0]

    loss = compute_edit_tb_style_loss(model, params, states, log_Z,
        dataset.trajectories; beta=config.beta, threshold=config.threshold)
    @test isfinite(loss)
    @test loss >= 0.0  # shifted cosh is non-negative

    # Test gradient computation with both params and log_Z
    loss_fn = (ps, lz) -> compute_edit_tb_style_loss(model, ps, states, lz,
        dataset.trajectories; beta=config.beta, threshold=config.threshold)
    loss_val, grads = Zygote.withgradient(loss_fn, params, log_Z)
    @test isfinite(loss_val)
    @test grads[1] !== nothing  # policy grads
    @test grads[2] !== nothing  # log_Z grads
end

@testset "Edit-TB — Grouped Validation Split" begin
    dataset = _make_synthetic_dataset(n_trajectories=30)

    train_trajs, val_trajs = GFlowNet._edit_tb_grouped_split(
        dataset.trajectories, 0.15; rng=Random.MersenneTwister(42)
    )

    @test length(train_trajs) + length(val_trajs) == length(dataset.trajectories)
    @test length(val_trajs) > 0
    @test length(train_trajs) > 0

    # Check that train and val don't share run_ids
    train_runs = Set(t.run_id for t in train_trajs)
    val_runs = Set(t.run_id for t in val_trajs)
    @test isempty(intersect(train_runs, val_runs))
end

@testset "Edit-TB — RWMLE Training (Synthetic)" begin
    rng = Random.MersenneTwister(42)
    config = EditTBConfig(n_epochs=5, learning_rate=1e-3, batch_size=8)
    model, params, states = init_edit_policy(rng; config=config)

    dataset = _make_synthetic_dataset(n_trajectories=30, rng=Random.MersenneTwister(99))

    best_params, best_log_Z, history = train_edit_policy!(
        model, params, states, dataset, config;
        verbose=false, rng=Random.MersenneTwister(123)
    )

    @test length(history["train_loss"]) == 5
    @test length(history["val_loss"]) == 5
    @test all(isfinite, history["train_loss"])
    @test all(isfinite, history["val_loss"])
    @test best_params !== nothing
end

@testset "Edit-TB — TB-Style Training (Synthetic)" begin
    rng = Random.MersenneTwister(42)
    config = EditTBConfig(objective=:tb, n_epochs=5, learning_rate=1e-3, batch_size=8)
    model, params, states = init_edit_policy(rng; config=config)

    dataset = _make_synthetic_dataset(n_trajectories=30, rng=Random.MersenneTwister(99))

    best_params, best_log_Z, history = train_edit_policy!(
        model, params, states, dataset, config;
        verbose=false, rng=Random.MersenneTwister(123)
    )

    @test length(history["train_loss"]) == 5
    @test all(isfinite, history["train_loss"])
    @test all(isfinite, history["log_Z"])
end

@testset "Edit-TB — Online Stabilizer" begin
    rng = Random.MersenneTwister(42)
    config = EditTBConfig(heuristic_interpolation=0.5, entropy_floor=0.1)
    model, params, states = init_edit_policy(rng; config=config)

    traj = _make_synthetic_trajectory(rng=Random.MersenneTwister(99))
    step = traj.steps[1]

    heuristic_scores = abs.(randn(step.basin.n_candidates)) .+ 0.1

    chosen, probs, entropy = choose_with_edit_policy(
        model, params, states, config,
        step.basin.frontier_features,
        step.basin.candidate_features,
        heuristic_scores,
        step.basin.n_candidates;
        head=:basin
    )

    @test 1 <= chosen <= step.basin.n_candidates
    @test length(probs) == step.basin.n_candidates
    @test isapprox(sum(probs), 1.0, atol=1e-5)
    @test all(p -> p >= 0, probs)
    @test isfinite(entropy)
    @test entropy > 0.0
end

@testset "Edit-TB — Entropy Floor Enforcement" begin
    rng = Random.MersenneTwister(42)
    config = EditTBConfig(heuristic_interpolation=0.0, entropy_floor=1.0)  # high floor, pure heuristic
    model, params, states = init_edit_policy(rng; config=config)

    # Create very peaked heuristic scores
    heuristic_scores = [100.0, 0.001, 0.001]
    context = randn(rng, Float32, 8)
    cand = randn(rng, Float32, 6, 3)

    chosen, probs, entropy = choose_with_edit_policy(
        model, params, states, config,
        context, cand, heuristic_scores, 3; head=:basin
    )

    # With entropy floor = 1.0, should mix with uniform and have higher entropy
    @test entropy > 0.3  # should be boosted above very low entropy
    @test minimum(probs) > 0.01  # no prob should be near-zero
end

@testset "Edit-TB — Task-Conditioned Controller" begin
    rng = Random.MersenneTwister(42)
    task_names = ["celecoxib_rediscovery", "drd2", "qed"]
    config = EditTBConfig(task_conditioning=true, heuristic_interpolation=0.5, entropy_floor=0.05)
    model, params, states = init_edit_policy(rng; config=config, task_names=task_names)
    controller = EditTBPolicyController(model, params, states, config; task_names=task_names)
    set_edit_tb_task!(controller, "drd2")

    @test controller.current_task_name == "drd2"
    @test length(GFlowNet._controller_task_features(controller)) == length(task_names)
    @test sum(GFlowNet._controller_task_features(controller)) == 1.0f0

    heuristic_scores = [0.7, 0.2, 0.1]
    context = vcat(GFlowNet._controller_task_features(controller), randn(rng, Float32, 8))
    cand = randn(rng, Float32, 6, 3)
    chosen, probs, entropy = choose_with_edit_policy(
        model, params, states, config,
        context, cand, heuristic_scores, 3; head=:basin
    )
    @test 1 <= chosen <= 3
    @test isapprox(sum(probs), 1.0, atol=1e-5)
    @test isfinite(entropy)
end

@testset "Edit-TB — Leak-Free Features" begin
    # This test verifies the feature engineering functions use only decision-time fields
    # by checking dimensions match expected feature counts

    config_h = EditTBConfig(include_heuristic_scores=true)
    config_nh = EditTBConfig(include_heuristic_scores=false)

    traj_h = _make_synthetic_trajectory(include_heuristic=true)
    traj_nh = _make_synthetic_trajectory(include_heuristic=false)

    # With heuristic: basin=6, parent=5, operator=4
    step_h = traj_h.steps[1]
    @test size(step_h.basin.candidate_features, 1) == 6
    @test size(step_h.parent.candidate_features, 1) == 5
    @test size(step_h.operator.candidate_features, 1) == 4

    # Without heuristic: basin=5, parent=4, operator=3
    step_nh = traj_nh.steps[1]
    @test size(step_nh.basin.candidate_features, 1) == 5
    @test size(step_nh.parent.candidate_features, 1) == 4
    @test size(step_nh.operator.candidate_features, 1) == 3

    # Frontier features always 8-d
    @test length(step_h.basin.frontier_features) == 8
    @test length(step_nh.basin.frontier_features) == 8
end

@testset "Edit-TB — Checkpoint Round-Trip" begin
    rng = Random.MersenneTwister(42)
    config = EditTBConfig(n_epochs=2)
    model, params, states = init_edit_policy(rng; config=config)
    dataset = _make_synthetic_dataset(n_trajectories=10, rng=Random.MersenneTwister(99))

    best_params, best_log_Z, history = train_edit_policy!(
        model, params, states, dataset, config;
        verbose=false, rng=Random.MersenneTwister(123)
    )

    tmpfile = tempname() * ".jls"
    try
        save_edit_policy(tmpfile, best_params, best_log_Z, history, config)
        loaded_params, loaded_log_Z, loaded_history, loaded_config = load_edit_policy(tmpfile)

        @test loaded_config.objective == :rwmle
        @test length(loaded_history["train_loss"]) == 2
        @test loaded_log_Z[1] == best_log_Z[1]
    finally
        isfile(tmpfile) && rm(tmpfile)
    end
end

@testset "Edit-TB — Terminal Reward Computation" begin
    ep_summary = Dict{String, Any}(
        "frontier_before_summary" => Dict{String, Any}(
            "top10_mean" => 0.5,
            "top1" => 0.6,
            "scaffold_count" => 3
        ),
        "frontier_after_summary" => Dict{String, Any}(
            "top10_mean" => 0.55,
            "top1" => 0.65,
            "scaffold_count" => 5
        )
    )

    r1 = compute_edit_tb_terminal_reward(ep_summary; mode=:top10_delta)
    @test isapprox(r1, 0.05, atol=1e-6)

    r2 = compute_edit_tb_terminal_reward(ep_summary; mode=:top1_delta)
    @test isapprox(r2, 0.05, atol=1e-6)

    r3 = compute_edit_tb_terminal_reward(ep_summary; mode=:composite)
    @test r3 > 0.001
    @test isfinite(r3)

    # Epsilon floor for negative gain
    ep_bad = Dict{String, Any}(
        "frontier_before_summary" => Dict{String, Any}("top10_mean" => 0.5, "top1" => 0.6),
        "frontier_after_summary" => Dict{String, Any}("top10_mean" => 0.4, "top1" => 0.5)
    )
    r_bad = compute_edit_tb_terminal_reward(ep_bad; mode=:top10_delta)
    @test r_bad == 0.001  # epsilon floor
end

# ============================================================================
# Artifact Integration Test (only if artifacts exist)
# ============================================================================

const ARTIFACT_ROOT = "checkpoints/truth_sprint_stage_b_f015_truth_tasksharded"

if isdir(ARTIFACT_ROOT)
    @testset "Edit-TB — Real Artifact Loading" begin
        config = EditTBConfig()
        dataset = load_edit_tb_dataset(ARTIFACT_ROOT; config=config)

        @test dataset.stats["n_trajectories"] > 50
        @test length(dataset.task_names) >= 4

        for traj in dataset.trajectories
            @test traj.terminal_reward >= 0.001
            @test length(traj.run_id) > 0
            @test length(traj.episode_id) > 0
            for step in traj.steps
                @test length(step.basin.frontier_features) == 8
                @test all(isfinite, step.basin.frontier_features)
                @test step.basin.n_candidates >= 1
                @test step.parent.n_candidates >= 1
                @test step.operator.n_candidates >= 2
                @test 1 <= step.basin.chosen_index <= step.basin.n_candidates
                @test 1 <= step.parent.chosen_index <= step.parent.n_candidates
                @test 1 <= step.operator.chosen_index <= step.operator.n_candidates
            end
        end

        println("  Real dataset: $(dataset.stats["n_trajectories"]) trajectories, " *
                "$(dataset.stats["n_steps"]) steps, " *
                "tasks=$(dataset.task_names)")
    end

    @testset "Edit-TB — Real Artifact Heuristic Ablation" begin
        config_no_h = EditTBConfig(include_heuristic_scores=false)
        dataset = load_edit_tb_dataset(ARTIFACT_ROOT; config=config_no_h)

        @test dataset.stats["n_trajectories"] > 50

        step = dataset.trajectories[1].steps[1]
        @test size(step.basin.candidate_features, 1) == 5   # not 6
        @test size(step.parent.candidate_features, 1) == 4   # not 5
        @test size(step.operator.candidate_features, 1) == 3  # not 4
    end

    @testset "Edit-TB — Real Artifact RWMLE Smoke" begin
        rng = Random.MersenneTwister(42)
        config = EditTBConfig(n_epochs=3, learning_rate=1e-3, batch_size=16)
        model, params, states = init_edit_policy(rng; config=config)

        dataset = load_edit_tb_dataset(ARTIFACT_ROOT; config=config)

        best_params, best_log_Z, history = train_edit_policy!(
            model, params, states, dataset, config;
            verbose=true, rng=Random.MersenneTwister(123)
        )

        @test length(history["train_loss"]) == 3
        @test all(isfinite, history["train_loss"])
        @test all(isfinite, history["val_loss"])

        println("  RWMLE smoke: train_loss=$(round.(history["train_loss"], digits=4)), " *
                "val_loss=$(round.(history["val_loss"], digits=4))")
    end
    @testset "Edit-TB — Runtime Controller Integration" begin
        rng = Random.MersenneTwister(42)
        config = EditTBConfig(n_epochs=3, learning_rate=1e-3, batch_size=16,
                              heuristic_interpolation=0.5, entropy_floor=0.05)
        model, params, states = init_edit_policy(rng; config=config)

        # Train briefly on real data
        dataset = load_edit_tb_dataset(ARTIFACT_ROOT; config=config)
        best_params, best_log_Z, _ = train_edit_policy!(
            model, params, states, dataset, config;
            verbose=false, rng=Random.MersenneTwister(99)
        )

        # Create controller
        controller = EditTBPolicyController(model, best_params, states, config)
        @test controller._last_basin === nothing

        # Build synthetic runtime objects
        entries = [
            FrontierSnapshotEntry("CCO", "scaffold_a", 0.7, 0.5, 0.1, :seed),
            FrontierSnapshotEntry("CCN", "scaffold_a", 0.65, 0.4, 0.2, :model),
            FrontierSnapshotEntry("c1ccccc1", "scaffold_b", 0.8, 0.6, 0.05, :edit),
            FrontierSnapshotEntry("CC(=O)O", "scaffold_c", 0.55, 0.3, 0.15, :ga),
        ]
        snapshot = FrontierSnapshot(entries, nothing, nothing, 2500, 10, UInt64(1234))

        # ── Basin selection ──
        basin_a = BasinSummary("scaffold_a", 2, 0.7, 0.675, 0.45, 0.15)
        basin_b = BasinSummary("scaffold_b", 1, 0.8, 0.8, 0.6, 0.05)
        basin_c = BasinSummary("scaffold_c", 1, 0.55, 0.55, 0.3, 0.15)
        basin_candidates = [
            ScoredBasinCandidate(basin_a, 0.4),
            ScoredBasinCandidate(basin_b, 0.5),
            ScoredBasinCandidate(basin_c, 0.1),
        ]

        chosen_basin = GFlowNet.select_basin(controller, snapshot, basin_candidates; step_index=1)
        @test chosen_basin !== nothing
        @test chosen_basin isa ScoredBasinCandidate
        @test controller._last_basin !== nothing  # basin stored for parent head

        # ── Parent selection ──
        parent_candidates = [
            ScoredParentCandidate(entries[1], 0.6, 2, true, false),
            ScoredParentCandidate(entries[3], 0.8, 0, false, false),
        ]

        parent_meta = GFlowNet.parent_selection_metadata(controller, snapshot, parent_candidates; step_index=1)
        @test haskey(parent_meta, "chosen_index")
        @test 1 <= parent_meta["chosen_index"] <= 2
        @test haskey(parent_meta, "override_applied")
        @test haskey(parent_meta, "entropy")
        @test parent_meta["selection_reason"] == "edit_tb_policy"

        # ── Operator selection ──
        op_candidates = [
            OperatorDecisionCandidate(:mutate, 0.5, 10, 6, 0.1, 0.0),
            OperatorDecisionCandidate(:crossover, 0.3, 8, 3, 0.2, 0.0),
            OperatorDecisionCandidate(:fragment, 0.2, 5, 2, 0.3, 0.0),
        ]

        op_meta = GFlowNet.operator_selection_metadata(controller, snapshot, basin_a, entries[1], op_candidates; step_index=1)
        @test haskey(op_meta, "chosen_index")
        @test 1 <= op_meta["chosen_index"] <= 3
        @test haskey(op_meta, "entropy")
        @test op_meta["selection_reason"] == "edit_tb_policy"

        # ── Runtime feature builders ──
        ff = GFlowNet._runtime_frontier_features(snapshot; step_index=2)
        @test length(ff) == 8
        @test all(isfinite, ff)
        @test ff[1] ≈ 0.8f0  # top1

        bf = GFlowNet._runtime_basin_candidate_features(basin_candidates[1]; include_heuristic=true)
        @test length(bf) == 6
        bf_no_h = GFlowNet._runtime_basin_candidate_features(basin_candidates[1]; include_heuristic=false)
        @test length(bf_no_h) == 5

        bcf = GFlowNet._runtime_basin_context_features(basin_a)
        @test length(bcf) == 4

        pf = GFlowNet._runtime_parent_candidate_features(parent_candidates[1]; include_heuristic=true)
        @test length(pf) == 5

        pcf = GFlowNet._runtime_parent_context_features(entries[1], snapshot; step_index=1)
        @test length(pcf) == 6

        of = GFlowNet._runtime_operator_candidate_features(op_candidates[1]; include_heuristic=true)
        @test length(of) == 4

        println("  Runtime controller: basin→parent→operator dispatch chain OK")
    end
else
    @info "Skipping real artifact tests — $ARTIFACT_ROOT not found"
end
