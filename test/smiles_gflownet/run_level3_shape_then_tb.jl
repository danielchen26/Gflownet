# ============================================================================
# Level 3 Search-State-Shaping Test
# ============================================================================
#
# Causal design:
#   Arm A: TB only (full budget)
#   Arm B: heuristic shaping -> TB
#   Arm C: learned shaping -> TB
#
# The theory object is not direct HE authorship. It is whether a shaping phase
# can create a frontier/replay state that makes later pure TB search better.
#
# Usage:
#   julia --project=. test/smiles_gflownet/run_level3_shape_then_tb.jl
#   LEVEL3_TOTAL_BUDGET=256 LEVEL3_REPEATS=2 julia --project=. test/smiles_gflownet/run_level3_shape_then_tb.jl

using Pkg
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(PROJECT_ROOT)

using GFlowNet
using Random
using Serialization
using Statistics
using Dates

include(joinpath(PROJECT_ROOT, "src", "applications", "smiles_gflownet.jl"))
include(joinpath(PROJECT_ROOT, "src", "utils", "visualization", "python", "rdkit_bridge.jl"))
include(joinpath(PROJECT_ROOT, "src", "utils", "visualization", "python", "oracle_bridge.jl"))
include(joinpath(PROJECT_ROOT, "src", "utils", "visualization", "core", "oracle_manager.jl"))
include(joinpath(PROJECT_ROOT, "src", "utils", "visualization", "core", "pmo_benchmark.jl"))

const SIBLING_MAIN_ROOT = normpath(joinpath(PROJECT_ROOT, "..", "Gflownet"))

logmsg(msg) = println("[$(Dates.format(now(), "HH:MM:SS"))] $msg")

function _env_or_empty(key::String)
    return haskey(ENV, key) ? strip(ENV[key]) : ""
end

function _first_existing_path(paths::Vector{String}; want_dir::Bool=false)
    for p in paths
        isempty(p) && continue
        pp = normpath(p)
        if want_dir
            isdir(pp) && return pp
        else
            isfile(pp) && return pp
        end
    end
    return ""
end

function _safe_mean(xs)
    vals = Float64[x for x in xs]
    isempty(vals) && return 0.0
    return mean(vals)
end

function _safe_std(xs)
    vals = Float64[x for x in xs]
    length(vals) <= 1 && return 0.0
    return std(vals)
end

# ============================================================================
# Config
# ============================================================================

const TOTAL_BUDGET = parse(Int, get(ENV, "LEVEL3_TOTAL_BUDGET", "256"))
const SHAPING_BUDGET = parse(Int, get(ENV, "LEVEL3_SHAPING_BUDGET", "64"))
const DOWNSTREAM_BUDGET = max(0, TOTAL_BUDGET - SHAPING_BUDGET)
const N_REPEATS = parse(Int, get(ENV, "LEVEL3_REPEATS", "2"))
const METRIC_STRIDE = parse(Int, get(ENV, "LEVEL3_METRIC_STRIDE", "16"))
const THEORY_EPOCHS = parse(Int, get(ENV, "LEVEL3_POLICY_EPOCHS", "40"))
const TASKS = [String(strip(t)) for t in split(get(ENV, "LEVEL3_TASKS", "qed,drd2,celecoxib_rediscovery"), ',') if !isempty(strip(t))]
const LOGDIR = get(ENV, "LEVEL3_LOGDIR", joinpath(PROJECT_ROOT, "checkpoints", "level3_shape_then_tb"))
const PMO_BATCH_SIZE = parse(Int, get(ENV, "PMO_BATCH_SIZE", "32"))
const PMO_REPLAY_RATIO = parse(Int, get(ENV, "PMO_REPLAY_RATIO", "4"))
const HE_BOOTSTRAP_SAMPLES = parse(Int, get(ENV, "LEVEL3_BOOTSTRAP_SAMPLES", "12"))
const HE_SHAPING_EPISODES = parse(Int, get(ENV, "LEVEL3_SHAPING_EPISODES", "2"))
const HE_HORIZON = parse(Int, get(ENV, "LEVEL3_HE_HORIZON", "3"))
const HE_MAX_STEP_ATTEMPTS = parse(Int, get(ENV, "LEVEL3_HE_MAX_STEP_ATTEMPTS", "3"))
const ARTIFACT_ROOT = let env_path = _env_or_empty("LEVEL3_ARTIFACT_ROOT")
    resolved = _first_existing_path(String[
        env_path,
        joinpath(PROJECT_ROOT, "checkpoints", "truth_sprint_stage_b_f015_truth_tasksharded"),
        joinpath(SIBLING_MAIN_ROOT, "checkpoints", "truth_sprint_stage_b_f015_truth_tasksharded"),
    ]; want_dir=true)
    isempty(resolved) && error("Could not locate truth-sprint HE artifacts. Set LEVEL3_ARTIFACT_ROOT.")
    resolved
end
const CHECKPOINT_PATH = let env_path = _env_or_empty("LEVEL3_PRETRAIN_CHECKPOINT")
    resolved = _first_existing_path(String[
        env_path,
        joinpath(PROJECT_ROOT, "checkpoints", "pretrain", "final.jls"),
        joinpath(SIBLING_MAIN_ROOT, "checkpoints", "pretrain", "final.jls"),
    ])
    isempty(resolved) && error("Could not locate pretrained checkpoint. Set LEVEL3_PRETRAIN_CHECKPOINT.")
    resolved
end
const TARGET_SMILES = Dict(
    "celecoxib_rediscovery" => "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
)
const ARM_NAMES = ["tb_only", "heuristic_shape_then_tb", "learned_shape_then_tb"]

mkpath(LOGDIR)

# ============================================================================
# Metric Tracking
# ============================================================================

mutable struct PhaseMetricTracker
    top_scores::Vector{Float64}
    calls_used::Int
    stride::Int
    checkpoints::Vector{Tuple{Int,Float64}}
end

PhaseMetricTracker(; stride::Int=16) = PhaseMetricTracker(Float64[], 0, stride, Tuple{Int,Float64}[])

function _tracker_update!(tracker::PhaseMetricTracker, score::Float64)
    push!(tracker.top_scores, score)
    sort!(tracker.top_scores, rev=true)
    if length(tracker.top_scores) > 10
        resize!(tracker.top_scores, 10)
    end
    tracker.calls_used += 1
    if tracker.calls_used % tracker.stride == 0
        top10_mean = isempty(tracker.top_scores) ? 0.0 : sum(tracker.top_scores) / length(tracker.top_scores)
        push!(tracker.checkpoints, (tracker.calls_used, top10_mean))
    end
    return nothing
end

function _tracker_auc(tracker::PhaseMetricTracker; min_call::Int=1)::Float64
    vals = Float64[val for (call, val) in tracker.checkpoints if call >= min_call]
    isempty(vals) && return 0.0
    return mean(vals)
end

function _tracker_top1(tracker::PhaseMetricTracker)::Float64
    return isempty(tracker.top_scores) ? 0.0 : tracker.top_scores[1]
end

function _tracker_top10(tracker::PhaseMetricTracker)::Float64
    return isempty(tracker.top_scores) ? 0.0 : sum(tracker.top_scores) / length(tracker.top_scores)
end

# ============================================================================
# State Summaries / Cloning
# ============================================================================

function _clone_replay_buffer(buf::Union{Nothing,SMILESReplayBuffer})
    isnothing(buf) && return nothing
    clone = SMILESReplayBuffer(buf.max_size)
    clone.entries = [SMILESReplayEntry(e.smiles, copy(e.tokens), e.reward) for e in buf.entries]
    clone.seen_smiles = copy(buf.seen_smiles)
    clone.needs_sort = buf.needs_sort
    clone.tb_deltas = copy(buf.tb_deltas)
    return clone
end

function _replay_topk_summary(buf::Union{Nothing,SMILESReplayBuffer}; topk::Int=10)
    if isnothing(buf) || isempty(buf)
        return Dict{String,Any}(
            "size" => 0,
            "top1" => 0.0,
            "top10_mean" => 0.0,
            "top_smiles" => Set{String}(),
        )
    end
    top_entries = get_top_molecules(buf, min(topk, length(buf)))
    rewards = [e.reward for e in top_entries]
    return Dict{String,Any}(
        "size" => length(buf),
        "top1" => isempty(rewards) ? 0.0 : rewards[1],
        "top10_mean" => isempty(rewards) ? 0.0 : mean(rewards),
        "top_smiles" => Set{String}(e.smiles for e in top_entries),
    )
end

function _cache_topk_summary(cache::Dict{String,Dict{String,Float64}}, task::String; topk::Int=10)
    pairs = [(smi, get(scores, task, 0.0)) for (smi, scores) in cache if haskey(scores, task)]
    isempty(pairs) && return Dict{String,Any}(
        "size" => 0,
        "top1" => 0.0,
        "top10_mean" => 0.0,
        "top_smiles" => Set{String}(),
    )
    sorted_pairs = sort(pairs, by=x -> -x[2])
    top_pairs = sorted_pairs[1:min(topk, length(sorted_pairs))]
    vals = [x[2] for x in top_pairs]
    return Dict{String,Any}(
        "size" => length(pairs),
        "top1" => isempty(vals) ? 0.0 : vals[1],
        "top10_mean" => isempty(vals) ? 0.0 : mean(vals),
        "top_smiles" => Set{String}(x[1] for x in top_pairs),
    )
end

function _set_overlap_fraction(a, b)::Float64
    sa = Set{String}(String(x) for x in a)
    sb = Set{String}(String(x) for x in b)
    isempty(sa) && return 0.0
    return count(in(sb), sa) / length(sa)
end

function _build_level3_he_config(; controller=nothing)
    return HierarchicalEditConfig(;
        horizon=HE_HORIZON,
        allow_crossover=true,
        allow_fragment_ops=false,
        max_step_attempts=HE_MAX_STEP_ATTEMPTS,
        min_exploration_per_operator=5,
        multi_child_min_reward_ratio=0.2,
        operator_prior_strength=4.0,
        use_learned_basin=!isnothing(controller),
        learned_basin_controller=controller,
        use_learned_parent=!isnothing(controller),
        learned_parent_controller=controller,
        use_learned_operator=!isnothing(controller),
        learned_operator_controller=controller,
    )
end

# ============================================================================
# Oracle Environment
# ============================================================================

function _flatten_task_cache(cache::Dict{String,Dict{String,Float64}}, task::String)
    result = Dict{String,Float64}()
    for (smi, scores) in cache
        if haskey(scores, task)
            result[smi] = scores[task]
        end
    end
    return result
end

function _make_oracle_phase(task::String, budget::Int;
                            existing_cache::Dict{String,Dict{String,Float64}}=Dict{String,Dict{String,Float64}}(),
                            tracker::PhaseMetricTracker=PhaseMetricTracker(stride=METRIC_STRIDE))
    oracle_mgr = OracleManager(
        [OracleConfig(task, 1.0)],
        budget,
        0,
        deepcopy(existing_cache),
        true,
    )
    OracleBridge.init_oracles!([task];
        cache_dir=joinpath(PROJECT_ROOT, "data", "tdc_cache"))

    oracle_cache = _flatten_task_cache(oracle_mgr.cache, task)

    function reward_fn_batch(smiles_list::Vector{String})::Vector{Float64}
        isempty(smiles_list) && return Float64[]

        uncached_raw = String[]
        seen = Set{String}()
        for smiles in smiles_list
            isempty(smiles) && continue
            canonical = canonicalize_smiles_identity(smiles)
            if !haskey(oracle_cache, canonical) && !(canonical in seen)
                push!(uncached_raw, smiles)
                push!(seen, canonical)
            end
        end

        if !isempty(uncached_raw) && !budget_exhausted(oracle_mgr)
            evaluate_molecules!(oracle_mgr, uncached_raw)
            for smiles in uncached_raw
                canonical = canonicalize_smiles_identity(smiles)
                if !haskey(oracle_cache, canonical)
                    score = lookup_score(oracle_mgr, smiles, task)
                    oracle_cache[canonical] = score
                    _tracker_update!(tracker, score)
                end
            end
        end

        return Float64[
            isempty(smiles) ? 0.0 : get(oracle_cache, canonicalize_smiles_identity(smiles), 0.0)
            for smiles in smiles_list
        ]
    end

    function reward_fn(smiles::String)::Float64
        scores = reward_fn_batch([smiles])
        return isempty(scores) ? 0.0 : scores[1]
    end

    return (mgr=oracle_mgr, reward_fn=reward_fn, reward_fn_batch=reward_fn_batch, reward_cache=oracle_cache, tracker=tracker)
end

# ============================================================================
# Shared Setup
# ============================================================================

logmsg("Loading pretrained checkpoint: $CHECKPOINT_PATH")
checkpoint = deserialize(CHECKPOINT_PATH)
const PRETRAINED_PARAMS = checkpoint["params"]
const PRETRAINED_STATES = checkpoint["states"]
const VOCAB = SMILESVocabulary()
actual_vocab_size = size(PRETRAINED_PARAMS.output.layer_2.weight, 1)
const POLICY_MODEL, _, _ = create_smiles_policy(; vocab_size=actual_vocab_size, hidden_dim=512, embed_dim=128, n_layers=3)

logmsg("Loading HE artifacts from: $ARTIFACT_ROOT")
train_config = EditTBConfig(
    n_epochs=THEORY_EPOCHS,
    reward_mode=:top10_delta,
    include_heuristic_scores=true,
    heuristic_interpolation=0.5,
    entropy_floor=0.05,
    task_conditioning=true,
)
dataset = load_edit_tb_dataset(ARTIFACT_ROOT; config=train_config)
logmsg("Dataset: $(dataset.stats["n_trajectories"]) trajectories, $(dataset.stats["n_steps"]) steps, tasks=$(join(dataset.task_names, ", "))")

rng = Random.MersenneTwister(42)
model, params, states = init_edit_policy(rng; config=train_config, task_names=dataset.task_names)
best_params, best_log_Z, history = train_edit_policy!(
    model, params, states, dataset, train_config;
    verbose=true, rng=Random.MersenneTwister(123)
)
controller = EditTBPolicyController(model, best_params, states, train_config; task_names=dataset.task_names)
policy_ckpt = joinpath(LOGDIR, "level3_task_conditioned_policy.jls")
save_edit_policy(policy_ckpt, best_params, best_log_Z, history, train_config)
logmsg("Saved Level 3 shaping controller: $policy_ckpt")

# ============================================================================
# Phase 1: Shaping
# ============================================================================

function _run_shaping_phase(task::String, arm::String, run_idx::Int)
    phase = _make_oracle_phase(task, SHAPING_BUDGET; tracker=PhaseMetricTracker(stride=METRIC_STRIDE))
    oracle_mgr = phase.mgr
    reward_fn = phase.reward_fn
    reward_fn_batch = phase.reward_fn_batch
    tracker = phase.tracker

    replay_buffer = SMILESReplayBuffer(5000)
    frontier_buffer = MolecularFrontierBuffer(5000)
    scaffold_filt = nothing
    target_smi = get(TARGET_SMILES, task, nothing)
    seed_pool = isnothing(target_smi) ? String[] : [target_smi]

    # Shared initialization used only inside the shaping budget
    seed_calls_before = oracle_mgr.calls_used
    if !isempty(seed_pool)
        _seed_memories!(seed_pool, reward_fn, VOCAB;
            reward_fn_batch=reward_fn_batch,
            replay_buffer=replay_buffer,
            frontier_buffer=frontier_buffer,
            scaffold_filter=scaffold_filt,
            augmentation_count=0,
            verbose=false)
    end
    seed_calls = oracle_mgr.calls_used - seed_calls_before

    bootstrap_calls_before = oracle_mgr.calls_used
    _bootstrap_frontier_from_model!(POLICY_MODEL, PRETRAINED_PARAMS, PRETRAINED_STATES, VOCAB, reward_fn;
        reward_fn_batch=reward_fn_batch,
        replay_buffer=replay_buffer,
        frontier_buffer=frontier_buffer,
        scaffold_filter=scaffold_filt,
        batch_size=HE_BOOTSTRAP_SAMPLES,
        min_frontier_entries=2,
        verbose=false)
    bootstrap_calls = oracle_mgr.calls_used - bootstrap_calls_before

    he_trajectory_buffer = EditTrajectoryBuffer(10000)
    he_diagnostics_buffer = HierarchicalEditDiagnosticsBuffer(10000)
    he_episode_summaries = Dict{String,Any}[]

    used_learned = arm == "learned_shape_then_tb"
    local_controller = nothing
    if used_learned
        set_edit_tb_task!(controller, task)
        controller._last_basin = nothing
        local_controller = controller
    end
    he_config = _build_level3_he_config(controller=local_controller)

    frontier_before = frontier_quality_summary(frontier_buffer; topk=10)
    episodes_run = 0
    total_he_calls = 0
    while !budget_exhausted(oracle_mgr) && episodes_run < HE_SHAPING_EPISODES && length(frontier_buffer) >= 2
        episodes_run += 1
        ep_calls_before = oracle_mgr.calls_used
        ep = run_hierarchical_edit_episode!(
            frontier_buffer,
            he_trajectory_buffer,
            reward_fn,
            VOCAB;
            reward_fn_batch=reward_fn_batch,
            diagnostics_buffer=he_diagnostics_buffer,
            config=he_config,
            target_smiles=target_smi,
            budget_remaining=budget_remaining(oracle_mgr),
            created_at_step=-episodes_run,
            task_name=task,
        )
        ep_calls_after = oracle_mgr.calls_used
        total_he_calls += (ep_calls_after - ep_calls_before)

        proposal_logs = _proposal_logs_for_episode(he_diagnostics_buffer, ep.episode_id)
        decision_logs = _decision_logs_for_episode(he_diagnostics_buffer, ep.episode_id)
        basin_logs = _basin_logs_for_episode(he_diagnostics_buffer, ep.episode_id)
        parent_logs = _parent_logs_for_episode(he_diagnostics_buffer, ep.episode_id)
        trajectory_entries = _trajectory_entries_for_episode(he_trajectory_buffer, ep.episode_id)
        push!(he_episode_summaries, _episode_summary(
            ep,
            trajectory_entries,
            proposal_logs,
            decision_logs,
            basin_logs,
            parent_logs;
            phase="shape",
            segment_index=0,
            episode_index=episodes_run,
            calls_before=ep_calls_before,
            calls_after=ep_calls_after,
            budget_remaining_before=max(0, SHAPING_BUDGET - ep_calls_before),
            budget_remaining_after=max(0, SHAPING_BUDGET - ep_calls_after),
            budget_cap_reached=budget_exhausted(oracle_mgr),
            frontier_before_summary=frontier_before,
            frontier_after_summary=frontier_quality_summary(frontier_buffer; topk=10),
            config=he_config,
            run_context=Dict{String,Any}(
                "task_name" => task,
                "arm" => arm,
                "run_index" => run_idx,
            ),
        ))

        # Feed shape discoveries into replay for downstream TB
        he_top = frontier_topk(frontier_buffer, min(32, length(frontier_buffer)); by=:reward)
        for entry in he_top
            entry.source in (:edit, :warmup) || continue
            tokens = try encode(VOCAB, entry.smiles) catch; Int[] end
            length(tokens) < 2 && continue
            add_to_replay!(replay_buffer, entry.smiles, tokens, entry.reward)
        end
        frontier_before = frontier_quality_summary(frontier_buffer; topk=10)
    end

    artifact_dir = joinpath(LOGDIR, "artifacts", arm, task, "run$(run_idx)", "shape")
    mkpath(artifact_dir)
    artifact_paths, diagnostics_summary = _build_he_artifacts!(
        artifact_dir,
        he_trajectory_buffer,
        he_diagnostics_buffer,
        he_episode_summaries,
    )

    handoff_frontier = frontier_quality_summary(frontier_buffer; topk=10)
    handoff_replay = _replay_topk_summary(replay_buffer; topk=10)
    handoff_cache = _cache_topk_summary(oracle_mgr.cache, task; topk=10)

    return Dict{String,Any}(
        "task" => task,
        "arm" => arm,
        "run_index" => run_idx,
        "oracle_cache" => deepcopy(oracle_mgr.cache),
        "oracle_cache_flat" => copy(phase.reward_cache),
        "tracker" => Dict(
            "calls_used" => tracker.calls_used,
            "auc" => _tracker_auc(tracker; min_call=1),
            "top1" => _tracker_top1(tracker),
            "top10_mean" => _tracker_top10(tracker),
            "checkpoints" => tracker.checkpoints,
        ),
        "replay_buffer" => _clone_replay_buffer(replay_buffer),
        "frontier_buffer" => GFlowNet._clone_frontier_buffer(frontier_buffer),
        "handoff_frontier" => handoff_frontier,
        "handoff_replay" => handoff_replay,
        "handoff_cache" => handoff_cache,
        "handoff_replay_smiles" => Set{String}(String(s) for s in handoff_replay["top_smiles"]),
        "handoff_frontier_smiles" => Set{String}(String(s) for s in handoff_frontier["top_smiles"]),
        "seed_calls" => seed_calls,
        "bootstrap_calls" => bootstrap_calls,
        "he_calls" => total_he_calls,
        "artifact_paths" => artifact_paths,
        "diagnostics_summary" => diagnostics_summary,
    )
end

# ============================================================================
# Phase 2: Downstream pure TB
# ============================================================================

function _build_tb_config(remaining_budget::Int)
    n_iters = max(1, cld(max(remaining_budget, 1), PMO_BATCH_SIZE))
    return FinetuningConfig(;
        n_iterations=n_iters,
        sample_batch_size=PMO_BATCH_SIZE,
        learning_rate=3e-5,
        gradient_clip_norm=1.0,
        kl_weight=0.01,
        kl_decay_schedule=:none,
        loss_type=:shifted_cosh,
        cosh_threshold=2.0,
        max_length=150,
        temperature=1.0,
        epsilon=0.05,
        log_frequency=5,
        reward_exponent=8.0,
        min_reward=0.01,
        training_mode=:tb,
        constructive_only=true,
        freeze_gru=true,
        reward_weighted=true,
        unfreeze_top_gru=true,
        use_replay=true,
        replay_ratio=PMO_REPLAY_RATIO,
        use_qgfn_sampling=false,
        lr_z_multiplier=10.0,
        lr_z=0.0,
        log_z_grad_clip=1.0,
        warmup_iters=0,
    )
end

function _run_downstream_tb(task::String, arm::String, run_idx::Int, shaping_state::Union{Nothing,Dict{String,Any}})
    target_smi = get(TARGET_SMILES, task, nothing)
    is_baseline = arm == "tb_only"
    budget = is_baseline ? TOTAL_BUDGET : DOWNSTREAM_BUDGET
    phase = _make_oracle_phase(task, budget;
        existing_cache=is_baseline ? Dict{String,Dict{String,Float64}}() : shaping_state["oracle_cache"],
        tracker=PhaseMetricTracker(stride=METRIC_STRIDE))
    oracle_mgr = phase.mgr
    reward_fn = phase.reward_fn
    reward_fn_batch = phase.reward_fn_batch
    tracker = phase.tracker

    replay_buffer = is_baseline ? SMILESReplayBuffer(5000) : _clone_replay_buffer(shaping_state["replay_buffer"])
    frontier_buffer = is_baseline ? MolecularFrontierBuffer(5000) : GFlowNet._clone_frontier_buffer(shaping_state["frontier_buffer"])

    if is_baseline && !isnothing(target_smi)
        _seed_memories!([target_smi], reward_fn, VOCAB;
            reward_fn_batch=reward_fn_batch,
            replay_buffer=replay_buffer,
            frontier_buffer=frontier_buffer,
            augmentation_count=0,
            verbose=false)
    end

    current_params = deepcopy(PRETRAINED_PARAMS)
    ref_params = deepcopy(PRETRAINED_PARAMS)
    ref_states = deepcopy(PRETRAINED_STATES)
    current_log_Z = 0.0
    segment = 0
    max_segments = 4

    while !budget_exhausted(oracle_mgr) && segment < max_segments
        segment += 1
        cfg = _build_tb_config(budget_remaining(oracle_mgr))
        logmsg("[$arm][$task][run$(run_idx)] downstream segment $segment budget=$(oracle_mgr.calls_used)/$budget")
        result = finetune_smiles_gflownet(
            POLICY_MODEL,
            VOCAB,
            current_params,
            PRETRAINED_STATES,
            ref_params,
            ref_states,
            reward_fn,
            cfg;
            reward_fn_batch=reward_fn_batch,
            replay_buffer=replay_buffer,
            budget_used=oracle_mgr.calls_used,
            total_budget=budget,
            log_Z_init=current_log_Z,
            verbose=true,
        )
        current_params = result.params
        current_log_Z = result.log_Z

        if !isempty(replay_buffer)
            for mol in get_top_molecules(replay_buffer, min(128, length(replay_buffer)))
                add_to_frontier!(frontier_buffer, mol.smiles;
                    reward=mol.reward,
                    source=:model,
                    operator=:sample)
            end
        end
    end

    final_frontier = frontier_quality_summary(frontier_buffer; topk=10)
    final_replay = _replay_topk_summary(replay_buffer; topk=10)
    final_cache = _cache_topk_summary(oracle_mgr.cache, task; topk=10)

    handoff_replay = isnothing(shaping_state) ? Dict{String,Any}("top_smiles" => Set{String}(), "top10_mean" => 0.0, "top1" => 0.0, "size" => 0) : shaping_state["handoff_replay"]
    handoff_frontier = isnothing(shaping_state) ? Dict{String,Any}("top_smiles" => Set{String}(), "top10_mean" => 0.0, "top1" => 0.0, "size" => 0) : shaping_state["handoff_frontier"]
    handoff_replay_smiles = isnothing(shaping_state) ? Set{String}() : shaping_state["handoff_replay_smiles"]
    handoff_frontier_smiles = isnothing(shaping_state) ? Set{String}() : shaping_state["handoff_frontier_smiles"]

    baseline_min_call = is_baseline ? (SHAPING_BUDGET + 1) : 1

    return Dict{String,Any}(
        "task" => task,
        "arm" => arm,
        "run_index" => run_idx,
        "downstream_tracker" => Dict(
            "calls_used" => tracker.calls_used,
            "downstream_auc_top10" => _tracker_auc(tracker; min_call=baseline_min_call),
            "full_auc_top10" => _tracker_auc(tracker; min_call=1),
            "phase_top1" => _tracker_top1(tracker),
            "phase_top10_mean" => _tracker_top10(tracker),
            "checkpoints" => tracker.checkpoints,
        ),
        "final_frontier" => final_frontier,
        "final_replay" => final_replay,
        "final_cache" => final_cache,
        "handoff_frontier" => handoff_frontier,
        "handoff_replay" => handoff_replay,
        "replay_top10_overlap_with_handoff" => _set_overlap_fraction(final_replay["top_smiles"], handoff_replay_smiles),
        "frontier_top10_overlap_with_handoff" => _set_overlap_fraction(final_frontier["top_smiles"], handoff_frontier_smiles),
        "replay_top10_gain_from_handoff" => Float64(final_replay["top10_mean"]) - Float64(get(handoff_replay, "top10_mean", 0.0)),
        "frontier_top10_gain_from_handoff" => Float64(final_frontier["top10_mean"]) - Float64(get(handoff_frontier, "top10_mean", 0.0)),
        "final_oracle_calls" => oracle_mgr.calls_used,
    )
end

# ============================================================================
# Arm Execution
# ============================================================================

function _run_arm(arm::String, task::String, run_idx::Int)
    if arm == "tb_only"
        downstream = _run_downstream_tb(task, arm, run_idx, nothing)
        return Dict{String,Any}(
            "shaping" => nothing,
            "downstream" => downstream,
        )
    end

    shaping = _run_shaping_phase(task, arm, run_idx)
    downstream = _run_downstream_tb(task, arm, run_idx, shaping)
    return Dict{String,Any}(
        "shaping" => shaping,
        "downstream" => downstream,
    )
end

# ============================================================================
# Execute Experiment
# ============================================================================

logmsg("============================================================")
logmsg("Level 3 Shape-then-TB Test")
logmsg("Total budget=$TOTAL_BUDGET | Shaping=$SHAPING_BUDGET | Downstream=$DOWNSTREAM_BUDGET")
logmsg("Repeats=$N_REPEATS | Tasks=$(join(TASKS, ", "))")
logmsg("Arms=$(join(ARM_NAMES, ", "))")
logmsg("============================================================")

all_results = Dict{String, Dict{String, Vector{Dict{String,Any}}}}()
for arm in ARM_NAMES
    task_results = Dict{String, Vector{Dict{String,Any}}}()
    logmsg("--- Arm: $arm ---")
    for task in TASKS
        runs = Dict{String,Any}[]
        for run_idx in 1:N_REPEATS
            push!(runs, _run_arm(arm, task, run_idx))
        end
        task_results[task] = runs
    end
    all_results[arm] = task_results
end

# ============================================================================
# Summaries / Verdict
# ============================================================================

summary_rows = Dict{String,Any}[]
for arm in ARM_NAMES
    for task in TASKS
        runs = all_results[arm][task]
        downstreams = [r["downstream"] for r in runs]
        shapings = [r["shaping"] for r in runs if !isnothing(r["shaping"])]
        push!(summary_rows, Dict{String,Any}(
            "arm" => arm,
            "task" => task,
            "downstream_auc_mean" => _safe_mean(r["downstream_tracker"]["downstream_auc_top10"] for r in downstreams),
            "downstream_auc_std" => _safe_std(r["downstream_tracker"]["downstream_auc_top10"] for r in downstreams),
            "final_top10_mean" => _safe_mean(r["final_cache"]["top10_mean"] for r in downstreams),
            "final_top1_mean" => _safe_mean(r["final_cache"]["top1"] for r in downstreams),
            "handoff_frontier_top10" => isempty(shapings) ? 0.0 : _safe_mean(s["handoff_frontier"]["top10_mean"] for s in shapings),
            "handoff_replay_top10" => isempty(shapings) ? 0.0 : _safe_mean(s["handoff_replay"]["top10_mean"] for s in shapings),
            "frontier_gain_mean" => _safe_mean(r["frontier_top10_gain_from_handoff"] for r in downstreams),
            "replay_gain_mean" => _safe_mean(r["replay_top10_gain_from_handoff"] for r in downstreams),
            "frontier_overlap_mean" => _safe_mean(r["frontier_top10_overlap_with_handoff"] for r in downstreams),
            "replay_overlap_mean" => _safe_mean(r["replay_top10_overlap_with_handoff"] for r in downstreams),
            "shaping_calls_mean" => isempty(shapings) ? 0.0 : _safe_mean(s["seed_calls"] + s["bootstrap_calls"] + s["he_calls"] for s in shapings),
        ))
    end
end

by_arm_task = Dict((row["arm"], row["task"]) => row for row in summary_rows)
deltas_vs_tb = Dict{String,Dict{String,Dict{String,Float64}}}()
for arm in ("heuristic_shape_then_tb", "learned_shape_then_tb")
    task_delta = Dict{String,Dict{String,Float64}}()
    for task in TASKS
        base = by_arm_task[("tb_only", task)]
        row = by_arm_task[(arm, task)]
        task_delta[task] = Dict(
            "downstream_auc_delta" => row["downstream_auc_mean"] - base["downstream_auc_mean"],
            "final_top10_delta" => row["final_top10_mean"] - base["final_top10_mean"],
            "final_top1_delta" => row["final_top1_mean"] - base["final_top1_mean"],
        )
    end
    deltas_vs_tb[arm] = task_delta
end

learned_vs_heuristic = Dict{String,Dict{String,Float64}}()
for task in TASKS
    h = by_arm_task[("heuristic_shape_then_tb", task)]
    l = by_arm_task[("learned_shape_then_tb", task)]
    learned_vs_heuristic[task] = Dict(
        "downstream_auc_delta" => l["downstream_auc_mean"] - h["downstream_auc_mean"],
        "final_top10_delta" => l["final_top10_mean"] - h["final_top10_mean"],
        "final_top1_delta" => l["final_top1_mean"] - h["final_top1_mean"],
        "frontier_overlap_delta" => l["frontier_overlap_mean"] - h["frontier_overlap_mean"],
        "replay_overlap_delta" => l["replay_overlap_mean"] - h["replay_overlap_mean"],
    )
end

mean_learned_auc_vs_tb = _safe_mean(v["downstream_auc_delta"] for v in values(deltas_vs_tb["learned_shape_then_tb"]))
mean_heuristic_auc_vs_tb = _safe_mean(v["downstream_auc_delta"] for v in values(deltas_vs_tb["heuristic_shape_then_tb"]))
mean_learned_vs_heuristic_auc = _safe_mean(v["downstream_auc_delta"] for v in values(learned_vs_heuristic))
mean_learned_frontier_overlap = _safe_mean(row["frontier_overlap_mean"] for row in summary_rows if row["arm"] == "learned_shape_then_tb")
mean_learned_replay_overlap = _safe_mean(row["replay_overlap_mean"] for row in summary_rows if row["arm"] == "learned_shape_then_tb")
mean_learned_frontier_gain = _safe_mean(row["frontier_gain_mean"] for row in summary_rows if row["arm"] == "learned_shape_then_tb")
catastrophic_stress = if ("celecoxib_rediscovery" in TASKS)
    by_arm_task[("learned_shape_then_tb", "celecoxib_rediscovery")]["final_top10_mean"] + 1e-12 <
        by_arm_task[("tb_only", "celecoxib_rediscovery")]["final_top10_mean"] - 0.10
else
    false
end
mediated_credit_alive = (mean_learned_frontier_overlap > 0.0) || (mean_learned_replay_overlap > 0.0)

verdict = if mean_learned_auc_vs_tb > 0.0 && mean_learned_vs_heuristic_auc >= 0.0 && !catastrophic_stress && mediated_credit_alive
    "WORKING_ENOUGH_TO_SCALE"
elseif mean_learned_auc_vs_tb > 0.0 && !catastrophic_stress
    "PROMISING_BUT_INCONCLUSIVE"
else
    "CURRENTLY_FALSIFIED"
end

logmsg("============================================================")
logmsg("LEVEL 3 SHAPE-THEN-TB SUMMARY")
for row in summary_rows
    logmsg("$(rpad(row["arm"], 28)) $(rpad(row["task"], 24)) downstream_auc=$(round(row["downstream_auc_mean"], digits=4))±$(round(row["downstream_auc_std"], digits=4)) final_top10=$(round(row["final_top10_mean"], digits=4)) frontier_gain=$(round(row["frontier_gain_mean"], digits=4)) overlap=$(round(row["frontier_overlap_mean"], digits=3))/$(round(row["replay_overlap_mean"], digits=3))")
end
logmsg("--- Deltas vs TB-only ---")
for arm in ("heuristic_shape_then_tb", "learned_shape_then_tb")
    logmsg("$arm")
    for task in TASKS
        δ = deltas_vs_tb[arm][task]
        logmsg("  $task: ΔDownstreamAUC=$(round(δ["downstream_auc_delta"], digits=4)) ΔFinalTop10=$(round(δ["final_top10_delta"], digits=4)) ΔFinalTop1=$(round(δ["final_top1_delta"], digits=4))")
    end
end
logmsg("--- Learned vs Heuristic ---")
for task in TASKS
    δ = learned_vs_heuristic[task]
    logmsg("  $task: ΔDownstreamAUC=$(round(δ["downstream_auc_delta"], digits=4)) ΔFinalTop10=$(round(δ["final_top10_delta"], digits=4)) ΔFrontierOverlap=$(round(δ["frontier_overlap_delta"], digits=4))")
end
logmsg("--- Verdict ---")
logmsg("VERDICT=$verdict")
logmsg("mean_learned_auc_vs_tb=$(round(mean_learned_auc_vs_tb, digits=4))")
logmsg("mean_heuristic_auc_vs_tb=$(round(mean_heuristic_auc_vs_tb, digits=4))")
logmsg("mean_learned_vs_heuristic_auc=$(round(mean_learned_vs_heuristic_auc, digits=4))")
logmsg("mean_learned_frontier_overlap=$(round(mean_learned_frontier_overlap, digits=4)) mean_learned_replay_overlap=$(round(mean_learned_replay_overlap, digits=4))")
logmsg("mean_learned_frontier_gain=$(round(mean_learned_frontier_gain, digits=4)) catastrophic_stress=$(catastrophic_stress)")

results_file = joinpath(LOGDIR, "level3_shape_then_tb_results.jls")
serialize(results_file, Dict(
    "timestamp" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
    "total_budget" => TOTAL_BUDGET,
    "shaping_budget" => SHAPING_BUDGET,
    "downstream_budget" => DOWNSTREAM_BUDGET,
    "repeats" => N_REPEATS,
    "tasks" => TASKS,
    "arms" => ARM_NAMES,
    "artifact_root" => ARTIFACT_ROOT,
    "pretrain_checkpoint" => CHECKPOINT_PATH,
    "policy_checkpoint" => policy_ckpt,
    "train_config" => train_config,
    "dataset_stats" => dataset.stats,
    "training_history" => history,
    "summary_rows" => summary_rows,
    "deltas_vs_tb" => deltas_vs_tb,
    "learned_vs_heuristic" => learned_vs_heuristic,
    "mean_learned_auc_vs_tb" => mean_learned_auc_vs_tb,
    "mean_heuristic_auc_vs_tb" => mean_heuristic_auc_vs_tb,
    "mean_learned_vs_heuristic_auc" => mean_learned_vs_heuristic_auc,
    "mean_learned_frontier_overlap" => mean_learned_frontier_overlap,
    "mean_learned_replay_overlap" => mean_learned_replay_overlap,
    "mean_learned_frontier_gain" => mean_learned_frontier_gain,
    "verdict" => verdict,
    "all_results" => all_results,
))
logmsg("Saved results: $results_file")
logmsg("Done.")
