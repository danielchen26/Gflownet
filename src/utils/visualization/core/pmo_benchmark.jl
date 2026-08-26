# PMO Benchmark Runner — Standardized 23-Task Molecular Optimization
#
# Implements the exact PMO evaluation protocol from:
#   "Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization"
#   Gao et al., NeurIPS 2022
#
# Protocol:
# - Budget: 10,000 oracle calls per task
# - Metric: AUC of top-10 average score (sampled every 100 calls)
# - 5 independent runs → mean ± std
# - Total score: sum of AUC top-10 across all 23 tasks

using Statistics: mean, std
using Serialization

const PMO_23_TASKS = OracleBridge.PMO_23_TASKS

# Known SOTA scores (AUC top-10, sum across all 23 tasks)
const SOTA_SCORES = Dict(
    "Genetic GFN"   => 16.2,
    "REINVENT"      => 15.2,
    "Mol GA"        => 15.7,
    "Graph GA"      => 14.8,
    "SMILES LSTM"   => 12.8,
    "Random"        => 5.1,
)

"""
    PMOResult

Result of running a single PMO benchmark task.
"""
struct PMOResult
    task_name::String
    auc_top10::Float64       # AUC of top-10 avg score (sampled every 100 calls)
    top1::Float64            # Best molecule score
    top10_mean::Float64      # Mean of top 10 molecules
    diversity::Float64       # Tanimoto diversity of top 100
    n_oracle_calls::Int      # Budget consumed
    unique_molecules::Int    # Total unique SMILES generated
    oracle_call_breakdown::Dict{String,Int}
    provenance_summary::Dict{String,Any}
    artifact_paths::Dict{String,String}
    diagnostics_summary::Dict{String,Any}
end

PMOResult(task_name::String,
          auc_top10::Float64,
          top1::Float64,
          top10_mean::Float64,
          diversity::Float64,
          n_oracle_calls::Int,
          unique_molecules::Int) = PMOResult(
    task_name,
    auc_top10,
    top1,
    top10_mean,
    diversity,
    n_oracle_calls,
    unique_molecules,
    Dict{String,Int}(),
    Dict{String,Any}(),
    Dict{String,String}(),
    Dict{String,Any}(),
)

PMOResult(task_name::String,
          auc_top10::Float64,
          top1::Float64,
          top10_mean::Float64,
          diversity::Float64,
          n_oracle_calls::Int,
          unique_molecules::Int,
          oracle_call_breakdown::AbstractDict{String,<:Integer},
          provenance_summary::AbstractDict{String,<:Any}) = PMOResult(
    task_name,
    auc_top10,
    top1,
    top10_mean,
    diversity,
    n_oracle_calls,
    unique_molecules,
    Dict{String,Int}(String(k) => Int(v) for (k, v) in pairs(oracle_call_breakdown)),
    Dict{String,Any}(String(k) => v for (k, v) in pairs(provenance_summary)),
    Dict{String,String}(),
    Dict{String,Any}(),
)

"""
    PMOBenchmarkReport

Report from running the full PMO benchmark (or subset).
"""
struct PMOBenchmarkReport
    task_results::Vector{PMOResult}
    total_score::Float64     # Sum of AUC top-10 across all tasks
    n_runs::Int              # Number of independent runs per task
    budget_per_task::Int     # Oracle budget per task
end

function _empty_pmo_oracle_call_breakdown()::Dict{String,Int}
    return Dict(
        "seed" => 0,
        "frontier_bootstrap" => 0,
        "model" => 0,
        "ga" => 0,
        "he_warmup" => 0,
        "he_interleaved" => 0,
    )
end

function _accumulate_oracle_calls!(breakdown::Dict{String,Int}, key::String, delta::Integer)
    breakdown[key] = get(breakdown, key, 0) + max(0, Int(delta))
    return nothing
end

function _collapse_pmo_source_key(source_key::AbstractString)::String
    if source_key == "model"
        return "tb"
    elseif source_key in ("ga", "mutation", "crossover")
        return "ga"
    elseif source_key in ("edit", "warmup")
        return "he"
    elseif source_key == "seed"
        return "seed"
    elseif source_key == "bootstrap"
        return "bootstrap"
    end
    return String(source_key)
end

function _collapsed_pmo_source_counts(counts)::Dict{String,Int}
    collapsed = Dict{String,Int}(
        "tb" => 0,
        "ga" => 0,
        "he" => 0,
        "seed" => 0,
        "bootstrap" => 0,
    )
    for (key, value) in pairs(counts)
        bucket = _collapse_pmo_source_key(String(key))
        collapsed[bucket] = get(collapsed, bucket, 0) + Int(value)
    end
    return collapsed
end

function _pmo_source_fractions(counts::Dict{String,Int})::Dict{String,Float64}
    total = sum(values(counts))
    fractions = Dict{String,Float64}()
    for key in ("tb", "ga", "he", "seed", "bootstrap")
        fractions[key] = total > 0 ? get(counts, key, 0) / total : 0.0
    end
    return fractions
end

function _pmo_provenance_summary(frontier_buffer)::Dict{String,Any}
    if isnothing(frontier_buffer) || isempty(frontier_buffer)
        return Dict(
            "frontier_size" => 0,
            "graph_unique_count" => 0,
            "overall_source_counts" => Dict{String,Int}("tb" => 0, "ga" => 0, "he" => 0, "seed" => 0, "bootstrap" => 0),
            "overall_source_fractions" => Dict{String,Float64}("tb" => 0.0, "ga" => 0.0, "he" => 0.0, "seed" => 0.0, "bootstrap" => 0.0),
            "topk_source_counts" => Dict{String,Int}("tb" => 0, "ga" => 0, "he" => 0, "seed" => 0, "bootstrap" => 0),
            "topk_source_fractions" => Dict{String,Float64}("tb" => 0.0, "ga" => 0.0, "he" => 0.0, "seed" => 0.0, "bootstrap" => 0.0),
            "overall_operators" => Dict{String,Int}(),
            "topk_operators" => Dict{String,Int}(),
            "topk_edit_operators" => Dict{String,Int}(),
            "top1_source" => "none",
            "top1_operator" => "none",
            "frontier_stats" => Dict{String,Any}(),
        )
    end

    stats = frontier_stats(frontier_buffer)
    source_summary = frontier_source_summary(frontier_buffer; topk=10)
    overall_counts = _collapsed_pmo_source_counts(get(source_summary, "overall", Dict{String,Int}()))
    topk_counts = _collapsed_pmo_source_counts(get(source_summary, "topk", Dict{String,Int}()))

    return Dict(
        "frontier_size" => Int(get(stats, "size", 0)),
        "graph_unique_count" => Int(get(stats, "graph_unique_count", 0)),
        "overall_source_counts" => overall_counts,
        "overall_source_fractions" => _pmo_source_fractions(overall_counts),
        "topk_source_counts" => topk_counts,
        "topk_source_fractions" => _pmo_source_fractions(topk_counts),
        "overall_operators" => get(source_summary, "overall_operators", Dict{String,Int}()),
        "topk_operators" => get(source_summary, "topk_operators", Dict{String,Int}()),
        "topk_edit_operators" => get(source_summary, "topk_edit_operators", Dict{String,Int}()),
        "top1_source" => get(source_summary, "top1_source", "none"),
        "top1_operator" => get(source_summary, "top1_operator", "none"),
        "frontier_stats" => stats,
    )
end

function _assert_heuristic_he_config(config::HierarchicalEditConfig)
    config.use_learned_basin && error("Stage B′ requires heuristic-only HE: use_learned_basin must be false")
    config.use_learned_parent && error("Stage B′ requires heuristic-only HE: use_learned_parent must be false")
    config.use_learned_operator && error("Stage B′ requires heuristic-only HE: use_learned_operator must be false")
    config.allow_fragment_ops && error("Stage B′ requires trusted operators only: allow_fragment_ops must be false")
    if !isnothing(config.operators)
        trusted = Set((:mutate, :terminate, :crossover))
        invalid = Symbol[op for op in config.operators if !(op in trusted)]
        isempty(invalid) || error("Stage B′ requires trusted operators only; invalid operators: $(invalid)")
    end
    return nothing
end

"""
    run_pmo_task(task_name; budget=10000, hidden_dim=256, lr=0.001) → PMOResult

Run a single PMO benchmark task using GFlowNet.

The GFlowNet is trained in benchmark_mode (oracle-only reward) until the
oracle budget is exhausted. Budget-driven, NOT episode-driven.
"""
function run_pmo_task(task_name::String;
                      budget::Int=10000,
                      hidden_dim::Int=256,
                      learning_rate::Float64=0.001,
                      batch_size::Int=32)::PMOResult
    @info "PMO task: $task_name (budget=$budget)"

    # Create oracle manager in benchmark_mode
    oracle_mgr = OracleManager(
        [OracleConfig(task_name, 1.0)],
        budget,
        0,
        Dict{String,Dict{String,Float64}}(),
        true  # benchmark_mode
    )

    # Initialize oracle
    OracleBridge.init_oracles!([task_name];
        cache_dir=joinpath(@__DIR__, "..", "..", "..", "..", "data", "tdc_cache"))

    # Create single-objective GFlowNet (no MOGFN needed for single task)
    model = create_molecular_gflownet(
        hidden_dim = hidden_dim,
        learning_rate = learning_rate,
        partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,
    )

    # Training loop — budget-driven
    all_smiles = Set{String}()
    sampling_config = GFlowNet.SamplingConfig(
        strategy = GFlowNet.STOCHASTIC_SAMPLING,
        temperature = 1.0,
        epsilon = 0.1,
        max_trajectory_length = 100
    )

    training_config = GFlowNet.TrainingConfig(
        objective = GFlowNet.TRAJECTORY_BALANCE,
        n_iterations = budget,  # Will stop by budget, not iterations
        batch_size = batch_size,
        learning_rate = learning_rate,
        temperature = 1.0,
        epsilon = 0.1,
        epsilon_decay = true,
        entropy_weight = 0.02,
        z_learning_rate_multiplier = 10.0,
        verbose = false
    )

    iteration = 0
    while !budget_exhausted(oracle_mgr)
        iteration += 1

        # Sample trajectories
        trajectories = [GFlowNet.sample_trajectory(model; config=sampling_config)
                        for _ in 1:batch_size]

        # Collect terminal SMILES
        terminal_smiles = String[]
        for traj in trajectories
            terminal = traj.states[end]
            if GFlowNet.is_terminal_state(terminal) && hasproperty(terminal, :smiles) && !isempty(terminal.smiles)
                push!(terminal_smiles, terminal.smiles)
                push!(all_smiles, terminal.smiles)
            end
        end

        # Batch pre-compute oracle scores
        if !isempty(terminal_smiles)
            evaluate_molecules!(oracle_mgr, unique(terminal_smiles))
        end

        # Check budget after evaluation
        budget_exhausted(oracle_mgr) && break

        # Train step
        try
            GFlowNet.train_step!(model, trajectories, training_config)
        catch
            continue  # Skip failed steps
        end

        # Progress logging every 100 iterations
        if iteration % 100 == 0
            @info "  PMO [$task_name] iter=$iteration budget_used=$(oracle_mgr.calls_used)/$(budget)"
        end
    end

    # Compute final metrics
    top_scores = sort(oracle_mgr.top_scores, rev=true)
    top1 = isempty(top_scores) ? 0.0 : top_scores[1]
    top10_mean = isempty(top_scores) ? 0.0 : mean(top_scores[1:min(10, length(top_scores))])

    # Compute diversity of top-100 molecules
    diversity = 0.0
    try
        sorted_cache = sort(collect(oracle_mgr.cache), by=x -> -get(x[2], task_name, 0.0))
        top100_smiles = [kv[1] for kv in sorted_cache[1:min(100, length(sorted_cache))]]
        if length(top100_smiles) >= 2
            fps = [RDKitBridge.compute_fingerprint(s) for s in top100_smiles]
            valid_fps = filter(fp -> sum(fp) > 0, fps)
            if length(valid_fps) >= 2
                div_stats = RDKitBridge.compute_diversity_stats(valid_fps)
                diversity = div_stats["internal_diversity_1"]
            end
        end
    catch
        # Diversity is non-critical
    end

    auc = compute_auc_top10(oracle_mgr)

    result = PMOResult(
        task_name,
        auc,
        top1,
        top10_mean,
        diversity,
        oracle_mgr.calls_used,
        length(all_smiles),
    )

    @info "PMO [$task_name] complete" auc_top10=auc top1=top1 top10=top10_mean diversity=diversity calls=oracle_mgr.calls_used

    return result
end

"""
    run_pmo_benchmark(; tasks=PMO_23_TASKS, n_runs=5, budget=10000) → PMOBenchmarkReport

Run the full PMO benchmark (or subset).
Each task is run n_runs times, reporting mean ± std.
"""
function run_pmo_benchmark(;
    tasks::Vector{String}=PMO_23_TASKS,
    n_runs::Int=5,
    budget::Int=10000,
    hidden_dim::Int=256,
    learning_rate::Float64=0.001,
)::PMOBenchmarkReport
    @info "Starting PMO benchmark" n_tasks=length(tasks) n_runs=n_runs budget=budget

    all_results = PMOResult[]

    for task in tasks
        task_aucs = Float64[]

        for run in 1:n_runs
            @info "  Run $run/$n_runs for $task"
            result = run_pmo_task(task;
                budget=budget, hidden_dim=hidden_dim, learning_rate=learning_rate)
            push!(task_aucs, result.auc_top10)

            # Store last run result for detailed metrics
            if run == n_runs
                push!(all_results, result)
            end
        end

        mean_auc = mean(task_aucs)
        std_auc = n_runs > 1 ? std(task_aucs) : 0.0
        @info "  $task: AUC top-10 = $(round(mean_auc, digits=4)) ± $(round(std_auc, digits=4))"
    end

    total_score = sum(r.auc_top10 for r in all_results)

    report = PMOBenchmarkReport(all_results, total_score, n_runs, budget)

    @info "PMO Benchmark Complete" total_score=total_score
    for (name, sota) in sort(collect(SOTA_SCORES), by=x->-x[2])
        @info "  $name: $sota $(total_score > sota ? "✓ (beat)" : "")"
    end

    return report
end

"""
    benchmark_results_to_dict(report::PMOBenchmarkReport) → Dict

Convert benchmark report to a Dict for JSON serialization.
"""
function benchmark_results_to_dict(report::PMOBenchmarkReport)::Dict
    tasks = [Dict(
        "task_name"             => r.task_name,
        "auc_top10"             => r.auc_top10,
        "top1"                  => r.top1,
        "top10_mean"            => r.top10_mean,
        "diversity"             => r.diversity,
        "n_oracle_calls"        => r.n_oracle_calls,
        "unique_molecules"      => r.unique_molecules,
        "oracle_call_breakdown" => r.oracle_call_breakdown,
        "provenance_summary"    => r.provenance_summary,
        "artifact_paths"        => r.artifact_paths,
        "diagnostics_summary"   => r.diagnostics_summary,
    ) for r in report.task_results]

    return Dict(
        "tasks"          => tasks,
        "total_score"    => report.total_score,
        "n_runs"         => report.n_runs,
        "budget_per_task" => report.budget_per_task,
        "sota_comparison" => SOTA_SCORES,
    )
end

function _artifact_safe_component(s::AbstractString)::String
    safe = replace(String(s), r"[^A-Za-z0-9._-]+" => "-")
    return isempty(safe) ? "artifact" : safe
end

function _trajectory_entries_for_episode(buf::Union{Nothing,EditTrajectoryBuffer}, episode_id::String)
    isnothing(buf) && return EditTrajectoryEntry[]
    return [entry for entry in buf.entries if get(entry.metadata, "episode_id", "") == episode_id]
end

function _decision_logs_for_episode(buf::Union{Nothing,HierarchicalEditDiagnosticsBuffer}, episode_id::String)
    isnothing(buf) && return HierarchicalEditDecisionLog[]
    return [log for log in buf.logs if log.episode_id == episode_id]
end

function _proposal_logs_for_episode(buf::Union{Nothing,HierarchicalEditDiagnosticsBuffer}, episode_id::String)
    isnothing(buf) && return HierarchicalEditProposalLog[]
    return [log for log in buf.proposal_logs if log.episode_id == episode_id]
end

function _basin_logs_for_episode(buf::Union{Nothing,HierarchicalEditDiagnosticsBuffer}, episode_id::String)
    isnothing(buf) && return BasinDecisionLog[]
    return [log for log in buf.basin_logs if log.episode_id == episode_id]
end

function _parent_logs_for_episode(buf::Union{Nothing,HierarchicalEditDiagnosticsBuffer}, episode_id::String)
    isnothing(buf) && return ParentDecisionLog[]
    return [log for log in buf.parent_logs if log.episode_id == episode_id]
end

function _episode_stop_reason(ep::HierarchicalEditEpisode,
                              proposal_logs::Vector{HierarchicalEditProposalLog},
                              decision_logs::Vector{HierarchicalEditDecisionLog},
                              horizon::Int)::String
    if ep.frontier_size_before == 0
        return "empty_frontier"
    elseif any(log -> log.terminated && log.commit_applied, decision_logs)
        return "terminated_commit"
    elseif ep.commits_applied == 0 && !isempty(proposal_logs) && all(log -> log.empty_after_filter, proposal_logs)
        return "no_valid_proposal"
    elseif ep.commits_applied == 0 && !isempty(proposal_logs)
        return "stagnation_or_low_value"
    elseif length(ep.steps) >= horizon
        return "horizon_exhausted"
    elseif ep.commits_applied > 0
        return "committed_without_terminate"
    end
    return "no_progress"
end

function _compact_frontier_summary(summary::Dict{String,Any})::Dict{String,Any}
    return Dict(
        "size" => Int(get(summary, "size", 0)),
        "top1" => Float64(get(summary, "top1", 0.0)),
        "top10_mean" => Float64(get(summary, "top10_mean", 0.0)),
        "graph_unique_count" => Int(get(summary, "graph_unique_count", 0)),
        "n_scaffolds" => Int(get(summary, "n_scaffolds", 0)),
    )
end

function _string_key_any_dict(x)::Dict{String,Any}
    if x isa Dict
        return Dict{String,Any}(String(k) => v for (k, v) in pairs(x))
    elseif x isa NamedTuple
        return Dict{String,Any}(String(k) => v for (k, v) in pairs(x))
    elseif isnothing(x)
        return Dict{String,Any}()
    else
        return Dict{String,Any}("value" => x)
    end
end

function _resolve_he_episode_selection(he_episode_selector,
                                       frontier_buffer::MolecularFrontierBuffer,
                                       task_name::String,
                                       budget_remaining::Int,
                                       created_at_step::Int,
                                       phase::String,
                                       episode_index::Int,
                                       base_config::HierarchicalEditConfig)::Tuple{HierarchicalEditConfig,Union{Nothing,Symbol},Dict{String,Any}}
    if isnothing(he_episode_selector)
        return base_config, nothing, Dict{String,Any}(
            "schema_version" => "o3_selector_metadata_v1",
            "selector_active" => false,
            "policy" => "none",
            "selected_schema" => "heuristic_default_h8",
            "phase" => phase,
            "episode_index" => episode_index,
            "created_at_step" => created_at_step,
            "budget_remaining" => budget_remaining,
        )
    end

    selection = he_episode_selector(frontier_buffer, task_name, budget_remaining, created_at_step, phase, episode_index, base_config)
    selected_config = base_config
    operator_override::Union{Nothing,Symbol} = nothing
    metadata = Dict{String,Any}()

    if selection isa Tuple
        length(selection) >= 1 && (selected_config = selection[1])
        length(selection) >= 2 && (operator_override = selection[2])
        length(selection) >= 3 && (metadata = _string_key_any_dict(selection[3]))
    elseif selection isa NamedTuple
        selected_config = get(selection, :config, get(selection, :selected_config, base_config))
        operator_override = get(selection, :operator_override, nothing)
        metadata = _string_key_any_dict(get(selection, :metadata, Dict{String,Any}()))
    elseif selection isa Dict
        selected_config = get(selection, "config", get(selection, "selected_config", base_config))
        operator_override = get(selection, "operator_override", nothing)
        metadata = _string_key_any_dict(get(selection, "metadata", Dict{String,Any}()))
    else
        error("he_episode_selector must return Tuple, NamedTuple, or Dict; got $(typeof(selection))")
    end

    selected_config isa HierarchicalEditConfig || error("he_episode_selector returned non-HierarchicalEditConfig config: $(typeof(selected_config))")
    (isnothing(operator_override) || operator_override isa Symbol) || error("he_episode_selector operator_override must be nothing or Symbol; got $(typeof(operator_override))")

    metadata["schema_version"] = get(metadata, "schema_version", "o3_selector_metadata_v1")
    metadata["selector_active"] = true
    metadata["phase"] = phase
    metadata["episode_index"] = episode_index
    metadata["created_at_step"] = created_at_step
    metadata["budget_remaining"] = budget_remaining
    metadata["selected_horizon"] = selected_config.horizon
    metadata["selected_topk_tracking"] = selected_config.topk_tracking
    metadata["operator_override"] = isnothing(operator_override) ? "mixed" : string(operator_override)
    return selected_config, operator_override, metadata
end

function _episode_summary(ep::HierarchicalEditEpisode,
                          trajectory_entries::Vector{EditTrajectoryEntry},
                          proposal_logs::Vector{HierarchicalEditProposalLog},
                          decision_logs::Vector{HierarchicalEditDecisionLog},
                          basin_logs::Vector{BasinDecisionLog},
                          parent_logs::Vector{ParentDecisionLog};
                          phase::String,
                          segment_index::Int,
                          episode_index::Int,
                          calls_before::Int,
                          calls_after::Int,
                          budget_remaining_before::Int,
                          budget_remaining_after::Int,
                          budget_cap_reached::Bool,
                          frontier_before_summary::Dict{String,Any},
                          frontier_after_summary::Dict{String,Any},
                          config::HierarchicalEditConfig,
                          run_context::Dict{String,Any},
                          he_selector_metadata::Dict{String,Any}=Dict{String,Any}())::Dict{String,Any}
    frontier_gain_values = Float64[get(entry.metadata, "frontier_utility_delta", 0.0) for entry in trajectory_entries]
    delta_top1_values = Float64[get(entry.metadata, "delta_top1", 0.0) for entry in trajectory_entries]
    delta_top10_values = Float64[get(entry.metadata, "delta_top10_mean", 0.0) for entry in trajectory_entries]
    enters_topk = any(Bool(get(entry.metadata, "enters_topk", false)) for entry in trajectory_entries)
    stop_reason = _episode_stop_reason(ep, proposal_logs, decision_logs, config.horizon)
    parent_smiles = isempty(parent_logs) ? String[] : [log.chosen_parent_smiles for log in parent_logs if !isempty(log.chosen_parent_smiles)]
    basin_scaffolds = isempty(basin_logs) ? String[] : [log.chosen_basin_scaffold for log in basin_logs if !isempty(log.chosen_basin_scaffold)]

    return Dict{String,Any}(
        "task_name" => get(run_context, "task_name", ep.task_name),
        "config_name" => get(run_context, "config_name", "unknown"),
        "run_index" => Int(get(run_context, "run_index", 0)),
        "phase" => phase,
        "segment_index" => segment_index,
        "episode_index" => episode_index,
        "episode_id" => ep.episode_id,
        "snapshot_id" => string(ep.snapshot_id),
        "calls_before" => calls_before,
        "calls_after" => calls_after,
        "calls_used" => max(0, calls_after - calls_before),
        "budget_remaining_before" => budget_remaining_before,
        "budget_remaining_after" => budget_remaining_after,
        "budget_cap_reached" => budget_cap_reached,
        "he_selector_metadata" => he_selector_metadata,
        "frontier_size_before" => ep.frontier_size_before,
        "frontier_size_after" => ep.frontier_size_after,
        "frontier_before_summary" => _compact_frontier_summary(frontier_before_summary),
        "frontier_after_summary" => _compact_frontier_summary(frontier_after_summary),
        "commits_applied" => ep.commits_applied,
        "step_count" => length(ep.steps),
        "best_smiles" => ep.best_smiles,
        "best_reward" => ep.best_reward,
        "improved_topk" => ep.improved_topk,
        "stop_reason" => stop_reason,
        "terminated_commit" => any(log -> log.terminated && log.commit_applied, decision_logs),
        "horizon_exhausted" => length(ep.steps) >= config.horizon,
        "no_valid_proposal" => stop_reason == "no_valid_proposal",
        "stagnation_or_low_value" => stop_reason == "stagnation_or_low_value",
        "frontier_gain_sum" => sum(frontier_gain_values),
        "frontier_gain_max" => isempty(frontier_gain_values) ? 0.0 : maximum(frontier_gain_values),
        "delta_top1_max" => isempty(delta_top1_values) ? 0.0 : maximum(delta_top1_values),
        "delta_top10_mean_max" => isempty(delta_top10_values) ? 0.0 : maximum(delta_top10_values),
        "enters_topk" => enters_topk,
        "chosen_parent_count" => length(parent_smiles),
        "unique_parent_count" => length(Set(parent_smiles)),
        "unique_basin_count" => length(Set(basin_scaffolds)),
    )
end

function _proposal_pipeline_summary(proposal_logs::Vector{HierarchicalEditProposalLog})::Dict{String,Any}
    if isempty(proposal_logs)
        return Dict{String,Any}(
            "episodes_with_proposals" => 0,
            "generated_total" => 0,
            "duplicate_total" => 0,
            "invalid_or_empty_total" => 0,
            "already_seen_total" => 0,
            "valid_after_filter_total" => 0,
            "committed_choice_total" => 0,
            "all_filtered_fraction" => 0.0,
        )
    end

    generated_total = sum(log.raw_candidate_count for log in proposal_logs)
    duplicate_total = sum(log.duplicate_candidate_count for log in proposal_logs)
    invalid_or_empty_total = sum(log.empty_child_count + log.self_child_count for log in proposal_logs)
    already_seen_total = sum(log.cached_child_count for log in proposal_logs)
    valid_after_filter_total = sum(log.unique_valid_count for log in proposal_logs)
    committed_choice_total = count(log -> !isempty(log.chosen_child_smiles), proposal_logs)
    all_filtered_fraction = mean(Float64[log.empty_after_filter for log in proposal_logs])

    return Dict{String,Any}(
        "episodes_with_proposals" => length(proposal_logs),
        "generated_total" => generated_total,
        "duplicate_total" => duplicate_total,
        "invalid_or_empty_total" => invalid_or_empty_total,
        "already_seen_total" => already_seen_total,
        "valid_after_filter_total" => valid_after_filter_total,
        "committed_choice_total" => committed_choice_total,
        "all_filtered_fraction" => all_filtered_fraction,
    )
end

function _run_capacity_summary(episode_summaries::Vector{Dict{String,Any}},
                               diagnostics_buffer::Union{Nothing,HierarchicalEditDiagnosticsBuffer})::Dict{String,Any}
    if isempty(episode_summaries)
        return Dict{String,Any}(
            "episode_count" => 0,
            "total_he_calls" => 0,
            "total_commits" => 0,
            "total_frontier_gain" => 0.0,
            "marginal_gain_per_call" => Float64[],
            "cumulative_calls_curve" => Int[],
            "cumulative_frontier_gain_curve" => Float64[],
            "cumulative_gain_per_call_curve" => Float64[],
            "stop_reason_counts" => Dict{String,Int}(),
            "phase_counts" => Dict{String,Int}(),
            "parent_reuse_distribution" => Dict{String,Int}(),
            "unique_basin_count" => 0,
        )
    end

    sorted_eps = sort(episode_summaries, by=ep -> (Int(get(ep, "segment_index", 0)), Int(get(ep, "episode_index", 0))))
    marginal_gain_per_call = Float64[]
    cumulative_calls_curve = Int[]
    cumulative_frontier_gain_curve = Float64[]
    cumulative_gain_per_call_curve = Float64[]
    stop_reason_counts = Dict{String,Int}()
    phase_counts = Dict{String,Int}()
    total_calls = 0
    total_frontier_gain = 0.0
    total_commits = 0

    for ep in sorted_eps
        calls_used = Int(get(ep, "calls_used", 0))
        frontier_gain = Float64(get(ep, "frontier_gain_sum", 0.0))
        total_calls += calls_used
        total_frontier_gain += frontier_gain
        total_commits += Int(get(ep, "commits_applied", 0))
        push!(marginal_gain_per_call, calls_used > 0 ? frontier_gain / calls_used : 0.0)
        push!(cumulative_calls_curve, total_calls)
        push!(cumulative_frontier_gain_curve, total_frontier_gain)
        push!(cumulative_gain_per_call_curve, total_calls > 0 ? total_frontier_gain / total_calls : 0.0)
        stop_reason = String(get(ep, "stop_reason", "unknown"))
        stop_reason_counts[stop_reason] = get(stop_reason_counts, stop_reason, 0) + 1
        phase = String(get(ep, "phase", "unknown"))
        phase_counts[phase] = get(phase_counts, phase, 0) + 1
    end

    parent_reuse_distribution = Dict{String,Int}()
    unique_basin_count = 0
    if !isnothing(diagnostics_buffer)
        for log in diagnostics_buffer.parent_logs
            isempty(log.chosen_parent_smiles) && continue
            parent_reuse_distribution[log.chosen_parent_smiles] = get(parent_reuse_distribution, log.chosen_parent_smiles, 0) + 1
        end
        unique_basin_count = length(Set(log.chosen_basin_scaffold for log in diagnostics_buffer.basin_logs if !isempty(log.chosen_basin_scaffold)))
    end

    return Dict{String,Any}(
        "episode_count" => length(sorted_eps),
        "total_he_calls" => total_calls,
        "total_commits" => total_commits,
        "total_frontier_gain" => total_frontier_gain,
        "marginal_gain_per_call" => marginal_gain_per_call,
        "cumulative_calls_curve" => cumulative_calls_curve,
        "cumulative_frontier_gain_curve" => cumulative_frontier_gain_curve,
        "cumulative_gain_per_call_curve" => cumulative_gain_per_call_curve,
        "stop_reason_counts" => stop_reason_counts,
        "phase_counts" => phase_counts,
        "parent_reuse_distribution" => parent_reuse_distribution,
        "unique_basin_count" => unique_basin_count,
    )
end

function _build_he_artifacts!(artifact_dir::Union{Nothing,String},
                              trajectory_buffer::Union{Nothing,EditTrajectoryBuffer},
                              diagnostics_buffer::Union{Nothing,HierarchicalEditDiagnosticsBuffer},
                              episode_summaries::Vector{Dict{String,Any}})::Tuple{Dict{String,String},Dict{String,Any}}
    if isnothing(artifact_dir)
        capacity_summary = _run_capacity_summary(episode_summaries, diagnostics_buffer)
        proposal_summary = _proposal_pipeline_summary(isnothing(diagnostics_buffer) ? HierarchicalEditProposalLog[] : diagnostics_buffer.proposal_logs)
        return Dict{String,String}(), Dict{String,Any}(
            "episode_summaries" => episode_summaries,
            "proposal_pipeline" => proposal_summary,
            "run_capacity" => capacity_summary,
        )
    end

    mkpath(artifact_dir)
    raw_trajectory_path = joinpath(artifact_dir, "he_raw_trajectory.jls")
    raw_diagnostics_path = joinpath(artifact_dir, "he_raw_diagnostics.jls")
    episode_summary_path = joinpath(artifact_dir, "he_episode_summary.jls")
    capacity_summary_path = joinpath(artifact_dir, "he_capacity_summary.jls")

    serialize(raw_trajectory_path, isnothing(trajectory_buffer) ? EditTrajectoryEntry[] : trajectory_buffer.entries)
    serialize(raw_diagnostics_path, isnothing(diagnostics_buffer) ? Dict{String,Any}() : Dict(
        "decision_logs" => diagnostics_buffer.logs,
        "proposal_logs" => diagnostics_buffer.proposal_logs,
        "basin_logs" => diagnostics_buffer.basin_logs,
        "parent_logs" => diagnostics_buffer.parent_logs,
        "operator_logs" => diagnostics_buffer.operator_logs,
    ))
    serialize(episode_summary_path, episode_summaries)

    proposal_summary = _proposal_pipeline_summary(isnothing(diagnostics_buffer) ? HierarchicalEditProposalLog[] : diagnostics_buffer.proposal_logs)
    capacity_summary = _run_capacity_summary(episode_summaries, diagnostics_buffer)
    compact_summary = Dict{String,Any}(
        "episode_summaries" => episode_summaries,
        "proposal_pipeline" => proposal_summary,
        "run_capacity" => capacity_summary,
    )
    serialize(capacity_summary_path, compact_summary)

    return Dict(
        "raw_trajectory" => raw_trajectory_path,
        "raw_diagnostics" => raw_diagnostics_path,
        "episode_summary" => episode_summary_path,
        "capacity_summary" => capacity_summary_path,
    ), compact_summary
end

# =============================================================================
# CAFE-GFN SMILES PMO Runner
# =============================================================================

function _add_seed_smiles!(replay_buffer, frontier_buffer, smiles::String, reward::Float64, vocab;
                           source::Symbol=:seed,
                           parent_smiles=nothing,
                           operator::Symbol=:seed,
                           scaffold_filter=nothing)
    if isnothing(replay_buffer) && isnothing(frontier_buffer)
        return false
    end
    if isempty(smiles) || reward <= 0.0
        return false
    end

    identity_smiles = canonicalize_smiles_identity(smiles)
    canonical_parent = isnothing(parent_smiles) ? nothing : canonicalize_smiles_identity(parent_smiles)

    tokens = try
        encode(vocab, identity_smiles)
    catch
        Int[]
    end

    if length(tokens) < 2
        return false
    end

    if !isnothing(scaffold_filter)
        should_add_molecule(scaffold_filter, identity_smiles) || return false
        register_molecule!(scaffold_filter, identity_smiles)
    end

    if !isnothing(replay_buffer)
        add_to_replay!(replay_buffer, identity_smiles, tokens, reward)
    end
    if !isnothing(frontier_buffer)
        add_to_frontier!(frontier_buffer, identity_smiles;
            reward=reward, source=source, parent_smiles=canonical_parent, operator=operator)
    end
    return true
end

function _seed_memories!(seed_smiles::Vector{String}, budget_oracle, vocab;
                         reward_fn_batch=nothing,
                         replay_buffer=nothing,
                         frontier_buffer=nothing,
                         scaffold_filter=nothing,
                         augmentation_count::Int=0,
                         verbose::Bool=true)
    isempty(seed_smiles) && return Dict("seeded" => 0, "augmented" => 0, "evaluated" => 0)

    seeded = 0
    augmented = 0
    unique_seeds = unique(canonicalize_smiles_identity(s) for s in seed_smiles if !isempty(strip(s)))
    seen_augmented = Set{String}(unique_seeds)
    augment_fn = augmentation_count > 0 ? create_augment_fn(vocab; n_augmentations=augmentation_count) : nothing

    rewards = if !isnothing(reward_fn_batch)
        try
            Float64[r for r in reward_fn_batch(unique_seeds)]
        catch
            Float64[try Float64(budget_oracle(s)) catch; 0.0 end for s in unique_seeds]
        end
    else
        Float64[try Float64(budget_oracle(s)) catch; 0.0 end for s in unique_seeds]
    end
    evaluated = length(unique_seeds)

    for (smi, reward) in zip(unique_seeds, rewards)
        if _add_seed_smiles!(replay_buffer, frontier_buffer, smi, reward, vocab;
                             source=:seed, operator=:seed, scaffold_filter=scaffold_filter)
            seeded += 1
        end

        if !isnothing(augment_fn) && reward > 0.0
            try
                for (aug_smi, _aug_tok) in augment_fn(smi)
                    canonical_aug = canonicalize_smiles_identity(aug_smi)
                    if canonical_aug != smi && !(canonical_aug in seen_augmented) &&
                       _add_seed_smiles!(replay_buffer, frontier_buffer, canonical_aug, reward, vocab;
                                         source=:augment, parent_smiles=smi,
                                         operator=:augment, scaffold_filter=scaffold_filter)
                        push!(seen_augmented, canonical_aug)
                        augmented += 1
                    end
                end
            catch
                # Augmentation failures are non-critical.
            end
        end
    end

    verbose && @info "Seeded PMO memories" seeded=seeded augmented=augmented evaluated=evaluated
    return Dict("seeded" => seeded, "augmented" => augmented, "evaluated" => evaluated)
end

function _bootstrap_frontier_from_model!(policy_model, params, states, vocab, budget_oracle;
                                         reward_fn_batch=nothing,
                                         replay_buffer=nothing,
                                         frontier_buffer=nothing,
                                         scaffold_filter=nothing,
                                         batch_size::Int=32,
                                         min_frontier_entries::Int=2,
                                         max_rounds::Int=4,
                                         verbose::Bool=true)
    isnothing(frontier_buffer) && return Dict("added" => 0, "evaluated" => 0, "rounds" => 0)
    length(frontier_buffer) >= min_frontier_entries && return Dict("added" => 0, "evaluated" => 0, "rounds" => 0)

    added = 0
    evaluated = 0
    rounds = 0

    while length(frontier_buffer) < min_frontier_entries && rounds < max_rounds
        rounds += 1
        samples = sample_smiles_batch(policy_model, params, states, vocab, batch_size;
            max_length=150, temperature=1.0, epsilon=0.05, constrained=true)

        unique_smiles = String[]
        seen = Set{String}()
        for (smi, _tok, _logp) in samples
            canonical = canonicalize_smiles_identity(smi)
            isempty(canonical) && continue
            canonical in seen && continue
            push!(unique_smiles, canonical)
            push!(seen, canonical)
        end
        isempty(unique_smiles) && continue

        rewards = if !isnothing(reward_fn_batch)
            try
                Float64[r for r in reward_fn_batch(unique_smiles)]
            catch
                Float64[try Float64(budget_oracle(s)) catch; 0.0 end for s in unique_smiles]
            end
        else
            Float64[try Float64(budget_oracle(s)) catch; 0.0 end for s in unique_smiles]
        end
        evaluated += length(unique_smiles)

        for (smi, reward) in zip(unique_smiles, rewards)
            if _add_seed_smiles!(replay_buffer, frontier_buffer, smi, reward, vocab;
                                 source=:bootstrap, operator=:bootstrap,
                                 scaffold_filter=scaffold_filter)
                added += 1
            end
        end
    end

    verbose && @info "Bootstrapped PMO frontier" added=added evaluated=evaluated rounds=rounds frontier_size=length(frontier_buffer)
    return Dict("added" => added, "evaluated" => evaluated, "rounds" => rounds)
end

"""
    run_smiles_pmo_task(task_name; budget=10000, pretrained_params=nothing,
                         pretrained_states=nothing, vocab=nothing,
                         training_mode=:tb, use_qgfn=false,
                         use_boosting=false, use_replay=true,
                         use_augmentation=false, use_scaffold_filter=false,
                         batch_size=32, verbose=true) → PMOResult

Run a single PMO benchmark task using SMILES-level CAFE-GFN with all improvements.

Integrates all CAFE-GFN components:
- TB v10 or RWMLE training with oracle rewards
- **Rank-based experience replay** (Genetic GFN, NeurIPS 2024)
- **QGFN Q-value masking** for budget-efficient sampling
- **SMILES augmentation** via randomized SMILES (zero oracle cost)
- **Scaffold diversity filter** to prevent mode collapse
- **Low KL** (0.01 vs previous 1.0) for faster exploration
- Sequential boosting with residual rewards

# Returns
PMOResult with AUC top-10, diversity, and other metrics
"""
function run_smiles_pmo_task(task_name::String;
                              budget::Int=10000,
                              pretrained_params=nothing,
                              pretrained_states=nothing,
                              vocab=nothing,
                              policy_model=nothing,
                              training_mode::Symbol=:tb,
                              use_qgfn::Bool=false,
                              use_boosting::Bool=false,
                              n_boost_rounds::Int=3,
                              use_replay::Bool=true,
                              replay_ratio::Int=4,
                              use_augmentation::Bool=false,
                              use_scaffold_filter::Bool=false,
                              batch_size::Int=32,
                              verbose::Bool=true,
                              # --- Novel features (Direction A & B) ---
                              beta_schedule::Symbol=:none,
                              beta_start::Float64=0.0,
                              beta_end::Float64=8.0,
                              delta_priority_replay::Bool=false,
                              # --- Enhanced hyperparameters (Genetic GFN-matched) ---
                              learning_rate::Float64=3e-5,
                              reward_exponent::Float64=8.0,
                              lr_z_multiplier::Float64=10.0,
                              lr_z::Float64=0.0,
                              log_z_grad_clip::Float64=1.0,
                              warmup_iters::Int=0,
                              kl_weight::Float64=0.01,
                              constructive_only::Bool=true,
                              reward_weighted::Bool=true,
                              n_iterations::Int=25,
                              # --- Per-step GA (Genetic GFN style) ---
                              ga_per_step::Bool=false,
                              ga_crossover::Int=8,
                              ga_mutation::Int=8,
                              # --- Loss function ---
                              loss_type::Symbol=:shifted_cosh,
                              # --- Frontier / seed initialization ---
                              track_frontier::Bool=false,
                              seed_smiles::Vector{String}=String[],
                              target_seed::Bool=false,
                              target_seed_augmentations::Int=8,
                              # --- CAFE-GFN 2.0: Scaffold-aware training ---
                              use_scaffold_aware::Bool=false,
                              use_graph_ga::Bool=false,
                              target_smiles::Union{Nothing,String}=nothing,
                              # --- Stage B′: shared frontier bootstrap ---
                              frontier_bootstrap_samples::Int=0,
                              frontier_bootstrap_min_entries::Int=2,
                              # --- Stage B: Hierarchical Edit integration ---
                              use_hierarchical_edit::Bool=false,
                              he_warmup_episodes::Int=8,
                              he_episodes_per_segment::Int=4,
                              he_budget_fraction::Float64=0.15,
                              he_config::HierarchicalEditConfig=HierarchicalEditConfig(),
                              he_artifact_dir::Union{Nothing,String}=nothing,
                              he_run_context::Dict{String,Any}=Dict{String,Any}(),
                              he_episode_selector=nothing,
                              allow_learned_he::Bool=false)::PMOResult
    lr_z_eff = lr_z > 0 ? lr_z : learning_rate * lr_z_multiplier
    @info "CAFE-GFN PMO task: $task_name (budget=$budget, mode=$training_mode, lr=$learning_rate, β=$reward_exponent, lr_z=$lr_z_eff, kl=$kl_weight, constr=$constructive_only, rw=$reward_weighted, replay=$(use_replay ? replay_ratio : 0), ga_step=$ga_per_step)"

    # --- Validate required arguments ---
    isnothing(pretrained_params) && error("pretrained_params required for CAFE-GFN PMO task")
    isnothing(pretrained_states) && error("pretrained_states required for CAFE-GFN PMO task")
    isnothing(vocab) && error("vocab required for CAFE-GFN PMO task")

    # Create policy model if not provided
    if isnothing(policy_model)
        actual_vocab_size = size(pretrained_params.output.layer_2.weight, 1)
        policy_model_local, _, _ = create_smiles_policy(;
            vocab_size=actual_vocab_size, hidden_dim=512, embed_dim=128, n_layers=3)
    else
        policy_model_local = policy_model
    end

    # --- Budget-tracked oracle ---
    oracle_mgr = OracleManager(
        [OracleConfig(task_name, 1.0)],
        budget, 0,
        Dict{String,Dict{String,Float64}}(),
        true  # benchmark_mode
    )

    # Initialize TDC oracle
    OracleBridge.init_oracles!([task_name];
        cache_dir=joinpath(@__DIR__, "..", "..", "..", "..", "data", "tdc_cache"))

    # Oracle wrapper for finetune_smiles_gflownet
    oracle_cache = Dict{String, Float64}()
    raw_reward_cache = Dict{String, Float64}()  # Raw (unshaped) rewards for PMO scoring

    function process_oracle_score!(smiles::String, score::Float64)::Float64
        canonical = canonicalize_smiles_identity(smiles)
        if !isnothing(scaffold_tracker)
            GFlowNet.update_scaffold_tracker!(scaffold_tracker, canonical, score)
            if scaffold_tracker.task_type == :unknown &&
               scaffold_tracker.global_stats.reward_count >= 30
                GFlowNet.detect_task_type!(scaffold_tracker)
                verbose && @info "Auto-detected task type: $(scaffold_tracker.task_type)"
            end
            if use_scaffold_aware
                shaped = GFlowNet.shape_reward(score, canonical, scaffold_tracker)
                raw_reward_cache[canonical] = score
                oracle_cache[canonical] = shaped
                return shaped
            end
        end

        oracle_cache[canonical] = score
        return score
    end

    function budget_oracle_batch(smiles_list::Vector{String})::Vector{Float64}
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
                haskey(oracle_cache, canonical) && continue
                score = lookup_score(oracle_mgr, smiles, task_name)
                process_oracle_score!(smiles, score)
            end
        end

        scores = Float64[]
        for smiles in smiles_list
            if isempty(smiles)
                push!(scores, 0.0)
            else
                canonical = canonicalize_smiles_identity(smiles)
                push!(scores, get(oracle_cache, canonical, 0.0))
            end
        end
        return scores
    end

    function budget_oracle(smiles::String)::Float64
        canonical = canonicalize_smiles_identity(smiles)
        haskey(oracle_cache, canonical) && return oracle_cache[canonical]
        scores = budget_oracle_batch([smiles])
        return isempty(scores) ? 0.0 : scores[1]
    end

    # --- FinetuningConfig with all improvements enabled ---
    finetune_config = if training_mode == :tb
        FinetuningConfig(;
            n_iterations=n_iterations,
            sample_batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_clip_norm=1.0,
            kl_weight=kl_weight,
            kl_decay_schedule=:none,
            loss_type=loss_type,
            cosh_threshold=2.0,
            max_length=150,
            temperature=1.0,
            epsilon=0.05,
            log_frequency=verbose ? 5 : 0,
            reward_exponent=reward_exponent,
            min_reward=0.01,
            training_mode=:tb,
            constructive_only=constructive_only,
            freeze_gru=true,
            reward_weighted=reward_weighted,
            unfreeze_top_gru=true,
            use_replay=use_replay,
            replay_ratio=replay_ratio,
            use_qgfn_sampling=use_qgfn,
            # Novel features (Direction A & B)
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            delta_priority_replay=delta_priority_replay,
            lr_z_multiplier=lr_z_multiplier,
            lr_z=lr_z,
            log_z_grad_clip=log_z_grad_clip,
            warmup_iters=warmup_iters,
        )
    elseif training_mode == :rwmle
        FinetuningConfig(;
            n_iterations=n_iterations,
            sample_batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_clip_norm=1.0,
            kl_weight=kl_weight,
            kl_decay_schedule=:none,
            loss_type=loss_type,
            cosh_threshold=2.0,
            max_length=150,
            temperature=1.0,
            epsilon=0.05,
            log_frequency=verbose ? 5 : 0,
            reward_exponent=reward_exponent,
            min_reward=0.01,
            training_mode=:rwmle,
            constructive_only=false,
            freeze_gru=false,
            reward_weighted=false,
            unfreeze_top_gru=false,
            use_replay=use_replay,
            replay_ratio=replay_ratio,
            use_qgfn_sampling=use_qgfn,
            lr_z_multiplier=lr_z_multiplier,
            lr_z=lr_z,
            log_z_grad_clip=log_z_grad_clip,
            warmup_iters=warmup_iters,
        )
    else
        error("Unknown training_mode: $training_mode. Use :tb or :rwmle")
    end

    ref_params = deepcopy(pretrained_params)
    ref_states = deepcopy(pretrained_states)

    # --- Shared replay buffer (persists across segments) ---
    replay_buffer = use_replay ? SMILESReplayBuffer(5000) : nothing

    # --- Frontier memory (first step toward hierarchical edit-flow search) ---
    # Force frontier tracking when HE is enabled
    frontier_buffer = (track_frontier || use_hierarchical_edit) ? MolecularFrontierBuffer(5000) : nothing

    # --- Stage B′ intrinsic accounting ---
    oracle_call_breakdown = _empty_pmo_oracle_call_breakdown()
    use_hierarchical_edit && !allow_learned_he && _assert_heuristic_he_config(he_config)

    # --- HE trajectory + diagnostics buffers (persist across segments for later artifact export) ---
    he_trajectory_buffer = use_hierarchical_edit ? EditTrajectoryBuffer(10000) : nothing
    he_diagnostics_buffer = use_hierarchical_edit ? HierarchicalEditDiagnosticsBuffer(10000) : nothing
    he_episode_summaries = Dict{String,Any}[]

    # --- QGFN setup ---
    q_function_net = nothing
    q_params_local = nothing
    q_states_local = nothing
    q_optimizer_local = nothing
    if use_qgfn
        actual_vocab_size = size(pretrained_params.output.layer_2.weight, 1)
        q_function_net, q_params_local, q_states_local = create_q_function(512, actual_vocab_size)
        q_opt = Optimisers.Adam(1e-4)
        q_optimizer_local = Optimisers.setup(q_opt, q_params_local)
    end

    # --- Augmentation function ---
    augment_fn = use_augmentation ? create_augment_fn(vocab; n_augmentations=4) : nothing

    # --- Scaffold diversity filter ---
    scaffold_filt = use_scaffold_filter ? ScaffoldFilter(; max_per_scaffold=25) : nothing

    # --- CAFE-GFN 2.0: Scaffold tracker (needed for Graph GA adaptive params OR scaffold-aware shaping) ---
    scaffold_tracker = (use_scaffold_aware || use_graph_ga) ?
        GFlowNet.ScaffoldTracker(; target_smiles=target_smiles) : nothing

    # --- Initial memory seeding (critical for structural tasks) ---
    seed_pool = copy(seed_smiles)
    if target_seed && !isnothing(target_smiles) && !(target_smiles in seed_pool)
        push!(seed_pool, target_smiles)
    end
    if !isempty(seed_pool)
        seed_start_calls = oracle_mgr.calls_used
        _seed_memories!(seed_pool, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            replay_buffer=replay_buffer,
            frontier_buffer=frontier_buffer,
            scaffold_filter=scaffold_filt,
            augmentation_count=target_seed ? target_seed_augmentations : 0,
            verbose=verbose)
        _accumulate_oracle_calls!(oracle_call_breakdown, "seed", oracle_mgr.calls_used - seed_start_calls)
    end

    # --- Stage B′: shared frontier bootstrap (fair identical starting state for TB and TB+HE) ---
    if frontier_bootstrap_samples > 0 && !isnothing(frontier_buffer) && length(frontier_buffer) < frontier_bootstrap_min_entries && !budget_exhausted(oracle_mgr)
        bootstrap_start_calls = oracle_mgr.calls_used
        _bootstrap_frontier_from_model!(policy_model_local, pretrained_params, pretrained_states, vocab, budget_oracle;
            reward_fn_batch=budget_oracle_batch,
            replay_buffer=replay_buffer,
            frontier_buffer=frontier_buffer,
            scaffold_filter=scaffold_filt,
            batch_size=frontier_bootstrap_samples,
            min_frontier_entries=frontier_bootstrap_min_entries,
            verbose=verbose)
        _accumulate_oracle_calls!(oracle_call_breakdown, "frontier_bootstrap", oracle_mgr.calls_used - bootstrap_start_calls)
    end

    he_budget_base = max(0, budget - get(oracle_call_breakdown, "frontier_bootstrap", 0))

    # --- Stage B: HE warmup phase (front-loaded structural search before model training) ---
    if use_hierarchical_edit && !isnothing(frontier_buffer) && length(frontier_buffer) >= 2 && !budget_exhausted(oracle_mgr)
        he_warmup_commits = 0
        he_total_budget = round(Int, he_budget_base * he_budget_fraction)
        he_warmup_budget_limit = max(1, round(Int, he_total_budget * 0.5))
        he_warmup_start_calls = oracle_mgr.calls_used

        for he_idx in 1:he_warmup_episodes
            budget_exhausted(oracle_mgr) && break
            (oracle_mgr.calls_used - he_warmup_start_calls) >= he_warmup_budget_limit && break

            episode_calls_before = oracle_mgr.calls_used
            budget_remaining_before = budget - oracle_mgr.calls_used
            selected_he_config, selected_operator_override, selector_metadata = _resolve_he_episode_selection(
                he_episode_selector,
                frontier_buffer,
                task_name,
                budget_remaining_before,
                -he_idx,
                "warmup",
                he_idx,
                he_config)
            frontier_before_summary = frontier_quality_summary(frontier_buffer; topk=selected_he_config.topk_tracking)
            ep = run_hierarchical_edit_episode!(
                frontier_buffer, he_trajectory_buffer, budget_oracle, vocab;
                reward_fn_batch=budget_oracle_batch,
                diagnostics_buffer=he_diagnostics_buffer,
                config=selected_he_config,
                target_smiles=target_smiles,
                budget_remaining=budget_remaining_before,
                created_at_step=-he_idx,
                task_name=task_name,
                operator_override=selected_operator_override)
            episode_calls_after = oracle_mgr.calls_used
            frontier_after_summary = frontier_quality_summary(frontier_buffer; topk=selected_he_config.topk_tracking)
            proposal_logs = _proposal_logs_for_episode(he_diagnostics_buffer, ep.episode_id)
            decision_logs = _decision_logs_for_episode(he_diagnostics_buffer, ep.episode_id)
            basin_logs = _basin_logs_for_episode(he_diagnostics_buffer, ep.episode_id)
            parent_logs = _parent_logs_for_episode(he_diagnostics_buffer, ep.episode_id)
            trajectory_entries = _trajectory_entries_for_episode(he_trajectory_buffer, ep.episode_id)
            budget_remaining_after = budget - oracle_mgr.calls_used
            budget_cap_reached = (oracle_mgr.calls_used - he_warmup_start_calls) >= he_warmup_budget_limit
            push!(he_episode_summaries, _episode_summary(
                ep,
                trajectory_entries,
                proposal_logs,
                decision_logs,
                basin_logs,
                parent_logs;
                phase="warmup",
                segment_index=0,
                episode_index=he_idx,
                calls_before=episode_calls_before,
                calls_after=episode_calls_after,
                budget_remaining_before=budget_remaining_before,
                budget_remaining_after=budget_remaining_after,
                budget_cap_reached=budget_cap_reached,
                frontier_before_summary=frontier_before_summary,
                frontier_after_summary=frontier_after_summary,
                config=selected_he_config,
                run_context=he_run_context,
                he_selector_metadata=selector_metadata,
            ))
            he_warmup_commits += ep.commits_applied
        end

        # B2: Feed HE frontier discoveries into the replay buffer for model training
        if !isnothing(replay_buffer) && he_warmup_commits > 0
            he_top = frontier_topk(frontier_buffer, min(64, length(frontier_buffer)); by=:reward)
            he_replay_added = 0
            for entry in he_top
                entry.source in (:edit, :warmup) || continue
                tokens = try encode(vocab, entry.smiles) catch; Int[] end
                length(tokens) < 2 && continue
                add_to_replay!(replay_buffer, entry.smiles, tokens, entry.reward)
                he_replay_added += 1
            end
            verbose && @info "HE warmup" episodes=he_warmup_episodes commits=he_warmup_commits replay_added=he_replay_added frontier_size=length(frontier_buffer)
        end
        _accumulate_oracle_calls!(oracle_call_breakdown, "he_warmup", oracle_mgr.calls_used - he_warmup_start_calls)
    end

    # --- Per-step GA function (Genetic GFN-style: 16 offspring per training step) ---
    per_step_ga_fn = if ga_per_step
        function(rb, _reward_fn)
            budget_exhausted(oracle_mgr) && return
            ga_start_calls = oracle_mgr.calls_used
            genetic_mols = generate_genetic_molecules(rb, vocab;
                n_crossover=ga_crossover, n_mutation=ga_mutation, n_augmentation=0,
                scaffold_filter=scaffold_filt)
            smiles_batch = String[gm_smi for (gm_smi, _) in genetic_mols]
            rewards_batch = budget_oracle_batch(smiles_batch)
            for ((gm_smi, gm_tok), gm_reward) in zip(genetic_mols, rewards_batch)
                budget_exhausted(oracle_mgr) && break
                if gm_reward > 0.01
                    add_to_replay!(rb, gm_smi, gm_tok, gm_reward)
                    if !isnothing(frontier_buffer)
                        add_to_frontier!(frontier_buffer, gm_smi;
                            reward=gm_reward, source=:ga, operator=:mutation)
                    end
                    if !isnothing(scaffold_filt)
                        register_molecule!(scaffold_filt, gm_smi)
                    end
                end
            end
            _accumulate_oracle_calls!(oracle_call_breakdown, "ga", oracle_mgr.calls_used - ga_start_calls)
        end
    else
        nothing
    end

    # --- Main training loop ---
    if use_boosting
        ensemble = BoostedGFlowNet()

        for round in 1:n_boost_rounds
            budget_exhausted(oracle_mgr) && break
            verbose && @info "Boosting round $round/$n_boost_rounds (budget: $(oracle_mgr.calls_used)/$budget)"

            residual_oracle = function(smiles::String)
                raw_reward = budget_oracle(smiles)
                return compute_residual_reward(raw_reward, ensemble, smiles)
            end
            residual_oracle_batch = function(smiles_list::Vector{String})
                raw_rewards = budget_oracle_batch(smiles_list)
                return Float64[compute_residual_reward(r, ensemble, s) for (s, r) in zip(smiles_list, raw_rewards)]
            end

            round_params = deepcopy(pretrained_params)
            round_replay = use_replay ? SMILESReplayBuffer(5000) : nothing
            segment = 0
            while !budget_exhausted(oracle_mgr) && segment < 8
                segment += 1
                round_segment_start_calls = oracle_mgr.calls_used
                round_segment_ga_before = get(oracle_call_breakdown, "ga", 0)
                result = finetune_smiles_gflownet(
                    policy_model_local, vocab, round_params, pretrained_states,
                    ref_params, ref_states, residual_oracle, finetune_config;
                    reward_fn_batch=residual_oracle_batch,
                    verbose=verbose,
                    replay_buffer=round_replay,
                    q_net=q_function_net, q_params=q_params_local,
                    q_states=q_states_local, q_optimizer=q_optimizer_local,
                    budget_used=oracle_mgr.calls_used, total_budget=budget,
                    augment_fn=nothing, scaffold_filter=scaffold_filt,
                    ga_fn=per_step_ga_fn)
                round_segment_total_calls = oracle_mgr.calls_used - round_segment_start_calls
                round_segment_ga_calls = get(oracle_call_breakdown, "ga", 0) - round_segment_ga_before
                _accumulate_oracle_calls!(oracle_call_breakdown, "model", round_segment_total_calls - round_segment_ga_calls)
                round_params = result.params
                q_params_local = result.q_params
                q_optimizer_local = result.q_optimizer
            end

            add_boosting_round!(ensemble, round_params, pretrained_states,
                0.0, length(oracle_cache))
        end
    else
        # Single model training: segmented fine-tuning until budget exhausted
        current_params = deepcopy(pretrained_params)
        best_params = nothing
        best_score = 0.0
        segment = 0
        current_log_Z = 0.0  # 0.0 triggers estimation in first segment

        while !budget_exhausted(oracle_mgr)
            segment += 1
            verbose && @info "Segment $segment (budget: $(oracle_mgr.calls_used)/$budget)"

            segment_start_calls = oracle_mgr.calls_used
            segment_ga_before = get(oracle_call_breakdown, "ga", 0)
            result = finetune_smiles_gflownet(
                policy_model_local, vocab, current_params, pretrained_states,
                ref_params, ref_states, budget_oracle, finetune_config;
                reward_fn_batch=budget_oracle_batch,
                verbose=verbose,
                replay_buffer=replay_buffer,
                q_net=q_function_net, q_params=q_params_local,
                q_states=q_states_local, q_optimizer=q_optimizer_local,
                budget_used=oracle_mgr.calls_used, total_budget=budget,
                augment_fn=nothing, scaffold_filter=scaffold_filt,
                ga_fn=per_step_ga_fn,
                log_Z_init=current_log_Z)
            segment_total_calls = oracle_mgr.calls_used - segment_start_calls
            segment_ga_calls = get(oracle_call_breakdown, "ga", 0) - segment_ga_before
            _accumulate_oracle_calls!(oracle_call_breakdown, "model", segment_total_calls - segment_ga_calls)

            current_params = result.params
            current_log_Z = result.log_Z  # Pass log_Z to next segment
            q_params_local = result.q_params
            q_optimizer_local = result.q_optimizer

            # Track best model
            if length(oracle_mgr.top_scores) >= 10
                current_score = mean(oracle_mgr.top_scores[1:10])
                if current_score > best_score
                    best_score = current_score
                    best_params = deepcopy(current_params)
                end
            end

            # Synchronize frontier memory from high-value replay entries.
            if !isnothing(frontier_buffer) && !isnothing(replay_buffer) && length(replay_buffer) > 0
                for mol in get_top_molecules(replay_buffer, min(128, length(replay_buffer)))
                    add_to_frontier!(frontier_buffer, mol.smiles;
                        reward=mol.reward, source=:model, operator=:sample)
                end
            end

            # --- Stage B1: Per-segment HE episodes (structural search interleaved with model training) ---
            local he_segment_commits = 0
            if use_hierarchical_edit && !isnothing(frontier_buffer) && length(frontier_buffer) >= 2 && !budget_exhausted(oracle_mgr)
                # Budget cap: remaining HE fraction (post-bootstrap shared budget base), divided across expected segments
                he_total_budget = round(Int, he_budget_base * he_budget_fraction)
                he_warmup_share = round(Int, he_total_budget * 0.5)
                he_segment_budget_pool = max(0, he_total_budget - he_warmup_share)
                he_per_segment_cap = max(1, round(Int, he_segment_budget_pool / max(1, 8)))  # ~8 segments expected
                he_segment_start_calls = oracle_mgr.calls_used

                for he_idx in 1:he_episodes_per_segment
                    budget_exhausted(oracle_mgr) && break
                    (oracle_mgr.calls_used - he_segment_start_calls) >= he_per_segment_cap && break

                    episode_calls_before = oracle_mgr.calls_used
                    budget_remaining_before = budget - oracle_mgr.calls_used
                    created_at_step = segment * 100 + he_idx
                    selected_he_config, selected_operator_override, selector_metadata = _resolve_he_episode_selection(
                        he_episode_selector,
                        frontier_buffer,
                        task_name,
                        budget_remaining_before,
                        created_at_step,
                        "interleaved",
                        he_idx,
                        he_config)
                    frontier_before_summary = frontier_quality_summary(frontier_buffer; topk=selected_he_config.topk_tracking)
                    ep = run_hierarchical_edit_episode!(
                        frontier_buffer, he_trajectory_buffer, budget_oracle, vocab;
                        reward_fn_batch=budget_oracle_batch,
                        diagnostics_buffer=he_diagnostics_buffer,
                        config=selected_he_config,
                        target_smiles=target_smiles,
                        budget_remaining=budget_remaining_before,
                        created_at_step=created_at_step,
                        task_name=task_name,
                        operator_override=selected_operator_override)
                    episode_calls_after = oracle_mgr.calls_used
                    frontier_after_summary = frontier_quality_summary(frontier_buffer; topk=selected_he_config.topk_tracking)
                    proposal_logs = _proposal_logs_for_episode(he_diagnostics_buffer, ep.episode_id)
                    decision_logs = _decision_logs_for_episode(he_diagnostics_buffer, ep.episode_id)
                    basin_logs = _basin_logs_for_episode(he_diagnostics_buffer, ep.episode_id)
                    parent_logs = _parent_logs_for_episode(he_diagnostics_buffer, ep.episode_id)
                    trajectory_entries = _trajectory_entries_for_episode(he_trajectory_buffer, ep.episode_id)
                    budget_remaining_after = budget - oracle_mgr.calls_used
                    budget_cap_reached = (oracle_mgr.calls_used - he_segment_start_calls) >= he_per_segment_cap
                    push!(he_episode_summaries, _episode_summary(
                        ep,
                        trajectory_entries,
                        proposal_logs,
                        decision_logs,
                        basin_logs,
                        parent_logs;
                        phase="interleaved",
                        segment_index=segment,
                        episode_index=he_idx,
                        calls_before=episode_calls_before,
                        calls_after=episode_calls_after,
                        budget_remaining_before=budget_remaining_before,
                        budget_remaining_after=budget_remaining_after,
                        budget_cap_reached=budget_cap_reached,
                        frontier_before_summary=frontier_before_summary,
                        frontier_after_summary=frontier_after_summary,
                        config=selected_he_config,
                        run_context=he_run_context,
                        he_selector_metadata=selector_metadata,
                    ))
                    he_segment_commits += ep.commits_applied
                end

                # B2: Feed new HE frontier entries into replay buffer for model learning
                if !isnothing(replay_buffer) && he_segment_commits > 0
                    he_top = frontier_topk(frontier_buffer, min(32, length(frontier_buffer)); by=:reward)
                    he_seg_replay_added = 0
                    for entry in he_top
                        entry.source in (:edit, :warmup) || continue
                        tokens = try encode(vocab, entry.smiles) catch; Int[] end
                        length(tokens) < 2 && continue
                        add_to_replay!(replay_buffer, entry.smiles, tokens, entry.reward)
                        he_seg_replay_added += 1
                    end
                    he_calls_used = oracle_mgr.calls_used - he_segment_start_calls
                    verbose && @info "HE segment $segment" episodes=he_episodes_per_segment commits=he_segment_commits replay_added=he_seg_replay_added calls=he_calls_used frontier_size=length(frontier_buffer)
                end
                _accumulate_oracle_calls!(oracle_call_breakdown, "he_interleaved", oracle_mgr.calls_used - he_segment_start_calls)
            end

            # --- CAFE-GFN 2.0: Between-segment Graph GA + adaptive params ---
            local segment_ga_calls = 0
            if use_graph_ga && !isnothing(scaffold_tracker) && !isnothing(replay_buffer) &&
               length(replay_buffer) >= 20 && !budget_exhausted(oracle_mgr)
                ga_start_calls = oracle_mgr.calls_used
                try
                    adaptive = GFlowNet.compute_adaptive_params(scaffold_tracker)
                    remaining = budget - oracle_mgr.calls_used
                    segment_ga_calls = GFlowNet.run_graph_ga_step!(
                        scaffold_tracker, replay_buffer, vocab,
                        budget_oracle, adaptive;
                        remaining_budget=remaining,
                        max_ga_per_segment=20)
                catch e
                    verbose && @warn "Graph GA step failed" exception=e
                end
                _accumulate_oracle_calls!(oracle_call_breakdown, "ga", oracle_mgr.calls_used - ga_start_calls)
                verbose && @info "Graph GA: $(segment_ga_calls) oracle calls"
            end

            # Fallback: BRICS GA when Graph GA produced nothing (or wasn't enabled)
            if segment_ga_calls == 0 && !ga_per_step && !isnothing(replay_buffer) &&
               length(replay_buffer) >= 20 && !budget_exhausted(oracle_mgr)
                ga_start_calls = oracle_mgr.calls_used
                try
                    genetic_mols = generate_genetic_molecules(replay_buffer, vocab;
                        n_crossover=4, n_mutation=4, n_augmentation=8,
                        scaffold_filter=scaffold_filt)
                    smiles_batch = String[gm_smi for (gm_smi, _) in genetic_mols]
                    rewards_batch = budget_oracle_batch(smiles_batch)
                    for ((gm_smi, gm_tok), gm_reward) in zip(genetic_mols, rewards_batch)
                        if gm_reward > 0.0
                            add_to_replay!(replay_buffer, gm_smi, gm_tok, gm_reward)
                            if !isnothing(frontier_buffer)
                                add_to_frontier!(frontier_buffer, gm_smi;
                                    reward=gm_reward, source=:ga, operator=:crossover)
                            end
                        end
                        budget_exhausted(oracle_mgr) && break
                    end
                catch
                    # GA operations may fail if RDKit unavailable — non-critical
                end
                _accumulate_oracle_calls!(oracle_call_breakdown, "ga", oracle_mgr.calls_used - ga_start_calls)
            end

            # --- Augmentation of top molecules (free data, no oracle cost) ---
            if use_augmentation && !isnothing(replay_buffer) && !isnothing(augment_fn) &&
               length(replay_buffer) >= 10
                try
                    top_mols = get_top_molecules(replay_buffer, min(20, length(replay_buffer)))
                    for mol in top_mols
                        aug_pairs = augment_fn(mol.smiles)
                        for (aug_smi, aug_tok) in aug_pairs
                            canonical_aug = canonicalize_smiles_identity(aug_smi)
                            if !isempty(canonical_aug) && canonical_aug != mol.smiles
                                add_to_replay!(replay_buffer, canonical_aug, aug_tok, mol.reward)
                            end
                        end
                    end
                catch
                end
            end

            segment >= 20 && break
        end
    end

    # --- Compute final metrics ---
    top_scores = sort(oracle_mgr.top_scores, rev=true)
    top1 = isempty(top_scores) ? 0.0 : top_scores[1]
    top10_mean = isempty(top_scores) ? 0.0 : mean(top_scores[1:min(10, length(top_scores))])

    # Diversity of top-100 molecules
    diversity = 0.0
    try
        sorted_cache = sort(collect(oracle_mgr.cache), by=x -> -get(x[2], task_name, 0.0))
        top100_smiles = [kv[1] for kv in sorted_cache[1:min(100, length(sorted_cache))]]
        if length(top100_smiles) >= 2
            fps = [RDKitBridge.compute_fingerprint(s) for s in top100_smiles]
            valid_fps = filter(fp -> sum(fp) > 0, fps)
            if length(valid_fps) >= 2
                div_stats = RDKitBridge.compute_diversity_stats(valid_fps)
                diversity = div_stats["internal_diversity_1"]
            end
        end
    catch
    end

    auc = compute_auc_top10(oracle_mgr)

    accounted_calls = sum(values(oracle_call_breakdown))
    oracle_call_breakdown["unattributed"] = max(0, oracle_mgr.calls_used - accounted_calls)
    oracle_call_breakdown["total"] = oracle_mgr.calls_used
    provenance_summary = _pmo_provenance_summary(frontier_buffer)

    artifact_paths = Dict{String,String}()
    diagnostics_summary = Dict{String,Any}()
    if use_hierarchical_edit
        artifact_paths, diagnostics_summary = _build_he_artifacts!(
            he_artifact_dir,
            he_trajectory_buffer,
            he_diagnostics_buffer,
            he_episode_summaries,
        )
    end

    result = PMOResult(
        task_name,
        auc,
        top1,
        top10_mean,
        diversity,
        oracle_mgr.calls_used,
        length(oracle_cache),
        oracle_call_breakdown,
        provenance_summary,
        artifact_paths,
        diagnostics_summary,
    )

    @info "CAFE-GFN [$task_name] complete" auc_top10=auc top1=top1 top10=top10_mean diversity=diversity

    # --- B3 / Stage B′: source-aware contribution diagnostics ---
    if !isempty(provenance_summary)
        topk_fracs = get(provenance_summary, "topk_source_fractions", Dict{String,Float64}())
        overall_fracs = get(provenance_summary, "overall_source_fractions", Dict{String,Float64}())
        @info "PMO provenance [$task_name]" tb_top10=round(get(topk_fracs, "tb", 0.0), digits=3) ga_top10=round(get(topk_fracs, "ga", 0.0), digits=3) he_top10=round(get(topk_fracs, "he", 0.0), digits=3) seed_top10=round(get(topk_fracs, "seed", 0.0), digits=3) bootstrap_top10=round(get(topk_fracs, "bootstrap", 0.0), digits=3) tb_frontier=round(get(overall_fracs, "tb", 0.0), digits=3) ga_frontier=round(get(overall_fracs, "ga", 0.0), digits=3) he_frontier=round(get(overall_fracs, "he", 0.0), digits=3) bootstrap_frontier=round(get(overall_fracs, "bootstrap", 0.0), digits=3)
    end
    if use_hierarchical_edit && !isempty(diagnostics_summary)
        run_capacity = get(diagnostics_summary, "run_capacity", Dict{String,Any}())
        @info "PMO HE diagnostics [$task_name]" episodes=Int(get(run_capacity, "episode_count", 0)) total_he_calls=Int(get(run_capacity, "total_he_calls", 0)) total_commits=Int(get(run_capacity, "total_commits", 0)) total_frontier_gain=round(Float64(get(run_capacity, "total_frontier_gain", 0.0)), digits=4) stop_reasons=get(run_capacity, "stop_reason_counts", Dict{String,Int}()) artifact_paths=artifact_paths
    end

    return result
end

"""
    run_smiles_pmo_benchmark(; tasks=PMO_23_TASKS, n_runs=5, budget=10000,
                               pretrained_params, pretrained_states, vocab,
                               training_mode=:tb, kwargs...) → PMOBenchmarkReport

Run the full PMO benchmark using CAFE-GFN SMILES runner.

Each task is run n_runs times with the actual training pipeline.

# Arguments
- `tasks`: Vector of PMO task names (default: all 23)
- `n_runs`: Independent runs per task (default: 5)
- `budget`: Oracle budget per task per run (default: 10000)
- `pretrained_params`, `pretrained_states`, `vocab`: Required CAFE-GFN components
- `training_mode`: `:tb` or `:rwmle`
- Additional kwargs passed to `run_smiles_pmo_task`
"""
function run_smiles_pmo_benchmark(;
    tasks::Vector{String}=PMO_23_TASKS,
    n_runs::Int=5,
    budget::Int=10000,
    pretrained_params=nothing,
    pretrained_states=nothing,
    vocab=nothing,
    policy_model=nothing,
    training_mode::Symbol=:tb,
    kwargs...
)::PMOBenchmarkReport
    @info "Starting CAFE-GFN PMO benchmark" n_tasks=length(tasks) n_runs=n_runs budget=budget mode=training_mode

    all_results = PMOResult[]

    for task in tasks
        task_aucs = Float64[]
        last_result = nothing

        for run in 1:n_runs
            @info "  Run $run/$n_runs for $task"
            result = run_smiles_pmo_task(task;
                budget=budget,
                pretrained_params=pretrained_params,
                pretrained_states=pretrained_states,
                vocab=vocab,
                policy_model=policy_model,
                training_mode=training_mode,
                kwargs...)
            push!(task_aucs, result.auc_top10)
            last_result = result
        end

        push!(all_results, last_result)

        mean_auc = mean(task_aucs)
        std_auc = n_runs > 1 ? std(task_aucs) : 0.0
        @info "  $task: AUC top-10 = $(round(mean_auc, digits=4)) ± $(round(std_auc, digits=4))"
    end

    total_score = sum(r.auc_top10 for r in all_results)
    report = PMOBenchmarkReport(all_results, total_score, n_runs, budget)

    @info "CAFE-GFN PMO Benchmark Complete" total_score=total_score
    for (name, sota) in sort(collect(SOTA_SCORES), by=x->-x[2])
        @info "  $name: $sota $(total_score > sota ? "✓ (beat)" : "")"
    end

    return report
end
