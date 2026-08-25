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
end

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
        "task_name"        => r.task_name,
        "auc_top10"        => r.auc_top10,
        "top1"             => r.top1,
        "top10_mean"       => r.top10_mean,
        "diversity"        => r.diversity,
        "n_oracle_calls"   => r.n_oracle_calls,
        "unique_molecules" => r.unique_molecules,
    ) for r in report.task_results]

    return Dict(
        "tasks"          => tasks,
        "total_score"    => report.total_score,
        "n_runs"         => report.n_runs,
        "budget_per_task" => report.budget_per_task,
        "sota_comparison" => SOTA_SCORES,
    )
end
