#!/usr/bin/env julia
# ============================================================
# Experiment: Training Objective Comparison on DRD2 Oracle
# ============================================================
#
# Compares TB, DB, STB, FM, TLM training objectives for
# fragment-based molecular GFlowNet on the DRD2 bioactivity task.
#
# Protocol:
#   - Budget: 10,000 oracle calls per run (PMO standard)
#   - Metric: AUC top-10 (sampled every 100 calls)
#   - 3 independent runs per objective → mean ± std
#   - Also records: top-1, top-10 mean, diversity, unique molecules
#
# Usage:
#   julia --project experiments/objective_comparison_drd2.jl [--budget 10000] [--runs 3] [--quick]

using Pkg
Pkg.activate(".")

using Random
using Statistics: mean, std
using Dates
using Printf
using GFlowNet

# ---- Load molecular generation domain ----
include(joinpath(@__DIR__, "..", "src", "utils", "visualization", "python", "rdkit_bridge.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "visualization", "python", "oracle_bridge.jl"))
include(joinpath(@__DIR__, "..", "src", "utils", "visualization", "core", "oracle_manager.jl"))
include(joinpath(@__DIR__, "..", "src", "applications", "molecular_generation.jl"))

# ============================================================
# Configuration
# ============================================================

# Parse CLI args
const QUICK_MODE = "--quick" in ARGS
const BUDGET = let
    idx = findfirst(==("--budget"), ARGS)
    idx !== nothing && idx < length(ARGS) ? parse(Int, ARGS[idx+1]) : (QUICK_MODE ? 1000 : 10000)
end
const N_RUNS = let
    idx = findfirst(==("--runs"), ARGS)
    idx !== nothing && idx < length(ARGS) ? parse(Int, ARGS[idx+1]) : (QUICK_MODE ? 1 : 3)
end
const HIDDEN_DIM = 256
const BATCH_SIZE = 32
const LR = 0.001
const ORACLE_NAME = "drd2"

# Objectives to compare
const OBJECTIVES = [
    (name = "TB",  objective = GFlowNet.TRAJECTORY_BALANCE,                include_backward = false, include_flow = false, is_mogfn = false),
    (name = "DB",  objective = GFlowNet.DETAILED_BALANCE,                  include_backward = true,  include_flow = false, is_mogfn = false),
    (name = "STB", objective = GFlowNet.SUB_TRAJECTORY_BALANCE,            include_backward = false, include_flow = true,  is_mogfn = false),
    (name = "FM",  objective = GFlowNet.FLOW_MATCHING,                     include_backward = false, include_flow = true,  is_mogfn = false),
    (name = "TLM", objective = GFlowNet.TRAJECTORY_LIKELIHOOD_MAXIMIZATION, include_backward = true,  include_flow = false, is_mogfn = false),
]

# ============================================================
# Result Types
# ============================================================

struct RunResult
    objective_name::String
    run_id::Int
    auc_top10::Float64
    top1::Float64
    top10_mean::Float64
    diversity::Float64
    n_oracle_calls::Int
    unique_molecules::Int
    wall_time_sec::Float64
    loss_history::Vector{Float64}
end

# ============================================================
# Core Experiment: Run one objective
# ============================================================

function run_single_experiment(obj_config; budget::Int, run_id::Int)
    @info "═══ $(obj_config.name) Run $run_id ═══" budget=budget

    t_start = time()

    # Create oracle manager in benchmark_mode (oracle-only reward)
    oracle_mgr = OracleManager(
        [OracleConfig(ORACLE_NAME, 1.0)],
        budget,
        0,
        Dict{String,Dict{String,Float64}}(),
        true  # benchmark_mode
    )

    # Create model with appropriate components for the objective
    if obj_config.is_mogfn
        model = create_mogfn_molecular_gflownet(
            hidden_dim = HIDDEN_DIM,
            learning_rate = LR,
            n_objectives = 1,  # Single-objective DRD2
            preference_dim = 64,
            include_backward = false,
        )
    else
        model = create_molecular_gflownet(
            hidden_dim = HIDDEN_DIM,
            learning_rate = LR,
            include_backward = obj_config.include_backward,
            include_flow_estimator = obj_config.include_flow,
            partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,
        )
    end

    # Sampling config
    sampling_config = GFlowNet.SamplingConfig(
        strategy = GFlowNet.STOCHASTIC_SAMPLING,
        temperature = 1.0,
        epsilon = 0.1,
        max_trajectory_length = 100,
    )

    # Training config
    training_config = GFlowNet.TrainingConfig(
        objective = obj_config.objective,
        n_iterations = budget,
        batch_size = BATCH_SIZE,
        learning_rate = LR,
        temperature = 1.0,
        epsilon = 0.1,
        epsilon_decay = true,
        entropy_weight = 0.02,
        z_learning_rate_multiplier = 10.0,
        sub_trajectory_length = 5,
        use_replay_buffer = true,
        replay_buffer_size = 5000,
        replay_ratio = 0.3,
        replay_priority_alpha = 0.6,
        verbose = false,
        tlm_backward_weight = 1.0,
        tlm_update_frequency = 1,
        tlm_entropy_coeff = 0.01,
    )

    # Training loop — budget-driven
    all_smiles = Set{String}()
    loss_history = Float64[]
    iteration = 0

    while !budget_exhausted(oracle_mgr)
        iteration += 1

        # Sample trajectories
        trajectories = Trajectory[]
        for _ in 1:BATCH_SIZE
            try
                traj = GFlowNet.sample_trajectory(model; config=sampling_config)
                push!(trajectories, traj)
            catch
                continue
            end
        end
        isempty(trajectories) && continue

        # Collect terminal SMILES
        terminal_smiles = String[]
        for traj in trajectories
            terminal = traj.states[end]
            if GFlowNet.is_terminal_state(terminal) && hasproperty(terminal, :smiles) && !isempty(terminal.smiles)
                push!(terminal_smiles, terminal.smiles)
                push!(all_smiles, terminal.smiles)
            end
        end

        # Batch evaluate oracle
        if !isempty(terminal_smiles)
            evaluate_molecules!(oracle_mgr, unique(terminal_smiles))
        end
        budget_exhausted(oracle_mgr) && break

        # Train step
        try
            GFlowNet.train_step!(model, trajectories, training_config)
            # Record loss (approximate from reward signal)
            if !isempty(terminal_smiles)
                avg_score = mean([lookup_score(oracle_mgr, s, ORACLE_NAME) for s in terminal_smiles])
                push!(loss_history, avg_score)
            end
        catch e
            @warn "Train step failed" iteration=iteration exception=e
            continue
        end

        # Progress logging every 50 iterations
        if iteration % 50 == 0
            top_scores = sort(oracle_mgr.top_scores, rev=true)
            top1_now = isempty(top_scores) ? 0.0 : top_scores[1]
            @info "  [$(obj_config.name)] iter=$iteration budget=$(oracle_mgr.calls_used)/$(budget) top1=$(round(top1_now, digits=4)) unique=$(length(all_smiles))"
        end
    end

    # Compute final metrics
    top_scores = sort(oracle_mgr.top_scores, rev=true)
    top1 = isempty(top_scores) ? 0.0 : top_scores[1]
    top10_mean_val = isempty(top_scores) ? 0.0 : mean(top_scores[1:min(10, length(top_scores))])

    # Diversity of top-100
    diversity = 0.0
    try
        sorted_cache = sort(collect(oracle_mgr.cache), by=x -> -get(x[2], ORACLE_NAME, 0.0))
        top100_smiles = [kv[1] for kv in sorted_cache[1:min(100, length(sorted_cache))]]
        if length(top100_smiles) >= 2
            fps = [RDKitBridge.compute_fingerprint(s) for s in top100_smiles]
            valid_fps = filter(fp -> sum(fp) > 0, fps)
            if length(valid_fps) >= 2
                div_stats = RDKitBridge.compute_diversity_stats(valid_fps)
                diversity = div_stats["internal_diversity_1"]
            end
        end
    catch e
        @warn "Diversity computation failed" exception=e
    end

    auc = compute_auc_top10(oracle_mgr)
    wall_time = time() - t_start

    result = RunResult(
        obj_config.name, run_id, auc, top1, top10_mean_val,
        diversity, oracle_mgr.calls_used, length(all_smiles),
        wall_time, loss_history
    )

    @info "═══ $(obj_config.name) Run $run_id DONE ═══" auc=round(auc, digits=4) top1=round(top1, digits=4) top10=round(top10_mean_val, digits=4) diversity=round(diversity, digits=3) time_sec=round(wall_time, digits=1) unique=length(all_smiles)

    return result
end

# ============================================================
# Results Serialization
# ============================================================

function results_to_dict(results::Vector{RunResult})
    by_objective = Dict{String,Vector{RunResult}}()
    for r in results
        push!(get!(by_objective, r.objective_name, RunResult[]), r)
    end

    summary = []
    for (name, runs) in sort(collect(by_objective), by=x->x[1])
        aucs = [r.auc_top10 for r in runs]
        top1s = [r.top1 for r in runs]
        top10s = [r.top10_mean for r in runs]
        divs = [r.diversity for r in runs]
        times = [r.wall_time_sec for r in runs]
        uniques = [r.unique_molecules for r in runs]

        push!(summary, Dict(
            "objective" => name,
            "n_runs" => length(runs),
            "auc_top10_mean" => mean(aucs),
            "auc_top10_std" => length(aucs) > 1 ? std(aucs) : 0.0,
            "top1_mean" => mean(top1s),
            "top1_std" => length(top1s) > 1 ? std(top1s) : 0.0,
            "top10_mean" => mean(top10s),
            "top10_std" => length(top10s) > 1 ? std(top10s) : 0.0,
            "diversity_mean" => mean(divs),
            "diversity_std" => length(divs) > 1 ? std(divs) : 0.0,
            "unique_molecules_mean" => mean(uniques),
            "wall_time_mean" => mean(times),
            "per_run" => [Dict(
                "run" => r.run_id,
                "auc_top10" => r.auc_top10,
                "top1" => r.top1,
                "top10_mean" => r.top10_mean,
                "diversity" => r.diversity,
                "n_oracle_calls" => r.n_oracle_calls,
                "unique_molecules" => r.unique_molecules,
                "wall_time_sec" => r.wall_time_sec,
            ) for r in runs]
        ))
    end

    return Dict(
        "experiment" => "objective_comparison_drd2",
        "oracle" => ORACLE_NAME,
        "budget" => BUDGET,
        "n_runs" => N_RUNS,
        "hidden_dim" => HIDDEN_DIM,
        "batch_size" => BATCH_SIZE,
        "learning_rate" => LR,
        "timestamp" => string(now()),
        "quick_mode" => QUICK_MODE,
        "results" => summary,
        "sota_reference" => Dict(
            "genetic_gfn_total" => 16.213,
            "reinvent_total" => 15.185,
            "fragment_gfn_total" => 9.929,
            "note" => "SOTA scores are total across 23 tasks; our experiment is single-task DRD2",
        ),
    )
end

function save_results(results::Vector{RunResult}, path::String)
    data = results_to_dict(results)

    # Manual JSON serialization (avoid JSON3 dependency)
    function to_json(x, indent=0)
        pad = "  " ^ indent
        if x isa Dict
            items = ["$pad  \"$(k)\": $(to_json(v, indent+1))" for (k,v) in sort(collect(x), by=first)]
            return "{\n$(join(items, ",\n"))\n$pad}"
        elseif x isa Vector
            if isempty(x)
                return "[]"
            end
            items = ["$pad  $(to_json(v, indent+1))" for v in x]
            return "[\n$(join(items, ",\n"))\n$pad]"
        elseif x isa AbstractString
            return "\"$(escape_string(x))\""
        elseif x isa Number
            isnan(x) && return "null"
            isinf(x) && return "null"
            return string(round(x, digits=6))
        elseif x isa Bool
            return string(x)
        else
            return "\"$(string(x))\""
        end
    end

    mkpath(dirname(path))
    write(path, to_json(data))
    @info "Results saved" path=path
end

# ============================================================
# Print Summary Table
# ============================================================

function print_summary(results::Vector{RunResult})
    by_objective = Dict{String,Vector{RunResult}}()
    for r in results
        push!(get!(by_objective, r.objective_name, RunResult[]), r)
    end

    println("\n" * "="^80)
    println("EXPERIMENT RESULTS: Training Objective Comparison on DRD2")
    println("Budget: $BUDGET oracle calls | Runs: $N_RUNS per objective")
    println("="^80)
    println()

    # Header
    @printf("%-6s  %12s  %10s  %10s  %10s  %8s  %8s\n",
            "Obj", "AUC Top-10", "Top-1", "Top-10 Avg", "Diversity", "Unique", "Time(s)")
    println("-"^76)

    sorted_objs = sort(collect(by_objective), by=x -> -mean([r.auc_top10 for r in x[2]]))
    for (name, runs) in sorted_objs
        aucs = [r.auc_top10 for r in runs]
        top1s = [r.top1 for r in runs]
        top10s = [r.top10_mean for r in runs]
        divs = [r.diversity for r in runs]
        uniques = [r.unique_molecules for r in runs]
        times = [r.wall_time_sec for r in runs]

        if length(runs) > 1
            @printf("%-6s  %5.4f±%.4f  %5.4f±%.3f  %5.4f±%.3f  %5.3f±%.3f  %6.0f  %6.0f\n",
                name,
                mean(aucs), std(aucs),
                mean(top1s), std(top1s),
                mean(top10s), std(top10s),
                mean(divs), std(divs),
                mean(uniques), mean(times))
        else
            @printf("%-6s  %12.4f  %10.4f  %10.4f  %10.3f  %8d  %8.0f\n",
                name, aucs[1], top1s[1], top10s[1], divs[1], Int(uniques[1]), times[1])
        end
    end

    println()
    println("SOTA Reference (full 23-task PMO):")
    println("  Genetic GFN: 16.213 | REINVENT: 15.185 | Fragment GFN: 9.929")
    println("  (Our experiment: single DRD2 task, not directly comparable)")
    println("="^80)
end

# ============================================================
# Main Entry Point
# ============================================================

function main()
    println("╔═══════════════════════════════════════════════════════════╗")
    println("║  GFlowNet Training Objective Comparison — DRD2 Oracle   ║")
    println("║  Budget: $BUDGET | Runs: $N_RUNS | Mode: $(QUICK_MODE ? "QUICK" : "FULL")          ║")
    println("╚═══════════════════════════════════════════════════════════╝")

    # Initialize Python/RDKit/TDC
    @info "Initializing RDKit bridge..."
    RDKitBridge.init_rdkit!()

    @info "Initializing TDC oracle: $ORACLE_NAME..."
    OracleBridge.init_oracles!([ORACLE_NAME])

    # Verify oracle works
    test_score = OracleBridge.evaluate("c1ccccc1", ORACLE_NAME)
    @info "Oracle sanity check" smiles="benzene" drd2_score=round(test_score, digits=4)

    # Run all experiments
    all_results = RunResult[]

    for obj_config in OBJECTIVES
        for run_id in 1:N_RUNS
            # Set different random seed per run
            Random.seed!(42 + run_id * 1000 + hash(obj_config.name))

            result = run_single_experiment(obj_config; budget=BUDGET, run_id=run_id)
            push!(all_results, result)
        end
    end

    # Print summary
    print_summary(all_results)

    # Save results
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    results_path = joinpath(@__DIR__, "..", "results", "$(timestamp)_objective_comparison_drd2.json")
    save_results(all_results, results_path)

    return all_results
end

# Run!
results = main()
