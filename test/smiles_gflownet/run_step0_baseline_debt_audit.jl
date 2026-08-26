#!/usr/bin/env julia

# Step 0 Baseline-Debt Audit
#
# Purpose:
#   Before spending more effort on Option-Flow integration, test whether already
#   implemented PMO baselines (QGFN, Boosting, QGFN+Boosting, heuristic HE)
#   close the relevant gap. This is a go/no-go audit, not a SOTA claim.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GFlowNet
using Serialization
using Statistics
using Random
using Dates
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "applications", "smiles_gflownet.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "rdkit_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "oracle_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "oracle_manager.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "pmo_benchmark.jl"))

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUTDIR = joinpath(ROOT, "checkpoints", "step0_baseline_debt_audit")
mkpath(OUTDIR)

const TARGET_SMILES = Dict(
    "albuterol_similarity" => "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "celecoxib_rediscovery" => "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "mestranol_similarity" => "C#C[C@]1(O)CC[C@H]2[C@@H]3CCc4cc(OC)ccc4[C@H]3CC[C@@]21C",
    "thiothixene_rediscovery" => "C=C(c1ccc(S(=O)(=O)N2CCN(C)CC2)cc1)c1cc2c(s1)Cc1ccccc1-2",
)

const CONFIRMATORY_ARMS = Set(["tb_only", "tb_qgfn", "tb_boosting", "tb_qgfn_boosting", "heuristic_he"])
const EXPLORATORY_ARMS = Set(["tb_ga_per_step"])

function logmsg(msg)
    println("[", Dates.format(now(), "HH:MM:SS"), "] ", msg)
    flush(stdout)
end

function parse_env_int(name::String, default::Int)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    return parse(Int, raw)
end

function parse_env_float(name::String, default::Float64)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    return parse(Float64, raw)
end

function parse_csv_strings(name::String, default::Vector{String})
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    return [String(strip(x)) for x in split(raw, ',') if !isempty(strip(x))]
end

function parse_csv_ints(name::String, default::Vector{Int})
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    return parse.(Int, strip.(split(raw, ',')))
end

function load_pretrain()
    checkpoint_path = joinpath(ROOT, "checkpoints", "pretrain", "final.jls")
    isfile(checkpoint_path) || error("Pretrained checkpoint not found: $(checkpoint_path)")
    checkpoint = deserialize(checkpoint_path)
    pretrained_params = checkpoint["params"]
    pretrained_states = checkpoint["states"]
    vocab = SMILESVocabulary()
    vocab_size = if haskey(checkpoint, "vocab_size")
        Int(checkpoint["vocab_size"])
    else
        Int(size(pretrained_params.output.layer_2.weight, 1))
    end
    policy_model, _, _ = create_smiles_policy(; vocab_size=vocab_size, hidden_dim=512, embed_dim=128, n_layers=3)
    return Dict{String,Any}(
        "checkpoint_path" => checkpoint_path,
        "pretrained_params" => pretrained_params,
        "pretrained_states" => pretrained_states,
        "vocab" => vocab,
        "vocab_size" => vocab_size,
        "policy_model" => policy_model,
    )
end

function arm_flags(arm::String)
    if arm == "tb_only"
        return Dict(
            :use_qgfn => false,
            :use_boosting => false,
            :use_hierarchical_edit => false,
            :ga_per_step => false,
        )
    elseif arm == "tb_qgfn"
        return Dict(:use_qgfn => true, :use_boosting => false, :use_hierarchical_edit => false, :ga_per_step => false)
    elseif arm == "tb_boosting"
        return Dict(:use_qgfn => false, :use_boosting => true, :use_hierarchical_edit => false, :ga_per_step => false)
    elseif arm == "tb_qgfn_boosting"
        return Dict(:use_qgfn => true, :use_boosting => true, :use_hierarchical_edit => false, :ga_per_step => false)
    elseif arm == "heuristic_he"
        return Dict(:use_qgfn => false, :use_boosting => false, :use_hierarchical_edit => true, :ga_per_step => false)
    elseif arm == "tb_ga_per_step"
        return Dict(:use_qgfn => false, :use_boosting => false, :use_hierarchical_edit => false, :ga_per_step => true)
    else
        error("unknown Step0 arm: $(arm)")
    end
end

function pmo_result_summary(result)
    return Dict{String,Any}(
        "task_name" => result.task_name,
        "auc_top10" => Float64(result.auc_top10),
        "top1" => Float64(result.top1),
        "top10_mean" => Float64(result.top10_mean),
        "diversity" => Float64(result.diversity),
        "n_oracle_calls" => Int(result.n_oracle_calls),
        "unique_molecules" => Int(result.unique_molecules),
        "oracle_call_breakdown" => result.oracle_call_breakdown,
        "provenance_summary" => result.provenance_summary,
        "artifact_paths" => result.artifact_paths,
        "diagnostics_summary" => result.diagnostics_summary,
    )
end

function make_he_config()
    return HierarchicalEditConfig(;
        horizon=8,
        topk_tracking=10,
        allow_crossover=true,
        min_exploration_per_operator=3,
        multi_child_min_reward_ratio=0.2,
        operator_prior_strength=4.0,
    )
end

function run_one(task::String,
                 arm::String,
                 seed::Int,
                 pretrain::Dict{String,Any};
                 budget::Int,
                 n_iters::Int,
                 batch_size::Int,
                 replay_ratio::Int,
                 boost_rounds::Int,
                 verbose::Bool)
    flags = arm_flags(arm)
    target_smi = get(TARGET_SMILES, task, nothing)
    Random.seed!(seed + abs(hash((task, arm))) % 10_000_000)
    he_enabled = Bool(flags[:use_hierarchical_edit])
    he_artifact_dir = he_enabled ? joinpath(OUTDIR, "artifacts", arm, task, "seed$(seed)") : nothing
    !isnothing(he_artifact_dir) && mkpath(he_artifact_dir)
    he_run_context = Dict{String,Any}(
        "step0_arm" => arm,
        "seed" => seed,
        "task" => task,
        "audit_mode" => get(ENV, "STEP0_MODE", "unknown"),
    )

    start_time = time()
    try
        result = run_smiles_pmo_task(task;
            budget=budget,
            pretrained_params=deepcopy(pretrain["pretrained_params"]),
            pretrained_states=deepcopy(pretrain["pretrained_states"]),
            vocab=pretrain["vocab"],
            policy_model=pretrain["policy_model"],
            training_mode=:tb,
            use_qgfn=Bool(flags[:use_qgfn]),
            use_boosting=Bool(flags[:use_boosting]),
            n_boost_rounds=boost_rounds,
            use_replay=true,
            replay_ratio=replay_ratio,
            batch_size=batch_size,
            n_iterations=n_iters,
            ga_per_step=Bool(flags[:ga_per_step]),
            ga_crossover=4,
            ga_mutation=4,
            track_frontier=true,
            target_smiles=target_smi,
            target_seed=!isnothing(target_smi),
            target_seed_augmentations=4,
            use_hierarchical_edit=he_enabled,
            he_warmup_episodes=he_enabled ? 2 : 0,
            he_episodes_per_segment=he_enabled ? 1 : 0,
            he_budget_fraction=0.15,
            he_config=he_enabled ? make_he_config() : HierarchicalEditConfig(),
            he_artifact_dir=he_artifact_dir,
            he_run_context=he_run_context,
            verbose=verbose,
        )
        elapsed = time() - start_time
        summary = pmo_result_summary(result)
        return Dict{String,Any}(
            "status" => "ok",
            "task" => task,
            "arm" => arm,
            "seed" => seed,
            "budget" => budget,
            "n_iterations" => n_iters,
            "batch_size" => batch_size,
            "replay_ratio" => replay_ratio,
            "boost_rounds" => boost_rounds,
            "flags" => Dict(String(k)=>v for (k, v) in flags),
            "elapsed_sec" => elapsed,
            "result_summary" => summary,
            "raw_result" => result,
            "error" => nothing,
        )
    catch e
        elapsed = time() - start_time
        bt = catch_backtrace()
        return Dict{String,Any}(
            "status" => "failed",
            "task" => task,
            "arm" => arm,
            "seed" => seed,
            "budget" => budget,
            "n_iterations" => n_iters,
            "batch_size" => batch_size,
            "replay_ratio" => replay_ratio,
            "boost_rounds" => boost_rounds,
            "flags" => Dict(String(k)=>v for (k, v) in flags),
            "elapsed_sec" => elapsed,
            "result_summary" => nothing,
            "raw_result" => nothing,
            "error" => sprint(showerror, e, bt),
        )
    end
end

function mean_or_nan(xs::Vector{Float64})
    isempty(xs) && return NaN
    return mean(xs)
end

function std_or_zero(xs::Vector{Float64})
    length(xs) <= 1 && return 0.0
    return std(xs)
end

function aggregate_rows(rows::Vector{Dict{String,Any}})
    groups = Dict{Tuple{String,String},Vector{Dict{String,Any}}}()
    for row in rows
        row["status"] == "ok" || continue
        key = (String(row["task"]), String(row["arm"]))
        groups[key] = get(groups, key, Dict{String,Any}[])
        push!(groups[key], row)
    end
    out = Dict{String,Any}[]
    for ((task, arm), vals) in groups
        aucs = [Float64(v["result_summary"]["auc_top10"]) for v in vals]
        top1 = [Float64(v["result_summary"]["top1"]) for v in vals]
        top10 = [Float64(v["result_summary"]["top10_mean"]) for v in vals]
        diversity = [Float64(v["result_summary"]["diversity"]) for v in vals]
        calls = [Float64(v["result_summary"]["n_oracle_calls"]) for v in vals]
        uniques = [Float64(v["result_summary"]["unique_molecules"]) for v in vals]
        elapsed = [Float64(v["elapsed_sec"]) for v in vals]
        push!(out, Dict{String,Any}(
            "task" => task,
            "arm" => arm,
            "n_ok" => length(vals),
            "auc_mean" => mean_or_nan(aucs),
            "auc_std" => std_or_zero(aucs),
            "top1_mean" => mean_or_nan(top1),
            "top10_mean" => mean_or_nan(top10),
            "diversity_mean" => mean_or_nan(diversity),
            "calls_mean" => mean_or_nan(calls),
            "unique_mean" => mean_or_nan(uniques),
            "elapsed_mean" => mean_or_nan(elapsed),
        ))
    end
    sort!(out, by = r -> (String(r["task"]), String(r["arm"])))
    return out
end

function overall_by_arm(agg::Vector{Dict{String,Any}})
    groups = Dict{String,Vector{Dict{String,Any}}}()
    for row in agg
        arm = String(row["arm"])
        groups[arm] = get(groups, arm, Dict{String,Any}[])
        push!(groups[arm], row)
    end
    out = Dict{String,Any}[]
    for (arm, rows) in groups
        aucs = [Float64(r["auc_mean"]) for r in rows if isfinite(Float64(r["auc_mean"]))]
        top10s = [Float64(r["top10_mean"]) for r in rows if isfinite(Float64(r["top10_mean"]))]
        push!(out, Dict{String,Any}(
            "arm" => arm,
            "tasks" => [String(r["task"]) for r in rows],
            "mean_auc" => mean_or_nan(aucs),
            "mean_top10" => mean_or_nan(top10s),
        ))
    end
    sort!(out, by = r -> -Float64(r["mean_auc"]))
    return out
end

function paired_deltas(rows::Vector{Dict{String,Any}}, baseline_arm::String)
    ok_rows = [r for r in rows if r["status"] == "ok"]
    index = Dict{Tuple{String,Int,String},Dict{String,Any}}()
    for r in ok_rows
        index[(String(r["task"]), Int(r["seed"]), String(r["arm"]))] = r
    end
    deltas = Dict{String,Any}[]
    for r in ok_rows
        arm = String(r["arm"])
        arm == baseline_arm && continue
        key = (String(r["task"]), Int(r["seed"]), baseline_arm)
        haskey(index, key) || continue
        base = index[key]
        auc = Float64(r["result_summary"]["auc_top10"])
        base_auc = Float64(base["result_summary"]["auc_top10"])
        push!(deltas, Dict{String,Any}(
            "task" => String(r["task"]),
            "seed" => Int(r["seed"]),
            "arm" => arm,
            "baseline_arm" => baseline_arm,
            "auc" => auc,
            "baseline_auc" => base_auc,
            "delta_auc" => auc - base_auc,
            "relative_delta_pct" => base_auc == 0.0 ? NaN : (auc / base_auc - 1.0) * 100.0,
        ))
    end
    return deltas
end

function gate_decision(agg::Vector{Dict{String,Any}}, rows::Vector{Dict{String,Any}})
    overall = Dict(String(r["arm"]) => Float64(r["mean_auc"]) for r in overall_by_arm(agg))
    failures = [r for r in rows if r["status"] != "ok"]
    heuristic = get(overall, "heuristic_he", NaN)
    tb_only = get(overall, "tb_only", NaN)
    qgfn = get(overall, "tb_qgfn", NaN)
    boosting = get(overall, "tb_boosting", NaN)
    qgfn_boosting = get(overall, "tb_qgfn_boosting", NaN)
    prior_o2 = 0.2819883389770848

    decisions = String[]
    if !isempty(failures)
        push!(decisions, "Some arms failed; treat audit as incomplete until failures are explained.")
    end
    if isfinite(heuristic) && ((isfinite(qgfn) && qgfn >= heuristic) || (isfinite(boosting) && boosting >= heuristic))
        push!(decisions, "QGFN or Boosting matches/exceeds heuristic HE: downgrade Option-Flow mainline to complementary scheduler/router unless integrated O3 adds value.")
    end
    if isfinite(qgfn_boosting) && isfinite(heuristic) && qgfn_boosting >= heuristic && qgfn_boosting > prior_o2
        push!(decisions, "QGFN+Boosting exceeds prior Option-Flow O2 reference and matches/exceeds heuristic: prioritize baseline pipeline optimization before Option-Flow integration.")
    end
    if isfinite(heuristic) && isfinite(tb_only) && heuristic <= tb_only
        push!(decisions, "Heuristic HE is weak/nonpositive versus TB-only: do not build Option-Flow only on HE; reconsider option substrate.")
    end
    if isfinite(heuristic) && all(isfinite.([tb_only, qgfn, boosting, qgfn_boosting])) && heuristic > maximum([tb_only, qgfn, boosting, qgfn_boosting])
        push!(decisions, "Heuristic HE remains strongest among audited baselines: Option-Flow must beat heuristic in integrated O3 or be downgraded.")
    end
    if isempty(decisions)
        push!(decisions, "No decisive go/no-go branch triggered; treat as inconclusive and inspect per-task paired deltas.")
    end

    return Dict{String,Any}(
        "overall_auc_by_arm" => overall,
        "prior_option_flow_o2_reference_auc" => prior_o2,
        "decisions" => decisions,
        "failed_runs" => length(failures),
        "status" => isempty(failures) ? "complete" : "incomplete_with_failures",
    )
end

function print_summary(agg, overall, gate)
    println("\n", "="^90)
    println("STEP 0 SUMMARY")
    println("="^90)
    println(rpad("Task", 26), rpad("Arm", 22), rpad("n", 5), rpad("AUC mean", 12), rpad("AUC std", 12), rpad("Top1", 10), rpad("Top10", 10), rpad("Calls", 10))
    println("-"^90)
    for row in agg
        @printf("%-26s%-22s%-5d%-12.6f%-12.6f%-10.4f%-10.4f%-10.1f\n",
            row["task"], row["arm"], row["n_ok"], row["auc_mean"], row["auc_std"], row["top1_mean"], row["top10_mean"], row["calls_mean"])
    end
    println("\nOVERALL BY ARM")
    for row in overall
        @printf("%-22s mean_auc=%.6f mean_top10=%.6f tasks=%s\n", row["arm"], row["mean_auc"], row["mean_top10"], join(row["tasks"], ","))
    end
    println("\nGO/NO-GO DECISIONS")
    for d in gate["decisions"]
        println("- ", d)
    end
end

function main()
    mode = strip(get(ENV, "STEP0_MODE", "smoke"))
    default_tasks = mode == "smoke" ? ["qed"] : ["qed", "drd2", "celecoxib_rediscovery"]
    tasks = parse_csv_strings("STEP0_TASKS", default_tasks)
    arms = parse_csv_strings("STEP0_ARMS", ["tb_only", "tb_qgfn", "tb_boosting", "tb_qgfn_boosting", "heuristic_he"])
    seeds = parse_csv_ints("STEP0_SEEDS", mode == "smoke" ? [17] : [17, 23])
    budget = parse_env_int("STEP0_BUDGET", mode == "smoke" ? 192 : 1000)
    n_iters = parse_env_int("STEP0_ITERS", mode == "smoke" ? 3 : 8)
    batch_size = parse_env_int("STEP0_BATCH", mode == "smoke" ? 8 : 16)
    replay_ratio = parse_env_int("STEP0_REPLAY_RATIO", 2)
    boost_rounds = parse_env_int("STEP0_BOOST_ROUNDS", 3)
    verbose = lowercase(strip(get(ENV, "STEP0_VERBOSE", "true"))) in ["1", "true", "yes", "y"]

    invalid = [a for a in arms if !(a in CONFIRMATORY_ARMS || a in EXPLORATORY_ARMS)]
    isempty(invalid) || error("Unknown Step0 arms: $(invalid)")

    logmsg("Step0 baseline debt audit mode=$(mode) tasks=$(tasks) arms=$(arms) seeds=$(seeds) budget=$(budget) iters=$(n_iters) batch=$(batch_size) boost_rounds=$(boost_rounds)")
    pretrain = load_pretrain()
    logmsg("Loaded checkpoint $(pretrain["checkpoint_path"]) vocab_size=$(pretrain["vocab_size"])")

    rows = Dict{String,Any}[]
    partial_path = joinpath(OUTDIR, "step0_$(mode)_partial_results.jls")
    for seed in seeds, task in tasks, arm in arms
        logmsg("RUN start task=$(task) arm=$(arm) seed=$(seed)")
        row = run_one(task, arm, seed, pretrain;
            budget=budget,
            n_iters=n_iters,
            batch_size=batch_size,
            replay_ratio=replay_ratio,
            boost_rounds=boost_rounds,
            verbose=verbose)
        push!(rows, row)
        partial_agg = aggregate_rows(rows)
        partial_overall = overall_by_arm(partial_agg)
        partial_gate = gate_decision(partial_agg, rows)
        serialize(partial_path, Dict{String,Any}(
            "created_at" => string(now()),
            "mode" => mode,
            "partial" => true,
            "tasks" => tasks,
            "arms" => arms,
            "seeds" => seeds,
            "budget" => budget,
            "n_iterations" => n_iters,
            "batch_size" => batch_size,
            "replay_ratio" => replay_ratio,
            "boost_rounds" => boost_rounds,
            "rows" => rows,
            "aggregate_rows" => partial_agg,
            "overall_by_arm" => partial_overall,
            "gate" => partial_gate,
        ))
        if row["status"] == "ok"
            s = row["result_summary"]
            logmsg("RUN ok task=$(task) arm=$(arm) seed=$(seed) auc=$(round(s["auc_top10"], digits=6)) top1=$(round(s["top1"], digits=4)) top10=$(round(s["top10_mean"], digits=4)) calls=$(s["n_oracle_calls"]) unique=$(s["unique_molecules"]) elapsed=$(round(row["elapsed_sec"], digits=1))")
        else
            logmsg("RUN failed task=$(task) arm=$(arm) seed=$(seed) elapsed=$(round(row["elapsed_sec"], digits=1))")
            println(row["error"])
            flush(stdout)
        end
    end

    agg = aggregate_rows(rows)
    overall = overall_by_arm(agg)
    delta_vs_he = paired_deltas(rows, "heuristic_he")
    delta_vs_tb = paired_deltas(rows, "tb_only")
    gate = gate_decision(agg, rows)
    print_summary(agg, overall, gate)

    bundle = Dict{String,Any}(
        "created_at" => string(now()),
        "authoritative_date_note" => "User supplied authoritative date/time: Friday, 2026-06-19 01:14 EDT",
        "mode" => mode,
        "tasks" => tasks,
        "arms" => arms,
        "seeds" => seeds,
        "budget" => budget,
        "n_iterations" => n_iters,
        "batch_size" => batch_size,
        "replay_ratio" => replay_ratio,
        "boost_rounds" => boost_rounds,
        "rows" => rows,
        "aggregate_rows" => agg,
        "overall_by_arm" => overall,
        "paired_delta_vs_heuristic_he" => delta_vs_he,
        "paired_delta_vs_tb_only" => delta_vs_tb,
        "gate" => gate,
        "limitations" => [
            "Step 0 is a baseline-debt audit, not a SOTA benchmark.",
            "Two seeds and 3 tasks are exploratory; strong claims require broader tasks/seeds/budgets.",
            "Prior Option-Flow O2 is reference-only because it used a standalone HE-option loop, not this integrated PMO pipeline.",
        ],
    )

    out = joinpath(OUTDIR, "step0_$(mode)_results.jls")
    latest = joinpath(OUTDIR, "step0_latest_results.jls")
    serialize(out, bundle)
    serialize(latest, bundle)
    logmsg("Saved results: $(abspath(out))")
    logmsg("Saved latest: $(abspath(latest))")

    if mode == "smoke"
        failed_confirmatory = [r for r in rows if r["arm"] in CONFIRMATORY_ARMS && r["status"] != "ok"]
        if !isempty(failed_confirmatory)
            error("Smoke failed for confirmatory arms: $([(r["task"], r["arm"], r["seed"]) for r in failed_confirmatory])")
        end
    end
end

main()
