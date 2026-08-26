#!/usr/bin/env julia

using Dates
using Serialization
using Statistics
using Random

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "src", "training", "option_flow_dataset.jl"))
include(joinpath(ROOT, "src", "training", "option_flow_model.jl"))
include(joinpath(ROOT, "src", "training", "option_flow_loss.jl"))
include(joinpath(ROOT, "src", "training", "option_flow_training.jl"))
include(joinpath(ROOT, "src", "training", "option_flow_real_catalog.jl"))

function logmsg(msg)
    println("[", Dates.format(now(), "HH:MM:SS"), "] ", msg)
    flush(stdout)
end

function parse_env_int(name::String, default::Int)
    value = get(ENV, name, "")
    isempty(value) && return default
    return parse(Int, value)
end

function parse_env_float(name::String, default::Float64)
    value = get(ENV, name, "")
    isempty(value) && return default
    return parse(Float64, value)
end

function parse_env_symbols(name::String, defaults::Vector{Symbol})
    value = get(ENV, name, "")
    isempty(value) && return defaults
    return Symbol.(strip.(split(value, ",")))
end

function parse_env_ints(name::String, defaults::Vector{Int})
    value = get(ENV, name, "")
    isempty(value) && return defaults
    return parse.(Int, strip.(split(value, ",")))
end

function default_real_roots()
    return String[
        joinpath(ROOT, "checkpoints", "final_theory_direct_test", "artifacts", "final_theory_v1"),
        joinpath(ROOT, "checkpoints", "level3_shape_then_tb", "artifacts", "heuristic_shape_then_tb"),
        joinpath(ROOT, "checkpoints", "level3_shape_then_tb", "artifacts", "learned_shape_then_tb"),
        "/Users/tianchichen/Documents/GitHub/Gflownet/checkpoints/truth_sprint_stage_b_f015_truth_tasksharded",
    ]
end

function parse_roots()
    value = get(ENV, "OPTION_FLOW_REAL_ROOTS", "")
    isempty(value) && return default_real_roots()
    return strip.(split(value, ":"))
end

function _finite_values(xs)
    vals = Float64[]
    for x in xs
        y = Float64(x)
        isfinite(y) && push!(vals, y)
    end
    return vals
end

function _mean_std(xs)
    vals = _finite_values(xs)
    isempty(vals) && return Dict("mean" => NaN, "std" => NaN, "n" => 0)
    return Dict("mean" => mean(vals), "std" => length(vals) > 1 ? std(vals) : 0.0, "n" => length(vals))
end

function _nested_get(d::Dict{String,Any}, path::Vector{String}, default=NaN)
    cur = d
    for (i, key) in enumerate(path)
        if i == length(path)
            return get(cur, key, default)
        end
        nxt = get(cur, key, nothing)
        nxt isa Dict{String,Any} || return default
        cur = nxt
    end
    return default
end

function aggregate_seed_results(seed_results::Vector{Dict{String,Any}})
    isempty(seed_results) && return Dict{String,Any}("n_seeds" => 0)
    metric_paths = Dict(
        "val_ce" => ["val_metrics", "mean_ce"],
        "val_uniform_ce" => ["val_metrics", "mean_uniform_ce"],
        "val_ce_gain" => ["val_metrics", "mean_ce_vs_uniform"],
        "val_top_quartile_lift" => ["val_metrics", "mean_top_quartile_lift"],
        "val_rank_correlation" => ["val_metrics", "mean_rank_correlation"],
        "val_entropy" => ["val_metrics", "mean_entropy"],
        "val_expected_utility" => ["val_metrics", "mean_expected_utility"],
        "val_uniform_expected_utility" => ["val_metrics", "mean_uniform_expected_utility"],
        "val_expected_utility_lift" => ["val_metrics", "mean_expected_utility_lift"],
        "val_expected_utility_lift_fraction" => ["val_metrics", "mean_expected_utility_lift_fraction"],
        "config_prior_ce" => ["config_prior", "mean_ce"],
        "config_prior_expected_utility_lift" => ["config_prior", "mean_expected_utility_lift"],
        "source_prior_ce" => ["source_prior", "mean_ce"],
        "source_prior_expected_utility_lift" => ["source_prior", "mean_expected_utility_lift"],
        "greedy_model_expected_utility" => ["val_metrics", "greedy_model", "mean_expected_utility"],
        "oracle_greedy_expected_utility" => ["val_metrics", "oracle_greedy", "mean_expected_utility"],
    )
    agg = Dict{String,Any}("n_seeds" => length(seed_results))
    for (name, path) in metric_paths
        agg[name] = _mean_std([_nested_get(sr, path, NaN) for sr in seed_results])
    end
    gates = [Bool(get(sr["gate"], "pass", false)) for sr in seed_results]
    agg["gate_pass_count"] = count(identity, gates)
    agg["gate_pass_fraction"] = mean(Float64.(gates))
    agg["all_seeds_pass"] = all(gates)
    agg["majority_seeds_pass"] = count(identity, gates) >= ceil(Int, length(gates) / 2)
    return agg
end

const STRUCTURAL_TASKS = Set(["celecoxib_rediscovery", "albuterol_similarity"])

function structural_task_signal(seed_results::Vector{Dict{String,Any}})
    task_rows = Dict{String,Vector{Dict{String,Any}}}()
    for sr in seed_results
        tb = get(sr, "task_breakdown", Dict{String,Any}())
        for (task, metrics_any) in tb
            task in STRUCTURAL_TASKS || continue
            metrics = metrics_any::Dict{String,Any}
            push!(get!(task_rows, task, Dict{String,Any}[]), metrics)
        end
    end
    out = Dict{String,Any}()
    for (task, rows) in task_rows
        ce_gain = _mean_std([get(r, "mean_ce_vs_uniform", NaN) for r in rows])
        util_lift = _mean_std([get(r, "mean_expected_utility_lift", NaN) for r in rows])
        top_lift = _mean_std([get(r, "mean_top_quartile_lift", NaN) for r in rows])
        out[task] = Dict{String,Any}(
            "ce_gain" => ce_gain,
            "utility_lift" => util_lift,
            "top_quartile_lift" => top_lift,
            "positive" => _num(ce_gain["mean"]; default=NaN) > 0.0 && _num(util_lift["mean"]; default=NaN) > 0.0,
        )
    end
    return out
end

function run_one_experiment(rows::Vector{Dict{String,Any}};
                            grouping::Symbol,
                            feature_mode::Symbol,
                            utility_key::Symbol,
                            seeds::Vector{Int},
                            n_epochs::Int,
                            lr::Float64,
                            hidden_dim::Int,
                            second_hidden_dim::Int,
                            validation_fraction::Float64,
                            min_candidates::Int,
                            max_candidates::Int)
    logmsg("Build catalogs grouping=$(grouping) feature_mode=$(feature_mode) utility=$(utility_key)")
    catalogs, enc, build_stats = build_summary_proxy_catalogs(rows;
        grouping=grouping,
        feature_mode=feature_mode,
        utility_key=utility_key,
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        seed=first(seeds),
    )
    headline_catalogs = filter_real_headline_catalogs(catalogs)
    catalog_stats = summarize_real_catalog_collection(headline_catalogs)
    logmsg("Catalogs raw=$(length(catalogs)) headline=$(length(headline_catalogs)) candidates=$(get(catalog_stats, "n_candidates", 0))")

    exp_result = Dict{String,Any}(
        "grouping" => string(grouping),
        "feature_mode" => string(feature_mode),
        "utility_key" => string(utility_key),
        "encoding" => Dict(
            "task_names" => enc.task_names,
            "phases" => enc.phases,
            "config_names" => enc.config_names,
            "source_families" => enc.source_families,
            "feature_mode" => string(enc.feature_mode),
            "grouping" => string(enc.grouping),
        ),
        "build_stats" => build_stats,
        "catalog_stats" => catalog_stats,
        "seed_results" => Dict{String,Any}[],
    )

    if length(headline_catalogs) < 2
        exp_result["status"] = "SKIPPED_TOO_FEW_HEADLINE_CATALOGS"
        return exp_result
    end

    for seed in seeds
        logmsg("Train seed=$(seed) grouping=$(grouping) feature=$(feature_mode)")
        result = train_option_flow_model(headline_catalogs; config=OptionFlowTrainingConfig(
            n_epochs=n_epochs,
            learning_rate=lr,
            hidden_dim=hidden_dim,
            second_hidden_dim=second_hidden_dim,
            validation_fraction=validation_fraction,
            seed=seed,
            verbose=false,
        ))
        val_metrics = evaluate_real_option_flow_model(result["params"], result["val_catalogs"])
        train_metrics = evaluate_real_option_flow_model(result["params"], result["train_catalogs"])
        gate = e1_summary_proxy_gate(val_metrics)
        config_prior = evaluate_metadata_prior_baseline(result["train_catalogs"], result["val_catalogs"]; metadata_key="config_name")
        source_prior = evaluate_metadata_prior_baseline(result["train_catalogs"], result["val_catalogs"]; metadata_key="source_family")
        uniform_val = uniform_policy_metrics(result["val_catalogs"])
        task_breakdown = real_catalog_task_breakdown(result["params"], result["val_catalogs"])
        logmsg("Seed $(seed): val_ce_gain=$(round(val_metrics["mean_ce_vs_uniform"], digits=4)) util_lift=$(round(val_metrics["mean_expected_utility_lift"], digits=4)) top_lift=$(round(val_metrics["mean_top_quartile_lift"], digits=4)) rank=$(round(val_metrics["mean_rank_correlation"], digits=4)) gate=$(gate["pass"])")
        push!(exp_result["seed_results"], Dict{String,Any}(
            "seed" => seed,
            "train_catalog_count" => length(result["train_catalogs"]),
            "val_catalog_count" => length(result["val_catalogs"]),
            "param_count" => result["param_count"],
            "history" => result["history"],
            "train_metrics" => train_metrics,
            "val_metrics" => val_metrics,
            "gate" => gate,
            "config_prior" => config_prior,
            "source_prior" => source_prior,
            "uniform_val" => uniform_val,
            "task_breakdown" => task_breakdown,
        ))
    end
    exp_result["aggregate"] = aggregate_seed_results(exp_result["seed_results"])
    exp_result["structural_task_signal"] = structural_task_signal(exp_result["seed_results"])
    exp_result["status"] = "DONE"
    return exp_result
end

function choose_headline_experiment(experiments::Vector{Dict{String,Any}})
    candidates = [e for e in experiments if get(e, "status", "") == "DONE" && get(e, "feature_mode", "") == "leak_free" && get(e, "grouping", "") == "task_phase_budget100"]
    !isempty(candidates) && return first(candidates)
    candidates = [e for e in experiments if get(e, "status", "") == "DONE" && get(e, "feature_mode", "") == "leak_free"]
    !isempty(candidates) && return first(candidates)
    return isempty(experiments) ? Dict{String,Any}() : first(experiments)
end

function main()
    roots = parse_roots()
    output_dir = get(ENV, "OPTION_FLOW_REAL_OUTPUT_DIR", joinpath(ROOT, "checkpoints", "option_flow_real_evidence_poc"))
    mkpath(output_dir)

    groupings = parse_env_symbols("OPTION_FLOW_REAL_GROUPINGS", [:task_phase_budget100, :task_phase_b100_t05, :task_phase])
    feature_modes = parse_env_symbols("OPTION_FLOW_REAL_FEATURE_MODES", [:leak_free, :policy_only, :leaky_upper])
    utility_key = Symbol(get(ENV, "OPTION_FLOW_REAL_UTILITY", "frontier_gain_sum"))
    seeds = parse_env_ints("OPTION_FLOW_REAL_SEEDS", [17, 23, 31])
    n_epochs = parse_env_int("OPTION_FLOW_REAL_EPOCHS", 220)
    lr = parse_env_float("OPTION_FLOW_REAL_LR", 0.012)
    hidden_dim = parse_env_int("OPTION_FLOW_REAL_HIDDEN", 96)
    second_hidden_dim = parse_env_int("OPTION_FLOW_REAL_HIDDEN2", 48)
    validation_fraction = parse_env_float("OPTION_FLOW_REAL_VAL_FRACTION", 0.25)
    min_candidates = parse_env_int("OPTION_FLOW_REAL_MIN_CANDIDATES", 3)
    max_candidates = parse_env_int("OPTION_FLOW_REAL_MAX_CANDIDATES", 12)

    logmsg("Option-Flow real evidence POC roots=$(roots)")
    audit = option_flow_real_artifact_audit(roots)
    logmsg("Audit: summaries=$(audit["summary_files"]) episodes=$(audit["episode_count"]) repeated_task_snapshot_groups=$(audit["repeated_task_snapshot_groups"])")
    logmsg("Raw diagnostics lightweight=$(audit["raw_diagnostics_status"]["sample_deserializes"]) raw trajectory lightweight=$(audit["raw_trajectory_status"]["sample_deserializes"])")

    rows, files, errors = load_he_summary_rows(roots)
    isempty(errors) || logmsg("Summary load errors: $(errors)")
    isempty(rows) && error("No summary rows loaded from roots")

    typed_path_stats = Dict{String,Any}("requested" => (:typed_path in feature_modes), "loaded" => 0)
    if :typed_path in feature_modes
        logmsg("Typed-path mode requested; loading lightweight raw-artifact stub")
        include(joinpath(@__DIR__, "support", "option_flow_raw_stub.jl"))
        rows, typed_path_stats = augment_rows_with_typed_path_features(rows)
        logmsg("Typed-path rows loaded=$(typed_path_stats["loaded"])/$(typed_path_stats["rows"]) missing=$(typed_path_stats["missing"]) errors=$(length(typed_path_stats["errors"]))")
    end

    experiments = Dict{String,Any}[]
    for grouping in groupings
        for feature_mode in feature_modes
            push!(experiments, run_one_experiment(rows;
                grouping=grouping,
                feature_mode=feature_mode,
                utility_key=utility_key,
                seeds=seeds,
                n_epochs=n_epochs,
                lr=lr,
                hidden_dim=hidden_dim,
                second_hidden_dim=second_hidden_dim,
                validation_fraction=validation_fraction,
                min_candidates=min_candidates,
                max_candidates=max_candidates,
            ))
        end
    end

    headline = choose_headline_experiment(experiments)
    headline_agg = get(headline, "aggregate", Dict{String,Any}())
    e1_verdict = if get(headline, "status", "") != "DONE"
        "E1_BLOCKED_OR_SKIPPED"
    elseif Bool(get(headline_agg, "majority_seeds_pass", false))
        "E1_SUMMARY_PROXY_SIGNAL_PRESENT"
    else
        "E1_SUMMARY_PROXY_SIGNAL_WEAK_OR_ABSENT"
    end
    structural = get(headline, "structural_task_signal", Dict{String,Any}())
    structural_positive = any(Bool(get(v, "positive", false)) for v in values(structural))

    bundle = Dict{String,Any}(
        "created_at" => string(now()),
        "authoritative_date_note" => "User supplied authoritative date/time: Thursday, 2026-06-11 16:06 EDT",
        "evidence_level" => "E1_summary_proxy",
        "audit" => audit,
        "loaded_summary_files" => files,
        "summary_load_errors" => errors,
        "typed_path_stats" => typed_path_stats,
        "runner_config" => Dict(
            "groupings" => string.(groupings),
            "feature_modes" => string.(feature_modes),
            "utility_key" => string(utility_key),
            "seeds" => seeds,
            "n_epochs" => n_epochs,
            "learning_rate" => lr,
            "hidden_dim" => hidden_dim,
            "second_hidden_dim" => second_hidden_dim,
            "validation_fraction" => validation_fraction,
            "min_candidates" => min_candidates,
            "max_candidates" => max_candidates,
        ),
        "experiments" => experiments,
        "headline" => headline,
        "e1_verdict" => e1_verdict,
        "structural_positive" => structural_positive,
        "limitations" => [
            "E1 uses matched summary proxy catalogs, not exact same-snapshot catalogs.",
            "Raw diagnostics/trajectory did not deserialize lightweight in the current audit unless the full GFlowNet type environment is available.",
            "Headline leak-free features exclude outcome fields; leaky_upper results are diagnostic only.",
            "This is not PMO/SOTA evidence; it is a real-artifact plausibility gate toward E2/E3.",
        ],
    )

    mode = get(ENV, "OPTION_FLOW_REAL_MODE", "e1_summary_proxy")
    out = joinpath(output_dir, "option_flow_real_evidence_poc_$(mode)_results.jls")
    latest = joinpath(output_dir, "option_flow_real_evidence_poc_latest_results.jls")
    serialize(out, bundle)
    serialize(latest, bundle)
    logmsg("VERDICT=$(e1_verdict) structural_positive=$(structural_positive)")
    logmsg("Saved results: $(abspath(out))")
    logmsg("Saved latest: $(abspath(latest))")
end

main()
