#!/usr/bin/env julia

using Dates
using Serialization
using Random
using Statistics

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "src", "training", "option_flow_dataset.jl"))
include(joinpath(ROOT, "src", "training", "option_flow_model.jl"))
include(joinpath(ROOT, "src", "training", "option_flow_loss.jl"))
include(joinpath(ROOT, "src", "training", "option_flow_training.jl"))

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

function main()
    mode = get(ENV, "OPTION_FLOW_POC_MODE", "smoke")
    seed = parse_env_int("OPTION_FLOW_SEED", 11)
    n_catalogs = parse_env_int("OPTION_FLOW_CATALOGS", mode == "smoke" ? 48 : 96)
    n_candidates = parse_env_int("OPTION_FLOW_CANDIDATES", 6)
    n_epochs = parse_env_int("OPTION_FLOW_EPOCHS", mode == "smoke" ? 120 : 250)
    lr = parse_env_float("OPTION_FLOW_LR", 0.015)
    output_dir = get(ENV, "OPTION_FLOW_OUTPUT_DIR", "checkpoints/option_flow_v0_poc")
    mkpath(output_dir)

    logmsg("Option-Flow v0 POC mode=$(mode) seed=$(seed) catalogs=$(n_catalogs) candidates=$(n_candidates)")
    catalogs = synthetic_option_flow_catalogs(
        n_catalogs=n_catalogs,
        n_candidates=n_candidates,
        state_dim=8,
        option_dim=10,
        seed=seed,
        tasks=["qed", "celecoxib_rediscovery", "drd2"],
    )
    stats = option_flow_catalog_stats(catalogs)
    logmsg("Catalog stats: $(stats)")

    result = train_option_flow_model(catalogs; config=OptionFlowTrainingConfig(
        n_epochs=n_epochs,
        learning_rate=lr,
        hidden_dim=96,
        second_hidden_dim=48,
        validation_fraction=0.25,
        seed=seed,
        verbose=true,
    ))

    train_metrics = result["train_metrics"]
    val_metrics = result["val_metrics"]
    logmsg("TRAIN metrics: $(train_metrics)")
    logmsg("VAL metrics: $(val_metrics)")

    verdict = if val_metrics["mean_ce"] < val_metrics["mean_uniform_ce"] &&
                 val_metrics["mean_top_quartile_mass"] > val_metrics["mean_uniform_top_quartile_mass"] &&
                 val_metrics["mean_rank_correlation"] > 0.1
        "POC_SIGNAL_PRESENT"
    else
        "POC_SIGNAL_WEAK_OR_ABSENT"
    end
    logmsg("VERDICT=$(verdict)")

    bundle = Dict{String,Any}(
        "mode" => mode,
        "seed" => seed,
        "catalog_stats" => stats,
        "train_metrics" => train_metrics,
        "val_metrics" => val_metrics,
        "history" => result["history"],
        "param_count" => result["param_count"],
        "verdict" => verdict,
    )
    out = joinpath(output_dir, "option_flow_v0_poc_$(mode)_seed$(seed)_results.jls")
    serialize(out, bundle)
    latest = joinpath(output_dir, "option_flow_v0_poc_latest_results.jls")
    serialize(latest, bundle)
    logmsg("Saved results: $(abspath(out))")
    logmsg("Saved latest alias: $(abspath(latest))")
end

main()
