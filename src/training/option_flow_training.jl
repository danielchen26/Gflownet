# Option-Flow v0 training loop
#
# This file intentionally uses a small manual backprop implementation instead of
# Zygote. The POC should stay lightweight and avoid loading the full molecular
# stack / PythonCall / RDKit just to validate finite-catalog option flow.

using Random
using Statistics

struct OptionFlowTrainingConfig
    n_epochs::Int
    learning_rate::Float64
    hidden_dim::Int
    second_hidden_dim::Int
    validation_fraction::Float64
    seed::Int
    verbose::Bool
end

function OptionFlowTrainingConfig(; n_epochs::Int=250,
                                   learning_rate::Float64=0.01,
                                   hidden_dim::Int=128,
                                   second_hidden_dim::Int=64,
                                   validation_fraction::Float64=0.25,
                                   seed::Int=17,
                                   verbose::Bool=false)
    n_epochs > 0 || throw(ArgumentError("n_epochs must be positive"))
    learning_rate > 0 || throw(ArgumentError("learning_rate must be positive"))
    return OptionFlowTrainingConfig(n_epochs, learning_rate, hidden_dim, second_hidden_dim, validation_fraction, seed, verbose)
end

function _zero_like_option_flow_params(params)
    return (
        W1 = zeros(Float32, size(params.W1)),
        b1 = zeros(Float32, size(params.b1)),
        W2 = zeros(Float32, size(params.W2)),
        b2 = zeros(Float32, size(params.b2)),
        W3 = zeros(Float32, size(params.W3)),
        b3 = zeros(Float32, size(params.b3)),
    )
end

function _add_option_flow_grads(a, b)
    return (
        W1 = a.W1 .+ b.W1,
        b1 = a.b1 .+ b.b1,
        W2 = a.W2 .+ b.W2,
        b2 = a.b2 .+ b.b2,
        W3 = a.W3 .+ b.W3,
        b3 = a.b3 .+ b.b3,
    )
end

function _scale_option_flow_grads(g, scale::Float32)
    return (
        W1 = scale .* g.W1,
        b1 = scale .* g.b1,
        W2 = scale .* g.W2,
        b2 = scale .* g.b2,
        W3 = scale .* g.W3,
        b3 = scale .* g.b3,
    )
end

function _sgd_update_option_flow_params(params, grads, lr::Float32)
    return (
        W1 = params.W1 .- lr .* grads.W1,
        b1 = params.b1 .- lr .* grads.b1,
        W2 = params.W2 .- lr .* grads.W2,
        b2 = params.b2 .- lr .* grads.b2,
        W3 = params.W3 .- lr .* grads.W3,
        b3 = params.b3 .- lr .* grads.b3,
    )
end

function _catalog_loss_and_grad(params, catalog::OptionFlowCatalog)
    X = option_flow_input_matrix(catalog)                         # input_dim × n
    Z1 = params.W1 * X .+ params.b1                               # h1 × n
    H1 = max.(Z1, 0.0f0)
    Z2 = params.W2 * H1 .+ params.b2                              # h2 × n
    H2 = max.(Z2, 0.0f0)
    logits = vec(params.W3 * H2 .+ params.b3)                     # n
    logp = option_flow_log_probs_from_logits(logits)
    probs = exp.(logp)
    target = catalog.target_probs
    loss = -sum(target .* logp)

    dlogits = reshape(probs .- target, 1, :)                      # 1 × n
    gW3 = dlogits * H2'
    gb3 = reshape(sum(dlogits; dims=2), 1, 1)

    dH2 = params.W3' * dlogits
    dZ2 = dH2 .* Float32.(Z2 .> 0.0f0)
    gW2 = dZ2 * H1'
    gb2 = sum(dZ2; dims=2)

    dH1 = params.W2' * dZ2
    dZ1 = dH1 .* Float32.(Z1 .> 0.0f0)
    gW1 = dZ1 * X'
    gb1 = sum(dZ1; dims=2)

    grads = (
        W1 = Float32.(gW1),
        b1 = Float32.(gb1),
        W2 = Float32.(gW2),
        b2 = Float32.(gb2),
        W3 = Float32.(gW3),
        b3 = Float32.(gb3),
    )
    return loss, grads
end

function _mean_loss_and_grad(params, catalogs::Vector{OptionFlowCatalog})
    isempty(catalogs) && throw(ArgumentError("cannot compute gradient on empty catalogs"))
    total_loss = 0.0f0
    total_grad = _zero_like_option_flow_params(params)
    for catalog in catalogs
        loss, grad = _catalog_loss_and_grad(params, catalog)
        total_loss += Float32(loss)
        total_grad = _add_option_flow_grads(total_grad, grad)
    end
    scale = 1.0f0 / Float32(length(catalogs))
    return total_loss * scale, _scale_option_flow_grads(total_grad, scale)
end

function train_option_flow_model(catalogs::Vector{OptionFlowCatalog};
                                 config::OptionFlowTrainingConfig=OptionFlowTrainingConfig())
    isempty(catalogs) && throw(ArgumentError("cannot train Option-Flow model on empty catalogs"))
    all(validate_option_flow_catalog(c) for c in catalogs) || throw(ArgumentError("invalid Option-Flow catalog detected"))
    input_dim = option_flow_input_dim(catalogs[1])
    all(option_flow_input_dim(c) == input_dim for c in catalogs) || throw(ArgumentError("all catalogs must share input dimension for v0 training"))

    train_catalogs, val_catalogs = grouped_split_option_flow_catalogs(catalogs;
        validation_fraction=config.validation_fraction,
        seed=config.seed)
    isempty(train_catalogs) && throw(ArgumentError("training split is empty"))

    model_config = create_option_flow_mlp(input_dim;
        hidden_dim=config.hidden_dim,
        second_hidden_dim=config.second_hidden_dim)
    rng = MersenneTwister(config.seed)
    params = init_option_flow_params(model_config; rng=rng)
    history = Vector{Dict{String,Any}}()
    lr = Float32(config.learning_rate)

    for epoch in 1:config.n_epochs
        loss_value, grads = _mean_loss_and_grad(params, train_catalogs)
        params = _sgd_update_option_flow_params(params, grads, lr)
        if epoch == 1 || epoch == config.n_epochs || epoch % max(1, config.n_epochs ÷ 10) == 0
            train_metrics = evaluate_option_flow_model(params, train_catalogs)
            val_metrics = evaluate_option_flow_model(params, val_catalogs)
            row = Dict{String,Any}(
                "epoch" => epoch,
                "train_loss" => Float64(loss_value),
                "train_mean_ce" => train_metrics["mean_ce"],
                "train_ce_vs_uniform" => train_metrics["mean_ce_vs_uniform"],
                "val_mean_ce" => val_metrics["mean_ce"],
                "val_ce_vs_uniform" => val_metrics["mean_ce_vs_uniform"],
                "val_top_quartile_lift" => val_metrics["mean_top_quartile_lift"],
                "val_rank_correlation" => val_metrics["mean_rank_correlation"],
            )
            push!(history, row)
            if config.verbose
                println("epoch=$(epoch) train_ce=$(round(train_metrics["mean_ce"], digits=4)) val_ce=$(round(val_metrics["mean_ce"], digits=4)) val_lift=$(round(val_metrics["mean_top_quartile_lift"], digits=4))")
            end
        end
    end

    return Dict{String,Any}(
        "model_config" => model_config,
        "params" => params,
        "train_catalogs" => train_catalogs,
        "val_catalogs" => val_catalogs,
        "history" => history,
        "train_metrics" => evaluate_option_flow_model(params, train_catalogs),
        "val_metrics" => evaluate_option_flow_model(params, val_catalogs),
        "param_count" => option_flow_param_count(params),
    )
end
