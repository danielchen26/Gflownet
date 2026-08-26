# Parent Controller Training
#
# Batch 1B′ scope:
# - offline regression-style parent learning over truthful attempt-level targets
# - linear vs small MLP parent scorers
# - conservative disagreement-aware metrics for celecoxib-first recipe selection

using Random
using Statistics

struct ParentControllerTrainingConfig
    hidden_dim::Int
    learning_rate::Float64
    n_epochs::Int
    train_fraction::Float64
    min_records::Int
    model_kind::Symbol
    grad_clip_norm::Float64
    feature_mode::Symbol

    function ParentControllerTrainingConfig(;
        hidden_dim::Int=32,
        learning_rate::Float64=5e-3,
        n_epochs::Int=40,
        train_fraction::Float64=0.8,
        min_records::Int=12,
        model_kind::Symbol=:linear,
        grad_clip_norm::Float64=5.0,
        feature_mode::Symbol=:basic,
    )
        new(hidden_dim, learning_rate, n_epochs, train_fraction, min_records, model_kind, grad_clip_norm, feature_mode)
    end
end

_parent_regression_records(dataset::ParentControllerDataset) = [record for record in dataset.records if length(record.candidate_features) >= 2]

function _score_parent_chosen(controller, record::ParentDecisionRecord)
    return Float32(parent_candidate_score(controller, record.candidate_features[record.chosen_index]))
end

function _parent_pearson_or_zero(xs::Vector{Float64}, ys::Vector{Float64})
    length(xs) < 2 && return 0.0
    (std(xs) == 0 || std(ys) == 0) && return 0.0
    return cor(xs, ys)
end

_parent_mean_or_zero(xs) = isempty(xs) ? 0.0 : mean(Float64.(xs))

function evaluate_parent_controller(controller,
                                    dataset::ParentControllerDataset)
    records = _parent_regression_records(dataset)
    if isempty(records)
        return Dict(
            "n_records" => 0,
            "rmse" => 0.0,
            "score_target_correlation" => 0.0,
            "heuristic_target_correlation" => 0.0,
            "frontier_utility_correlation" => 0.0,
            "heuristic_frontier_utility_correlation" => 0.0,
            "enters_topk_correlation" => 0.0,
            "productive_degenerate_margin" => 0.0,
            "mean_chosen_rank" => 0.0,
            "mean_heuristic_chosen_rank" => 0.0,
            "disagreement_rate" => 0.0,
            "high_confidence_disagreement_rate" => 0.0,
            "ambiguous_disagreement_rate" => 0.0,
            "strong_heuristic_disagreement_rate" => 0.0,
            "heuristic_ambiguous_fraction" => 0.0,
            "mean_learned_margin" => 0.0,
            "mean_learned_advantage_vs_heuristic" => 0.0,
            "preserve_strong_heuristic_rate" => 0.0,
        )
    end

    preds = Float64[]
    targets = Float64[]
    heuristics = Float64[]
    frontier_targets = Float64[]
    enters_topk_targets = Float64[]
    productive_labels = Bool[]
    degenerate_labels = Bool[]
    chosen_ranks = Float64[]
    heuristic_ranks = Float64[]
    disagreement_flags = Bool[]
    high_conf_disagreement_flags = Bool[]
    ambiguous_disagreement_flags = Bool[]
    strong_heuristic_disagreement_flags = Bool[]
    strong_heuristic_preserved_flags = Bool[]
    heuristic_ambiguous_flags = Bool[]
    learned_margins = Float64[]
    learned_advantages = Float64[]

    for record in records
        pred = Float64(_score_parent_chosen(controller, record))
        target = Float64(record.target_value)
        heuristic = Float64(record.heuristic_scores[record.chosen_index])
        push!(preds, pred)
        push!(targets, target)
        push!(heuristics, heuristic)
        push!(frontier_targets, Float64(record.chosen_frontier_utility_delta))
        push!(enters_topk_targets, record.chosen_enters_topk ? 1.0 : 0.0)
        productive = record.outcome_class == "productive" || record.outcome_class == "weak_productive"
        degenerate = record.outcome_class == "degenerate"
        push!(productive_labels, productive)
        push!(degenerate_labels, degenerate)

        learned_scores = Float32[parent_candidate_score(controller, features) for features in record.candidate_features]
        learned_order = sortperm(learned_scores, rev=true)
        heuristic_order = sortperm(record.heuristic_scores, rev=true)
        push!(chosen_ranks, findfirst(==(record.chosen_index), learned_order))
        push!(heuristic_ranks, findfirst(==(record.chosen_index), heuristic_order))

        learned_top = learned_order[1]
        heuristic_top = heuristic_order[1]
        disagreement = learned_top != heuristic_top
        push!(disagreement_flags, disagreement)
        learned_margin = length(learned_order) >= 2 ? learned_scores[learned_order[1]] - learned_scores[learned_order[2]] : learned_scores[learned_order[1]]
        push!(learned_margins, Float64(learned_margin))
        learned_advantage = learned_scores[learned_top] - learned_scores[heuristic_top]
        push!(learned_advantages, Float64(learned_advantage))
        push!(high_conf_disagreement_flags, disagreement && learned_margin >= 0.1f0)

        heuristic_margin = length(heuristic_order) >= 2 ? record.heuristic_scores[heuristic_order[1]] - record.heuristic_scores[heuristic_order[2]] : record.heuristic_scores[heuristic_order[1]]
        heuristic_ambiguous = heuristic_margin < 0.1f0
        push!(heuristic_ambiguous_flags, heuristic_ambiguous)
        push!(ambiguous_disagreement_flags, heuristic_ambiguous && disagreement)
        push!(strong_heuristic_disagreement_flags, heuristic_margin >= 0.1f0 && disagreement)
        if heuristic_margin >= 0.1f0
            push!(strong_heuristic_preserved_flags, learned_top == heuristic_top)
        end
    end

    rmse = sqrt(mean((preds .- targets) .^ 2))
    prod_scores = [preds[i] for i in eachindex(preds) if productive_labels[i]]
    deg_scores = [preds[i] for i in eachindex(preds) if degenerate_labels[i]]
    return Dict(
        "n_records" => length(records),
        "rmse" => rmse,
        "score_target_correlation" => _parent_pearson_or_zero(preds, targets),
        "heuristic_target_correlation" => _parent_pearson_or_zero(heuristics, targets),
        "frontier_utility_correlation" => _parent_pearson_or_zero(preds, frontier_targets),
        "heuristic_frontier_utility_correlation" => _parent_pearson_or_zero(heuristics, frontier_targets),
        "enters_topk_correlation" => _parent_pearson_or_zero(preds, enters_topk_targets),
        "productive_degenerate_margin" => _parent_mean_or_zero(prod_scores) - _parent_mean_or_zero(deg_scores),
        "mean_chosen_rank" => mean(chosen_ranks),
        "mean_heuristic_chosen_rank" => mean(heuristic_ranks),
        "disagreement_rate" => mean(Float64.(disagreement_flags)),
        "high_confidence_disagreement_rate" => mean(Float64.(high_conf_disagreement_flags)),
        "ambiguous_disagreement_rate" => mean(Float64.(ambiguous_disagreement_flags)),
        "strong_heuristic_disagreement_rate" => mean(Float64.(strong_heuristic_disagreement_flags)),
        "heuristic_ambiguous_fraction" => mean(Float64.(heuristic_ambiguous_flags)),
        "mean_learned_margin" => _parent_mean_or_zero(learned_margins),
        "mean_learned_advantage_vs_heuristic" => _parent_mean_or_zero(learned_advantages),
        "preserve_strong_heuristic_rate" => isempty(strong_heuristic_preserved_flags) ? 1.0 : mean(Float64.(strong_heuristic_preserved_flags)),
    )
end

function _linear_grad(controller::LearnedParentController,
                      record::ParentDecisionRecord)
    x = Float32.(record.candidate_features[record.chosen_index])
    y = Float32(record.target_value)
    ŷ = parent_candidate_score(controller, x)
    err = ŷ - y
    return 0.5f0 * err^2, err .* x, err
end

function _mlp_grad(controller::MLPParentController,
                   record::ParentDecisionRecord)
    x = Float32.(record.candidate_features[record.chosen_index])
    y = Float32(record.target_value)
    z1 = controller.W1 * x .+ controller.b1
    h = tanh.(z1)
    ŷ = dot(controller.W2, h) + controller.b2
    err = ŷ - y
    loss = 0.5f0 * err^2

    dW2 = err .* h
    db2 = err
    dh = err .* controller.W2
    dz1 = dh .* (1 .- h.^2)
    dW1 = dz1 * transpose(x)
    db1 = dz1
    return loss, dW1, db1, dW2, db2
end

function _train_linear(dataset::ParentControllerDataset,
                       config::ParentControllerTrainingConfig,
                       rng::AbstractRNG)
    records = _parent_regression_records(dataset)
    input_dim = length(records[1].candidate_features[1])
    controller = create_learned_parent_controller(input_dim; hidden_dim=config.hidden_dim, feature_mode=config.feature_mode, rng=rng)
    history = Vector{Dict{String,Any}}()

    train_dataset, val_dataset = split_parent_controller_dataset(dataset; train_fraction=config.train_fraction, rng=rng)
    train_records = _parent_regression_records(train_dataset)
    val_records = _parent_regression_records(val_dataset)

    for epoch in 1:config.n_epochs
        grad_w = zeros(Float32, controller.input_dim)
        grad_b = 0.0f0
        total_loss = 0.0f0

        for record in train_records
            loss, dW, db = _linear_grad(controller, record)
            total_loss += loss
            grad_w .+= dW
            grad_b += db
        end

        n = max(length(train_records), 1)
        grad_w ./= n
        grad_b /= n
        grad_norm = sqrt(sum(abs2, grad_w) + grad_b^2)
        if grad_norm > config.grad_clip_norm && grad_norm > 0
            scale = Float32(config.grad_clip_norm / grad_norm)
            grad_w .*= scale
            grad_b *= scale
        end
        mean_loss = total_loss / n
        controller = LearnedParentController(
            controller.weights .- Float32(config.learning_rate) .* grad_w,
            controller.bias - Float32(config.learning_rate) * grad_b,
            controller.input_dim,
            controller.feature_mode,
        )

        train_eval = evaluate_parent_controller(controller, ParentControllerDataset(train_records))
        val_eval = evaluate_parent_controller(controller, ParentControllerDataset(val_records))
        push!(history, Dict(
            "epoch" => epoch,
            "train_loss" => Float64(mean_loss),
            "train_rmse" => train_eval["rmse"],
            "val_rmse" => val_eval["rmse"],
            "val_corr" => val_eval["score_target_correlation"],
        ))
    end

    return controller, train_dataset, val_dataset, history
end

function _train_mlp(dataset::ParentControllerDataset,
                    config::ParentControllerTrainingConfig,
                    rng::AbstractRNG)
    records = _parent_regression_records(dataset)
    input_dim = length(records[1].candidate_features[1])
    controller = create_mlp_parent_controller(input_dim; hidden_dim=config.hidden_dim, feature_mode=config.feature_mode, rng=rng)
    history = Vector{Dict{String,Any}}()

    train_dataset, val_dataset = split_parent_controller_dataset(dataset; train_fraction=config.train_fraction, rng=rng)
    train_records = _parent_regression_records(train_dataset)
    val_records = _parent_regression_records(val_dataset)

    for epoch in 1:config.n_epochs
        grad_W1 = zeros(Float32, size(controller.W1))
        grad_b1 = zeros(Float32, size(controller.b1))
        grad_W2 = zeros(Float32, size(controller.W2))
        grad_b2 = 0.0f0
        total_loss = 0.0f0

        for record in train_records
            loss, dW1, db1, dW2, db2 = _mlp_grad(controller, record)
            total_loss += loss
            grad_W1 .+= dW1
            grad_b1 .+= db1
            grad_W2 .+= dW2
            grad_b2 += db2
        end

        n = max(length(train_records), 1)
        grad_W1 ./= n
        grad_b1 ./= n
        grad_W2 ./= n
        grad_b2 /= n
        grad_norm = sqrt(sum(abs2, grad_W1) + sum(abs2, grad_b1) + sum(abs2, grad_W2) + grad_b2^2)
        if grad_norm > config.grad_clip_norm && grad_norm > 0
            scale = Float32(config.grad_clip_norm / grad_norm)
            grad_W1 .*= scale
            grad_b1 .*= scale
            grad_W2 .*= scale
            grad_b2 *= scale
        end
        mean_loss = total_loss / n

        controller = MLPParentController(
            controller.W1 .- Float32(config.learning_rate) .* grad_W1,
            controller.b1 .- Float32(config.learning_rate) .* grad_b1,
            controller.W2 .- Float32(config.learning_rate) .* grad_W2,
            controller.b2 - Float32(config.learning_rate) * grad_b2,
            controller.input_dim,
            controller.hidden_dim,
            controller.feature_mode,
        )

        train_eval = evaluate_parent_controller(controller, ParentControllerDataset(train_records))
        val_eval = evaluate_parent_controller(controller, ParentControllerDataset(val_records))
        push!(history, Dict(
            "epoch" => epoch,
            "train_loss" => Float64(mean_loss),
            "train_rmse" => train_eval["rmse"],
            "val_rmse" => val_eval["rmse"],
            "val_corr" => val_eval["score_target_correlation"],
        ))
    end

    return controller, train_dataset, val_dataset, history
end

function train_parent_controller(dataset::ParentControllerDataset;
                                 rng::AbstractRNG=Random.MersenneTwister(0),
                                 config::ParentControllerTrainingConfig=ParentControllerTrainingConfig())
    total_records = length(_parent_regression_records(dataset))
    if total_records < config.min_records
        error("Need at least $(config.min_records) parent records, got $(total_records)")
    end

    controller, train_dataset, val_dataset, history = if config.model_kind == :mlp
        _train_mlp(dataset, config, rng)
    else
        _train_linear(dataset, config, rng)
    end

    return controller, Dict(
        "model_kind" => String(config.model_kind),
        "dataset_stats" => parent_controller_dataset_stats(dataset),
        "train_stats" => parent_controller_dataset_stats(train_dataset),
        "val_stats" => parent_controller_dataset_stats(val_dataset),
        "train_eval" => evaluate_parent_controller(controller, train_dataset),
        "val_eval" => evaluate_parent_controller(controller, val_dataset),
        "history" => history,
    )
end

function compare_parent_regressors(dataset::ParentControllerDataset;
                                   rng::AbstractRNG=Random.MersenneTwister(0),
                                   linear_config::ParentControllerTrainingConfig=ParentControllerTrainingConfig(model_kind=:linear),
                                   mlp_config::ParentControllerTrainingConfig=ParentControllerTrainingConfig(model_kind=:mlp))
    linear_controller, linear_summary = train_parent_controller(dataset; rng=rng, config=linear_config)
    mlp_controller, mlp_summary = train_parent_controller(dataset; rng=Random.MersenneTwister(rand(rng, UInt)), config=mlp_config)
    return Dict(
        "linear" => Dict("controller" => linear_controller, "summary" => linear_summary),
        "mlp" => Dict("controller" => mlp_controller, "summary" => mlp_summary),
    )
end
