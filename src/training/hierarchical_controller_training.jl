# Hierarchical Controller Training
#
# Batch 1A.1 scope:
# - offline regression-style basin learning over repaired attempt-level targets
# - linear vs small MLP capacity comparison
# - offline metrics that better reflect online usefulness than pure action imitation

struct BasinControllerTrainingConfig
    hidden_dim::Int
    learning_rate::Float64
    n_epochs::Int
    train_fraction::Float64
    min_records::Int
    model_kind::Symbol
    grad_clip_norm::Float64
    feature_mode::Symbol

    function BasinControllerTrainingConfig(;
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

function _regression_records(dataset::BasinControllerDataset)
    return [record for record in dataset.records if length(record.candidate_features) >= 2]
end

function _score_chosen(controller, record::BasinDecisionRecord)
    return Float32(basin_candidate_score(controller, record.candidate_features[record.chosen_index]))
end

function _pearson_or_zero(xs::Vector{Float64}, ys::Vector{Float64})
    length(xs) < 2 && return 0.0
    (std(xs) == 0 || std(ys) == 0) && return 0.0
    return cor(xs, ys)
end


function _mean_or_zero(xs)
    isempty(xs) && return 0.0
    return mean(Float64.(xs))
end

function _balanced_accuracy(pred_labels::Vector{Bool}, true_labels::Vector{Bool})
    length(pred_labels) == length(true_labels) || return 0.0
    isempty(pred_labels) && return 0.0
    positives = findall(identity, true_labels)
    negatives = findall(!, true_labels)
    if isempty(positives) || isempty(negatives)
        return 0.5
    end
    tpr = mean(Float64[pred_labels[i] for i in positives])
    tnr = mean(Float64[!pred_labels[i] for i in negatives])
    return 0.5 * (tpr + tnr)
end

function _bucket_productivity(preds::Vector{Float64}, labels::Vector{Bool}; n_buckets::Int=3)
    isempty(preds) && return Float64[]
    order = sortperm(preds)
    buckets = [Float64[] for _ in 1:n_buckets]
    for (rank, idx) in enumerate(order)
        bucket = min(n_buckets, max(1, ceil(Int, rank * n_buckets / length(order))))
        push!(buckets[bucket], labels[idx] ? 1.0 : 0.0)
    end
    return [_mean_or_zero(bucket) for bucket in buckets]
end

function evaluate_basin_controller(controller,
                                   dataset::BasinControllerDataset)
    records = _regression_records(dataset)
    if isempty(records)
        return Dict(
            "n_records" => 0,
            "rmse" => 0.0,
            "score_target_correlation" => 0.0,
            "heuristic_target_correlation" => 0.0,
            "productive_accuracy" => 0.0,
            "mean_chosen_rank" => 0.0,
            "mean_heuristic_chosen_rank" => 0.0,
            "frontier_utility_correlation" => 0.0,
            "heuristic_frontier_utility_correlation" => 0.0,
            "enters_topk_correlation" => 0.0,
            "heuristic_enters_topk_correlation" => 0.0,
            "productive_degenerate_margin" => 0.0,
            "heuristic_productive_degenerate_margin" => 0.0,
            "degenerate_balanced_accuracy" => 0.0,
            "score_bucket_productivity" => Float64[],
            "heuristic_bucket_productivity" => Float64[],
        )
    end

    preds = Float64[]
    targets = Float64[]
    heuristics = Float64[]
    frontier_targets = Float64[]
    enters_topk_targets = Float64[]
    productive_labels = Bool[]
    degenerate_labels = Bool[]
    productive_correct = 0
    chosen_ranks = Float64[]
    heuristic_ranks = Float64[]

    for record in records
        pred = Float64(_score_chosen(controller, record))
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
        productive_correct += ((pred > 0) == record.success_label)

        learned_scores = Float32[basin_candidate_score(controller, features) for features in record.candidate_features]
        learned_order = sortperm(learned_scores, rev=true)
        heuristic_order = sortperm(record.heuristic_scores, rev=true)
        push!(chosen_ranks, findfirst(==(record.chosen_index), learned_order))
        push!(heuristic_ranks, findfirst(==(record.chosen_index), heuristic_order))
    end

    rmse = sqrt(mean((preds .- targets) .^ 2))
    prod_scores = [preds[i] for i in eachindex(preds) if productive_labels[i]]
    deg_scores = [preds[i] for i in eachindex(preds) if degenerate_labels[i]]
    heuristic_prod_scores = [heuristics[i] for i in eachindex(heuristics) if productive_labels[i]]
    heuristic_deg_scores = [heuristics[i] for i in eachindex(heuristics) if degenerate_labels[i]]
    pred_bucket_rates = _bucket_productivity(preds, productive_labels)
    heuristic_bucket_rates = _bucket_productivity(heuristics, productive_labels)
    degenerate_pred_labels = [pred <= 0 for pred in preds]

    return Dict(
        "n_records" => length(records),
        "rmse" => rmse,
        "score_target_correlation" => _pearson_or_zero(preds, targets),
        "heuristic_target_correlation" => _pearson_or_zero(heuristics, targets),
        "productive_accuracy" => productive_correct / length(records),
        "mean_chosen_rank" => mean(chosen_ranks),
        "mean_heuristic_chosen_rank" => mean(heuristic_ranks),
        "mean_positive_prediction" => _mean_or_zero([p for (p, t) in zip(preds, targets) if t > 0]),
        "mean_nonpositive_prediction" => _mean_or_zero([p for (p, t) in zip(preds, targets) if t <= 0]),
        "frontier_utility_correlation" => _pearson_or_zero(preds, frontier_targets),
        "heuristic_frontier_utility_correlation" => _pearson_or_zero(heuristics, frontier_targets),
        "enters_topk_correlation" => _pearson_or_zero(preds, enters_topk_targets),
        "heuristic_enters_topk_correlation" => _pearson_or_zero(heuristics, enters_topk_targets),
        "productive_mean_prediction" => _mean_or_zero(prod_scores),
        "degenerate_mean_prediction" => _mean_or_zero(deg_scores),
        "productive_degenerate_margin" => _mean_or_zero(prod_scores) - _mean_or_zero(deg_scores),
        "heuristic_productive_degenerate_margin" => _mean_or_zero(heuristic_prod_scores) - _mean_or_zero(heuristic_deg_scores),
        "degenerate_balanced_accuracy" => _balanced_accuracy(degenerate_pred_labels, degenerate_labels),
        "score_bucket_productivity" => pred_bucket_rates,
        "heuristic_bucket_productivity" => heuristic_bucket_rates,
    )
end

function _linear_grad(controller::LearnedBasinController,
                      record::BasinDecisionRecord)
    x = Float32.(record.candidate_features[record.chosen_index])
    y = Float32(record.target_value)
    ŷ = basin_candidate_score(controller, x)
    err = ŷ - y
    return 0.5f0 * err^2, err .* x, err
end

function _mlp_grad(controller::MLPBasinController,
                   record::BasinDecisionRecord)
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

function _train_linear(dataset::BasinControllerDataset,
                       config::BasinControllerTrainingConfig,
                       rng::AbstractRNG)
    records = _regression_records(dataset)
    input_dim = length(records[1].candidate_features[1])
    controller = create_learned_basin_controller(input_dim; hidden_dim=config.hidden_dim, feature_mode=config.feature_mode, rng=rng)
    history = Vector{Dict{String,Any}}()

    train_dataset, val_dataset = split_basin_controller_dataset(dataset; train_fraction=config.train_fraction, rng=rng)
    train_records = _regression_records(train_dataset)
    val_records = _regression_records(val_dataset)

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
        controller = LearnedBasinController(
            controller.weights .- Float32(config.learning_rate) .* grad_w,
            controller.bias - Float32(config.learning_rate) * grad_b,
            controller.input_dim,
            controller.feature_mode,
        )

        train_eval = evaluate_basin_controller(controller, BasinControllerDataset(train_records))
        val_eval = evaluate_basin_controller(controller, BasinControllerDataset(val_records))
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

function _train_mlp(dataset::BasinControllerDataset,
                    config::BasinControllerTrainingConfig,
                    rng::AbstractRNG)
    records = _regression_records(dataset)
    input_dim = length(records[1].candidate_features[1])
    controller = create_mlp_basin_controller(input_dim; hidden_dim=config.hidden_dim, feature_mode=config.feature_mode, rng=rng)
    history = Vector{Dict{String,Any}}()

    train_dataset, val_dataset = split_basin_controller_dataset(dataset; train_fraction=config.train_fraction, rng=rng)
    train_records = _regression_records(train_dataset)
    val_records = _regression_records(val_dataset)

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

        controller = MLPBasinController(
            controller.W1 .- Float32(config.learning_rate) .* grad_W1,
            controller.b1 .- Float32(config.learning_rate) .* grad_b1,
            controller.W2 .- Float32(config.learning_rate) .* grad_W2,
            controller.b2 - Float32(config.learning_rate) * grad_b2,
            controller.input_dim,
            controller.hidden_dim,
            controller.feature_mode,
        )

        train_eval = evaluate_basin_controller(controller, BasinControllerDataset(train_records))
        val_eval = evaluate_basin_controller(controller, BasinControllerDataset(val_records))
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

function train_basin_controller(dataset::BasinControllerDataset;
                                rng::AbstractRNG=Random.MersenneTwister(0),
                                config::BasinControllerTrainingConfig=BasinControllerTrainingConfig())
    total_records = length(_regression_records(dataset))
    if total_records < config.min_records
        error("Need at least $(config.min_records) basin records, got $(total_records)")
    end

    controller, train_dataset, val_dataset, history = if config.model_kind == :mlp
        _train_mlp(dataset, config, rng)
    else
        _train_linear(dataset, config, rng)
    end

    return controller, Dict(
        "model_kind" => String(config.model_kind),
        "dataset_stats" => basin_controller_dataset_stats(dataset),
        "train_stats" => basin_controller_dataset_stats(train_dataset),
        "val_stats" => basin_controller_dataset_stats(val_dataset),
        "train_eval" => evaluate_basin_controller(controller, train_dataset),
        "val_eval" => evaluate_basin_controller(controller, val_dataset),
        "history" => history,
    )
end

function compare_basin_regressors(dataset::BasinControllerDataset;
                                  rng::AbstractRNG=Random.MersenneTwister(0),
                                  linear_config::BasinControllerTrainingConfig=BasinControllerTrainingConfig(model_kind=:linear),
                                  mlp_config::BasinControllerTrainingConfig=BasinControllerTrainingConfig(model_kind=:mlp))
    linear_controller, linear_summary = train_basin_controller(dataset; rng=rng, config=linear_config)
    mlp_controller, mlp_summary = train_basin_controller(dataset; rng=Random.MersenneTwister(rand(rng, UInt)), config=mlp_config)
    return Dict(
        "linear" => Dict("controller" => linear_controller, "summary" => linear_summary),
        "mlp" => Dict("controller" => mlp_controller, "summary" => mlp_summary),
    )
end
