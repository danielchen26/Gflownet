# Operator Controller Training
#
# Batch 1H scope:
# - separate offline eligibility learning from conditional operator ranking
# - keep models simple and interpretable
# - evaluate activation quality separately from ranking quality

using Random
using Statistics

struct OperatorControllerTrainingConfig
    learning_rate::Float64
    n_epochs::Int
    train_fraction::Float64
    min_records::Int
    grad_clip_norm::Float64
    feature_mode::Symbol
    eligibility_threshold::Float32

    function OperatorControllerTrainingConfig(;
        learning_rate::Float64=1e-3,
        n_epochs::Int=40,
        train_fraction::Float64=0.8,
        min_records::Int=12,
        grad_clip_norm::Float64=5.0,
        feature_mode::Symbol=:basic,
        eligibility_threshold::Float32=0.50f0,
    )
        new(learning_rate, n_epochs, train_fraction, min_records, grad_clip_norm, feature_mode, eligibility_threshold)
    end
end

_operator_mean_or_zero(xs) = isempty(xs) ? 0.0 : mean(Float64.(xs))

function _operator_pearson_or_zero(xs::Vector{Float64}, ys::Vector{Float64})
    length(xs) < 2 && return 0.0
    (std(xs) == 0 || std(ys) == 0) && return 0.0
    return cor(xs, ys)
end

function filter_operator_controller_dataset(dataset::OperatorControllerDataset;
                                            eligible_only::Bool=false,
                                            state_labels::Union{Nothing,Vector{String}}=nothing)
    records = dataset.records
    if eligible_only
        records = [record for record in records if record.controller_eligible]
    end
    if !isnothing(state_labels)
        allowed = Set(state_labels)
        records = [record for record in records if record.state_label in allowed]
    end
    return OperatorControllerDataset(records)
end

_operator_regression_records(dataset::OperatorControllerDataset) = [record for record in dataset.records if length(record.candidate_features) >= 2]

function _score_operator_chosen(controller, record::OperatorDecisionRecord)
    return Float32(operator_candidate_score(controller, record.candidate_features[record.chosen_index]))
end

function evaluate_operator_eligibility_model(model,
                                             dataset::OperatorControllerDataset;
                                             threshold::Float32=0.50f0)
    records = dataset.records
    if isempty(records)
        return Dict(
            "n_records" => 0,
            "positive_fraction" => 0.0,
            "predicted_eligible_fraction" => 0.0,
            "overall_accuracy" => 0.0,
            "eligible_recall" => 0.0,
            "invariant_preserve_rate" => 1.0,
            "ambiguous_activation_rate" => 0.0,
            "degenerate_activation_rate" => 0.0,
            "mean_score" => 0.0,
        )
    end

    labels = Bool[r.controller_eligible for r in records]
    scores = Float64[operator_eligibility_score(model, r.eligibility_features) for r in records]
    preds = Bool[s >= threshold for s in scores]

    eligible_mask = [r.controller_eligible for r in records]
    invariant_mask = [r.state_label == "invariant" for r in records]
    ambiguous_mask = [r.state_label == "ambiguous" for r in records]
    degenerate_mask = [r.state_label == "degenerate" for r in records]

    return Dict(
        "n_records" => length(records),
        "positive_fraction" => mean(Float64.(labels)),
        "predicted_eligible_fraction" => mean(Float64.(preds)),
        "overall_accuracy" => mean(Float64[preds[i] == labels[i] for i in eachindex(labels)]),
        "eligible_recall" => any(eligible_mask) ? mean(Float64[preds[i] for i in eachindex(preds) if eligible_mask[i]]) : 0.0,
        "invariant_preserve_rate" => any(invariant_mask) ? mean(Float64[!preds[i] for i in eachindex(preds) if invariant_mask[i]]) : 1.0,
        "ambiguous_activation_rate" => any(ambiguous_mask) ? mean(Float64[preds[i] for i in eachindex(preds) if ambiguous_mask[i]]) : 0.0,
        "degenerate_activation_rate" => any(degenerate_mask) ? mean(Float64[preds[i] for i in eachindex(preds) if degenerate_mask[i]]) : 0.0,
        "mean_score" => mean(scores),
    )
end

function evaluate_operator_controller(controller,
                                      dataset::OperatorControllerDataset)
    records = _operator_regression_records(dataset)
    if isempty(records)
        return Dict(
            "n_records" => 0,
            "rmse" => 0.0,
            "score_target_correlation" => 0.0,
            "heuristic_target_correlation" => 0.0,
            "frontier_utility_correlation" => 0.0,
            "enters_topk_correlation" => 0.0,
            "mean_chosen_rank" => 0.0,
            "mean_heuristic_chosen_rank" => 0.0,
            "robust_state_agreement" => 0.0,
            "eligible_state_agreement" => 0.0,
            "invariant_preserve_rate" => 1.0,
            "ambiguous_disagreement_rate" => 0.0,
            "mean_learned_margin" => 0.0,
            "mean_learned_advantage_vs_heuristic" => 0.0,
        )
    end

    preds = Float64[]
    targets = Float64[]
    heuristics = Float64[]
    frontier_targets = Float64[]
    enters_topk_targets = Float64[]
    chosen_ranks = Float64[]
    heuristic_ranks = Float64[]
    robust_agree = Bool[]
    eligible_agree = Bool[]
    invariant_preserve = Bool[]
    ambiguous_disagreement = Bool[]
    learned_margins = Float64[]
    learned_advantages = Float64[]

    for record in records
        pred = Float64(_score_operator_chosen(controller, record))
        target = Float64(record.target_value)
        heuristic = Float64(record.heuristic_scores[record.chosen_index])
        push!(preds, pred)
        push!(targets, target)
        push!(heuristics, heuristic)
        push!(frontier_targets, Float64(record.chosen_frontier_utility_delta))
        push!(enters_topk_targets, record.chosen_enters_topk ? 1.0 : 0.0)

        learned_scores = Float32[operator_candidate_score(controller, features) for features in record.candidate_features]
        learned_order = sortperm(learned_scores, rev=true)
        heuristic_order = sortperm(record.heuristic_scores, rev=true)
        push!(chosen_ranks, findfirst(==(record.chosen_index), learned_order))
        push!(heuristic_ranks, findfirst(==(record.chosen_index), heuristic_order))

        learned_top = learned_order[1]
        heuristic_top = heuristic_order[1]
        learned_margin = length(learned_order) >= 2 ? learned_scores[learned_order[1]] - learned_scores[learned_order[2]] : learned_scores[learned_order[1]]
        push!(learned_margins, Float64(learned_margin))
        push!(learned_advantages, Float64(learned_scores[learned_top] - learned_scores[heuristic_top]))

        if record.state_label == "robust_operator"
            push!(robust_agree, learned_top == record.chosen_index)
        end
        if record.controller_eligible
            push!(eligible_agree, learned_top == record.chosen_index)
        end
        if record.state_label == "invariant"
            push!(invariant_preserve, learned_top == heuristic_top)
        end
        if record.state_label == "ambiguous"
            push!(ambiguous_disagreement, learned_top != heuristic_top)
        end
    end

    rmse = sqrt(mean((preds .- targets) .^ 2))
    return Dict(
        "n_records" => length(records),
        "rmse" => rmse,
        "score_target_correlation" => _operator_pearson_or_zero(preds, targets),
        "heuristic_target_correlation" => _operator_pearson_or_zero(heuristics, targets),
        "frontier_utility_correlation" => _operator_pearson_or_zero(preds, frontier_targets),
        "enters_topk_correlation" => _operator_pearson_or_zero(preds, enters_topk_targets),
        "mean_chosen_rank" => mean(chosen_ranks),
        "mean_heuristic_chosen_rank" => mean(heuristic_ranks),
        "robust_state_agreement" => isempty(robust_agree) ? 0.0 : mean(Float64.(robust_agree)),
        "eligible_state_agreement" => isempty(eligible_agree) ? 0.0 : mean(Float64.(eligible_agree)),
        "invariant_preserve_rate" => isempty(invariant_preserve) ? 1.0 : mean(Float64.(invariant_preserve)),
        "ambiguous_disagreement_rate" => isempty(ambiguous_disagreement) ? 0.0 : mean(Float64.(ambiguous_disagreement)),
        "mean_learned_margin" => _operator_mean_or_zero(learned_margins),
        "mean_learned_advantage_vs_heuristic" => _operator_mean_or_zero(learned_advantages),
    )
end

function _eligibility_grad(model::LearnedOperatorEligibilityModel,
                           record::OperatorDecisionRecord;
                           pos_weight::Float32=1.0f0)
    x = Float32.(record.eligibility_features)
    y = record.controller_eligible ? 1.0f0 : 0.0f0
    p = operator_eligibility_score(model, x)
    scale = y > 0.5f0 ? pos_weight : 1.0f0
    err = scale * (p - y)
    loss = -scale * (y * log(max(p, 1f-6)) + (1.0f0 - y) * log(max(1.0f0 - p, 1f-6)))
    return loss, err .* x, err
end

function _linear_grad(controller::LearnedOperatorController,
                      record::OperatorDecisionRecord)
    x = Float32.(record.candidate_features[record.chosen_index])
    y = Float32(record.target_value)
    yhat = operator_candidate_score(controller, x)
    err = yhat - y
    return 0.5f0 * err^2, err .* x, err
end

function train_operator_eligibility_model(dataset::OperatorControllerDataset;
                                          rng::AbstractRNG=Random.MersenneTwister(0),
                                          config::OperatorControllerTrainingConfig=OperatorControllerTrainingConfig())
    total_records = length(dataset.records)
    if total_records < config.min_records
        error("Need at least $(config.min_records) operator-eligibility records, got $(total_records)")
    end

    input_dim = length(dataset.records[1].eligibility_features)
    model = create_learned_operator_eligibility_model(input_dim; rng=rng)
    history = Vector{Dict{String,Any}}()

    train_dataset, val_dataset = split_operator_controller_dataset(dataset; train_fraction=config.train_fraction, rng=rng)
    train_records = train_dataset.records
    val_records = val_dataset.records
    n_pos = count(r -> r.controller_eligible, train_records)
    n_neg = length(train_records) - n_pos
    pos_weight = Float32(clamp(n_neg / max(n_pos, 1), 1.0, 4.0))

    function _eligibility_score_fn(ev::AbstractDict{String,<:Any})
        score = 0.55 * get(ev, "eligible_recall", 0.0)
        score += 0.30 * get(ev, "invariant_preserve_rate", 0.0)
        score += 0.10 * get(ev, "overall_accuracy", 0.0)
        score += 0.05 * min(get(ev, "predicted_eligible_fraction", 0.0), 0.5)
        score -= 0.20 * get(ev, "ambiguous_activation_rate", 0.0)
        score -= 0.15 * get(ev, "degenerate_activation_rate", 0.0)
        score -= 0.10 * abs(get(ev, "predicted_eligible_fraction", 0.0) - get(ev, "positive_fraction", 0.0))
        return score
    end

    for epoch in 1:config.n_epochs
        grad_w = zeros(Float32, model.input_dim)
        grad_b = 0.0f0
        total_loss = 0.0f0

        for record in train_records
            loss, dW, db = _eligibility_grad(model, record; pos_weight=pos_weight)
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
        model = LearnedOperatorEligibilityModel(
            model.weights .- Float32(config.learning_rate) .* grad_w,
            model.bias - Float32(config.learning_rate) * grad_b,
            model.input_dim,
        )

        train_eval = evaluate_operator_eligibility_model(model, OperatorControllerDataset(train_records); threshold=config.eligibility_threshold)
        val_eval = evaluate_operator_eligibility_model(model, OperatorControllerDataset(val_records); threshold=config.eligibility_threshold)
        push!(history, Dict(
            "epoch" => epoch,
            "train_loss" => Float64(mean_loss),
            "train_acc" => train_eval["overall_accuracy"],
            "val_acc" => val_eval["overall_accuracy"],
            "val_recall" => val_eval["eligible_recall"],
        ))
    end

    thresholds = Float32[0.25f0, 0.30f0, 0.35f0, 0.40f0, 0.45f0, 0.50f0, 0.55f0, 0.60f0, 0.65f0, 0.70f0]
    best_threshold = config.eligibility_threshold
    best_eval = evaluate_operator_eligibility_model(model, val_dataset; threshold=best_threshold)
    best_score = _eligibility_score_fn(best_eval)
    for threshold in thresholds
        ev = evaluate_operator_eligibility_model(model, val_dataset; threshold=threshold)
        score = _eligibility_score_fn(ev)
        if score > best_score
            best_score = score
            best_threshold = threshold
            best_eval = ev
        end
    end

    return model, Dict(
        "dataset_stats" => operator_controller_dataset_stats(dataset),
        "train_stats" => operator_controller_dataset_stats(train_dataset),
        "val_stats" => operator_controller_dataset_stats(val_dataset),
        "train_eval" => evaluate_operator_eligibility_model(model, train_dataset; threshold=best_threshold),
        "val_eval" => best_eval,
        "history" => history,
        "pos_weight" => Float64(pos_weight),
        "calibrated_threshold" => Float64(best_threshold),
        "calibrated_score" => Float64(best_score),
    )
end

function train_operator_controller(dataset::OperatorControllerDataset;
                                   rng::AbstractRNG=Random.MersenneTwister(0),
                                   config::OperatorControllerTrainingConfig=OperatorControllerTrainingConfig())
    records = _operator_regression_records(dataset)
    total_records = length(records)
    if total_records < config.min_records
        error("Need at least $(config.min_records) operator records, got $(total_records)")
    end

    input_dim = length(records[1].candidate_features[1])
    controller = create_learned_operator_controller(input_dim; feature_mode=config.feature_mode, rng=rng)
    history = Vector{Dict{String,Any}}()

    train_dataset, val_dataset = split_operator_controller_dataset(dataset; train_fraction=config.train_fraction, rng=rng)
    train_records = _operator_regression_records(train_dataset)
    val_records = _operator_regression_records(val_dataset)

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
        controller = LearnedOperatorController(
            controller.weights .- Float32(config.learning_rate) .* grad_w,
            controller.bias - Float32(config.learning_rate) * grad_b,
            controller.input_dim,
            controller.feature_mode,
        )

        train_eval = evaluate_operator_controller(controller, OperatorControllerDataset(train_records))
        val_eval = evaluate_operator_controller(controller, OperatorControllerDataset(val_records))
        push!(history, Dict(
            "epoch" => epoch,
            "train_loss" => Float64(mean_loss),
            "train_rmse" => train_eval["rmse"],
            "val_rmse" => val_eval["rmse"],
            "val_corr" => val_eval["score_target_correlation"],
        ))
    end

    return controller, Dict(
        "dataset_stats" => operator_controller_dataset_stats(dataset),
        "train_stats" => operator_controller_dataset_stats(train_dataset),
        "val_stats" => operator_controller_dataset_stats(val_dataset),
        "train_eval" => evaluate_operator_controller(controller, train_dataset),
        "val_eval" => evaluate_operator_controller(controller, val_dataset),
        "history" => history,
    )
end
