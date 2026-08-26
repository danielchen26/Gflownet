# Option Value Training
#
# Batch 1M scope:
# - preserve the stable linear ranker from Batch 1K/1L
# - add truthful override-target construction relative to the entry-local comparator
# - add a calibrated ordinal policy with separate ranking and preserve-vs-override confidence heads

using Random
using Statistics

struct OptionValueTrainingConfig
    learning_rate::Float64
    n_epochs::Int
    train_fraction::Float64
    min_records::Int
    grad_clip_norm::Float64
    feature_mode::Symbol
    objective_mode::Symbol
    pairwise_margin::Float64
    pairwise_weight::Float64

    function OptionValueTrainingConfig(;
        learning_rate::Float64=1e-3,
        n_epochs::Int=40,
        train_fraction::Float64=0.8,
        min_records::Int=12,
        grad_clip_norm::Float64=5.0,
        feature_mode::Symbol=:augmented,
        objective_mode::Symbol=:regression,
        pairwise_margin::Float64=0.05,
        pairwise_weight::Float64=0.5,
    )
        new(learning_rate, n_epochs, train_fraction, min_records, grad_clip_norm,
            feature_mode, objective_mode, pairwise_margin, pairwise_weight)
    end
end

struct OptionCalibrationConfig
    learning_rate::Float64
    n_epochs::Int
    grad_clip_norm::Float64
    override_gain_threshold::Float64
    ambiguity_threshold::Float64
    confidence_threshold_candidates::Vector{Float64}
    confidence_low_threshold_candidates::Vector{Float64}
    confidence_high_threshold_candidates::Vector{Float64}

    function OptionCalibrationConfig(;
        learning_rate::Float64=5e-3,
        n_epochs::Int=60,
        grad_clip_norm::Float64=5.0,
        override_gain_threshold::Float64=0.02,
        ambiguity_threshold::Float64=0.05,
        confidence_threshold_candidates::Vector{Float64}=[0.50, 0.55, 0.60, 0.65, 0.70],
        confidence_low_threshold_candidates::Vector{Float64}=[0.35, 0.40, 0.45, 0.50],
        confidence_high_threshold_candidates::Vector{Float64}=[0.55, 0.60, 0.65, 0.70],
    )
        new(learning_rate, n_epochs, grad_clip_norm, override_gain_threshold, ambiguity_threshold,
            confidence_threshold_candidates, confidence_low_threshold_candidates, confidence_high_threshold_candidates)
    end
end

struct OptionOverrideSample
    snapshot_id::String
    features::Vector{Float32}
    label::Float32
end

_option_mean_or_zero(xs) = isempty(xs) ? 0.0 : mean(Float64.(xs))

function _option_pearson_or_zero(xs::Vector{Float64}, ys::Vector{Float64})
    length(xs) < 2 && return 0.0
    (std(xs) == 0 || std(ys) == 0) && return 0.0
    value = cor(xs, ys)
    return isfinite(value) ? value : 0.0
end

function _confidence_bucket_summary(confidences::Vector{Float64}, labels::Vector{Bool})
    isempty(confidences) && return Dict{String,Any}()
    bounds = [0.0, 0.25, 0.50, 0.75, 1.01]
    summary = Dict{String,Any}()
    for idx in 1:length(bounds)-1
        lo = bounds[idx]
        hi = bounds[idx + 1]
        mask = [lo <= conf < hi for conf in confidences]
        count = sum(mask)
        label = string(round(lo, digits=2), "-", round(min(hi, 1.0), digits=2))
        if count == 0
            summary[label] = Dict("count" => 0, "mean_confidence" => 0.0, "helpful_rate" => 0.0)
        else
            idxs = findall(identity, mask)
            bucket_conf = confidences[idxs]
            bucket_labels = labels[idxs]
            summary[label] = Dict(
                "count" => count,
                "mean_confidence" => mean(bucket_conf),
                "helpful_rate" => mean(Float64.(bucket_labels)),
            )
        end
    end
    return summary
end

function _linear_option_grad(model::LearnedOptionValueModel,
                             record::OptionValueRecord)
    x = Float32.(record.features)
    y = Float32(record.option_value)
    yhat = option_value_score(model, x)
    err = yhat - y
    return 0.5f0 * err^2, err .* x, err
end

function _pairwise_option_grad(model::LearnedOptionValueModel,
                               records::Vector{OptionValueRecord};
                               margin::Float64=0.05)
    grad_w = zeros(Float32, model.input_dim)
    grad_b = 0.0f0
    total_loss = 0.0f0
    pair_count = 0

    for i in 1:length(records)-1
        for j in i+1:length(records)
            delta = Float32(records[i].option_value - records[j].option_value)
            abs(delta) <= 1f-6 && continue
            pair_count += 1
            sign_ij = delta > 0 ? 1.0f0 : -1.0f0
            xi = Float32.(records[i].features)
            xj = Float32.(records[j].features)
            si = option_value_score(model, xi)
            sj = option_value_score(model, xj)
            pair_gap = sign_ij * (si - sj)
            loss = max(0.0f0, Float32(margin) - pair_gap)
            total_loss += loss
            if loss > 0
                weight = max(abs(delta), 0.1f0)
                grad_w .-= weight .* sign_ij .* (xi .- xj)
            end
        end
    end

    if pair_count == 0
        return 0.0f0, grad_w, grad_b, 0
    end
    return total_loss / pair_count, grad_w ./ pair_count, grad_b, pair_count
end

function _group_option_records(dataset::OptionValueDataset)
    groups = Dict{String,Vector{OptionValueRecord}}()
    for record in dataset.records
        push!(get!(groups, record.snapshot_id, OptionValueRecord[]), record)
    end
    return groups
end

function _model_option_score(model::LearnedOptionValueModel, record::OptionValueRecord)
    return Float64(option_value_score(model, record.features))
end

function _model_option_score(policy::CalibratedOrdinalOptionPolicy, record::OptionValueRecord)
    return Float64(option_value_score(policy.ranking_model, record.features))
end

function _entry_local_state(records::Vector{OptionValueRecord}; ambiguity_threshold::Float64=0.05)
    local_scores = Float64[record.step1_local_utility for record in records]
    local_idx = argmax(local_scores)
    sorted_local = sort(local_scores; rev=true)
    local_margin = length(sorted_local) >= 2 ? sorted_local[1] - sorted_local[2] : Inf
    local_ambiguous = isfinite(local_margin) && local_margin <= ambiguity_threshold
    return local_idx, local_scores, local_margin, local_ambiguous
end

function _selection_metadata(local_idx::Int,
                             learned_idx::Int,
                             chosen_idx::Int,
                             local_margin::Float64,
                             learned_advantage::Float64,
                             local_ambiguous::Bool,
                             selection_rule::Symbol;
                             selected_confidence::Float64=0.0,
                             challenger_confidence::Float64=0.0,
                             uncertain_band::Bool=false,
                             predicted_override::Bool=(chosen_idx != local_idx),
                             challenger_helpful::Bool=false,
                             strong_challenger_helpful::Bool=false,
                             positive_override_opportunity::Bool=false,
                             override_gain_threshold::Float64=0.02)
    return Dict{String,Any}(
        "local_index" => local_idx,
        "learned_index" => learned_idx,
        "chosen_index" => chosen_idx,
        "local_margin" => local_margin,
        "learned_advantage_vs_local" => learned_advantage,
        "local_ambiguous" => local_ambiguous,
        "selection_rule" => String(selection_rule),
        "override_applied" => chosen_idx != local_idx,
        "preserved_to_local" => chosen_idx == local_idx,
        "selected_confidence" => selected_confidence,
        "challenger_confidence" => challenger_confidence,
        "uncertain_band" => uncertain_band,
        "predicted_override" => predicted_override,
        "challenger_helpful" => challenger_helpful,
        "strong_challenger_helpful" => strong_challenger_helpful,
        "positive_override_opportunity" => positive_override_opportunity,
        "override_gain_threshold" => override_gain_threshold,
    )
end

function _select_option_record(model::LearnedOptionValueModel,
                               records::Vector{OptionValueRecord};
                               selection_rule::Symbol=:argmax,
                               override_margin::Float64=0.05,
                               ambiguity_threshold::Float64=0.05)
    isempty(records) && error("Cannot select from empty option-record set")
    scores = Float64[_model_option_score(model, record) for record in records]
    learned_idx = argmax(scores)
    local_idx, _, local_margin, local_ambiguous = _entry_local_state(records; ambiguity_threshold=ambiguity_threshold)
    learned_advantage = scores[learned_idx] - scores[local_idx]

    chosen_idx = if selection_rule == :argmax
        learned_idx
    elseif selection_rule == :local_anchored
        (learned_idx != local_idx && learned_advantage > override_margin) ? learned_idx : local_idx
    elseif selection_rule == :ambiguity_gated
        if learned_idx != local_idx && ((local_ambiguous && learned_advantage > 0.0) || learned_advantage > override_margin)
            learned_idx
        else
            local_idx
        end
    else
        error("Unknown option selection rule: $(selection_rule)")
    end

    return records[chosen_idx], _selection_metadata(local_idx, learned_idx, chosen_idx, local_margin,
        learned_advantage, local_ambiguous, selection_rule)
end

function _challenger_override_features(policy::CalibratedOrdinalOptionPolicy,
                                       records::Vector{OptionValueRecord},
                                       scores::Vector{Float64},
                                       learned_idx::Int,
                                       local_idx::Int,
                                       local_margin::Float64)
    learned_record = records[learned_idx]
    rank_gap = scores[learned_idx] - scores[local_idx]
    feats = option_override_feature_vector(learned_record;
        ranking_score=scores[learned_idx],
        ranking_gap_vs_entry_local=rank_gap,
        local_margin=local_margin)
    return feats, rank_gap
end

function _select_option_record(policy::CalibratedOrdinalOptionPolicy,
                               records::Vector{OptionValueRecord};
                               selection_rule::Symbol=policy.selection_rule,
                               override_margin::Float64=Float64(policy.confidence_threshold),
                               ambiguity_threshold::Float64=Float64(policy.ambiguity_threshold))
    isempty(records) && error("Cannot select from empty option-record set")
    scores = Float64[_model_option_score(policy, record) for record in records]
    learned_idx = argmax(scores)
    local_idx, _, local_margin, local_ambiguous = _entry_local_state(records; ambiguity_threshold=ambiguity_threshold)
    learned_advantage = scores[learned_idx] - scores[local_idx]

    feats, rank_gap = _challenger_override_features(policy, records, scores, learned_idx, local_idx, local_margin)
    challenger_confidence = Float64(option_override_confidence(policy, feats))
    challenger_helpful = records[learned_idx].override_helpful
    strong_challenger_helpful = records[learned_idx].strong_override_helpful
    positive_override_opportunity = any(record.override_helpful for record in records if !record.entry_local_baseline)

    predicted_override = false
    uncertain_band = false
    if learned_idx != local_idx
        if selection_rule == :confidence_threshold
            predicted_override = challenger_confidence >= Float64(policy.confidence_threshold)
        elseif selection_rule == :confidence_band
            predicted_override = challenger_confidence >= Float64(policy.confidence_high_threshold)
            uncertain_band = !predicted_override && challenger_confidence >= Float64(policy.confidence_low_threshold)
        elseif selection_rule == :anchored_confidence
            predicted_override = challenger_confidence >= Float64(policy.confidence_high_threshold) ||
                                 (local_ambiguous && challenger_confidence >= Float64(policy.confidence_low_threshold))
            uncertain_band = !predicted_override && challenger_confidence >= Float64(policy.confidence_low_threshold)
        else
            error("Unknown calibrated option selection rule: $(selection_rule)")
        end
    end

    chosen_idx = predicted_override ? learned_idx : local_idx
    selected_confidence = chosen_idx == local_idx ? 0.0 : challenger_confidence
    return records[chosen_idx], _selection_metadata(local_idx, learned_idx, chosen_idx, local_margin,
        rank_gap, local_ambiguous, selection_rule;
        selected_confidence=selected_confidence,
        challenger_confidence=challenger_confidence,
        uncertain_band=uncertain_band,
        predicted_override=predicted_override,
        challenger_helpful=challenger_helpful,
        strong_challenger_helpful=strong_challenger_helpful,
        positive_override_opportunity=positive_override_opportunity,
        override_gain_threshold=Float64(policy.override_gain_threshold))
end

function _empty_option_eval(selection_rule::String)
    return Dict(
        "n_records" => 0,
        "n_snapshots" => 0,
        "rmse" => 0.0,
        "score_target_correlation" => 0.0,
        "score_continuation_correlation" => 0.0,
        "mean_selected_option_value" => 0.0,
        "mean_best_option_value" => 0.0,
        "mean_local_surface_utility" => 0.0,
        "mean_entry_context_surface_utility" => 0.0,
        "mean_entry_local_candidate_option_value" => 0.0,
        "mean_local_candidate_option_value" => 0.0,
        "mean_regret_vs_best" => 0.0,
        "mean_gain_vs_local_surface" => 0.0,
        "mean_gain_vs_entry_context_surface" => 0.0,
        "mean_gain_vs_entry_local_candidate" => 0.0,
        "mean_gain_vs_local_candidate" => 0.0,
        "selection_hit_rate" => 0.0,
        "reorder_fraction_vs_entry_local" => 0.0,
        "reorder_fraction_vs_local" => 0.0,
        "override_rate" => 0.0,
        "preserve_rate" => 0.0,
        "predicted_override_fraction" => 0.0,
        "ambiguous_local_fraction" => 0.0,
        "continuation_sensitive_fraction" => 0.0,
        "continuation_sensitive_gain_vs_entry_local_candidate" => 0.0,
        "continuation_sensitive_gain_vs_local_candidate" => 0.0,
        "invariant_gain_vs_entry_local_candidate" => 0.0,
        "invariant_gain_vs_local_candidate" => 0.0,
        "positive_override_opportunity_fraction" => 0.0,
        "override_precision" => 0.0,
        "override_recall" => 0.0,
        "mean_selected_confidence" => 0.0,
        "mean_challenger_confidence" => 0.0,
        "uncertain_fraction" => 0.0,
        "challenger_helpful_fraction" => 0.0,
        "confidence_bucket_summary" => Dict{String,Any}(),
        "selection_rule" => selection_rule,
    )
end

function _evaluate_option_predictions(model, dataset::OptionValueDataset)
    preds = Float64[]
    targets = Float64[]
    continuation_targets = Float64[]
    for record in dataset.records
        push!(preds, _model_option_score(model, record))
        push!(targets, Float64(record.option_value))
        push!(continuation_targets, Float64(record.continuation_gain))
    end
    rmse = isempty(preds) ? 0.0 : sqrt(mean((preds .- targets) .^ 2))
    return preds, targets, continuation_targets, rmse
end

function _evaluate_option_policy(model, dataset::OptionValueDataset;
                                 selection_rule::Symbol,
                                 override_margin::Float64=0.05,
                                 ambiguity_threshold::Float64=0.05)
    isempty(dataset) && return _empty_option_eval(String(selection_rule))

    preds, targets, continuation_targets, rmse = _evaluate_option_predictions(model, dataset)

    selected_values = Float64[]
    best_values = Float64[]
    local_values = Float64[]
    entry_values = Float64[]
    entry_local_values = Float64[]
    regrets = Float64[]
    gains_vs_local = Float64[]
    gains_vs_entry = Float64[]
    gains_vs_entry_local = Float64[]
    hit_rate = Bool[]
    reorder_flags = Bool[]
    override_flags = Bool[]
    preserve_flags = Bool[]
    predicted_override_flags = Bool[]
    ambiguous_flags = Bool[]
    continuation_sensitive_flags = Bool[]
    continuation_sensitive_gains = Float64[]
    invariant_gains = Float64[]
    positive_override_opportunity_flags = Bool[]
    override_helpful_flags = Bool[]
    selected_confidences = Float64[]
    challenger_confidences = Float64[]
    uncertain_flags = Bool[]
    challenger_helpful_flags = Bool[]

    grouped = _group_option_records(dataset)
    for records in values(grouped)
        selected, meta = _select_option_record(model, records;
            selection_rule=selection_rule,
            override_margin=override_margin,
            ambiguity_threshold=ambiguity_threshold)
        best_idx = argmax(Float64[record.option_value for record in records])
        local_idx = Int(meta["local_index"])
        best = records[best_idx]
        local_choice = records[local_idx]
        snapshot_sensitive = any(record.continuation_sensitive for record in records)
        gain_vs_entry_local = Float64(selected.option_value - local_choice.option_value)

        push!(selected_values, Float64(selected.option_value))
        push!(best_values, Float64(best.option_value))
        push!(local_values, Float64(selected.local_surface_utility))
        push!(entry_values, Float64(selected.entry_context_surface_utility))
        push!(entry_local_values, Float64(local_choice.option_value))
        push!(regrets, Float64(best.option_value - selected.option_value))
        push!(gains_vs_local, Float64(selected.option_value - selected.local_surface_utility))
        push!(gains_vs_entry, Float64(selected.option_value - selected.entry_context_surface_utility))
        push!(gains_vs_entry_local, gain_vs_entry_local)
        push!(hit_rate, selected.object_id == best.object_id)
        push!(reorder_flags, selected.object_id != local_choice.object_id)
        override_applied = Bool(meta["override_applied"])
        push!(override_flags, override_applied)
        push!(preserve_flags, Bool(meta["preserved_to_local"]))
        push!(predicted_override_flags, Bool(get(meta, "predicted_override", override_applied)))
        push!(ambiguous_flags, Bool(meta["local_ambiguous"]))
        push!(continuation_sensitive_flags, snapshot_sensitive)
        push!(positive_override_opportunity_flags, Bool(get(meta, "positive_override_opportunity", false)))
        push!(selected_confidences, Float64(get(meta, "selected_confidence", 0.0)))
        push!(challenger_confidences, Float64(get(meta, "challenger_confidence", 0.0)))
        push!(uncertain_flags, Bool(get(meta, "uncertain_band", false)))
        push!(challenger_helpful_flags, Bool(get(meta, "challenger_helpful", false)))
        push!(override_helpful_flags, override_applied && (gain_vs_entry_local > Float64(get(meta, "override_gain_threshold", 0.02))))
        if snapshot_sensitive
            push!(continuation_sensitive_gains, gain_vs_entry_local)
        else
            push!(invariant_gains, gain_vs_entry_local)
        end
    end

    override_precision = any(override_flags) ? mean(Float64.(override_helpful_flags[override_flags])) : 0.0
    positive_opportunities = sum(positive_override_opportunity_flags)
    override_recall = positive_opportunities > 0 ?
        mean(Float64[override_flags[idx] for idx in eachindex(override_flags) if positive_override_opportunity_flags[idx]]) : 0.0

    return Dict(
        "n_records" => length(dataset.records),
        "n_snapshots" => length(grouped),
        "rmse" => rmse,
        "score_target_correlation" => _option_pearson_or_zero(preds, targets),
        "score_continuation_correlation" => _option_pearson_or_zero(preds, continuation_targets),
        "mean_selected_option_value" => _option_mean_or_zero(selected_values),
        "mean_best_option_value" => _option_mean_or_zero(best_values),
        "mean_local_surface_utility" => _option_mean_or_zero(local_values),
        "mean_entry_context_surface_utility" => _option_mean_or_zero(entry_values),
        "mean_entry_local_candidate_option_value" => _option_mean_or_zero(entry_local_values),
        "mean_local_candidate_option_value" => _option_mean_or_zero(entry_local_values),
        "mean_regret_vs_best" => _option_mean_or_zero(regrets),
        "mean_gain_vs_local_surface" => _option_mean_or_zero(gains_vs_local),
        "mean_gain_vs_entry_context_surface" => _option_mean_or_zero(gains_vs_entry),
        "mean_gain_vs_entry_local_candidate" => _option_mean_or_zero(gains_vs_entry_local),
        "mean_gain_vs_local_candidate" => _option_mean_or_zero(gains_vs_entry_local),
        "selection_hit_rate" => isempty(hit_rate) ? 0.0 : mean(Float64.(hit_rate)),
        "reorder_fraction_vs_entry_local" => isempty(reorder_flags) ? 0.0 : mean(Float64.(reorder_flags)),
        "reorder_fraction_vs_local" => isempty(reorder_flags) ? 0.0 : mean(Float64.(reorder_flags)),
        "override_rate" => isempty(override_flags) ? 0.0 : mean(Float64.(override_flags)),
        "preserve_rate" => isempty(preserve_flags) ? 0.0 : mean(Float64.(preserve_flags)),
        "predicted_override_fraction" => isempty(predicted_override_flags) ? 0.0 : mean(Float64.(predicted_override_flags)),
        "ambiguous_local_fraction" => isempty(ambiguous_flags) ? 0.0 : mean(Float64.(ambiguous_flags)),
        "continuation_sensitive_fraction" => isempty(continuation_sensitive_flags) ? 0.0 : mean(Float64.(continuation_sensitive_flags)),
        "continuation_sensitive_gain_vs_entry_local_candidate" => _option_mean_or_zero(continuation_sensitive_gains),
        "continuation_sensitive_gain_vs_local_candidate" => _option_mean_or_zero(continuation_sensitive_gains),
        "invariant_gain_vs_entry_local_candidate" => _option_mean_or_zero(invariant_gains),
        "invariant_gain_vs_local_candidate" => _option_mean_or_zero(invariant_gains),
        "positive_override_opportunity_fraction" => isempty(positive_override_opportunity_flags) ? 0.0 : mean(Float64.(positive_override_opportunity_flags)),
        "override_precision" => override_precision,
        "override_recall" => override_recall,
        "mean_selected_confidence" => _option_mean_or_zero(selected_confidences),
        "mean_challenger_confidence" => _option_mean_or_zero(challenger_confidences),
        "uncertain_fraction" => isempty(uncertain_flags) ? 0.0 : mean(Float64.(uncertain_flags)),
        "challenger_helpful_fraction" => isempty(challenger_helpful_flags) ? 0.0 : mean(Float64.(challenger_helpful_flags)),
        "confidence_bucket_summary" => _confidence_bucket_summary(challenger_confidences, challenger_helpful_flags),
        "selection_rule" => String(selection_rule),
    )
end

function evaluate_option_value_model(model::LearnedOptionValueModel,
                                     dataset::OptionValueDataset;
                                     selection_rule::Symbol=:argmax,
                                     override_margin::Float64=0.05,
                                     ambiguity_threshold::Float64=0.05)
    return _evaluate_option_policy(model, dataset;
        selection_rule=selection_rule,
        override_margin=override_margin,
        ambiguity_threshold=ambiguity_threshold)
end

function evaluate_option_value_model(policy::CalibratedOrdinalOptionPolicy,
                                     dataset::OptionValueDataset;
                                     selection_rule::Symbol=policy.selection_rule,
                                     override_margin::Float64=Float64(policy.confidence_threshold),
                                     ambiguity_threshold::Float64=Float64(policy.ambiguity_threshold))
    return _evaluate_option_policy(policy, dataset;
        selection_rule=selection_rule,
        override_margin=override_margin,
        ambiguity_threshold=ambiguity_threshold)
end

function train_option_value_model(dataset::OptionValueDataset;
                                  rng::AbstractRNG=Random.MersenneTwister(0),
                                  config::OptionValueTrainingConfig=OptionValueTrainingConfig())
    total_records = length(dataset.records)
    if total_records < config.min_records
        error("Need at least $(config.min_records) option-value records, got $(total_records)")
    end

    input_dim = length(dataset.records[1].features)
    model = create_learned_option_value_model(input_dim; feature_mode=config.feature_mode, rng=rng)
    history = Vector{Dict{String,Any}}()

    train_dataset, val_dataset = split_option_value_dataset(dataset; train_fraction=config.train_fraction, rng=rng)
    train_records = train_dataset.records
    train_groups = collect(values(_group_option_records(train_dataset)))

    best_model = deepcopy(model)
    best_score = -Inf
    best_val = Dict{String,Any}()

    for epoch in 1:config.n_epochs
        grad_w = zeros(Float32, model.input_dim)
        grad_b = 0.0f0
        total_loss = 0.0f0
        total_weight = 0.0f0

        if config.objective_mode in (:regression, :hybrid)
            reg_grad_w = zeros(Float32, model.input_dim)
            reg_grad_b = 0.0f0
            reg_loss = 0.0f0
            for record in train_records
                loss, dW, db = _linear_option_grad(model, record)
                reg_loss += loss
                reg_grad_w .+= dW
                reg_grad_b += db
            end
            n = max(length(train_records), 1)
            reg_grad_w ./= n
            reg_grad_b /= n
            reg_loss /= n
            weight = config.objective_mode == :hybrid ? (1.0 - config.pairwise_weight) : 1.0
            grad_w .+= Float32(weight) .* reg_grad_w
            grad_b += Float32(weight) * reg_grad_b
            total_loss += Float32(weight) * reg_loss
            total_weight += Float32(weight)
        end

        if config.objective_mode in (:pairwise, :hybrid)
            pair_grad_w = zeros(Float32, model.input_dim)
            pair_grad_b = 0.0f0
            pair_loss = 0.0f0
            pair_batches = 0
            for records in train_groups
                length(records) < 2 && continue
                loss, dW, db, count = _pairwise_option_grad(model, records; margin=config.pairwise_margin)
                count == 0 && continue
                pair_loss += loss
                pair_grad_w .+= dW
                pair_grad_b += db
                pair_batches += 1
            end
            if pair_batches > 0
                pair_grad_w ./= pair_batches
                pair_grad_b /= pair_batches
                pair_loss /= pair_batches
                weight = config.objective_mode == :hybrid ? config.pairwise_weight : 1.0
                grad_w .+= Float32(weight) .* pair_grad_w
                grad_b += Float32(weight) * pair_grad_b
                total_loss += Float32(weight) * pair_loss
                total_weight += Float32(weight)
            end
        end

        if total_weight > 0
            grad_w ./= total_weight
            grad_b /= total_weight
            total_loss /= total_weight
        end

        grad_norm = sqrt(sum(abs2, grad_w) + grad_b^2)
        if grad_norm > config.grad_clip_norm && grad_norm > 0
            scale = Float32(config.grad_clip_norm / grad_norm)
            grad_w .*= scale
            grad_b *= scale
        end

        new_weights = model.weights .- Float32(config.learning_rate) .* grad_w
        new_bias = model.bias - Float32(config.learning_rate) * grad_b
        model = LearnedOptionValueModel(new_weights, new_bias, model.input_dim, model.feature_mode)

        train_eval = evaluate_option_value_model(model, train_dataset)
        val_eval = evaluate_option_value_model(model, val_dataset)
        push!(history, Dict(
            "epoch" => epoch,
            "loss" => total_loss,
            "train_eval" => train_eval,
            "val_eval" => val_eval,
        ))

        score = get(val_eval, "mean_gain_vs_entry_local_candidate", 0.0)
        score += 0.20 * get(val_eval, "selection_hit_rate", 0.0)
        score += 0.10 * get(val_eval, "score_target_correlation", 0.0)
        score += 0.05 * get(val_eval, "continuation_sensitive_gain_vs_entry_local_candidate", 0.0)
        score -= 0.10 * get(val_eval, "mean_regret_vs_best", 0.0)
        if score > best_score
            best_score = score
            best_model = deepcopy(model)
            best_val = val_eval
        end
    end

    final_eval = evaluate_option_value_model(best_model, dataset)
    return Dict(
        "model" => best_model,
        "history" => history,
        "train_dataset" => train_dataset,
        "val_dataset" => val_dataset,
        "best_val_eval" => best_val,
        "final_eval" => final_eval,
        "dataset_stats" => option_value_dataset_stats(dataset),
    )
end

function _build_override_samples(ranking_model::LearnedOptionValueModel,
                                 dataset::OptionValueDataset;
                                 override_gain_threshold::Float64=0.02,
                                 ambiguity_threshold::Float64=0.05)
    samples = OptionOverrideSample[]
    for records in values(_group_option_records(dataset))
        scores = Float64[_model_option_score(ranking_model, record) for record in records]
        local_idx, _, local_margin, _ = _entry_local_state(records; ambiguity_threshold=ambiguity_threshold)
        for idx in eachindex(records)
            record = records[idx]
            features = option_override_feature_vector(record;
                ranking_score=scores[idx],
                ranking_gap_vs_entry_local=scores[idx] - scores[local_idx],
                local_margin=local_margin)
            label = (!record.entry_local_baseline && record.gain_vs_entry_local_candidate > override_gain_threshold) ? 1.0f0 : 0.0f0
            push!(samples, OptionOverrideSample(record.snapshot_id, features, label))
        end
    end
    return samples
end

function _evaluate_override_samples(weights::Vector{Float32},
                                    bias::Float32,
                                    samples::Vector{OptionOverrideSample};
                                    threshold::Float64=0.5)
    isempty(samples) && return Dict(
        "n_samples" => 0,
        "positive_fraction" => 0.0,
        "accuracy" => 0.0,
        "precision" => 0.0,
        "recall" => 0.0,
        "score_label_correlation" => 0.0,
        "mean_confidence" => 0.0,
    )
    scores = Float64[]
    labels = Float64[sample.label for sample in samples]
    preds = Bool[]
    for sample in samples
        score = Float64(_sigmoid32(dot(weights, sample.features) + bias))
        push!(scores, score)
        push!(preds, score >= threshold)
    end
    positives = labels .> 0.5
    accuracy = mean(Float64[preds[i] == positives[i] for i in eachindex(preds)])
    precision = any(preds) ? mean(Float64[positives[i] for i in eachindex(preds) if preds[i]]) : 0.0
    recall = any(positives) ? mean(Float64[preds[i] for i in eachindex(preds) if positives[i]]) : 0.0
    return Dict(
        "n_samples" => length(samples),
        "positive_fraction" => mean(labels),
        "accuracy" => accuracy,
        "precision" => precision,
        "recall" => recall,
        "score_label_correlation" => _option_pearson_or_zero(scores, labels),
        "mean_confidence" => mean(scores),
    )
end

function train_option_override_confidence_model(ranking_model::LearnedOptionValueModel,
                                                dataset::OptionValueDataset;
                                                rng::AbstractRNG=Random.MersenneTwister(0),
                                                config::OptionCalibrationConfig=OptionCalibrationConfig())
    samples = _build_override_samples(ranking_model, dataset;
        override_gain_threshold=config.override_gain_threshold,
        ambiguity_threshold=config.ambiguity_threshold)
    isempty(samples) && error("Need at least one override sample")

    input_dim = length(samples[1].features)
    weights = 0.01f0 .* randn(rng, Float32, input_dim)
    bias = 0.0f0
    history = Dict{String,Any}[]

    positives = sum(sample.label > 0.5 for sample in samples)
    negatives = length(samples) - positives
    pos_weight = positives > 0 ? length(samples) / (2 * positives) : 1.0
    neg_weight = negatives > 0 ? length(samples) / (2 * negatives) : 1.0

    for epoch in 1:config.n_epochs
        grad_w = zeros(Float32, input_dim)
        grad_b = 0.0f0
        total_loss = 0.0f0
        for sample in samples
            y = sample.label
            score = _sigmoid32(dot(weights, sample.features) + bias)
            weight = y > 0.5 ? pos_weight : neg_weight
            err = Float32(weight) * (score - y)
            grad_w .+= err .* sample.features
            grad_b += err
            total_loss += -Float32(weight) * (y * log(score + 1f-6) + (1.0f0 - y) * log(1.0f0 - score + 1f-6))
        end
        n = max(length(samples), 1)
        grad_w ./= n
        grad_b /= n
        total_loss /= n

        grad_norm = sqrt(sum(abs2, grad_w) + grad_b^2)
        if grad_norm > config.grad_clip_norm && grad_norm > 0
            scale = Float32(config.grad_clip_norm / grad_norm)
            grad_w .*= scale
            grad_b *= scale
        end

        weights .-= Float32(config.learning_rate) .* grad_w
        bias -= Float32(config.learning_rate) * grad_b

        push!(history, Dict(
            "epoch" => epoch,
            "loss" => total_loss,
            "eval" => _evaluate_override_samples(weights, bias, samples),
        ))
    end

    return Dict(
        "weights" => weights,
        "bias" => bias,
        "input_dim" => input_dim,
        "history" => history,
        "final_eval" => _evaluate_override_samples(weights, bias, samples),
        "n_samples" => length(samples),
    )
end

function _option_policy_behavior_score(ev::AbstractDict{String,<:Any})
    score = 0.35 * get(ev, "mean_gain_vs_local_surface", 0.0)
    score += 0.20 * get(ev, "mean_gain_vs_entry_local_candidate", 0.0)
    score += 0.15 * get(ev, "selection_hit_rate", 0.0)
    score += 0.15 * get(ev, "override_precision", 0.0)
    score += 0.10 * get(ev, "override_recall", 0.0)
    score -= 0.10 * get(ev, "mean_regret_vs_best", 0.0)

    override_rate = get(ev, "override_rate", 0.0)
    if override_rate < 0.05
        score -= 0.20 * (0.05 - override_rate)
    elseif override_rate > 0.85
        score -= 0.40 * (override_rate - 0.85)
    end
    return score
end

function train_calibrated_ordinal_option_policy(dataset::OptionValueDataset;
                                                rng::AbstractRNG=Random.MersenneTwister(0),
                                                ranking_config::OptionValueTrainingConfig=OptionValueTrainingConfig(objective_mode=:pairwise),
                                                calibration_config::OptionCalibrationConfig=OptionCalibrationConfig(),
                                                selection_rule::Symbol=:confidence_threshold)
    ranking_result = train_option_value_model(dataset; rng=rng, config=ranking_config)
    ranking_model = ranking_result["model"]
    train_dataset = ranking_result["train_dataset"]
    val_dataset = ranking_result["val_dataset"]

    confidence_result = train_option_override_confidence_model(ranking_model, train_dataset;
        rng=rng,
        config=calibration_config)

    best_policy = nothing
    best_val_eval = Dict{String,Any}()
    best_score = -Inf
    searched = Dict{String,Any}[]

    function maybe_update_policy(low_thr::Float64, high_thr::Float64, main_thr::Float64)
        policy = CalibratedOrdinalOptionPolicy(
            ranking_model,
            confidence_result["weights"],
            confidence_result["bias"],
            confidence_result["input_dim"],
            selection_rule,
            Float32(main_thr),
            Float32(low_thr),
            Float32(high_thr),
            Float32(calibration_config.ambiguity_threshold),
            Float32(calibration_config.override_gain_threshold),
        )
        val_eval = evaluate_option_value_model(policy, val_dataset)
        score = _option_policy_behavior_score(val_eval)
        push!(searched, Dict(
            "selection_rule" => String(selection_rule),
            "confidence_threshold" => main_thr,
            "confidence_low_threshold" => low_thr,
            "confidence_high_threshold" => high_thr,
            "score" => score,
            "val_eval" => val_eval,
        ))
        if score > best_score
            best_score = score
            best_policy = policy
            best_val_eval = val_eval
        end
    end

    if selection_rule == :confidence_threshold
        for thr in calibration_config.confidence_threshold_candidates
            maybe_update_policy(thr, thr, thr)
        end
    elseif selection_rule in (:confidence_band, :anchored_confidence)
        for low_thr in calibration_config.confidence_low_threshold_candidates
            for high_thr in calibration_config.confidence_high_threshold_candidates
                low_thr < high_thr || continue
                maybe_update_policy(low_thr, high_thr, high_thr)
            end
        end
    else
        error("Unknown calibrated option policy selection rule: $(selection_rule)")
    end

    isnothing(best_policy) && error("Failed to calibrate any option policy")

    final_eval = evaluate_option_value_model(best_policy, dataset)
    return Dict(
        "policy" => best_policy,
        "ranking_result" => ranking_result,
        "confidence_result" => confidence_result,
        "searched_policies" => searched,
        "train_dataset" => train_dataset,
        "val_dataset" => val_dataset,
        "best_val_eval" => best_val_eval,
        "final_eval" => final_eval,
        "dataset_stats" => option_value_dataset_stats(dataset),
    )
end
