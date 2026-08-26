# Option Value Dataset Utilities
#
# Batch 1M scope:
# - preserve truthful short-horizon option/subtrajectory records from Batch 1J/1K probes
# - preserve Batch 1L within-snapshot relative context features
# - add truthful override-target construction relative to the entry-local comparator
# - support calibrated preserve-vs-override policies over the same bounded option object

using Random
using Statistics

struct OptionValueRecord
    snapshot_id::String
    task_name::String
    horizon::Int
    object_id::String
    entry_key::String
    basin_scaffold::String
    parent_smiles::String
    operator::Symbol
    features::Vector{Float32}
    step1_local_utility::Float32
    option_value::Float32
    continuation_gain::Float32
    best_reward_reached::Float32
    topk_hit_fraction::Float32
    gain_vs_baseline::Float32
    local_surface_utility::Float32
    entry_context_surface_utility::Float32
    local_rank_fraction::Float32
    local_margin_to_top::Float32
    local_centered_utility::Float32
    basin_rank_fraction::Float32
    parent_rank_fraction::Float32
    continuation_ratio::Float32
    continuation_sensitive::Bool
    local_ambiguous::Bool
    all_degenerate::Bool
    gain_vs_entry_local_candidate::Float32
    override_helpful::Bool
    strong_override_helpful::Bool
    entry_local_baseline::Bool
    snapshot_has_override_opportunity::Bool
    snapshot_has_strong_override_opportunity::Bool
end

mutable struct OptionValueDataset
    records::Vector{OptionValueRecord}
end

OptionValueDataset() = OptionValueDataset(OptionValueRecord[])
Base.length(dataset::OptionValueDataset) = length(dataset.records)
Base.isempty(dataset::OptionValueDataset) = isempty(dataset.records)

function _rank_fractions(values::Vector{Float64})
    n = length(values)
    n == 0 && return Float32[]
    n == 1 && return Float32[1.0f0]
    order = sortperm(values; rev=true)
    ranks = zeros(Int, n)
    for (rank, idx) in enumerate(order)
        ranks[idx] = rank
    end
    return Float32[(n - ranks[idx]) / (n - 1) for idx in eachindex(values)]
end

function _snapshot_relative_contexts(records::Vector{<:AbstractDict{String,<:Any}};
                                     override_margin::Float64=0.02,
                                     strong_override_margin::Float64=0.05)
    n = length(records)
    n == 0 && return Dict{String,Float64}[]

    local_vals = Float64[Float64(get(record, "step1_local_utility", 0.0)) for record in records]
    basin_vals = Float64[Float64(get(record, "basin_score", 0.0)) for record in records]
    parent_vals = Float64[Float64(get(record, "parent_reward", 0.0)) for record in records]
    option_vals = Float64[Float64(get(record, "option_value", 0.0)) for record in records]
    continuation_vals = Float64[Float64(get(record, "continuation_gain", 0.0)) for record in records]

    local_rank = _rank_fractions(local_vals)
    basin_rank = _rank_fractions(basin_vals)
    parent_rank = _rank_fractions(parent_vals)

    top_local = maximum(local_vals)
    mean_local = mean(local_vals)
    sorted_local = sort(local_vals; rev=true)
    second_local = length(sorted_local) >= 2 ? sorted_local[2] : sorted_local[1]
    top_margin = top_local - second_local
    local_idx = argmax(local_vals)
    local_candidate_value = option_vals[local_idx]

    snapshot_has_override = any((idx != local_idx) && ((option_vals[idx] - local_candidate_value) > override_margin) for idx in eachindex(records))
    snapshot_has_strong_override = any((idx != local_idx) && ((option_vals[idx] - local_candidate_value) > strong_override_margin) for idx in eachindex(records))

    contexts = Dict{String,Float64}[]
    for idx in eachindex(records)
        option_value = option_vals[idx]
        continuation_gain = continuation_vals[idx]
        step1_value = local_vals[idx]
        continuation_ratio = abs(option_value) <= 1e-6 ? continuation_gain : continuation_gain / max(abs(option_value), 1e-6)
        continuation_sensitive = continuation_gain > max(0.05, 0.15 * abs(step1_value))
        gain_vs_entry_local = option_value - local_candidate_value
        is_entry_local_baseline = idx == local_idx
        push!(contexts, Dict{String,Float64}(
            "local_rank_fraction" => Float64(local_rank[idx]),
            "local_margin_to_top" => top_local - local_vals[idx],
            "local_centered_utility" => local_vals[idx] - mean_local,
            "basin_rank_fraction" => Float64(basin_rank[idx]),
            "parent_rank_fraction" => Float64(parent_rank[idx]),
            "continuation_ratio" => continuation_ratio,
            "continuation_sensitive" => continuation_sensitive ? 1.0 : 0.0,
            "local_ambiguous" => top_margin <= 0.05 ? 1.0 : 0.0,
            "gain_vs_entry_local_candidate" => gain_vs_entry_local,
            "override_helpful" => (!is_entry_local_baseline && gain_vs_entry_local > override_margin) ? 1.0 : 0.0,
            "strong_override_helpful" => (!is_entry_local_baseline && gain_vs_entry_local > strong_override_margin) ? 1.0 : 0.0,
            "entry_local_baseline" => is_entry_local_baseline ? 1.0 : 0.0,
            "snapshot_has_override_opportunity" => snapshot_has_override ? 1.0 : 0.0,
            "snapshot_has_strong_override_opportunity" => snapshot_has_strong_override ? 1.0 : 0.0,
        ))
    end
    return contexts
end

function option_value_feature_vector(record::AbstractDict{String,<:Any};
                                     feature_mode::Symbol=:basic,
                                     snapshot_context::Union{Nothing,AbstractDict{String,<:Any}}=nothing)
    operator = Symbol(get(record, "operator", "terminate"))
    op_flags = Float32[
        operator == :mutate ? 1.0 : 0.0,
        operator == :crossover ? 1.0 : 0.0,
        operator == :terminate ? 1.0 : 0.0,
    ]

    base_context = Float32[
        Float32(get(record, "basin_score", 0.0)),
        Float32(get(record, "parent_reward", 0.0)),
        Float32(get(record, "parent_novelty_score", 0.0)),
        Float32(get(record, "parent_tb_delta_abs", 0.0)),
        Float32(get(record, "horizon", 0)),
    ]

    isnothing(snapshot_context) && (snapshot_context = Dict{String,Float64}())
    relative_context = Float32[
        Float32(get(snapshot_context, "local_rank_fraction", 0.0)),
        Float32(get(snapshot_context, "local_margin_to_top", 0.0)),
        Float32(get(snapshot_context, "local_centered_utility", 0.0)),
        Float32(get(snapshot_context, "basin_rank_fraction", 0.0)),
        Float32(get(snapshot_context, "parent_rank_fraction", 0.0)),
    ]

    continuation_context = Float32[
        Float32(get(record, "continuation_gain", 0.0)),
        Float32(get(record, "gain_vs_baseline", 0.0)),
        Float32(get(record, "topk_hit_fraction", 0.0)),
        Float32(get(snapshot_context, "continuation_ratio", 0.0)),
        Float32(get(snapshot_context, "continuation_sensitive", 0.0)),
        Float32(get(snapshot_context, "local_ambiguous", 0.0)),
    ]

    interaction_context = Float32[
        Float32(get(record, "step1_local_utility", 0.0) * get(record, "continuation_gain", 0.0)),
        Float32(get(record, "basin_score", 0.0) * (operator == :mutate ? 1.0 : 0.0)),
        Float32(get(record, "parent_reward", 0.0) * (operator == :crossover ? 1.0 : 0.0)),
    ]

    if feature_mode == :basic
        return vcat(base_context, op_flags)
    elseif feature_mode == :augmented
        return vcat(base_context, op_flags, relative_context, continuation_context, interaction_context)
    else
        error("Unknown option-value feature mode: $(feature_mode)")
    end
end

function option_override_feature_vector(record::OptionValueRecord;
                                        ranking_score::Real=0.0,
                                        ranking_gap_vs_entry_local::Real=0.0,
                                        local_margin::Real=record.local_margin_to_top)
    op_flags = Float32[
        record.operator == :mutate ? 1.0 : 0.0,
        record.operator == :crossover ? 1.0 : 0.0,
        record.operator == :terminate ? 1.0 : 0.0,
    ]
    return vcat(Float32[
        Float32(ranking_score),
        Float32(ranking_gap_vs_entry_local),
        Float32(local_margin),
        record.step1_local_utility,
        record.local_rank_fraction,
        record.local_margin_to_top,
        record.local_centered_utility,
        record.basin_rank_fraction,
        record.parent_rank_fraction,
        record.continuation_ratio,
        record.continuation_sensitive ? 1.0f0 : 0.0f0,
        record.local_ambiguous ? 1.0f0 : 0.0f0,
        record.topk_hit_fraction,
        record.best_reward_reached,
    ], op_flags)
end

function extract_option_value_dataset(bridge_runs::Vector{<:AbstractDict};
                                      task_name::Union{Nothing,String}=nothing,
                                      feature_mode::Symbol=:augmented,
                                      family::String="C",
                                      override_margin::Float64=0.02,
                                      strong_override_margin::Float64=0.05)
    dataset = OptionValueDataset()
    for run in bridge_runs
        probe = get(run, "probe", Dict{String,Any}())
        bridge = get(run, "bridge", Dict{String,Any}())
        raw_records = extract_option_subtrajectory_records(probe)
        filtered_records = Dict{String,Any}[]
        for record in raw_records
            String(get(record, "family", "")) == family || continue
            !isnothing(task_name) && String(get(record, "task_name", "")) != task_name && continue
            push!(filtered_records, record)
        end
        isempty(filtered_records) && continue

        local_surface_utility = Float32(get(bridge, "local_object_utility", get(bridge, "local_gain_vs_baseline", 0.0)))
        entry_context_utility = Float32(get(bridge, "entry_context_object_utility", get(bridge, "entry_context_gain_vs_baseline", 0.0)))
        contexts = _snapshot_relative_contexts(filtered_records;
            override_margin=override_margin,
            strong_override_margin=strong_override_margin)

        for (record, context) in zip(filtered_records, contexts)
            push!(dataset.records, OptionValueRecord(
                String(get(record, "snapshot_id", "")),
                String(get(record, "task_name", "unknown")),
                Int(get(record, "horizon", 0)),
                String(get(record, "object_id", "")),
                String(get(record, "entry_key", "")),
                String(get(record, "basin_scaffold", "")),
                String(get(record, "parent_smiles", "")),
                Symbol(get(record, "operator", "terminate")),
                option_value_feature_vector(record; feature_mode=feature_mode, snapshot_context=context),
                Float32(get(record, "step1_local_utility", 0.0)),
                Float32(get(record, "option_value", 0.0)),
                Float32(get(record, "continuation_gain", 0.0)),
                Float32(get(record, "best_reward_reached", 0.0)),
                Float32(get(record, "topk_hit_fraction", 0.0)),
                Float32(get(record, "gain_vs_baseline", 0.0)),
                local_surface_utility,
                entry_context_utility,
                Float32(get(context, "local_rank_fraction", 0.0)),
                Float32(get(context, "local_margin_to_top", 0.0)),
                Float32(get(context, "local_centered_utility", 0.0)),
                Float32(get(context, "basin_rank_fraction", 0.0)),
                Float32(get(context, "parent_rank_fraction", 0.0)),
                Float32(get(context, "continuation_ratio", 0.0)),
                Bool(get(context, "continuation_sensitive", 0.0) > 0.5),
                Bool(get(context, "local_ambiguous", 0.0) > 0.5),
                Bool(get(record, "all_degenerate", false)),
                Float32(get(context, "gain_vs_entry_local_candidate", 0.0)),
                Bool(get(context, "override_helpful", 0.0) > 0.5),
                Bool(get(context, "strong_override_helpful", 0.0) > 0.5),
                Bool(get(context, "entry_local_baseline", 0.0) > 0.5),
                Bool(get(context, "snapshot_has_override_opportunity", 0.0) > 0.5),
                Bool(get(context, "snapshot_has_strong_override_opportunity", 0.0) > 0.5),
            ))
        end
    end
    return dataset
end

function split_option_value_dataset(dataset::OptionValueDataset;
                                    train_fraction::Float64=0.8,
                                    rng::AbstractRNG=Random.MersenneTwister(0))
    isempty(dataset) && return OptionValueDataset(), OptionValueDataset()
    group_to_indices = Dict{String,Vector{Int}}()
    for (idx, record) in enumerate(dataset.records)
        push!(get!(group_to_indices, record.snapshot_id, Int[]), idx)
    end
    group_ids = collect(keys(group_to_indices))
    Random.shuffle!(rng, group_ids)
    n_train = clamp(round(Int, train_fraction * length(group_ids)), 1, length(group_ids))
    train_groups = Set(group_ids[1:n_train])
    train_records = OptionValueRecord[]
    val_records = OptionValueRecord[]
    for record in dataset.records
        if record.snapshot_id in train_groups
            push!(train_records, record)
        else
            push!(val_records, record)
        end
    end
    return OptionValueDataset(train_records), OptionValueDataset(val_records)
end

function option_value_dataset_stats(dataset::OptionValueDataset)
    if isempty(dataset)
        return Dict(
            "size" => 0,
            "feature_dim" => 0,
            "n_snapshots" => 0,
            "mean_option_value" => 0.0,
            "mean_step1_local_utility" => 0.0,
            "mean_continuation_gain" => 0.0,
            "mean_gain_vs_baseline" => 0.0,
            "mean_local_surface_utility" => 0.0,
            "mean_entry_context_surface_utility" => 0.0,
            "mean_gain_vs_entry_local_candidate" => 0.0,
            "continuation_sensitive_fraction" => 0.0,
            "local_ambiguous_fraction" => 0.0,
            "override_positive_fraction" => 0.0,
            "strong_override_positive_fraction" => 0.0,
            "entry_local_baseline_fraction" => 0.0,
            "snapshot_override_opportunity_fraction" => 0.0,
            "snapshot_strong_override_opportunity_fraction" => 0.0,
            "operator_counts" => Dict{String,Int}(),
            "mean_records_per_snapshot" => 0.0,
        )
    end

    operator_counts = Dict{String,Int}()
    snapshot_counts = Dict{String,Int}()
    for record in dataset.records
        op = String(record.operator)
        operator_counts[op] = get(operator_counts, op, 0) + 1
        snapshot_counts[record.snapshot_id] = get(snapshot_counts, record.snapshot_id, 0) + 1
    end

    return Dict(
        "size" => length(dataset.records),
        "feature_dim" => length(dataset.records[1].features),
        "n_snapshots" => length(snapshot_counts),
        "mean_option_value" => mean(Float64[r.option_value for r in dataset.records]),
        "mean_step1_local_utility" => mean(Float64[r.step1_local_utility for r in dataset.records]),
        "mean_continuation_gain" => mean(Float64[r.continuation_gain for r in dataset.records]),
        "mean_gain_vs_baseline" => mean(Float64[r.gain_vs_baseline for r in dataset.records]),
        "mean_local_surface_utility" => mean(Float64[r.local_surface_utility for r in dataset.records]),
        "mean_entry_context_surface_utility" => mean(Float64[r.entry_context_surface_utility for r in dataset.records]),
        "mean_gain_vs_entry_local_candidate" => mean(Float64[r.gain_vs_entry_local_candidate for r in dataset.records]),
        "continuation_sensitive_fraction" => mean(Float64[r.continuation_sensitive for r in dataset.records]),
        "local_ambiguous_fraction" => mean(Float64[r.local_ambiguous for r in dataset.records]),
        "override_positive_fraction" => mean(Float64[r.override_helpful for r in dataset.records]),
        "strong_override_positive_fraction" => mean(Float64[r.strong_override_helpful for r in dataset.records]),
        "entry_local_baseline_fraction" => mean(Float64[r.entry_local_baseline for r in dataset.records]),
        "snapshot_override_opportunity_fraction" => mean(Float64[r.snapshot_has_override_opportunity for r in dataset.records]),
        "snapshot_strong_override_opportunity_fraction" => mean(Float64[r.snapshot_has_strong_override_opportunity for r in dataset.records]),
        "operator_counts" => operator_counts,
        "mean_records_per_snapshot" => mean(Float64.(collect(values(snapshot_counts)))),
    )
end
