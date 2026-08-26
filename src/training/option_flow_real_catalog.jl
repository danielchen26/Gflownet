# Option-Flow real-artifact catalog utilities
#
# Evidence level E1: summary-only, leak-free proxy catalogs from real HE/PMO
# artifacts. This file intentionally avoids loading the full molecular/RDKit stack.
# It only depends on plain Dict summaries stored in he_episode_summary.jls and
# he_capacity_summary.jls.

using Serialization
using Statistics
using Random

const OPTION_FLOW_REAL_HEADLINE_FORBIDDEN_FIELDS = Set([
    "frontier_gain_sum", "frontier_gain_max", "delta_top1_max", "delta_top10_mean_max",
    "best_reward", "commits_applied", "step_count", "calls_used", "frontier_size_after",
    "frontier_after_summary", "improved_topk", "enters_topk", "no_valid_proposal",
    "budget_cap_reached", "budget_remaining_after", "calls_after", "best_smiles",
    "child_reward", "reward_delta", "frontier_utility_delta", "chosen_reward", "reward_q75", "reward_max",
])

const OPTION_FLOW_TYPED_PATH_FEATURE_DIM = 32

struct OptionFlowRealEncoding
    task_names::Vector{String}
    phases::Vector{String}
    config_names::Vector{String}
    source_families::Vector{String}
    feature_mode::Symbol
    grouping::Symbol
end

_sorted_unique(xs) = sort(unique(String.(xs)))

function _onehot(value::String, categories::Vector{String})
    out = zeros(Float32, length(categories))
    idx = findfirst(==(value), categories)
    idx === nothing || (out[idx] = 1.0f0)
    return out
end

function _num(x; default::Float64=0.0)
    x === nothing && return default
    x isa Missing && return default
    x isa Number && return Float64(x)
    try
        return parse(Float64, string(x))
    catch
        return default
    end
end

function _str(x; default::String="unknown")
    x === nothing && return default
    s = string(x)
    isempty(s) ? default : s
end

function _stable_hash64(s::AbstractString)
    h = UInt64(0xcbf29ce484222325)
    prime = UInt64(0x100000001b3)
    for b in codeunits(s)
        h ⊻= UInt64(b)
        h *= prime
    end
    return h
end

function _normalized_path(path::String)
    return replace(path, '\\' => '/')
end

function infer_option_flow_source_family(path::String)
    p = _normalized_path(path)
    if occursin("final_theory_direct_test", p)
        return "final_theory_v1"
    elseif occursin("level3_shape_then_tb/artifacts/heuristic_shape_then_tb", p)
        return "level3_heuristic_shape_then_tb"
    elseif occursin("level3_shape_then_tb/artifacts/learned_shape_then_tb", p)
        return "level3_learned_shape_then_tb"
    elseif occursin("truth_sprint_stage_b_f015_truth_tasksharded", p)
        return "stage_b_tb_he_full_locked"
    elseif occursin("edit_tb_pilot_smoke/artifacts/heuristic_baseline", p)
        return "edit_tb_heuristic_baseline"
    elseif occursin("edit_tb_pilot_smoke/artifacts/learned_rwmle", p)
        return "edit_tb_learned_rwmle"
    elseif occursin("edit_tb_pilot_smoke/artifacts/interpolated_050", p)
        return "edit_tb_interpolated_050"
    elseif occursin("checkpoints/", p)
        tail = split(p, "checkpoints/")[end]
        return first(split(tail, "/"))
    else
        return basename(dirname(path))
    end
end

function discover_he_summary_files(roots::Vector{String}; filename::String="he_episode_summary.jls")
    files = String[]
    for root in roots
        isdir(root) || continue
        for (dir, _, fs) in walkdir(root)
            if filename in fs
                push!(files, joinpath(dir, filename))
            end
        end
    end
    return sort(unique(files))
end

function _try_deserialize(path::String)
    try
        return deserialize(path), nothing
    catch err
        return nothing, string(typeof(err), ": ", sprint(showerror, err))
    end
end

function _summary_row_from_episode(ep::AbstractDict, path::String)
    before = get(ep, "frontier_before_summary", Dict{String,Any}())
    after = get(ep, "frontier_after_summary", Dict{String,Any}())
    before = before isa AbstractDict ? before : Dict{String,Any}()
    after = after isa AbstractDict ? after : Dict{String,Any}()

    top1_before = _num(get(before, "top1", 0.0))
    best_reward = _num(get(ep, "best_reward", 0.0))

    row = Dict{String,Any}(
        "summary_path" => path,
        "artifact_dir" => dirname(path),
        "source_family" => infer_option_flow_source_family(path),
        "task_name" => _str(get(ep, "task_name", "unknown")),
        "phase" => _str(get(ep, "phase", "unknown")),
        "config_name" => _str(get(ep, "config_name", "unknown")),
        "snapshot_id" => _str(get(ep, "snapshot_id", "unknown")),
        "episode_id" => _str(get(ep, "episode_id", "unknown")),
        "run_index" => _num(get(ep, "run_index", 0.0)),
        "episode_index" => _num(get(ep, "episode_index", 0.0)),
        "segment_index" => _num(get(ep, "segment_index", 0.0)),
        "budget_remaining_before" => _num(get(ep, "budget_remaining_before", 0.0)),
        "budget_remaining_after" => _num(get(ep, "budget_remaining_after", 0.0)),
        "calls_before" => _num(get(ep, "calls_before", 0.0)),
        "calls_after" => _num(get(ep, "calls_after", 0.0)),
        "calls_used" => _num(get(ep, "calls_used", 0.0)),
        "frontier_size_before" => _num(get(ep, "frontier_size_before", get(before, "size", 0.0))),
        "frontier_size_after" => _num(get(ep, "frontier_size_after", get(after, "size", 0.0))),
        "before_top1" => top1_before,
        "before_top10_mean" => _num(get(before, "top10_mean", 0.0)),
        "before_n_scaffolds" => _num(get(before, "n_scaffolds", 0.0)),
        "before_graph_unique_count" => _num(get(before, "graph_unique_count", 0.0)),
        "after_top1" => _num(get(after, "top1", 0.0)),
        "after_top10_mean" => _num(get(after, "top10_mean", 0.0)),
        "after_n_scaffolds" => _num(get(after, "n_scaffolds", 0.0)),
        "frontier_gain_sum" => _num(get(ep, "frontier_gain_sum", 0.0)),
        "frontier_gain_max" => _num(get(ep, "frontier_gain_max", 0.0)),
        "delta_top1_max" => _num(get(ep, "delta_top1_max", 0.0)),
        "delta_top10_mean_max" => _num(get(ep, "delta_top10_mean_max", 0.0)),
        "best_reward" => best_reward,
        "best_delta_vs_initial_top1" => best_reward - top1_before,
        "commits_applied" => _num(get(ep, "commits_applied", 0.0)),
        "step_count" => _num(get(ep, "step_count", 0.0)),
        "unique_parent_count" => _num(get(ep, "unique_parent_count", 0.0)),
        "unique_basin_count" => _num(get(ep, "unique_basin_count", 0.0)),
        "enters_topk" => Bool(get(ep, "enters_topk", false)),
        "improved_topk" => Bool(get(ep, "improved_topk", false)),
        "no_valid_proposal" => Bool(get(ep, "no_valid_proposal", false)),
        "budget_cap_reached" => Bool(get(ep, "budget_cap_reached", false)),
    )
    return row
end

function load_he_summary_rows(roots::Vector{String})
    rows = Dict{String,Any}[]
    files = discover_he_summary_files(roots)
    errors = Dict{String,String}()
    for path in files
        obj, err = _try_deserialize(path)
        if err !== nothing
            errors[path] = err
            continue
        end
        obj isa AbstractVector || (errors[path] = "expected Vector summary, got $(typeof(obj))"; continue)
        for ep in obj
            ep isa AbstractDict || continue
            push!(rows, _summary_row_from_episode(ep, path))
        end
    end
    return rows, files, errors
end

function _countmap_string(values)
    counts = Dict{String,Int}()
    for v in values
        s = string(v)
        counts[s] = get(counts, s, 0) + 1
    end
    return counts
end

function option_flow_real_artifact_audit(roots::Vector{String})
    rows, summary_files, summary_errors = load_he_summary_rows(roots)
    capacity_files = discover_he_summary_files(roots; filename="he_capacity_summary.jls")
    raw_diag_files = discover_he_summary_files(roots; filename="he_raw_diagnostics.jls")
    raw_traj_files = discover_he_summary_files(roots; filename="he_raw_trajectory.jls")

    function sample_status(files)
        isempty(files) && return Dict{String,Any}("count" => 0, "sample_deserializes" => false, "sample_error" => "no files")
        _, err = _try_deserialize(first(files))
        return Dict{String,Any}(
            "count" => length(files),
            "sample_path" => first(files),
            "sample_deserializes" => err === nothing,
            "sample_error" => err === nothing ? "" : err,
        )
    end

    snapshot_counts = _countmap_string([string(r["task_name"], "::", r["snapshot_id"]) for r in rows])
    repeated = Dict(k => v for (k, v) in snapshot_counts if v > 1)

    return Dict{String,Any}(
        "roots" => roots,
        "summary_files" => length(summary_files),
        "summary_errors" => summary_errors,
        "episode_count" => length(rows),
        "task_counts" => _countmap_string([r["task_name"] for r in rows]),
        "phase_counts" => _countmap_string([r["phase"] for r in rows]),
        "config_counts" => _countmap_string([r["config_name"] for r in rows]),
        "source_counts" => _countmap_string([r["source_family"] for r in rows]),
        "unique_task_snapshot_groups" => length(snapshot_counts),
        "repeated_task_snapshot_groups" => length(repeated),
        "max_task_snapshot_group_size" => isempty(snapshot_counts) ? 0 : maximum(values(snapshot_counts)),
        "repeated_task_snapshot_examples" => collect(Iterators.take(repeated, 10)),
        "capacity_status" => sample_status(capacity_files),
        "raw_diagnostics_status" => sample_status(raw_diag_files),
        "raw_trajectory_status" => sample_status(raw_traj_files),
    )
end

function build_option_flow_real_encoding(rows::Vector{Dict{String,Any}};
                                         feature_mode::Symbol=:leak_free,
                                         grouping::Symbol=:task_phase_budget100)
    isempty(rows) && throw(ArgumentError("cannot build real encoding from empty rows"))
    return OptionFlowRealEncoding(
        _sorted_unique([r["task_name"] for r in rows]),
        _sorted_unique([r["phase"] for r in rows]),
        _sorted_unique([r["config_name"] for r in rows]),
        _sorted_unique([r["source_family"] for r in rows]),
        feature_mode,
        grouping,
    )
end

_bucket(x::Real, width::Real) = floor(Int, Float64(x) / Float64(width))

function option_flow_proxy_group_key(row::Dict{String,Any}, grouping::Symbol)
    task = string(row["task_name"])
    phase = string(row["phase"])
    b100 = _bucket(_num(row["budget_remaining_before"]), 100.0)
    b50 = _bucket(_num(row["budget_remaining_before"]), 50.0)
    s50 = _bucket(_num(row["frontier_size_before"]), 50.0)
    t05 = _bucket(_num(row["before_top10_mean"]), 0.05)
    if grouping == :task_phase
        return string(task, "|", phase)
    elseif grouping == :task_phase_budget100
        return string(task, "|", phase, "|b100=", b100)
    elseif grouping == :task_phase_budget50
        return string(task, "|", phase, "|b50=", b50)
    elseif grouping == :task_phase_size50
        return string(task, "|", phase, "|s50=", s50)
    elseif grouping == :task_phase_top10_05
        return string(task, "|", phase, "|t05=", t05)
    elseif grouping == :task_phase_b100_t05
        return string(task, "|", phase, "|b100=", b100, "|t05=", t05)
    elseif grouping == :task_phase_b100_size50
        return string(task, "|", phase, "|b100=", b100, "|s50=", s50)
    else
        throw(ArgumentError("unknown proxy grouping $(grouping)"))
    end
end

function _safe_mean(xs)
    isempty(xs) && return 0.0
    return mean(Float64.(xs))
end

function _scale_budget(x) Float32(_num(x) / 3000.0) end
function _scale_calls(x) Float32(_num(x) / 10000.0) end
function _scale_size(x) Float32(_num(x) / 512.0) end
function _scale_scaffolds(x) Float32(_num(x) / 128.0) end
function _scale_run(x) Float32(_num(x) / 10.0) end
function _scale_episode(x) Float32(_num(x) / 100.0) end
function _scale_segment(x) Float32(_num(x) / 10.0) end

function option_flow_real_state_features(group_rows::Vector{Dict{String,Any}}, enc::OptionFlowRealEncoding)
    isempty(group_rows) && throw(ArgumentError("cannot build state features for empty group"))
    task = string(group_rows[1]["task_name"])
    phase = string(group_rows[1]["phase"])
    budgets = [_num(r["budget_remaining_before"]) for r in group_rows]
    calls = [_num(r["calls_before"]) for r in group_rows]
    sizes = [_num(r["frontier_size_before"]) for r in group_rows]
    top1s = [_num(r["before_top1"]) for r in group_rows]
    top10s = [_num(r["before_top10_mean"]) for r in group_rows]
    scaffolds = [_num(r["before_n_scaffolds"]) for r in group_rows]
    graphs = [_num(r["before_graph_unique_count"]) for r in group_rows]
    numeric = Float32[
        _safe_mean(budgets) / 3000.0,
        _safe_mean(calls) / 10000.0,
        _safe_mean(sizes) / 512.0,
        _safe_mean(top1s),
        _safe_mean(top10s),
        _safe_mean(scaffolds) / 128.0,
        _safe_mean(graphs) / 128.0,
        _safe_mean(top1s .- top10s),
        length(group_rows) / 32.0,
    ]
    return Float32.(vcat(_onehot(task, enc.task_names), _onehot(phase, enc.phases), numeric))
end

function _prop(obj, name::Symbol, default=nothing)
    try
        return getproperty(obj, name)
    catch
        return default
    end
end

function _mean_prop(logs, name::Symbol; scale::Float64=1.0, default::Float64=0.0)
    isempty(logs) && return default
    vals = Float64[]
    for log in logs
        push!(vals, _num(_prop(log, name, default)) / scale)
    end
    return mean(vals)
end

function _mean_candidate_len(logs, field::Symbol; scale::Float64=1.0)
    isempty(logs) && return 0.0
    vals = Float64[]
    for log in logs
        candidates = _prop(log, field, [])
        push!(vals, length(candidates) / scale)
    end
    return mean(vals)
end

function _mean_chosen_fraction(logs, field::Symbol)
    isempty(logs) && return 0.0
    vals = Float64[]
    for log in logs
        candidates = _prop(log, field, [])
        n = max(1, length(candidates))
        idx = clamp(round(Int, _num(_prop(log, :chosen_index, 1))), 1, n)
        push!(vals, n == 1 ? 0.0 : (idx - 1) / (n - 1))
    end
    return mean(vals)
end

function _mean_bool_prop(logs, name::Symbol)
    isempty(logs) && return 0.0
    vals = Float64[]
    for log in logs
        push!(vals, Bool(_prop(log, name, false)) ? 1.0 : 0.0)
    end
    return mean(vals)
end

function _operator_rate(operator_logs, opname::Symbol)
    isempty(operator_logs) && return 0.0
    count(log -> Symbol(_prop(log, :chosen_operator, :unknown)) == opname, operator_logs) / length(operator_logs)
end

function typed_path_feature_vector(diagnostics::AbstractDict)
    basin_logs = get(diagnostics, "basin_logs", [])
    parent_logs = get(diagnostics, "parent_logs", [])
    operator_logs = get(diagnostics, "operator_logs", [])

    feats = Float32[
        length(basin_logs) / 10.0,
        _mean_prop(basin_logs, :frontier_size; scale=512.0),
        _mean_prop(basin_logs, :frontier_top1),
        _mean_prop(basin_logs, :frontier_top10_mean),
        _mean_prop(basin_logs, :frontier_scaffold_count; scale=128.0),
        _mean_candidate_len(basin_logs, :candidate_basins; scale=32.0),
        _mean_chosen_fraction(basin_logs, :candidate_basins),
        _mean_prop(basin_logs, :chosen_basin_score),
        length(parent_logs) / 10.0,
        _mean_candidate_len(parent_logs, :candidate_parents; scale=32.0),
        _mean_chosen_fraction(parent_logs, :candidate_parents),
        _mean_prop(parent_logs, :chosen_parent_score),
        _mean_prop(parent_logs, :heuristic_margin),
        _mean_prop(parent_logs, :heuristic_entropy),
        _mean_bool_prop(parent_logs, :override_applied),
        _mean_bool_prop(parent_logs, :abstained_to_heuristic),
        length(operator_logs) / 10.0,
        _mean_candidate_len(operator_logs, :candidate_operators; scale=8.0),
        _mean_chosen_fraction(operator_logs, :candidate_operators),
        _mean_prop(operator_logs, :chosen_heuristic_score),
        _mean_prop(operator_logs, :heuristic_margin),
        _mean_prop(operator_logs, :heuristic_entropy),
        _mean_prop(operator_logs, :eligibility_score),
        _mean_bool_prop(operator_logs, :acted_on),
        _mean_bool_prop(operator_logs, :preserved_to_heuristic),
        _mean_bool_prop(operator_logs, :override_applied),
        _mean_bool_prop(operator_logs, :abstained_to_heuristic),
        _operator_rate(operator_logs, :mutate),
        _operator_rate(operator_logs, :crossover),
        _operator_rate(operator_logs, :terminate),
        _operator_rate(operator_logs, :sample),
        _operator_rate(operator_logs, :fragment),
    ]
    length(feats) == OPTION_FLOW_TYPED_PATH_FEATURE_DIM || throw(AssertionError("typed path feature dimension changed"))
    return feats
end

function augment_rows_with_typed_path_features(rows::Vector{Dict{String,Any}})
    out = Dict{String,Any}[]
    errors = Dict{String,String}()
    loaded = 0
    missing = 0
    for row in rows
        newrow = copy(row)
        raw_path = joinpath(string(row["artifact_dir"]), "he_raw_diagnostics.jls")
        if isfile(raw_path)
            obj, err = _try_deserialize(raw_path)
            if err === nothing && obj isa AbstractDict
                newrow["typed_path_features"] = typed_path_feature_vector(obj)
                newrow["typed_path_status"] = "loaded"
                loaded += 1
            else
                newrow["typed_path_features"] = zeros(Float32, OPTION_FLOW_TYPED_PATH_FEATURE_DIM)
                newrow["typed_path_status"] = "error"
                errors[raw_path] = err === nothing ? "raw diagnostics was not a Dict" : err
            end
        else
            newrow["typed_path_features"] = zeros(Float32, OPTION_FLOW_TYPED_PATH_FEATURE_DIM)
            newrow["typed_path_status"] = "missing"
            missing += 1
        end
        push!(out, newrow)
    end
    stats = Dict{String,Any}(
        "rows" => length(rows),
        "loaded" => loaded,
        "missing" => missing,
        "errors" => errors,
        "feature_dim" => OPTION_FLOW_TYPED_PATH_FEATURE_DIM,
    )
    return out, stats
end

function option_flow_real_option_features(row::Dict{String,Any}, enc::OptionFlowRealEncoding)
    config_feats = _onehot(string(row["config_name"]), enc.config_names)
    source_feats = _onehot(string(row["source_family"]), enc.source_families)
    schedule_feats = Float32[
        _scale_run(row["run_index"]),
        _scale_episode(row["episode_index"]),
        _scale_segment(row["segment_index"]),
    ]
    enc.feature_mode == :policy_only && return Float32.(vcat(config_feats, source_feats))

    precontext_feats = Float32[
        _scale_budget(row["budget_remaining_before"]),
        _scale_calls(row["calls_before"]),
        _scale_size(row["frontier_size_before"]),
        Float32(_num(row["before_top1"])),
        Float32(_num(row["before_top10_mean"])),
        _scale_scaffolds(row["before_n_scaffolds"]),
        _scale_scaffolds(row["before_graph_unique_count"]),
        Float32(_num(row["before_top1"]) - _num(row["before_top10_mean"])),
    ]

    if enc.feature_mode == :leak_free
        return Float32.(vcat(config_feats, source_feats, schedule_feats, precontext_feats))
    elseif enc.feature_mode == :typed_path
        typed_feats = haskey(row, "typed_path_features") ? Float32.(row["typed_path_features"]) : zeros(Float32, OPTION_FLOW_TYPED_PATH_FEATURE_DIM)
        return Float32.(vcat(config_feats, source_feats, schedule_feats, precontext_feats, typed_feats))
    elseif enc.feature_mode == :leaky_upper
        leaky_feats = Float32[
            Float32(_num(row["frontier_gain_sum"])),
            Float32(_num(row["frontier_gain_max"])),
            Float32(_num(row["delta_top10_mean_max"])),
            Float32(_num(row["delta_top1_max"])),
            Float32(_num(row["best_delta_vs_initial_top1"])),
            Float32(_num(row["commits_applied"]) / 32.0),
            Float32(_num(row["step_count"]) / 32.0),
            Float32(_num(row["calls_used"]) / 128.0),
        ]
        return Float32.(vcat(config_feats, source_feats, schedule_feats, precontext_feats, leaky_feats))
    else
        throw(ArgumentError("unknown feature_mode $(enc.feature_mode)"))
    end
end

function option_flow_real_utility(row::Dict{String,Any}; utility_key::Symbol=:frontier_gain_sum)
    if utility_key == :frontier_gain_sum
        return max(0.0, _num(row["frontier_gain_sum"]))
    elseif utility_key == :delta_top10_mean_max
        return max(0.0, _num(row["delta_top10_mean_max"]))
    elseif utility_key == :delta_top1_max
        return max(0.0, _num(row["delta_top1_max"]))
    elseif utility_key == :best_delta_vs_initial_top1
        return max(0.0, _num(row["best_delta_vs_initial_top1"]))
    elseif utility_key == :best_reward
        return max(0.0, _num(row["best_reward"]))
    else
        throw(ArgumentError("unknown real utility key $(utility_key)"))
    end
end

function catalog_has_option_feature_variation(catalog::OptionFlowCatalog; tol::Float64=1.0e-6)
    length(catalog.candidates) <= 1 && return false
    mat = reduce(hcat, [c.option_features for c in catalog.candidates])
    for i in 1:size(mat, 1)
        if maximum(mat[i, :]) - minimum(mat[i, :]) > tol
            return true
        end
    end
    return false
end

function _chunk_group_rows(rows::Vector{Dict{String,Any}}, max_candidates::Int, min_candidates::Int, rng::AbstractRNG)
    if max_candidates <= 0 || length(rows) <= max_candidates
        return [rows]
    end
    shuffled = shuffle(rng, rows)
    chunks = Vector{Vector{Dict{String,Any}}}()
    i = 1
    while i <= length(shuffled)
        j = min(length(shuffled), i + max_candidates - 1)
        chunk = shuffled[i:j]
        if length(chunk) < min_candidates && !isempty(chunks)
            append!(chunks[end], chunk)
        else
            push!(chunks, chunk)
        end
        i = j + 1
    end
    return chunks
end

function build_summary_proxy_catalogs(rows::Vector{Dict{String,Any}};
                                      grouping::Symbol=:task_phase_budget100,
                                      feature_mode::Symbol=:leak_free,
                                      utility_key::Symbol=:frontier_gain_sum,
                                      min_candidates::Int=3,
                                      max_candidates::Int=12,
                                      seed::Int=17,
                                      epsilon::Float64=1.0e-6)
    isempty(rows) && return OptionFlowCatalog[], build_option_flow_real_encoding([Dict("task_name"=>"unknown", "phase"=>"unknown", "config_name"=>"unknown", "source_family"=>"unknown")]; feature_mode=feature_mode, grouping=grouping), Dict{String,Any}()
    min_candidates >= 2 || throw(ArgumentError("min_candidates must be at least 2"))
    enc = build_option_flow_real_encoding(rows; feature_mode=feature_mode, grouping=grouping)
    groups = Dict{String,Vector{Dict{String,Any}}}()
    for row in rows
        key = option_flow_proxy_group_key(row, grouping)
        push!(get!(groups, key, Dict{String,Any}[]), row)
    end
    rng = MersenneTwister(seed)
    catalogs = OptionFlowCatalog[]
    skipped_small = 0
    for (key, group_rows) in sort(collect(groups); by=x -> x[1])
        length(group_rows) >= min_candidates || (skipped_small += 1; continue)
        for chunk in _chunk_group_rows(group_rows, max_candidates, min_candidates, rng)
            length(chunk) >= min_candidates || (skipped_small += 1; continue)
            state = option_flow_real_state_features(chunk, enc)
            option_features = [option_flow_real_option_features(row, enc) for row in chunk]
            utilities = [option_flow_real_utility(row; utility_key=utility_key) for row in chunk]
            option_ids = [string(row["episode_id"]) for row in chunk]
            metadata = [copy(row) for row in chunk]
            for md in metadata
                md["proxy_group_key"] = key
                md["feature_mode"] = string(feature_mode)
                md["utility_key"] = string(utility_key)
                md["evidence_level"] = "E1_summary_proxy"
                md["forbidden_headline_inputs"] = collect(OPTION_FLOW_REAL_HEADLINE_FORBIDDEN_FIELDS)
            end
            snapshot_id = _stable_hash64(string(grouping, "|", key, "|", length(catalogs) + 1))
            task = string(chunk[1]["task_name"])
            push!(catalogs, make_option_flow_catalog(task, snapshot_id, state, option_features, utilities;
                option_ids=option_ids,
                metadata=metadata,
                epsilon=epsilon))
        end
    end
    stats = Dict{String,Any}(
        "grouping" => string(grouping),
        "feature_mode" => string(feature_mode),
        "utility_key" => string(utility_key),
        "n_rows" => length(rows),
        "n_raw_groups" => length(groups),
        "n_catalogs" => length(catalogs),
        "skipped_small_groups" => skipped_small,
        "n_feature_varied_catalogs" => count(c -> catalog_has_option_feature_variation(c), catalogs),
        "n_informative_catalogs" => count(c -> c.informative, catalogs),
        "candidate_count_total" => sum(length(c.candidates) for c in catalogs; init=0),
    )
    return catalogs, enc, stats
end

function filter_real_headline_catalogs(catalogs::Vector{OptionFlowCatalog}; require_feature_variation::Bool=true)
    return [c for c in catalogs if c.informative && (!require_feature_variation || catalog_has_option_feature_variation(c))]
end

function option_flow_expected_utility(probs::AbstractVector{<:Real}, catalog::OptionFlowCatalog)
    return sum(Float64.(probs) .* option_flow_utilities(catalog))
end

function _catalog_ce_for_probs(probs::AbstractVector{<:Real}, catalog::OptionFlowCatalog)
    p = Float64.(probs)
    return -sum(Float64.(catalog.target_probs) .* log.(p .+ 1.0e-12))
end

function evaluate_probability_policy(catalogs::Vector{OptionFlowCatalog}, probs_list::Vector{Vector{Float32}})
    length(catalogs) == length(probs_list) || throw(ArgumentError("catalog/probability list length mismatch"))
    isempty(catalogs) && return Dict{String,Any}(
        "n_catalogs" => 0,
        "mean_ce" => NaN,
        "mean_expected_utility" => NaN,
        "mean_uniform_expected_utility" => NaN,
        "mean_expected_utility_lift" => NaN,
        "mean_expected_utility_lift_fraction" => NaN,
        "mean_top_quartile_mass" => NaN,
        "mean_entropy" => NaN,
    )
    ces = Float64[]
    eus = Float64[]
    uniform_eus = Float64[]
    top_masses = Float64[]
    entropies = Float64[]
    for (catalog, probs) in zip(catalogs, probs_list)
        push!(ces, _catalog_ce_for_probs(probs, catalog))
        push!(eus, option_flow_expected_utility(probs, catalog))
        uniform_probs = fill(Float32(1.0 / length(catalog.candidates)), length(catalog.candidates))
        push!(uniform_eus, option_flow_expected_utility(uniform_probs, catalog))
        push!(top_masses, top_utility_mass(probs, catalog))
        push!(entropies, option_entropy(probs))
    end
    lifts = eus .- uniform_eus
    return Dict{String,Any}(
        "n_catalogs" => length(catalogs),
        "mean_ce" => mean(ces),
        "mean_expected_utility" => mean(eus),
        "mean_uniform_expected_utility" => mean(uniform_eus),
        "mean_expected_utility_lift" => mean(lifts),
        "mean_expected_utility_lift_fraction" => mean(lifts ./ max.(abs.(uniform_eus), 1.0e-12)),
        "mean_top_quartile_mass" => mean(top_masses),
        "mean_entropy" => mean(entropies),
    )
end

function evaluate_real_option_flow_model(params, catalogs::Vector{OptionFlowCatalog})
    base = evaluate_option_flow_model(params, catalogs)
    probs_list = [option_flow_probs(params, c) for c in catalogs]
    real = evaluate_probability_policy(catalogs, probs_list)
    greedy_probs = Vector{Float32}[]
    for catalog in catalogs
        logits = option_flow_logits(params, catalog)
        p = zeros(Float32, length(logits))
        p[argmax(logits)] = 1.0f0
        push!(greedy_probs, p)
    end
    greedy = evaluate_probability_policy(catalogs, greedy_probs)
    oracle_probs = Vector{Float32}[]
    for catalog in catalogs
        utilities = option_flow_utilities(catalog)
        p = zeros(Float32, length(utilities))
        p[argmax(utilities)] = 1.0f0
        push!(oracle_probs, p)
    end
    oracle = evaluate_probability_policy(catalogs, oracle_probs)
    return merge(base, Dict{String,Any}(
        "mean_expected_utility" => real["mean_expected_utility"],
        "mean_uniform_expected_utility" => real["mean_uniform_expected_utility"],
        "mean_expected_utility_lift" => real["mean_expected_utility_lift"],
        "mean_expected_utility_lift_fraction" => real["mean_expected_utility_lift_fraction"],
        "greedy_model" => greedy,
        "oracle_greedy" => oracle,
    ))
end

function fit_metadata_prior(catalogs::Vector{OptionFlowCatalog}; metadata_key::String="config_name", epsilon::Float64=1.0e-6)
    sums = Dict{String,Float64}()
    counts = Dict{String,Int}()
    total_sum = 0.0
    total_count = 0
    for catalog in catalogs
        for c in catalog.candidates
            key = string(get(c.metadata, metadata_key, "unknown"))
            u = max(0.0, c.utility)
            sums[key] = get(sums, key, 0.0) + u
            counts[key] = get(counts, key, 0) + 1
            total_sum += u
            total_count += 1
        end
    end
    prior = Dict{String,Float64}()
    for (k, s) in sums
        prior[k] = (s + epsilon) / (counts[k] + epsilon)
    end
    overall = (total_sum + epsilon) / (max(total_count, 1) + epsilon)
    return prior, overall
end

function metadata_prior_probs(catalog::OptionFlowCatalog, prior::Dict{String,Float64}, overall::Float64; metadata_key::String="config_name")
    scores = Float64[]
    for c in catalog.candidates
        key = string(get(c.metadata, metadata_key, "unknown"))
        push!(scores, max(0.0, get(prior, key, overall)) + 1.0e-6)
    end
    total = sum(scores)
    return Float32.(scores ./ total)
end

function evaluate_metadata_prior_baseline(train_catalogs::Vector{OptionFlowCatalog},
                                          eval_catalogs::Vector{OptionFlowCatalog};
                                          metadata_key::String="config_name")
    prior, overall = fit_metadata_prior(train_catalogs; metadata_key=metadata_key)
    probs = [metadata_prior_probs(c, prior, overall; metadata_key=metadata_key) for c in eval_catalogs]
    metrics = evaluate_probability_policy(eval_catalogs, probs)
    return merge(metrics, Dict{String,Any}(
        "metadata_key" => metadata_key,
        "prior" => prior,
        "overall" => overall,
    ))
end

function uniform_policy_metrics(catalogs::Vector{OptionFlowCatalog})
    probs = [fill(Float32(1.0 / length(c.candidates)), length(c.candidates)) for c in catalogs]
    return evaluate_probability_policy(catalogs, probs)
end

function real_catalog_task_breakdown(params, catalogs::Vector{OptionFlowCatalog})
    tasks = sort(unique(c.task_name for c in catalogs))
    return Dict(task => evaluate_real_option_flow_model(params, [c for c in catalogs if c.task_name == task]) for task in tasks)
end

function summarize_real_catalog_collection(catalogs::Vector{OptionFlowCatalog})
    base = option_flow_catalog_stats(catalogs)
    task_counts = _countmap_string([c.task_name for c in catalogs])
    candidate_task_counts = Dict{String,Int}()
    feature_varied = count(c -> catalog_has_option_feature_variation(c), catalogs)
    for c in catalogs
        candidate_task_counts[c.task_name] = get(candidate_task_counts, c.task_name, 0) + length(c.candidates)
    end
    return merge(base, Dict{String,Any}(
        "task_catalog_counts" => task_counts,
        "task_candidate_counts" => candidate_task_counts,
        "feature_varied_catalogs" => feature_varied,
    ))
end

function e1_summary_proxy_gate(metrics::Dict{String,Any})
    ce_gain = _num(get(metrics, "mean_ce_vs_uniform", NaN); default=NaN)
    util_lift = _num(get(metrics, "mean_expected_utility_lift", NaN); default=NaN)
    top_lift = _num(get(metrics, "mean_top_quartile_lift", NaN); default=NaN)
    entropy = _num(get(metrics, "mean_entropy", NaN); default=NaN)
    uniform_entropy = _num(get(metrics, "mean_uniform_ce", NaN); default=NaN)
    rank_corr = _num(get(metrics, "mean_rank_correlation", NaN); default=NaN)
    pass = isfinite(ce_gain) && ce_gain > 0.0 &&
           isfinite(util_lift) && util_lift > 0.0 &&
           isfinite(top_lift) && top_lift > 0.0 &&
           isfinite(entropy) && isfinite(uniform_entropy) && entropy >= 0.40 * uniform_entropy &&
           isfinite(rank_corr) && rank_corr > 0.0
    return Dict{String,Any}(
        "gate" => "E1_summary_proxy",
        "pass" => pass,
        "ce_gain_positive" => ce_gain > 0.0,
        "utility_lift_positive" => util_lift > 0.0,
        "top_quartile_lift_positive" => top_lift > 0.0,
        "entropy_noncollapse" => entropy >= 0.40 * uniform_entropy,
        "rank_correlation_positive" => rank_corr > 0.0,
        "ce_gain" => ce_gain,
        "utility_lift" => util_lift,
        "top_quartile_lift" => top_lift,
        "entropy" => entropy,
        "uniform_entropy" => uniform_entropy,
        "rank_correlation" => rank_corr,
    )
end
