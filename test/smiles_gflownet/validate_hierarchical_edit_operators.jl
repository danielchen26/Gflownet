#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GFlowNet
using Statistics: mean
using Serialization

include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "rdkit_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "oracle_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "oracle_manager.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "pmo_benchmark.jl"))

const OUTDIR = joinpath(@__DIR__, "..", "..", "checkpoints", "hierarchical_edit_baseline")
mkpath(OUTDIR)

const TARGET_SMILES = Dict(
    "albuterol_similarity" => "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "celecoxib_rediscovery" => "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
)
const DEFAULT_BOOTSTRAP_SEEDS = ["CCO", "CCN", "CCC", "CCCl", "CC(=O)O", "c1ccccc1"]
const TASK_BOOTSTRAP_SEEDS = Dict(
    "albuterol_similarity" => ["CC(C)(C)N", "NCCO", "Oc1ccccc1", "Oc1ccc(O)cc1"],
    "celecoxib_rediscovery" => ["Cc1ccccc1", "NS(=O)(=O)c1ccccc1", "FC(F)(F)c1ccccc1", "c1ccn[nH]1"],
    "drd2" => ["N1CCCCC1", "CN1CCNCC1", "c1ccncc1", "CCN(CC)CC", "c1ccccc1Cl"],
)

parse_bool_env(name::String, default::Bool=false) = lowercase(strip(get(ENV, name, string(default)))) in ["1", "true", "yes", "y"]

function bootstrap_seed_pool(task_name::String; target_seed::Bool=true, target_smiles::Union{Nothing,String}=nothing)
    pool = String[]
    append!(pool, DEFAULT_BOOTSTRAP_SEEDS)
    append!(pool, get(TASK_BOOTSTRAP_SEEDS, task_name, String[]))
    if target_seed && !isnothing(target_smiles)
        push!(pool, target_smiles)
    end
    return unique(pool)
end

function bootstrap_frontier_warmup!(frontier_buffer::MolecularFrontierBuffer,
                                    oracle_mgr,
                                    budget_oracle_batch,
                                    vocab;
                                    task_name::String,
                                    target_smiles::Union{Nothing,String}=nothing,
                                    rounds::Int=1,
                                    max_parents::Int=4,
                                    max_candidates::Int=6,
                                    verbose::Bool=true)
    added = 0
    for round_idx in 1:rounds
        budget_exhausted(oracle_mgr) && break
        isempty(frontier_buffer) && break

        snapshot = create_frontier_snapshot(frontier_buffer;
            max_entries=min(48, length(frontier_buffer)),
            target_smiles=target_smiles,
            budget_remaining=budget_remaining(oracle_mgr),
            created_at_step=-round_idx)
        parents = frontier_topk(frontier_buffer, min(max_parents, length(frontier_buffer)); by=:reward)
        isempty(parents) && break

        candidate_smiles = String[]
        candidate_meta = Tuple{String,Symbol}[]
        seen = Set{String}()
        for parent in parents
            for op in trusted_edit_operators()
                op == :terminate && continue
                partner = op == :crossover ? choose_partner(snapshot, parent.smiles) : nothing
                proposals, _ = propose_edit_with_diagnostics(parent.smiles, op, vocab;
                    partner_smiles=partner,
                    max_candidates=max_candidates)
                for proposal in proposals
                    child = proposal.child_smiles
                    if isempty(child) || haskey(frontier_buffer.seen_smiles, child) || (child in seen)
                        continue
                    end
                    push!(candidate_smiles, child)
                    push!(candidate_meta, (parent.smiles, op))
                    push!(seen, child)
                end
            end
        end

        isempty(candidate_smiles) && continue
        rewards = budget_oracle_batch(candidate_smiles[1:min(end, max_parents * max_candidates)])
        metas = candidate_meta[1:length(rewards)]
        smiles = candidate_smiles[1:length(rewards)]
        order = sortperm(rewards, rev=true)
        round_added = 0
        for idx in order[1:min(length(order), max_parents)]
            reward = rewards[idx]
            reward <= 0.0 && continue
            parent_smiles, op = metas[idx]
            before = length(frontier_buffer)
            add_to_frontier!(frontier_buffer, smiles[idx];
                reward=reward,
                source=:edit,
                parent_smiles=parent_smiles,
                operator=op)
            round_added += length(frontier_buffer) > before ? 1 : 0
        end
        added += round_added
        verbose && println("Warmup[$(task_name)] round=$(round_idx) added=$(round_added) frontier=$(length(frontier_buffer))")
    end
    return added
end

function budget_oracle_fns(task_name::String, budget::Int)
    oracle_mgr = OracleManager(
        [OracleConfig(task_name, 1.0)],
        budget,
        0,
        Dict{String,Dict{String,Float64}}(),
        true,
    )

    OracleBridge.init_oracles!([task_name];
        cache_dir=joinpath(@__DIR__, "..", "..", "data", "tdc_cache"))

    oracle_cache = Dict{String, Float64}()

    function budget_oracle_batch(smiles_list::Vector{String})
        isempty(smiles_list) && return Float64[]

        uncached = String[]
        seen = Set{String}()
        for smiles in smiles_list
            isempty(smiles) && continue
            if !haskey(oracle_cache, smiles) && !(smiles in seen)
                push!(uncached, smiles)
                push!(seen, smiles)
            end
        end

        if !isempty(uncached) && !budget_exhausted(oracle_mgr)
            evaluate_molecules!(oracle_mgr, uncached)
            for smiles in uncached
                score = if haskey(oracle_mgr.cache, smiles) && haskey(oracle_mgr.cache[smiles], task_name)
                    lookup_score(oracle_mgr, smiles, task_name)
                else
                    0.0
                end
                oracle_cache[smiles] = score
            end
        end

        return Float64[get(oracle_cache, smiles, 0.0) for smiles in smiles_list]
    end

    function budget_oracle(smiles::String)
        haskey(oracle_cache, smiles) && return oracle_cache[smiles]
        scores = budget_oracle_batch([smiles])
        return isempty(scores) ? 0.0 : scores[1]
    end

    return oracle_mgr, budget_oracle, budget_oracle_batch
end

function validate_operator(task_name::String, operator::Symbol;
                           budget::Int=128,
                           trials::Int=12,
                           max_candidates::Int=6,
                           target_seed::Bool=true,
                           warmup_rounds::Int=1,
                           verbose::Bool=true)
    oracle_mgr, budget_oracle, budget_oracle_batch = budget_oracle_fns(task_name, budget)
    target_smiles = get(TARGET_SMILES, task_name, nothing)
    vocab = SMILESVocabulary()
    frontier_buffer = MolecularFrontierBuffer(5000)

    seed_pool = bootstrap_seed_pool(task_name; target_seed=target_seed, target_smiles=target_smiles)
    _seed_memories!(seed_pool, budget_oracle, vocab;
        reward_fn_batch=budget_oracle_batch,
        frontier_buffer=frontier_buffer,
        augmentation_count=!isnothing(target_smiles) && target_seed ? 8 : 2,
        verbose=verbose)
    warmup_added = bootstrap_frontier_warmup!(frontier_buffer, oracle_mgr, budget_oracle_batch, vocab;
        task_name=task_name,
        target_smiles=target_smiles,
        rounds=warmup_rounds,
        verbose=verbose)

    attempts = 0
    nonempty_trials = 0
    empty_after_filter_trials = 0
    raw_candidate_count = 0
    duplicate_candidate_count = 0
    empty_child_count = 0
    self_child_count = 0
    cached_child_count = 0
    valid_child_count = 0
    positive_delta_count = 0
    chosen_positive_delta_count = 0
    chosen_reward_deltas = Float64[]
    enters_topk_count = 0
    same_family_count = 0
    cross_family_count = 0
    no_scaffold_count = 0

    while attempts < trials && !budget_exhausted(oracle_mgr)
        attempts += 1
        snapshot = create_frontier_snapshot(frontier_buffer;
            max_entries=32,
            target_smiles=target_smiles,
            budget_remaining=budget_remaining(oracle_mgr),
            created_at_step=attempts)
        isempty(snapshot.entries) && break

        basin = sample_basin(snapshot)
        basin === nothing && break
        parent = sample_parent(snapshot; basin=basin)
        parent === nothing && break

        partner = operator == :crossover ? choose_partner(snapshot, parent.smiles) : nothing
        proposals, proposal_diag = propose_edit_with_diagnostics(parent.smiles, operator, vocab;
            partner_smiles=partner,
            max_candidates=max_candidates)

        raw_candidate_count += Int(proposal_diag["raw_candidate_count"])
        duplicate_candidate_count += Int(proposal_diag["duplicate_candidate_count"])
        empty_child_count += Int(proposal_diag["empty_child_count"])
        self_child_count += Int(proposal_diag["self_child_count"])

        filtered = EditProposal[]
        for proposal in proposals
            if proposal.operator != :terminate && haskey(frontier_buffer.seen_smiles, proposal.child_smiles)
                cached_child_count += 1
                continue
            end
            push!(filtered, proposal)
        end

        if isempty(filtered)
            empty_after_filter_trials += 1
            continue
        end

        nonempty_trials += 1
        child_smiles = [p.child_smiles for p in filtered]
        valid_child_count += length(child_smiles)

        rewards = budget_oracle_batch(child_smiles)
        positive_delta_count += count(r -> r > parent.reward, rewards)

        best_idx = argmax(rewards)
        chosen_smiles = child_smiles[best_idx]
        chosen_reward = rewards[best_idx]
        delta = chosen_reward - parent.reward
        push!(chosen_reward_deltas, delta)
        chosen_positive_delta_count += delta > 0 ? 1 : 0

        before = frontier_quality_summary(frontier_buffer; topk=10)
        add_to_frontier!(frontier_buffer, chosen_smiles;
            reward=chosen_reward,
            source=:edit,
            parent_smiles=parent.smiles,
            operator=operator)
        after = frontier_quality_summary(frontier_buffer; topk=10)
        utility = compute_frontier_utility_delta(before, after, chosen_smiles, get_scaffold(chosen_smiles))
        enters_topk_count += utility["enters_topk"] ? 1 : 0

        family_transition = if isempty(parent.scaffold) || isempty(get_scaffold(chosen_smiles))
            :no_scaffold
        elseif parent.scaffold == get_scaffold(chosen_smiles)
            :same_family
        else
            :cross_family
        end
        same_family_count += family_transition == :same_family ? 1 : 0
        cross_family_count += family_transition == :cross_family ? 1 : 0
        no_scaffold_count += family_transition == :no_scaffold ? 1 : 0
    end

    return Dict(
        "task_name" => task_name,
        "operator" => String(operator),
        "budget" => budget,
        "trials_requested" => trials,
        "trials_attempted" => attempts,
        "nonempty_trials" => nonempty_trials,
        "empty_after_filter_trials" => empty_after_filter_trials,
        "raw_candidate_count" => raw_candidate_count,
        "duplicate_candidate_count" => duplicate_candidate_count,
        "empty_child_count" => empty_child_count,
        "self_child_count" => self_child_count,
        "cached_child_count" => cached_child_count,
        "valid_child_count" => valid_child_count,
        "positive_reward_delta_rate" => valid_child_count == 0 ? 0.0 : positive_delta_count / valid_child_count,
        "chosen_positive_delta_rate" => nonempty_trials == 0 ? 0.0 : chosen_positive_delta_count / nonempty_trials,
        "mean_chosen_reward_delta" => isempty(chosen_reward_deltas) ? 0.0 : mean(chosen_reward_deltas),
        "topk_entry_rate" => nonempty_trials == 0 ? 0.0 : enters_topk_count / nonempty_trials,
        "same_family_rate" => nonempty_trials == 0 ? 0.0 : same_family_count / nonempty_trials,
        "cross_family_rate" => nonempty_trials == 0 ? 0.0 : cross_family_count / nonempty_trials,
        "no_scaffold_rate" => nonempty_trials == 0 ? 0.0 : no_scaffold_count / nonempty_trials,
        "warmup_added" => warmup_added,
        "oracle_calls_used" => oracle_mgr.calls_used,
        "frontier_size" => length(frontier_buffer),
    )
end

const TASK = get(ENV, "PMO_TASK", "albuterol_similarity")
const BUDGET = parse(Int, get(ENV, "PMO_BUDGET", "128"))
const TRIALS = parse(Int, get(ENV, "HE_TRIALS", "12"))
const TARGET_SEED = parse_bool_env("HE_TARGET_SEED", true)
const WARMUP_ROUNDS = parse(Int, get(ENV, "HE_BOOTSTRAP_WARMUP_ROUNDS", "1"))

results = Dict{String,Any}()
for op in trusted_edit_operators()
    println("Validating operator $(op) on task $(TASK) ...")
    results[String(op)] = validate_operator(TASK, op;
        budget=BUDGET,
        trials=TRIALS,
        target_seed=TARGET_SEED,
        warmup_rounds=WARMUP_ROUNDS,
        verbose=true)
end

outfile = joinpath(OUTDIR, "operator_validation_$(TASK).jls")
serialize(outfile, results)
println("Saved operator validation results to: $(outfile)")

for op in sort(collect(keys(results)))
    r = results[op]
    op_name = r["operator"]
    chosen_rate = round(r["chosen_positive_delta_rate"], digits=3)
    mean_delta = round(r["mean_chosen_reward_delta"], digits=4)
    topk_rate = round(r["topk_entry_rate"], digits=3)
    calls_used = r["oracle_calls_used"]
    raw_count = r["raw_candidate_count"]
    cached = r["cached_child_count"]
    empty_trials = r["empty_after_filter_trials"]
    println("$(op_name): chosen+Δ=$(chosen_rate) | meanΔ=$(mean_delta) | topk=$(topk_rate) | raw=$(raw_count) | cached=$(cached) | empty_trials=$(empty_trials) | calls=$(calls_used)")
end
