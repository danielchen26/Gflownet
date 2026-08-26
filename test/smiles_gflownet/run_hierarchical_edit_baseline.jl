#!/usr/bin/env julia
# Hierarchical Edit Baseline Runner
#
# Stage A0 runner:
# - canonical graph-identity accounting
# - source attribution (seed / augment / warmup / edit)
# - structural ablation matrix
# - optional full hierarchical edit rollout

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GFlowNet
using Random
using Serialization
using Statistics: mean, std

include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "rdkit_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "python", "oracle_bridge.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "oracle_manager.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils", "visualization", "core", "pmo_benchmark.jl"))

const OUTDIR = joinpath(@__DIR__, "..", "..", "checkpoints", "hierarchical_edit_baseline")
mkpath(OUTDIR)

const DEFAULT_TASKS = [
    "albuterol_similarity",
    "celecoxib_rediscovery",
    "drd2",
]

const TARGET_SMILES = Dict(
    "albuterol_similarity" => "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "celecoxib_rediscovery" => "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "mestranol_similarity" => "C#C[C@]1(O)CC[C@H]2[C@@H]3CCc4cc(OC)ccc4[C@H]3CC[C@@]21C",
    "thiothixene_rediscovery" => "C=C(c1ccc(S(=O)(=O)N2CCN(C)CC2)cc1)c1cc2c(s1)Cc1ccccc1-2",
    "troglitazone_rediscovery" => "Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O",
    "aripiprazole_similarity" => "Clc1ccc2c(c1)N(CCCCN1CCN(c3nsc4ccccc34)CC1)C(=O)CC2",
)

const DEFAULT_BOOTSTRAP_SEEDS = [
    "CCO",
    "CCN",
    "CCC",
    "CCCl",
    "CC(=O)O",
    "c1ccccc1",
]

const TASK_BOOTSTRAP_SEEDS = Dict(
    "albuterol_similarity" => [
        "CC(C)(C)N",
        "NCCO",
        "Oc1ccccc1",
        "Oc1ccc(O)cc1",
        "CC(C)O",
    ],
    "celecoxib_rediscovery" => [
        "Cc1ccccc1",
        "NS(=O)(=O)c1ccccc1",
        "FC(F)(F)c1ccccc1",
        "c1ccn[nH]1",
        "c1ccccc1S(N)(=O)=O",
    ],
    "drd2" => [
        "N1CCCCC1",
        "CN1CCNCC1",
        "c1ccncc1",
        "CCN(CC)CC",
        "c1ccccc1Cl",
    ],
    "mestranol_similarity" => [
        "c1cc(OC)ccc1",           # anisole (methoxy aromatic ring)
        "C1CCC2CCCCC2C1",         # decalin (fused cyclohexanes)
        "C#CO",                    # ethynol (ethynyl fragment)
        "C1CCC2(CC1)CCCC2",       # spiro-bicyclic
        "CC12CCC3CCCCC3C1CCC2O",  # steroid skeleton fragment
    ],
    "thiothixene_rediscovery" => [
        "c1ccc2c(c1)Sc1ccccc1C2", # thioxanthene core
        "CS(=O)(=O)N1CCNCC1",     # sulfonyl-piperazine
        "CN1CCNCC1",              # N-methylpiperazine
        "c1ccc(S(=O)(=O)N)cc1",   # benzenesulfonamide
        "c1ccc2sccc2c1",          # benzothiophene
    ],
    "troglitazone_rediscovery" => [
        "Cc1cc(O)c(C)c(C)c1O",   # trimethylhydroquinone
        "O=C1NC(=O)CS1",          # thiazolidinedione
        "c1ccc(CC2CCCC2)cc1",     # phenylcyclopentane
        "OCC(O)CO",               # glycerol
        "CC1(C)CCc2ccccc2O1",     # chroman
    ],
    "aripiprazole_similarity" => [
        "Clc1ccccc1",             # chlorobenzene
        "O=C1CCc2ccccc2N1",       # 2-oxindole
        "N1CCNCC1",               # piperazine
        "c1ccnc2ccccc12",         # quinoline
        "NCCCCN",                 # butanediamine linker
    ],
)

function parse_list_env(name::String, default::Vector{String})
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    return [String(strip(x)) for x in split(raw, ',') if !isempty(strip(x))]
end

function parse_float_list_env(name::String, default::Vector{Float64})
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    return [parse(Float64, strip(x)) for x in split(raw, ',') if !isempty(strip(x))]
end
parse_bool_env(name::String, default::Bool=false) = lowercase(strip(get(ENV, name, string(default)))) in ["1", "true", "yes", "y"]

function task_regime(task_name::String, target_smiles)
    return isnothing(target_smiles) ? :sparse_or_property : :structural
end

function bootstrap_seed_pool(task_name::String;
                             user_seed_smiles::Vector{String}=String[],
                             target_smiles::Union{Nothing,String}=nothing,
                             target_seed::Bool=true)
    seed_pool = String[]
    append!(seed_pool, DEFAULT_BOOTSTRAP_SEEDS)
    append!(seed_pool, get(TASK_BOOTSTRAP_SEEDS, task_name, String[]))
    append!(seed_pool, user_seed_smiles)
    if target_seed && !isnothing(target_smiles)
        push!(seed_pool, target_smiles)
    end
    canonical_pool = [canonicalize_smiles_identity(s) for s in seed_pool if !isempty(strip(s))]
    return unique(filter(!isempty, canonical_pool))
end

function bootstrap_augmentation_count(task_name::String, target_smiles, target_seed::Bool; enable_augmentation::Bool=true)
    (!target_seed || !enable_augmentation) && return 0
    return task_regime(task_name, target_smiles) == :structural ? 12 : 4
end

function build_budget_oracles(task_name::String, budget::Int)
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

        uncached_raw = String[]
        seen = Set{String}()
        for smiles in smiles_list
            isempty(smiles) && continue
            canonical = canonicalize_smiles_identity(smiles)
            if !haskey(oracle_cache, canonical) && !(canonical in seen)
                push!(uncached_raw, smiles)
                push!(seen, canonical)
            end
        end

        if !isempty(uncached_raw) && !budget_exhausted(oracle_mgr)
            evaluate_molecules!(oracle_mgr, uncached_raw)
            for smiles in uncached_raw
                canonical = canonicalize_smiles_identity(smiles)
                oracle_cache[canonical] = lookup_score(oracle_mgr, smiles, task_name)
            end
        end

        return Float64[get(oracle_cache, canonicalize_smiles_identity(smiles), 0.0) for smiles in smiles_list]
    end

    function budget_oracle(smiles::String)
        canonical = canonicalize_smiles_identity(smiles)
        haskey(oracle_cache, canonical) && return oracle_cache[canonical]
        scores = budget_oracle_batch([smiles])
        return isempty(scores) ? 0.0 : scores[1]
    end

    return oracle_mgr, budget_oracle, budget_oracle_batch
end

function bootstrap_frontier_warmup!(task_name::String,
                                    frontier_buffer::MolecularFrontierBuffer,
                                    oracle_mgr,
                                    budget_oracle_batch,
                                    vocab;
                                    target_smiles::Union{Nothing,String}=nothing,
                                    rounds::Int=1,
                                    max_parents::Int=4,
                                    max_candidates::Int=6,
                                    verbose::Bool=true)
    added_total = 0
    evaluated_total = 0

    for round_idx in 1:rounds
        budget_exhausted(oracle_mgr) && break
        isempty(frontier_buffer) && break

        snapshot = create_frontier_snapshot(frontier_buffer;
            max_entries=min(48, length(frontier_buffer)),
            target_smiles=target_smiles,
            budget_remaining=budget_remaining(oracle_mgr),
            created_at_step=-round_idx)
        isempty(snapshot.entries) && break

        parents = frontier_topk(frontier_buffer, min(max_parents, length(frontier_buffer)); by=:reward)
        candidate_smiles = String[]
        candidate_meta = Tuple{String,Symbol}[]
        seen_candidates = Set{String}()

        for parent in parents
            for op in trusted_edit_operators()
                op == :terminate && continue
                partner = op == :crossover ? choose_partner(snapshot, parent.smiles) : nothing
                proposals, _ = propose_edit_with_diagnostics(parent.smiles, op, vocab;
                    partner_smiles=partner,
                    max_candidates=max_candidates)
                for proposal in proposals
                    child = canonicalize_smiles_identity(proposal.child_smiles)
                    if isempty(child) || haskey(frontier_buffer.seen_smiles, child) || (child in seen_candidates)
                        continue
                    end
                    push!(candidate_smiles, child)
                    push!(candidate_meta, (parent.smiles, op))
                    push!(seen_candidates, child)
                end
            end
        end

        isempty(candidate_smiles) && continue
        eval_limit = min(length(candidate_smiles), max(max_parents * max_candidates, 1))
        candidate_smiles = candidate_smiles[1:eval_limit]
        candidate_meta = candidate_meta[1:eval_limit]
        rewards = budget_oracle_batch(candidate_smiles)
        evaluated_total += length(candidate_smiles)

        idxs = sortperm(rewards, rev=true)
        round_added = 0
        for idx in idxs[1:min(length(idxs), max_parents)]
            reward = rewards[idx]
            child = candidate_smiles[idx]
            reward <= 0.0 && continue
            haskey(frontier_buffer.seen_smiles, child) && continue
            parent_smiles, op = candidate_meta[idx]
            add_to_frontier!(frontier_buffer, child;
                reward=reward,
                source=:warmup,
                parent_smiles=parent_smiles,
                operator=op)
            added_total += 1
            round_added += 1
        end

        verbose && println("HE[$(task_name)] warmup_round=$(round_idx) evaluated=$(length(candidate_smiles)) added=$(round_added) frontier=$(length(frontier_buffer))")
        round_added == 0 && break
    end

    return Dict(
        "warmup_rounds" => rounds,
        "warmup_added" => added_total,
        "warmup_evaluated" => evaluated_total,
    )
end

function hierarchical_result(task_name::String,
                             oracle_mgr,
                             diagnostics_buffer::HierarchicalEditDiagnosticsBuffer,
                             frontier_buffer::MolecularFrontierBuffer;
                             regime_name::String,
                             run_stats::Dict{String,Any}=Dict{String,Any}(),
                             bootstrap_stats::Dict{String,Any}=Dict{String,Any}())
    top_scores = sort(oracle_mgr.top_scores, rev=true)
    top1 = isempty(top_scores) ? 0.0 : top_scores[1]
    top10_mean = isempty(top_scores) ? 0.0 : mean(top_scores[1:min(10, length(top_scores))])
    auc = compute_auc_top10(oracle_mgr)
    diversity = 0.0
    unique_molecules = length(keys(oracle_mgr.cache))
    source_summary = frontier_source_summary(frontier_buffer; topk=10)

    return PMOResult(
        task_name,
        auc,
        top1,
        top10_mean,
        diversity,
        oracle_mgr.calls_used,
        unique_molecules,
    ), Dict(
        "regime_name" => regime_name,
        "diagnostic_stats" => decision_log_stats(diagnostics_buffer),
        "proposal_stats" => proposal_log_stats(diagnostics_buffer),
        "frontier_stats" => frontier_stats(frontier_buffer),
        "frontier_source_summary" => source_summary,
        "decision_logs" => length(diagnostics_buffer.logs),
        "proposal_logs" => length(diagnostics_buffer.proposal_logs),
        "basin_logs" => length(diagnostics_buffer.basin_logs),
        "parent_logs" => length(diagnostics_buffer.parent_logs),
        "operator_logs" => length(diagnostics_buffer.operator_logs),
        "decision_logs_raw" => copy(diagnostics_buffer.logs),
        "proposal_logs_raw" => copy(diagnostics_buffer.proposal_logs),
        "basin_logs_raw" => copy(diagnostics_buffer.basin_logs),
        "parent_logs_raw" => copy(diagnostics_buffer.parent_logs),
        "operator_logs_raw" => copy(diagnostics_buffer.operator_logs),
        "graph_unique_molecules" => unique_molecules,
        "budget_fraction_used" => oracle_mgr.budget > 0 ? oracle_mgr.calls_used / oracle_mgr.budget : 0.0,
        "run_stats" => run_stats,
        "bootstrap_stats" => bootstrap_stats,
    )
end

function run_hierarchical_edit_pmo_task(task_name::String;
                                        budget::Int=256,
                                        n_episodes::Int=48,
                                        target_seed::Bool=true,
                                        seed_smiles::Vector{String}=String[],
                                        enable_augmentation::Bool=true,
                                        enable_warmup::Bool=true,
                                        bootstrap_warmup_rounds::Int=1,
                                        run_episodes::Bool=true,
                                        regime_name::String="full",
                                        config::HierarchicalEditConfig=HierarchicalEditConfig(),
                                        verbose::Bool=true)
    oracle_mgr, budget_oracle, budget_oracle_batch = build_budget_oracles(task_name, budget)
    frontier_buffer = MolecularFrontierBuffer(5000)
    trajectory_buffer = EditTrajectoryBuffer(10000)
    diagnostics_buffer = HierarchicalEditDiagnosticsBuffer(10000)
    vocab = SMILESVocabulary()
    target_smiles = get(TARGET_SMILES, task_name, nothing)

    seed_pool = bootstrap_seed_pool(task_name;
        user_seed_smiles=seed_smiles,
        target_smiles=target_smiles,
        target_seed=target_seed)
    seed_stats = _seed_memories!(seed_pool, budget_oracle, vocab;
        reward_fn_batch=budget_oracle_batch,
        frontier_buffer=frontier_buffer,
        augmentation_count=bootstrap_augmentation_count(task_name, target_smiles, target_seed; enable_augmentation=enable_augmentation),
        verbose=verbose)

    warmup_stats = enable_warmup ? bootstrap_frontier_warmup!(task_name, frontier_buffer, oracle_mgr, budget_oracle_batch, vocab;
        target_smiles=target_smiles,
        rounds=bootstrap_warmup_rounds,
        verbose=verbose) : Dict("warmup_rounds" => 0, "warmup_added" => 0, "warmup_evaluated" => 0)

    episodes_executed = 0
    stagnant_episodes = 0
    stagnant_stop = false
    if run_episodes
        for episode_idx in 1:n_episodes
            budget_exhausted(oracle_mgr) && break
            ep = run_hierarchical_edit_episode!(frontier_buffer, trajectory_buffer, budget_oracle, vocab;
                reward_fn_batch=budget_oracle_batch,
                diagnostics_buffer=diagnostics_buffer,
                config=config,
                target_smiles=target_smiles,
                budget_remaining=budget_remaining(oracle_mgr),
                created_at_step=episode_idx,
                task_name=task_name)
            episodes_executed += 1

            if ep.commits_applied == 0
                stagnant_episodes += 1
            elseif ep.improved_topk
                stagnant_episodes = 0
            else
                stagnant_episodes += 1
            end

            verbose && println(
                "HE[$(task_name):$(regime_name)] episode=$(episode_idx) calls=$(oracle_mgr.calls_used)/$(budget) commits=$(ep.commits_applied) improved=$(ep.improved_topk) best=$(round(ep.best_reward, digits=4)) proposal_logs=$(length(diagnostics_buffer.proposal_logs))"
            )
            if stagnant_episodes >= 8
                stagnant_stop = true
                break
            end
        end
    end

    return hierarchical_result(task_name, oracle_mgr, diagnostics_buffer, frontier_buffer;
        regime_name=regime_name,
        run_stats=Dict{String,Any}(
            "episodes_executed" => episodes_executed,
            "stagnant_stop" => stagnant_stop,
            "final_stagnant_episodes" => stagnant_episodes,
            "budget_used" => oracle_mgr.calls_used,
            "budget_total" => budget,
        ),
        bootstrap_stats=Dict(
            "seed_stats" => seed_stats,
            "warmup_stats" => warmup_stats,
            "regime" => String(task_regime(task_name, target_smiles)),
            "augmentation_enabled" => enable_augmentation,
            "warmup_enabled" => enable_warmup,
            "episodes_enabled" => run_episodes,
        ))
end

function print_attribution_report(task::String, name::String, result, extra)
    println()
    println("=== Attribution Report: $(task) / $(name) ===")
    source_summary = extra["frontier_source_summary"]
    fstats = extra["frontier_stats"]

    # Overall frontier composition
    println("  Frontier size: $(fstats["size"]) | Graph-unique: $(fstats["graph_unique_count"]) | Ratio: $(round(fstats["string_vs_graph_ratio"], digits=3))")
    println("  Scaffolds: $(fstats["n_scaffolds"]) | Mean reward: $(round(fstats["mean_reward"], digits=4)) | Max reward: $(round(fstats["max_reward"], digits=4))")

    # Source composition for full frontier and top-k
    for (label, counts) in [("Frontier (all)", source_summary["overall"]), ("Top-10", source_summary["topk"])]
        total = max(sum(values(counts)), 1)
        parts = String[]
        for src in ["seed", "augment", "warmup", "edit", "ga", "model"]
            n = get(counts, src, 0)
            n > 0 && push!(parts, "$(src)=$(n)($(round(100*n/total, digits=1))%)")
        end
        println("  $(label): $(join(parts, ", "))")
    end

    # Key metrics
    println("  AUC=$(round(result.auc_top10, digits=4)) | Top1=$(round(result.top1, digits=4)) | Top10=$(round(result.top10_mean, digits=4))")
    println("  Oracle calls: $(result.n_oracle_calls) | Graph-unique molecules: $(extra["graph_unique_molecules"])")
    oracle_eff = extra["graph_unique_molecules"] > 0 ? round(100 * extra["graph_unique_molecules"] / max(result.n_oracle_calls, 1), digits=1) : 0.0
    println("  Oracle efficiency (graph-unique / calls): $(oracle_eff)%")

    run_stats = extra["run_stats"]
    println("  Budget fraction used: $(round(100 * extra["budget_fraction_used"], digits=1))% | Episodes executed: $(run_stats["episodes_executed"]) | Stagnant stop: $(run_stats["stagnant_stop"])")

    # Actual top-1 attribution
    topk_sources = source_summary["topk"]
    top1_src = get(source_summary, "top1_source", "none")
    top1_op = get(source_summary, "top1_operator", "none")
    println("  Top-1 source: $(top1_src) | Top-1 operator: $(top1_op)")

    # Edit contribution fraction (fraction of top-10 from :edit)
    edit_count = get(topk_sources, "edit", 0)
    topk_total = max(source_summary["topk_size"], 1)
    println("  Edit contribution to top-10: $(edit_count)/$(topk_total) ($(round(100*edit_count/topk_total, digits=1))%)")

    edit_op_counts = get(source_summary, "topk_edit_operators", Dict{String,Int}())
    if !isempty(edit_op_counts)
        op_total = max(sum(values(edit_op_counts)), 1)
        op_parts = String[]
        for op in ["mutate", "crossover", "terminate", "augment", "sample"]
            n = get(edit_op_counts, op, 0)
            n > 0 && push!(op_parts, "$(op)=$(n)($(round(100*n/op_total, digits=1))%)")
        end
        !isempty(op_parts) && println("  Edit operator split in top-10: $(join(op_parts, ", "))")
    end

    # Proposal stats (if episodes were run)
    pstats = extra["proposal_stats"]
    if pstats["size"] > 0
        println("  Proposal stats: empty_after_filter=$(round(pstats["empty_after_filter_fraction"], digits=3)) | chosen+Δ=$(round(pstats["chosen_positive_delta_fraction"], digits=3)) | mean_raw=$(round(pstats["mean_raw_candidate_count"], digits=2))")
    end
    println()
end

function print_regime_comparison(task::String, task_results::Dict)
    println()
    println("=== Regime Comparison: $(task) ===")
    println("  Regime                  | AUC    | Top1   | Top10  | Calls | GraphUniq | Edit%")
    println("  " * "-"^80)
    for name in ["seed_only", "seed_plus_augmentation", "seed_plus_warmup", "seed_warmup_episodes",
                 "mutate_only", "crossover_only", "mixed_trusted"]
        !haskey(task_results, name) && continue
        r = task_results[name]["hierarchical_edit"]
        e = task_results[name]["extra"]
        topk_sources = e["frontier_source_summary"]["topk"]
        edit_pct = round(100 * get(topk_sources, "edit", 0) / max(e["frontier_source_summary"]["topk_size"], 1), digits=1)
        padded_name = rpad(name, 25)
        println("  $(padded_name) | $(lpad(string(round(r.auc_top10, digits=4)), 6)) | $(lpad(string(round(r.top1, digits=4)), 6)) | $(lpad(string(round(r.top10_mean, digits=4)), 6)) | $(lpad(string(r.n_oracle_calls), 5)) | $(lpad(string(e["graph_unique_molecules"]), 9)) | $(lpad(string(edit_pct), 5))%")
    end

    # Compute edit-vs-bootstrap delta
    if haskey(task_results, "seed_plus_warmup") && haskey(task_results, "seed_warmup_episodes")
        warmup_top10 = task_results["seed_plus_warmup"]["hierarchical_edit"].top10_mean
        episodes_top10 = task_results["seed_warmup_episodes"]["hierarchical_edit"].top10_mean
        delta = episodes_top10 - warmup_top10
        println()
        println("  Edit-vs-bootstrap delta (Top10): $(round(delta, digits=4)) ($(delta > 0 ? "edit helps" : delta < 0 ? "edit hurts" : "no effect"))")
    end
    println()
end

function run_ablation_matrix(tasks::Vector{String};
                             budget::Int,
                             n_episodes::Int,
                             target_seed::Bool,
                             bootstrap_warmup_rounds::Int,
                             max_step_attempts::Int,
                             max_operator_candidates::Int,
                             include_operator_comparison::Bool=true)
    # Stage A0.4: Base ablation regimes (seed/augment/warmup/episodes)
    regimes = [
        Dict("name" => "seed_only", "enable_augmentation" => false, "enable_warmup" => false, "run_episodes" => false, "operators" => nothing),
        Dict("name" => "seed_plus_augmentation", "enable_augmentation" => true, "enable_warmup" => false, "run_episodes" => false, "operators" => nothing),
        Dict("name" => "seed_plus_warmup", "enable_augmentation" => false, "enable_warmup" => true, "run_episodes" => false, "operators" => nothing),
        Dict("name" => "seed_warmup_episodes", "enable_augmentation" => true, "enable_warmup" => true, "run_episodes" => true, "operators" => nothing),
    ]

    # Stage A0.5: Operator-isolated regimes
    if include_operator_comparison
        append!(regimes, [
            Dict("name" => "mutate_only", "enable_augmentation" => true, "enable_warmup" => true, "run_episodes" => true, "operators" => [:mutate, :terminate]),
            Dict("name" => "crossover_only", "enable_augmentation" => true, "enable_warmup" => true, "run_episodes" => true, "operators" => [:crossover, :terminate]),
            Dict("name" => "mixed_trusted", "enable_augmentation" => true, "enable_warmup" => true, "run_episodes" => true, "operators" => [:mutate, :crossover, :terminate]),
        ])
    end

    all_results = Dict{String,Any}()
    for task in tasks
        println("\n" * "="^80)
        println("Running Stage A0 ablation matrix on $(task) ...")
        println("="^80)
        task_results = Dict{String,Any}()
        for regime in regimes
            name = regime["name"]
            result, extra = run_hierarchical_edit_pmo_task(task;
                budget=budget,
                n_episodes=n_episodes,
                target_seed=target_seed,
                enable_augmentation=regime["enable_augmentation"],
                enable_warmup=regime["enable_warmup"],
                bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                run_episodes=regime["run_episodes"],
                regime_name=name,
                config=HierarchicalEditConfig(
                    max_step_attempts=max_step_attempts,
                    max_operator_candidates=max_operator_candidates,
                    operators=regime["operators"],
                ),
                verbose=true)
            task_results[name] = Dict("hierarchical_edit" => result, "extra" => extra)

            # A0.3: Structured attribution report per regime
            print_attribution_report(task, name, result, extra)
        end

        # A0.4: Cross-regime comparison table
        print_regime_comparison(task, task_results)

        all_results[task] = task_results
    end

    # Decision gate summary
    println("\n" * "="^80)
    println("STAGE A0 DECISION GATE SUMMARY")
    println("="^80)
    for task in tasks
        !haskey(all_results, task) && continue
        task_results = all_results[task]

        # Q1: Graph identity correctness
        ratios = Float64[]
        for (name, data) in task_results
            ratio = get(data["extra"]["frontier_stats"], "string_vs_graph_ratio", 1.0)
            push!(ratios, ratio)
        end
        identity_ok = all(r -> r <= 1.05, ratios)

        # Q2: Edit contribution
        edit_delta = 0.0
        if haskey(task_results, "seed_plus_warmup") && haskey(task_results, "seed_warmup_episodes")
            edit_delta = task_results["seed_warmup_episodes"]["hierarchical_edit"].top10_mean -
                        task_results["seed_plus_warmup"]["hierarchical_edit"].top10_mean
        end

        # Q3: Best operator
        best_op = "none"
        best_op_top10 = 0.0
        for name in ["mutate_only", "crossover_only", "mixed_trusted"]
            haskey(task_results, name) || continue
            t10 = task_results[name]["hierarchical_edit"].top10_mean
            if t10 > best_op_top10
                best_op_top10 = t10
                best_op = name
            end
        end

        println("  $(task):")
        println("    Q1 Graph identity clean: $(identity_ok ? "YES" : "NO") (ratios: $(round.(ratios, digits=3)))")
        println("    Q2 Edit-vs-bootstrap delta: $(round(edit_delta, digits=4)) ($(edit_delta > 0.01 ? "PASS" : edit_delta > 0.0 ? "MARGINAL" : "FAIL"))")
        println("    Q3 Best operator regime: $(best_op) (Top10=$(round(best_op_top10, digits=4)))")
    end
    println()

    return all_results
end

function run_a10_repeat_checks(;
    budget::Int,
    n_episodes::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    repeats::Int=3,
)
    structural_best = Dict(
        "albuterol_similarity" => "mutate_only",
        "celecoxib_rediscovery" => "mixed_trusted",
    )

    function regime_kwargs(name::String)
        if name == "seed_plus_warmup"
            return Dict(
                :enable_augmentation => false,
                :enable_warmup => true,
                :run_episodes => false,
                :operators => nothing,
            )
        elseif name == "mutate_only"
            return Dict(
                :enable_augmentation => true,
                :enable_warmup => true,
                :run_episodes => true,
                :operators => [:mutate, :terminate],
            )
        elseif name == "crossover_only"
            return Dict(
                :enable_augmentation => true,
                :enable_warmup => true,
                :run_episodes => true,
                :operators => [:crossover, :terminate],
            )
        elseif name == "mixed_trusted"
            return Dict(
                :enable_augmentation => true,
                :enable_warmup => true,
                :run_episodes => true,
                :operators => [:mutate, :crossover, :terminate],
            )
        else
            error("Unknown A1.0 regime: $(name)")
        end
    end

    suite = Dict(
        "albuterol_similarity" => ["seed_plus_warmup", structural_best["albuterol_similarity"]],
        "celecoxib_rediscovery" => ["seed_plus_warmup", structural_best["celecoxib_rediscovery"]],
        "drd2" => ["mixed_trusted", "mutate_only"],
    )

    all_results = Dict{String,Any}()
    println("\n" * "="^80)
    println("RUNNING A1.0 ATTRIBUTION + BUDGET STABILITY CHECKS")
    println("="^80)

    for task in ["albuterol_similarity", "celecoxib_rediscovery", "drd2"]
        println("\nTask: $(task)")
        task_results = Dict{String,Any}()
        for regime_name in suite[task]
            runs = Vector{Dict{String,Any}}()
            spec = regime_kwargs(regime_name)
            for repeat_idx in 1:repeats
                result, extra = run_hierarchical_edit_pmo_task(task;
                    budget=budget,
                    n_episodes=n_episodes,
                    target_seed=target_seed,
                    enable_augmentation=spec[:enable_augmentation],
                    enable_warmup=spec[:enable_warmup],
                    bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                    run_episodes=spec[:run_episodes],
                    regime_name="$(regime_name)-r$(repeat_idx)",
                    config=HierarchicalEditConfig(
                        max_step_attempts=max_step_attempts,
                        max_operator_candidates=max_operator_candidates,
                        operators=spec[:operators],
                    ),
                    verbose=true)
                print_attribution_report(task, "$(regime_name)-r$(repeat_idx)", result, extra)
                push!(runs, Dict(
                    "result" => result,
                    "extra" => extra,
                ))
            end
            task_results[regime_name] = runs
        end
        all_results[task] = task_results
    end

    println("\n" * "="^80)
    println("A1.0 SUMMARY")
    println("="^80)
    for (task, task_results) in all_results
        println("\n$(task):")
        for (regime_name, runs) in task_results
            aucs = [r["result"].auc_top10 for r in runs]
            top10s = [r["result"].top10_mean for r in runs]
            edit_fracs = [get(r["extra"]["frontier_source_summary"]["topk"], "edit", 0) / max(r["extra"]["frontier_source_summary"]["topk_size"], 1) for r in runs]
            budget_fracs = [r["extra"]["budget_fraction_used"] for r in runs]
            graph_per_call = [r["extra"]["graph_unique_molecules"] / max(r["result"].n_oracle_calls, 1) for r in runs]
            stagnant_stops = [r["extra"]["run_stats"]["stagnant_stop"] for r in runs]
            top1_sources = [get(r["extra"]["frontier_source_summary"], "top1_source", "none") for r in runs]
            edit_ops = [repr(get(r["extra"]["frontier_source_summary"], "topk_edit_operators", Dict{String,Int}())) for r in runs]

            auc_std = length(aucs) > 1 ? std(aucs) : 0.0
            top10_std = length(top10s) > 1 ? std(top10s) : 0.0
            println("  $(rpad(regime_name, 20)) | AUC=$(round(mean(aucs), digits=4))±$(round(auc_std, digits=4)) | Top10=$(round(mean(top10s), digits=4))±$(round(top10_std, digits=4)) | EditTop10=$(round(100*mean(edit_fracs), digits=1))% | BudgetUsed=$(round(100*mean(budget_fracs), digits=1))% | Graph/Call=$(round(mean(graph_per_call), digits=3)) | StagnantStops=$(sum(stagnant_stops))/$(length(stagnant_stops))")
            println("    Top1 sources: $(join(top1_sources, ", "))")
            println("    Edit operator splits: $(join(edit_ops, " | "))")
        end
    end

    return all_results
end

function _window_subset(logs, fraction::Float64)
    isempty(logs) && return logs[1:0]
    count = clamp(ceil(Int, fraction * length(logs)), 1, length(logs))
    return logs[1:count]
end

function _diagnostic_window_summary(proposal_logs, decision_logs)
    overall = Dict{String,Any}(
        "proposal_count" => length(proposal_logs),
        "decision_count" => length(decision_logs),
        "mean_raw_candidate_count" => isempty(proposal_logs) ? 0.0 : mean(Float64[l.raw_candidate_count for l in proposal_logs]),
        "empty_after_filter_fraction" => isempty(proposal_logs) ? 0.0 : mean(Float64[l.empty_after_filter for l in proposal_logs]),
        "chosen_positive_delta_fraction" => isempty(proposal_logs) ? 0.0 : mean(Float64[l.chosen_reward_delta > 0 for l in proposal_logs if !isempty(l.chosen_child_smiles)]),
        "topk_entry_fraction" => isempty(decision_logs) ? 0.0 : mean(Float64[l.enters_topk for l in decision_logs]),
        "mean_frontier_utility_delta" => isempty(decision_logs) ? 0.0 : mean(Float64[l.frontier_utility_delta for l in decision_logs]),
    )
    if isnan(overall["chosen_positive_delta_fraction"])
        overall["chosen_positive_delta_fraction"] = 0.0
    end

    by_operator = Dict{String,Any}()
    all_ops = unique(vcat(Symbol[l.operator for l in proposal_logs], Symbol[l.operator for l in decision_logs]))
    for op in [:mutate, :crossover, :terminate]
        op in all_ops || continue
        op_props = [l for l in proposal_logs if l.operator == op]
        op_decisions = [l for l in decision_logs if l.operator == op]
        chosen_props = [l for l in op_props if !isempty(l.chosen_child_smiles)]
        stats = Dict{String,Any}(
            "proposal_count" => length(op_props),
            "decision_count" => length(op_decisions),
            "mean_raw_candidate_count" => isempty(op_props) ? 0.0 : mean(Float64[l.raw_candidate_count for l in op_props]),
            "empty_after_filter_fraction" => isempty(op_props) ? 0.0 : mean(Float64[l.empty_after_filter for l in op_props]),
            "chosen_positive_delta_fraction" => isempty(chosen_props) ? 0.0 : mean(Float64[l.chosen_reward_delta > 0 for l in chosen_props]),
            "topk_entry_fraction" => isempty(op_decisions) ? 0.0 : mean(Float64[l.enters_topk for l in op_decisions]),
            "mean_frontier_utility_delta" => isempty(op_decisions) ? 0.0 : mean(Float64[l.frontier_utility_delta for l in op_decisions]),
        )
        by_operator[String(op)] = stats
    end

    return Dict("overall" => overall, "by_operator" => by_operator)
end

function compute_bridge_diagnostics(extra::Dict{String,Any})
    proposal_logs = get(extra, "proposal_logs_raw", Vector{HierarchicalEditProposalLog}())
    decision_logs = get(extra, "decision_logs_raw", Vector{HierarchicalEditDecisionLog}())
    return Dict{String,Any}(
        "early" => _diagnostic_window_summary(_window_subset(proposal_logs, 0.25), _window_subset(decision_logs, 0.25)),
        "mid" => _diagnostic_window_summary(_window_subset(proposal_logs, 0.50), _window_subset(decision_logs, 0.50)),
        "full" => _diagnostic_window_summary(proposal_logs, decision_logs),
    )
end

function _mean_run_metric(runs, accessor)
    values = Float64[]
    for run in runs
        push!(values, Float64(accessor(run)))
    end
    return isempty(values) ? 0.0 : mean(values)
end

function _std_run_metric(runs, accessor)
    values = Float64[]
    for run in runs
        push!(values, Float64(accessor(run)))
    end
    return length(values) > 1 ? std(values) : 0.0
end

function _operator_window_parts(runs, window_name::String)
    parts = String[]
    for op in ["mutate", "crossover", "terminate"]
        has_any = any(haskey(get(run["diagnostics"][window_name]["by_operator"], op, Dict{String,Any}()), "proposal_count") for run in runs)
        has_any || continue
        raw = _mean_run_metric(runs, run -> get(get(run["diagnostics"][window_name]["by_operator"], op, Dict{String,Any}()), "mean_raw_candidate_count", 0.0))
        empty_frac = _mean_run_metric(runs, run -> get(get(run["diagnostics"][window_name]["by_operator"], op, Dict{String,Any}()), "empty_after_filter_fraction", 0.0))
        positive_frac = _mean_run_metric(runs, run -> get(get(run["diagnostics"][window_name]["by_operator"], op, Dict{String,Any}()), "chosen_positive_delta_fraction", 0.0))
        topk_frac = _mean_run_metric(runs, run -> get(get(run["diagnostics"][window_name]["by_operator"], op, Dict{String,Any}()), "topk_entry_fraction", 0.0))
        push!(parts, "$(op){raw=$(round(raw, digits=2)), empty=$(round(empty_frac, digits=3)), +Δ=$(round(positive_frac, digits=3)), topk=$(round(topk_frac, digits=3))}")
    end
    return isempty(parts) ? "none" : join(parts, " | ")
end

function bridge_regime_kwargs(name::String)
    if name == "mutate_only"
        return Dict(
            :enable_augmentation => true,
            :enable_warmup => true,
            :run_episodes => true,
            :operators => [:mutate, :terminate],
            :use_operator_adaptation => false,
            :operator_sampling_weights => nothing,
        )
    elseif name == "crossover_only"
        return Dict(
            :enable_augmentation => true,
            :enable_warmup => true,
            :run_episodes => true,
            :operators => [:crossover, :terminate],
            :use_operator_adaptation => false,
            :operator_sampling_weights => nothing,
        )
    elseif name == "mixed_trusted"
        return Dict(
            :enable_augmentation => true,
            :enable_warmup => true,
            :run_episodes => true,
            :operators => [:mutate, :crossover, :terminate],
            :use_operator_adaptation => false,
            :operator_sampling_weights => nothing,
        )
    elseif name == "mutate_biased_mixed"
        return Dict(
            :enable_augmentation => true,
            :enable_warmup => true,
            :run_episodes => true,
            :operators => [:mutate, :crossover, :terminate],
            :use_operator_adaptation => false,
            :operator_sampling_weights => Dict(:mutate => 0.75, :crossover => 0.25),
        )
    elseif name == "mutate_dominant_mixed"
        return Dict(
            :enable_augmentation => true,
            :enable_warmup => true,
            :run_episodes => true,
            :operators => [:mutate, :crossover, :terminate],
            :use_operator_adaptation => false,
            :operator_sampling_weights => Dict(:mutate => 0.85, :crossover => 0.15),
        )
    else
        error("Unknown bridge regime: $(name)")
    end
end

function run_a11_bridge_checks(tasks::Vector{String};
    budget::Int,
    n_episodes::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    repeats::Int=3,
)
    diagnostic_suite = Dict(
        "albuterol_similarity" => ["mutate_only", "crossover_only", "mixed_trusted"],
        "celecoxib_rediscovery" => ["mutate_only", "crossover_only", "mixed_trusted"],
        "drd2" => ["mutate_only", "crossover_only", "mixed_trusted"],
    )
    probe_suite = Dict(
        "celecoxib_rediscovery" => ["mutate_biased_mixed"],
        "drd2" => ["mutate_dominant_mixed"],
    )

    all_results = Dict{String,Any}()
    println("\n" * "="^80)
    println("RUNNING A1.1 / A1.2 BRIDGE CHECKS")
    println("="^80)

    for task in tasks
        println("\nTask: $(task)")
        task_results = Dict{String,Any}()
        combined_regimes = vcat(get(diagnostic_suite, task, String[]), get(probe_suite, task, String[]))
        for regime_name in combined_regimes
            runs = Vector{Dict{String,Any}}()
            spec = bridge_regime_kwargs(regime_name)
            for repeat_idx in 1:repeats
                result, extra = run_hierarchical_edit_pmo_task(task;
                    budget=budget,
                    n_episodes=n_episodes,
                    target_seed=target_seed,
                    enable_augmentation=spec[:enable_augmentation],
                    enable_warmup=spec[:enable_warmup],
                    bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                    run_episodes=spec[:run_episodes],
                    regime_name="$(regime_name)-r$(repeat_idx)",
                    config=HierarchicalEditConfig(
                        max_step_attempts=max_step_attempts,
                        max_operator_candidates=max_operator_candidates,
                        operators=spec[:operators],
                        use_operator_adaptation=spec[:use_operator_adaptation],
                        operator_sampling_weights=spec[:operator_sampling_weights],
                    ),
                    verbose=true)
                diagnostics = compute_bridge_diagnostics(extra)
                print_attribution_report(task, "$(regime_name)-r$(repeat_idx)", result, extra)
                println("  Bridge diagnostics ($(regime_name)-r$(repeat_idx)):")
                for window_name in ["early", "mid", "full"]
                    window = diagnostics[window_name]
                    overall = window["overall"]
                    single_run_parts = _operator_window_parts([Dict("diagnostics" => diagnostics)], window_name)
                    mean_raw = overall["mean_raw_candidate_count"]
                    empty_frac = overall["empty_after_filter_fraction"]
                    positive_frac = overall["chosen_positive_delta_fraction"]
                    topk_frac = overall["topk_entry_fraction"]
                    mean_util = overall["mean_frontier_utility_delta"]
                    println("    $(window_name): raw=$(round(mean_raw, digits=2)) | empty=$(round(empty_frac, digits=3)) | +Δ=$(round(positive_frac, digits=3)) | topk=$(round(topk_frac, digits=3)) | util=$(round(mean_util, digits=3))")
                    println("      ops: $(single_run_parts)")
                end
                println()
                push!(runs, Dict(
                    "result" => result,
                    "extra" => extra,
                    "diagnostics" => diagnostics,
                ))
            end
            task_results[regime_name] = runs
        end
        all_results[task] = task_results
    end

    println("\n" * "="^80)
    println("A1.1 / A1.2 BRIDGE SUMMARY")
    println("="^80)
    for task in tasks
        haskey(all_results, task) || continue
        task_results = all_results[task]
        println("\n$(task):")
        println("  Part A — diagnostic matrix")
        for regime_name in get(diagnostic_suite, task, String[])
            haskey(task_results, regime_name) || continue
            runs = task_results[regime_name]
            auc_mean = _mean_run_metric(runs, run -> run["result"].auc_top10)
            auc_std = _std_run_metric(runs, run -> run["result"].auc_top10)
            top10_mean = _mean_run_metric(runs, run -> run["result"].top10_mean)
            top10_std = _std_run_metric(runs, run -> run["result"].top10_mean)
            edit_top10 = 100 * _mean_run_metric(runs, run -> get(run["extra"]["frontier_source_summary"]["topk"], "edit", 0) / max(run["extra"]["frontier_source_summary"]["topk_size"], 1))
            budget_used = 100 * _mean_run_metric(runs, run -> run["extra"]["budget_fraction_used"])
            graph_per_call = _mean_run_metric(runs, run -> run["extra"]["graph_unique_molecules"] / max(run["result"].n_oracle_calls, 1))
            early_parts = _operator_window_parts(runs, "early")
            mid_parts = _operator_window_parts(runs, "mid")
            full_parts = _operator_window_parts(runs, "full")
            println("    $(rpad(regime_name, 22)) | AUC=$(round(auc_mean, digits=4))±$(round(auc_std, digits=4)) | Top10=$(round(top10_mean, digits=4))±$(round(top10_std, digits=4)) | EditTop10=$(round(edit_top10, digits=1))% | BudgetUsed=$(round(budget_used, digits=1))% | Graph/Call=$(round(graph_per_call, digits=3))")
            println("      early ops: $(early_parts)")
            println("      mid ops:   $(mid_parts)")
            println("      full ops:  $(full_parts)")
        end

        probe_regimes = get(probe_suite, task, String[])
        if !isempty(probe_regimes)
            println("  Part B — bounded probes")
            baseline_runs = get(task_results, "mixed_trusted", Vector{Dict{String,Any}}())
            baseline_top10 = isempty(baseline_runs) ? 0.0 : _mean_run_metric(baseline_runs, run -> run["result"].top10_mean)
            baseline_auc = isempty(baseline_runs) ? 0.0 : _mean_run_metric(baseline_runs, run -> run["result"].auc_top10)
            baseline_edit = isempty(baseline_runs) ? 0.0 : 100 * _mean_run_metric(baseline_runs, run -> get(run["extra"]["frontier_source_summary"]["topk"], "edit", 0) / max(run["extra"]["frontier_source_summary"]["topk_size"], 1))
            baseline_budget = isempty(baseline_runs) ? 0.0 : 100 * _mean_run_metric(baseline_runs, run -> run["extra"]["budget_fraction_used"])
            for regime_name in probe_regimes
                haskey(task_results, regime_name) || continue
                runs = task_results[regime_name]
                probe_top10 = _mean_run_metric(runs, run -> run["result"].top10_mean)
                probe_auc = _mean_run_metric(runs, run -> run["result"].auc_top10)
                probe_edit = 100 * _mean_run_metric(runs, run -> get(run["extra"]["frontier_source_summary"]["topk"], "edit", 0) / max(run["extra"]["frontier_source_summary"]["topk_size"], 1))
                probe_budget = 100 * _mean_run_metric(runs, run -> run["extra"]["budget_fraction_used"])
                early_parts = _operator_window_parts(runs, "early")
                full_parts = _operator_window_parts(runs, "full")
                println("    $(rpad(regime_name, 22)) | AUC=$(round(probe_auc, digits=4)) | Top10=$(round(probe_top10, digits=4)) | EditTop10=$(round(probe_edit, digits=1))% | BudgetUsed=$(round(probe_budget, digits=1))%")
                println("      vs mixed_trusted: ΔAUC=$(round(probe_auc - baseline_auc, digits=4)) | ΔTop10=$(round(probe_top10 - baseline_top10, digits=4)) | ΔEditTop10=$(round(probe_edit - baseline_edit, digits=1))% | ΔBudget=$(round(probe_budget - baseline_budget, digits=1))%")
                println("      early ops: $(early_parts)")
                println("      full ops:  $(full_parts)")
            end
        end
    end

    return Dict(
        "diagnostic_suite" => diagnostic_suite,
        "probe_suite" => probe_suite,
        "results_by_task" => all_results,
    )
end

function run_a12_confirm_checks(tasks::Vector{String};
    budget::Int,
    n_episodes::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    celecoxib_repeats::Int=5,
    control_repeats::Int=3,
    sanity_repeats::Int=2,
    include_control_anchor::Bool=true,
)
    default_suite = Dict(
        "celecoxib_rediscovery" => [("mixed_trusted", celecoxib_repeats), ("mutate_biased_mixed", celecoxib_repeats), ("mutate_only", celecoxib_repeats)],
        "drd2" => include_control_anchor ? [("mutate_only", control_repeats), ("mutate_biased_mixed", control_repeats), ("mixed_trusted", control_repeats)] : [("mutate_only", control_repeats), ("mutate_biased_mixed", control_repeats)],
        "albuterol_similarity" => [("mutate_only", sanity_repeats), ("mutate_biased_mixed", sanity_repeats)],
    )
    task_order = [task for task in ["celecoxib_rediscovery", "drd2", "albuterol_similarity"] if task in tasks]

    all_results = Dict{String,Any}()
    println("\n" * "="^80)
    println("RUNNING A1.2 STATIC ALLOCATION CONFIRMATION")
    println("="^80)

    for task in task_order
        println("\nTask: $(task)")
        task_results = Dict{String,Any}()
        for (regime_name, repeats) in default_suite[task]
            runs = Vector{Dict{String,Any}}()
            spec = bridge_regime_kwargs(regime_name)
            for repeat_idx in 1:repeats
                result, extra = run_hierarchical_edit_pmo_task(task;
                    budget=budget,
                    n_episodes=n_episodes,
                    target_seed=target_seed,
                    enable_augmentation=spec[:enable_augmentation],
                    enable_warmup=spec[:enable_warmup],
                    bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                    run_episodes=spec[:run_episodes],
                    regime_name="$(regime_name)-r$(repeat_idx)",
                    config=HierarchicalEditConfig(
                        max_step_attempts=max_step_attempts,
                        max_operator_candidates=max_operator_candidates,
                        operators=spec[:operators],
                        use_operator_adaptation=spec[:use_operator_adaptation],
                        operator_sampling_weights=spec[:operator_sampling_weights],
                    ),
                    verbose=true)
                diagnostics = compute_bridge_diagnostics(extra)
                print_attribution_report(task, "$(regime_name)-r$(repeat_idx)", result, extra)
                push!(runs, Dict(
                    "result" => result,
                    "extra" => extra,
                    "diagnostics" => diagnostics,
                ))
            end
            task_results[regime_name] = runs
        end
        all_results[task] = task_results
    end

    println("\n" * "="^80)
    println("A1.2 CONFIRMATION SUMMARY")
    println("="^80)
    for task in task_order
        task_results = all_results[task]
        println("\n$(task):")
        for regime_name in keys(task_results)
            runs = task_results[regime_name]
            auc_mean = _mean_run_metric(runs, run -> run["result"].auc_top10)
            auc_std = _std_run_metric(runs, run -> run["result"].auc_top10)
            top10_mean = _mean_run_metric(runs, run -> run["result"].top10_mean)
            top10_std = _std_run_metric(runs, run -> run["result"].top10_mean)
            budget_used = 100 * _mean_run_metric(runs, run -> run["extra"]["budget_fraction_used"])
            graph_per_call = _mean_run_metric(runs, run -> run["extra"]["graph_unique_molecules"] / max(run["result"].n_oracle_calls, 1))
            edit_top10 = 100 * _mean_run_metric(runs, run -> get(run["extra"]["frontier_source_summary"]["topk"], "edit", 0) / max(run["extra"]["frontier_source_summary"]["topk_size"], 1))
            full_parts = _operator_window_parts(runs, "full")
            println("  $(rpad(regime_name, 22)) | AUC=$(round(auc_mean, digits=4))±$(round(auc_std, digits=4)) | Top10=$(round(top10_mean, digits=4))±$(round(top10_std, digits=4)) | EditTop10=$(round(edit_top10, digits=1))% | BudgetUsed=$(round(budget_used, digits=1))% | Graph/Call=$(round(graph_per_call, digits=3))")
            println("    full ops: $(full_parts)")
        end
    end

    decisions = Dict{String,Any}()
    if haskey(all_results, "celecoxib_rediscovery")
        task_results = all_results["celecoxib_rediscovery"]
        mixed_runs = task_results["mixed_trusted"]
        biased_runs = task_results["mutate_biased_mixed"]
        mixed_top10 = _mean_run_metric(mixed_runs, run -> run["result"].top10_mean)
        mixed_auc = _mean_run_metric(mixed_runs, run -> run["result"].auc_top10)
        mixed_budget = 100 * _mean_run_metric(mixed_runs, run -> run["extra"]["budget_fraction_used"])
        mixed_graph = _mean_run_metric(mixed_runs, run -> run["extra"]["graph_unique_molecules"] / max(run["result"].n_oracle_calls, 1))
        biased_top10 = _mean_run_metric(biased_runs, run -> run["result"].top10_mean)
        biased_auc = _mean_run_metric(biased_runs, run -> run["result"].auc_top10)
        biased_budget = 100 * _mean_run_metric(biased_runs, run -> run["extra"]["budget_fraction_used"])
        biased_graph = _mean_run_metric(biased_runs, run -> run["extra"]["graph_unique_molecules"] / max(run["result"].n_oracle_calls, 1))
        delta_top10 = biased_top10 - mixed_top10
        delta_auc = biased_auc - mixed_auc
        delta_budget = biased_budget - mixed_budget
        delta_graph = biased_graph - mixed_graph
        promote = (delta_top10 >= 0.01) && (delta_auc >= -1e-6) && (delta_budget >= -2.0) && (delta_graph >= -0.01)
        decisions["celecoxib_promote_mutate_biased_mixed"] = promote
        decisions["celecoxib_delta_top10"] = delta_top10
        decisions["celecoxib_delta_auc"] = delta_auc
        decisions["celecoxib_delta_budget"] = delta_budget
        decisions["celecoxib_delta_graph"] = delta_graph
    end

    if haskey(all_results, "drd2")
        task_results = all_results["drd2"]
        mutate_runs = task_results["mutate_only"]
        biased_runs = task_results["mutate_biased_mixed"]
        mutate_top10 = _mean_run_metric(mutate_runs, run -> run["result"].top10_mean)
        mutate_auc = _mean_run_metric(mutate_runs, run -> run["result"].auc_top10)
        mutate_budget = 100 * _mean_run_metric(mutate_runs, run -> run["extra"]["budget_fraction_used"])
        biased_top10 = _mean_run_metric(biased_runs, run -> run["result"].top10_mean)
        biased_auc = _mean_run_metric(biased_runs, run -> run["result"].auc_top10)
        biased_budget = 100 * _mean_run_metric(biased_runs, run -> run["extra"]["budget_fraction_used"])
        not_clearly_worse = ((biased_top10 - mutate_top10) >= -0.001) && ((biased_auc - mutate_auc) >= -0.001) && ((biased_budget - mutate_budget) >= -2.0)
        decisions["drd2_mutate_biased_not_clearly_worse"] = not_clearly_worse
        decisions["drd2_delta_top10_vs_mutate_only"] = biased_top10 - mutate_top10
        decisions["drd2_delta_auc_vs_mutate_only"] = biased_auc - mutate_auc
    end

    println("\n" * "="^80)
    println("A1.2 DECISION SUMMARY")
    println("="^80)
    if haskey(decisions, "celecoxib_promote_mutate_biased_mixed")
        println("  celecoxib promote mutate_biased_mixed: $(decisions["celecoxib_promote_mutate_biased_mixed"] ? "YES" : "NO") | ΔTop10=$(round(decisions["celecoxib_delta_top10"], digits=4)) | ΔAUC=$(round(decisions["celecoxib_delta_auc"], digits=4)) | ΔBudget=$(round(decisions["celecoxib_delta_budget"], digits=2)) | ΔGraph=$(round(decisions["celecoxib_delta_graph"], digits=4))")
    end
    if haskey(decisions, "drd2_mutate_biased_not_clearly_worse")
        println("  drd2 mutate_biased_mixed not clearly worse than mutate_only: $(decisions["drd2_mutate_biased_not_clearly_worse"] ? "YES" : "NO") | ΔTop10=$(round(decisions["drd2_delta_top10_vs_mutate_only"], digits=4)) | ΔAUC=$(round(decisions["drd2_delta_auc_vs_mutate_only"], digits=4))")
    end
    if get(decisions, "celecoxib_promote_mutate_biased_mixed", false) && get(decisions, "drd2_mutate_biased_not_clearly_worse", false)
        println("  Recommendation: single mutate_biased_mixed default is tentatively acceptable.")
    elseif get(decisions, "celecoxib_promote_mutate_biased_mixed", false)
        println("  Recommendation: use task-aware presets (mutate_biased_mixed for celecoxib-like tasks, mutate_only for drd2-like tasks).")
    else
        println("  Recommendation: do not promote mutate_biased_mixed yet; keep simpler task baselines.")
    end

    return Dict(
        "suite" => default_suite,
        "results_by_task" => all_results,
        "decisions" => decisions,
    )
end

function run_c2_basin_controller_checks(tasks::Vector{String};
    budget::Int,
    n_episodes::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=3,
    eval_repeats::Int=2,
    training_epochs::Int=20,
    max_promoted::Int=2,
    celecoxib_repeats::Int=5,
    control_repeats::Int=3,
    sanity_repeats::Int=3,
)
    println("\n" * "="^80)
    println("RUNNING DIRECTION C V2 BATCH 1A.2 — TASK-AWARE BASIN REFINEMENT")
    println("="^80)

    function class_counts_repr(counts)
        isempty(counts) && return "none"
        parts = String[]
        for key in sort(collect(keys(counts)))
            push!(parts, "$(key)=$(counts[key])")
        end
        return join(parts, ", ")
    end

    mode_repr(mode::Symbol) = String(mode)

    function online_summary(runs)
        isempty(runs) && return Dict("auc" => NaN, "top10" => NaN, "budget" => NaN, "graph_per_call" => NaN)
        return Dict(
            "auc" => _mean_run_metric(runs, run -> run["result"].auc_top10),
            "top10" => _mean_run_metric(runs, run -> run["result"].top10_mean),
            "budget" => _mean_run_metric(runs, run -> run["extra"]["budget_fraction_used"]),
            "graph_per_call" => _mean_run_metric(runs, run -> run["extra"]["graph_unique_molecules"] / max(run["result"].n_oracle_calls, 1)),
        )
    end

    function task_eval_repeats(task::String)
        if task == "celecoxib_rediscovery"
            return celecoxib_repeats
        elseif task == "drd2"
            return control_repeats
        elseif task == "albuterol_similarity"
            return sanity_repeats
        else
            return eval_repeats
        end
    end

    function offline_recipe_score(task::String, val_eval::Dict{String,Any})
        margin_score = tanh(get(val_eval, "productive_degenerate_margin", 0.0))
        score = 1.0 * get(val_eval, "score_target_correlation", 0.0)
        score += 0.35 * get(val_eval, "frontier_utility_correlation", 0.0)
        score += 0.25 * get(val_eval, "enters_topk_correlation", 0.0)
        score += 0.20 * margin_score
        score += 0.10 * (get(val_eval, "score_target_correlation", 0.0) - get(val_eval, "heuristic_target_correlation", 0.0))
        score += 0.10 * (get(val_eval, "frontier_utility_correlation", 0.0) - get(val_eval, "heuristic_frontier_utility_correlation", 0.0))
        score -= 0.10 * get(val_eval, "rmse", 0.0)
        if task == "drd2"
            score += 0.25 * (get(val_eval, "degenerate_balanced_accuracy", 0.5) - 0.5)
        end
        return score
    end

    function recipe_name(feature_mode::Symbol, target_mode::Symbol, model_kind::Symbol)
        return "$(mode_repr(feature_mode))__$(mode_repr(target_mode))__$(mode_repr(model_kind))"
    end

    feature_modes = [:basic, :augmented]
    target_modes = [:blended, :ordinal_productivity, :risk_adjusted_utility]
    model_kinds = [:linear, :mlp]

    all_results = Dict{String,Any}()
    decisions = Dict{String,Any}()

    for task in tasks
        println("\nTask: $(task)")
        collection_runs = Vector{Dict{String,Any}}()
        basin_logs = BasinDecisionLog[]
        proposal_logs = HierarchicalEditProposalLog[]
        decision_logs = HierarchicalEditDecisionLog[]

        for repeat_idx in 1:data_repeats
            result, extra = run_hierarchical_edit_pmo_task(task;
                budget=budget,
                n_episodes=n_episodes,
                target_seed=target_seed,
                enable_augmentation=true,
                enable_warmup=true,
                bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                run_episodes=true,
                regime_name="basin-data-r$(repeat_idx)",
                config=HierarchicalEditConfig(
                    max_step_attempts=max_step_attempts,
                    max_operator_candidates=max_operator_candidates,
                    basin_candidate_limit=8,
                    use_learned_basin=false,
                ),
                verbose=true)
            push!(collection_runs, Dict("result" => result, "extra" => extra))
            append!(basin_logs, extra["basin_logs_raw"])
            append!(proposal_logs, extra["proposal_logs_raw"])
            append!(decision_logs, extra["decision_logs_raw"])
        end

        audit = audit_basin_dataset_coverage(basin_logs, proposal_logs, decision_logs)
        audit_basin_logs = audit["basin_logs"]
        audit_matched = audit["matched_attempt_outcomes"]
        audit_prop_cov = round(100 * audit["proposal_coverage_fraction"], digits=1)
        audit_dec_cov = round(100 * audit["decision_coverage_fraction"], digits=1)
        audit_empty = round(100 * audit["empty_after_filter_fraction"], digits=1)
        audit_classes = class_counts_repr(audit["class_counts"])
        println("  Audit: basin_logs=$(audit_basin_logs) | matched=$(audit_matched) | proposal_cov=$(audit_prop_cov)% | decision_cov=$(audit_dec_cov)% | empty=$(audit_empty)%")
        println("  Audit classes: $(audit_classes)")

        dataset_summaries = Dict{String,Any}()
        recipe_results = Vector{Dict{String,Any}}()
        promoted_runtime = Vector{Dict{String,Any}}()

        for feature_mode in feature_modes
            for target_mode in target_modes
                dataset = extract_basin_controller_dataset(basin_logs, proposal_logs, decision_logs;
                    target_mode=target_mode,
                    feature_mode=feature_mode)
                stats = basin_controller_dataset_stats(dataset)
                dataset_key = "$(mode_repr(feature_mode))__$(mode_repr(target_mode))"
                dataset_summaries[dataset_key] = Dict(
                    "feature_mode" => mode_repr(feature_mode),
                    "target_mode" => mode_repr(target_mode),
                    "dataset_stats" => stats,
                )
                stats_size = stats["size"]
                stats_positive = round(100 * stats["positive_fraction"], digits=1)
                stats_feature_dim = stats["feature_dim"]
                stats_classes = class_counts_repr(stats["class_counts"])
                println("  Dataset $(dataset_key): size=$(stats_size) | positive=$(stats_positive)% | feat_dim=$(stats_feature_dim) | classes=$(stats_classes)")

                for model_kind in model_kinds
                    config = BasinControllerTrainingConfig(
                        n_epochs=training_epochs,
                        model_kind=model_kind,
                        hidden_dim=32,
                        learning_rate=model_kind == :linear ? 1e-3 : 5e-3,
                        feature_mode=feature_mode,
                    )
                    recipe_name_value = recipe_name(feature_mode, target_mode, model_kind)
                    recipe = Dict{String,Any}(
                        "task" => task,
                        "feature_mode" => mode_repr(feature_mode),
                        "target_mode" => mode_repr(target_mode),
                        "model_kind" => mode_repr(model_kind),
                        "recipe_name" => recipe_name_value,
                        "dataset_stats" => stats,
                        "error" => nothing,
                        "summary" => Dict{String,Any}(),
                        "offline_score" => -Inf,
                        "controller_path" => nothing,
                    )
                    try
                        controller, summary = train_basin_controller(dataset;
                            config=config,
                            rng=Random.MersenneTwister(hash((task, feature_mode, target_mode, model_kind))))
                        val_eval = summary["val_eval"]
                        score = offline_recipe_score(task, val_eval)
                        controller_path = joinpath(OUTDIR, "learned_basin_$(task)_$(recipe_name_value).jls")
                        save_learned_basin_controller(controller_path, controller)
                        recipe["summary"] = summary
                        recipe["offline_score"] = score
                        recipe["controller_path"] = controller_path
                        push!(promoted_runtime, Dict(
                            "recipe_name" => recipe_name_value,
                            "controller" => controller,
                            "controller_path" => controller_path,
                            "feature_mode" => recipe["feature_mode"],
                            "target_mode" => recipe["target_mode"],
                            "model_kind" => recipe["model_kind"],
                            "summary" => summary,
                            "offline_score" => score,
                        ))
                        recipe_corr = round(val_eval["score_target_correlation"], digits=4)
                        recipe_fu_corr = round(val_eval["frontier_utility_correlation"], digits=4)
                        recipe_topk_corr = round(val_eval["enters_topk_correlation"], digits=4)
                        recipe_margin = round(val_eval["productive_degenerate_margin"], digits=4)
                        recipe_deg_bal = round(val_eval["degenerate_balanced_accuracy"], digits=4)
                        println("    Recipe $(rpad(recipe_name_value, 42)) | score=$(round(score, digits=4)) | corr=$(recipe_corr) | fu_corr=$(recipe_fu_corr) | topk_corr=$(recipe_topk_corr) | margin=$(recipe_margin) | deg_bal=$(recipe_deg_bal)")
                    catch err
                        recipe["error"] = sprint(showerror, err)
                        recipe_error = recipe["error"]
                        println("    Recipe $(rpad(recipe_name_value, 42)) | failed: $(recipe_error)")
                    end
                    push!(recipe_results, recipe)
                end
            end
        end

        sort!(promoted_runtime, by=r -> (-r["offline_score"], r["recipe_name"]))
        promoted_runtime = promoted_runtime[1:min(max_promoted, length(promoted_runtime))]
        promoted_recipe_names = [item["recipe_name"] for item in promoted_runtime]
        promoted_recipe_str = isempty(promoted_recipe_names) ? "none" : join(promoted_recipe_names, ", ")
        println("  Promoted offline recipes: $(promoted_recipe_str)")

        heuristic_runs = Vector{Dict{String,Any}}()
        promoted_online = Dict{String,Any}()
        for repeat_idx in 1:task_eval_repeats(task)
            heuristic_result, heuristic_extra = run_hierarchical_edit_pmo_task(task;
                budget=budget,
                n_episodes=n_episodes,
                target_seed=target_seed,
                enable_augmentation=true,
                enable_warmup=true,
                bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                run_episodes=true,
                regime_name="heuristic-basin-r$(repeat_idx)",
                config=HierarchicalEditConfig(
                    max_step_attempts=max_step_attempts,
                    max_operator_candidates=max_operator_candidates,
                    basin_candidate_limit=8,
                    use_learned_basin=false,
                ),
                verbose=true)
            push!(heuristic_runs, Dict("result" => heuristic_result, "extra" => heuristic_extra))

            for promoted in promoted_runtime
                name = promoted["recipe_name"]
                runs = get!(promoted_online, name, Vector{Dict{String,Any}}())
                learned_result, learned_extra = run_hierarchical_edit_pmo_task(task;
                    budget=budget,
                    n_episodes=n_episodes,
                    target_seed=target_seed,
                    enable_augmentation=true,
                    enable_warmup=true,
                    bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                    run_episodes=true,
                    regime_name="$(name)-r$(repeat_idx)",
                    config=HierarchicalEditConfig(
                        max_step_attempts=max_step_attempts,
                        max_operator_candidates=max_operator_candidates,
                        basin_candidate_limit=8,
                        use_learned_basin=true,
                        learned_basin_controller=promoted["controller"],
                    ),
                    verbose=true)
                push!(runs, Dict("result" => learned_result, "extra" => learned_extra))
            end
        end

        heuristic_summary = online_summary(heuristic_runs)
        heuristic_auc = round(heuristic_summary["auc"], digits=4)
        heuristic_top10 = round(heuristic_summary["top10"], digits=4)
        heuristic_budget = round(100 * heuristic_summary["budget"], digits=1)
        heuristic_graph = round(heuristic_summary["graph_per_call"], digits=4)
        println("  Online heuristic: AUC=$(heuristic_auc) | Top10=$(heuristic_top10) | Budget=$(heuristic_budget)% | Graph/Call=$(heuristic_graph)")
        online_recipe_summaries = Dict{String,Any}()
        best_recipe_name = nothing
        best_delta = -Inf
        best_offline_rank = 0
        for (rank_idx, promoted) in enumerate(promoted_runtime)
            name = promoted["recipe_name"]
            summary = online_summary(get(promoted_online, name, Vector{Dict{String,Any}}()))
            delta_top10 = summary["top10"] - heuristic_summary["top10"]
            delta_auc = summary["auc"] - heuristic_summary["auc"]
            delta_graph = summary["graph_per_call"] - heuristic_summary["graph_per_call"]
            online_recipe_summaries[name] = Dict(
                "summary" => summary,
                "delta_top10" => delta_top10,
                "delta_auc" => delta_auc,
                "delta_graph_per_call" => delta_graph,
                "offline_score" => promoted["offline_score"],
                "offline_rank" => rank_idx,
                "feature_mode" => promoted["feature_mode"],
                "target_mode" => promoted["target_mode"],
                "model_kind" => promoted["model_kind"],
                "controller_path" => promoted["controller_path"],
            )
            online_auc = round(summary["auc"], digits=4)
            online_top10 = round(summary["top10"], digits=4)
            online_delta_top10 = round(delta_top10, digits=4)
            online_delta_auc = round(delta_auc, digits=4)
            online_delta_graph = round(delta_graph, digits=4)
            println("  Online $(rpad(name, 42)) | AUC=$(online_auc) | Top10=$(online_top10) | ΔTop10=$(online_delta_top10) | ΔAUC=$(online_delta_auc) | ΔGraph/Call=$(online_delta_graph)")
            if delta_top10 > best_delta
                best_delta = delta_top10
                best_recipe_name = name
                best_offline_rank = rank_idx
            end
        end

        all_results[task] = Dict(
            "collection_runs" => collection_runs,
            "audit" => audit,
            "dataset_summaries" => dataset_summaries,
            "recipe_results" => recipe_results,
            "promoted_recipes" => promoted_recipe_names,
            "heuristic_runs" => heuristic_runs,
            "promoted_online_runs" => promoted_online,
            "online_recipe_summaries" => online_recipe_summaries,
        )

        decisions[task] = Dict(
            "best_recipe_name" => best_recipe_name,
            "best_delta_top10" => best_delta,
            "offline_online_alignment" => !isnothing(best_recipe_name) && best_offline_rank == 1,
        )
    end

    println("\n" * "="^80)
    println("BATCH 1A.2 SUMMARY")
    println("="^80)
    for task in tasks
        haskey(all_results, task) || continue
        task_result = all_results[task]
        decision = decisions[task]
        promoted_str = isempty(task_result["promoted_recipes"]) ? "none" : join(task_result["promoted_recipes"], ", ")
        best_name = something(decision["best_recipe_name"], "none")
        best_delta_repr = round(decision["best_delta_top10"], digits=4)
        aligned_repr = decision["offline_online_alignment"] ? "YES" : "NO"
        println("\n$(task):")
        println("  promoted recipes: $(promoted_str)")
        println("  best online recipe: $(best_name) | ΔTop10=$(best_delta_repr) | offline-online aligned=$(aligned_repr)")
    end

    expected_tasks = [task for task in ["celecoxib_rediscovery", "drd2", "albuterol_similarity"] if task in tasks]
    if all(task -> haskey(decisions, task), expected_tasks)
        celecoxib_delta = get(get(decisions, "celecoxib_rediscovery", Dict{String,Any}()), "best_delta_top10", -Inf)
        drd2_delta = get(get(decisions, "drd2", Dict{String,Any}()), "best_delta_top10", -Inf)
        albuterol_delta = get(get(decisions, "albuterol_similarity", Dict{String,Any}()), "best_delta_top10", -Inf)
        celecoxib_align = get(get(decisions, "celecoxib_rediscovery", Dict{String,Any}()), "offline_online_alignment", false)
        promote = celecoxib_delta >= 0.015 && drd2_delta >= -0.001 && albuterol_delta >= -0.01 && celecoxib_align
        decisions["global_recommendation"] = promote ? "PROMOTE_BASIN_ONLY" : "HOLD_AT_BASIN_ONLY"
        global_rec = decisions["global_recommendation"]
        celecoxib_align_repr = celecoxib_align ? "YES" : "NO"
        println("\nGlobal recommendation: $(global_rec) | celecoxib ΔTop10=$(round(celecoxib_delta, digits=4)) | drd2 ΔTop10=$(round(drd2_delta, digits=4)) | albuterol ΔTop10=$(round(albuterol_delta, digits=4)) | celecoxib alignment=$(celecoxib_align_repr)")
    end

    return Dict(
        "results_by_task" => all_results,
        "decisions" => decisions,
        "max_promoted" => max_promoted,
    )
end


function run_c3_parent_controller_checks(tasks::Vector{String};
    budget::Int,
    n_episodes::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=2,
    training_epochs::Int=20,
    parent_candidate_limit::Int=16,
    celecoxib_repeats::Int=5,
    control_repeats::Int=3,
    sanity_repeats::Int=2,
)
    println("\n" * "="^80)
    println("RUNNING BATCH 1B′ — FRONTIER-CONDITIONED PARENT CONTROLLER PIVOT")
    println("="^80)

    function class_counts_repr(counts)
        isempty(counts) && return "none"
        join(["$(key)=$(counts[key])" for key in sort(collect(keys(counts)))], ", ")
    end

    online_summary(runs) = isempty(runs) ? Dict("auc" => NaN, "top10" => NaN, "budget" => NaN, "graph_per_call" => NaN) : Dict(
        "auc" => _mean_run_metric(runs, run -> run["result"].auc_top10),
        "top10" => _mean_run_metric(runs, run -> run["result"].top10_mean),
        "budget" => _mean_run_metric(runs, run -> run["extra"]["budget_fraction_used"]),
        "graph_per_call" => _mean_run_metric(runs, run -> run["extra"]["graph_unique_molecules"] / max(run["result"].n_oracle_calls, 1)),
    )

    function parent_offline_score(val_eval::AbstractDict{String,<:Any})
        score = 1.0 * get(val_eval, "score_target_correlation", 0.0)
        score += 0.40 * get(val_eval, "frontier_utility_correlation", 0.0)
        score += 0.20 * get(val_eval, "enters_topk_correlation", 0.0)
        score += 0.15 * get(val_eval, "productive_degenerate_margin", 0.0)
        score += 0.15 * get(val_eval, "preserve_strong_heuristic_rate", 0.0)
        score -= 0.20 * get(val_eval, "disagreement_rate", 0.0)
        score -= 0.20 * get(val_eval, "high_confidence_disagreement_rate", 0.0)
        score -= 0.10 * get(val_eval, "rmse", 0.0)
        return score
    end

    celecoxib_task = "celecoxib_rediscovery"
    if !(celecoxib_task in tasks)
        error("Batch 1B′ requires celecoxib_rediscovery in PMO_TASKS")
    end

    all_results = Dict{String,Any}()
    decisions = Dict{String,Any}()

    println("\nTask: $(celecoxib_task) [offline collection]")
    collection_runs = Vector{Dict{String,Any}}()
    parent_logs = ParentDecisionLog[]
    proposal_logs = HierarchicalEditProposalLog[]
    decision_logs = HierarchicalEditDecisionLog[]
    for repeat_idx in 1:data_repeats
        result, extra = run_hierarchical_edit_pmo_task(celecoxib_task;
            budget=budget,
            n_episodes=n_episodes,
            target_seed=target_seed,
            enable_augmentation=true,
            enable_warmup=true,
            bootstrap_warmup_rounds=bootstrap_warmup_rounds,
            run_episodes=true,
            regime_name="parent-data-r$(repeat_idx)",
            config=HierarchicalEditConfig(
                max_step_attempts=max_step_attempts,
                max_operator_candidates=max_operator_candidates,
                parent_candidate_limit=parent_candidate_limit,
                use_learned_parent=false,
            ),
            verbose=true)
        push!(collection_runs, Dict("result" => result, "extra" => extra))
        append!(parent_logs, extra["parent_logs_raw"])
        append!(proposal_logs, extra["proposal_logs_raw"])
        append!(decision_logs, extra["decision_logs_raw"])
    end

    audit = audit_parent_dataset_coverage(parent_logs, proposal_logs, decision_logs)
    println("  Parent audit: logs=$(audit["parent_logs"]) | matched=$(audit["matched_attempt_outcomes"]) | proposal_cov=$(round(100*audit["proposal_coverage_fraction"], digits=1))% | decision_cov=$(round(100*audit["decision_coverage_fraction"], digits=1))% | empty=$(round(100*audit["empty_after_filter_fraction"], digits=1))%")
    println("  Parent classes: $(class_counts_repr(audit["class_counts"]))")

    parent_recipe_results = Vector{Dict{String,Any}}()
    target_modes = [:blended, :ordinal_productivity]
    feature_modes = [:basic, :augmented]
    model_kinds = [:linear, :mlp]
    promoted_base = nothing
    promoted_name = ""
    promoted_score = -Inf
    promoted_summary = Dict{String,Any}()

    for feature_mode in feature_modes
        for target_mode in target_modes
            dataset = extract_parent_controller_dataset(parent_logs, proposal_logs, decision_logs;
                target_mode=target_mode,
                feature_mode=feature_mode)
            stats = parent_controller_dataset_stats(dataset)
            stats_repr = class_counts_repr(stats["class_counts"])
            println("  Parent dataset $(feature_mode)__$(target_mode): size=$(stats["size"]) | positive=$(round(100*stats["positive_fraction"], digits=1))% | feat_dim=$(stats["feature_dim"]) | classes=$(stats_repr)")
            for model_kind in model_kinds
                recipe_name = "$(feature_mode)__$(target_mode)__$(model_kind)"
                recipe = Dict{String,Any}(
                    "recipe_name" => recipe_name,
                    "feature_mode" => String(feature_mode),
                    "target_mode" => String(target_mode),
                    "model_kind" => String(model_kind),
                    "dataset_stats" => stats,
                    "error" => nothing,
                    "summary" => Dict{String,Any}(),
                    "offline_score" => -Inf,
                )
                try
                    config = ParentControllerTrainingConfig(
                        n_epochs=training_epochs,
                        model_kind=model_kind,
                        feature_mode=feature_mode,
                        hidden_dim=32,
                        learning_rate=model_kind == :linear ? 1e-3 : 5e-3,
                    )
                    controller, summary = train_parent_controller(dataset;
                        config=config,
                        rng=Random.MersenneTwister(hash((celecoxib_task, feature_mode, target_mode, model_kind))))
                    val_eval = summary["val_eval"]
                    score = parent_offline_score(val_eval)
                    recipe["summary"] = summary
                    recipe["offline_score"] = score
                    push!(parent_recipe_results, recipe)
                    println("    Parent recipe $(rpad(recipe_name, 36)) | score=$(round(score, digits=4)) | corr=$(round(val_eval["score_target_correlation"], digits=4)) | fu_corr=$(round(val_eval["frontier_utility_correlation"], digits=4)) | disagree=$(round(val_eval["disagreement_rate"], digits=4)) | preserve=$(round(val_eval["preserve_strong_heuristic_rate"], digits=4))")
                    if score > promoted_score
                        promoted_score = score
                        promoted_base = controller
                        promoted_name = recipe_name
                        promoted_summary = summary
                    end
                catch err
                    recipe["error"] = sprint(showerror, err)
                    push!(parent_recipe_results, recipe)
                    println("    Parent recipe $(rpad(recipe_name, 36)) | failed: $(recipe["error"])")
                end
            end
        end
    end

    if isnothing(promoted_base)
        decisions["global_recommendation"] = "RETHINK_FIRST_LEARNED_FACTOR"
        all_results[celecoxib_task] = Dict(
            "collection_runs" => collection_runs,
            "audit" => audit,
            "recipe_results" => parent_recipe_results,
        )
        println("\nNo parent controller trained successfully; recommendation: RETHINK_FIRST_LEARNED_FACTOR")
        return Dict("results_by_task" => all_results, "decisions" => decisions)
    end

    anchored_controller = create_anchored_parent_controller(promoted_base; override_margin=0.05f0, preserve_margin=0.15f0)
    candidate_controllers = Dict(
        promoted_name => promoted_base,
        "anchored__$(promoted_name)" => anchored_controller,
    )
    println("\nPromoted parent controllers for online gate: $(join(collect(keys(candidate_controllers)), ", "))")

    heuristic_runs = Vector{Dict{String,Any}}()
    online_runs = Dict{String,Any}(name => Vector{Dict{String,Any}}() for name in keys(candidate_controllers))
    for repeat_idx in 1:celecoxib_repeats
        heuristic_result, heuristic_extra = run_hierarchical_edit_pmo_task(celecoxib_task;
            budget=budget,
            n_episodes=n_episodes,
            target_seed=target_seed,
            enable_augmentation=true,
            enable_warmup=true,
            bootstrap_warmup_rounds=bootstrap_warmup_rounds,
            run_episodes=true,
            regime_name="heuristic-parent-r$(repeat_idx)",
            config=HierarchicalEditConfig(
                max_step_attempts=max_step_attempts,
                max_operator_candidates=max_operator_candidates,
                parent_candidate_limit=parent_candidate_limit,
                use_learned_parent=false,
            ),
            verbose=true)
        push!(heuristic_runs, Dict("result" => heuristic_result, "extra" => heuristic_extra))

        for (name, controller) in candidate_controllers
            learned_result, learned_extra = run_hierarchical_edit_pmo_task(celecoxib_task;
                budget=budget,
                n_episodes=n_episodes,
                target_seed=target_seed,
                enable_augmentation=true,
                enable_warmup=true,
                bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                run_episodes=true,
                regime_name="$(name)-r$(repeat_idx)",
                config=HierarchicalEditConfig(
                    max_step_attempts=max_step_attempts,
                    max_operator_candidates=max_operator_candidates,
                    parent_candidate_limit=parent_candidate_limit,
                    use_learned_parent=true,
                    learned_parent_controller=controller,
                ),
                verbose=true)
            push!(online_runs[name], Dict("result" => learned_result, "extra" => learned_extra))
        end
    end

    heuristic_summary = online_summary(heuristic_runs)
    println("\nCelecoxib online heuristic: AUC=$(round(heuristic_summary["auc"], digits=4)) | Top10=$(round(heuristic_summary["top10"], digits=4)) | Graph/Call=$(round(heuristic_summary["graph_per_call"], digits=4))")
    best_controller_name = ""
    best_delta = -Inf
    best_delta_auc = -Inf
    best_delta_graph = 0.0
    online_summaries = Dict{String,Any}()
    for (name, runs) in online_runs
        summary = online_summary(runs)
        delta_top10 = summary["top10"] - heuristic_summary["top10"]
        delta_auc = summary["auc"] - heuristic_summary["auc"]
        delta_graph = summary["graph_per_call"] - heuristic_summary["graph_per_call"]
        online_summaries[name] = Dict(
            "summary" => summary,
            "delta_top10" => delta_top10,
            "delta_auc" => delta_auc,
            "delta_graph_per_call" => delta_graph,
        )
        println("  Celecoxib online $(rpad(name, 36)) | AUC=$(round(summary["auc"], digits=4)) | Top10=$(round(summary["top10"], digits=4)) | ΔTop10=$(round(delta_top10, digits=4)) | ΔAUC=$(round(delta_auc, digits=4)) | ΔGraph/Call=$(round(delta_graph, digits=4))")
        if delta_top10 > best_delta
            best_delta = delta_top10
            best_delta_auc = delta_auc
            best_delta_graph = delta_graph
            best_controller_name = name
        end
    end

    celecoxib_pass = best_delta >= 0.015 && best_delta_auc >= -1e-6 && best_delta_graph >= -0.01
    decisions["celecoxib_best_parent_controller"] = best_controller_name
    decisions["celecoxib_delta_top10"] = best_delta
    decisions["celecoxib_delta_auc"] = best_delta_auc
    decisions["celecoxib_pass"] = celecoxib_pass
    all_results[celecoxib_task] = Dict(
        "collection_runs" => collection_runs,
        "audit" => audit,
        "recipe_results" => parent_recipe_results,
        "promoted_controllers" => collect(keys(candidate_controllers)),
        "heuristic_runs" => heuristic_runs,
        "online_runs" => online_runs,
        "online_summaries" => online_summaries,
        "promoted_summary" => promoted_summary,
    )

    if celecoxib_pass
        println("\nCelecoxib pass achieved; running minimal transfer sanity checks.")
        promoted_controller = candidate_controllers[best_controller_name]
        for (task_name, repeats) in [("drd2", control_repeats), ("albuterol_similarity", sanity_repeats)]
            task_name in tasks || continue
            heuristic_task_runs = Vector{Dict{String,Any}}()
            learned_task_runs = Vector{Dict{String,Any}}()
            for repeat_idx in 1:repeats
                heuristic_result, heuristic_extra = run_hierarchical_edit_pmo_task(task_name;
                    budget=budget,
                    n_episodes=n_episodes,
                    target_seed=target_seed,
                    enable_augmentation=true,
                    enable_warmup=true,
                    bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                    run_episodes=true,
                    regime_name="heuristic-parent-$(task_name)-r$(repeat_idx)",
                    config=HierarchicalEditConfig(
                        max_step_attempts=max_step_attempts,
                        max_operator_candidates=max_operator_candidates,
                        parent_candidate_limit=parent_candidate_limit,
                        use_learned_parent=false,
                    ),
                    verbose=true)
                push!(heuristic_task_runs, Dict("result" => heuristic_result, "extra" => heuristic_extra))

                learned_result, learned_extra = run_hierarchical_edit_pmo_task(task_name;
                    budget=budget,
                    n_episodes=n_episodes,
                    target_seed=target_seed,
                    enable_augmentation=true,
                    enable_warmup=true,
                    bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                    run_episodes=true,
                    regime_name="$(best_controller_name)-$(task_name)-r$(repeat_idx)",
                    config=HierarchicalEditConfig(
                        max_step_attempts=max_step_attempts,
                        max_operator_candidates=max_operator_candidates,
                        parent_candidate_limit=parent_candidate_limit,
                        use_learned_parent=true,
                        learned_parent_controller=promoted_controller,
                    ),
                    verbose=true)
                push!(learned_task_runs, Dict("result" => learned_result, "extra" => learned_extra))
            end
            heur_summary = online_summary(heuristic_task_runs)
            learn_summary = online_summary(learned_task_runs)
            delta_top10 = learn_summary["top10"] - heur_summary["top10"]
            println("  Transfer $(task_name): heuristic Top10=$(round(heur_summary["top10"], digits=4)) | learned Top10=$(round(learn_summary["top10"], digits=4)) | ΔTop10=$(round(delta_top10, digits=4))")
            all_results[task_name] = Dict(
                "heuristic_runs" => heuristic_task_runs,
                "learned_runs" => learned_task_runs,
                "delta_top10" => delta_top10,
            )
        end
        decisions["global_recommendation"] = "PROMOTE_PARENT_AS_FIRST_LEARNED_FACTOR"
    else
        decisions["global_recommendation"] = "HOLD_AT_HEURISTIC_PARENT_CONTROL"
    end

    println("\nGlobal recommendation: $(decisions["global_recommendation"]) | celecoxib best=$(best_controller_name) | ΔTop10=$(round(best_delta, digits=4)) | ΔAUC=$(round(best_delta_auc, digits=4))")
    return Dict("results_by_task" => all_results, "decisions" => decisions)
end

function run_c4_parent_semantics_checks(tasks::Vector{String};
    budget::Int,
    n_episodes::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=2,
    training_epochs::Int=20,
    parent_candidate_limit::Int=16,
    celecoxib_repeats::Int=5,
    control_repeats::Int=3,
    sanity_repeats::Int=2,
)
    println("\n" * "="^80)
    println("RUNNING BATCH 1C — PARENT CREDIT SEMANTICS + TRANSFER-SAFE ABSTENTION")
    println("="^80)

    class_counts_repr(counts) = isempty(counts) ? "none" : join(["$(key)=$(counts[key])" for key in sort(collect(keys(counts)))], ", ")
    online_summary(runs) = isempty(runs) ? Dict("auc" => NaN, "top10" => NaN, "budget" => NaN, "graph_per_call" => NaN) : Dict(
        "auc" => _mean_run_metric(runs, run -> run["result"].auc_top10),
        "top10" => _mean_run_metric(runs, run -> run["result"].top10_mean),
        "budget" => _mean_run_metric(runs, run -> run["extra"]["budget_fraction_used"]),
        "graph_per_call" => _mean_run_metric(runs, run -> run["extra"]["graph_unique_molecules"] / max(run["result"].n_oracle_calls, 1)),
    )

    function parent_policy_summary(runs)
        logs = ParentDecisionLog[]
        for run in runs
            append!(logs, get(run["extra"], "parent_logs_raw", ParentDecisionLog[]))
        end
        if isempty(logs)
            return Dict(
                "override_rate" => 0.0,
                "abstain_rate" => 0.0,
                "ambiguous_fraction" => 0.0,
                "ambiguous_override_rate" => 0.0,
                "strong_heuristic_override_rate" => 0.0,
                "mean_heuristic_margin" => 0.0,
                "mean_learned_margin" => 0.0,
                "reason_counts" => Dict{String,Int}(),
            )
        end
        reason_counts = Dict{String,Int}()
        for log in logs
            reason_counts[log.selection_reason] = get(reason_counts, log.selection_reason, 0) + 1
        end
        return Dict(
            "override_rate" => mean(Float64[log.override_applied for log in logs]),
            "abstain_rate" => mean(Float64[log.abstained_to_heuristic for log in logs]),
            "ambiguous_fraction" => mean(Float64[log.heuristic_margin < 0.1 for log in logs]),
            "ambiguous_override_rate" => mean(Float64[(log.heuristic_margin < 0.1) && log.override_applied for log in logs]),
            "strong_heuristic_override_rate" => mean(Float64[(log.heuristic_margin >= 0.1) && log.override_applied for log in logs]),
            "mean_heuristic_margin" => mean(Float64[log.heuristic_margin for log in logs]),
            "mean_learned_margin" => mean(Float64[log.learned_margin for log in logs]),
            "reason_counts" => reason_counts,
        )
    end

    function parent_offline_score(val_eval::AbstractDict{String,<:Any})
        score = 1.00 * get(val_eval, "score_target_correlation", 0.0)
        score += 0.45 * get(val_eval, "frontier_utility_correlation", 0.0)
        score += 0.20 * get(val_eval, "enters_topk_correlation", 0.0)
        score += 0.20 * get(val_eval, "productive_degenerate_margin", 0.0)
        score += 0.15 * get(val_eval, "preserve_strong_heuristic_rate", 0.0)
        score += 0.10 * get(val_eval, "ambiguous_disagreement_rate", 0.0)
        score += 0.05 * get(val_eval, "mean_learned_advantage_vs_heuristic", 0.0)
        score -= 0.20 * get(val_eval, "strong_heuristic_disagreement_rate", 0.0)
        score -= 0.10 * get(val_eval, "rmse", 0.0)
        return score
    end

    celecoxib_task = "celecoxib_rediscovery"
    celecoxib_task in tasks || error("Batch 1C requires celecoxib_rediscovery in PMO_TASKS")

    all_results = Dict{String,Any}()
    decisions = Dict{String,Any}()

    println("\nTask: $(celecoxib_task) [offline collection]")
    collection_runs = Vector{Dict{String,Any}}()
    parent_logs = ParentDecisionLog[]
    proposal_logs = HierarchicalEditProposalLog[]
    decision_logs = HierarchicalEditDecisionLog[]
    for repeat_idx in 1:data_repeats
        result, extra = run_hierarchical_edit_pmo_task(celecoxib_task;
            budget=budget,
            n_episodes=n_episodes,
            target_seed=target_seed,
            enable_augmentation=true,
            enable_warmup=true,
            bootstrap_warmup_rounds=bootstrap_warmup_rounds,
            run_episodes=true,
            regime_name="parent-semantics-data-r$(repeat_idx)",
            config=HierarchicalEditConfig(
                max_step_attempts=max_step_attempts,
                max_operator_candidates=max_operator_candidates,
                parent_candidate_limit=parent_candidate_limit,
                use_learned_parent=false,
            ),
            verbose=true)
        push!(collection_runs, Dict("result" => result, "extra" => extra))
        append!(parent_logs, extra["parent_logs_raw"])
        append!(proposal_logs, extra["proposal_logs_raw"])
        append!(decision_logs, extra["decision_logs_raw"])
    end

    audit = audit_parent_dataset_coverage(parent_logs, proposal_logs, decision_logs)
    println("  Parent audit: logs=$(audit["parent_logs"]) | matched=$(audit["matched_attempt_outcomes"]) | proposal_cov=$(round(100*audit["proposal_coverage_fraction"], digits=1))% | decision_cov=$(round(100*audit["decision_coverage_fraction"], digits=1))% | empty=$(round(100*audit["empty_after_filter_fraction"], digits=1))%")
    println("  Parent classes: $(class_counts_repr(audit["class_counts"]))")

    parent_recipe_results = Vector{Dict{String,Any}}()
    trained_controllers = Dict{String,Any}()
    target_modes = [:blended, :heuristic_adjusted_blended, :relative_blended, :risk_adjusted_advantage, :ordinal_productivity]
    feature_modes = [:basic, :augmented]
    model_kinds = [:linear, :mlp]
    promoted_base = nothing
    promoted_name = ""
    promoted_score = -Inf
    promoted_summary = Dict{String,Any}()

    for feature_mode in feature_modes
        for target_mode in target_modes
            dataset = extract_parent_controller_dataset(parent_logs, proposal_logs, decision_logs;
                target_mode=target_mode,
                feature_mode=feature_mode)
            stats = parent_controller_dataset_stats(dataset)
            stats_repr = class_counts_repr(stats["class_counts"])
            reason_repr = class_counts_repr(stats["selection_reason_counts"])
            println("  Parent dataset $(feature_mode)__$(target_mode): size=$(stats["size"]) | positive=$(round(100*stats["positive_fraction"], digits=1))% | feat_dim=$(stats["feature_dim"]) | ambig=$(round(100*stats["heuristic_ambiguous_fraction"], digits=1))% | hmargin=$(round(stats["mean_heuristic_margin"], digits=4)) | reasons=$(reason_repr) | classes=$(stats_repr)")
            for model_kind in model_kinds
                recipe_name = "$(feature_mode)__$(target_mode)__$(model_kind)"
                recipe = Dict{String,Any}(
                    "recipe_name" => recipe_name,
                    "feature_mode" => String(feature_mode),
                    "target_mode" => String(target_mode),
                    "model_kind" => String(model_kind),
                    "dataset_stats" => stats,
                    "error" => nothing,
                    "summary" => Dict{String,Any}(),
                    "offline_score" => -Inf,
                )
                try
                    config = ParentControllerTrainingConfig(
                        n_epochs=training_epochs,
                        model_kind=model_kind,
                        feature_mode=feature_mode,
                        hidden_dim=32,
                        learning_rate=model_kind == :linear ? 1e-3 : 5e-3,
                    )
                    controller, summary = train_parent_controller(dataset;
                        config=config,
                        rng=Random.MersenneTwister(hash((celecoxib_task, feature_mode, target_mode, model_kind, :c4))))
                    val_eval = summary["val_eval"]
                    score = parent_offline_score(val_eval)
                    recipe["summary"] = summary
                    recipe["offline_score"] = score
                    trained_controllers[recipe_name] = controller
                    push!(parent_recipe_results, recipe)
                    println("    Parent recipe $(rpad(recipe_name, 36)) | score=$(round(score, digits=4)) | corr=$(round(val_eval["score_target_correlation"], digits=4)) | fu_corr=$(round(val_eval["frontier_utility_correlation"], digits=4)) | ambig_dis=$(round(val_eval["ambiguous_disagreement_rate"], digits=4)) | strong_dis=$(round(val_eval["strong_heuristic_disagreement_rate"], digits=4)) | preserve=$(round(val_eval["preserve_strong_heuristic_rate"], digits=4))")
                    if score > promoted_score
                        promoted_score = score
                        promoted_base = controller
                        promoted_name = recipe_name
                        promoted_summary = summary
                    end
                catch err
                    recipe["error"] = sprint(showerror, err)
                    push!(parent_recipe_results, recipe)
                    println("    Parent recipe $(rpad(recipe_name, 36)) | failed: $(recipe["error"])")
                end
            end
        end
    end

    if isnothing(promoted_base)
        decisions["global_recommendation"] = "RETHINK_PARENT_CREDIT_OBJECT"
        all_results[celecoxib_task] = Dict(
            "collection_runs" => collection_runs,
            "audit" => audit,
            "recipe_results" => parent_recipe_results,
        )
        println("\nNo Batch 1C parent controller trained successfully; recommendation: RETHINK_PARENT_CREDIT_OBJECT")
        return Dict("results_by_task" => all_results, "decisions" => decisions)
    end

    candidate_controllers = Dict{String,Any}()
    if haskey(trained_controllers, "augmented__blended__mlp")
        candidate_controllers["anchored__augmented__blended__mlp"] = create_anchored_parent_controller(trained_controllers["augmented__blended__mlp"];
            override_margin=0.05f0, preserve_margin=0.15f0, learned_confidence_margin=0.05f0)
    end
    candidate_controllers["anchored__$(promoted_name)"] = create_anchored_parent_controller(promoted_base;
        override_margin=0.05f0, preserve_margin=0.15f0, learned_confidence_margin=0.05f0)
    candidate_controllers["cautious__anchored__$(promoted_name)"] = create_anchored_parent_controller(promoted_base;
        override_margin=0.08f0, preserve_margin=0.12f0, learned_confidence_margin=0.08f0)
    println("\nPromoted parent controllers for Batch 1C online gate: $(join(collect(keys(candidate_controllers)), ", "))")

    heuristic_runs = Vector{Dict{String,Any}}()
    deterministic_runs = Vector{Dict{String,Any}}()
    online_runs = Dict{String,Any}(name => Vector{Dict{String,Any}}() for name in keys(candidate_controllers))
    for repeat_idx in 1:celecoxib_repeats
        heuristic_result, heuristic_extra = run_hierarchical_edit_pmo_task(celecoxib_task;
            budget=budget,
            n_episodes=n_episodes,
            target_seed=target_seed,
            enable_augmentation=true,
            enable_warmup=true,
            bootstrap_warmup_rounds=bootstrap_warmup_rounds,
            run_episodes=true,
            regime_name="heuristic-parent-c4-r$(repeat_idx)",
            config=HierarchicalEditConfig(
                max_step_attempts=max_step_attempts,
                max_operator_candidates=max_operator_candidates,
                parent_candidate_limit=parent_candidate_limit,
                use_learned_parent=false,
            ),
            verbose=true)
        push!(heuristic_runs, Dict("result" => heuristic_result, "extra" => heuristic_extra))

        deterministic_result, deterministic_extra = run_hierarchical_edit_pmo_task(celecoxib_task;
            budget=budget,
            n_episodes=n_episodes,
            target_seed=target_seed,
            enable_augmentation=true,
            enable_warmup=true,
            bootstrap_warmup_rounds=bootstrap_warmup_rounds,
            run_episodes=true,
            regime_name="heuristic-top-parent-c4-r$(repeat_idx)",
            config=HierarchicalEditConfig(
                max_step_attempts=max_step_attempts,
                max_operator_candidates=max_operator_candidates,
                parent_candidate_limit=parent_candidate_limit,
                use_learned_parent=true,
                learned_parent_controller=HeuristicTopParentController(),
            ),
            verbose=true)
        push!(deterministic_runs, Dict("result" => deterministic_result, "extra" => deterministic_extra))

        for (name, controller) in candidate_controllers
            learned_result, learned_extra = run_hierarchical_edit_pmo_task(celecoxib_task;
                budget=budget,
                n_episodes=n_episodes,
                target_seed=target_seed,
                enable_augmentation=true,
                enable_warmup=true,
                bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                run_episodes=true,
                regime_name="$(name)-r$(repeat_idx)",
                config=HierarchicalEditConfig(
                    max_step_attempts=max_step_attempts,
                    max_operator_candidates=max_operator_candidates,
                    parent_candidate_limit=parent_candidate_limit,
                    use_learned_parent=true,
                    learned_parent_controller=controller,
                ),
                verbose=true)
            push!(online_runs[name], Dict("result" => learned_result, "extra" => learned_extra))
        end
    end

    heuristic_summary = online_summary(heuristic_runs)
    heuristic_policy = parent_policy_summary(heuristic_runs)
    deterministic_summary = online_summary(deterministic_runs)
    deterministic_policy = parent_policy_summary(deterministic_runs)
    println("\nCelecoxib online heuristic-sampled: AUC=$(round(heuristic_summary["auc"], digits=4)) | Top10=$(round(heuristic_summary["top10"], digits=4)) | Graph/Call=$(round(heuristic_summary["graph_per_call"], digits=4))")
    println("Celecoxib online heuristic-top:     AUC=$(round(deterministic_summary["auc"], digits=4)) | Top10=$(round(deterministic_summary["top10"], digits=4)) | Graph/Call=$(round(deterministic_summary["graph_per_call"], digits=4)) | override=$(round(deterministic_policy["override_rate"], digits=4))")
    current_anchor_top10 = haskey(candidate_controllers, "anchored__augmented__blended__mlp") ? NaN : deterministic_summary["top10"]
    best_controller_name = ""
    best_delta = -Inf
    best_delta_auc = -Inf
    best_delta_graph = 0.0
    online_summaries = Dict{String,Any}()
    for (name, runs) in online_runs
        summary = online_summary(runs)
        policy = parent_policy_summary(runs)
        delta_top10 = summary["top10"] - deterministic_summary["top10"]
        delta_auc = summary["auc"] - deterministic_summary["auc"]
        delta_graph = summary["graph_per_call"] - deterministic_summary["graph_per_call"]
        delta_vs_sampled = summary["top10"] - heuristic_summary["top10"]
        online_summaries[name] = Dict(
            "summary" => summary,
            "policy" => policy,
            "delta_top10" => delta_top10,
            "delta_auc" => delta_auc,
            "delta_graph_per_call" => delta_graph,
            "delta_top10_vs_sampled" => delta_vs_sampled,
        )
        if name == "anchored__augmented__blended__mlp"
            current_anchor_top10 = summary["top10"]
        end
        println("  Celecoxib online $(rpad(name, 36)) | AUC=$(round(summary["auc"], digits=4)) | Top10=$(round(summary["top10"], digits=4)) | ΔTop10_vs_top=$(round(delta_top10, digits=4)) | ΔTop10_vs_sampled=$(round(delta_vs_sampled, digits=4)) | override=$(round(policy["override_rate"], digits=4)) | abstain=$(round(policy["abstain_rate"], digits=4)) | strongOv=$(round(policy["strong_heuristic_override_rate"], digits=4))")
        if delta_top10 > best_delta
            best_delta = delta_top10
            best_delta_auc = delta_auc
            best_delta_graph = delta_graph
            best_controller_name = name
        end
    end

    best_top10 = online_summaries[best_controller_name]["summary"]["top10"]
    delta_vs_current_anchor = isfinite(current_anchor_top10) ? best_top10 - current_anchor_top10 : best_delta
    celecoxib_pass = best_delta >= 0.0 && best_delta_auc >= -1e-6 && best_delta_graph >= -0.01 && delta_vs_current_anchor >= -1e-6
    decisions["celecoxib_best_parent_controller"] = best_controller_name
    decisions["celecoxib_delta_top10_vs_heuristic_top"] = best_delta
    decisions["celecoxib_delta_auc_vs_heuristic_top"] = best_delta_auc
    decisions["celecoxib_delta_top10_vs_current_anchor"] = delta_vs_current_anchor
    decisions["celecoxib_pass"] = celecoxib_pass
    all_results[celecoxib_task] = Dict(
        "collection_runs" => collection_runs,
        "audit" => audit,
        "recipe_results" => parent_recipe_results,
        "promoted_controllers" => collect(keys(candidate_controllers)),
        "heuristic_runs" => heuristic_runs,
        "deterministic_heuristic_runs" => deterministic_runs,
        "online_runs" => online_runs,
        "online_summaries" => online_summaries,
        "promoted_summary" => promoted_summary,
    )

    if celecoxib_pass
        println("\nCelecoxib held under Batch 1C; running minimal transfer sanity checks.")
        promoted_controller = candidate_controllers[best_controller_name]
        for (task_name, repeats) in [("drd2", control_repeats), ("albuterol_similarity", sanity_repeats)]
            task_name in tasks || continue
            heuristic_task_runs = Vector{Dict{String,Any}}()
            deterministic_task_runs = Vector{Dict{String,Any}}()
            learned_task_runs = Vector{Dict{String,Any}}()
            for repeat_idx in 1:repeats
                heuristic_result, heuristic_extra = run_hierarchical_edit_pmo_task(task_name;
                    budget=budget,
                    n_episodes=n_episodes,
                    target_seed=target_seed,
                    enable_augmentation=true,
                    enable_warmup=true,
                    bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                    run_episodes=true,
                    regime_name="heuristic-parent-c4-$(task_name)-r$(repeat_idx)",
                    config=HierarchicalEditConfig(
                        max_step_attempts=max_step_attempts,
                        max_operator_candidates=max_operator_candidates,
                        parent_candidate_limit=parent_candidate_limit,
                        use_learned_parent=false,
                    ),
                    verbose=true)
                push!(heuristic_task_runs, Dict("result" => heuristic_result, "extra" => heuristic_extra))

                deterministic_result, deterministic_extra = run_hierarchical_edit_pmo_task(task_name;
                    budget=budget,
                    n_episodes=n_episodes,
                    target_seed=target_seed,
                    enable_augmentation=true,
                    enable_warmup=true,
                    bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                    run_episodes=true,
                    regime_name="heuristic-top-parent-c4-$(task_name)-r$(repeat_idx)",
                    config=HierarchicalEditConfig(
                        max_step_attempts=max_step_attempts,
                        max_operator_candidates=max_operator_candidates,
                        parent_candidate_limit=parent_candidate_limit,
                        use_learned_parent=true,
                        learned_parent_controller=HeuristicTopParentController(),
                    ),
                    verbose=true)
                push!(deterministic_task_runs, Dict("result" => deterministic_result, "extra" => deterministic_extra))

                learned_result, learned_extra = run_hierarchical_edit_pmo_task(task_name;
                    budget=budget,
                    n_episodes=n_episodes,
                    target_seed=target_seed,
                    enable_augmentation=true,
                    enable_warmup=true,
                    bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                    run_episodes=true,
                    regime_name="$(best_controller_name)-$(task_name)-r$(repeat_idx)",
                    config=HierarchicalEditConfig(
                        max_step_attempts=max_step_attempts,
                        max_operator_candidates=max_operator_candidates,
                        parent_candidate_limit=parent_candidate_limit,
                        use_learned_parent=true,
                        learned_parent_controller=promoted_controller,
                    ),
                    verbose=true)
                push!(learned_task_runs, Dict("result" => learned_result, "extra" => learned_extra))
            end
            heur_summary = online_summary(heuristic_task_runs)
            det_summary = online_summary(deterministic_task_runs)
            learn_summary = online_summary(learned_task_runs)
            learn_policy = parent_policy_summary(learned_task_runs)
            delta_top10 = learn_summary["top10"] - det_summary["top10"]
            println("  Transfer $(task_name): sampled=$(round(heur_summary["top10"], digits=4)) | top=$(round(det_summary["top10"], digits=4)) | learned=$(round(learn_summary["top10"], digits=4)) | ΔTop10_vs_top=$(round(delta_top10, digits=4)) | override=$(round(learn_policy["override_rate"], digits=4)) | strongOv=$(round(learn_policy["strong_heuristic_override_rate"], digits=4))")
            all_results[task_name] = Dict(
                "heuristic_runs" => heuristic_task_runs,
                "deterministic_heuristic_runs" => deterministic_task_runs,
                "learned_runs" => learned_task_runs,
                "delta_top10" => delta_top10,
                "learned_policy" => learn_policy,
            )
        end
        decisions["global_recommendation"] = "KEEP_OR_PROMOTE_TRANSFER_SAFE_PARENT_CONTROL"
    else
        decisions["global_recommendation"] = "KEEP_CURRENT_ANCHORED_PARENT_BASELINE"
    end

    println("\nGlobal recommendation: $(decisions["global_recommendation"]) | celecoxib best=$(best_controller_name) | ΔTop10_vs_top=$(round(best_delta, digits=4)) | ΔTop10_vs_anchor=$(round(delta_vs_current_anchor, digits=4))")
    return Dict("results_by_task" => all_results, "decisions" => decisions)
end

function run_c5_parent_causality_probe(tasks::Vector{String};
    budget::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=3,
    parent_candidate_limit::Int=4,
)
    println("\n" * "="^80)
    println("RUNNING BATCH 1D — PAIRED / FORCED-OVERRIDE PARENT CAUSALITY PROBE")
    println("="^80)

    function classify_probe(probe::Dict{String,Any})
        parents = get(probe, "parent_summaries", Dict{String,Any}[])
        isempty(parents) && return Dict(
            "classification" => "empty_state",
            "best_minus_heuristic" => 0.0,
            "utility_spread" => 0.0,
            "heuristic_margin" => 0.0,
            "best_parent_index" => 0,
            "heuristic_parent_index" => 0,
        )

        util(parent) = begin
            u = parent["best_frontier_utility_delta"]
            isfinite(u) ? u : -1e9
        end
        utilities = [util(parent) for parent in parents]
        heuristic_idx = 1
        best_idx = argmax(utilities)
        best_utility = utilities[best_idx]
        heuristic_utility = utilities[heuristic_idx]
        valid_utilities = [u for u in utilities if u > -1e8]
        spread = isempty(valid_utilities) ? 0.0 : maximum(valid_utilities) - minimum(valid_utilities)
        heuristic_scores = [parent["heuristic_score"] for parent in parents]
        heuristic_margin = length(heuristic_scores) >= 2 ? heuristic_scores[1] - heuristic_scores[2] : heuristic_scores[1]
        best_ops = Set(String(parent["best_operator"]) for parent in parents if !parent["all_degenerate"])
        all_degenerate = all(parent["all_degenerate"] for parent in parents)
        gap = best_utility - heuristic_utility
        threshold = 0.02

        classification = if all_degenerate
            "degenerate_state"
        elseif best_idx != heuristic_idx && gap > threshold && heuristic_margin >= 0.1
            "false_confidence_heuristic_state"
        elseif length(best_ops) > 1 && spread > threshold
            "parent_operator_interaction_state"
        elseif spread > threshold
            "parent_sensitive_opportunity_state"
        else
            "heuristic_dominant_invariant_state"
        end

        return Dict(
            "classification" => classification,
            "best_minus_heuristic" => gap,
            "utility_spread" => spread,
            "heuristic_margin" => heuristic_margin,
            "best_parent_index" => best_idx,
            "heuristic_parent_index" => heuristic_idx,
        )
    end

    function summarize_probe_runs(runs::Vector{Dict{String,Any}})
        classifications = Dict{String,Int}()
        gaps = Float64[]
        spreads = Float64[]
        margins = Float64[]
        heuristic_best_matches = Bool[]
        false_confidence = 0
        interaction = 0
        sensitive = 0
        degenerate = 0

        for run in runs
            cls = run["classification"]
            classifications[cls] = get(classifications, cls, 0) + 1
            push!(gaps, run["best_minus_heuristic"])
            push!(spreads, run["utility_spread"])
            push!(margins, run["heuristic_margin"])
            push!(heuristic_best_matches, run["best_parent_index"] == run["heuristic_parent_index"])
            false_confidence += cls == "false_confidence_heuristic_state" ? 1 : 0
            interaction += cls == "parent_operator_interaction_state" ? 1 : 0
            sensitive += (cls == "parent_sensitive_opportunity_state" || cls == "false_confidence_heuristic_state" || cls == "parent_operator_interaction_state") ? 1 : 0
            degenerate += cls == "degenerate_state" ? 1 : 0
        end

        n = max(length(runs), 1)
        return Dict(
            "n_runs" => length(runs),
            "classifications" => classifications,
            "mean_best_minus_heuristic" => isempty(gaps) ? 0.0 : mean(gaps),
            "mean_utility_spread" => isempty(spreads) ? 0.0 : mean(spreads),
            "mean_heuristic_margin" => isempty(margins) ? 0.0 : mean(margins),
            "heuristic_best_match_fraction" => isempty(heuristic_best_matches) ? 0.0 : mean(Float64.(heuristic_best_matches)),
            "sensitive_fraction" => sensitive / n,
            "false_confidence_fraction" => false_confidence / n,
            "interaction_fraction" => interaction / n,
            "degenerate_fraction" => degenerate / n,
        )
    end

    function class_counts_repr(counts)
        isempty(counts) && return "none"
        join(["$(key)=$(counts[key])" for key in sort(collect(keys(counts)))], ", ")
    end

    celecoxib_task = "celecoxib_rediscovery"
    celecoxib_task in tasks || error("Batch 1D requires celecoxib_rediscovery in PMO_TASKS")

    all_results = Dict{String,Any}()
    decisions = Dict{String,Any}()

    for task_name in tasks
        println("\nTask: $(task_name) [causal parent probe]")
        probe_runs = Vector{Dict{String,Any}}()
        for repeat_idx in 1:data_repeats
            oracle_mgr, budget_oracle, budget_oracle_batch = build_budget_oracles(task_name, budget)
            frontier_buffer = MolecularFrontierBuffer(5000)
            vocab = SMILESVocabulary()
            target_smiles = get(TARGET_SMILES, task_name, nothing)
            seed_pool = bootstrap_seed_pool(task_name;
                user_seed_smiles=String[],
                target_smiles=target_smiles,
                target_seed=target_seed)
            seed_stats = _seed_memories!(seed_pool, budget_oracle, vocab;
                reward_fn_batch=budget_oracle_batch,
                frontier_buffer=frontier_buffer,
                augmentation_count=bootstrap_augmentation_count(task_name, target_smiles, target_seed; enable_augmentation=true),
                verbose=true)
            warmup_stats = bootstrap_frontier_warmup!(task_name, frontier_buffer, oracle_mgr, budget_oracle_batch, vocab;
                target_smiles=target_smiles,
                rounds=bootstrap_warmup_rounds,
                verbose=true)

            cfg = HierarchicalEditConfig(
                max_step_attempts=max_step_attempts,
                max_operator_candidates=max_operator_candidates,
                parent_candidate_limit=parent_candidate_limit,
            )
            probe = probe_parent_interventions(frontier_buffer, budget_oracle, vocab;
                reward_fn_batch=budget_oracle_batch,
                config=cfg,
                target_smiles=target_smiles,
                budget_remaining=budget_remaining(oracle_mgr),
                created_at_step=repeat_idx,
                task_name=task_name,
                max_parents=parent_candidate_limit)
            diagnosis = classify_probe(probe)
            run = Dict{String,Any}(
                "probe" => probe,
                "classification" => diagnosis["classification"],
                "best_minus_heuristic" => diagnosis["best_minus_heuristic"],
                "utility_spread" => diagnosis["utility_spread"],
                "heuristic_margin" => diagnosis["heuristic_margin"],
                "best_parent_index" => diagnosis["best_parent_index"],
                "heuristic_parent_index" => diagnosis["heuristic_parent_index"],
                "seed_stats" => seed_stats,
                "warmup_stats" => warmup_stats,
                "calls_used" => oracle_mgr.calls_used,
            )
            push!(probe_runs, run)
            println("  Probe r$(repeat_idx): class=$(run["classification"]) | best-heur=$(round(run["best_minus_heuristic"], digits=4)) | spread=$(round(run["utility_spread"], digits=4)) | hmargin=$(round(run["heuristic_margin"], digits=4)) | calls=$(run["calls_used"]) ")
        end

        summary = summarize_probe_runs(probe_runs)
        println("  Summary: classes=$(class_counts_repr(summary["classifications"])) | sensitive=$(round(100*summary["sensitive_fraction"], digits=1))% | false_conf=$(round(100*summary["false_confidence_fraction"], digits=1))% | interaction=$(round(100*summary["interaction_fraction"], digits=1))% | mean_gap=$(round(summary["mean_best_minus_heuristic"], digits=4))")
        all_results[task_name] = Dict(
            "probe_runs" => probe_runs,
            "summary" => summary,
        )
    end

    cele_summary = all_results[celecoxib_task]["summary"]
    decisions["celecoxib_sensitive_fraction"] = cele_summary["sensitive_fraction"]
    decisions["celecoxib_false_confidence_fraction"] = cele_summary["false_confidence_fraction"]
    decisions["celecoxib_interaction_fraction"] = cele_summary["interaction_fraction"]
    decisions["celecoxib_mean_best_minus_heuristic"] = cele_summary["mean_best_minus_heuristic"]

    decisions["global_recommendation"] = if cele_summary["interaction_fraction"] >= 0.34
        "SHIFT_TO_JOINT_PARENT_OPERATOR_FACTOR"
    elseif cele_summary["sensitive_fraction"] >= 0.34
        "KEEP_PARENT_AS_FIRST_CAUSAL_FACTOR"
    elseif cele_summary["degenerate_fraction"] >= 0.5
        "SHIFT_TO_DEGENERACY_OR_FRONTIER_SHAPING"
    else
        "PARENT_MOSTLY_CONTEXT_UNDER_CURRENT_GEOMETRY"
    end

    println("\nGlobal recommendation: $(decisions["global_recommendation"]) | celecoxib sensitive=$(round(100*cele_summary["sensitive_fraction"], digits=1))% | false_conf=$(round(100*cele_summary["false_confidence_fraction"], digits=1))% | interaction=$(round(100*cele_summary["interaction_fraction"], digits=1))% | mean_gap=$(round(cele_summary["mean_best_minus_heuristic"], digits=4))")
    return Dict("results_by_task" => all_results, "decisions" => decisions)
end

function run_c6_parent_operator_causality_probe(tasks::Vector{String};
    budget::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=5,
    parent_candidate_limit::Int=4,
    max_basin_contexts::Int=2,
)
    println("\n" * "="^80)
    println("RUNNING BATCH 1E — REAUDITED JOINT PARENT-OPERATOR CAUSAL PROBE")
    println("="^80)

    function empty_joint_summary()
        return Dict{String,Any}(
            "candidate_mode" => "frontier",
            "basin_scaffold" => "",
            "matched_budget" => 0,
            "heuristic_margin" => 0.0,
            "heuristic_parent_index" => 0,
            "best_parent_index_normalized" => 0,
            "best_joint_parent_index_normalized" => 0,
            "best_joint_operator_normalized" => "none",
            "heuristic_parent_optimistic_utility_normalized" => -Inf,
            "best_parent_optimistic_utility_normalized" => -Inf,
            "parent_gap_normalized" => 0.0,
            "best_joint_utility_normalized" => -Inf,
            "joint_gap_vs_heuristic_parent_normalized" => 0.0,
            "parent_main_effect_normalized" => 0.0,
            "operator_main_effect_normalized" => 0.0,
            "interaction_effect_normalized" => 0.0,
            "best_pair_interaction_residual_normalized" => 0.0,
            "operator_instability_across_parents" => 0.0,
            "parent_instability_across_operators" => 0.0,
            "all_degenerate" => true,
            "parent_summaries" => Dict{String,Any}[],
        )
    end

    function primary_joint_summary(probe::Dict{String,Any})
        summaries = get(probe, "basin_summaries", Dict{String,Any}[])
        return isempty(summaries) ? empty_joint_summary() : summaries[1]
    end

    function basin_slice_diagnostics(slice_probe::Union{Nothing,Dict{String,Any}})
        if isnothing(slice_probe)
            return Dict("basin_conditioned" => false, "operator_changed" => false, "interaction_gap" => 0.0, "joint_gap" => 0.0)
        end
        summaries = get(slice_probe, "basin_summaries", Dict{String,Any}[])
        if length(summaries) < 2
            return Dict("basin_conditioned" => false, "operator_changed" => false, "interaction_gap" => 0.0, "joint_gap" => 0.0)
        end
        s1, s2 = summaries[1], summaries[2]
        operator_changed = String(s1["best_joint_operator_normalized"]) != String(s2["best_joint_operator_normalized"]) &&
                           String(s1["best_joint_operator_normalized"]) != "none" &&
                           String(s2["best_joint_operator_normalized"]) != "none"
        interaction_gap = abs(Float64(s1["interaction_effect_normalized"]) - Float64(s2["interaction_effect_normalized"]))
        joint_gap = abs(Float64(s1["best_joint_utility_normalized"]) - Float64(s2["best_joint_utility_normalized"]))
        basin_conditioned = operator_changed || interaction_gap > 0.015 || joint_gap > 0.015
        return Dict(
            "basin_conditioned" => basin_conditioned,
            "operator_changed" => operator_changed,
            "interaction_gap" => interaction_gap,
            "joint_gap" => joint_gap,
        )
    end

    function classify_joint_probe(frontier_probe::Dict{String,Any}, slice_probe::Union{Nothing,Dict{String,Any}}=nothing)
        summary = primary_joint_summary(frontier_probe)
        slice_diag = basin_slice_diagnostics(slice_probe)
        parent_effect = Float64(summary["parent_main_effect_normalized"])
        operator_effect = Float64(summary["operator_main_effect_normalized"])
        interaction_effect = Float64(summary["interaction_effect_normalized"])
        joint_gap = Float64(summary["joint_gap_vs_heuristic_parent_normalized"])
        parent_gap = Float64(summary["parent_gap_normalized"])
        heuristic_margin = Float64(summary["heuristic_margin"])
        operator_instability = Float64(summary["operator_instability_across_parents"])
        parent_instability = Float64(summary["parent_instability_across_operators"])
        threshold = 0.01
        dominance_margin = 0.005

        classification = if Bool(summary["all_degenerate"])
            "degenerate_joint_state"
        elseif Bool(slice_diag["basin_conditioned"]) && max(interaction_effect, joint_gap, parent_gap) > threshold
            "basin_conditioned_joint_state"
        elseif joint_gap > threshold && Int(summary["best_joint_parent_index_normalized"]) != Int(summary["heuristic_parent_index"]) && heuristic_margin >= 0.1
            "false_confidence_joint_state"
        elseif interaction_effect > max(parent_effect, operator_effect) + dominance_margin && interaction_effect > threshold && (operator_instability > 0.0 || parent_instability > 0.0)
            "joint_interaction_state"
        elseif parent_effect > max(operator_effect, interaction_effect) + dominance_margin && parent_effect > threshold
            "parent_dominant_state"
        elseif operator_effect > max(parent_effect, interaction_effect) + dominance_margin && operator_effect > threshold
            "operator_dominant_state"
        elseif max(parent_effect, operator_effect, interaction_effect, joint_gap) <= threshold
            "joint_invariant_state"
        else
            "ambiguous_joint_state"
        end

        return Dict{String,Any}(
            "classification" => classification,
            "parent_main_effect_normalized" => parent_effect,
            "operator_main_effect_normalized" => operator_effect,
            "interaction_effect_normalized" => interaction_effect,
            "joint_gap_vs_heuristic_parent_normalized" => joint_gap,
            "parent_gap_normalized" => parent_gap,
            "heuristic_margin" => heuristic_margin,
            "matched_budget" => Int(summary["matched_budget"]),
            "best_joint_operator_normalized" => String(summary["best_joint_operator_normalized"]),
            "best_joint_parent_index_normalized" => Int(summary["best_joint_parent_index_normalized"]),
            "heuristic_parent_index" => Int(summary["heuristic_parent_index"]),
            "operator_instability_across_parents" => operator_instability,
            "parent_instability_across_operators" => parent_instability,
            "basin_conditioned" => Bool(slice_diag["basin_conditioned"]),
            "basin_operator_changed" => Bool(slice_diag["operator_changed"]),
            "basin_interaction_gap" => Float64(slice_diag["interaction_gap"]),
            "basin_joint_gap" => Float64(slice_diag["joint_gap"]),
        )
    end

    function summarize_joint_runs(runs::Vector{Dict{String,Any}})
        classifications = Dict{String,Int}()
        parent_effects = Float64[]
        operator_effects = Float64[]
        interaction_effects = Float64[]
        joint_gaps = Float64[]
        parent_gaps = Float64[]
        heuristic_margins = Float64[]
        matched_budgets = Int[]
        basin_conditioned = 0
        degenerate = 0
        ambiguous = 0
        parent_dominant = 0
        operator_dominant = 0
        joint_interaction = 0
        false_confidence = 0
        invariant = 0

        for run in runs
            cls = String(run["classification"])
            classifications[cls] = get(classifications, cls, 0) + 1
            push!(parent_effects, Float64(run["parent_main_effect_normalized"]))
            push!(operator_effects, Float64(run["operator_main_effect_normalized"]))
            push!(interaction_effects, Float64(run["interaction_effect_normalized"]))
            push!(joint_gaps, Float64(run["joint_gap_vs_heuristic_parent_normalized"]))
            push!(parent_gaps, Float64(run["parent_gap_normalized"]))
            push!(heuristic_margins, Float64(run["heuristic_margin"]))
            push!(matched_budgets, Int(run["matched_budget"]))
            basin_conditioned += Bool(run["basin_conditioned"]) ? 1 : 0
            degenerate += cls == "degenerate_joint_state" ? 1 : 0
            ambiguous += cls == "ambiguous_joint_state" ? 1 : 0
            parent_dominant += cls == "parent_dominant_state" ? 1 : 0
            operator_dominant += cls == "operator_dominant_state" ? 1 : 0
            joint_interaction += cls == "joint_interaction_state" ? 1 : 0
            false_confidence += cls == "false_confidence_joint_state" ? 1 : 0
            invariant += cls == "joint_invariant_state" ? 1 : 0
        end

        n = max(length(runs), 1)
        return Dict(
            "n_runs" => length(runs),
            "classifications" => classifications,
            "mean_parent_main_effect_normalized" => isempty(parent_effects) ? 0.0 : mean(parent_effects),
            "mean_operator_main_effect_normalized" => isempty(operator_effects) ? 0.0 : mean(operator_effects),
            "mean_interaction_effect_normalized" => isempty(interaction_effects) ? 0.0 : mean(interaction_effects),
            "mean_joint_gap_normalized" => isempty(joint_gaps) ? 0.0 : mean(joint_gaps),
            "mean_parent_gap_normalized" => isempty(parent_gaps) ? 0.0 : mean(parent_gaps),
            "mean_heuristic_margin" => isempty(heuristic_margins) ? 0.0 : mean(heuristic_margins),
            "mean_matched_budget" => isempty(matched_budgets) ? 0.0 : mean(Float64.(matched_budgets)),
            "basin_conditioned_fraction" => basin_conditioned / n,
            "degenerate_fraction" => degenerate / n,
            "ambiguous_fraction" => ambiguous / n,
            "parent_dominant_fraction" => parent_dominant / n,
            "operator_dominant_fraction" => operator_dominant / n,
            "joint_interaction_fraction" => joint_interaction / n,
            "false_confidence_fraction" => false_confidence / n,
            "invariant_fraction" => invariant / n,
            "interpretable" => (ambiguous + degenerate) / n < 0.75,
        )
    end

    function class_counts_repr(counts)
        isempty(counts) && return "none"
        join(["$(key)=$(counts[key])" for key in sort(collect(keys(counts)))], ", ")
    end

    celecoxib_task = "celecoxib_rediscovery"
    celecoxib_task in tasks || error("Batch 1E requires celecoxib_rediscovery in PMO_TASKS")

    all_results = Dict{String,Any}()
    decisions = Dict{String,Any}()

    for task_name in tasks
        println("\nTask: $(task_name) [joint parent-operator causal probe]")
        probe_runs = Vector{Dict{String,Any}}()
        for repeat_idx in 1:data_repeats
            oracle_mgr, budget_oracle, budget_oracle_batch = build_budget_oracles(task_name, budget)
            frontier_buffer = MolecularFrontierBuffer(5000)
            vocab = SMILESVocabulary()
            target_smiles = get(TARGET_SMILES, task_name, nothing)
            seed_pool = bootstrap_seed_pool(task_name;
                user_seed_smiles=String[],
                target_smiles=target_smiles,
                target_seed=target_seed)
            seed_stats = _seed_memories!(seed_pool, budget_oracle, vocab;
                reward_fn_batch=budget_oracle_batch,
                frontier_buffer=frontier_buffer,
                augmentation_count=bootstrap_augmentation_count(task_name, target_smiles, target_seed; enable_augmentation=true),
                verbose=true)
            warmup_stats = bootstrap_frontier_warmup!(task_name, frontier_buffer, oracle_mgr, budget_oracle_batch, vocab;
                target_smiles=target_smiles,
                rounds=bootstrap_warmup_rounds,
                verbose=true)

            cfg = HierarchicalEditConfig(
                max_step_attempts=max_step_attempts,
                max_operator_candidates=max_operator_candidates,
                parent_candidate_limit=parent_candidate_limit,
                basin_candidate_limit=max_basin_contexts,
            )
            frontier_probe = probe_parent_interventions(frontier_buffer, budget_oracle, vocab;
                reward_fn_batch=budget_oracle_batch,
                config=cfg,
                target_smiles=target_smiles,
                budget_remaining=budget_remaining(oracle_mgr),
                created_at_step=repeat_idx,
                task_name=task_name,
                max_parents=parent_candidate_limit,
                max_basins=1,
                restrict_parents_to_basin=false)
            basin_slice_probe = probe_parent_interventions(frontier_buffer, budget_oracle, vocab;
                reward_fn_batch=budget_oracle_batch,
                config=cfg,
                target_smiles=target_smiles,
                budget_remaining=budget_remaining(oracle_mgr),
                created_at_step=repeat_idx,
                task_name=task_name,
                max_parents=parent_candidate_limit,
                max_basins=max_basin_contexts,
                restrict_parents_to_basin=true)
            diagnosis = classify_joint_probe(frontier_probe, basin_slice_probe)
            run = Dict{String,Any}(
                "frontier_probe" => frontier_probe,
                "basin_slice_probe" => basin_slice_probe,
                "classification" => diagnosis["classification"],
                "parent_main_effect_normalized" => diagnosis["parent_main_effect_normalized"],
                "operator_main_effect_normalized" => diagnosis["operator_main_effect_normalized"],
                "interaction_effect_normalized" => diagnosis["interaction_effect_normalized"],
                "joint_gap_vs_heuristic_parent_normalized" => diagnosis["joint_gap_vs_heuristic_parent_normalized"],
                "parent_gap_normalized" => diagnosis["parent_gap_normalized"],
                "heuristic_margin" => diagnosis["heuristic_margin"],
                "matched_budget" => diagnosis["matched_budget"],
                "best_joint_operator_normalized" => diagnosis["best_joint_operator_normalized"],
                "basin_conditioned" => diagnosis["basin_conditioned"],
                "basin_operator_changed" => diagnosis["basin_operator_changed"],
                "basin_interaction_gap" => diagnosis["basin_interaction_gap"],
                "basin_joint_gap" => diagnosis["basin_joint_gap"],
                "seed_stats" => seed_stats,
                "warmup_stats" => warmup_stats,
                "calls_used" => oracle_mgr.calls_used,
            )
            push!(probe_runs, run)
            println("  Probe r$(repeat_idx): class=$(run["classification"]) | p=$(round(run["parent_main_effect_normalized"], digits=4)) | o=$(round(run["operator_main_effect_normalized"], digits=4)) | i=$(round(run["interaction_effect_normalized"], digits=4)) | jointGap=$(round(run["joint_gap_vs_heuristic_parent_normalized"], digits=4)) | basin=$(run["basin_conditioned"]) | mb=$(run["matched_budget"]) | calls=$(run["calls_used"]) ")
        end

        summary = summarize_joint_runs(probe_runs)
        println("  Summary: classes=$(class_counts_repr(summary["classifications"])) | parent=$(round(summary["mean_parent_main_effect_normalized"], digits=4)) | operator=$(round(summary["mean_operator_main_effect_normalized"], digits=4)) | interaction=$(round(summary["mean_interaction_effect_normalized"], digits=4)) | jointGap=$(round(summary["mean_joint_gap_normalized"], digits=4)) | basinCond=$(round(100*summary["basin_conditioned_fraction"], digits=1))% | ambiguous=$(round(100*summary["ambiguous_fraction"], digits=1))%")
        all_results[task_name] = Dict(
            "probe_runs" => probe_runs,
            "summary" => summary,
        )
    end

    cele_summary = all_results[celecoxib_task]["summary"]
    decisions["celecoxib_parent_main_effect_normalized"] = cele_summary["mean_parent_main_effect_normalized"]
    decisions["celecoxib_operator_main_effect_normalized"] = cele_summary["mean_operator_main_effect_normalized"]
    decisions["celecoxib_interaction_effect_normalized"] = cele_summary["mean_interaction_effect_normalized"]
    decisions["celecoxib_joint_gap_normalized"] = cele_summary["mean_joint_gap_normalized"]
    decisions["celecoxib_basin_conditioned_fraction"] = cele_summary["basin_conditioned_fraction"]
    decisions["celecoxib_interpretable"] = cele_summary["interpretable"]

    decisions["global_recommendation"] = if !Bool(cele_summary["interpretable"])
        "DEEPER_CAUSAL_STAGE_REQUIRED"
    elseif cele_summary["basin_conditioned_fraction"] >= 0.4
        "SHIFT_TO_BASIN_CONDITIONED_PARENT_OPERATOR_FACTOR"
    elseif cele_summary["joint_interaction_fraction"] >= 0.4
        "SHIFT_TO_JOINT_PARENT_OPERATOR_FACTOR"
    elseif cele_summary["operator_dominant_fraction"] >= 0.4
        "SHIFT_TO_OPERATOR_CONDITIONED_FACTOR"
    elseif cele_summary["parent_dominant_fraction"] >= 0.4
        "KEEP_PARENT_AS_FIRST_ACTIVE_FACTOR"
    elseif cele_summary["false_confidence_fraction"] >= 0.4
        "ACTIVE_PARENT_CONTROL_ON_FALSE_CONFIDENCE_STATES"
    else
        "NO_CLEAR_ACTIVE_FACTOR_YET"
    end

    println("\nGlobal recommendation: $(decisions["global_recommendation"]) | celecoxib parent=$(round(cele_summary["mean_parent_main_effect_normalized"], digits=4)) | operator=$(round(cele_summary["mean_operator_main_effect_normalized"], digits=4)) | interaction=$(round(cele_summary["mean_interaction_effect_normalized"], digits=4)) | basinCond=$(round(100*cele_summary["basin_conditioned_fraction"], digits=1))% | interpretable=$(cele_summary["interpretable"])")
    return Dict("results_by_task" => all_results, "decisions" => decisions)
end


function run_c7_basin_conditioned_operator_probe(tasks::Vector{String};
    budget::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=5,
    parent_candidate_limit::Int=4,
    max_basin_contexts::Int=2,
)
    println("
" * "="^80)
    println("RUNNING BATCH 1F — BASIN-CONDITIONED OPERATOR OPPORTUNITY PROBE")
    println("="^80)

    function empty_operator_summary()
        return Dict{String,Any}(
            "candidate_mode" => "frontier",
            "basin_scaffold" => "",
            "matched_budget" => 0,
            "heuristic_margin" => 0.0,
            "parent_main_effect_normalized" => 0.0,
            "operator_main_effect_normalized" => 0.0,
            "interaction_effect_normalized" => 0.0,
            "parent_main_effect_k1" => 0.0,
            "operator_main_effect_k1" => 0.0,
            "interaction_effect_k1" => 0.0,
            "parent_main_effect_k2" => 0.0,
            "operator_main_effect_k2" => 0.0,
            "interaction_effect_k2" => 0.0,
            "parent_main_effect_k4" => 0.0,
            "operator_main_effect_k4" => 0.0,
            "interaction_effect_k4" => 0.0,
            "operator_mean_top2_effect" => 0.0,
            "operator_mean_top4_effect" => 0.0,
            "interaction_mean_top2_effect" => 0.0,
            "interaction_mean_top4_effect" => 0.0,
            "operator_instability_across_parents" => 0.0,
            "parent_instability_across_operators" => 0.0,
            "all_degenerate" => true,
        )
    end

    function primary_operator_summary(probe::Dict{String,Any})
        summaries = get(probe, "basin_summaries", Dict{String,Any}[])
        return isempty(summaries) ? empty_operator_summary() : summaries[1]
    end

    function operator_basin_slice_diagnostics(slice_probe::Union{Nothing,Dict{String,Any}})
        if isnothing(slice_probe)
            return Dict("basin_conditioned" => false, "operator_changed_strict" => false, "operator_gap_k2" => 0.0, "operator_gap_top2" => 0.0)
        end
        summaries = get(slice_probe, "basin_summaries", Dict{String,Any}[])
        if length(summaries) < 2
            return Dict("basin_conditioned" => false, "operator_changed_strict" => false, "operator_gap_k2" => 0.0, "operator_gap_top2" => 0.0)
        end
        s1, s2 = summaries[1], summaries[2]
        operator_changed = String(s1["best_joint_operator_normalized"]) != String(s2["best_joint_operator_normalized"]) &&
                           String(s1["best_joint_operator_normalized"]) != "none" &&
                           String(s2["best_joint_operator_normalized"]) != "none"
        operator_gap_k2 = abs(Float64(s1["operator_main_effect_k2"]) - Float64(s2["operator_main_effect_k2"]))
        operator_gap_top2 = abs(Float64(s1["operator_mean_top2_effect"]) - Float64(s2["operator_mean_top2_effect"]))
        basin_conditioned = operator_changed || operator_gap_k2 > 0.015 || operator_gap_top2 > 0.015
        return Dict(
            "basin_conditioned" => basin_conditioned,
            "operator_changed_strict" => operator_changed,
            "operator_gap_k2" => operator_gap_k2,
            "operator_gap_top2" => operator_gap_top2,
        )
    end

    function classify_operator_probe(frontier_probe::Dict{String,Any}, slice_probe::Union{Nothing,Dict{String,Any}}=nothing)
        summary = primary_operator_summary(frontier_probe)
        slice_diag = operator_basin_slice_diagnostics(slice_probe)
        p_strict = Float64(summary["parent_main_effect_normalized"])
        o_strict = Float64(summary["operator_main_effect_normalized"])
        i_strict = Float64(summary["interaction_effect_normalized"])
        p_k1 = Float64(summary["parent_main_effect_k1"])
        o_k1 = Float64(summary["operator_main_effect_k1"])
        p_k2 = Float64(summary["parent_main_effect_k2"])
        o_k2 = Float64(summary["operator_main_effect_k2"])
        p_k4 = Float64(summary["parent_main_effect_k4"])
        o_k4 = Float64(summary["operator_main_effect_k4"])
        o_top2 = Float64(summary["operator_mean_top2_effect"])
        o_top4 = Float64(summary["operator_mean_top4_effect"])
        i_top2 = Float64(summary["interaction_mean_top2_effect"])
        heuristic_margin = Float64(summary["heuristic_margin"])
        matched_budget = Int(summary["matched_budget"])
        operator_instability = Float64(summary["operator_instability_across_parents"])
        parent_instability = Float64(summary["parent_instability_across_operators"])
        threshold = 0.01
        dominance_margin = 0.01

        robust_strict = o_strict > p_strict + dominance_margin && o_strict > i_strict + 0.0 && o_strict > threshold
        robust_k1 = o_k1 > p_k1 + dominance_margin && o_k1 > threshold
        robust_k2 = o_k2 > p_k2 + dominance_margin && o_k2 > threshold
        robust_k4 = o_k4 > p_k4 + dominance_margin && o_k4 > threshold
        robust_top2 = o_top2 > threshold
        robust_top4 = o_top4 > threshold
        operator_robust = robust_strict && robust_k1 && robust_k2 && robust_top2
        k1_only = robust_k1 && !robust_k2 && !robust_top2
        proposal_geometry = max(o_strict, o_k1, o_k2) > threshold && max(o_top2, o_top4) < 0.5 * max(o_k1, 1e-8)
        joint_context = i_strict > threshold && (operator_instability > 0.0 || parent_instability > 0.0)
        invariant = max(o_strict, o_k1, o_k2, o_k4, o_top2, o_top4, p_strict, i_strict) <= threshold

        classification = if Bool(summary["all_degenerate"])
            "degenerate_state"
        elseif Bool(slice_diag["basin_conditioned"]) && operator_robust
            "basin_conditioned_operator_state"
        elseif operator_robust
            "operator_dominant_robust_state"
        elseif k1_only
            "operator_dominant_k1_only_state"
        elseif proposal_geometry
            "proposal_geometry_state"
        elseif joint_context
            "joint_parent_operator_state"
        elseif invariant
            "invariant_state"
        else
            "ambiguous_state"
        end

        return Dict{String,Any}(
            "classification" => classification,
            "parent_main_effect_normalized" => p_strict,
            "operator_main_effect_normalized" => o_strict,
            "interaction_effect_normalized" => i_strict,
            "parent_main_effect_k1" => p_k1,
            "operator_main_effect_k1" => o_k1,
            "parent_main_effect_k2" => p_k2,
            "operator_main_effect_k2" => o_k2,
            "parent_main_effect_k4" => p_k4,
            "operator_main_effect_k4" => o_k4,
            "operator_mean_top2_effect" => o_top2,
            "operator_mean_top4_effect" => o_top4,
            "interaction_mean_top2_effect" => i_top2,
            "heuristic_margin" => heuristic_margin,
            "matched_budget" => matched_budget,
            "operator_instability_across_parents" => operator_instability,
            "parent_instability_across_operators" => parent_instability,
            "basin_conditioned" => Bool(slice_diag["basin_conditioned"]),
            "operator_changed_strict" => Bool(slice_diag["operator_changed_strict"]),
            "operator_gap_k2" => Float64(slice_diag["operator_gap_k2"]),
            "operator_gap_top2" => Float64(slice_diag["operator_gap_top2"]),
            "operator_robust" => operator_robust,
            "k1_only" => k1_only,
            "proposal_geometry" => proposal_geometry,
        )
    end

    function summarize_operator_runs(runs::Vector{Dict{String,Any}})
        classifications = Dict{String,Int}()
        o_strict = Float64[]
        o_k1 = Float64[]
        o_k2 = Float64[]
        o_k4 = Float64[]
        o_top2 = Float64[]
        o_top4 = Float64[]
        p_strict = Float64[]
        matched_budgets = Int[]
        basin_conditioned = 0
        operator_robust = 0
        k1_only = 0
        proposal_geometry = 0
        joint_context = 0
        ambiguous = 0
        degenerate = 0

        for run in runs
            cls = String(run["classification"])
            classifications[cls] = get(classifications, cls, 0) + 1
            push!(o_strict, Float64(run["operator_main_effect_normalized"]))
            push!(o_k1, Float64(run["operator_main_effect_k1"]))
            push!(o_k2, Float64(run["operator_main_effect_k2"]))
            push!(o_k4, Float64(run["operator_main_effect_k4"]))
            push!(o_top2, Float64(run["operator_mean_top2_effect"]))
            push!(o_top4, Float64(run["operator_mean_top4_effect"]))
            push!(p_strict, Float64(run["parent_main_effect_normalized"]))
            push!(matched_budgets, Int(run["matched_budget"]))
            basin_conditioned += Bool(run["basin_conditioned"]) ? 1 : 0
            operator_robust += Bool(run["operator_robust"]) ? 1 : 0
            k1_only += Bool(run["k1_only"]) ? 1 : 0
            proposal_geometry += Bool(run["proposal_geometry"]) ? 1 : 0
            joint_context += cls == "joint_parent_operator_state" ? 1 : 0
            ambiguous += cls == "ambiguous_state" ? 1 : 0
            degenerate += cls == "degenerate_state" ? 1 : 0
        end

        n = max(length(runs), 1)
        return Dict(
            "n_runs" => length(runs),
            "classifications" => classifications,
            "mean_parent_main_effect_normalized" => isempty(p_strict) ? 0.0 : mean(p_strict),
            "mean_operator_main_effect_normalized" => isempty(o_strict) ? 0.0 : mean(o_strict),
            "mean_operator_main_effect_k1" => isempty(o_k1) ? 0.0 : mean(o_k1),
            "mean_operator_main_effect_k2" => isempty(o_k2) ? 0.0 : mean(o_k2),
            "mean_operator_main_effect_k4" => isempty(o_k4) ? 0.0 : mean(o_k4),
            "mean_operator_mean_top2_effect" => isempty(o_top2) ? 0.0 : mean(o_top2),
            "mean_operator_mean_top4_effect" => isempty(o_top4) ? 0.0 : mean(o_top4),
            "mean_matched_budget" => isempty(matched_budgets) ? 0.0 : mean(Float64.(matched_budgets)),
            "basin_conditioned_fraction" => basin_conditioned / n,
            "operator_robust_fraction" => operator_robust / n,
            "k1_only_fraction" => k1_only / n,
            "proposal_geometry_fraction" => proposal_geometry / n,
            "joint_context_fraction" => joint_context / n,
            "ambiguous_fraction" => ambiguous / n,
            "degenerate_fraction" => degenerate / n,
            "interpretable" => (ambiguous + degenerate) / n < 0.75,
        )
    end

    function class_counts_repr(counts)
        isempty(counts) && return "none"
        join(["$(key)=$(counts[key])" for key in sort(collect(keys(counts)))], ", ")
    end

    celecoxib_task = "celecoxib_rediscovery"
    celecoxib_task in tasks || error("Batch 1F requires celecoxib_rediscovery in PMO_TASKS")

    all_results = Dict{String,Any}()
    decisions = Dict{String,Any}()

    for task_name in tasks
        println("
Task: $(task_name) [basin-conditioned operator probe]")
        probe_runs = Vector{Dict{String,Any}}()
        for repeat_idx in 1:data_repeats
            oracle_mgr, budget_oracle, budget_oracle_batch = build_budget_oracles(task_name, budget)
            frontier_buffer = MolecularFrontierBuffer(5000)
            vocab = SMILESVocabulary()
            target_smiles = get(TARGET_SMILES, task_name, nothing)
            seed_pool = bootstrap_seed_pool(task_name;
                user_seed_smiles=String[],
                target_smiles=target_smiles,
                target_seed=target_seed)
            seed_stats = _seed_memories!(seed_pool, budget_oracle, vocab;
                reward_fn_batch=budget_oracle_batch,
                frontier_buffer=frontier_buffer,
                augmentation_count=bootstrap_augmentation_count(task_name, target_smiles, target_seed; enable_augmentation=true),
                verbose=true)
            warmup_stats = bootstrap_frontier_warmup!(task_name, frontier_buffer, oracle_mgr, budget_oracle_batch, vocab;
                target_smiles=target_smiles,
                rounds=bootstrap_warmup_rounds,
                verbose=true)

            cfg = HierarchicalEditConfig(
                max_step_attempts=max_step_attempts,
                max_operator_candidates=max_operator_candidates,
                parent_candidate_limit=parent_candidate_limit,
                basin_candidate_limit=max_basin_contexts,
            )
            frontier_probe = probe_parent_interventions(frontier_buffer, budget_oracle, vocab;
                reward_fn_batch=budget_oracle_batch,
                config=cfg,
                target_smiles=target_smiles,
                budget_remaining=budget_remaining(oracle_mgr),
                created_at_step=repeat_idx,
                task_name=task_name,
                max_parents=parent_candidate_limit,
                max_basins=1,
                restrict_parents_to_basin=false)
            basin_slice_probe = probe_parent_interventions(frontier_buffer, budget_oracle, vocab;
                reward_fn_batch=budget_oracle_batch,
                config=cfg,
                target_smiles=target_smiles,
                budget_remaining=budget_remaining(oracle_mgr),
                created_at_step=repeat_idx,
                task_name=task_name,
                max_parents=parent_candidate_limit,
                max_basins=max_basin_contexts,
                restrict_parents_to_basin=true)
            diagnosis = classify_operator_probe(frontier_probe, basin_slice_probe)
            run = Dict{String,Any}(
                "frontier_probe" => frontier_probe,
                "basin_slice_probe" => basin_slice_probe,
                "classification" => diagnosis["classification"],
                "parent_main_effect_normalized" => diagnosis["parent_main_effect_normalized"],
                "operator_main_effect_normalized" => diagnosis["operator_main_effect_normalized"],
                "interaction_effect_normalized" => diagnosis["interaction_effect_normalized"],
                "parent_main_effect_k1" => diagnosis["parent_main_effect_k1"],
                "operator_main_effect_k1" => diagnosis["operator_main_effect_k1"],
                "parent_main_effect_k2" => diagnosis["parent_main_effect_k2"],
                "operator_main_effect_k2" => diagnosis["operator_main_effect_k2"],
                "parent_main_effect_k4" => diagnosis["parent_main_effect_k4"],
                "operator_main_effect_k4" => diagnosis["operator_main_effect_k4"],
                "operator_mean_top2_effect" => diagnosis["operator_mean_top2_effect"],
                "operator_mean_top4_effect" => diagnosis["operator_mean_top4_effect"],
                "heuristic_margin" => diagnosis["heuristic_margin"],
                "matched_budget" => diagnosis["matched_budget"],
                "basin_conditioned" => diagnosis["basin_conditioned"],
                "operator_changed_strict" => diagnosis["operator_changed_strict"],
                "operator_gap_k2" => diagnosis["operator_gap_k2"],
                "operator_gap_top2" => diagnosis["operator_gap_top2"],
                "operator_robust" => diagnosis["operator_robust"],
                "k1_only" => diagnosis["k1_only"],
                "proposal_geometry" => diagnosis["proposal_geometry"],
                "seed_stats" => seed_stats,
                "warmup_stats" => warmup_stats,
                "calls_used" => oracle_mgr.calls_used,
            )
            push!(probe_runs, run)
            println("  Probe r$(repeat_idx): class=$(run["classification"]) | strict=$(round(run["operator_main_effect_normalized"], digits=4)) | k1=$(round(run["operator_main_effect_k1"], digits=4)) | k2=$(round(run["operator_main_effect_k2"], digits=4)) | top2=$(round(run["operator_mean_top2_effect"], digits=4)) | basin=$(run["basin_conditioned"]) | mb=$(run["matched_budget"]) | calls=$(run["calls_used"]) ")
        end

        summary = summarize_operator_runs(probe_runs)
        println("  Summary: classes=$(class_counts_repr(summary["classifications"])) | strict=$(round(summary["mean_operator_main_effect_normalized"], digits=4)) | k1=$(round(summary["mean_operator_main_effect_k1"], digits=4)) | k2=$(round(summary["mean_operator_main_effect_k2"], digits=4)) | top2=$(round(summary["mean_operator_mean_top2_effect"], digits=4)) | basinCond=$(round(100*summary["basin_conditioned_fraction"], digits=1))% | robust=$(round(100*summary["operator_robust_fraction"], digits=1))% | proposalGeom=$(round(100*summary["proposal_geometry_fraction"], digits=1))%")
        all_results[task_name] = Dict("probe_runs" => probe_runs, "summary" => summary)
    end

    cele_summary = all_results[celecoxib_task]["summary"]
    decisions["celecoxib_operator_main_effect_normalized"] = cele_summary["mean_operator_main_effect_normalized"]
    decisions["celecoxib_operator_main_effect_k1"] = cele_summary["mean_operator_main_effect_k1"]
    decisions["celecoxib_operator_main_effect_k2"] = cele_summary["mean_operator_main_effect_k2"]
    decisions["celecoxib_operator_mean_top2_effect"] = cele_summary["mean_operator_mean_top2_effect"]
    decisions["celecoxib_basin_conditioned_fraction"] = cele_summary["basin_conditioned_fraction"]
    decisions["celecoxib_operator_robust_fraction"] = cele_summary["operator_robust_fraction"]
    decisions["celecoxib_proposal_geometry_fraction"] = cele_summary["proposal_geometry_fraction"]
    decisions["celecoxib_interpretable"] = cele_summary["interpretable"]

    decisions["global_recommendation"] = if !Bool(cele_summary["interpretable"])
        "DEEPER_CAUSAL_STAGE_REQUIRED"
    elseif cele_summary["proposal_geometry_fraction"] >= 0.5
        "SHIFT_TO_PROPOSAL_SELECTION_SEAM"
    elseif cele_summary["operator_robust_fraction"] >= 0.5 && cele_summary["basin_conditioned_fraction"] >= 0.5
        "SHIFT_TO_BASIN_CONDITIONED_OPERATOR_CONTROLLER"
    elseif cele_summary["operator_robust_fraction"] >= 0.5
        "SHIFT_TO_OPERATOR_CONTROLLER"
    elseif cele_summary["joint_context_fraction"] >= 0.5
        "KEEP_BASIN_CONDITIONED_JOINT_FACTOR"
    else
        "NO_CLEAR_ACTIVE_FACTOR_YET"
    end

    println("
Global recommendation: $(decisions["global_recommendation"]) | celecoxib strict=$(round(cele_summary["mean_operator_main_effect_normalized"], digits=4)) | k1=$(round(cele_summary["mean_operator_main_effect_k1"], digits=4)) | k2=$(round(cele_summary["mean_operator_main_effect_k2"], digits=4)) | top2=$(round(cele_summary["mean_operator_mean_top2_effect"], digits=4)) | basinCond=$(round(100*cele_summary["basin_conditioned_fraction"], digits=1))% | robust=$(round(100*cele_summary["operator_robust_fraction"], digits=1))%")
    return Dict("results_by_task" => all_results, "decisions" => decisions)
end


function run_c8_operator_controller_checks(tasks::Vector{String};
    budget::Int,
    n_episodes::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=3,
    training_epochs::Int=20,
    celecoxib_repeats::Int=5,
    control_repeats::Int=3,
    sanity_repeats::Int=3,
)
    println("\n" * "="^80)
    println("RUNNING BATCH 1G — CELECOXIB-FIRST BASIN-CONDITIONED OPERATOR CONTROLLER")
    println("="^80)

    class_counts_repr(counts) = isempty(counts) ? "none" : join(["$(key)=$(counts[key])" for key in sort(collect(keys(counts)))], ", ")
    online_summary(runs) = isempty(runs) ? Dict("auc" => NaN, "top10" => NaN, "budget" => NaN, "graph_per_call" => NaN) : Dict(
        "auc" => _mean_run_metric(runs, run -> run["result"].auc_top10),
        "top10" => _mean_run_metric(runs, run -> run["result"].top10_mean),
        "budget" => _mean_run_metric(runs, run -> run["extra"]["budget_fraction_used"]),
        "graph_per_call" => _mean_run_metric(runs, run -> run["extra"]["graph_unique_molecules"] / max(run["result"].n_oracle_calls, 1)),
    )

    function operator_policy_summary(runs)
        logs = OperatorDecisionLog[]
        for run in runs
            append!(logs, get(run["extra"], "operator_logs_raw", OperatorDecisionLog[]))
        end
        if isempty(logs)
            return Dict(
                "override_rate" => 0.0,
                "abstain_rate" => 0.0,
                "strong_heuristic_override_rate" => 0.0,
                "mean_heuristic_margin" => 0.0,
                "mean_learned_margin" => 0.0,
                "reason_counts" => Dict{String,Int}(),
            )
        end
        reason_counts = Dict{String,Int}()
        for log in logs
            reason_counts[log.selection_reason] = get(reason_counts, log.selection_reason, 0) + 1
        end
        return Dict(
            "override_rate" => mean(Float64[log.override_applied for log in logs]),
            "abstain_rate" => mean(Float64[log.abstained_to_heuristic for log in logs]),
            "strong_heuristic_override_rate" => mean(Float64[(log.heuristic_margin >= 0.1) && log.override_applied for log in logs]),
            "mean_heuristic_margin" => mean(Float64[log.heuristic_margin for log in logs]),
            "mean_learned_margin" => mean(Float64[log.learned_margin for log in logs]),
            "reason_counts" => reason_counts,
        )
    end

    function operator_offline_score(val_eval::AbstractDict{String,<:Any})
        score = 1.0 * get(val_eval, "score_target_correlation", 0.0)
        score += 0.40 * get(val_eval, "frontier_utility_correlation", 0.0)
        score += 0.25 * get(val_eval, "robust_state_agreement", 0.0)
        score += 0.20 * get(val_eval, "eligible_state_agreement", 0.0)
        score += 0.15 * get(val_eval, "invariant_preserve_rate", 0.0)
        score -= 0.15 * get(val_eval, "ambiguous_disagreement_rate", 0.0)
        score -= 0.10 * get(val_eval, "rmse", 0.0)
        return score
    end

    celecoxib_task = "celecoxib_rediscovery"
    celecoxib_task in tasks || error("Batch 1G requires celecoxib_rediscovery in PMO_TASKS")

    all_results = Dict{String,Any}()
    decisions = Dict{String,Any}()

    println("\nTask: $(celecoxib_task) [offline collection]")
    collection_runs = Vector{Dict{String,Any}}()
    operator_logs = OperatorDecisionLog[]
    proposal_logs = HierarchicalEditProposalLog[]
    decision_logs = HierarchicalEditDecisionLog[]
    for repeat_idx in 1:data_repeats
        result, extra = run_hierarchical_edit_pmo_task(celecoxib_task;
            budget=budget,
            n_episodes=n_episodes,
            target_seed=target_seed,
            enable_augmentation=true,
            enable_warmup=true,
            bootstrap_warmup_rounds=bootstrap_warmup_rounds,
            run_episodes=true,
            regime_name="operator-data-r$(repeat_idx)",
            config=HierarchicalEditConfig(
                max_step_attempts=max_step_attempts,
                max_operator_candidates=max_operator_candidates,
                use_learned_operator=false,
            ),
            verbose=true)
        push!(collection_runs, Dict("result" => result, "extra" => extra))
        append!(operator_logs, get(extra, "operator_logs_raw", OperatorDecisionLog[]))
        append!(proposal_logs, extra["proposal_logs_raw"])
        append!(decision_logs, extra["decision_logs_raw"])
    end

    audit = audit_operator_dataset_coverage(operator_logs, proposal_logs, decision_logs)
    println("  Operator audit: logs=$(audit["operator_logs"]) | matched=$(audit["matched_attempt_outcomes"]) | proposal_cov=$(round(100*audit["proposal_coverage_fraction"], digits=1))% | decision_cov=$(round(100*audit["decision_coverage_fraction"], digits=1))% | empty=$(round(100*audit["empty_after_filter_fraction"], digits=1))%")
    println("  Outcome classes: $(class_counts_repr(audit["class_counts"]))")
    println("  State labels: $(class_counts_repr(audit["state_counts"]))")

    recipe_results = Vector{Dict{String,Any}}()
    trained_controllers = Dict{String,Any}()
    promoted_name = ""
    promoted_base = nothing
    promoted_score = -Inf
    promoted_summary = Dict{String,Any}()

    for feature_mode in [:basic, :augmented]
        dataset = extract_operator_controller_dataset(operator_logs, proposal_logs, decision_logs; feature_mode=feature_mode)
        stats = operator_controller_dataset_stats(dataset)
        println("  Operator dataset $(feature_mode): size=$(stats["size"]) | positive=$(round(100*stats["positive_fraction"], digits=1))% | eligible=$(round(100*stats["controller_eligible_fraction"], digits=1))% | feat_dim=$(stats["feature_dim"]) | states=$(class_counts_repr(stats["state_counts"]))")
        recipe = Dict{String,Any}(
            "recipe_name" => String(feature_mode),
            "feature_mode" => String(feature_mode),
            "dataset_stats" => stats,
            "error" => nothing,
            "summary" => Dict{String,Any}(),
            "offline_score" => -Inf,
        )
        try
            controller, summary = train_operator_controller(dataset;
                config=OperatorControllerTrainingConfig(
                    n_epochs=training_epochs,
                    feature_mode=feature_mode,
                    learning_rate=1e-3,
                ),
                rng=Random.MersenneTwister(hash((celecoxib_task, feature_mode, :c8))))
            score = operator_offline_score(summary["val_eval"])
            recipe["summary"] = summary
            recipe["offline_score"] = score
            trained_controllers[String(feature_mode)] = controller
            push!(recipe_results, recipe)
            val_eval = summary["val_eval"]
            println("    Operator recipe $(rpad(String(feature_mode), 12)) | score=$(round(score, digits=4)) | corr=$(round(val_eval["score_target_correlation"], digits=4)) | fu_corr=$(round(val_eval["frontier_utility_correlation"], digits=4)) | robust=$(round(val_eval["robust_state_agreement"], digits=4)) | preserve=$(round(val_eval["invariant_preserve_rate"], digits=4))")
            if score > promoted_score
                promoted_score = score
                promoted_name = String(feature_mode)
                promoted_base = controller
                promoted_summary = summary
            end
        catch err
            recipe["error"] = sprint(showerror, err)
            push!(recipe_results, recipe)
            println("    Operator recipe $(rpad(String(feature_mode), 12)) | failed: $(recipe["error"])")
        end
    end

    all_results[celecoxib_task] = Dict(
        "collection_runs" => collection_runs,
        "audit" => audit,
        "recipe_results" => recipe_results,
    )

    if isnothing(promoted_base)
        decisions["global_recommendation"] = "KEEP_OPERATOR_SEAM_REVISE_CONTROLLER_FORM"
        println("\nNo operator controller trained successfully; recommendation: KEEP_OPERATOR_SEAM_REVISE_CONTROLLER_FORM")
        return Dict("results_by_task" => all_results, "decisions" => decisions)
    end

    anchored_controller = create_anchored_operator_controller(promoted_base;
        override_margin=0.05f0,
        preserve_margin=0.15f0,
        learned_confidence_margin=0.05f0)
    candidate_controllers = Dict(
        promoted_name => promoted_base,
        "anchored__$(promoted_name)" => anchored_controller,
    )
    println("\nPromoted operator controllers for online gate: $(join(collect(keys(candidate_controllers)), ", "))")

    heuristic_runs = Vector{Dict{String,Any}}()
    deterministic_runs = Vector{Dict{String,Any}}()
    online_runs = Dict{String,Any}(name => Vector{Dict{String,Any}}() for name in keys(candidate_controllers))
    for repeat_idx in 1:celecoxib_repeats
        heuristic_result, heuristic_extra = run_hierarchical_edit_pmo_task(celecoxib_task;
            budget=budget,
            n_episodes=n_episodes,
            target_seed=target_seed,
            enable_augmentation=true,
            enable_warmup=true,
            bootstrap_warmup_rounds=bootstrap_warmup_rounds,
            run_episodes=true,
            regime_name="heuristic-operator-c8-r$(repeat_idx)",
            config=HierarchicalEditConfig(
                max_step_attempts=max_step_attempts,
                max_operator_candidates=max_operator_candidates,
                use_learned_operator=false,
            ),
            verbose=true)
        push!(heuristic_runs, Dict("result" => heuristic_result, "extra" => heuristic_extra))

        deterministic_result, deterministic_extra = run_hierarchical_edit_pmo_task(celecoxib_task;
            budget=budget,
            n_episodes=n_episodes,
            target_seed=target_seed,
            enable_augmentation=true,
            enable_warmup=true,
            bootstrap_warmup_rounds=bootstrap_warmup_rounds,
            run_episodes=true,
            regime_name="heuristic-top-operator-c8-r$(repeat_idx)",
            config=HierarchicalEditConfig(
                max_step_attempts=max_step_attempts,
                max_operator_candidates=max_operator_candidates,
                use_learned_operator=true,
                learned_operator_controller=HeuristicTopOperatorController(),
            ),
            verbose=true)
        push!(deterministic_runs, Dict("result" => deterministic_result, "extra" => deterministic_extra))

        for (name, controller) in candidate_controllers
            learned_result, learned_extra = run_hierarchical_edit_pmo_task(celecoxib_task;
                budget=budget,
                n_episodes=n_episodes,
                target_seed=target_seed,
                enable_augmentation=true,
                enable_warmup=true,
                bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                run_episodes=true,
                regime_name="$(name)-r$(repeat_idx)",
                config=HierarchicalEditConfig(
                    max_step_attempts=max_step_attempts,
                    max_operator_candidates=max_operator_candidates,
                    use_learned_operator=true,
                    learned_operator_controller=controller,
                ),
                verbose=true)
            push!(online_runs[name], Dict("result" => learned_result, "extra" => learned_extra))
        end
    end

    heuristic_summary = online_summary(heuristic_runs)
    deterministic_summary = online_summary(deterministic_runs)
    deterministic_policy = operator_policy_summary(deterministic_runs)
    println("\nCelecoxib online heuristic-sampled: AUC=$(round(heuristic_summary["auc"], digits=4)) | Top10=$(round(heuristic_summary["top10"], digits=4)) | Graph/Call=$(round(heuristic_summary["graph_per_call"], digits=4))")
    println("Celecoxib online heuristic-top:     AUC=$(round(deterministic_summary["auc"], digits=4)) | Top10=$(round(deterministic_summary["top10"], digits=4)) | Graph/Call=$(round(deterministic_summary["graph_per_call"], digits=4)) | override=$(round(deterministic_policy["override_rate"], digits=4))")

    best_controller_name = ""
    best_delta = -Inf
    best_delta_auc = -Inf
    best_policy = Dict{String,Any}()
    online_summaries = Dict{String,Any}()
    for (name, runs) in online_runs
        summary = online_summary(runs)
        policy = operator_policy_summary(runs)
        delta_top10 = summary["top10"] - deterministic_summary["top10"]
        delta_auc = summary["auc"] - deterministic_summary["auc"]
        delta_vs_sampled = summary["top10"] - heuristic_summary["top10"]
        online_summaries[name] = Dict(
            "summary" => summary,
            "policy" => policy,
            "delta_top10" => delta_top10,
            "delta_auc" => delta_auc,
            "delta_top10_vs_sampled" => delta_vs_sampled,
        )
        println("  Celecoxib online $(rpad(name, 24)) | AUC=$(round(summary["auc"], digits=4)) | Top10=$(round(summary["top10"], digits=4)) | ΔTop10_vs_top=$(round(delta_top10, digits=4)) | ΔTop10_vs_sampled=$(round(delta_vs_sampled, digits=4)) | override=$(round(policy["override_rate"], digits=4)) | abstain=$(round(policy["abstain_rate"], digits=4)) | strongOv=$(round(policy["strong_heuristic_override_rate"], digits=4))")
        if delta_top10 > best_delta
            best_delta = delta_top10
            best_delta_auc = delta_auc
            best_controller_name = name
            best_policy = policy
        end
    end

    celecoxib_pass = best_delta >= 0.015 && best_delta_auc >= -1e-6 && get(best_policy, "override_rate", 0.0) >= 0.05 && get(best_policy, "override_rate", 0.0) <= 0.80 && get(best_policy, "strong_heuristic_override_rate", 0.0) <= 0.50
    decisions["celecoxib_best_operator_controller"] = best_controller_name
    decisions["celecoxib_delta_top10_vs_heuristic_top"] = best_delta
    decisions["celecoxib_delta_auc_vs_heuristic_top"] = best_delta_auc
    decisions["celecoxib_override_rate"] = get(best_policy, "override_rate", 0.0)
    decisions["celecoxib_abstain_rate"] = get(best_policy, "abstain_rate", 0.0)
    decisions["celecoxib_pass"] = celecoxib_pass
    all_results[celecoxib_task]["promoted_controllers"] = collect(keys(candidate_controllers))
    all_results[celecoxib_task]["heuristic_runs"] = heuristic_runs
    all_results[celecoxib_task]["deterministic_heuristic_runs"] = deterministic_runs
    all_results[celecoxib_task]["online_runs"] = online_runs
    all_results[celecoxib_task]["online_summaries"] = online_summaries
    all_results[celecoxib_task]["promoted_summary"] = promoted_summary

    if celecoxib_pass
        promoted_controller = candidate_controllers[best_controller_name]
        println("\nCelecoxib passed Batch 1G gate; running minimal controls.")
        for (task_name, repeats) in [("drd2", control_repeats), ("albuterol_similarity", sanity_repeats)]
            task_name in tasks || continue
            deterministic_task_runs = Vector{Dict{String,Any}}()
            learned_task_runs = Vector{Dict{String,Any}}()
            for repeat_idx in 1:repeats
                det_result, det_extra = run_hierarchical_edit_pmo_task(task_name;
                    budget=budget,
                    n_episodes=n_episodes,
                    target_seed=target_seed,
                    enable_augmentation=true,
                    enable_warmup=true,
                    bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                    run_episodes=true,
                    regime_name="heuristic-top-operator-c8-$(task_name)-r$(repeat_idx)",
                    config=HierarchicalEditConfig(
                        max_step_attempts=max_step_attempts,
                        max_operator_candidates=max_operator_candidates,
                        use_learned_operator=true,
                        learned_operator_controller=HeuristicTopOperatorController(),
                    ),
                    verbose=true)
                push!(deterministic_task_runs, Dict("result" => det_result, "extra" => det_extra))

                learn_result, learn_extra = run_hierarchical_edit_pmo_task(task_name;
                    budget=budget,
                    n_episodes=n_episodes,
                    target_seed=target_seed,
                    enable_augmentation=true,
                    enable_warmup=true,
                    bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                    run_episodes=true,
                    regime_name="$(best_controller_name)-$(task_name)-r$(repeat_idx)",
                    config=HierarchicalEditConfig(
                        max_step_attempts=max_step_attempts,
                        max_operator_candidates=max_operator_candidates,
                        use_learned_operator=true,
                        learned_operator_controller=promoted_controller,
                    ),
                    verbose=true)
                push!(learned_task_runs, Dict("result" => learn_result, "extra" => learn_extra))
            end
            det_summary = online_summary(deterministic_task_runs)
            learn_summary = online_summary(learned_task_runs)
            learn_policy = operator_policy_summary(learned_task_runs)
            delta_top10 = learn_summary["top10"] - det_summary["top10"]
            println("  Transfer $(task_name): top=$(round(det_summary["top10"], digits=4)) | learned=$(round(learn_summary["top10"], digits=4)) | ΔTop10_vs_top=$(round(delta_top10, digits=4)) | override=$(round(learn_policy["override_rate"], digits=4))")
            all_results[task_name] = Dict(
                "deterministic_heuristic_runs" => deterministic_task_runs,
                "learned_runs" => learned_task_runs,
                "delta_top10" => delta_top10,
                "learned_policy" => learn_policy,
            )
        end
        decisions["global_recommendation"] = "PROMOTE_BASIN_CONDITIONED_OPERATOR_CONTROLLER"
    else
        decisions["global_recommendation"] = "KEEP_OPERATOR_SEAM_REVISE_CONTROLLER_FORM"
    end

    println("\nGlobal recommendation: $(decisions["global_recommendation"]) | celecoxib best=$(best_controller_name) | ΔTop10_vs_top=$(round(best_delta, digits=4)) | override=$(round(get(best_policy, "override_rate", 0.0), digits=4))")
    return Dict("results_by_task" => all_results, "decisions" => decisions)
end

function run_c9_operator_eligibility_checks(tasks::Vector{String};
    budget::Int,
    n_episodes::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=3,
    training_epochs::Int=20,
    celecoxib_repeats::Int=5,
    control_repeats::Int=3,
    sanity_repeats::Int=3,
)
    println("\n" * "="^80)
    println("RUNNING BATCH 1H — CELECOXIB-FIRST OPERATOR ELIGIBILITY / STATE-GATING")
    println("="^80)

    class_counts_repr(counts) = isempty(counts) ? "none" : join(["$(key)=$(counts[key])" for key in sort(collect(keys(counts)))], ", ")
    online_summary(runs) = isempty(runs) ? Dict("auc" => NaN, "top10" => NaN, "budget" => NaN, "graph_per_call" => NaN) : Dict(
        "auc" => _mean_run_metric(runs, run -> run["result"].auc_top10),
        "top10" => _mean_run_metric(runs, run -> run["result"].top10_mean),
        "budget" => _mean_run_metric(runs, run -> run["extra"]["budget_fraction_used"]),
        "graph_per_call" => _mean_run_metric(runs, run -> run["extra"]["graph_unique_molecules"] / max(run["result"].n_oracle_calls, 1)),
    )

    function operator_policy_summary(runs)
        logs = OperatorDecisionLog[]
        for run in runs
            append!(logs, get(run["extra"], "operator_logs_raw", OperatorDecisionLog[]))
        end
        if isempty(logs)
            return Dict(
                "predicted_eligible_fraction" => 0.0,
                "acted_on_fraction" => 0.0,
                "override_rate" => 0.0,
                "preserved_fraction" => 0.0,
                "abstain_rate" => 0.0,
                "strong_heuristic_override_rate" => 0.0,
                "mean_eligibility_score" => 0.0,
                "mean_heuristic_margin" => 0.0,
                "mean_learned_margin" => 0.0,
                "reason_counts" => Dict{String,Int}(),
            )
        end
        reason_counts = Dict{String,Int}()
        for log in logs
            reason_counts[log.selection_reason] = get(reason_counts, log.selection_reason, 0) + 1
        end
        return Dict(
            "predicted_eligible_fraction" => mean(Float64[log.predicted_eligible for log in logs]),
            "acted_on_fraction" => mean(Float64[log.acted_on for log in logs]),
            "override_rate" => mean(Float64[log.override_applied for log in logs]),
            "preserved_fraction" => mean(Float64[log.preserved_to_heuristic for log in logs]),
            "abstain_rate" => mean(Float64[log.abstained_to_heuristic for log in logs]),
            "strong_heuristic_override_rate" => mean(Float64[(log.heuristic_margin >= 0.1) && log.override_applied for log in logs]),
            "mean_eligibility_score" => mean(Float64[log.eligibility_score for log in logs]),
            "mean_heuristic_margin" => mean(Float64[log.heuristic_margin for log in logs]),
            "mean_learned_margin" => mean(Float64[log.learned_margin for log in logs]),
            "reason_counts" => reason_counts,
        )
    end

    function eligibility_offline_score(val_eval::AbstractDict{String,<:Any})
        score = 0.55 * get(val_eval, "eligible_recall", 0.0)
        score += 0.30 * get(val_eval, "invariant_preserve_rate", 0.0)
        score += 0.10 * get(val_eval, "overall_accuracy", 0.0)
        score -= 0.20 * get(val_eval, "ambiguous_activation_rate", 0.0)
        score -= 0.15 * get(val_eval, "degenerate_activation_rate", 0.0)
        return score
    end

    function ranking_offline_score(val_eval::AbstractDict{String,<:Any})
        score = 0.55 * get(val_eval, "score_target_correlation", 0.0)
        score += 0.25 * get(val_eval, "frontier_utility_correlation", 0.0)
        score += 0.20 * get(val_eval, "robust_state_agreement", 0.0)
        score += 0.10 * get(val_eval, "eligible_state_agreement", 0.0)
        score += 0.10 * get(val_eval, "invariant_preserve_rate", 0.0)
        score -= 0.10 * get(val_eval, "ambiguous_disagreement_rate", 0.0)
        score -= 0.10 * get(val_eval, "rmse", 0.0)
        return score
    end

    celecoxib_task = "celecoxib_rediscovery"
    celecoxib_task in tasks || error("Batch 1H requires celecoxib_rediscovery in PMO_TASKS")
    default_gating_threshold = 0.55f0

    all_results = Dict{String,Any}()
    decisions = Dict{String,Any}()

    println("\nTask: $(celecoxib_task) [offline collection]")
    collection_runs = Vector{Dict{String,Any}}()
    operator_logs = OperatorDecisionLog[]
    proposal_logs = HierarchicalEditProposalLog[]
    decision_logs = HierarchicalEditDecisionLog[]
    for repeat_idx in 1:data_repeats
        result, extra = run_hierarchical_edit_pmo_task(celecoxib_task;
            budget=budget,
            n_episodes=n_episodes,
            target_seed=target_seed,
            enable_augmentation=true,
            enable_warmup=true,
            bootstrap_warmup_rounds=bootstrap_warmup_rounds,
            run_episodes=true,
            regime_name="operator-eligibility-data-r$(repeat_idx)",
            config=HierarchicalEditConfig(
                max_step_attempts=max_step_attempts,
                max_operator_candidates=max_operator_candidates,
                use_learned_operator=false,
            ),
            verbose=true)
        push!(collection_runs, Dict("result" => result, "extra" => extra))
        append!(operator_logs, get(extra, "operator_logs_raw", OperatorDecisionLog[]))
        append!(proposal_logs, extra["proposal_logs_raw"])
        append!(decision_logs, extra["decision_logs_raw"])
    end

    audit = audit_operator_dataset_coverage(operator_logs, proposal_logs, decision_logs)
    println("  Operator audit: logs=$(audit["operator_logs"]) | matched=$(audit["matched_attempt_outcomes"]) | proposal_cov=$(round(100*audit["proposal_coverage_fraction"], digits=1))% | decision_cov=$(round(100*audit["decision_coverage_fraction"], digits=1))% | empty=$(round(100*audit["empty_after_filter_fraction"], digits=1))%")
    println("  Outcome classes: $(class_counts_repr(audit["class_counts"]))")
    println("  State labels: $(class_counts_repr(audit["state_counts"]))")

    recipe_results = Vector{Dict{String,Any}}()
    best_recipe_name = ""
    best_recipe_score = -Inf
    best_recipe = Dict{String,Any}()

    for feature_mode in [:basic, :augmented]
        dataset = extract_operator_controller_dataset(operator_logs, proposal_logs, decision_logs; feature_mode=feature_mode)
        stats = operator_controller_dataset_stats(dataset)
        eligible_dataset = filter_operator_controller_dataset(dataset; eligible_only=true)
        eligible_stats = operator_controller_dataset_stats(eligible_dataset)
        println("  Operator dataset $(feature_mode): size=$(stats["size"]) | eligible=$(round(100*stats["controller_eligible_fraction"], digits=1))% | acted=$(round(100*stats["acted_on_fraction"], digits=1))% | feat_dim=$(stats["feature_dim"]) | eligibility_dim=$(stats["eligibility_dim"]) | states=$(class_counts_repr(stats["state_counts"]))")
        println("    Eligible subset $(feature_mode): size=$(eligible_stats["size"]) | states=$(class_counts_repr(eligible_stats["state_counts"]))")

        recipe = Dict{String,Any}(
            "recipe_name" => String(feature_mode),
            "feature_mode" => String(feature_mode),
            "dataset_stats" => stats,
            "eligible_stats" => eligible_stats,
            "eligibility_error" => nothing,
            "ranking_error" => nothing,
            "eligibility_summary" => Dict{String,Any}(),
            "ranking_summary" => Dict{String,Any}(),
            "offline_score" => -Inf,
        )

        try
            eligibility_model, eligibility_summary = train_operator_eligibility_model(dataset;
                config=OperatorControllerTrainingConfig(
                    n_epochs=training_epochs,
                    feature_mode=feature_mode,
                    learning_rate=2e-3,
                    eligibility_threshold=default_gating_threshold,
                ),
                rng=Random.MersenneTwister(hash((celecoxib_task, feature_mode, :eligibility, :c9))))
            eligibility_val = eligibility_summary["val_eval"]
            eligibility_score = eligibility_offline_score(eligibility_val)
            recipe["eligibility_summary"] = eligibility_summary
            calibrated_threshold = get(eligibility_summary, "calibrated_threshold", Float64(default_gating_threshold))
            println("    Eligibility $(rpad(String(feature_mode), 11)) | score=$(round(eligibility_score, digits=4)) | thr=$(round(calibrated_threshold, digits=3)) | recall=$(round(eligibility_val["eligible_recall"], digits=4)) | preserve=$(round(eligibility_val["invariant_preserve_rate"], digits=4)) | ambigAct=$(round(eligibility_val["ambiguous_activation_rate"], digits=4)) | predElig=$(round(eligibility_val["predicted_eligible_fraction"], digits=4))")

            ranking_controller, ranking_summary = train_operator_controller(eligible_dataset;
                config=OperatorControllerTrainingConfig(
                    n_epochs=training_epochs,
                    feature_mode=feature_mode,
                    learning_rate=1e-3,
                    eligibility_threshold=default_gating_threshold,
                ),
                rng=Random.MersenneTwister(hash((celecoxib_task, feature_mode, :ranking, :c9))))
            ranking_val = ranking_summary["val_eval"]
            ranking_score = ranking_offline_score(ranking_val)
            combined_score = 0.55 * eligibility_score + 0.45 * ranking_score
            recipe["ranking_summary"] = ranking_summary
            recipe["offline_score"] = combined_score
            recipe["eligibility_model"] = eligibility_model
            recipe["ranking_controller"] = ranking_controller
            push!(recipe_results, recipe)
            println("    Ranking    $(rpad(String(feature_mode), 11)) | score=$(round(ranking_score, digits=4)) | corr=$(round(ranking_val["score_target_correlation"], digits=4)) | fu_corr=$(round(ranking_val["frontier_utility_correlation"], digits=4)) | robust=$(round(ranking_val["robust_state_agreement"], digits=4)) | eligible=$(round(ranking_val["eligible_state_agreement"], digits=4))")
            println("    Combined   $(rpad(String(feature_mode), 11)) | offline_score=$(round(combined_score, digits=4))")
            if combined_score > best_recipe_score
                best_recipe_score = combined_score
                best_recipe_name = String(feature_mode)
                best_recipe = recipe
            end
        catch err
            msg = sprint(showerror, err)
            if isempty(recipe["eligibility_summary"])
                recipe["eligibility_error"] = msg
            else
                recipe["ranking_error"] = msg
            end
            push!(recipe_results, recipe)
            println("    Batch 1H recipe $(rpad(String(feature_mode), 11)) | failed: $(msg)")
        end
    end

    all_results[celecoxib_task] = Dict(
        "collection_runs" => collection_runs,
        "audit" => audit,
        "recipe_results" => recipe_results,
    )

    if isempty(best_recipe_name)
        decisions["global_recommendation"] = "KEEP_OPERATOR_SEAM_REVISE_ELIGIBILITY_FORM"
        println("\nNo Batch 1H recipe trained successfully; recommendation: KEEP_OPERATOR_SEAM_REVISE_ELIGIBILITY_FORM")
        return Dict("results_by_task" => all_results, "decisions" => decisions)
    end

    eligibility_model = best_recipe["eligibility_model"]
    ranking_controller = best_recipe["ranking_controller"]
    gating_threshold = Float32(get(get(best_recipe, "eligibility_summary", Dict{String,Any}()), "calibrated_threshold", Float64(default_gating_threshold)))
    ungated_name = "ungated__$(best_recipe_name)"
    gated_name = "gated__$(best_recipe_name)"
    anchored_gated_name = "anchored_gated__$(best_recipe_name)"

    candidate_controllers = Dict(
        ungated_name => ranking_controller,
        gated_name => create_gated_operator_controller(eligibility_model, ranking_controller; eligibility_threshold=gating_threshold),
        anchored_gated_name => create_gated_operator_controller(
            eligibility_model,
            create_anchored_operator_controller(ranking_controller;
                override_margin=0.05f0,
                preserve_margin=0.15f0,
                learned_confidence_margin=0.05f0);
            eligibility_threshold=gating_threshold),
    )
    println("\nPromoted Batch 1H controllers for online gate: $(join(collect(keys(candidate_controllers)), ", "))")

    heuristic_runs = Vector{Dict{String,Any}}()
    deterministic_runs = Vector{Dict{String,Any}}()
    online_runs = Dict{String,Any}(name => Vector{Dict{String,Any}}() for name in keys(candidate_controllers))
    for repeat_idx in 1:celecoxib_repeats
        heuristic_result, heuristic_extra = run_hierarchical_edit_pmo_task(celecoxib_task;
            budget=budget,
            n_episodes=n_episodes,
            target_seed=target_seed,
            enable_augmentation=true,
            enable_warmup=true,
            bootstrap_warmup_rounds=bootstrap_warmup_rounds,
            run_episodes=true,
            regime_name="heuristic-operator-c9-r$(repeat_idx)",
            config=HierarchicalEditConfig(
                max_step_attempts=max_step_attempts,
                max_operator_candidates=max_operator_candidates,
                use_learned_operator=false,
            ),
            verbose=true)
        push!(heuristic_runs, Dict("result" => heuristic_result, "extra" => heuristic_extra))

        deterministic_result, deterministic_extra = run_hierarchical_edit_pmo_task(celecoxib_task;
            budget=budget,
            n_episodes=n_episodes,
            target_seed=target_seed,
            enable_augmentation=true,
            enable_warmup=true,
            bootstrap_warmup_rounds=bootstrap_warmup_rounds,
            run_episodes=true,
            regime_name="heuristic-top-operator-c9-r$(repeat_idx)",
            config=HierarchicalEditConfig(
                max_step_attempts=max_step_attempts,
                max_operator_candidates=max_operator_candidates,
                use_learned_operator=true,
                learned_operator_controller=HeuristicTopOperatorController(),
            ),
            verbose=true)
        push!(deterministic_runs, Dict("result" => deterministic_result, "extra" => deterministic_extra))

        for (name, controller) in candidate_controllers
            learned_result, learned_extra = run_hierarchical_edit_pmo_task(celecoxib_task;
                budget=budget,
                n_episodes=n_episodes,
                target_seed=target_seed,
                enable_augmentation=true,
                enable_warmup=true,
                bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                run_episodes=true,
                regime_name="$(name)-r$(repeat_idx)",
                config=HierarchicalEditConfig(
                    max_step_attempts=max_step_attempts,
                    max_operator_candidates=max_operator_candidates,
                    use_learned_operator=true,
                    learned_operator_controller=controller,
                ),
                verbose=true)
            push!(online_runs[name], Dict("result" => learned_result, "extra" => learned_extra))
        end
    end

    heuristic_summary = online_summary(heuristic_runs)
    deterministic_summary = online_summary(deterministic_runs)
    deterministic_policy = operator_policy_summary(deterministic_runs)
    println("\nCelecoxib online heuristic-sampled: AUC=$(round(heuristic_summary["auc"], digits=4)) | Top10=$(round(heuristic_summary["top10"], digits=4)) | Graph/Call=$(round(heuristic_summary["graph_per_call"], digits=4))")
    println("Celecoxib online heuristic-top:     AUC=$(round(deterministic_summary["auc"], digits=4)) | Top10=$(round(deterministic_summary["top10"], digits=4)) | Graph/Call=$(round(deterministic_summary["graph_per_call"], digits=4)) | predElig=$(round(deterministic_policy["predicted_eligible_fraction"], digits=4)) | acted=$(round(deterministic_policy["acted_on_fraction"], digits=4))")

    best_controller_name = ""
    best_delta = -Inf
    best_delta_auc = -Inf
    best_policy = Dict{String,Any}()
    online_summaries = Dict{String,Any}()
    for (name, runs) in online_runs
        summary = online_summary(runs)
        policy = operator_policy_summary(runs)
        delta_top10 = summary["top10"] - deterministic_summary["top10"]
        delta_auc = summary["auc"] - deterministic_summary["auc"]
        delta_vs_sampled = summary["top10"] - heuristic_summary["top10"]
        online_summaries[name] = Dict(
            "summary" => summary,
            "policy" => policy,
            "delta_top10" => delta_top10,
            "delta_auc" => delta_auc,
            "delta_top10_vs_sampled" => delta_vs_sampled,
        )
        println("  Celecoxib online $(rpad(name, 28)) | AUC=$(round(summary["auc"], digits=4)) | Top10=$(round(summary["top10"], digits=4)) | ΔTop10_vs_top=$(round(delta_top10, digits=4)) | predElig=$(round(policy["predicted_eligible_fraction"], digits=4)) | acted=$(round(policy["acted_on_fraction"], digits=4)) | preserve=$(round(policy["preserved_fraction"], digits=4)) | override=$(round(policy["override_rate"], digits=4)) | strongOv=$(round(policy["strong_heuristic_override_rate"], digits=4))")
        if startswith(name, "gated") || startswith(name, "anchored_gated")
            if delta_top10 > best_delta
                best_delta = delta_top10
                best_delta_auc = delta_auc
                best_controller_name = name
                best_policy = policy
            end
        end
    end

    ungated_delta = get(get(online_summaries, ungated_name, Dict{String,Any}()), "delta_top10", -Inf)
    celecoxib_pass = !isempty(best_controller_name) &&
        best_delta >= 0.015 &&
        best_delta_auc >= -1e-6 &&
        best_delta > ungated_delta + 1e-6 &&
        get(best_policy, "predicted_eligible_fraction", 0.0) >= 0.05 &&
        get(best_policy, "acted_on_fraction", 0.0) >= 0.05 &&
        get(best_policy, "override_rate", 0.0) >= 0.05 &&
        get(best_policy, "override_rate", 0.0) <= 0.60 &&
        get(best_policy, "strong_heuristic_override_rate", 0.0) <= 0.35

    decisions["celecoxib_best_operator_controller"] = best_controller_name
    decisions["celecoxib_best_recipe"] = best_recipe_name
    decisions["celecoxib_gating_threshold"] = gating_threshold
    decisions["celecoxib_delta_top10_vs_heuristic_top"] = best_delta
    decisions["celecoxib_delta_auc_vs_heuristic_top"] = best_delta_auc
    decisions["celecoxib_delta_top10_vs_ungated"] = best_delta - ungated_delta
    decisions["celecoxib_predicted_eligible_fraction"] = get(best_policy, "predicted_eligible_fraction", 0.0)
    decisions["celecoxib_acted_on_fraction"] = get(best_policy, "acted_on_fraction", 0.0)
    decisions["celecoxib_override_rate"] = get(best_policy, "override_rate", 0.0)
    decisions["celecoxib_abstain_rate"] = get(best_policy, "abstain_rate", 0.0)
    decisions["celecoxib_pass"] = celecoxib_pass
    all_results[celecoxib_task]["promoted_controllers"] = collect(keys(candidate_controllers))
    all_results[celecoxib_task]["heuristic_runs"] = heuristic_runs
    all_results[celecoxib_task]["deterministic_heuristic_runs"] = deterministic_runs
    all_results[celecoxib_task]["online_runs"] = online_runs
    all_results[celecoxib_task]["online_summaries"] = online_summaries
    all_results[celecoxib_task]["best_recipe"] = best_recipe

    if celecoxib_pass
        promoted_controller = candidate_controllers[best_controller_name]
        println("\nCelecoxib passed Batch 1H gate; running minimal controls.")
        for (task_name, repeats) in [("drd2", control_repeats), ("albuterol_similarity", sanity_repeats)]
            task_name in tasks || continue
            deterministic_task_runs = Vector{Dict{String,Any}}()
            learned_task_runs = Vector{Dict{String,Any}}()
            for repeat_idx in 1:repeats
                det_result, det_extra = run_hierarchical_edit_pmo_task(task_name;
                    budget=budget,
                    n_episodes=n_episodes,
                    target_seed=target_seed,
                    enable_augmentation=true,
                    enable_warmup=true,
                    bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                    run_episodes=true,
                    regime_name="heuristic-top-operator-c9-$(task_name)-r$(repeat_idx)",
                    config=HierarchicalEditConfig(
                        max_step_attempts=max_step_attempts,
                        max_operator_candidates=max_operator_candidates,
                        use_learned_operator=true,
                        learned_operator_controller=HeuristicTopOperatorController(),
                    ),
                    verbose=true)
                push!(deterministic_task_runs, Dict("result" => det_result, "extra" => det_extra))

                learn_result, learn_extra = run_hierarchical_edit_pmo_task(task_name;
                    budget=budget,
                    n_episodes=n_episodes,
                    target_seed=target_seed,
                    enable_augmentation=true,
                    enable_warmup=true,
                    bootstrap_warmup_rounds=bootstrap_warmup_rounds,
                    run_episodes=true,
                    regime_name="$(best_controller_name)-$(task_name)-r$(repeat_idx)",
                    config=HierarchicalEditConfig(
                        max_step_attempts=max_step_attempts,
                        max_operator_candidates=max_operator_candidates,
                        use_learned_operator=true,
                        learned_operator_controller=promoted_controller,
                    ),
                    verbose=true)
                push!(learned_task_runs, Dict("result" => learn_result, "extra" => learn_extra))
            end
            det_summary = online_summary(deterministic_task_runs)
            learn_summary = online_summary(learned_task_runs)
            learn_policy = operator_policy_summary(learned_task_runs)
            delta_top10 = learn_summary["top10"] - det_summary["top10"]
            println("  Transfer $(task_name): top=$(round(det_summary["top10"], digits=4)) | learned=$(round(learn_summary["top10"], digits=4)) | ΔTop10_vs_top=$(round(delta_top10, digits=4)) | predElig=$(round(learn_policy["predicted_eligible_fraction"], digits=4)) | acted=$(round(learn_policy["acted_on_fraction"], digits=4))")
            all_results[task_name] = Dict(
                "deterministic_heuristic_runs" => deterministic_task_runs,
                "learned_runs" => learned_task_runs,
                "delta_top10" => delta_top10,
                "learned_policy" => learn_policy,
            )
        end
        decisions["global_recommendation"] = "PROMOTE_STATE_GATED_OPERATOR_CONTROLLER"
    else
        decisions["global_recommendation"] = "KEEP_OPERATOR_SEAM_REVISE_ELIGIBILITY_FORM"
    end

    println("\nGlobal recommendation: $(decisions["global_recommendation"]) | celecoxib best=$(best_controller_name) | ΔTop10_vs_top=$(round(best_delta, digits=4)) | acted=$(round(get(best_policy, "acted_on_fraction", 0.0), digits=4)) | override=$(round(get(best_policy, "override_rate", 0.0), digits=4))")
    return Dict("results_by_task" => all_results, "decisions" => decisions)
end

function run_c10_coupled_option_probe(tasks::Vector{String};
    budget::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=5,
    parent_candidate_limit::Int=4,
    max_basin_contexts::Int=2,
    option_horizon::Int=3,
)
    println("\n" * "="^80)
    println("RUNNING BATCH 1I — FINAL-GOAL-ORIENTED COUPLED HIERARCHY OPTION PROBE")
    println("="^80)

    function class_counts_repr(counts)
        isempty(counts) && return "none"
        return join(["$(key)=$(counts[key])" for key in sort(collect(keys(counts)))], ", ")
    end

    function classify_coupled_option_probe(probe::Dict{String,Any})
        summary = get(probe, "summary", Dict{String,Any}())
        if isempty(summary)
            return Dict{String,Any}(
                "classification" => "degenerate_option_state",
                "family_a_gain_vs_baseline" => 0.0,
                "family_b_gain_vs_baseline" => 0.0,
                "family_c_gain_vs_baseline" => 0.0,
                "parent_coupling_gain" => 0.0,
                "basin_coupling_gain" => 0.0,
                "continuation_gain" => 0.0,
                "heuristic_baseline_utility" => -Inf,
            )
        end

        a_gain = Float64(get(summary, "family_a_gain_vs_baseline", 0.0))
        b_gain = Float64(get(summary, "family_b_gain_vs_baseline", 0.0))
        c_gain = Float64(get(summary, "family_c_gain_vs_baseline", 0.0))
        parent_coupling = Float64(get(summary, "parent_coupling_gain", 0.0))
        basin_coupling = Float64(get(summary, "basin_coupling_gain", 0.0))
        continuation_gain = Float64(get(summary, "family_c_best_continuation_gain", 0.0))
        heuristic_utility = Float64(get(summary, "heuristic_baseline_utility", -Inf))
        best_basin_range = Float64(get(summary, "best_basins_utility_range", 0.0))
        threshold = 0.01
        dominance_margin = 0.015
        max_gain = max(a_gain, b_gain, c_gain)

        classification = if Bool(get(summary, "all_degenerate", true))
            "degenerate_option_state"
        elseif max_gain <= threshold
            "invariant_horizon_state"
        elseif basin_coupling > max(parent_coupling, continuation_gain, threshold) + 0.005 && best_basin_range > threshold
            "frontier_allocation_suspected"
        elseif c_gain > max(a_gain, b_gain) + dominance_margin && (parent_coupling > threshold || basin_coupling > threshold || continuation_gain > threshold)
            "coupled_option_dominant"
        elseif continuation_gain > threshold && c_gain > threshold
            "continuation_sensitive"
        elseif abs(c_gain - a_gain) <= threshold && abs(b_gain - a_gain) <= threshold
            "local_object_sufficient"
        else
            "ambiguous_option_state"
        end

        return Dict{String,Any}(
            "classification" => classification,
            "family_a_gain_vs_baseline" => a_gain,
            "family_b_gain_vs_baseline" => b_gain,
            "family_c_gain_vs_baseline" => c_gain,
            "parent_coupling_gain" => parent_coupling,
            "basin_coupling_gain" => basin_coupling,
            "continuation_gain" => continuation_gain,
            "heuristic_baseline_utility" => heuristic_utility,
            "best_basin_utility_range" => best_basin_range,
        )
    end

    function summarize_coupled_option_runs(runs::Vector{Dict{String,Any}})
        classifications = Dict{String,Int}()
        a_gains = Float64[]
        b_gains = Float64[]
        c_gains = Float64[]
        parent_couplings = Float64[]
        basin_couplings = Float64[]
        continuation_gains = Float64[]
        heuristic_utils = Float64[]
        basin_ranges = Float64[]
        coupled_count = 0
        local_count = 0
        frontier_alloc = 0
        continuation = 0
        invariant = 0
        degenerate = 0
        ambiguous = 0

        for run in runs
            cls = String(run["classification"])
            classifications[cls] = get(classifications, cls, 0) + 1
            push!(a_gains, Float64(run["family_a_gain_vs_baseline"]))
            push!(b_gains, Float64(run["family_b_gain_vs_baseline"]))
            push!(c_gains, Float64(run["family_c_gain_vs_baseline"]))
            push!(parent_couplings, Float64(run["parent_coupling_gain"]))
            push!(basin_couplings, Float64(run["basin_coupling_gain"]))
            push!(continuation_gains, Float64(run["continuation_gain"]))
            push!(heuristic_utils, Float64(run["heuristic_baseline_utility"]))
            push!(basin_ranges, Float64(run["best_basin_utility_range"]))
            coupled_count += cls == "coupled_option_dominant" ? 1 : 0
            local_count += cls == "local_object_sufficient" ? 1 : 0
            frontier_alloc += cls == "frontier_allocation_suspected" ? 1 : 0
            continuation += cls == "continuation_sensitive" ? 1 : 0
            invariant += cls == "invariant_horizon_state" ? 1 : 0
            degenerate += cls == "degenerate_option_state" ? 1 : 0
            ambiguous += cls == "ambiguous_option_state" ? 1 : 0
        end

        n = max(length(runs), 1)
        return Dict(
            "n_runs" => length(runs),
            "classifications" => classifications,
            "mean_family_a_gain_vs_baseline" => isempty(a_gains) ? 0.0 : mean(a_gains),
            "mean_family_b_gain_vs_baseline" => isempty(b_gains) ? 0.0 : mean(b_gains),
            "mean_family_c_gain_vs_baseline" => isempty(c_gains) ? 0.0 : mean(c_gains),
            "mean_parent_coupling_gain" => isempty(parent_couplings) ? 0.0 : mean(parent_couplings),
            "mean_basin_coupling_gain" => isempty(basin_couplings) ? 0.0 : mean(basin_couplings),
            "mean_continuation_gain" => isempty(continuation_gains) ? 0.0 : mean(continuation_gains),
            "mean_heuristic_baseline_utility" => isempty(heuristic_utils) ? 0.0 : mean(heuristic_utils),
            "mean_best_basin_utility_range" => isempty(basin_ranges) ? 0.0 : mean(basin_ranges),
            "coupled_option_fraction" => coupled_count / n,
            "local_object_fraction" => local_count / n,
            "frontier_allocation_fraction" => frontier_alloc / n,
            "continuation_sensitive_fraction" => continuation / n,
            "invariant_fraction" => invariant / n,
            "degenerate_fraction" => degenerate / n,
            "ambiguous_fraction" => ambiguous / n,
            "interpretable" => (ambiguous + degenerate) / n < 0.75,
        )
    end

    function run_probe_task(task_name::String, repeats::Int)
        println("\nTask: $(task_name) [coupled hierarchy option probe]")
        probe_runs = Vector{Dict{String,Any}}()
        for repeat_idx in 1:repeats
            oracle_mgr, budget_oracle, budget_oracle_batch = build_budget_oracles(task_name, budget)
            frontier_buffer = MolecularFrontierBuffer(5000)
            vocab = SMILESVocabulary()
            target_smiles = get(TARGET_SMILES, task_name, nothing)
            seed_pool = bootstrap_seed_pool(task_name;
                user_seed_smiles=String[],
                target_smiles=target_smiles,
                target_seed=target_seed)
            seed_stats = _seed_memories!(seed_pool, budget_oracle, vocab;
                reward_fn_batch=budget_oracle_batch,
                frontier_buffer=frontier_buffer,
                augmentation_count=bootstrap_augmentation_count(task_name, target_smiles, target_seed; enable_augmentation=true),
                verbose=true)
            warmup_stats = bootstrap_frontier_warmup!(task_name, frontier_buffer, oracle_mgr, budget_oracle_batch, vocab;
                target_smiles=target_smiles,
                rounds=bootstrap_warmup_rounds,
                verbose=true)

            cfg = HierarchicalEditConfig(
                max_step_attempts=max_step_attempts,
                max_operator_candidates=max_operator_candidates,
                parent_candidate_limit=parent_candidate_limit,
                basin_candidate_limit=max_basin_contexts,
            )
            probe = probe_coupled_hierarchy_options(frontier_buffer, budget_oracle, vocab;
                reward_fn_batch=budget_oracle_batch,
                config=cfg,
                target_smiles=target_smiles,
                budget_remaining=budget_remaining(oracle_mgr),
                created_at_step=repeat_idx,
                task_name=task_name,
                max_parents=parent_candidate_limit,
                max_basins=max_basin_contexts,
                horizon=option_horizon)
            diagnosis = classify_coupled_option_probe(probe)
            run = Dict{String,Any}(
                "probe" => probe,
                "classification" => diagnosis["classification"],
                "family_a_gain_vs_baseline" => diagnosis["family_a_gain_vs_baseline"],
                "family_b_gain_vs_baseline" => diagnosis["family_b_gain_vs_baseline"],
                "family_c_gain_vs_baseline" => diagnosis["family_c_gain_vs_baseline"],
                "parent_coupling_gain" => diagnosis["parent_coupling_gain"],
                "basin_coupling_gain" => diagnosis["basin_coupling_gain"],
                "continuation_gain" => diagnosis["continuation_gain"],
                "heuristic_baseline_utility" => diagnosis["heuristic_baseline_utility"],
                "best_basin_utility_range" => diagnosis["best_basin_utility_range"],
                "seed_stats" => seed_stats,
                "warmup_stats" => warmup_stats,
                "calls_used" => oracle_mgr.calls_used,
            )
            push!(probe_runs, run)
            println("  Probe r$(repeat_idx): class=$(run["classification"]) | A=$(round(run["family_a_gain_vs_baseline"], digits=4)) | B=$(round(run["family_b_gain_vs_baseline"], digits=4)) | C=$(round(run["family_c_gain_vs_baseline"], digits=4)) | parent=$(round(run["parent_coupling_gain"], digits=4)) | basin=$(round(run["basin_coupling_gain"], digits=4)) | cont=$(round(run["continuation_gain"], digits=4)) | calls=$(run["calls_used"]) ")
        end

        summary = summarize_coupled_option_runs(probe_runs)
        println("  Summary: classes=$(class_counts_repr(summary["classifications"])) | A=$(round(summary["mean_family_a_gain_vs_baseline"], digits=4)) | B=$(round(summary["mean_family_b_gain_vs_baseline"], digits=4)) | C=$(round(summary["mean_family_c_gain_vs_baseline"], digits=4)) | parent=$(round(summary["mean_parent_coupling_gain"], digits=4)) | basin=$(round(summary["mean_basin_coupling_gain"], digits=4)) | cont=$(round(summary["mean_continuation_gain"], digits=4)) | coupled=$(round(100*summary["coupled_option_fraction"], digits=1))%")
        return Dict("probe_runs" => probe_runs, "summary" => summary)
    end

    celecoxib_task = "celecoxib_rediscovery"
    celecoxib_task in tasks || error("Batch 1I requires celecoxib_rediscovery in PMO_TASKS")

    all_results = Dict{String,Any}()
    decisions = Dict{String,Any}()

    all_results[celecoxib_task] = run_probe_task(celecoxib_task, data_repeats)
    cele_summary = all_results[celecoxib_task]["summary"]

    if Bool(cele_summary["interpretable"])
        for task_name in ["drd2", "albuterol_similarity"]
            task_name in tasks || continue
            all_results[task_name] = run_probe_task(task_name, 3)
        end
    end

    decisions["celecoxib_mean_family_a_gain_vs_baseline"] = cele_summary["mean_family_a_gain_vs_baseline"]
    decisions["celecoxib_mean_family_b_gain_vs_baseline"] = cele_summary["mean_family_b_gain_vs_baseline"]
    decisions["celecoxib_mean_family_c_gain_vs_baseline"] = cele_summary["mean_family_c_gain_vs_baseline"]
    decisions["celecoxib_mean_parent_coupling_gain"] = cele_summary["mean_parent_coupling_gain"]
    decisions["celecoxib_mean_basin_coupling_gain"] = cele_summary["mean_basin_coupling_gain"]
    decisions["celecoxib_mean_continuation_gain"] = cele_summary["mean_continuation_gain"]
    decisions["celecoxib_coupled_option_fraction"] = cele_summary["coupled_option_fraction"]
    decisions["celecoxib_frontier_allocation_fraction"] = cele_summary["frontier_allocation_fraction"]
    decisions["celecoxib_local_object_fraction"] = cele_summary["local_object_fraction"]
    decisions["celecoxib_interpretable"] = cele_summary["interpretable"]

    decisions["global_recommendation"] = if !Bool(cele_summary["interpretable"])
        "INCONCLUSIVE_REDESIGN_PROBE"
    elseif cele_summary["frontier_allocation_fraction"] >= 0.5
        "PIVOT_TO_FRONTIER_ALLOCATION_OBJECT"
    elseif cele_summary["coupled_option_fraction"] >= 0.5 || cele_summary["continuation_sensitive_fraction"] >= 0.5
        "PIVOT_TO_COUPLED_SHORT_HORIZON_OBJECT"
    elseif cele_summary["local_object_fraction"] >= 0.5
        "STAY_LOCAL_RECALIBRATE_CONTROLLER"
    else
        "MIXED_OBJECT_GEOMETRY_DEEPER_PROBE_REQUIRED"
    end

    println("\nGlobal recommendation: $(decisions["global_recommendation"]) | celecoxib A=$(round(cele_summary["mean_family_a_gain_vs_baseline"], digits=4)) | B=$(round(cele_summary["mean_family_b_gain_vs_baseline"], digits=4)) | C=$(round(cele_summary["mean_family_c_gain_vs_baseline"], digits=4)) | parent=$(round(cele_summary["mean_parent_coupling_gain"], digits=4)) | basin=$(round(cele_summary["mean_basin_coupling_gain"], digits=4)) | cont=$(round(cele_summary["mean_continuation_gain"], digits=4))")
    return Dict("results_by_task" => all_results, "decisions" => decisions)
end

function run_c11_subtrajectory_bridge(tasks::Vector{String};
    budget::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=5,
    parent_candidate_limit::Int=4,
    max_basin_contexts::Int=2,
    option_horizon::Int=3,
)
    println("\n" * "="^80)
    println("RUNNING BATCH 1J — THEORY-DERIVED SHORT-HORIZON SUBTRAJECTORY / CONTINUATION-VALUE BRIDGE")
    println("="^80)

    function class_counts_repr(counts)
        isempty(counts) && return "none"
        return join(["$(key)=$(counts[key])" for key in sort(collect(keys(counts)))], ", ")
    end

    function classify_subtrajectory_bridge(bridge::Dict{String,Any})
        local_gain = Float64(get(bridge, "local_gain_vs_baseline", 0.0))
        entry_gain = Float64(get(bridge, "entry_context_gain_vs_baseline", 0.0))
        sub_gain = Float64(get(bridge, "subtrajectory_gain_vs_baseline", 0.0))
        entry_over_local = Float64(get(bridge, "entry_context_gain_vs_local", 0.0))
        sub_over_entry = Float64(get(bridge, "subtrajectory_gain_vs_entry", 0.0))
        reorder = Float64(get(bridge, "entry_reorder_fraction", 0.0))
        frontier_range = Float64(get(bridge, "best_basin_utility_range", 0.0))
        continuation_gain = Float64(get(bridge, "max_entry_continuation_gain", 0.0))
        threshold = 0.01
        dominance_margin = 0.015
        max_gain = max(local_gain, entry_gain, sub_gain)

        classification = if Bool(get(bridge, "all_degenerate", true))
            "degenerate_subtrajectory_state"
        elseif max_gain <= threshold
            "invariant_subtrajectory_state"
        elseif frontier_range > max(entry_over_local, sub_over_entry, continuation_gain, threshold) + 0.005
            "frontier_allocation_suspected"
        elseif sub_over_entry > dominance_margin && (reorder > 0.0 || continuation_gain > threshold)
            "subtrajectory_value_dominant"
        elseif entry_over_local > dominance_margin && sub_over_entry <= threshold
            "entry_context_dominant"
        elseif abs(entry_over_local) <= threshold && abs(sub_over_entry) <= threshold && local_gain > threshold
            "local_surface_sufficient"
        else
            "ambiguous_subtrajectory_state"
        end

        return Dict{String,Any}(
            "classification" => classification,
            "local_gain_vs_baseline" => local_gain,
            "entry_context_gain_vs_baseline" => entry_gain,
            "subtrajectory_gain_vs_baseline" => sub_gain,
            "entry_context_gain_vs_local" => entry_over_local,
            "subtrajectory_gain_vs_entry" => sub_over_entry,
            "entry_reorder_fraction" => reorder,
            "best_basin_utility_range" => frontier_range,
            "max_entry_continuation_gain" => continuation_gain,
            "local_surface_correlation" => Float64(get(bridge, "local_surface_correlation", 0.0)),
            "entry_surface_correlation" => Float64(get(bridge, "entry_surface_correlation", 0.0)),
        )
    end

    function summarize_subtrajectory_runs(runs::Vector{Dict{String,Any}})
        classifications = Dict{String,Int}()
        local_gains = Float64[]
        entry_gains = Float64[]
        sub_gains = Float64[]
        entry_over_local = Float64[]
        sub_over_entry = Float64[]
        reorder_fractions = Float64[]
        local_corrs = Float64[]
        entry_corrs = Float64[]
        frontier_ranges = Float64[]
        max_continuations = Float64[]
        subtrajectory_count = 0
        entry_count = 0
        local_count = 0
        frontier_count = 0
        invariant = 0
        degenerate = 0
        ambiguous = 0

        for run in runs
            cls = String(run["classification"])
            classifications[cls] = get(classifications, cls, 0) + 1
            push!(local_gains, Float64(run["local_gain_vs_baseline"]))
            push!(entry_gains, Float64(run["entry_context_gain_vs_baseline"]))
            push!(sub_gains, Float64(run["subtrajectory_gain_vs_baseline"]))
            push!(entry_over_local, Float64(run["entry_context_gain_vs_local"]))
            push!(sub_over_entry, Float64(run["subtrajectory_gain_vs_entry"]))
            push!(reorder_fractions, Float64(run["entry_reorder_fraction"]))
            push!(local_corrs, Float64(run["local_surface_correlation"]))
            push!(entry_corrs, Float64(run["entry_surface_correlation"]))
            push!(frontier_ranges, Float64(run["best_basin_utility_range"]))
            push!(max_continuations, Float64(run["max_entry_continuation_gain"]))
            subtrajectory_count += cls == "subtrajectory_value_dominant" ? 1 : 0
            entry_count += cls == "entry_context_dominant" ? 1 : 0
            local_count += cls == "local_surface_sufficient" ? 1 : 0
            frontier_count += cls == "frontier_allocation_suspected" ? 1 : 0
            invariant += cls == "invariant_subtrajectory_state" ? 1 : 0
            degenerate += cls == "degenerate_subtrajectory_state" ? 1 : 0
            ambiguous += cls == "ambiguous_subtrajectory_state" ? 1 : 0
        end

        n = max(length(runs), 1)
        return Dict(
            "n_runs" => length(runs),
            "classifications" => classifications,
            "mean_local_gain_vs_baseline" => isempty(local_gains) ? 0.0 : mean(local_gains),
            "mean_entry_context_gain_vs_baseline" => isempty(entry_gains) ? 0.0 : mean(entry_gains),
            "mean_subtrajectory_gain_vs_baseline" => isempty(sub_gains) ? 0.0 : mean(sub_gains),
            "mean_entry_context_gain_vs_local" => isempty(entry_over_local) ? 0.0 : mean(entry_over_local),
            "mean_subtrajectory_gain_vs_entry" => isempty(sub_over_entry) ? 0.0 : mean(sub_over_entry),
            "mean_subtrajectory_gain_vs_local" => (isempty(entry_over_local) || isempty(sub_over_entry)) ? 0.0 : mean(entry_over_local) + mean(sub_over_entry),
            "mean_entry_reorder_fraction" => isempty(reorder_fractions) ? 0.0 : mean(reorder_fractions),
            "mean_local_surface_correlation" => isempty(local_corrs) ? 0.0 : mean(local_corrs),
            "mean_entry_surface_correlation" => isempty(entry_corrs) ? 0.0 : mean(entry_corrs),
            "mean_best_basin_utility_range" => isempty(frontier_ranges) ? 0.0 : mean(frontier_ranges),
            "mean_max_entry_continuation_gain" => isempty(max_continuations) ? 0.0 : mean(max_continuations),
            "subtrajectory_value_fraction" => subtrajectory_count / n,
            "entry_context_fraction" => entry_count / n,
            "local_surface_fraction" => local_count / n,
            "frontier_allocation_fraction" => frontier_count / n,
            "invariant_fraction" => invariant / n,
            "degenerate_fraction" => degenerate / n,
            "ambiguous_fraction" => ambiguous / n,
            "interpretable" => (ambiguous + degenerate) / n < 0.75,
        )
    end

    function run_bridge_task(task_name::String, repeats::Int)
        println("\nTask: $(task_name) [short-horizon subtrajectory bridge]")
        bridge_runs = Vector{Dict{String,Any}}()
        for repeat_idx in 1:repeats
            oracle_mgr, budget_oracle, budget_oracle_batch = build_budget_oracles(task_name, budget)
            frontier_buffer = MolecularFrontierBuffer(5000)
            vocab = SMILESVocabulary()
            target_smiles = get(TARGET_SMILES, task_name, nothing)
            seed_pool = bootstrap_seed_pool(task_name;
                user_seed_smiles=String[],
                target_smiles=target_smiles,
                target_seed=target_seed)
            seed_stats = _seed_memories!(seed_pool, budget_oracle, vocab;
                reward_fn_batch=budget_oracle_batch,
                frontier_buffer=frontier_buffer,
                augmentation_count=bootstrap_augmentation_count(task_name, target_smiles, target_seed; enable_augmentation=true),
                verbose=true)
            warmup_stats = bootstrap_frontier_warmup!(task_name, frontier_buffer, oracle_mgr, budget_oracle_batch, vocab;
                target_smiles=target_smiles,
                rounds=bootstrap_warmup_rounds,
                verbose=true)

            cfg = HierarchicalEditConfig(
                max_step_attempts=max_step_attempts,
                max_operator_candidates=max_operator_candidates,
                parent_candidate_limit=parent_candidate_limit,
                basin_candidate_limit=max_basin_contexts,
            )
            probe = probe_coupled_hierarchy_options(frontier_buffer, budget_oracle, vocab;
                reward_fn_batch=budget_oracle_batch,
                config=cfg,
                target_smiles=target_smiles,
                budget_remaining=budget_remaining(oracle_mgr),
                created_at_step=repeat_idx,
                task_name=task_name,
                max_parents=parent_candidate_limit,
                max_basins=max_basin_contexts,
                horizon=option_horizon)
            bridge = compare_option_value_surfaces(probe)
            diagnosis = classify_subtrajectory_bridge(bridge)
            run = Dict{String,Any}(
                "probe" => probe,
                "bridge" => bridge,
                "classification" => diagnosis["classification"],
                "local_gain_vs_baseline" => diagnosis["local_gain_vs_baseline"],
                "entry_context_gain_vs_baseline" => diagnosis["entry_context_gain_vs_baseline"],
                "subtrajectory_gain_vs_baseline" => diagnosis["subtrajectory_gain_vs_baseline"],
                "entry_context_gain_vs_local" => diagnosis["entry_context_gain_vs_local"],
                "subtrajectory_gain_vs_entry" => diagnosis["subtrajectory_gain_vs_entry"],
                "entry_reorder_fraction" => diagnosis["entry_reorder_fraction"],
                "best_basin_utility_range" => diagnosis["best_basin_utility_range"],
                "max_entry_continuation_gain" => diagnosis["max_entry_continuation_gain"],
                "local_surface_correlation" => diagnosis["local_surface_correlation"],
                "entry_surface_correlation" => diagnosis["entry_surface_correlation"],
                "seed_stats" => seed_stats,
                "warmup_stats" => warmup_stats,
                "calls_used" => oracle_mgr.calls_used,
            )
            push!(bridge_runs, run)
            println("  Bridge r$(repeat_idx): class=$(run["classification"]) | local=$(round(run["local_gain_vs_baseline"], digits=4)) | entry=$(round(run["entry_context_gain_vs_baseline"], digits=4)) | sub=$(round(run["subtrajectory_gain_vs_baseline"], digits=4)) | entryΔ=$(round(run["entry_context_gain_vs_local"], digits=4)) | subΔ=$(round(run["subtrajectory_gain_vs_entry"], digits=4)) | reorder=$(round(run["entry_reorder_fraction"], digits=2)) | calls=$(run["calls_used"]) ")
        end

        summary = summarize_subtrajectory_runs(bridge_runs)
        println("  Summary: classes=$(class_counts_repr(summary["classifications"])) | local=$(round(summary["mean_local_gain_vs_baseline"], digits=4)) | entry=$(round(summary["mean_entry_context_gain_vs_baseline"], digits=4)) | sub=$(round(summary["mean_subtrajectory_gain_vs_baseline"], digits=4)) | entryΔ=$(round(summary["mean_entry_context_gain_vs_local"], digits=4)) | subΔ=$(round(summary["mean_subtrajectory_gain_vs_entry"], digits=4)) | reorder=$(round(summary["mean_entry_reorder_fraction"], digits=2))")
        return Dict("bridge_runs" => bridge_runs, "summary" => summary)
    end

    celecoxib_task = "celecoxib_rediscovery"
    celecoxib_task in tasks || error("Batch 1J requires celecoxib_rediscovery in PMO_TASKS")

    all_results = Dict{String,Any}()
    decisions = Dict{String,Any}()

    all_results[celecoxib_task] = run_bridge_task(celecoxib_task, data_repeats)
    cele_summary = all_results[celecoxib_task]["summary"]

    if Bool(cele_summary["interpretable"])
        for task_name in ["drd2", "albuterol_similarity"]
            task_name in tasks || continue
            all_results[task_name] = run_bridge_task(task_name, 3)
        end
    end

    decisions["celecoxib_mean_local_gain_vs_baseline"] = cele_summary["mean_local_gain_vs_baseline"]
    decisions["celecoxib_mean_entry_context_gain_vs_baseline"] = cele_summary["mean_entry_context_gain_vs_baseline"]
    decisions["celecoxib_mean_subtrajectory_gain_vs_baseline"] = cele_summary["mean_subtrajectory_gain_vs_baseline"]
    decisions["celecoxib_mean_entry_context_gain_vs_local"] = cele_summary["mean_entry_context_gain_vs_local"]
    decisions["celecoxib_mean_subtrajectory_gain_vs_entry"] = cele_summary["mean_subtrajectory_gain_vs_entry"]
    decisions["celecoxib_mean_subtrajectory_gain_vs_local"] = cele_summary["mean_subtrajectory_gain_vs_local"]
    decisions["celecoxib_mean_entry_reorder_fraction"] = cele_summary["mean_entry_reorder_fraction"]
    decisions["celecoxib_subtrajectory_value_fraction"] = cele_summary["subtrajectory_value_fraction"]
    decisions["celecoxib_entry_context_fraction"] = cele_summary["entry_context_fraction"]
    decisions["celecoxib_frontier_allocation_fraction"] = cele_summary["frontier_allocation_fraction"]
    decisions["celecoxib_local_surface_fraction"] = cele_summary["local_surface_fraction"]
    decisions["celecoxib_interpretable"] = cele_summary["interpretable"]

    decisions["global_recommendation"] = if !Bool(cele_summary["interpretable"])
        "INCONCLUSIVE_REDESIGN_BRIDGE"
    elseif cele_summary["frontier_allocation_fraction"] >= 0.5
        "PIVOT_TO_FRONTIER_ALLOCATION_OBJECT"
    elseif (cele_summary["subtrajectory_value_fraction"] >= 0.5 || cele_summary["mean_subtrajectory_gain_vs_entry"] >= 0.015) && cele_summary["mean_subtrajectory_gain_vs_local"] >= 0.015
        "PROMOTE_SUBTRAJECTORY_VALUE_OBJECT"
    elseif cele_summary["entry_context_fraction"] >= 0.5
        "ENTRY_CONTEXT_STILL_MATTERS_RETAIN_HIERARCHY"
    elseif cele_summary["local_surface_fraction"] >= 0.5
        "STAY_LOCAL_RECALIBRATE_CONTROLLER"
    else
        "MIXED_OBJECT_GEOMETRY_DEEPER_PROBE_REQUIRED"
    end

    println("\nGlobal recommendation: $(decisions["global_recommendation"]) | celecoxib local=$(round(cele_summary["mean_local_gain_vs_baseline"], digits=4)) | entry=$(round(cele_summary["mean_entry_context_gain_vs_baseline"], digits=4)) | sub=$(round(cele_summary["mean_subtrajectory_gain_vs_baseline"], digits=4)) | entryΔ=$(round(cele_summary["mean_entry_context_gain_vs_local"], digits=4)) | subΔ=$(round(cele_summary["mean_subtrajectory_gain_vs_entry"], digits=4)) | local→sub=$(round(cele_summary["mean_subtrajectory_gain_vs_local"], digits=4))")
    return Dict("results_by_task" => all_results, "decisions" => decisions)
end

function run_c12_option_value_checks(tasks::Vector{String};
    budget::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=5,
    training_epochs::Int=30,
    parent_candidate_limit::Int=4,
    max_basin_contexts::Int=2,
    option_horizon::Int=3,
    celecoxib_repeats::Int=5,
    control_repeats::Int=3,
)
    println("\n" * "="^80)
    println("RUNNING BATCH 1K — FIRST LEARNED SHORT-HORIZON OPTION-VALUE / CONTINUATION-CONTROL STAGE")
    println("="^80)

    function collect_bridge_run(task_name::String, repeat_idx::Int)
        oracle_mgr, budget_oracle, budget_oracle_batch = build_budget_oracles(task_name, budget)
        frontier_buffer = MolecularFrontierBuffer(5000)
        vocab = SMILESVocabulary()
        target_smiles = get(TARGET_SMILES, task_name, nothing)
        seed_pool = bootstrap_seed_pool(task_name;
            user_seed_smiles=String[],
            target_smiles=target_smiles,
            target_seed=target_seed)
        seed_stats = _seed_memories!(seed_pool, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            frontier_buffer=frontier_buffer,
            augmentation_count=bootstrap_augmentation_count(task_name, target_smiles, target_seed; enable_augmentation=true),
            verbose=true)
        warmup_stats = bootstrap_frontier_warmup!(task_name, frontier_buffer, oracle_mgr, budget_oracle_batch, vocab;
            target_smiles=target_smiles,
            rounds=bootstrap_warmup_rounds,
            verbose=true)

        cfg = HierarchicalEditConfig(
            max_step_attempts=max_step_attempts,
            max_operator_candidates=max_operator_candidates,
            parent_candidate_limit=parent_candidate_limit,
            basin_candidate_limit=max_basin_contexts,
        )
        probe = probe_coupled_hierarchy_options(frontier_buffer, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            config=cfg,
            target_smiles=target_smiles,
            budget_remaining=budget_remaining(oracle_mgr),
            created_at_step=repeat_idx,
            task_name=task_name,
            max_parents=parent_candidate_limit,
            max_basins=max_basin_contexts,
            horizon=option_horizon)
        bridge = compare_option_value_surfaces(probe)
        return Dict(
            "probe" => probe,
            "bridge" => bridge,
            "seed_stats" => seed_stats,
            "warmup_stats" => warmup_stats,
            "calls_used" => oracle_mgr.calls_used,
        )
    end

    function offline_recipe_score(ev)
        score = get(ev, "mean_gain_vs_local_surface", 0.0)
        score += 0.25 * get(ev, "selection_hit_rate", 0.0)
        score += 0.10 * get(ev, "score_target_correlation", 0.0)
        score -= 0.10 * get(ev, "mean_regret_vs_best", 0.0)
        return score
    end

    celecoxib_task = "celecoxib_rediscovery"
    celecoxib_task in tasks || error("Batch 1K requires celecoxib_rediscovery in PMO_TASKS")

    println("\nCollecting celecoxib option-value training data ...")
    train_bridge_runs = [collect_bridge_run(celecoxib_task, idx) for idx in 1:data_repeats]

    recipe_results = Dict{String,Any}()
    for feature_mode in [:basic, :augmented]
        recipe_name = String(feature_mode)
        dataset = extract_option_value_dataset(train_bridge_runs; task_name=celecoxib_task, feature_mode=feature_mode)
        println("  Offline recipe $(recipe_name): records=$(length(dataset)) snapshots=$(option_value_dataset_stats(dataset)["n_snapshots"])")
        trained = train_option_value_model(dataset;
            rng=MersenneTwister(0),
            config=OptionValueTrainingConfig(n_epochs=training_epochs, min_records=12, feature_mode=feature_mode))
        val_eval = trained["best_val_eval"]
        final_eval = trained["final_eval"]
        recipe_results[recipe_name] = Dict(
            "feature_mode" => feature_mode,
            "dataset_stats" => trained["dataset_stats"],
            "best_val_eval" => val_eval,
            "final_eval" => final_eval,
            "offline_score" => offline_recipe_score(val_eval),
            "model" => trained["model"],
        )
        println("    score=$(round(recipe_results[recipe_name]["offline_score"], digits=4)) | val_gain_vs_local=$(round(get(val_eval, "mean_gain_vs_local_surface", 0.0), digits=4)) | val_hit=$(round(get(val_eval, "selection_hit_rate", 0.0), digits=4))")
    end

    promoted_recipe_names = sort(collect(keys(recipe_results)); by=name -> recipe_results[name]["offline_score"], rev=true)
    promoted_recipe_names = promoted_recipe_names[1:min(2, length(promoted_recipe_names))]

    function evaluate_recipe_online(task_name::String, repeats::Int, recipe_name::String)
        recipe = recipe_results[recipe_name]
        eval_bridge_runs = [collect_bridge_run(task_name, idx) for idx in 1:repeats]
        eval_dataset = extract_option_value_dataset(eval_bridge_runs; task_name=task_name, feature_mode=recipe["feature_mode"])
        online_eval = evaluate_option_value_model(recipe["model"], eval_dataset)
        return Dict(
            "bridge_runs" => eval_bridge_runs,
            "dataset_stats" => option_value_dataset_stats(eval_dataset),
            "online_eval" => online_eval,
        )
    end

    all_results = Dict{String,Any}()
    celecoxib_online = Dict{String,Any}()
    println("\nRunning celecoxib-first online gate ...")
    for recipe_name in promoted_recipe_names
        result = evaluate_recipe_online(celecoxib_task, celecoxib_repeats, recipe_name)
        celecoxib_online[recipe_name] = result
        online_eval = result["online_eval"]
        println("  $(recipe_name): selected=$(round(get(online_eval, "mean_selected_option_value", 0.0), digits=4)) | Δvs local=$(round(get(online_eval, "mean_gain_vs_local_surface", 0.0), digits=4)) | Δvs entry=$(round(get(online_eval, "mean_gain_vs_entry_context_surface", 0.0), digits=4)) | hit=$(round(get(online_eval, "selection_hit_rate", 0.0), digits=4))")
    end

    best_recipe_name = promoted_recipe_names[argmax([get(celecoxib_online[name]["online_eval"], "mean_gain_vs_local_surface", -Inf) for name in promoted_recipe_names])]
    best_online = celecoxib_online[best_recipe_name]["online_eval"]

    all_results[celecoxib_task] = Dict(
        "train_bridge_runs" => train_bridge_runs,
        "recipe_results" => recipe_results,
        "promoted_recipe_names" => promoted_recipe_names,
        "online_results" => celecoxib_online,
        "best_recipe_name" => best_recipe_name,
        "best_online_eval" => best_online,
    )

    if get(best_online, "mean_gain_vs_local_surface", 0.0) > -0.05
        for task_name in ["drd2", "albuterol_similarity"]
            task_name in tasks || continue
            println("\nRunning control task $(task_name) with best recipe $(best_recipe_name) ...")
            control_result = evaluate_recipe_online(task_name, control_repeats, best_recipe_name)
            all_results[task_name] = control_result
            online_eval = control_result["online_eval"]
            println("  $(task_name): selected=$(round(get(online_eval, "mean_selected_option_value", 0.0), digits=4)) | Δvs local=$(round(get(online_eval, "mean_gain_vs_local_surface", 0.0), digits=4)) | Δvs entry=$(round(get(online_eval, "mean_gain_vs_entry_context_surface", 0.0), digits=4)) | hit=$(round(get(online_eval, "selection_hit_rate", 0.0), digits=4))")
        end
    end

    decisions = Dict{String,Any}(
        "best_recipe_name" => best_recipe_name,
        "celecoxib_mean_gain_vs_local_surface" => get(best_online, "mean_gain_vs_local_surface", 0.0),
        "celecoxib_mean_gain_vs_entry_context_surface" => get(best_online, "mean_gain_vs_entry_context_surface", 0.0),
        "celecoxib_mean_regret_vs_best" => get(best_online, "mean_regret_vs_best", 0.0),
        "celecoxib_selection_hit_rate" => get(best_online, "selection_hit_rate", 0.0),
    )

    decisions["global_recommendation"] = if get(best_online, "mean_gain_vs_local_surface", 0.0) >= 0.015 && get(best_online, "selection_hit_rate", 0.0) >= 0.25
        "PROMOTE_LEARNED_OPTION_VALUE_BRIDGE"
    elseif get(best_online, "mean_gain_vs_local_surface", 0.0) > 0.0
        "VALUE_YES_POLICY_CONSERVATIVE_NEXT"
    elseif get(best_online, "mean_gain_vs_entry_context_surface", 0.0) > 0.0
        "VALUE_BEATS_ENTRY_BUT_NOT_LOCAL"
    else
        "HOLD_OPTION_VALUE_REVISE_BRIDGE"
    end

    println("\nGlobal recommendation: $(decisions["global_recommendation"]) | best=$(best_recipe_name) | celecoxib Δvs local=$(round(decisions["celecoxib_mean_gain_vs_local_surface"], digits=4)) | Δvs entry=$(round(decisions["celecoxib_mean_gain_vs_entry_context_surface"], digits=4)) | hit=$(round(decisions["celecoxib_selection_hit_rate"], digits=4))")
    return Dict("results_by_task" => all_results, "decisions" => decisions)
end


function run_c13_option_value_refinement_checks(tasks::Vector{String};
    budget::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=5,
    training_epochs::Int=30,
    parent_candidate_limit::Int=4,
    max_basin_contexts::Int=2,
    option_horizon::Int=3,
    celecoxib_repeats::Int=5,
    control_repeats::Int=3,
)
    println("\n" * "="^80)
    println("RUNNING BATCH 1L — OPTION-VALUE BRIDGE REFINEMENT")
    println("="^80)

    function collect_bridge_run(task_name::String, repeat_idx::Int)
        oracle_mgr, budget_oracle, budget_oracle_batch = build_budget_oracles(task_name, budget)
        frontier_buffer = MolecularFrontierBuffer(5000)
        vocab = SMILESVocabulary()
        target_smiles = get(TARGET_SMILES, task_name, nothing)
        seed_pool = bootstrap_seed_pool(task_name;
            user_seed_smiles=String[],
            target_smiles=target_smiles,
            target_seed=target_seed)
        seed_stats = _seed_memories!(seed_pool, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            frontier_buffer=frontier_buffer,
            augmentation_count=bootstrap_augmentation_count(task_name, target_smiles, target_seed; enable_augmentation=true),
            verbose=true)
        warmup_stats = bootstrap_frontier_warmup!(task_name, frontier_buffer, oracle_mgr, budget_oracle_batch, vocab;
            target_smiles=target_smiles,
            rounds=bootstrap_warmup_rounds,
            verbose=true)

        cfg = HierarchicalEditConfig(
            max_step_attempts=max_step_attempts,
            max_operator_candidates=max_operator_candidates,
            parent_candidate_limit=parent_candidate_limit,
            basin_candidate_limit=max_basin_contexts,
        )
        probe = probe_coupled_hierarchy_options(frontier_buffer, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            config=cfg,
            target_smiles=target_smiles,
            budget_remaining=budget_remaining(oracle_mgr),
            created_at_step=repeat_idx,
            task_name=task_name,
            max_parents=parent_candidate_limit,
            max_basins=max_basin_contexts,
            horizon=option_horizon)
        bridge = compare_option_value_surfaces(probe)
        return Dict(
            "probe" => probe,
            "bridge" => bridge,
            "seed_stats" => seed_stats,
            "warmup_stats" => warmup_stats,
            "calls_used" => oracle_mgr.calls_used,
        )
    end

    function offline_recipe_score(ev)
        score = 0.55 * get(ev, "mean_gain_vs_local_candidate", 0.0)
        score += 0.20 * get(ev, "selection_hit_rate", 0.0)
        score += 0.10 * get(ev, "score_target_correlation", 0.0)
        score += 0.10 * get(ev, "continuation_sensitive_gain_vs_local_candidate", 0.0)
        score += 0.05 * get(ev, "mean_gain_vs_local_surface", 0.0)
        score -= 0.10 * get(ev, "mean_regret_vs_best", 0.0)
        return score
    end

    celecoxib_task = "celecoxib_rediscovery"
    celecoxib_task in tasks || error("Batch 1L requires celecoxib_rediscovery in PMO_TASKS")

    println("\nCollecting celecoxib option-value refinement data ...")
    train_bridge_runs = [collect_bridge_run(celecoxib_task, idx) for idx in 1:data_repeats]

    recipe_results = Dict{String,Any}()
    base_datasets = Dict{Symbol,OptionValueDataset}()
    for feature_mode in [:basic, :augmented]
        dataset = extract_option_value_dataset(train_bridge_runs; task_name=celecoxib_task, feature_mode=feature_mode)
        base_datasets[feature_mode] = dataset
        stats = option_value_dataset_stats(dataset)
        stats_snapshots = stats["n_snapshots"]
        stats_feature_dim = stats["feature_dim"]
        stats_continuation = round(get(stats, "continuation_sensitive_fraction", 0.0), digits=4)
        println("  Feature mode $(feature_mode): records=$(length(dataset)) snapshots=$(stats_snapshots) feature_dim=$(stats_feature_dim) continuation=$(stats_continuation)")
        for objective_mode in [:regression, :pairwise, :hybrid]
            trained = train_option_value_model(dataset;
                rng=MersenneTwister(0),
                config=OptionValueTrainingConfig(
                    n_epochs=training_epochs,
                    min_records=12,
                    feature_mode=feature_mode,
                    objective_mode=objective_mode,
                ))
            for selection_rule in [:argmax, :local_anchored, :ambiguity_gated]
                val_eval = evaluate_option_value_model(trained["model"], trained["val_dataset"];
                    selection_rule=selection_rule,
                    override_margin=0.05,
                    ambiguity_threshold=0.05)
                final_eval = evaluate_option_value_model(trained["model"], dataset;
                    selection_rule=selection_rule,
                    override_margin=0.05,
                    ambiguity_threshold=0.05)
                recipe_name = string(feature_mode, "__", objective_mode, "__", selection_rule)
                recipe_results[recipe_name] = Dict(
                    "feature_mode" => feature_mode,
                    "objective_mode" => objective_mode,
                    "selection_rule" => selection_rule,
                    "override_margin" => 0.05,
                    "ambiguity_threshold" => 0.05,
                    "dataset_stats" => trained["dataset_stats"],
                    "best_val_eval" => val_eval,
                    "final_eval" => final_eval,
                    "offline_score" => offline_recipe_score(val_eval),
                    "model" => trained["model"],
                )
                recipe_score_repr = round(recipe_results[recipe_name]["offline_score"], digits=4)
                val_gain_local_candidate = round(get(val_eval, "mean_gain_vs_local_candidate", 0.0), digits=4)
                val_gain_local = round(get(val_eval, "mean_gain_vs_local_surface", 0.0), digits=4)
                val_hit = round(get(val_eval, "selection_hit_rate", 0.0), digits=4)
                val_reorder = round(get(val_eval, "reorder_fraction_vs_local", 0.0), digits=4)
                val_override = round(get(val_eval, "override_rate", 0.0), digits=4)
                println("    $(recipe_name): score=$(recipe_score_repr) | valΔlocalCand=$(val_gain_local_candidate) | valΔlocal=$(val_gain_local) | hit=$(val_hit) | reorder=$(val_reorder) | override=$(val_override)")
            end
        end
    end

    promoted_recipe_names = sort(collect(keys(recipe_results)); by=name -> recipe_results[name]["offline_score"], rev=true)
    promoted_recipe_names = promoted_recipe_names[1:min(3, length(promoted_recipe_names))]

    function evaluate_recipe_online(task_name::String, repeats::Int, recipe_name::String)
        recipe = recipe_results[recipe_name]
        eval_bridge_runs = [collect_bridge_run(task_name, idx) for idx in 1:repeats]
        eval_dataset = extract_option_value_dataset(eval_bridge_runs; task_name=task_name, feature_mode=recipe["feature_mode"])
        online_eval = evaluate_option_value_model(recipe["model"], eval_dataset;
            selection_rule=recipe["selection_rule"],
            override_margin=recipe["override_margin"],
            ambiguity_threshold=recipe["ambiguity_threshold"])
        return Dict(
            "bridge_runs" => eval_bridge_runs,
            "dataset_stats" => option_value_dataset_stats(eval_dataset),
            "online_eval" => online_eval,
        )
    end

    all_results = Dict{String,Any}()
    celecoxib_online = Dict{String,Any}()
    println("\nRunning celecoxib-first Batch 1L online gate ...")
    for recipe_name in promoted_recipe_names
        result = evaluate_recipe_online(celecoxib_task, celecoxib_repeats, recipe_name)
        celecoxib_online[recipe_name] = result
        online_eval = result["online_eval"]
        online_selected = round(get(online_eval, "mean_selected_option_value", 0.0), digits=4)
        online_gain_local_candidate = round(get(online_eval, "mean_gain_vs_local_candidate", 0.0), digits=4)
        online_gain_local = round(get(online_eval, "mean_gain_vs_local_surface", 0.0), digits=4)
        online_gain_entry = round(get(online_eval, "mean_gain_vs_entry_context_surface", 0.0), digits=4)
        online_hit = round(get(online_eval, "selection_hit_rate", 0.0), digits=4)
        online_reorder = round(get(online_eval, "reorder_fraction_vs_local", 0.0), digits=4)
        online_override = round(get(online_eval, "override_rate", 0.0), digits=4)
        println("  $(recipe_name): selected=$(online_selected) | Δvs localCand=$(online_gain_local_candidate) | Δvs local=$(online_gain_local) | Δvs entry=$(online_gain_entry) | hit=$(online_hit) | reorder=$(online_reorder) | override=$(online_override)")
    end

    best_recipe_name = promoted_recipe_names[argmax([get(celecoxib_online[name]["online_eval"], "mean_gain_vs_local_surface", -Inf) for name in promoted_recipe_names])]
    best_online = celecoxib_online[best_recipe_name]["online_eval"]

    all_results[celecoxib_task] = Dict(
        "train_bridge_runs" => train_bridge_runs,
        "recipe_results" => recipe_results,
        "promoted_recipe_names" => promoted_recipe_names,
        "online_results" => celecoxib_online,
        "best_recipe_name" => best_recipe_name,
        "best_online_eval" => best_online,
    )

    if get(best_online, "mean_gain_vs_local_surface", 0.0) >= 0.0
        for task_name in ["drd2", "albuterol_similarity"]
            task_name in tasks || continue
            println("\nRunning control task $(task_name) with best recipe $(best_recipe_name) ...")
            control_result = evaluate_recipe_online(task_name, control_repeats, best_recipe_name)
            all_results[task_name] = control_result
            online_eval = control_result["online_eval"]
            control_selected = round(get(online_eval, "mean_selected_option_value", 0.0), digits=4)
            control_gain_local_candidate = round(get(online_eval, "mean_gain_vs_local_candidate", 0.0), digits=4)
            control_gain_local = round(get(online_eval, "mean_gain_vs_local_surface", 0.0), digits=4)
            control_gain_entry = round(get(online_eval, "mean_gain_vs_entry_context_surface", 0.0), digits=4)
            control_hit = round(get(online_eval, "selection_hit_rate", 0.0), digits=4)
            control_override = round(get(online_eval, "override_rate", 0.0), digits=4)
            println("  $(task_name): selected=$(control_selected) | Δvs localCand=$(control_gain_local_candidate) | Δvs local=$(control_gain_local) | Δvs entry=$(control_gain_entry) | hit=$(control_hit) | override=$(control_override)")
        end
    end

    decisions = Dict{String,Any}(
        "best_recipe_name" => best_recipe_name,
        "celecoxib_mean_gain_vs_local_candidate" => get(best_online, "mean_gain_vs_local_candidate", 0.0),
        "celecoxib_mean_gain_vs_local_surface" => get(best_online, "mean_gain_vs_local_surface", 0.0),
        "celecoxib_mean_gain_vs_entry_context_surface" => get(best_online, "mean_gain_vs_entry_context_surface", 0.0),
        "celecoxib_mean_regret_vs_best" => get(best_online, "mean_regret_vs_best", 0.0),
        "celecoxib_selection_hit_rate" => get(best_online, "selection_hit_rate", 0.0),
        "celecoxib_reorder_fraction_vs_local" => get(best_online, "reorder_fraction_vs_local", 0.0),
        "celecoxib_override_rate" => get(best_online, "override_rate", 0.0),
        "best_selection_rule" => String(recipe_results[best_recipe_name]["selection_rule"]),
        "best_objective_mode" => String(recipe_results[best_recipe_name]["objective_mode"]),
    )

    decisions["global_recommendation"] = if get(best_online, "mean_gain_vs_local_surface", 0.0) >= 0.015 && get(best_online, "selection_hit_rate", 0.0) >= 0.25
        "PROMOTE_REFINED_OPTION_VALUE_BRIDGE"
    elseif get(best_online, "mean_gain_vs_local_surface", 0.0) >= 0.0 && recipe_results[best_recipe_name]["selection_rule"] != :argmax && get(best_online, "override_rate", 0.0) > 0.0
        "SELECTION_RULE_IS_REAL_MISSING_PIECE"
    elseif get(best_online, "mean_gain_vs_local_surface", 0.0) >= 0.0
        "REFINED_BRIDGE_NONNEGATIVE_KEEP_CAUTIOUS"
    elseif get(best_online, "mean_gain_vs_entry_context_surface", 0.0) > 0.0
        "OBJECT_RETAINED_BRIDGE_FORM_STILL_INADEQUATE"
    else
        "DEEPER_OPTION_CONTROL_MISMATCH_REMAINS"
    end

    global_rec = decisions["global_recommendation"]
    cele_gain_local_candidate = round(decisions["celecoxib_mean_gain_vs_local_candidate"], digits=4)
    cele_gain_local = round(decisions["celecoxib_mean_gain_vs_local_surface"], digits=4)
    cele_gain_entry = round(decisions["celecoxib_mean_gain_vs_entry_context_surface"], digits=4)
    cele_hit = round(decisions["celecoxib_selection_hit_rate"], digits=4)
    cele_reorder = round(decisions["celecoxib_reorder_fraction_vs_local"], digits=4)
    cele_override = round(decisions["celecoxib_override_rate"], digits=4)
    println("\nGlobal recommendation: $(global_rec) | best=$(best_recipe_name) | ΔlocalCand=$(cele_gain_local_candidate) | Δlocal=$(cele_gain_local) | Δentry=$(cele_gain_entry) | hit=$(cele_hit) | reorder=$(cele_reorder) | override=$(cele_override)")
    return Dict("results_by_task" => all_results, "decisions" => decisions)
end

function run_c14_calibrated_ordinal_option_checks(tasks::Vector{String};
    budget::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=5,
    training_epochs::Int=30,
    calibration_epochs::Int=50,
    parent_candidate_limit::Int=4,
    max_basin_contexts::Int=2,
    option_horizon::Int=3,
    celecoxib_repeats::Int=5,
    control_repeats::Int=3,
)
    println("\n" * "="^80)
    println("RUNNING BATCH 1M — CALIBRATED ORDINAL OPTION-SELECTION BRIDGE")
    println("="^80)

    function collect_bridge_run(task_name::String, repeat_idx::Int)
        oracle_mgr, budget_oracle, budget_oracle_batch = build_budget_oracles(task_name, budget)
        frontier_buffer = MolecularFrontierBuffer(5000)
        vocab = SMILESVocabulary()
        target_smiles = get(TARGET_SMILES, task_name, nothing)
        seed_pool = bootstrap_seed_pool(task_name;
            user_seed_smiles=String[],
            target_smiles=target_smiles,
            target_seed=target_seed)
        seed_stats = _seed_memories!(seed_pool, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            frontier_buffer=frontier_buffer,
            augmentation_count=bootstrap_augmentation_count(task_name, target_smiles, target_seed; enable_augmentation=true),
            verbose=true)
        warmup_stats = bootstrap_frontier_warmup!(task_name, frontier_buffer, oracle_mgr, budget_oracle_batch, vocab;
            target_smiles=target_smiles,
            rounds=bootstrap_warmup_rounds,
            verbose=true)

        cfg = HierarchicalEditConfig(
            max_step_attempts=max_step_attempts,
            max_operator_candidates=max_operator_candidates,
            parent_candidate_limit=parent_candidate_limit,
            basin_candidate_limit=max_basin_contexts,
        )
        probe = probe_coupled_hierarchy_options(frontier_buffer, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            config=cfg,
            target_smiles=target_smiles,
            budget_remaining=budget_remaining(oracle_mgr),
            created_at_step=repeat_idx,
            task_name=task_name,
            max_parents=parent_candidate_limit,
            max_basins=max_basin_contexts,
            horizon=option_horizon)
        bridge = compare_option_value_surfaces(probe)
        return Dict(
            "probe" => probe,
            "bridge" => bridge,
            "seed_stats" => seed_stats,
            "warmup_stats" => warmup_stats,
            "calls_used" => oracle_mgr.calls_used,
        )
    end

    function offline_recipe_score(ev)
        score = 0.35 * get(ev, "mean_gain_vs_local_surface", 0.0)
        score += 0.20 * get(ev, "mean_gain_vs_entry_local_candidate", 0.0)
        score += 0.15 * get(ev, "selection_hit_rate", 0.0)
        score += 0.15 * get(ev, "override_precision", 0.0)
        score += 0.10 * get(ev, "override_recall", 0.0)
        score -= 0.10 * get(ev, "mean_regret_vs_best", 0.0)
        override_rate = get(ev, "override_rate", 0.0)
        if override_rate < 0.05
            score -= 0.25 * (0.05 - override_rate)
        elseif override_rate > 0.85
            score -= 0.40 * (override_rate - 0.85)
        end
        return score
    end

    celecoxib_task = "celecoxib_rediscovery"
    celecoxib_task in tasks || error("Batch 1M requires celecoxib_rediscovery in PMO_TASKS")

    println("\nCollecting celecoxib calibrated-ordinal data ...")
    train_bridge_runs = [collect_bridge_run(celecoxib_task, idx) for idx in 1:data_repeats]

    recipe_results = Dict{String,Any}()
    for feature_mode in [:augmented]
        dataset = extract_option_value_dataset(train_bridge_runs; task_name=celecoxib_task, feature_mode=feature_mode)
        stats = option_value_dataset_stats(dataset)
        stats_snapshots = stats["n_snapshots"]
        stats_feature_dim = stats["feature_dim"]
        stats_override = round(get(stats, "override_positive_fraction", 0.0), digits=4)
        println("  Feature mode $(feature_mode): records=$(length(dataset)) snapshots=$(stats_snapshots) feature_dim=$(stats_feature_dim) override+=$(stats_override)")

        for objective_mode in [:pairwise, :hybrid]
            ranking_result = train_option_value_model(dataset;
                rng=MersenneTwister(0),
                config=OptionValueTrainingConfig(
                    n_epochs=training_epochs,
                    min_records=12,
                    feature_mode=feature_mode,
                    objective_mode=objective_mode,
                ))
            argmax_name = string(feature_mode, "__", objective_mode, "__argmax_diagnostic")
            argmax_val = evaluate_option_value_model(ranking_result["model"], ranking_result["val_dataset"];
                selection_rule=:argmax,
                override_margin=0.05,
                ambiguity_threshold=0.05)
            argmax_final = evaluate_option_value_model(ranking_result["model"], dataset;
                selection_rule=:argmax,
                override_margin=0.05,
                ambiguity_threshold=0.05)
            recipe_results[argmax_name] = Dict(
                "feature_mode" => feature_mode,
                "objective_mode" => objective_mode,
                "selection_rule" => :argmax,
                "recipe_kind" => "ordinal_argmax_diagnostic",
                "dataset_stats" => ranking_result["dataset_stats"],
                "best_val_eval" => argmax_val,
                "final_eval" => argmax_final,
                "offline_score" => offline_recipe_score(argmax_val),
                "model_like" => ranking_result["model"],
            )
            argmax_score = round(recipe_results[argmax_name]["offline_score"], digits=4)
            argmax_gain_local = round(get(argmax_val, "mean_gain_vs_local_surface", 0.0), digits=4)
            argmax_gain_entry_local = round(get(argmax_val, "mean_gain_vs_entry_local_candidate", 0.0), digits=4)
            argmax_override = round(get(argmax_val, "override_rate", 0.0), digits=4)
            println("    $(argmax_name): score=$(argmax_score) | valΔlocal=$(argmax_gain_local) | valΔentryLocal=$(argmax_gain_entry_local) | override=$(argmax_override)")

            for selection_rule in [:confidence_threshold, :confidence_band, :anchored_confidence]
                calibrated = train_calibrated_ordinal_option_policy(dataset;
                    rng=MersenneTwister(0),
                    ranking_config=OptionValueTrainingConfig(
                        n_epochs=training_epochs,
                        min_records=12,
                        feature_mode=feature_mode,
                        objective_mode=objective_mode,
                    ),
                    calibration_config=OptionCalibrationConfig(
                        n_epochs=calibration_epochs,
                        override_gain_threshold=0.02,
                    ),
                    selection_rule=selection_rule)
                recipe_name = string(feature_mode, "__", objective_mode, "__", selection_rule)
                val_eval = calibrated["best_val_eval"]
                final_eval = calibrated["final_eval"]
                recipe_results[recipe_name] = Dict(
                    "feature_mode" => feature_mode,
                    "objective_mode" => objective_mode,
                    "selection_rule" => selection_rule,
                    "recipe_kind" => "calibrated_ordinal",
                    "dataset_stats" => calibrated["dataset_stats"],
                    "confidence_result" => calibrated["confidence_result"],
                    "best_val_eval" => val_eval,
                    "final_eval" => final_eval,
                    "offline_score" => offline_recipe_score(val_eval),
                    "model_like" => calibrated["policy"],
                )
                recipe_score = round(recipe_results[recipe_name]["offline_score"], digits=4)
                val_gain_local = round(get(val_eval, "mean_gain_vs_local_surface", 0.0), digits=4)
                val_gain_entry_local = round(get(val_eval, "mean_gain_vs_entry_local_candidate", 0.0), digits=4)
                val_hit = round(get(val_eval, "selection_hit_rate", 0.0), digits=4)
                val_override = round(get(val_eval, "override_rate", 0.0), digits=4)
                val_precision = round(get(val_eval, "override_precision", 0.0), digits=4)
                println("    $(recipe_name): score=$(recipe_score) | valΔlocal=$(val_gain_local) | valΔentryLocal=$(val_gain_entry_local) | hit=$(val_hit) | override=$(val_override) | precision=$(val_precision)")
            end
        end
    end

    promoted_recipe_names = sort(collect(keys(recipe_results)); by=name -> recipe_results[name]["offline_score"], rev=true)
    promoted_recipe_names = promoted_recipe_names[1:min(4, length(promoted_recipe_names))]

    function evaluate_recipe_online(task_name::String, repeats::Int, recipe_name::String)
        recipe = recipe_results[recipe_name]
        eval_bridge_runs = [collect_bridge_run(task_name, idx) for idx in 1:repeats]
        eval_dataset = extract_option_value_dataset(eval_bridge_runs; task_name=task_name, feature_mode=recipe["feature_mode"])
        model_like = recipe["model_like"]
        online_eval = if model_like isa LearnedOptionValueModel
            evaluate_option_value_model(model_like, eval_dataset;
                selection_rule=recipe["selection_rule"],
                override_margin=0.05,
                ambiguity_threshold=0.05)
        else
            evaluate_option_value_model(model_like, eval_dataset)
        end
        return Dict(
            "bridge_runs" => eval_bridge_runs,
            "dataset_stats" => option_value_dataset_stats(eval_dataset),
            "online_eval" => online_eval,
        )
    end

    all_results = Dict{String,Any}()
    celecoxib_online = Dict{String,Any}()
    println("\nRunning celecoxib-first Batch 1M online gate ...")
    for recipe_name in promoted_recipe_names
        result = evaluate_recipe_online(celecoxib_task, celecoxib_repeats, recipe_name)
        celecoxib_online[recipe_name] = result
        online_eval = result["online_eval"]
        online_selected = round(get(online_eval, "mean_selected_option_value", 0.0), digits=4)
        online_gain_entry_local = round(get(online_eval, "mean_gain_vs_entry_local_candidate", 0.0), digits=4)
        online_gain_local = round(get(online_eval, "mean_gain_vs_local_surface", 0.0), digits=4)
        online_hit = round(get(online_eval, "selection_hit_rate", 0.0), digits=4)
        online_override = round(get(online_eval, "override_rate", 0.0), digits=4)
        online_precision = round(get(online_eval, "override_precision", 0.0), digits=4)
        online_recall = round(get(online_eval, "override_recall", 0.0), digits=4)
        online_conf = round(get(online_eval, "mean_challenger_confidence", 0.0), digits=4)
        println("  $(recipe_name): selected=$(online_selected) | Δvs entryLocal=$(online_gain_entry_local) | Δvs local=$(online_gain_local) | hit=$(online_hit) | override=$(online_override) | precision=$(online_precision) | recall=$(online_recall) | conf=$(online_conf)")
    end

    best_recipe_name = promoted_recipe_names[argmax([get(celecoxib_online[name]["online_eval"], "mean_gain_vs_local_surface", -Inf) for name in promoted_recipe_names])]
    best_online = celecoxib_online[best_recipe_name]["online_eval"]
    best_recipe = recipe_results[best_recipe_name]

    all_results[celecoxib_task] = Dict(
        "train_bridge_runs" => train_bridge_runs,
        "recipe_results" => recipe_results,
        "promoted_recipe_names" => promoted_recipe_names,
        "online_results" => celecoxib_online,
        "best_recipe_name" => best_recipe_name,
        "best_online_eval" => best_online,
    )

    interpretable_celecoxib = get(best_online, "mean_gain_vs_local_surface", 0.0) >= 0.0 && 0.05 <= get(best_online, "override_rate", 0.0) <= 0.95
    if interpretable_celecoxib
        for task_name in ["drd2", "albuterol_similarity"]
            task_name in tasks || continue
            println("\nRunning control task $(task_name) with best recipe $(best_recipe_name) ...")
            control_result = evaluate_recipe_online(task_name, control_repeats, best_recipe_name)
            all_results[task_name] = control_result
            online_eval = control_result["online_eval"]
            control_selected = round(get(online_eval, "mean_selected_option_value", 0.0), digits=4)
            control_gain_entry_local = round(get(online_eval, "mean_gain_vs_entry_local_candidate", 0.0), digits=4)
            control_gain_local = round(get(online_eval, "mean_gain_vs_local_surface", 0.0), digits=4)
            control_hit = round(get(online_eval, "selection_hit_rate", 0.0), digits=4)
            control_override = round(get(online_eval, "override_rate", 0.0), digits=4)
            control_precision = round(get(online_eval, "override_precision", 0.0), digits=4)
            println("  $(task_name): selected=$(control_selected) | Δvs entryLocal=$(control_gain_entry_local) | Δvs local=$(control_gain_local) | hit=$(control_hit) | override=$(control_override) | precision=$(control_precision)")
        end
    end

    decisions = Dict{String,Any}(
        "best_recipe_name" => best_recipe_name,
        "best_recipe_kind" => best_recipe["recipe_kind"],
        "celecoxib_mean_gain_vs_entry_local_candidate" => get(best_online, "mean_gain_vs_entry_local_candidate", 0.0),
        "celecoxib_mean_gain_vs_local_surface" => get(best_online, "mean_gain_vs_local_surface", 0.0),
        "celecoxib_mean_regret_vs_best" => get(best_online, "mean_regret_vs_best", 0.0),
        "celecoxib_selection_hit_rate" => get(best_online, "selection_hit_rate", 0.0),
        "celecoxib_override_rate" => get(best_online, "override_rate", 0.0),
        "celecoxib_override_precision" => get(best_online, "override_precision", 0.0),
        "celecoxib_override_recall" => get(best_online, "override_recall", 0.0),
        "celecoxib_mean_challenger_confidence" => get(best_online, "mean_challenger_confidence", 0.0),
        "best_selection_rule" => String(best_recipe["selection_rule"]),
        "best_objective_mode" => String(best_recipe["objective_mode"]),
    )

    decisions["global_recommendation"] = if get(best_online, "mean_gain_vs_local_surface", 0.0) >= 0.01 && 0.05 <= get(best_online, "override_rate", 0.0) <= 0.80 && get(best_online, "selection_hit_rate", 0.0) > 0.0 && get(best_online, "override_precision", 0.0) >= 0.5 && best_recipe["recipe_kind"] == "calibrated_ordinal"
        "PROMOTE_CALIBRATED_ORDINAL_BRIDGE"
    elseif get(best_online, "mean_gain_vs_local_surface", 0.0) >= 0.0 && 0.05 <= get(best_online, "override_rate", 0.0) <= 0.90 && best_recipe["recipe_kind"] == "calibrated_ordinal"
        "CALIBRATED_BRIDGE_NONNEGATIVE_KEEP_CAUTIOUS"
    elseif get(best_online, "mean_gain_vs_local_surface", 0.0) >= 0.0
        "ORDINAL_SIGNAL_REAL_CONTROL_STILL_NOT_TRUSTWORTHY"
    elseif get(best_online, "mean_gain_vs_entry_local_candidate", 0.0) > 0.0
        "KEEP_OPTION_OBJECT_MOVE_TO_FRONTIER_ALLOCATION_IF_REPEATED"
    else
        "CALIBRATED_BRIDGE_FAILED_MOVE_UPWARD"
    end

    global_rec = decisions["global_recommendation"]
    cele_gain_entry_local = round(decisions["celecoxib_mean_gain_vs_entry_local_candidate"], digits=4)
    cele_gain_local = round(decisions["celecoxib_mean_gain_vs_local_surface"], digits=4)
    cele_hit = round(decisions["celecoxib_selection_hit_rate"], digits=4)
    cele_override = round(decisions["celecoxib_override_rate"], digits=4)
    cele_precision = round(decisions["celecoxib_override_precision"], digits=4)
    cele_recall = round(decisions["celecoxib_override_recall"], digits=4)
    println("\nGlobal recommendation: $(global_rec) | best=$(best_recipe_name) | ΔentryLocal=$(cele_gain_entry_local) | Δlocal=$(cele_gain_local) | hit=$(cele_hit) | override=$(cele_override) | precision=$(cele_precision) | recall=$(cele_recall)")
    return Dict("results_by_task" => all_results, "decisions" => decisions)
end

function run_c15_frontier_allocation_probe(tasks::Vector{String};
    budget::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=5,
    parent_candidate_limit::Int=4,
    max_basin_contexts::Int=2,
    option_horizon::Int=3,
    control_repeats::Int=3,
    region_families::Vector{String}=["basin", "parent_novelty", "continuation"],
    max_allocation_budget::Int=2,
)
    println("\n" * "="^80)
    println("RUNNING BATCH 1N — FRONTIER-ALLOCATION / OPPORTUNITY-ROUTING CAUSAL PROBE")
    println("="^80)

    function collect_probe_run(task_name::String, repeat_idx::Int)
        oracle_mgr, budget_oracle, budget_oracle_batch = build_budget_oracles(task_name, budget)
        frontier_buffer = MolecularFrontierBuffer(5000)
        vocab = SMILESVocabulary()
        target_smiles = get(TARGET_SMILES, task_name, nothing)
        seed_pool = bootstrap_seed_pool(task_name;
            user_seed_smiles=String[],
            target_smiles=target_smiles,
            target_seed=target_seed)
        seed_stats = _seed_memories!(seed_pool, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            frontier_buffer=frontier_buffer,
            augmentation_count=bootstrap_augmentation_count(task_name, target_smiles, target_seed; enable_augmentation=true),
            verbose=true)
        warmup_stats = bootstrap_frontier_warmup!(task_name, frontier_buffer, oracle_mgr, budget_oracle_batch, vocab;
            target_smiles=target_smiles,
            rounds=bootstrap_warmup_rounds,
            verbose=true)

        cfg = HierarchicalEditConfig(
            max_step_attempts=max_step_attempts,
            max_operator_candidates=max_operator_candidates,
            parent_candidate_limit=parent_candidate_limit,
            basin_candidate_limit=max_basin_contexts,
        )
        probe = probe_frontier_allocation_opportunities(frontier_buffer, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            config=cfg,
            target_smiles=target_smiles,
            budget_remaining=budget_remaining(oracle_mgr),
            created_at_step=repeat_idx,
            task_name=task_name,
            max_parents=parent_candidate_limit,
            max_basins=max_basin_contexts,
            horizon=option_horizon,
            region_families=region_families,
            max_allocation_budget=max_allocation_budget)
        return Dict(
            "probe" => probe,
            "seed_stats" => seed_stats,
            "warmup_stats" => warmup_stats,
            "calls_used" => oracle_mgr.calls_used,
        )
    end

    function summarize_family(runs::Vector{<:AbstractDict}, family_name::String)
        family_summaries = Dict{String,Any}[]
        for run in runs
            probe = get(run, "probe", Dict{String,Any}())
            for summary in get(probe, "region_family_summaries", Dict{String,Any}[])
                if String(get(summary, "family_name", "")) == family_name
                    push!(family_summaries, summary)
                end
            end
        end
        isempty(family_summaries) && return Dict(
            "family_name" => family_name,
            "n_runs" => 0,
            "n_valid" => 0,
            "state_counts" => Dict{String,Int}(),
            "allocation_sensitive_fraction" => 0.0,
            "heuristic_dominant_fraction" => 0.0,
            "opportunity_routing_fraction" => 0.0,
            "mean_best_vs_heuristic_gap" => 0.0,
            "mean_best_vs_uniform_gap" => 0.0,
            "mean_heuristic_vs_anti_gap" => 0.0,
            "mean_region_opportunity_range" => 0.0,
            "mean_matched_budget" => 0.0,
            "interpretable" => false,
        )

        state_counts = Dict{String,Int}()
        best_vs_heuristic = Float64[]
        best_vs_uniform = Float64[]
        heuristic_vs_anti = Float64[]
        opportunity_ranges = Float64[]
        matched_budgets = Float64[]
        valid_count = 0
        for summary in family_summaries
            state = String(get(summary, "state_label", "taxonomy_ambiguous_state"))
            state_counts[state] = get(state_counts, state, 0) + 1
            if !Bool(get(summary, "all_degenerate", true))
                valid_count += 1
                push!(best_vs_heuristic, Float64(get(summary, "best_vs_heuristic_gap", 0.0)))
                push!(best_vs_uniform, Float64(get(summary, "best_vs_uniform_gap", 0.0)))
                push!(heuristic_vs_anti, Float64(get(summary, "heuristic_vs_anti_gap", 0.0)))
                push!(opportunity_ranges, Float64(get(summary, "region_opportunity_range", 0.0)))
                push!(matched_budgets, Float64(get(summary, "matched_budget", 0)))
            end
        end
        n_total = length(family_summaries)
        sensitive = get(state_counts, "allocation_sensitive_state", 0) + get(state_counts, "opportunity_routing_state", 0)
        heuristic_dom = get(state_counts, "heuristic_frontier_dominant_state", 0)
        opportunity = get(state_counts, "opportunity_routing_state", 0)
        mean_gap = isempty(best_vs_heuristic) ? 0.0 : mean(best_vs_heuristic)
        return Dict(
            "family_name" => family_name,
            "n_runs" => n_total,
            "n_valid" => valid_count,
            "state_counts" => state_counts,
            "allocation_sensitive_fraction" => n_total == 0 ? 0.0 : sensitive / n_total,
            "heuristic_dominant_fraction" => n_total == 0 ? 0.0 : heuristic_dom / n_total,
            "opportunity_routing_fraction" => n_total == 0 ? 0.0 : opportunity / n_total,
            "mean_best_vs_heuristic_gap" => mean_gap,
            "mean_best_vs_uniform_gap" => isempty(best_vs_uniform) ? 0.0 : mean(best_vs_uniform),
            "mean_heuristic_vs_anti_gap" => isempty(heuristic_vs_anti) ? 0.0 : mean(heuristic_vs_anti),
            "mean_region_opportunity_range" => isempty(opportunity_ranges) ? 0.0 : mean(opportunity_ranges),
            "mean_matched_budget" => isempty(matched_budgets) ? 0.0 : mean(matched_budgets),
            "interpretable" => valid_count > 0 && (mean_gap > 0.01 || opportunity > 0 || heuristic_dom > 0),
        )
    end

    function summarize_task(runs::Vector{<:AbstractDict})
        family_summaries = Dict{String,Any}()
        for family_name in region_families
            family_summaries[family_name] = summarize_family(runs, family_name)
        end
        family_names = collect(keys(family_summaries))
        isempty(family_names) && return Dict(
            "family_summaries" => family_summaries,
            "primary_family" => "",
            "primary_summary" => Dict{String,Any}(),
        )
        primary_family = if "basin" in family_names
            "basin"
        else
            family_names[argmax([get(family_summaries[name], "mean_best_vs_heuristic_gap", -Inf) for name in family_names])]
        end
        primary_summary = family_summaries[primary_family]
        return Dict(
            "family_summaries" => family_summaries,
            "primary_family" => primary_family,
            "primary_summary" => primary_summary,
        )
    end

    celecoxib_task = "celecoxib_rediscovery"
    celecoxib_task in tasks || error("Batch 1N requires celecoxib_rediscovery in PMO_TASKS")

    println("\nRunning celecoxib-first Batch 1N probe ...")
    celecoxib_runs = [collect_probe_run(celecoxib_task, idx) for idx in 1:data_repeats]
    celecoxib_summary = summarize_task(celecoxib_runs)
    cele_primary_family = celecoxib_summary["primary_family"]
    cele_primary = celecoxib_summary["primary_summary"]
    cele_gap = round(get(cele_primary, "mean_best_vs_heuristic_gap", 0.0), digits=4)
    cele_sens = round(get(cele_primary, "allocation_sensitive_fraction", 0.0), digits=4)
    cele_route = round(get(cele_primary, "opportunity_routing_fraction", 0.0), digits=4)
    cele_heur = round(get(cele_primary, "heuristic_dominant_fraction", 0.0), digits=4)
    println("  primary_family=$(cele_primary_family) | mean_gap=$(cele_gap) | sensitive=$(cele_sens) | routing=$(cele_route) | heuristic=$(cele_heur)")

    results_by_task = Dict{String,Any}(
        celecoxib_task => Dict(
            "runs" => celecoxib_runs,
            "summary" => celecoxib_summary,
        )
    )

    cele_interpretable = get(cele_primary, "interpretable", false)
    if cele_interpretable
        for task_name in ["drd2", "albuterol_similarity"]
            task_name in tasks || continue
            println("\nRunning control frontier-allocation probe on $(task_name) ...")
            task_runs = [collect_probe_run(task_name, idx) for idx in 1:control_repeats]
            task_summary = summarize_task(task_runs)
            primary_family = task_summary["primary_family"]
            primary = task_summary["primary_summary"]
            control_gap = round(get(primary, "mean_best_vs_heuristic_gap", 0.0), digits=4)
            control_sens = round(get(primary, "allocation_sensitive_fraction", 0.0), digits=4)
            control_route = round(get(primary, "opportunity_routing_fraction", 0.0), digits=4)
            println("  primary_family=$(primary_family) | mean_gap=$(control_gap) | sensitive=$(control_sens) | routing=$(control_route)")
            results_by_task[task_name] = Dict(
                "runs" => task_runs,
                "summary" => task_summary,
            )
        end
    end

    decisions = Dict{String,Any}(
        "celecoxib_primary_family" => cele_primary_family,
        "celecoxib_mean_best_vs_heuristic_gap" => get(cele_primary, "mean_best_vs_heuristic_gap", 0.0),
        "celecoxib_mean_best_vs_uniform_gap" => get(cele_primary, "mean_best_vs_uniform_gap", 0.0),
        "celecoxib_allocation_sensitive_fraction" => get(cele_primary, "allocation_sensitive_fraction", 0.0),
        "celecoxib_heuristic_dominant_fraction" => get(cele_primary, "heuristic_dominant_fraction", 0.0),
        "celecoxib_opportunity_routing_fraction" => get(cele_primary, "opportunity_routing_fraction", 0.0),
        "celecoxib_interpretable" => get(cele_primary, "interpretable", false),
    )

    decisions["global_recommendation"] = if get(cele_primary, "mean_best_vs_heuristic_gap", 0.0) > 0.05 && get(cele_primary, "opportunity_routing_fraction", 0.0) >= 0.4
        "PROCEED_TO_LEARNED_FRONTIER_ALLOCATOR"
    elseif get(cele_primary, "mean_best_vs_heuristic_gap", 0.0) > 0.02 && get(cele_primary, "allocation_sensitive_fraction", 0.0) >= 0.4
        "FRONTIER_ROUTING_REAL_BUT_NARROW"
    elseif get(cele_primary, "heuristic_dominant_fraction", 0.0) >= 0.6
        "HEURISTIC_FRONTIER_ALREADY_STRONG"
    else
        "UPWARD_BRANCH_WEAK_OR_INCONCLUSIVE"
    end

    global_rec = decisions["global_recommendation"]
    println("\nGlobal recommendation: $(global_rec) | family=$(cele_primary_family) | mean_gap=$(cele_gap) | sensitive=$(cele_sens) | routing=$(cele_route) | heuristic=$(cele_heur)")
    return Dict("results_by_task" => results_by_task, "decisions" => decisions)
end


function run_c16_selective_frontier_allocator_checks(tasks::Vector{String};
    budget::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=8,
    parent_candidate_limit::Int=4,
    max_basin_contexts::Int=2,
    option_horizon::Int=3,
    control_repeats::Int=3,
    max_allocation_budget::Int=2,
    override_threshold::Float64=0.01,
    train_fraction::Float64=0.75,
)
    println("\n" * "="^80)
    println("RUNNING BATCH 1O — SELECTIVE BASIN-CENTERED FRONTIER ALLOCATOR")
    println("="^80)

    function collect_probe_run(task_name::String, repeat_idx::Int)
        oracle_mgr, budget_oracle, budget_oracle_batch = build_budget_oracles(task_name, budget)
        frontier_buffer = MolecularFrontierBuffer(5000)
        vocab = SMILESVocabulary()
        target_smiles = get(TARGET_SMILES, task_name, nothing)
        seed_pool = bootstrap_seed_pool(task_name;
            user_seed_smiles=String[],
            target_smiles=target_smiles,
            target_seed=target_seed)
        seed_stats = _seed_memories!(seed_pool, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            frontier_buffer=frontier_buffer,
            augmentation_count=bootstrap_augmentation_count(task_name, target_smiles, target_seed; enable_augmentation=true),
            verbose=true)
        warmup_stats = bootstrap_frontier_warmup!(task_name, frontier_buffer, oracle_mgr, budget_oracle_batch, vocab;
            target_smiles=target_smiles,
            rounds=bootstrap_warmup_rounds,
            verbose=true)

        cfg = HierarchicalEditConfig(
            max_step_attempts=max_step_attempts,
            max_operator_candidates=max_operator_candidates,
            parent_candidate_limit=parent_candidate_limit,
            basin_candidate_limit=max_basin_contexts,
        )
        probe = probe_frontier_allocation_opportunities(frontier_buffer, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            config=cfg,
            target_smiles=target_smiles,
            budget_remaining=budget_remaining(oracle_mgr),
            created_at_step=repeat_idx,
            task_name=task_name,
            max_parents=parent_candidate_limit,
            max_basins=max_basin_contexts,
            horizon=option_horizon,
            region_families=["basin"],
            max_allocation_budget=max_allocation_budget)
        return Dict(
            "probe" => probe,
            "seed_stats" => seed_stats,
            "warmup_stats" => warmup_stats,
            "calls_used" => oracle_mgr.calls_used,
        )
    end

    function run_task(task_name::String, repeats::Int)
        runs = [collect_probe_run(task_name, idx) for idx in 1:repeats]
        dataset = extract_frontier_allocation_dataset(runs;
            family_name="basin",
            override_threshold=override_threshold)
        trained = train_selective_frontier_allocator(dataset;
            rng=MersenneTwister(0),
            train_fraction=train_fraction,
            family_name="basin")
        return Dict(
            "runs" => runs,
            "dataset" => dataset,
            "dataset_stats" => frontier_allocation_dataset_stats(dataset),
            "training" => trained,
        )
    end

    function summarize_task(task_result::Dict{String,Any})
        dataset_stats = get(task_result, "dataset_stats", Dict{String,Any}())
        trained = get(task_result, "training", Dict{String,Any}())
        model = get(trained, "model", nothing)
        if isnothing(model)
            return Dict(
                "dataset_stats" => dataset_stats,
                "interpretable" => false,
                "reason" => get(trained, "reason", "no_model"),
                "selective_val" => Dict{String,Any}(),
                "heuristic_val" => Dict{String,Any}(),
                "always_override_val" => Dict{String,Any}(),
                "oracle_val" => Dict{String,Any}(),
                "final_eval" => Dict{String,Any}(),
            )
        end
        selective_val = get(trained, "val_eval", Dict{String,Any}())
        heuristic_val = get(trained, "heuristic_val_eval", Dict{String,Any}())
        always_override_val = get(trained, "always_override_val_eval", Dict{String,Any}())
        oracle_val = get(trained, "oracle_val_eval", Dict{String,Any}())
        final_eval = get(trained, "final_eval", Dict{String,Any}())
        override_fraction = Float64(get(selective_val, "override_fraction", 0.0))
        abstention_fraction = Float64(get(selective_val, "abstention_fraction", 0.0))
        interpretable = Int(get(selective_val, "n_snapshots", 0)) > 0 && 0.0 < abstention_fraction < 1.0 && 0.0 < override_fraction < 1.0
        return Dict(
            "dataset_stats" => dataset_stats,
            "interpretable" => interpretable,
            "selective_val" => selective_val,
            "heuristic_val" => heuristic_val,
            "always_override_val" => always_override_val,
            "oracle_val" => oracle_val,
            "final_eval" => final_eval,
            "override_threshold" => get(trained, "override_threshold", 0.5f0),
            "margin_threshold" => get(trained, "margin_threshold", 0.0f0),
        )
    end

    celecoxib_task = "celecoxib_rediscovery"
    celecoxib_task in tasks || error("Batch 1O requires celecoxib_rediscovery in PMO_TASKS")

    println("\nRunning celecoxib-first Batch 1O selective allocator ...")
    celecoxib_result = run_task(celecoxib_task, data_repeats)
    celecoxib_summary = summarize_task(celecoxib_result)
    cele_selective = celecoxib_summary["selective_val"]
    cele_gain = round(get(cele_selective, "mean_gain_vs_heuristic", 0.0), digits=4)
    cele_regret = round(get(cele_selective, "mean_regret_vs_best_region", 0.0), digits=4)
    cele_override = round(get(cele_selective, "override_fraction", 0.0), digits=4)
    cele_abstain = round(get(cele_selective, "abstention_fraction", 0.0), digits=4)
    cele_precision = round(get(cele_selective, "override_precision", 0.0), digits=4)
    println("  val_gain=$(cele_gain) | val_regret=$(cele_regret) | override=$(cele_override) | abstain=$(cele_abstain) | precision=$(cele_precision)")

    results_by_task = Dict{String,Any}(
        celecoxib_task => Dict(
            "result" => celecoxib_result,
            "summary" => celecoxib_summary,
        )
    )

    cele_interpretable = get(celecoxib_summary, "interpretable", false) && get(cele_selective, "mean_gain_vs_heuristic", -Inf) >= 0.0
    if cele_interpretable
        for task_name in ["drd2", "albuterol_similarity"]
            task_name in tasks || continue
            println("\nRunning control selective allocator on $(task_name) ...")
            task_result = run_task(task_name, control_repeats)
            task_summary = summarize_task(task_result)
            selective = task_summary["selective_val"]
            control_gain = round(get(selective, "mean_gain_vs_heuristic", 0.0), digits=4)
            control_override = round(get(selective, "override_fraction", 0.0), digits=4)
            control_abstain = round(get(selective, "abstention_fraction", 0.0), digits=4)
            println("  val_gain=$(control_gain) | override=$(control_override) | abstain=$(control_abstain)")
            results_by_task[task_name] = Dict(
                "result" => task_result,
                "summary" => task_summary,
            )
        end
    end

    cele_final = get(celecoxib_summary, "final_eval", Dict{String,Any}())
    cele_always = get(celecoxib_summary, "always_override_val", Dict{String,Any}())
    cele_oracle = get(celecoxib_summary, "oracle_val", Dict{String,Any}())
    decisions = Dict{String,Any}(
        "celecoxib_dataset_size" => get(celecoxib_summary["dataset_stats"], "size", 0),
        "celecoxib_override_positive_fraction" => get(celecoxib_summary["dataset_stats"], "override_positive_fraction", 0.0),
        "celecoxib_val_mean_gain_vs_heuristic" => get(cele_selective, "mean_gain_vs_heuristic", 0.0),
        "celecoxib_val_mean_regret_vs_best_region" => get(cele_selective, "mean_regret_vs_best_region", 0.0),
        "celecoxib_val_override_fraction" => get(cele_selective, "override_fraction", 0.0),
        "celecoxib_val_abstention_fraction" => get(cele_selective, "abstention_fraction", 0.0),
        "celecoxib_val_override_precision" => get(cele_selective, "override_precision", 0.0),
        "celecoxib_val_override_recall" => get(cele_selective, "override_recall", 0.0),
        "celecoxib_val_basin_choice_accuracy" => get(cele_selective, "basin_choice_accuracy", 0.0),
        "celecoxib_final_mean_gain_vs_heuristic" => get(cele_final, "mean_gain_vs_heuristic", 0.0),
        "celecoxib_always_override_val_gain" => get(cele_always, "mean_gain_vs_heuristic", 0.0),
        "celecoxib_oracle_val_gain" => get(cele_oracle, "mean_gain_vs_heuristic", 0.0),
        "celecoxib_interpretable" => get(celecoxib_summary, "interpretable", false),
    )

    decisions["global_recommendation"] = if get(cele_selective, "mean_gain_vs_heuristic", 0.0) > 0.01 && get(cele_final, "mean_gain_vs_heuristic", 0.0) >= 0.0 && 0.05 <= get(cele_selective, "override_fraction", 0.0) <= 0.95 && 0.05 <= get(cele_selective, "abstention_fraction", 0.0) <= 0.95 && get(cele_selective, "mean_regret_vs_best_region", Inf) < get(get(celecoxib_summary, "heuristic_val", Dict{String,Any}()), "mean_regret_vs_best_region", Inf)
        "PROMOTE_SELECTIVE_FRONTIER_ALLOCATOR"
    elseif get(cele_selective, "mean_gain_vs_heuristic", 0.0) >= 0.0 && 0.05 <= get(cele_selective, "override_fraction", 0.0) <= 0.95 && 0.05 <= get(cele_selective, "abstention_fraction", 0.0) <= 0.95
        "SELECTIVE_ALLOCATOR_NONNEGATIVE_KEEP_CAUTIOUS"
    elseif get(cele_always, "mean_gain_vs_heuristic", 0.0) > 0.0
        "UPPER_SEAM_REAL_SELECTIVITY_MODEL_WEAK"
    elseif get(cele_selective, "abstention_fraction", 0.0) >= 0.95
        "ALWAYS_ABSTAIN_COLLAPSE"
    elseif get(cele_selective, "override_fraction", 0.0) >= 0.95
        "ALWAYS_OVERRIDE_COLLAPSE"
    else
        "SELECTIVE_ALLOCATOR_FAILED_OR_INCONCLUSIVE"
    end

    global_rec = decisions["global_recommendation"]
    println("\nGlobal recommendation: $(global_rec) | val_gain=$(cele_gain) | override=$(cele_override) | abstain=$(cele_abstain) | oracle_gap=$(round(get(cele_oracle, "mean_gain_vs_heuristic", 0.0), digits=4))")
    return Dict("results_by_task" => results_by_task, "decisions" => decisions)
end


function run_c17_opportunity_state_probe(tasks::Vector{String};
    budget::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=10,
    parent_candidate_limit::Int=4,
    max_basin_contexts::Int=2,
    option_horizon::Int=3,
    control_repeats::Int=3,
    max_allocation_budget::Int=2,
    state_threshold::Float64=0.01,
    train_fraction::Float64=0.75,
)
    println("\n" * "="^80)
    println("RUNNING BATCH 1P — OPPORTUNITY-STATE / OVERRIDE-ELIGIBILITY PROBE")
    println("="^80)

    function collect_probe_run(task_name::String, repeat_idx::Int)
        oracle_mgr, budget_oracle, budget_oracle_batch = build_budget_oracles(task_name, budget)
        frontier_buffer = MolecularFrontierBuffer(5000)
        vocab = SMILESVocabulary()
        target_smiles = get(TARGET_SMILES, task_name, nothing)
        seed_pool = bootstrap_seed_pool(task_name;
            user_seed_smiles=String[],
            target_smiles=target_smiles,
            target_seed=target_seed)
        seed_stats = _seed_memories!(seed_pool, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            frontier_buffer=frontier_buffer,
            augmentation_count=bootstrap_augmentation_count(task_name, target_smiles, target_seed; enable_augmentation=true),
            verbose=true)
        warmup_stats = bootstrap_frontier_warmup!(task_name, frontier_buffer, oracle_mgr, budget_oracle_batch, vocab;
            target_smiles=target_smiles,
            rounds=bootstrap_warmup_rounds,
            verbose=true)

        cfg = HierarchicalEditConfig(
            max_step_attempts=max_step_attempts,
            max_operator_candidates=max_operator_candidates,
            parent_candidate_limit=parent_candidate_limit,
            basin_candidate_limit=max_basin_contexts,
        )
        probe = probe_frontier_allocation_opportunities(frontier_buffer, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            config=cfg,
            target_smiles=target_smiles,
            budget_remaining=budget_remaining(oracle_mgr),
            created_at_step=repeat_idx,
            task_name=task_name,
            max_parents=parent_candidate_limit,
            max_basins=max_basin_contexts,
            horizon=option_horizon,
            region_families=["basin"],
            max_allocation_budget=max_allocation_budget)
        return Dict(
            "probe" => probe,
            "seed_stats" => seed_stats,
            "warmup_stats" => warmup_stats,
            "calls_used" => oracle_mgr.calls_used,
        )
    end

    function run_task(task_name::String, repeats::Int)
        runs = [collect_probe_run(task_name, idx) for idx in 1:repeats]
        dataset = extract_opportunity_state_dataset(runs;
            family_name="basin",
            state_threshold=state_threshold)
        trained = train_opportunity_state_detector(dataset;
            rng=MersenneTwister(0),
            train_fraction=train_fraction,
            family_name="basin")
        return Dict(
            "runs" => runs,
            "dataset" => dataset,
            "dataset_stats" => opportunity_state_dataset_stats(dataset),
            "training" => trained,
        )
    end

    function summarize_task(task_result::Dict{String,Any})
        dataset_stats = get(task_result, "dataset_stats", Dict{String,Any}())
        trained = get(task_result, "training", Dict{String,Any}())
        model = get(trained, "model", nothing)
        if isnothing(model)
            return Dict(
                "dataset_stats" => dataset_stats,
                "interpretable" => false,
                "reason" => get(trained, "reason", "no_model"),
                "val_eval" => Dict{String,Any}(),
                "final_eval" => Dict{String,Any}(),
                "conditional_val_eval" => Dict{String,Any}(),
                "conditional_final_eval" => Dict{String,Any}(),
                "oracle_val_eval" => Dict{String,Any}(),
                "oracle_conditional_val_eval" => Dict{String,Any}(),
            )
        end
        val_eval = get(trained, "val_eval", Dict{String,Any}())
        final_eval = get(trained, "final_eval", Dict{String,Any}())
        conditional_val = get(trained, "conditional_val_eval", Dict{String,Any}())
        conditional_final = get(trained, "conditional_final_eval", Dict{String,Any}())
        oracle_val = get(trained, "oracle_val_eval", Dict{String,Any}())
        oracle_conditional_val = get(trained, "oracle_conditional_val_eval", Dict{String,Any}())
        gap_sep = Float64(get(val_eval, "mean_gap_predicted_positive", 0.0)) - Float64(get(val_eval, "mean_gap_predicted_negative", 0.0))
        predicted_positive = Float64(get(val_eval, "predicted_positive_fraction", 0.0))
        interpretable = Int(get(val_eval, "n_snapshots", 0)) > 0 && gap_sep > 0.01 && 0.0 < predicted_positive < 1.0
        return Dict(
            "dataset_stats" => dataset_stats,
            "interpretable" => interpretable,
            "val_eval" => val_eval,
            "final_eval" => final_eval,
            "conditional_val_eval" => conditional_val,
            "conditional_final_eval" => conditional_final,
            "oracle_val_eval" => oracle_val,
            "oracle_conditional_val_eval" => oracle_conditional_val,
            "threshold" => get(trained, "threshold", 0.5f0),
        )
    end

    celecoxib_task = "celecoxib_rediscovery"
    celecoxib_task in tasks || error("Batch 1P requires celecoxib_rediscovery in PMO_TASKS")

    println("\nRunning celecoxib-first Batch 1P state probe ...")
    celecoxib_result = run_task(celecoxib_task, data_repeats)
    celecoxib_summary = summarize_task(celecoxib_result)
    cele_val = celecoxib_summary["val_eval"]
    cele_cond = celecoxib_summary["conditional_val_eval"]
    cele_gap_sep = round(get(cele_val, "mean_gap_predicted_positive", 0.0) - get(cele_val, "mean_gap_predicted_negative", 0.0), digits=4)
    cele_pred_pos = round(get(cele_val, "predicted_positive_fraction", 0.0), digits=4)
    cele_prec = round(get(cele_val, "precision", 0.0), digits=4)
    cele_recall = round(get(cele_val, "recall", 0.0), digits=4)
    cele_cond_gain = round(get(cele_cond, "mean_gain_vs_heuristic", 0.0), digits=4)
    println("  val_pred_pos=$(cele_pred_pos) | gap_sep=$(cele_gap_sep) | precision=$(cele_prec) | recall=$(cele_recall) | cond_gain=$(cele_cond_gain)")

    results_by_task = Dict{String,Any}(
        celecoxib_task => Dict(
            "result" => celecoxib_result,
            "summary" => celecoxib_summary,
        )
    )

    cele_interpretable = get(celecoxib_summary, "interpretable", false)
    if cele_interpretable
        for task_name in ["drd2", "albuterol_similarity"]
            task_name in tasks || continue
            println("\nRunning control state probe on $(task_name) ...")
            task_result = run_task(task_name, control_repeats)
            task_summary = summarize_task(task_result)
            val_eval = task_summary["val_eval"]
            cond_eval = task_summary["conditional_val_eval"]
            control_pred = round(get(val_eval, "predicted_positive_fraction", 0.0), digits=4)
            control_gap_sep = round(get(val_eval, "mean_gap_predicted_positive", 0.0) - get(val_eval, "mean_gap_predicted_negative", 0.0), digits=4)
            control_cond = round(get(cond_eval, "mean_gain_vs_heuristic", 0.0), digits=4)
            println("  val_pred_pos=$(control_pred) | gap_sep=$(control_gap_sep) | cond_gain=$(control_cond)")
            results_by_task[task_name] = Dict(
                "result" => task_result,
                "summary" => task_summary,
            )
        end
    end

    cele_final = get(celecoxib_summary, "final_eval", Dict{String,Any}())
    cele_cond_final = get(celecoxib_summary, "conditional_final_eval", Dict{String,Any}())
    cele_oracle_cond = get(celecoxib_summary, "oracle_conditional_val_eval", Dict{String,Any}())
    decisions = Dict{String,Any}(
        "celecoxib_dataset_size" => get(celecoxib_summary["dataset_stats"], "size", 0),
        "celecoxib_override_eligible_fraction" => get(celecoxib_summary["dataset_stats"], "override_eligible_fraction", 0.0),
        "celecoxib_val_predicted_positive_fraction" => get(cele_val, "predicted_positive_fraction", 0.0),
        "celecoxib_val_precision" => get(cele_val, "precision", 0.0),
        "celecoxib_val_recall" => get(cele_val, "recall", 0.0),
        "celecoxib_val_gap_positive" => get(cele_val, "mean_gap_predicted_positive", 0.0),
        "celecoxib_val_gap_negative" => get(cele_val, "mean_gap_predicted_negative", 0.0),
        "celecoxib_val_gap_separation" => get(cele_val, "mean_gap_predicted_positive", 0.0) - get(cele_val, "mean_gap_predicted_negative", 0.0),
        "celecoxib_val_routing_sensitive_fraction_in_positive" => get(cele_val, "routing_sensitive_fraction_in_predicted_positive", 0.0),
        "celecoxib_val_heuristic_dominant_fraction_in_negative" => get(cele_val, "heuristic_dominant_fraction_in_predicted_negative", 0.0),
        "celecoxib_conditional_val_gain" => get(cele_cond, "mean_gain_vs_heuristic", 0.0),
        "celecoxib_conditional_final_gain" => get(cele_cond_final, "mean_gain_vs_heuristic", 0.0),
        "celecoxib_oracle_conditional_val_gain" => get(cele_oracle_cond, "mean_gain_vs_heuristic", 0.0),
        "celecoxib_interpretable" => get(celecoxib_summary, "interpretable", false),
    )

    decisions["global_recommendation"] = if get(cele_val, "predicted_positive_fraction", 0.0) > 0.0 && get(cele_val, "predicted_positive_fraction", 0.0) < 1.0 && (get(cele_val, "mean_gap_predicted_positive", 0.0) - get(cele_val, "mean_gap_predicted_negative", 0.0)) > 0.02 && get(cele_cond, "mean_gain_vs_heuristic", 0.0) > 0.0
        "PROMOTE_OPPORTUNITY_STATE_DETECTOR"
    elseif (get(cele_val, "mean_gap_predicted_positive", 0.0) - get(cele_val, "mean_gap_predicted_negative", 0.0)) > 0.0
        "STATE_SPLIT_REAL_BUT_WEAK"
    elseif get(cele_val, "predicted_positive_fraction", 0.0) == 0.0
        "ALWAYS_NEGATIVE_COLLAPSE"
    elseif get(cele_val, "predicted_positive_fraction", 0.0) == 1.0
        "ALWAYS_POSITIVE_COLLAPSE"
    else
        "STATE_PROBE_FAILED_OR_INCONCLUSIVE"
    end

    global_rec = decisions["global_recommendation"]
    println("\nGlobal recommendation: $(global_rec) | pred_pos=$(cele_pred_pos) | gap_sep=$(cele_gap_sep) | cond_gain=$(cele_cond_gain) | oracle_cond=$(round(get(cele_oracle_cond, "mean_gain_vs_heuristic", 0.0), digits=4))")
    return Dict("results_by_task" => results_by_task, "decisions" => decisions)
end

function run_c18_opportunity_state_repeatability(tasks::Vector{String};
    budget::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=16,
    num_splits::Int=5,
    parent_candidate_limit::Int=4,
    max_basin_contexts::Int=2,
    option_horizon::Int=3,
    control_repeats::Int=3,
    max_allocation_budget::Int=2,
    state_threshold::Float64=0.01,
    train_fraction::Float64=0.75,
    threshold_perturbations::Vector{Float64}=[-0.05, 0.0, 0.05],
)
    println("\n" * "="^80)
    println("RUNNING BATCH 1Q — OPPORTUNITY-STATE REPEATABILITY GATE")
    println("="^80)

    function collect_probe_run(task_name::String, repeat_idx::Int)
        oracle_mgr, budget_oracle, budget_oracle_batch = build_budget_oracles(task_name, budget)
        frontier_buffer = MolecularFrontierBuffer(5000)
        vocab = SMILESVocabulary()
        target_smiles = get(TARGET_SMILES, task_name, nothing)
        seed_pool = bootstrap_seed_pool(task_name;
            user_seed_smiles=String[],
            target_smiles=target_smiles,
            target_seed=target_seed)
        seed_stats = _seed_memories!(seed_pool, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            frontier_buffer=frontier_buffer,
            augmentation_count=bootstrap_augmentation_count(task_name, target_smiles, target_seed; enable_augmentation=true),
            verbose=true)
        warmup_stats = bootstrap_frontier_warmup!(task_name, frontier_buffer, oracle_mgr, budget_oracle_batch, vocab;
            target_smiles=target_smiles,
            rounds=bootstrap_warmup_rounds,
            verbose=true)

        cfg = HierarchicalEditConfig(
            max_step_attempts=max_step_attempts,
            max_operator_candidates=max_operator_candidates,
            parent_candidate_limit=parent_candidate_limit,
            basin_candidate_limit=max_basin_contexts,
        )
        probe = probe_frontier_allocation_opportunities(frontier_buffer, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            config=cfg,
            target_smiles=target_smiles,
            budget_remaining=budget_remaining(oracle_mgr),
            created_at_step=repeat_idx,
            task_name=task_name,
            max_parents=parent_candidate_limit,
            max_basins=max_basin_contexts,
            horizon=option_horizon,
            region_families=["basin"],
            max_allocation_budget=max_allocation_budget)
        return Dict(
            "probe" => probe,
            "seed_stats" => seed_stats,
            "warmup_stats" => warmup_stats,
            "calls_used" => oracle_mgr.calls_used,
        )
    end

    function run_task(task_name::String, repeats::Int)
        runs = [collect_probe_run(task_name, idx) for idx in 1:repeats]
        dataset = extract_opportunity_state_dataset(runs;
            family_name="basin",
            state_threshold=state_threshold)
        repeatability = evaluate_opportunity_state_repeatability(dataset;
            rng=MersenneTwister(0),
            num_splits=num_splits,
            train_fraction=train_fraction,
            family_name="basin",
            threshold_perturbations=threshold_perturbations)
        full_training = train_opportunity_state_detector(dataset;
            rng=MersenneTwister(0),
            train_fraction=train_fraction,
            family_name="basin")
        return Dict(
            "runs" => runs,
            "dataset" => dataset,
            "dataset_stats" => opportunity_state_dataset_stats(dataset),
            "repeatability" => repeatability,
            "full_training" => full_training,
        )
    end

    function summarize_task(task_result::Dict{String,Any})
        dataset_stats = get(task_result, "dataset_stats", Dict{String,Any}())
        repeatability = get(task_result, "repeatability", Dict{String,Any}())
        repeat_summary = get(repeatability, "summary", Dict{String,Any}())
        full_training = get(task_result, "full_training", Dict{String,Any}())
        full_final_eval = get(full_training, "final_eval", Dict{String,Any}())
        full_conditional_final = get(full_training, "conditional_final_eval", Dict{String,Any}())
        return Dict(
            "dataset_stats" => dataset_stats,
            "repeatability_summary" => repeat_summary,
            "repeatability_splits" => get(repeatability, "splits", Dict{String,Any}[]),
            "interpretable" => Bool(get(repeat_summary, "repeatability_safe", false)),
            "recommendation" => get(repeat_summary, "recommendation", "STATE_SPLIT_NOT_HELD_OUT_SAFE"),
            "full_final_eval" => full_final_eval,
            "full_conditional_final_eval" => full_conditional_final,
            "reason" => get(repeatability, "reason", get(full_training, "reason", "ok")),
        )
    end

    celecoxib_task = "celecoxib_rediscovery"
    celecoxib_task in tasks || error("Batch 1Q requires celecoxib_rediscovery in PMO_TASKS")

    println("\nRunning celecoxib-first Batch 1Q repeatability gate ...")
    celecoxib_result = run_task(celecoxib_task, data_repeats)
    celecoxib_summary = summarize_task(celecoxib_result)
    cele_repeat = celecoxib_summary["repeatability_summary"]
    cele_nondeg = round(get(cele_repeat, "nondegenerate_fraction", 0.0), digits=4)
    cele_gap = round(get(cele_repeat, "median_gap_separation", 0.0), digits=4)
    cele_cond = round(get(cele_repeat, "median_conditional_gain", 0.0), digits=4)
    cele_robust = round(get(cele_repeat, "median_threshold_robust_fraction", 0.0), digits=4)
    println("  nondeg=$(cele_nondeg) | median_gap=$(cele_gap) | median_cond_gain=$(cele_cond) | median_threshold_robust=$(cele_robust)")

    results_by_task = Dict{String,Any}(
        celecoxib_task => Dict(
            "result" => celecoxib_result,
            "summary" => celecoxib_summary,
        )
    )

    cele_interpretable = get(celecoxib_summary, "interpretable", false)
    if cele_interpretable
        for task_name in ["drd2", "albuterol_similarity"]
            task_name in tasks || continue
            println("\nRunning control repeatability gate on $(task_name) ...")
            task_result = run_task(task_name, control_repeats)
            task_summary = summarize_task(task_result)
            repeat_summary = task_summary["repeatability_summary"]
            control_nondeg = round(get(repeat_summary, "nondegenerate_fraction", 0.0), digits=4)
            control_gap = round(get(repeat_summary, "median_gap_separation", 0.0), digits=4)
            control_cond = round(get(repeat_summary, "median_conditional_gain", 0.0), digits=4)
            println("  nondeg=$(control_nondeg) | median_gap=$(control_gap) | median_cond_gain=$(control_cond) | rec=$(task_summary["recommendation"])")
            results_by_task[task_name] = Dict(
                "result" => task_result,
                "summary" => task_summary,
            )
        end
    end

    cele_full_cond = get(celecoxib_summary["full_conditional_final_eval"], "mean_gain_vs_heuristic", 0.0)
    decisions = Dict{String,Any}(
        "celecoxib_dataset_size" => get(celecoxib_summary["dataset_stats"], "size", 0),
        "celecoxib_override_eligible_fraction" => get(celecoxib_summary["dataset_stats"], "override_eligible_fraction", 0.0),
        "celecoxib_num_splits" => get(cele_repeat, "n_splits", 0),
        "celecoxib_nondegenerate_fraction" => get(cele_repeat, "nondegenerate_fraction", 0.0),
        "celecoxib_zero_positive_collapse_fraction" => get(cele_repeat, "zero_positive_collapse_fraction", 0.0),
        "celecoxib_all_positive_collapse_fraction" => get(cele_repeat, "all_positive_collapse_fraction", 0.0),
        "celecoxib_median_predicted_positive_fraction" => get(cele_repeat, "median_predicted_positive_fraction", 0.0),
        "celecoxib_median_gap_separation" => get(cele_repeat, "median_gap_separation", 0.0),
        "celecoxib_median_conditional_gain" => get(cele_repeat, "median_conditional_gain", 0.0),
        "celecoxib_mean_routing_sensitive_fraction_in_positive" => get(cele_repeat, "mean_routing_sensitive_fraction_in_positive", 0.0),
        "celecoxib_mean_heuristic_dominant_fraction_in_negative" => get(cele_repeat, "mean_heuristic_dominant_fraction_in_negative", 0.0),
        "celecoxib_median_threshold_robust_fraction" => get(cele_repeat, "median_threshold_robust_fraction", 0.0),
        "celecoxib_full_conditional_final_gain" => cele_full_cond,
        "celecoxib_interpretable" => get(celecoxib_summary, "interpretable", false),
        "global_recommendation" => get(celecoxib_summary, "recommendation", "STATE_SPLIT_NOT_HELD_OUT_SAFE"),
    )

    global_rec = decisions["global_recommendation"]
    println("\nGlobal recommendation: $(global_rec) | nondeg=$(cele_nondeg) | median_gap=$(cele_gap) | median_cond_gain=$(cele_cond) | median_threshold_robust=$(cele_robust) | full_cond=$(round(cele_full_cond, digits=4))")
    return Dict("results_by_task" => results_by_task, "decisions" => decisions)
end


function run_c19_intervention_geometry_audit(tasks::Vector{String};
    budget::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=16,
    parent_candidate_limit::Int=4,
    max_basin_contexts::Int=2,
    option_horizon::Int=3,
    control_repeats::Int=3,
    max_allocation_budget::Int=2,
    state_threshold::Float64=0.01,
    stability_threshold::Float64=0.05,
)
    println("\n" * "="^80)
    println("RUNNING BATCH 1R — INTERVENTION-GEOMETRY AUDIT")
    println("="^80)

    function collect_probe_run(task_name::String, repeat_idx::Int)
        oracle_mgr, budget_oracle, budget_oracle_batch = build_budget_oracles(task_name, budget)
        frontier_buffer = MolecularFrontierBuffer(5000)
        vocab = SMILESVocabulary()
        target_smiles = get(TARGET_SMILES, task_name, nothing)
        seed_pool = bootstrap_seed_pool(task_name;
            user_seed_smiles=String[],
            target_smiles=target_smiles,
            target_seed=target_seed)
        seed_stats = _seed_memories!(seed_pool, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            frontier_buffer=frontier_buffer,
            augmentation_count=bootstrap_augmentation_count(task_name, target_smiles, target_seed; enable_augmentation=true),
            verbose=true)
        warmup_stats = bootstrap_frontier_warmup!(task_name, frontier_buffer, oracle_mgr, budget_oracle_batch, vocab;
            target_smiles=target_smiles,
            rounds=bootstrap_warmup_rounds,
            verbose=true)

        cfg = HierarchicalEditConfig(
            max_step_attempts=max_step_attempts,
            max_operator_candidates=max_operator_candidates,
            parent_candidate_limit=parent_candidate_limit,
            basin_candidate_limit=max_basin_contexts,
        )
        probe = probe_frontier_allocation_opportunities(frontier_buffer, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            config=cfg,
            target_smiles=target_smiles,
            budget_remaining=budget_remaining(oracle_mgr),
            created_at_step=repeat_idx,
            task_name=task_name,
            max_parents=parent_candidate_limit,
            max_basins=max_basin_contexts,
            horizon=option_horizon,
            region_families=["basin"],
            max_allocation_budget=max_allocation_budget)
        return Dict(
            "probe" => probe,
            "seed_stats" => seed_stats,
            "warmup_stats" => warmup_stats,
            "calls_used" => oracle_mgr.calls_used,
        )
    end

    function run_task(task_name::String, repeats::Int)
        runs = [collect_probe_run(task_name, idx) for idx in 1:repeats]
        atlas = extract_intervention_geometry_atlas(runs;
            family_name="basin",
            state_threshold=state_threshold,
            stability_threshold=stability_threshold)
        return Dict(
            "runs" => runs,
            "atlas" => atlas,
        )
    end

    function summarize_task(task_result::Dict{String,Any})
        atlas = get(task_result, "atlas", Dict{String,Any}())
        stats = get(atlas, "stats", Dict{String,Any}())
        comparison = get(atlas, "comparison", Dict{String,Any}())
        return Dict(
            "atlas_stats" => stats,
            "comparison" => comparison,
            "records" => get(atlas, "records", Dict{String,Any}[]),
            "interpretable" => Bool(get(atlas, "interpretable", false)),
            "recommendation" => get(atlas, "recommendation", "NO_INTERVENTION_GEOMETRY"),
        )
    end

    celecoxib_task = "celecoxib_rediscovery"
    celecoxib_task in tasks || error("Batch 1R requires celecoxib_rediscovery in PMO_TASKS")

    println("\nRunning celecoxib-first Batch 1R intervention-geometry audit ...")
    celecoxib_result = run_task(celecoxib_task, data_repeats)
    celecoxib_summary = summarize_task(celecoxib_result)
    cele_stats = celecoxib_summary["atlas_stats"]
    cele_comp = celecoxib_summary["comparison"]
    cele_active = get(cele_stats, "n_active_regimes", 0)
    cele_stable_delta = round(get(cele_comp, "stable_vs_pooled_gap_mean_delta", 0.0), digits=4)
    cele_std_impr = round(get(cele_comp, "stable_vs_pooled_gap_std_improvement", 0.0), digits=4)
    cele_ambig = round(get(cele_comp, "pooled_positive_ambiguous_fraction", 0.0), digits=4)
    println("  active_regimes=$(cele_active) | stable_gap_delta=$(cele_stable_delta) | gap_std_improvement=$(cele_std_impr) | pooled_positive_ambiguous_fraction=$(cele_ambig)")

    results_by_task = Dict{String,Any}(
        celecoxib_task => Dict(
            "result" => celecoxib_result,
            "summary" => celecoxib_summary,
        )
    )

    cele_interpretable = get(celecoxib_summary, "interpretable", false)
    if cele_interpretable
        for task_name in ["drd2", "albuterol_similarity"]
            task_name in tasks || continue
            println("\nRunning control intervention-geometry audit on $(task_name) ...")
            task_result = run_task(task_name, control_repeats)
            task_summary = summarize_task(task_result)
            comp = task_summary["comparison"]
            stats = task_summary["atlas_stats"]
            active = get(stats, "n_active_regimes", 0)
            stable_delta = round(get(comp, "stable_vs_pooled_gap_mean_delta", 0.0), digits=4)
            std_impr = round(get(comp, "stable_vs_pooled_gap_std_improvement", 0.0), digits=4)
            println("  active_regimes=$(active) | stable_gap_delta=$(stable_delta) | gap_std_improvement=$(std_impr) | rec=$(task_summary["recommendation"])")
            results_by_task[task_name] = Dict(
                "result" => task_result,
                "summary" => task_summary,
            )
        end
    end

    decisions = Dict{String,Any}(
        "celecoxib_dataset_size" => get(cele_stats, "size", 0),
        "celecoxib_override_eligible_fraction" => get(cele_stats, "override_eligible_fraction", 0.0),
        "celecoxib_stable_intervention_fraction" => get(cele_stats, "stable_intervention_fraction", 0.0),
        "celecoxib_ambiguous_positive_fraction" => get(cele_stats, "ambiguous_positive_fraction", 0.0),
        "celecoxib_active_regimes" => cele_active,
        "celecoxib_regime_counts" => get(cele_stats, "regime_counts", Dict{String,Int}()),
        "celecoxib_stable_vs_pooled_gap_mean_delta" => get(cele_comp, "stable_vs_pooled_gap_mean_delta", 0.0),
        "celecoxib_stable_vs_pooled_gap_std_improvement" => get(cele_comp, "stable_vs_pooled_gap_std_improvement", 0.0),
        "celecoxib_pooled_positive_ambiguous_fraction" => get(cele_comp, "pooled_positive_ambiguous_fraction", 0.0),
        "celecoxib_mixture_explains_instability" => get(cele_comp, "mixture_explains_instability", false),
        "celecoxib_compact_taxonomy" => get(cele_comp, "compact_taxonomy", false),
        "celecoxib_interpretable" => cele_interpretable,
        "global_recommendation" => get(celecoxib_summary, "recommendation", "NO_INTERVENTION_GEOMETRY"),
    )

    global_rec = decisions["global_recommendation"]
    println("\nGlobal recommendation: $(global_rec) | active_regimes=$(cele_active) | stable_gap_delta=$(cele_stable_delta) | gap_std_improvement=$(cele_std_impr) | pooled_positive_ambiguous_fraction=$(cele_ambig)")
    return Dict("results_by_task" => results_by_task, "decisions" => decisions)
end


function run_c20_sparse_positive_operating_point_audit(tasks::Vector{String};
    budget::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=16,
    num_splits::Int=5,
    parent_candidate_limit::Int=4,
    max_basin_contexts::Int=2,
    option_horizon::Int=3,
    control_repeats::Int=3,
    max_allocation_budget::Int=2,
    train_fraction::Float64=0.75,
    fixed_thresholds::Vector{Float64}=[0.45, 0.50, 0.55, 0.60],
    max_positive_fraction::Float64=0.35,
    min_positive_count::Int=1,
    min_train_precision::Float64=0.75,
    state_threshold::Float64=0.01,
)
    println("\n" * "="^80)
    println("RUNNING BATCH 1S — SPARSE-POSITIVE OPERATING-POINT AUDIT")
    println("="^80)

    function collect_probe_run(task_name::String, repeat_idx::Int)
        oracle_mgr, budget_oracle, budget_oracle_batch = build_budget_oracles(task_name, budget)
        frontier_buffer = MolecularFrontierBuffer(5000)
        vocab = SMILESVocabulary()
        target_smiles = get(TARGET_SMILES, task_name, nothing)
        seed_pool = bootstrap_seed_pool(task_name;
            user_seed_smiles=String[],
            target_smiles=target_smiles,
            target_seed=target_seed)
        seed_stats = _seed_memories!(seed_pool, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            frontier_buffer=frontier_buffer,
            augmentation_count=bootstrap_augmentation_count(task_name, target_smiles, target_seed; enable_augmentation=true),
            verbose=true)
        warmup_stats = bootstrap_frontier_warmup!(task_name, frontier_buffer, oracle_mgr, budget_oracle_batch, vocab;
            target_smiles=target_smiles,
            rounds=bootstrap_warmup_rounds,
            verbose=true)

        cfg = HierarchicalEditConfig(
            max_step_attempts=max_step_attempts,
            max_operator_candidates=max_operator_candidates,
            parent_candidate_limit=parent_candidate_limit,
            basin_candidate_limit=max_basin_contexts,
        )
        probe = probe_frontier_allocation_opportunities(frontier_buffer, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            config=cfg,
            target_smiles=target_smiles,
            budget_remaining=budget_remaining(oracle_mgr),
            created_at_step=repeat_idx,
            task_name=task_name,
            max_parents=parent_candidate_limit,
            max_basins=max_basin_contexts,
            horizon=option_horizon,
            region_families=["basin"],
            max_allocation_budget=max_allocation_budget)
        return Dict(
            "probe" => probe,
            "seed_stats" => seed_stats,
            "warmup_stats" => warmup_stats,
            "calls_used" => oracle_mgr.calls_used,
        )
    end

    function run_task(task_name::String, repeats::Int)
        runs = [collect_probe_run(task_name, idx) for idx in 1:repeats]
        dataset = extract_opportunity_state_dataset(runs;
            family_name="basin",
            state_threshold=state_threshold)
        audit = evaluate_sparse_positive_operating_points(dataset;
            rng=MersenneTwister(0),
            num_splits=num_splits,
            train_fraction=train_fraction,
            family_name="basin",
            rule_families=[:precision_guarded, :fraction_capped, :guarded_fallback],
            fixed_thresholds=fixed_thresholds,
            max_positive_fraction=max_positive_fraction,
            min_positive_count=min_positive_count,
            min_train_precision=min_train_precision)
        return Dict(
            "runs" => runs,
            "dataset" => dataset,
            "dataset_stats" => opportunity_state_dataset_stats(dataset),
            "audit" => audit,
        )
    end

    function summarize_task(task_result::Dict{String,Any})
        audit = get(task_result, "audit", Dict{String,Any}())
        dataset_stats = get(task_result, "dataset_stats", Dict{String,Any}())
        summary = get(audit, "summary", Dict{String,Any}())
        rule_summaries = get(audit, "rule_summaries", Dict{String,Any}())
        best_rule = String(get(summary, "best_rule", "none"))
        best_rule_summary = get(rule_summaries, best_rule, Dict{String,Any}())
        interpretable = Bool(get(best_rule_summary, "promotion_safe", false))
        recommendation = String(get(summary, "recommendation", "NO_SPARSE_POSITIVE_OPERATING_POINT"))
        return Dict(
            "dataset_stats" => dataset_stats,
            "audit_summary" => summary,
            "rule_summaries" => rule_summaries,
            "best_rule_summary" => best_rule_summary,
            "interpretable" => interpretable,
            "recommendation" => recommendation,
        )
    end

    celecoxib_task = "celecoxib_rediscovery"
    celecoxib_task in tasks || error("Batch 1S requires celecoxib_rediscovery in PMO_TASKS")

    println("\nRunning celecoxib-first Batch 1S sparse-positive operating-point audit ...")
    celecoxib_result = run_task(celecoxib_task, data_repeats)
    celecoxib_summary = summarize_task(celecoxib_result)
    cele_best = celecoxib_summary["best_rule_summary"]
    best_rule = get(celecoxib_summary["audit_summary"], "best_rule", "none")
    cele_nondeg = round(get(cele_best, "nondegenerate_fraction", 0.0), digits=4)
    cele_pos = round(get(cele_best, "median_predicted_positive_fraction", 0.0), digits=4)
    cele_gap = round(get(cele_best, "median_gap_separation", 0.0), digits=4)
    cele_gain = round(get(cele_best, "median_conditional_gain", 0.0), digits=4)
    println("  best_rule=$(best_rule) | nondeg=$(cele_nondeg) | median_pos=$(cele_pos) | median_gap=$(cele_gap) | median_gain=$(cele_gain)")

    results_by_task = Dict{String,Any}(
        celecoxib_task => Dict(
            "result" => celecoxib_result,
            "summary" => celecoxib_summary,
        )
    )

    cele_interpretable = get(celecoxib_summary, "interpretable", false)
    if cele_interpretable
        for task_name in ["drd2", "albuterol_similarity"]
            task_name in tasks || continue
            println("\nRunning control sparse-positive audit on $(task_name) ...")
            task_result = run_task(task_name, control_repeats)
            task_summary = summarize_task(task_result)
            best_rule = get(task_summary["audit_summary"], "best_rule", "none")
            best = task_summary["best_rule_summary"]
            control_nondeg = round(get(best, "nondegenerate_fraction", 0.0), digits=4)
            control_pos = round(get(best, "median_predicted_positive_fraction", 0.0), digits=4)
            control_gain = round(get(best, "median_conditional_gain", 0.0), digits=4)
            println("  best_rule=$(best_rule) | nondeg=$(control_nondeg) | median_pos=$(control_pos) | median_gain=$(control_gain) | rec=$(task_summary["recommendation"])")
            results_by_task[task_name] = Dict(
                "result" => task_result,
                "summary" => task_summary,
            )
        end
    end

    decisions = Dict{String,Any}(
        "celecoxib_dataset_size" => get(celecoxib_summary["dataset_stats"], "size", 0),
        "celecoxib_override_eligible_fraction" => get(celecoxib_summary["dataset_stats"], "override_eligible_fraction", 0.0),
        "celecoxib_best_rule" => get(celecoxib_summary["audit_summary"], "best_rule", "none"),
        "celecoxib_best_rule_summary" => cele_best,
        "celecoxib_interpretable" => cele_interpretable,
        "global_recommendation" => get(celecoxib_summary, "recommendation", "NO_SPARSE_POSITIVE_OPERATING_POINT"),
    )

    global_rec = decisions["global_recommendation"]
    println("\nGlobal recommendation: $(global_rec) | best_rule=$(best_rule) | nondeg=$(cele_nondeg) | median_pos=$(cele_pos) | median_gap=$(cele_gap) | median_gain=$(cele_gain)")
    return Dict("results_by_task" => results_by_task, "decisions" => decisions)
end


function run_c21_representation_semantics_repair_audit(tasks::Vector{String};
    budget::Int,
    target_seed::Bool,
    bootstrap_warmup_rounds::Int,
    max_step_attempts::Int,
    max_operator_candidates::Int,
    data_repeats::Int=16,
    num_splits::Int=5,
    parent_candidate_limit::Int=4,
    max_basin_contexts::Int=2,
    option_horizon::Int=3,
    control_repeats::Int=3,
    max_allocation_budget::Int=2,
    train_fraction::Float64=0.75,
    state_threshold::Float64=0.01,
    stability_threshold::Float64=0.05,
)
    println("\n" * "="^80)
    println("RUNNING BATCH 1T — REPRESENTATION / SEMANTICS REPAIR AUDIT")
    println("="^80)

    function collect_probe_run(task_name::String, repeat_idx::Int)
        oracle_mgr, budget_oracle, budget_oracle_batch = build_budget_oracles(task_name, budget)
        frontier_buffer = MolecularFrontierBuffer(5000)
        vocab = SMILESVocabulary()
        target_smiles = get(TARGET_SMILES, task_name, nothing)
        seed_pool = bootstrap_seed_pool(task_name;
            user_seed_smiles=String[],
            target_smiles=target_smiles,
            target_seed=target_seed)
        seed_stats = _seed_memories!(seed_pool, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            frontier_buffer=frontier_buffer,
            augmentation_count=bootstrap_augmentation_count(task_name, target_smiles, target_seed; enable_augmentation=true),
            verbose=true)
        warmup_stats = bootstrap_frontier_warmup!(task_name, frontier_buffer, oracle_mgr, budget_oracle_batch, vocab;
            target_smiles=target_smiles,
            rounds=bootstrap_warmup_rounds,
            verbose=true)

        cfg = HierarchicalEditConfig(
            max_step_attempts=max_step_attempts,
            max_operator_candidates=max_operator_candidates,
            parent_candidate_limit=parent_candidate_limit,
            basin_candidate_limit=max_basin_contexts,
        )
        probe = probe_frontier_allocation_opportunities(frontier_buffer, budget_oracle, vocab;
            reward_fn_batch=budget_oracle_batch,
            config=cfg,
            target_smiles=target_smiles,
            budget_remaining=budget_remaining(oracle_mgr),
            created_at_step=repeat_idx,
            task_name=task_name,
            max_parents=parent_candidate_limit,
            max_basins=max_basin_contexts,
            horizon=option_horizon,
            region_families=["basin"],
            max_allocation_budget=max_allocation_budget)
        return Dict(
            "probe" => probe,
            "seed_stats" => seed_stats,
            "warmup_stats" => warmup_stats,
            "calls_used" => oracle_mgr.calls_used,
            "budget_total" => budget,
            "created_at_step" => repeat_idx,
        )
    end

    function run_task(task_name::String, repeats::Int)
        runs = [collect_probe_run(task_name, idx) for idx in 1:repeats]
        dataset = extract_opportunity_repair_audit_dataset(runs;
            family_name="basin",
            state_threshold=state_threshold,
            stability_threshold=stability_threshold)
        audit = evaluate_opportunity_representation_semantics_repair(dataset;
            rng=MersenneTwister(0),
            num_splits=num_splits,
            train_fraction=train_fraction)
        return Dict(
            "runs" => runs,
            "dataset" => dataset,
            "dataset_stats" => opportunity_repair_audit_dataset_stats(dataset),
            "audit" => audit,
        )
    end

    function summarize_task(task_result::Dict{String,Any})
        audit = get(task_result, "audit", Dict{String,Any}())
        dataset_stats = get(task_result, "dataset_stats", Dict{String,Any}())
        summary = get(audit, "summary", Dict{String,Any}())
        verdict = String(get(summary, "verdict", "V5_NO_DECISIVE_REPAIR_SIGNAL"))
        decisive = Bool(get(summary, "decisive", false))
        return Dict(
            "dataset_stats" => dataset_stats,
            "audit_summary" => summary,
            "decisive" => decisive,
            "verdict" => verdict,
        )
    end

    celecoxib_task = "celecoxib_rediscovery"
    celecoxib_task in tasks || error("Batch 1T requires celecoxib_rediscovery in PMO_TASKS")

    println("\nRunning celecoxib-first Batch 1T repair audit ...")
    celecoxib_result = run_task(celecoxib_task, data_repeats)
    celecoxib_summary = summarize_task(celecoxib_result)
    cele_audit = celecoxib_summary["audit_summary"]
    println("  verdict=$(celecoxib_summary["verdict"]) | decisive=$(celecoxib_summary["decisive"]) | rep_branch=$(get(cele_audit, "best_representation_branch", "baseline")) | rep_metric=$(round(get(cele_audit, "best_representation_metric", 0.0), digits=4)) | robust_metric=$(round(get(cele_audit, "robust_best_metric", 0.0), digits=4)) | ordinal_metric=$(round(get(cele_audit, "ordinal_best_metric", 0.0), digits=4))")

    results_by_task = Dict{String,Any}(
        celecoxib_task => Dict(
            "result" => celecoxib_result,
            "summary" => celecoxib_summary,
        )
    )

    if celecoxib_summary["decisive"]
        for task_name in ["drd2", "albuterol_similarity"]
            task_name in tasks || continue
            println("\nRunning control representation / semantics audit on $(task_name) ...")
            task_result = run_task(task_name, control_repeats)
            task_summary = summarize_task(task_result)
            task_audit = task_summary["audit_summary"]
            println("  verdict=$(task_summary["verdict"]) | decisive=$(task_summary["decisive"]) | rep_metric=$(round(get(task_audit, "best_representation_metric", 0.0), digits=4)) | robust_metric=$(round(get(task_audit, "robust_best_metric", 0.0), digits=4)) | ordinal_metric=$(round(get(task_audit, "ordinal_best_metric", 0.0), digits=4))")
            results_by_task[task_name] = Dict(
                "result" => task_result,
                "summary" => task_summary,
            )
        end
    end

    decisions = Dict{String,Any}(
        "celecoxib_dataset_size" => get(celecoxib_summary["dataset_stats"], "size", 0),
        "celecoxib_current_positive_fraction" => get(celecoxib_summary["dataset_stats"], "current_positive_fraction", 0.0),
        "celecoxib_robust_positive_fraction" => get(celecoxib_summary["dataset_stats"], "robust_positive_fraction", 0.0),
        "celecoxib_verdict" => celecoxib_summary["verdict"],
        "celecoxib_decisive" => celecoxib_summary["decisive"],
        "celecoxib_audit_summary" => cele_audit,
        "global_recommendation" => celecoxib_summary["verdict"],
    )

    println("\nGlobal recommendation: $(decisions["global_recommendation"]) | decisive=$(decisions["celecoxib_decisive"]) | strongest=$(round(get(cele_audit, "strongest_metric", 0.0), digits=4))")
    return Dict("results_by_task" => results_by_task, "decisions" => decisions)
end

const TASKS = parse_list_env("PMO_TASKS", DEFAULT_TASKS)
const BUDGET = parse(Int, get(ENV, "PMO_BUDGET", "256"))
const N_EPISODES = parse(Int, get(ENV, "HE_EPISODES", "48"))
const TARGET_SEED = parse_bool_env("HE_TARGET_SEED", true)
const COMPARE_TOKEN = parse_bool_env("COMPARE_TOKEN", false)
const BOOTSTRAP_WARMUP_ROUNDS = parse(Int, get(ENV, "HE_BOOTSTRAP_WARMUP_ROUNDS", "1"))
const MAX_STEP_ATTEMPTS = parse(Int, get(ENV, "HE_MAX_STEP_ATTEMPTS", "3"))
const MAX_OPERATOR_CANDIDATES = parse(Int, get(ENV, "HE_MAX_OPERATOR_CANDIDATES", "8"))
const ABLATION_MATRIX = parse_bool_env("HE_ABLATION_MATRIX", false)
const OPERATOR_COMPARISON = parse_bool_env("HE_OPERATOR_COMPARISON", true)
const A1_ZERO_CHECKS = parse_bool_env("HE_A10_CHECKS", false)
const A1_ZERO_REPEATS = parse(Int, get(ENV, "HE_A10_REPEATS", "3"))
const A1_ONE_BRIDGE = parse_bool_env("HE_A11_BRIDGE", false)
const A1_ONE_REPEATS = parse(Int, get(ENV, "HE_A11_REPEATS", "3"))
const A1_TWO_CONFIRM = parse_bool_env("HE_A12_CONFIRM", false)
const A1_TWO_CELECOXIB_REPEATS = parse(Int, get(ENV, "HE_A12_CELECOXIB_REPEATS", "5"))
const A1_TWO_CONTROL_REPEATS = parse(Int, get(ENV, "HE_A12_CONTROL_REPEATS", "3"))
const A1_TWO_SANITY_REPEATS = parse(Int, get(ENV, "HE_A12_SANITY_REPEATS", "2"))
const A1_TWO_CONTROL_ANCHOR = parse_bool_env("HE_A12_CONTROL_ANCHOR", true)
const C2_BASIN_CHECKS = parse_bool_env("HE_C2_BASIN", false)
const C2_DATA_REPEATS = parse(Int, get(ENV, "HE_C2_DATA_REPEATS", "3"))
const C2_EVAL_REPEATS = parse(Int, get(ENV, "HE_C2_EVAL_REPEATS", "2"))
const C2_TRAIN_EPOCHS = parse(Int, get(ENV, "HE_C2_TRAIN_EPOCHS", "20"))
const C2_MAX_PROMOTED = parse(Int, get(ENV, "HE_C2_MAX_PROMOTED", "2"))
const C2_CELECOXIB_REPEATS = parse(Int, get(ENV, "HE_C2_CELECOXIB_REPEATS", "5"))
const C2_CONTROL_REPEATS = parse(Int, get(ENV, "HE_C2_CONTROL_REPEATS", "3"))
const C2_SANITY_REPEATS = parse(Int, get(ENV, "HE_C2_SANITY_REPEATS", "3"))
const C3_PARENT_CHECKS = parse_bool_env("HE_C3_PARENT", false)
const C3_DATA_REPEATS = parse(Int, get(ENV, "HE_C3_DATA_REPEATS", "2"))
const C3_TRAIN_EPOCHS = parse(Int, get(ENV, "HE_C3_TRAIN_EPOCHS", "20"))
const C3_PARENT_CANDIDATE_LIMIT = parse(Int, get(ENV, "HE_C3_PARENT_CANDIDATE_LIMIT", "16"))
const C3_CELECOXIB_REPEATS = parse(Int, get(ENV, "HE_C3_CELECOXIB_REPEATS", "5"))
const C3_CONTROL_REPEATS = parse(Int, get(ENV, "HE_C3_CONTROL_REPEATS", "3"))
const C3_SANITY_REPEATS = parse(Int, get(ENV, "HE_C3_SANITY_REPEATS", "2"))
const C4_PARENT_SEMANTICS = parse_bool_env("HE_C4_PARENT", false)
const C4_DATA_REPEATS = parse(Int, get(ENV, "HE_C4_DATA_REPEATS", "2"))
const C4_TRAIN_EPOCHS = parse(Int, get(ENV, "HE_C4_TRAIN_EPOCHS", "20"))
const C4_PARENT_CANDIDATE_LIMIT = parse(Int, get(ENV, "HE_C4_PARENT_CANDIDATE_LIMIT", "16"))
const C4_CELECOXIB_REPEATS = parse(Int, get(ENV, "HE_C4_CELECOXIB_REPEATS", "5"))
const C4_CONTROL_REPEATS = parse(Int, get(ENV, "HE_C4_CONTROL_REPEATS", "3"))
const C4_SANITY_REPEATS = parse(Int, get(ENV, "HE_C4_SANITY_REPEATS", "2"))
const C5_PARENT_CAUSALITY = parse_bool_env("HE_C5_PARENT", false)
const C5_DATA_REPEATS = parse(Int, get(ENV, "HE_C5_DATA_REPEATS", "3"))
const C5_PARENT_CANDIDATE_LIMIT = parse(Int, get(ENV, "HE_C5_PARENT_CANDIDATE_LIMIT", "4"))
const C6_PARENT_OPERATOR = parse_bool_env("HE_C6_PARENT_OPERATOR", false)
const C6_DATA_REPEATS = parse(Int, get(ENV, "HE_C6_DATA_REPEATS", "5"))
const C6_PARENT_CANDIDATE_LIMIT = parse(Int, get(ENV, "HE_C6_PARENT_CANDIDATE_LIMIT", "4"))
const C6_MAX_BASIN_CONTEXTS = parse(Int, get(ENV, "HE_C6_MAX_BASIN_CONTEXTS", "2"))
const C7_BASIN_OPERATOR = parse_bool_env("HE_C7_BASIN_OPERATOR", false)
const C7_DATA_REPEATS = parse(Int, get(ENV, "HE_C7_DATA_REPEATS", "5"))
const C7_PARENT_CANDIDATE_LIMIT = parse(Int, get(ENV, "HE_C7_PARENT_CANDIDATE_LIMIT", "4"))
const C7_MAX_BASIN_CONTEXTS = parse(Int, get(ENV, "HE_C7_MAX_BASIN_CONTEXTS", "2"))
const C8_OPERATOR_CONTROLLER = parse_bool_env("HE_C8_OPERATOR", false)
const C8_DATA_REPEATS = parse(Int, get(ENV, "HE_C8_DATA_REPEATS", "3"))
const C8_TRAIN_EPOCHS = parse(Int, get(ENV, "HE_C8_TRAIN_EPOCHS", "20"))
const C8_CELECOXIB_REPEATS = parse(Int, get(ENV, "HE_C8_CELECOXIB_REPEATS", "5"))
const C8_CONTROL_REPEATS = parse(Int, get(ENV, "HE_C8_CONTROL_REPEATS", "3"))
const C8_SANITY_REPEATS = parse(Int, get(ENV, "HE_C8_SANITY_REPEATS", "3"))
const C9_OPERATOR_ELIGIBILITY = parse_bool_env("HE_C9_OPERATOR_ELIGIBILITY", false)
const C9_DATA_REPEATS = parse(Int, get(ENV, "HE_C9_DATA_REPEATS", "3"))
const C9_TRAIN_EPOCHS = parse(Int, get(ENV, "HE_C9_TRAIN_EPOCHS", "20"))
const C9_CELECOXIB_REPEATS = parse(Int, get(ENV, "HE_C9_CELECOXIB_REPEATS", "5"))
const C9_CONTROL_REPEATS = parse(Int, get(ENV, "HE_C9_CONTROL_REPEATS", "3"))
const C9_SANITY_REPEATS = parse(Int, get(ENV, "HE_C9_SANITY_REPEATS", "3"))
const C10_COUPLED_OPTION = parse_bool_env("HE_C10_COUPLED_OPTION", false)
const C10_DATA_REPEATS = parse(Int, get(ENV, "HE_C10_DATA_REPEATS", "5"))
const C10_PARENT_CANDIDATE_LIMIT = parse(Int, get(ENV, "HE_C10_PARENT_CANDIDATE_LIMIT", "4"))
const C10_MAX_BASIN_CONTEXTS = parse(Int, get(ENV, "HE_C10_MAX_BASIN_CONTEXTS", "2"))
const C10_OPTION_HORIZON = parse(Int, get(ENV, "HE_C10_OPTION_HORIZON", "3"))
const C11_SUBTRAJECTORY = parse_bool_env("HE_C11_SUBTRAJECTORY", false)
const C11_DATA_REPEATS = parse(Int, get(ENV, "HE_C11_DATA_REPEATS", "5"))
const C11_PARENT_CANDIDATE_LIMIT = parse(Int, get(ENV, "HE_C11_PARENT_CANDIDATE_LIMIT", "4"))
const C11_MAX_BASIN_CONTEXTS = parse(Int, get(ENV, "HE_C11_MAX_BASIN_CONTEXTS", "2"))
const C11_OPTION_HORIZON = parse(Int, get(ENV, "HE_C11_OPTION_HORIZON", "3"))
const C12_OPTION_VALUE = parse_bool_env("HE_C12_OPTION_VALUE", false)
const C12_DATA_REPEATS = parse(Int, get(ENV, "HE_C12_DATA_REPEATS", "5"))
const C12_TRAIN_EPOCHS = parse(Int, get(ENV, "HE_C12_TRAIN_EPOCHS", "30"))
const C12_PARENT_CANDIDATE_LIMIT = parse(Int, get(ENV, "HE_C12_PARENT_CANDIDATE_LIMIT", "4"))
const C12_MAX_BASIN_CONTEXTS = parse(Int, get(ENV, "HE_C12_MAX_BASIN_CONTEXTS", "2"))
const C12_OPTION_HORIZON = parse(Int, get(ENV, "HE_C12_OPTION_HORIZON", "3"))
const C12_CELECOXIB_REPEATS = parse(Int, get(ENV, "HE_C12_CELECOXIB_REPEATS", "5"))
const C12_CONTROL_REPEATS = parse(Int, get(ENV, "HE_C12_CONTROL_REPEATS", "3"))
const C13_OPTION_REFINEMENT = parse_bool_env("HE_C13_OPTION_REFINEMENT", false)
const C13_DATA_REPEATS = parse(Int, get(ENV, "HE_C13_DATA_REPEATS", "5"))
const C13_TRAIN_EPOCHS = parse(Int, get(ENV, "HE_C13_TRAIN_EPOCHS", "30"))
const C13_PARENT_CANDIDATE_LIMIT = parse(Int, get(ENV, "HE_C13_PARENT_CANDIDATE_LIMIT", "4"))
const C13_MAX_BASIN_CONTEXTS = parse(Int, get(ENV, "HE_C13_MAX_BASIN_CONTEXTS", "2"))
const C13_OPTION_HORIZON = parse(Int, get(ENV, "HE_C13_OPTION_HORIZON", "3"))
const C13_CELECOXIB_REPEATS = parse(Int, get(ENV, "HE_C13_CELECOXIB_REPEATS", "5"))
const C13_CONTROL_REPEATS = parse(Int, get(ENV, "HE_C13_CONTROL_REPEATS", "3"))
const C14_CALIBRATED_ORDINAL = parse_bool_env("HE_C14_CALIBRATED_ORDINAL", false)
const C14_DATA_REPEATS = parse(Int, get(ENV, "HE_C14_DATA_REPEATS", "5"))
const C14_TRAIN_EPOCHS = parse(Int, get(ENV, "HE_C14_TRAIN_EPOCHS", "30"))
const C14_CALIBRATION_EPOCHS = parse(Int, get(ENV, "HE_C14_CALIBRATION_EPOCHS", "50"))
const C14_PARENT_CANDIDATE_LIMIT = parse(Int, get(ENV, "HE_C14_PARENT_CANDIDATE_LIMIT", "4"))
const C14_MAX_BASIN_CONTEXTS = parse(Int, get(ENV, "HE_C14_MAX_BASIN_CONTEXTS", "2"))
const C14_OPTION_HORIZON = parse(Int, get(ENV, "HE_C14_OPTION_HORIZON", "3"))
const C14_CELECOXIB_REPEATS = parse(Int, get(ENV, "HE_C14_CELECOXIB_REPEATS", "5"))
const C14_CONTROL_REPEATS = parse(Int, get(ENV, "HE_C14_CONTROL_REPEATS", "3"))
const C15_FRONTIER_ALLOCATION = parse_bool_env("HE_C15_FRONTIER_ALLOCATION", false)
const C15_DATA_REPEATS = parse(Int, get(ENV, "HE_C15_DATA_REPEATS", "5"))
const C15_PARENT_CANDIDATE_LIMIT = parse(Int, get(ENV, "HE_C15_PARENT_CANDIDATE_LIMIT", "4"))
const C15_MAX_BASIN_CONTEXTS = parse(Int, get(ENV, "HE_C15_MAX_BASIN_CONTEXTS", "2"))
const C15_OPTION_HORIZON = parse(Int, get(ENV, "HE_C15_OPTION_HORIZON", "3"))
const C15_CONTROL_REPEATS = parse(Int, get(ENV, "HE_C15_CONTROL_REPEATS", "3"))
const C15_REGION_FAMILIES = parse_list_env("HE_C15_REGION_FAMILIES", ["basin", "parent_novelty", "continuation"])
const C15_MAX_ALLOCATION_BUDGET = parse(Int, get(ENV, "HE_C15_MAX_ALLOCATION_BUDGET", "2"))
const C16_SELECTIVE_ALLOCATOR = parse_bool_env("HE_C16_SELECTIVE_ALLOCATOR", false)
const C16_DATA_REPEATS = parse(Int, get(ENV, "HE_C16_DATA_REPEATS", "8"))
const C16_PARENT_CANDIDATE_LIMIT = parse(Int, get(ENV, "HE_C16_PARENT_CANDIDATE_LIMIT", "4"))
const C16_MAX_BASIN_CONTEXTS = parse(Int, get(ENV, "HE_C16_MAX_BASIN_CONTEXTS", "2"))
const C16_OPTION_HORIZON = parse(Int, get(ENV, "HE_C16_OPTION_HORIZON", "3"))
const C16_CONTROL_REPEATS = parse(Int, get(ENV, "HE_C16_CONTROL_REPEATS", "3"))
const C16_MAX_ALLOCATION_BUDGET = parse(Int, get(ENV, "HE_C16_MAX_ALLOCATION_BUDGET", "2"))
const C16_OVERRIDE_THRESHOLD = parse(Float64, get(ENV, "HE_C16_OVERRIDE_THRESHOLD", "0.01"))
const C16_TRAIN_FRACTION = parse(Float64, get(ENV, "HE_C16_TRAIN_FRACTION", "0.75"))
const C17_OPPORTUNITY_STATE = parse_bool_env("HE_C17_OPPORTUNITY_STATE", false)
const C17_DATA_REPEATS = parse(Int, get(ENV, "HE_C17_DATA_REPEATS", "10"))
const C17_PARENT_CANDIDATE_LIMIT = parse(Int, get(ENV, "HE_C17_PARENT_CANDIDATE_LIMIT", "4"))
const C17_MAX_BASIN_CONTEXTS = parse(Int, get(ENV, "HE_C17_MAX_BASIN_CONTEXTS", "2"))
const C17_OPTION_HORIZON = parse(Int, get(ENV, "HE_C17_OPTION_HORIZON", "3"))
const C17_CONTROL_REPEATS = parse(Int, get(ENV, "HE_C17_CONTROL_REPEATS", "3"))
const C17_MAX_ALLOCATION_BUDGET = parse(Int, get(ENV, "HE_C17_MAX_ALLOCATION_BUDGET", "2"))
const C17_STATE_THRESHOLD = parse(Float64, get(ENV, "HE_C17_STATE_THRESHOLD", "0.01"))
const C17_TRAIN_FRACTION = parse(Float64, get(ENV, "HE_C17_TRAIN_FRACTION", "0.75"))
const C18_OPPORTUNITY_REPEATABILITY = parse_bool_env("HE_C18_OPPORTUNITY_REPEATABILITY", false)
const C18_DATA_REPEATS = parse(Int, get(ENV, "HE_C18_DATA_REPEATS", "16"))
const C18_NUM_SPLITS = parse(Int, get(ENV, "HE_C18_NUM_SPLITS", "5"))
const C18_PARENT_CANDIDATE_LIMIT = parse(Int, get(ENV, "HE_C18_PARENT_CANDIDATE_LIMIT", "4"))
const C18_MAX_BASIN_CONTEXTS = parse(Int, get(ENV, "HE_C18_MAX_BASIN_CONTEXTS", "2"))
const C18_OPTION_HORIZON = parse(Int, get(ENV, "HE_C18_OPTION_HORIZON", "3"))
const C18_CONTROL_REPEATS = parse(Int, get(ENV, "HE_C18_CONTROL_REPEATS", "3"))
const C18_MAX_ALLOCATION_BUDGET = parse(Int, get(ENV, "HE_C18_MAX_ALLOCATION_BUDGET", "2"))
const C18_STATE_THRESHOLD = parse(Float64, get(ENV, "HE_C18_STATE_THRESHOLD", "0.01"))
const C18_TRAIN_FRACTION = parse(Float64, get(ENV, "HE_C18_TRAIN_FRACTION", "0.75"))
const C18_THRESHOLD_PERTURBATIONS = parse_float_list_env("HE_C18_THRESHOLD_PERTURBATIONS", [-0.05, 0.0, 0.05])
const C19_INTERVENTION_GEOMETRY = parse_bool_env("HE_C19_INTERVENTION_GEOMETRY", false)
const C19_DATA_REPEATS = parse(Int, get(ENV, "HE_C19_DATA_REPEATS", "16"))
const C19_PARENT_CANDIDATE_LIMIT = parse(Int, get(ENV, "HE_C19_PARENT_CANDIDATE_LIMIT", "4"))
const C19_MAX_BASIN_CONTEXTS = parse(Int, get(ENV, "HE_C19_MAX_BASIN_CONTEXTS", "2"))
const C19_OPTION_HORIZON = parse(Int, get(ENV, "HE_C19_OPTION_HORIZON", "3"))
const C19_CONTROL_REPEATS = parse(Int, get(ENV, "HE_C19_CONTROL_REPEATS", "3"))
const C19_MAX_ALLOCATION_BUDGET = parse(Int, get(ENV, "HE_C19_MAX_ALLOCATION_BUDGET", "2"))
const C19_STATE_THRESHOLD = parse(Float64, get(ENV, "HE_C19_STATE_THRESHOLD", "0.01"))
const C19_STABILITY_THRESHOLD = parse(Float64, get(ENV, "HE_C19_STABILITY_THRESHOLD", "0.05"))

const C21_REPRESENTATION_SEMANTICS_AUDIT = parse_bool_env("HE_C21_REPRESENTATION_SEMANTICS_AUDIT", false)
const C21_DATA_REPEATS = parse(Int, get(ENV, "HE_C21_DATA_REPEATS", "16"))
const C21_NUM_SPLITS = parse(Int, get(ENV, "HE_C21_NUM_SPLITS", "5"))
const C21_PARENT_CANDIDATE_LIMIT = parse(Int, get(ENV, "HE_C21_PARENT_CANDIDATE_LIMIT", "4"))
const C21_MAX_BASIN_CONTEXTS = parse(Int, get(ENV, "HE_C21_MAX_BASIN_CONTEXTS", "2"))
const C21_OPTION_HORIZON = parse(Int, get(ENV, "HE_C21_OPTION_HORIZON", "3"))
const C21_CONTROL_REPEATS = parse(Int, get(ENV, "HE_C21_CONTROL_REPEATS", "3"))
const C21_MAX_ALLOCATION_BUDGET = parse(Int, get(ENV, "HE_C21_MAX_ALLOCATION_BUDGET", "2"))
const C21_TRAIN_FRACTION = parse(Float64, get(ENV, "HE_C21_TRAIN_FRACTION", "0.75"))
const C21_STATE_THRESHOLD = parse(Float64, get(ENV, "HE_C21_STATE_THRESHOLD", "0.01"))
const C21_STABILITY_THRESHOLD = parse(Float64, get(ENV, "HE_C21_STABILITY_THRESHOLD", "0.05"))
const C20_SPARSE_POSITIVE_AUDIT = parse_bool_env("HE_C20_SPARSE_POSITIVE_AUDIT", false)
const C20_DATA_REPEATS = parse(Int, get(ENV, "HE_C20_DATA_REPEATS", "16"))
const C20_NUM_SPLITS = parse(Int, get(ENV, "HE_C20_NUM_SPLITS", "5"))
const C20_PARENT_CANDIDATE_LIMIT = parse(Int, get(ENV, "HE_C20_PARENT_CANDIDATE_LIMIT", "4"))
const C20_MAX_BASIN_CONTEXTS = parse(Int, get(ENV, "HE_C20_MAX_BASIN_CONTEXTS", "2"))
const C20_OPTION_HORIZON = parse(Int, get(ENV, "HE_C20_OPTION_HORIZON", "3"))
const C20_CONTROL_REPEATS = parse(Int, get(ENV, "HE_C20_CONTROL_REPEATS", "3"))
const C20_MAX_ALLOCATION_BUDGET = parse(Int, get(ENV, "HE_C20_MAX_ALLOCATION_BUDGET", "2"))
const C20_TRAIN_FRACTION = parse(Float64, get(ENV, "HE_C20_TRAIN_FRACTION", "0.75"))
const C20_FIXED_THRESHOLDS = parse_float_list_env("HE_C20_FIXED_THRESHOLDS", [0.45, 0.50, 0.55, 0.60])
const C20_MAX_POSITIVE_FRACTION = parse(Float64, get(ENV, "HE_C20_MAX_POSITIVE_FRACTION", "0.35"))
const C20_MIN_POSITIVE_COUNT = parse(Int, get(ENV, "HE_C20_MIN_POSITIVE_COUNT", "1"))
const C20_MIN_TRAIN_PRECISION = parse(Float64, get(ENV, "HE_C20_MIN_TRAIN_PRECISION", "0.75"))
const C20_STATE_THRESHOLD = parse(Float64, get(ENV, "HE_C20_STATE_THRESHOLD", "0.01"))

all_results = if A1_TWO_CONFIRM
    run_a12_confirm_checks(TASKS;
        budget=BUDGET,
        n_episodes=N_EPISODES,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        celecoxib_repeats=A1_TWO_CELECOXIB_REPEATS,
        control_repeats=A1_TWO_CONTROL_REPEATS,
        sanity_repeats=A1_TWO_SANITY_REPEATS,
        include_control_anchor=A1_TWO_CONTROL_ANCHOR,
    )
elseif A1_ONE_BRIDGE
    run_a11_bridge_checks(TASKS;
        budget=BUDGET,
        n_episodes=N_EPISODES,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        repeats=A1_ONE_REPEATS,
    )
elseif A1_ZERO_CHECKS
    run_a10_repeat_checks(
        budget=BUDGET,
        n_episodes=N_EPISODES,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        repeats=A1_ZERO_REPEATS,
    )
elseif ABLATION_MATRIX
    run_ablation_matrix(TASKS;
        budget=BUDGET,
        n_episodes=N_EPISODES,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        include_operator_comparison=OPERATOR_COMPARISON)
elseif C2_BASIN_CHECKS
    run_c2_basin_controller_checks(TASKS;
        budget=BUDGET,
        n_episodes=N_EPISODES,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C2_DATA_REPEATS,
        eval_repeats=C2_EVAL_REPEATS,
        training_epochs=C2_TRAIN_EPOCHS,
        max_promoted=C2_MAX_PROMOTED,
        celecoxib_repeats=C2_CELECOXIB_REPEATS,
        control_repeats=C2_CONTROL_REPEATS,
        sanity_repeats=C2_SANITY_REPEATS,
    )
elseif C3_PARENT_CHECKS
    run_c3_parent_controller_checks(TASKS;
        budget=BUDGET,
        n_episodes=N_EPISODES,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C3_DATA_REPEATS,
        training_epochs=C3_TRAIN_EPOCHS,
        parent_candidate_limit=C3_PARENT_CANDIDATE_LIMIT,
        celecoxib_repeats=C3_CELECOXIB_REPEATS,
        control_repeats=C3_CONTROL_REPEATS,
        sanity_repeats=C3_SANITY_REPEATS,
    )
elseif C4_PARENT_SEMANTICS
    run_c4_parent_semantics_checks(TASKS;
        budget=BUDGET,
        n_episodes=N_EPISODES,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C4_DATA_REPEATS,
        training_epochs=C4_TRAIN_EPOCHS,
        parent_candidate_limit=C4_PARENT_CANDIDATE_LIMIT,
        celecoxib_repeats=C4_CELECOXIB_REPEATS,
        control_repeats=C4_CONTROL_REPEATS,
        sanity_repeats=C4_SANITY_REPEATS,
    )
elseif C8_OPERATOR_CONTROLLER
    run_c8_operator_controller_checks(TASKS;
        budget=BUDGET,
        n_episodes=N_EPISODES,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C8_DATA_REPEATS,
        training_epochs=C8_TRAIN_EPOCHS,
        celecoxib_repeats=C8_CELECOXIB_REPEATS,
        control_repeats=C8_CONTROL_REPEATS,
        sanity_repeats=C8_SANITY_REPEATS,
    )
elseif C9_OPERATOR_ELIGIBILITY
    run_c9_operator_eligibility_checks(TASKS;
        budget=BUDGET,
        n_episodes=N_EPISODES,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C9_DATA_REPEATS,
        training_epochs=C9_TRAIN_EPOCHS,
        celecoxib_repeats=C9_CELECOXIB_REPEATS,
        control_repeats=C9_CONTROL_REPEATS,
        sanity_repeats=C9_SANITY_REPEATS,
    )
elseif C21_REPRESENTATION_SEMANTICS_AUDIT
    run_c21_representation_semantics_repair_audit(TASKS;
        budget=BUDGET,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C21_DATA_REPEATS,
        num_splits=C21_NUM_SPLITS,
        parent_candidate_limit=C21_PARENT_CANDIDATE_LIMIT,
        max_basin_contexts=C21_MAX_BASIN_CONTEXTS,
        option_horizon=C21_OPTION_HORIZON,
        control_repeats=C21_CONTROL_REPEATS,
        max_allocation_budget=C21_MAX_ALLOCATION_BUDGET,
        train_fraction=C21_TRAIN_FRACTION,
        state_threshold=C21_STATE_THRESHOLD,
        stability_threshold=C21_STABILITY_THRESHOLD,
    )
elseif C20_SPARSE_POSITIVE_AUDIT
    run_c20_sparse_positive_operating_point_audit(TASKS;
        budget=BUDGET,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C20_DATA_REPEATS,
        num_splits=C20_NUM_SPLITS,
        parent_candidate_limit=C20_PARENT_CANDIDATE_LIMIT,
        max_basin_contexts=C20_MAX_BASIN_CONTEXTS,
        option_horizon=C20_OPTION_HORIZON,
        control_repeats=C20_CONTROL_REPEATS,
        max_allocation_budget=C20_MAX_ALLOCATION_BUDGET,
        train_fraction=C20_TRAIN_FRACTION,
        fixed_thresholds=C20_FIXED_THRESHOLDS,
        max_positive_fraction=C20_MAX_POSITIVE_FRACTION,
        min_positive_count=C20_MIN_POSITIVE_COUNT,
        min_train_precision=C20_MIN_TRAIN_PRECISION,
        state_threshold=C20_STATE_THRESHOLD,
    )
elseif C19_INTERVENTION_GEOMETRY
    run_c19_intervention_geometry_audit(TASKS;
        budget=BUDGET,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C19_DATA_REPEATS,
        parent_candidate_limit=C19_PARENT_CANDIDATE_LIMIT,
        max_basin_contexts=C19_MAX_BASIN_CONTEXTS,
        option_horizon=C19_OPTION_HORIZON,
        control_repeats=C19_CONTROL_REPEATS,
        max_allocation_budget=C19_MAX_ALLOCATION_BUDGET,
        state_threshold=C19_STATE_THRESHOLD,
        stability_threshold=C19_STABILITY_THRESHOLD,
    )
elseif C18_OPPORTUNITY_REPEATABILITY
    run_c18_opportunity_state_repeatability(TASKS;
        budget=BUDGET,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C18_DATA_REPEATS,
        num_splits=C18_NUM_SPLITS,
        parent_candidate_limit=C18_PARENT_CANDIDATE_LIMIT,
        max_basin_contexts=C18_MAX_BASIN_CONTEXTS,
        option_horizon=C18_OPTION_HORIZON,
        control_repeats=C18_CONTROL_REPEATS,
        max_allocation_budget=C18_MAX_ALLOCATION_BUDGET,
        state_threshold=C18_STATE_THRESHOLD,
        train_fraction=C18_TRAIN_FRACTION,
        threshold_perturbations=C18_THRESHOLD_PERTURBATIONS,
    )
elseif C17_OPPORTUNITY_STATE
    run_c17_opportunity_state_probe(TASKS;
        budget=BUDGET,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C17_DATA_REPEATS,
        parent_candidate_limit=C17_PARENT_CANDIDATE_LIMIT,
        max_basin_contexts=C17_MAX_BASIN_CONTEXTS,
        option_horizon=C17_OPTION_HORIZON,
        control_repeats=C17_CONTROL_REPEATS,
        max_allocation_budget=C17_MAX_ALLOCATION_BUDGET,
        state_threshold=C17_STATE_THRESHOLD,
        train_fraction=C17_TRAIN_FRACTION,
    )
elseif C16_SELECTIVE_ALLOCATOR
    run_c16_selective_frontier_allocator_checks(TASKS;
        budget=BUDGET,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C16_DATA_REPEATS,
        parent_candidate_limit=C16_PARENT_CANDIDATE_LIMIT,
        max_basin_contexts=C16_MAX_BASIN_CONTEXTS,
        option_horizon=C16_OPTION_HORIZON,
        control_repeats=C16_CONTROL_REPEATS,
        max_allocation_budget=C16_MAX_ALLOCATION_BUDGET,
        override_threshold=C16_OVERRIDE_THRESHOLD,
        train_fraction=C16_TRAIN_FRACTION,
    )
elseif C15_FRONTIER_ALLOCATION
    run_c15_frontier_allocation_probe(TASKS;
        budget=BUDGET,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C15_DATA_REPEATS,
        parent_candidate_limit=C15_PARENT_CANDIDATE_LIMIT,
        max_basin_contexts=C15_MAX_BASIN_CONTEXTS,
        option_horizon=C15_OPTION_HORIZON,
        control_repeats=C15_CONTROL_REPEATS,
        region_families=C15_REGION_FAMILIES,
        max_allocation_budget=C15_MAX_ALLOCATION_BUDGET,
    )
elseif C14_CALIBRATED_ORDINAL
    run_c14_calibrated_ordinal_option_checks(TASKS;
        budget=BUDGET,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C14_DATA_REPEATS,
        training_epochs=C14_TRAIN_EPOCHS,
        calibration_epochs=C14_CALIBRATION_EPOCHS,
        parent_candidate_limit=C14_PARENT_CANDIDATE_LIMIT,
        max_basin_contexts=C14_MAX_BASIN_CONTEXTS,
        option_horizon=C14_OPTION_HORIZON,
        celecoxib_repeats=C14_CELECOXIB_REPEATS,
        control_repeats=C14_CONTROL_REPEATS,
    )
elseif C13_OPTION_REFINEMENT
    run_c13_option_value_refinement_checks(TASKS;
        budget=BUDGET,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C13_DATA_REPEATS,
        training_epochs=C13_TRAIN_EPOCHS,
        parent_candidate_limit=C13_PARENT_CANDIDATE_LIMIT,
        max_basin_contexts=C13_MAX_BASIN_CONTEXTS,
        option_horizon=C13_OPTION_HORIZON,
        celecoxib_repeats=C13_CELECOXIB_REPEATS,
        control_repeats=C13_CONTROL_REPEATS,
    )
elseif C12_OPTION_VALUE
    run_c12_option_value_checks(TASKS;
        budget=BUDGET,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C12_DATA_REPEATS,
        training_epochs=C12_TRAIN_EPOCHS,
        parent_candidate_limit=C12_PARENT_CANDIDATE_LIMIT,
        max_basin_contexts=C12_MAX_BASIN_CONTEXTS,
        option_horizon=C12_OPTION_HORIZON,
        celecoxib_repeats=C12_CELECOXIB_REPEATS,
        control_repeats=C12_CONTROL_REPEATS,
    )
elseif C11_SUBTRAJECTORY
    run_c11_subtrajectory_bridge(TASKS;
        budget=BUDGET,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C11_DATA_REPEATS,
        parent_candidate_limit=C11_PARENT_CANDIDATE_LIMIT,
        max_basin_contexts=C11_MAX_BASIN_CONTEXTS,
        option_horizon=C11_OPTION_HORIZON,
    )
elseif C10_COUPLED_OPTION
    run_c10_coupled_option_probe(TASKS;
        budget=BUDGET,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C10_DATA_REPEATS,
        parent_candidate_limit=C10_PARENT_CANDIDATE_LIMIT,
        max_basin_contexts=C10_MAX_BASIN_CONTEXTS,
        option_horizon=C10_OPTION_HORIZON,
    )
elseif C7_BASIN_OPERATOR
    run_c7_basin_conditioned_operator_probe(TASKS;
        budget=BUDGET,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C7_DATA_REPEATS,
        parent_candidate_limit=C7_PARENT_CANDIDATE_LIMIT,
        max_basin_contexts=C7_MAX_BASIN_CONTEXTS,
    )
elseif C6_PARENT_OPERATOR
    run_c6_parent_operator_causality_probe(TASKS;
        budget=BUDGET,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C6_DATA_REPEATS,
        parent_candidate_limit=C6_PARENT_CANDIDATE_LIMIT,
        max_basin_contexts=C6_MAX_BASIN_CONTEXTS,
    )
elseif C5_PARENT_CAUSALITY
    run_c5_parent_causality_probe(TASKS;
        budget=BUDGET,
        target_seed=TARGET_SEED,
        bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
        max_step_attempts=MAX_STEP_ATTEMPTS,
        max_operator_candidates=MAX_OPERATOR_CANDIDATES,
        data_repeats=C5_DATA_REPEATS,
        parent_candidate_limit=C5_PARENT_CANDIDATE_LIMIT,
    )
else
    results = Dict{String,Any}()
    for task in TASKS
        println("Running hierarchical edit baseline on $(task) ...")
        result, extra = run_hierarchical_edit_pmo_task(task;
            budget=BUDGET,
            n_episodes=N_EPISODES,
            target_seed=TARGET_SEED,
            enable_augmentation=true,
            enable_warmup=true,
            bootstrap_warmup_rounds=BOOTSTRAP_WARMUP_ROUNDS,
            run_episodes=true,
            regime_name="full",
            config=HierarchicalEditConfig(
                max_step_attempts=MAX_STEP_ATTEMPTS,
                max_operator_candidates=MAX_OPERATOR_CANDIDATES,
            ),
            verbose=true)
        results[task] = Dict("hierarchical_edit" => result, "extra" => extra)
        source_repr = repr(extra["frontier_source_summary"])
        println("HE[$(task)] AUC=$(round(result.auc_top10, digits=4)) | Top1=$(round(result.top1, digits=4)) | Top10=$(round(result.top10_mean, digits=4)) | Calls=$(result.n_oracle_calls) | GraphUnique=$(extra["graph_unique_molecules"])")
        println("HE[$(task)] frontier_sources=$(source_repr)")

        if COMPARE_TOKEN
            println("COMPARE_TOKEN=1 requested, but token-baseline invocation is intentionally left explicit for controlled matched runs.")
        end
    end
    results
end

outfile = joinpath(OUTDIR,
    A1_TWO_CONFIRM ? "hierarchical_edit_a12_confirm_results.jls" :
    A1_ONE_BRIDGE ? "hierarchical_edit_a11_bridge_results.jls" :
    A1_ZERO_CHECKS ? "hierarchical_edit_a10_checks_results.jls" :
    ABLATION_MATRIX ? "hierarchical_edit_stage_a0_ablation_results.jls" :
    C2_BASIN_CHECKS ? "hierarchical_edit_c2_basin_investigation_results.jls" :
    C3_PARENT_CHECKS ? "hierarchical_edit_c3_parent_results.jls" :
    C4_PARENT_SEMANTICS ? "hierarchical_edit_c4_parent_semantics_results.jls" :
    C8_OPERATOR_CONTROLLER ? "hierarchical_edit_c8_operator_controller_results.jls" :
    C9_OPERATOR_ELIGIBILITY ? "hierarchical_edit_c9_operator_eligibility_results.jls" :
    C21_REPRESENTATION_SEMANTICS_AUDIT ? "hierarchical_edit_c21_representation_semantics_results.jls" :
    C20_SPARSE_POSITIVE_AUDIT ? "hierarchical_edit_c20_sparse_positive_results.jls" :
    C19_INTERVENTION_GEOMETRY ? "hierarchical_edit_c19_intervention_geometry_results.jls" :
    C18_OPPORTUNITY_REPEATABILITY ? "hierarchical_edit_c18_opportunity_repeatability_results.jls" :
    C17_OPPORTUNITY_STATE ? "hierarchical_edit_c17_opportunity_state_results.jls" :
    C16_SELECTIVE_ALLOCATOR ? "hierarchical_edit_c16_selective_allocator_results.jls" :
    C15_FRONTIER_ALLOCATION ? "hierarchical_edit_c15_frontier_allocation_results.jls" :
    C14_CALIBRATED_ORDINAL ? "hierarchical_edit_c14_calibrated_ordinal_results.jls" :
    C13_OPTION_REFINEMENT ? "hierarchical_edit_c13_option_value_refinement_results.jls" :
    C12_OPTION_VALUE ? "hierarchical_edit_c12_option_value_results.jls" :
    C11_SUBTRAJECTORY ? "hierarchical_edit_c11_subtrajectory_bridge_results.jls" :
    C10_COUPLED_OPTION ? "hierarchical_edit_c10_coupled_option_results.jls" :
    C7_BASIN_OPERATOR ? "hierarchical_edit_c7_basin_operator_results.jls" :
    C6_PARENT_OPERATOR ? "hierarchical_edit_c6_parent_operator_causality_results.jls" :
    C5_PARENT_CAUSALITY ? "hierarchical_edit_c5_parent_causality_results.jls" :
    "hierarchical_edit_baseline_results.jls")

serialize(outfile, Dict(
    "tasks" => TASKS,
    "budget" => BUDGET,
    "episodes" => N_EPISODES,
    "target_seed" => TARGET_SEED,
    "bootstrap_warmup_rounds" => BOOTSTRAP_WARMUP_ROUNDS,
    "max_step_attempts" => MAX_STEP_ATTEMPTS,
    "max_operator_candidates" => MAX_OPERATOR_CANDIDATES,
    "flags" => Dict(
        "ablation_matrix" => ABLATION_MATRIX,
        "a10_checks" => A1_ZERO_CHECKS,
        "a11_bridge" => A1_ONE_BRIDGE,
        "a12_confirm" => A1_TWO_CONFIRM,
        "c2_basin_checks" => C2_BASIN_CHECKS,
        "c3_parent_checks" => C3_PARENT_CHECKS,
        "c4_parent_semantics" => C4_PARENT_SEMANTICS,
        "c5_parent_causality" => C5_PARENT_CAUSALITY,
        "c6_parent_operator" => C6_PARENT_OPERATOR,
        "c7_basin_operator" => C7_BASIN_OPERATOR,
        "c8_operator_controller" => C8_OPERATOR_CONTROLLER,
        "c9_operator_eligibility" => C9_OPERATOR_ELIGIBILITY,
        "c21_representation_semantics_audit" => C21_REPRESENTATION_SEMANTICS_AUDIT,
        "c20_sparse_positive_audit" => C20_SPARSE_POSITIVE_AUDIT,
        "c19_intervention_geometry" => C19_INTERVENTION_GEOMETRY,
        "c18_opportunity_repeatability" => C18_OPPORTUNITY_REPEATABILITY,
        "c17_opportunity_state" => C17_OPPORTUNITY_STATE,
        "c16_selective_allocator" => C16_SELECTIVE_ALLOCATOR,
        "c15_frontier_allocation" => C15_FRONTIER_ALLOCATION,
        "c10_coupled_option" => C10_COUPLED_OPTION,
        "c11_subtrajectory" => C11_SUBTRAJECTORY,
        "c12_option_value" => C12_OPTION_VALUE,
        "c13_option_refinement" => C13_OPTION_REFINEMENT,
        "c14_calibrated_ordinal" => C14_CALIBRATED_ORDINAL,
    ),
    "results" => all_results,
))
println("Saved hierarchical edit results to: $(outfile)")
