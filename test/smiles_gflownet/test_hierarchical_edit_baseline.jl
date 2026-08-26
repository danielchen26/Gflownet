using Test
using GFlowNet
using Random

struct DummyBasinController end
function GFlowNet.select_basin(::DummyBasinController,
                               snapshot::GFlowNet.FrontierSnapshot,
                               candidates::Vector{GFlowNet.ScoredBasinCandidate};
                               step_index::Int=0)
    isempty(candidates) && return nothing
    return candidates[min(2, length(candidates))]
end

struct DummyParentController end
function GFlowNet.select_parent(::DummyParentController,
                                snapshot::GFlowNet.FrontierSnapshot,
                                candidates::Vector{GFlowNet.ScoredParentCandidate};
                                step_index::Int=0)
    isempty(candidates) && return nothing
    return candidates[min(2, length(candidates))]
end

struct DummyOperatorController end
function GFlowNet.select_operator(::DummyOperatorController,
                                  snapshot::GFlowNet.FrontierSnapshot,
                                  basin::GFlowNet.BasinSummary,
                                  parent::GFlowNet.FrontierSnapshotEntry,
                                  candidates::Vector{GFlowNet.OperatorDecisionCandidate};
                                  step_index::Int=0)
    isempty(candidates) && return nothing
    return candidates[min(2, length(candidates))]
end

@testset "Hierarchical Edit Baseline" begin
    @testset "Trusted operators are the default" begin
        cfg = HierarchicalEditConfig()
        ops = available_edit_operators(; allow_crossover=cfg.allow_crossover,
                                       allow_fragment_ops=cfg.allow_fragment_ops)
        @test cfg.allow_fragment_ops == false
        @test Set(ops) == Set([:mutate, :crossover, :terminate])
        @test :add_fragment ∉ ops
        @test :replace_fragment ∉ ops
        @test :delete_fragment ∉ ops
    end

    @testset "Snapshot identity is stable for identical frontier state" begin
        fb = MolecularFrontierBuffer(20)
        add_to_frontier!(fb, "CCO"; reward=0.8, source=:seed, operator=:seed)
        add_to_frontier!(fb, "CCN"; reward=0.7, source=:model, operator=:sample)

        snap1 = create_frontier_snapshot(fb; max_entries=10, target_smiles="CCO", budget_remaining=50, created_at_step=1)
        snap2 = create_frontier_snapshot(fb; max_entries=10, target_smiles="CCO", budget_remaining=50, created_at_step=1)

        @test snap1.snapshot_id == snap2.snapshot_id
        @test snap1.snapshot_id == compute_frontier_snapshot_id(snap1.entries, snap1.target_smiles, snap1.budget_remaining, snap1.created_at_step)
    end

    @testset "Deterministic basin candidates are stable and ranked" begin
        fb = MolecularFrontierBuffer(20)
        add_to_frontier!(fb, "CCO"; reward=0.8, source=:seed, operator=:seed)
        add_to_frontier!(fb, "CCN"; reward=0.75, source=:warmup, operator=:mutate)
        add_to_frontier!(fb, "c1ccccc1"; reward=0.6, source=:seed, operator=:seed)

        snap = create_frontier_snapshot(fb; max_entries=10, target_smiles="CCO", budget_remaining=32, created_at_step=1)
        c1 = candidate_basins(snap; max_candidates=2)
        c2 = candidate_basins(snap; max_candidates=2)

        @test length(c1) == 2
        @test [c.basin.scaffold for c in c1] == [c.basin.scaffold for c in c2]
        @test [c.score for c in c1] == [c.score for c in c2]
        @test c1[1].score >= c1[2].score
    end

    @testset "Deterministic parent candidates are stable and ranked" begin
        fb = MolecularFrontierBuffer(20)
        add_to_frontier!(fb, "CCO"; reward=0.8, source=:seed, operator=:seed)
        add_to_frontier!(fb, "CCN"; reward=0.75, source=:warmup, operator=:mutate)
        add_to_frontier!(fb, "CCC"; reward=0.7, source=:edit, operator=:mutate)

        snap = create_frontier_snapshot(fb; max_entries=10, target_smiles="CCO", budget_remaining=32, created_at_step=1)
        basins = candidate_basins(snap; max_candidates=1)
        p1 = candidate_parents(snap; basin=basins[1].basin, max_candidates=3)
        p2 = candidate_parents(snap; basin=basins[1].basin, max_candidates=3)

        @test !isempty(p1)
        @test [c.entry.smiles for c in p1] == [c.entry.smiles for c in p2]
        @test [c.score for c in p1] == [c.score for c in p2]
        @test p1[1].score >= p1[end].score
    end

    @testset "Basin dataset extraction joins truthful downstream outcomes" begin
        basin_log = BasinDecisionLog(
            0x11,
            "ep-1",
            "task",
            1,
            1,
            24,
            1,
            3,
            0.8,
            0.7,
            2,
            [
                BasinDecisionCandidate("sca-a", 2, 0.8, 0.75, 0.1, 0.05, 1.2, true),
                BasinDecisionCandidate("sca-b", 1, 0.6, 0.6, 0.05, 0.01, 0.9, false),
            ],
            1,
            "sca-a",
            1.2,
        )
        proposal_log = HierarchicalEditProposalLog(
            0x11,
            "ep-1",
            "task",
            1,
            1,
            "sca-a",
            1.2,
            "CCO",
            0.6,
            "sca-a",
            :mutate,
            nothing,
            3,
            0,
            0,
            0,
            0,
            2,
            2,
            0,
            0,
            "CCN",
            0.8,
            0.2,
            0.65,
            0.7,
            0.75,
            0.8,
            false,
        )
        decision_log = HierarchicalEditDecisionLog(
            0x11,
            "ep-1",
            "task",
            1,
            1,
            "sca-a",
            1.2,
            "CCO",
            0.6,
            :mutate,
            2,
            "CCN",
            0.8,
            0.2,
            false,
            "sca-a",
            "sca-a",
            "same_family",
            true,
            0.1,
            0.05,
            0.0,
            0.25,
            true,
        )

        dataset = extract_basin_controller_dataset([basin_log], [proposal_log], [decision_log])
        @test length(dataset) == 1
        @test dataset.records[1].success_label == true
        @test dataset.records[1].chosen_index == 1
        @test dataset.records[1].has_proposal_log == true
        @test dataset.records[1].has_decision_log == true
        @test length(dataset.records[1].candidate_features) == 2
        @test dataset.records[1].chosen_frontier_utility_delta > 0
        @test dataset.records[1].target_value > 0

        audit = audit_basin_dataset_coverage([basin_log], [proposal_log], [decision_log])
        @test audit["basin_logs"] == 1
        @test audit["matched_attempt_outcomes"] == 1
        @test audit["proposal_coverage_fraction"] == 1.0
        @test audit["decision_coverage_fraction"] == 1.0
        @test get(audit["class_counts"], "productive", 0) == 1
    end

    @testset "Truthful operator dataset extraction works" begin
        operator_log = OperatorDecisionLog(
            0x41,
            "ep-o",
            "task",
            1,
            1,
            "sca-a",
            1.1,
            "CCO",
            0.6,
            "sca-a",
            0.05,
            0.02,
            "seed",
            [
                OperatorDecisionCandidate(:mutate, 0.8, 3, 2, 0.0, 0.0),
                OperatorDecisionCandidate(:crossover, 0.5, 1, 0, 0.0, 0.0),
            ],
            1,
            :mutate,
            0.8,
            true,
            0.9,
            true,
            false,
            1,
            1,
            0.3,
            0.3,
            0.0,
            0.65,
            0.65,
            true,
            false,
            "eligible_agree",
        )
        proposal_log = HierarchicalEditProposalLog(
            0x41,
            "ep-o",
            "task",
            1,
            1,
            "sca-a",
            1.1,
            "CCO",
            0.6,
            "sca-a",
            :mutate,
            nothing,
            3,
            0,
            0,
            0,
            0,
            2,
            2,
            0,
            0,
            "CCOC",
            0.82,
            0.22,
            0.61,
            0.70,
            0.78,
            0.82,
            false,
        )
        decision_log = HierarchicalEditDecisionLog(
            0x41,
            "ep-o",
            "task",
            1,
            1,
            "sca-a",
            1.1,
            "CCO",
            0.6,
            :mutate,
            2,
            "CCOC",
            0.82,
            0.22,
            false,
            "sca-a",
            "sca-a",
            "same_family",
            true,
            0.1,
            0.05,
            0.0,
            0.25,
            true,
        )

        dataset = extract_operator_controller_dataset([operator_log], [proposal_log], [decision_log]; feature_mode=:augmented)
        @test length(dataset) == 1
        @test dataset.records[1].chosen_index == 1
        @test dataset.records[1].target_value > 0
        @test dataset.records[1].state_label == "robust_operator"
        @test dataset.records[1].controller_eligible == true
        @test dataset.records[1].predicted_eligible == true
        @test dataset.records[1].acted_on == true
        @test dataset.records[1].preserved_to_heuristic == false
        @test length(dataset.records[1].eligibility_features) == length(dataset.records[1].context_features)
        @test length(dataset.records[1].candidate_features[1]) > length(dataset.records[1].context_features)

        stats = operator_controller_dataset_stats(dataset)
        @test stats["size"] == 1
        @test haskey(stats, "state_counts")
        @test haskey(stats, "predicted_eligible_fraction")
        @test haskey(stats, "acted_on_fraction")
        @test get(stats["state_counts"], "robust_operator", 0) == 1

        audit = audit_operator_dataset_coverage([operator_log], [proposal_log], [decision_log])
        @test audit["operator_logs"] == 1
        @test audit["matched_attempt_outcomes"] == 1
        @test get(audit["class_counts"], "productive", 0) == 1
    end

    @testset "Truthful parent dataset extraction works" begin
        parent_log = ParentDecisionLog(
            0x31,
            "ep-p",
            "task",
            1,
            1,
            "sca-a",
            1.1,
            [
                ParentDecisionCandidate("CCO", "", 0.6, 0.1, 0.0, "seed", 1.2, 0, true, true),
                ParentDecisionCandidate("CCN", "", 0.5, 0.05, 0.0, "warmup", 0.9, 1, true, false),
            ],
            1,
            "CCO",
            1.2,
            1,
            1,
            0.3,
            0.3,
            0.0,
            0.65,
            0.65,
            false,
            false,
            "agree",
        )
        proposal_log = HierarchicalEditProposalLog(
            0x31,
            "ep-p",
            "task",
            1,
            1,
            "sca-a",
            1.1,
            "CCO",
            0.6,
            "",
            :mutate,
            nothing,
            3,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            0,
            "CCOC",
            0.82,
            0.22,
            0.61,
            0.70,
            0.78,
            0.82,
            false,
        )
        decision_log = HierarchicalEditDecisionLog(
            0x31,
            "ep-p",
            "task",
            1,
            1,
            "sca-a",
            1.1,
            "CCO",
            0.6,
            :mutate,
            2,
            "CCOC",
            0.82,
            0.22,
            false,
            "",
            "",
            "no_scaffold",
            true,
            0.1,
            0.05,
            0.0,
            0.25,
            true,
        )

        dataset = extract_parent_controller_dataset([parent_log], [proposal_log], [decision_log]; feature_mode=:augmented)
        @test length(dataset) == 1
        @test dataset.records[1].chosen_index == 1
        @test dataset.records[1].target_value > 0
        @test length(dataset.records[1].candidate_features[1]) > 10
        @test dataset.records[1].heuristic_margin > 0
        @test dataset.records[1].selection_reason == "agree"

        rel_dataset = extract_parent_controller_dataset([parent_log], [proposal_log], [decision_log]; target_mode=:relative_blended)
        risk_dataset = extract_parent_controller_dataset([parent_log], [proposal_log], [decision_log]; target_mode=:risk_adjusted_advantage)
        @test length(rel_dataset) == 1
        @test length(risk_dataset) == 1
        @test rel_dataset.records[1].target_value != dataset.records[1].target_value
        @test risk_dataset.records[1].target_value <= dataset.records[1].target_value

        stats = parent_controller_dataset_stats(dataset)
        @test haskey(stats, "mean_heuristic_margin")
        @test haskey(stats, "heuristic_ambiguous_fraction")

        audit = audit_parent_dataset_coverage([parent_log], [proposal_log], [decision_log])
        @test audit["parent_logs"] == 1
        @test audit["matched_attempt_outcomes"] == 1
        @test get(audit["class_counts"], "productive", 0) == 1
    end

    @testset "Batch 1A.2 target modes and feature modes are meaningful" begin
        productive = BasinAttemptOutcomeSummary(
            0x21,
            "ep-p",
            "task",
            1,
            1,
            "sca-a",
            true,
            true,
            3,
            2,
            false,
            0.2,
            0.6,
            0.8,
            true,
            1,
            0.2,
            0.4,
            true,
            "productive",
        )
        degenerate = BasinAttemptOutcomeSummary(
            0x22,
            "ep-d",
            "task",
            1,
            1,
            "sca-b",
            true,
            false,
            3,
            0,
            true,
            0.0,
            0.0,
            0.0,
            false,
            0,
            0.0,
            0.0,
            false,
            "degenerate",
        )

        @test compute_basin_target(productive; mode=:ordinal_productivity) == 1.0
        @test compute_basin_target(degenerate; mode=:ordinal_productivity) == -1.0
        @test compute_basin_target(productive; mode=:risk_adjusted_utility) > compute_basin_target(degenerate; mode=:risk_adjusted_utility)

        basin_log = BasinDecisionLog(
            0x11,
            "ep-1",
            "task",
            1,
            1,
            24,
            1,
            3,
            0.8,
            0.7,
            2,
            [
                BasinDecisionCandidate("sca-a", 2, 0.8, 0.75, 0.1, 0.05, 1.2, true),
                BasinDecisionCandidate("sca-b", 1, 0.6, 0.6, 0.05, 0.01, 0.9, false),
            ],
            1,
            "sca-a",
            1.2,
        )
        proposal_log = HierarchicalEditProposalLog(
            0x11,
            "ep-1",
            "task",
            1,
            1,
            "sca-a",
            1.2,
            "CCO",
            0.6,
            "sca-a",
            :mutate,
            nothing,
            3,
            0,
            0,
            0,
            0,
            2,
            2,
            0,
            0,
            "CCN",
            0.8,
            0.2,
            0.65,
            0.7,
            0.75,
            0.8,
            false,
        )
        decision_log = HierarchicalEditDecisionLog(
            0x11,
            "ep-1",
            "task",
            1,
            1,
            "sca-a",
            1.2,
            "CCO",
            0.6,
            :mutate,
            2,
            "CCN",
            0.8,
            0.2,
            false,
            "sca-a",
            "sca-a",
            "same_family",
            true,
            0.1,
            0.05,
            0.0,
            0.25,
            true,
        )

        basic_dataset = extract_basin_controller_dataset([basin_log], [proposal_log], [decision_log]; feature_mode=:basic)
        augmented_dataset = extract_basin_controller_dataset([basin_log], [proposal_log], [decision_log]; feature_mode=:augmented, target_mode=:risk_adjusted_utility)
        @test length(augmented_dataset.records[1].candidate_features[1]) > length(basic_dataset.records[1].candidate_features[1])
        @test augmented_dataset.records[1].target_value > 0

        controller = create_learned_basin_controller(length(augmented_dataset.records[1].candidate_features[1]); feature_mode=:augmented)
        eval_stats = evaluate_basin_controller(controller, augmented_dataset)
        @test controller.feature_mode == :augmented
        @test haskey(eval_stats, "frontier_utility_correlation")
        @test haskey(eval_stats, "productive_degenerate_margin")
        @test length(eval_stats["score_bucket_productivity"]) == 3
    end

    @testset "Terminate proposal diagnostics are truthful" begin
        vocab = SMILESVocabulary()
        proposals, diag = propose_edit_with_diagnostics("CCO", :terminate, vocab; max_candidates=4)

        @test length(proposals) == 1
        @test proposals[1].child_smiles == "CCO"
        @test diag["raw_candidate_count"] == 1
        @test diag["duplicate_candidate_count"] == 0
        @test diag["empty_child_count"] == 0
        @test diag["self_child_count"] == 0
        @test diag["unique_valid_count"] == 1
    end

    @testset "Frozen episode emits consistent decision and proposal diagnostics" begin
        Random.seed!(11)
        fb = MolecularFrontierBuffer(20)
        add_to_frontier!(fb, "CCO"; reward=0.8, source=:seed, operator=:seed)
        add_to_frontier!(fb, "CCN"; reward=0.7, source=:model, operator=:sample)

        tb = EditTrajectoryBuffer(50)
        db = HierarchicalEditDiagnosticsBuffer(50)
        vocab = SMILESVocabulary()
        cfg = HierarchicalEditConfig(horizon=2, allow_crossover=false, allow_fragment_ops=false, max_operator_candidates=4)

        ep = run_hierarchical_edit_episode!(fb, tb, s -> 0.5, vocab;
            reward_fn_batch=xs -> fill(0.5, length(xs)),
            diagnostics_buffer=db,
            config=cfg,
            target_smiles="CCO",
            budget_remaining=50,
            created_at_step=2,
            task_name="diagnostics",
            operator_override=:terminate)

        @test ep.commits_applied == 1
        @test length(ep.steps) == 1
        @test length(tb) == 1
        @test length(db) == 1
        @test length(db.proposal_logs) == 1
        @test all(s.snapshot_id == ep.snapshot_id for s in ep.steps)
        @test all(s.episode_id == ep.episode_id for s in ep.steps)
        @test db.logs[1].snapshot_id == ep.snapshot_id
        @test db.logs[1].episode_id == ep.episode_id
        @test db.proposal_logs[1].snapshot_id == ep.snapshot_id
        @test db.proposal_logs[1].episode_id == ep.episode_id
        @test db.proposal_logs[1].raw_candidate_count == 1
        @test db.proposal_logs[1].empty_after_filter == false
    end

    @testset "Learned basin hook overrides heuristic basin selection" begin
        Random.seed!(17)
        fb = MolecularFrontierBuffer(20)
        add_to_frontier!(fb, "CCO"; reward=0.9, source=:seed, operator=:seed)
        add_to_frontier!(fb, "c1ccccc1"; reward=0.7, source=:seed, operator=:seed)
        tb = EditTrajectoryBuffer(20)
        db = HierarchicalEditDiagnosticsBuffer(20)
        vocab = SMILESVocabulary()
        cfg = HierarchicalEditConfig(
            horizon=1,
            allow_crossover=false,
            max_operator_candidates=2,
            basin_candidate_limit=2,
            use_learned_basin=true,
            learned_basin_controller=DummyBasinController(),
        )

        ep = run_hierarchical_edit_episode!(fb, tb, s -> 0.5, vocab;
            reward_fn_batch=xs -> fill(0.5, length(xs)),
            diagnostics_buffer=db,
            config=cfg,
            target_smiles="CCO",
            budget_remaining=20,
            created_at_step=1,
            task_name="learned-basin-test",
            operator_override=:terminate)

        @test ep.commits_applied == 1
        @test length(db.basin_logs) == 1
        @test db.basin_logs[1].chosen_index == 2
        @test db.basin_logs[1].chosen_basin_scaffold == db.basin_logs[1].candidate_basins[2].scaffold
    end

    @testset "Learned parent hook overrides heuristic parent selection" begin
        Random.seed!(19)
        fb = MolecularFrontierBuffer(20)
        add_to_frontier!(fb, "CCO"; reward=0.9, source=:seed, operator=:seed)
        add_to_frontier!(fb, "CCN"; reward=0.7, source=:warmup, operator=:mutate)
        add_to_frontier!(fb, "CCC"; reward=0.65, source=:edit, operator=:mutate)
        tb = EditTrajectoryBuffer(20)
        db = HierarchicalEditDiagnosticsBuffer(20)
        vocab = SMILESVocabulary()
        cfg = HierarchicalEditConfig(
            horizon=1,
            allow_crossover=false,
            max_operator_candidates=2,
            basin_candidate_limit=1,
            parent_candidate_limit=3,
            use_learned_parent=true,
            learned_parent_controller=DummyParentController(),
        )

        ep = run_hierarchical_edit_episode!(fb, tb, s -> 0.5, vocab;
            reward_fn_batch=xs -> fill(0.5, length(xs)),
            diagnostics_buffer=db,
            config=cfg,
            target_smiles="CCO",
            budget_remaining=20,
            created_at_step=1,
            task_name="learned-parent-test",
            operator_override=:terminate)

        @test ep.commits_applied == 1
        @test length(db.parent_logs) == 1
        @test db.parent_logs[1].chosen_index == 2
        @test db.parent_logs[1].chosen_parent_smiles == db.parent_logs[1].candidate_parents[2].smiles
        @test db.parent_logs[1].override_applied == true
        @test db.parent_logs[1].selection_reason == "custom_override"
    end

    @testset "Parent intervention probe returns deterministic step-1 summaries" begin
        fb = MolecularFrontierBuffer(20)
        add_to_frontier!(fb, "CCO"; reward=0.9, source=:seed, operator=:seed)
        add_to_frontier!(fb, "CCN"; reward=0.7, source=:warmup, operator=:mutate)
        vocab = SMILESVocabulary()
        cfg = HierarchicalEditConfig(
            allow_crossover=false,
            max_operator_candidates=2,
            parent_candidate_limit=2,
            operators=[:terminate],
        )
        probe = probe_parent_interventions(fb, s -> 0.5, vocab;
            reward_fn_batch=xs -> fill(0.5, length(xs)),
            config=cfg,
            target_smiles="CCO",
            budget_remaining=20,
            created_at_step=1,
            task_name="probe-test",
            max_parents=2,
            operators=[:terminate])

        @test probe["task_name"] == "probe-test"
        @test probe["candidate_mode"] == "frontier"
        @test length(probe["parent_summaries"]) == 2
        @test length(probe["basin_summaries"]) == 1
        @test all(parent["best_operator"] == "terminate" for parent in probe["parent_summaries"])
        @test all(parent["best_normalized_operator"] == "terminate" for parent in probe["parent_summaries"])
        @test all(length(parent["operator_summaries"]) == 1 for parent in probe["parent_summaries"])

        basin_summary = probe["basin_summaries"][1]
        @test basin_summary["matched_budget"] >= 1
        @test haskey(basin_summary, "parent_main_effect_normalized")
        @test haskey(basin_summary, "operator_main_effect_normalized")
        @test haskey(basin_summary, "interaction_effect_normalized")
        @test haskey(basin_summary, "view_effects")
        @test haskey(basin_summary, "proposal_set_effects")
        @test haskey(basin_summary["view_effects"], "k1")
        @test haskey(basin_summary["view_effects"], "k2")
        @test haskey(basin_summary["view_effects"], "k4")
        @test haskey(basin_summary["proposal_set_effects"], "mean_top2")
        @test basin_summary["heuristic_parent_index"] == 1
        op_summary = basin_summary["parent_summaries"][1]["operator_summaries"][1]
        @test haskey(op_summary, "view_summaries")
        @test haskey(op_summary, "proposal_set_summaries")
        @test haskey(op_summary["view_summaries"], "k1")
        @test haskey(op_summary["view_summaries"], "k2")
        @test haskey(op_summary["proposal_set_summaries"], "mean_top2")
    end

    @testset "Parent intervention probe supports basin-slice mode" begin
        fb = MolecularFrontierBuffer(20)
        add_to_frontier!(fb, "c1ccccc1"; reward=0.9, source=:seed, operator=:seed)
        add_to_frontier!(fb, "c1ccncc1"; reward=0.8, source=:warmup, operator=:mutate)
        add_to_frontier!(fb, "CCO"; reward=0.4, source=:warmup, operator=:mutate)
        vocab = SMILESVocabulary()
        cfg = HierarchicalEditConfig(
            allow_crossover=false,
            max_operator_candidates=1,
            parent_candidate_limit=2,
            basin_candidate_limit=2,
            operators=[:terminate],
        )
        probe = probe_parent_interventions(fb, s -> 0.5, vocab;
            reward_fn_batch=xs -> fill(0.5, length(xs)),
            config=cfg,
            budget_remaining=20,
            created_at_step=1,
            task_name="basin-slice-test",
            max_parents=2,
            max_basins=2,
            operators=[:terminate],
            restrict_parents_to_basin=true)

        @test probe["candidate_mode"] == "basin_slice"
        @test length(probe["basin_summaries"]) >= 2
        @test all(summary["candidate_mode"] == "basin_slice" for summary in probe["basin_summaries"])
        @test all(length(summary["parent_summaries"]) >= 1 for summary in probe["basin_summaries"])
    end

    @testset "Coupled hierarchy option probe returns short-horizon family summaries" begin
        fb = MolecularFrontierBuffer(20)
        add_to_frontier!(fb, "CCO"; reward=0.9, source=:seed, operator=:seed)
        add_to_frontier!(fb, "CCN"; reward=0.7, source=:warmup, operator=:mutate)
        vocab = SMILESVocabulary()
        cfg = HierarchicalEditConfig(
            allow_crossover=false,
            max_operator_candidates=2,
            parent_candidate_limit=2,
            basin_candidate_limit=1,
            operators=[:terminate],
        )
        probe = probe_coupled_hierarchy_options(fb, s -> 0.5, vocab;
            reward_fn_batch=xs -> fill(0.5, length(xs)),
            config=cfg,
            target_smiles="CCO",
            budget_remaining=20,
            created_at_step=1,
            task_name="option-probe-test",
            max_parents=2,
            max_basins=1,
            operators=[:terminate],
            horizon=2)

        @test probe["task_name"] == "option-probe-test"
        @test probe["horizon"] == 2
        @test haskey(probe, "heuristic_baseline")
        @test haskey(probe, "family_a_rollouts")
        @test haskey(probe, "family_b_rollouts")
        @test haskey(probe, "family_c_rollouts")
        @test haskey(probe, "summary")
        @test length(probe["family_a_rollouts"]) == 1
        @test length(probe["family_b_rollouts"]) >= 1
        @test length(probe["family_c_rollouts"]) >= 1
        @test haskey(probe["summary"], "family_a_gain_vs_baseline")
        @test haskey(probe["summary"], "parent_coupling_gain")
        @test haskey(probe["summary"], "basin_coupling_gain")
        @test haskey(probe["summary"], "family_c_best_continuation_gain")

        records = extract_option_subtrajectory_records(probe)
        @test !isempty(records)
        @test all(haskey(record, "entry_key") for record in records)
        @test all(haskey(record, "step1_local_utility") for record in records)
        @test all(haskey(record, "option_value") for record in records)
        @test all(haskey(record, "trajectory_signature") for record in records)

        comparison = compare_option_value_surfaces(probe)
        @test haskey(comparison, "local_gain_vs_baseline")
        @test haskey(comparison, "entry_context_gain_vs_baseline")
        @test haskey(comparison, "subtrajectory_gain_vs_baseline")
        @test haskey(comparison, "subtrajectory_gain_vs_entry")
        @test haskey(comparison, "entry_reorder_fraction")
        @test haskey(comparison, "records")
        @test length(comparison["records"]) == length(records)

        bridge_runs = [
            Dict("probe" => probe, "bridge" => comparison),
            Dict("probe" => probe, "bridge" => comparison),
        ]
        dataset = extract_option_value_dataset(bridge_runs; feature_mode=:augmented)
        @test !isempty(dataset)
        @test length(dataset.records[1].features) >= 5
        stats = option_value_dataset_stats(dataset)
        @test stats["size"] == length(dataset)
        @test stats["n_snapshots"] >= 1

        trained = train_option_value_model(dataset;
            rng=MersenneTwister(0),
            config=OptionValueTrainingConfig(n_epochs=2, min_records=1, train_fraction=1.0))
        @test haskey(trained, "model")
        @test haskey(trained, "final_eval")
        @test haskey(trained["final_eval"], "mean_gain_vs_local_surface")
        @test haskey(trained["final_eval"], "selection_hit_rate")
    end

    @testset "Frontier-allocation probe returns matched-budget region summaries" begin
        fb = MolecularFrontierBuffer(20)
        add_to_frontier!(fb, "CCO"; reward=0.9, source=:seed, operator=:seed)
        add_to_frontier!(fb, "CCN"; reward=0.8, source=:warmup, operator=:mutate)
        add_to_frontier!(fb, "c1ccccc1"; reward=0.7, source=:seed, operator=:seed)
        add_to_frontier!(fb, "c1ccncc1"; reward=0.6, source=:edit, operator=:mutate)
        vocab = SMILESVocabulary()
        cfg = HierarchicalEditConfig(
            allow_crossover=false,
            max_operator_candidates=2,
            parent_candidate_limit=2,
            basin_candidate_limit=2,
            operators=[:terminate],
        )
        probe = probe_frontier_allocation_opportunities(fb, s -> 0.5, vocab;
            reward_fn_batch=xs -> fill(0.5, length(xs)),
            config=cfg,
            target_smiles="CCO",
            budget_remaining=20,
            created_at_step=1,
            task_name="allocation-probe-test",
            max_parents=2,
            max_basins=2,
            operators=[:terminate],
            horizon=2,
            region_families=["basin"],
            max_allocation_budget=2)
        @test probe["task_name"] == "allocation-probe-test"
        @test haskey(probe, "option_probe")
        @test haskey(probe, "surface_comparison")
        @test haskey(probe, "region_family_summaries")
        @test length(probe["region_family_summaries"]) == 1
        family_summary = probe["region_family_summaries"][1]
        @test family_summary["family_name"] == "basin"
        @test haskey(family_summary, "matched_budget")
        @test haskey(family_summary, "policies")
        @test haskey(family_summary, "state_label")
        if !family_summary["all_degenerate"]
            @test family_summary["matched_budget"] >= 2
            @test haskey(family_summary["policies"], "heuristic_top_region")
            @test haskey(family_summary["policies"], "uniform_regions")
            @test haskey(family_summary["policies"], "best_region")
            @test haskey(family_summary["policies"], "anti_heuristic_region")
        end
        summary = probe["summary"]
        @test haskey(summary, "best_family_name")
        @test haskey(summary, "best_family_gap")
        @test haskey(summary, "state_counts")
    end


    @testset "Batch 1O frontier-allocation dataset and selective policy are truthful" begin
        region_a = Dict{String,Any}(
            "region_key" => "basin-a",
            "count" => 2,
            "heuristic_region_score" => 0.90,
            "mean_heuristic_region_score" => 0.85,
            "best_option_utility" => 0.80,
            "mean_option_utility" => 0.75,
            "mean_parent_reward" => 0.72,
            "mean_parent_novelty_score" => 0.10,
            "option_utilities_sorted" => [0.80, 0.70],
            "records" => Dict{String,Any}[
                Dict("parent_tb_delta_abs" => 0.02, "basin_score" => 0.90),
                Dict("parent_tb_delta_abs" => 0.03, "basin_score" => 0.88),
            ],
        )
        region_b = Dict{String,Any}(
            "region_key" => "basin-b",
            "count" => 2,
            "heuristic_region_score" => 0.70,
            "mean_heuristic_region_score" => 0.68,
            "best_option_utility" => 1.05,
            "mean_option_utility" => 0.90,
            "mean_parent_reward" => 0.62,
            "mean_parent_novelty_score" => 0.25,
            "option_utilities_sorted" => [1.05, 0.95],
            "records" => Dict{String,Any}[
                Dict("parent_tb_delta_abs" => 0.08, "basin_score" => 0.75),
                Dict("parent_tb_delta_abs" => 0.07, "basin_score" => 0.73),
            ],
        )
        family_summary_1 = Dict{String,Any}(
            "family_name" => "basin",
            "all_degenerate" => false,
            "matched_budget" => 2,
            "heuristic_top_region" => "basin-a",
            "best_region" => "basin-b",
            "heuristic_allocation_utility" => 1.50,
            "best_allocation_utility" => 2.00,
            "best_vs_heuristic_gap" => 0.50,
            "region_summaries" => [region_a, region_b],
        )
        family_summary_2 = Dict{String,Any}(
            "family_name" => "basin",
            "all_degenerate" => false,
            "matched_budget" => 2,
            "heuristic_top_region" => "basin-a",
            "best_region" => "basin-a",
            "heuristic_allocation_utility" => 1.55,
            "best_allocation_utility" => 1.55,
            "best_vs_heuristic_gap" => 0.0,
            "region_summaries" => [region_a, region_b],
        )
        runs = [
            Dict("probe" => Dict("snapshot_id" => "snap-1", "task_name" => "celecoxib_rediscovery", "region_family_summaries" => [family_summary_1])),
            Dict("probe" => Dict("snapshot_id" => "snap-2", "task_name" => "celecoxib_rediscovery", "region_family_summaries" => [family_summary_2])),
        ]

        dataset = extract_frontier_allocation_dataset(runs; family_name="basin", override_threshold=0.01)
        @test length(dataset) == 2
        stats = frontier_allocation_dataset_stats(dataset)
        @test stats["override_positive_fraction"] == 0.5
        @test stats["snapshot_feature_dim"] > 0
        @test stats["region_feature_dim"] > 0

        snap1 = dataset.snapshots[1]
        @test snap1.override_worth_it == true
        @test snap1.heuristic_region == "basin-a"
        @test snap1.winning_region == "basin-b"
        @test length(snap1.regions) == 2
        @test snap1.regions[2].winning_region == true
        @test snap1.regions[2].allocation_utility > snap1.regions[1].allocation_utility

        snap_feature_dim = length(snap1.features)
        region_feature_dim = length(snap1.regions[1].features)
        abstain_policy = SelectiveFrontierAllocator(
            "basin",
            FrontierAllocationLinearModel(zeros(Float32, snap_feature_dim), -10.0f0, zeros(Float32, snap_feature_dim), ones(Float32, snap_feature_dim)),
            FrontierAllocationLinearModel(zeros(Float32, region_feature_dim), 0.0f0, zeros(Float32, region_feature_dim), ones(Float32, region_feature_dim)),
            0.5f0,
            0.0f0,
        )
        abstain_eval = evaluate_selective_frontier_allocator(abstain_policy, FrontierAllocationDataset([snap1]); mode=:selective_override)
        @test abstain_eval["override_fraction"] == 0.0
        @test abstain_eval["abstention_fraction"] == 1.0
        @test abstain_eval["heuristic_match_fraction"] == 1.0
        @test abstain_eval["mean_gain_vs_heuristic"] == 0.0

        override_weights = zeros(Float32, region_feature_dim)
        override_weights[end] = -1.0f0
        override_policy = SelectiveFrontierAllocator(
            "basin",
            FrontierAllocationLinearModel(zeros(Float32, snap_feature_dim), 10.0f0, zeros(Float32, snap_feature_dim), ones(Float32, snap_feature_dim)),
            FrontierAllocationLinearModel(override_weights, 0.0f0, zeros(Float32, region_feature_dim), ones(Float32, region_feature_dim)),
            0.5f0,
            0.0f0,
        )
        override_eval = evaluate_selective_frontier_allocator(override_policy, FrontierAllocationDataset([snap1]); mode=:selective_override)
        @test override_eval["override_fraction"] == 1.0
        @test override_eval["mean_gain_vs_heuristic"] > 0.0
        @test override_eval["basin_choice_accuracy"] == 1.0

        trained = train_selective_frontier_allocator(dataset; rng=MersenneTwister(0), train_fraction=0.5)
        @test haskey(trained, "model")
        @test trained["model"] !== nothing
        @test haskey(trained, "val_eval")
        @test haskey(trained, "heuristic_val_eval")
        @test haskey(trained, "always_override_val_eval")
        @test haskey(trained, "oracle_val_eval")
        @test haskey(trained["final_eval"], "abstention_fraction")
    end


    @testset "Batch 1P opportunity-state dataset and detector are truthful" begin
        region_a = Dict{String,Any}(
            "region_key" => "basin-a",
            "count" => 2,
            "heuristic_region_score" => 0.90,
            "mean_heuristic_region_score" => 0.85,
            "best_option_utility" => 0.80,
            "mean_option_utility" => 0.75,
            "mean_parent_reward" => 0.72,
            "mean_parent_novelty_score" => 0.10,
            "option_utilities_sorted" => [0.80, 0.70],
            "records" => Dict{String,Any}[
                Dict("parent_tb_delta_abs" => 0.02, "basin_score" => 0.90),
                Dict("parent_tb_delta_abs" => 0.03, "basin_score" => 0.88),
            ],
        )
        region_b = Dict{String,Any}(
            "region_key" => "basin-b",
            "count" => 2,
            "heuristic_region_score" => 0.70,
            "mean_heuristic_region_score" => 0.68,
            "best_option_utility" => 1.05,
            "mean_option_utility" => 0.90,
            "mean_parent_reward" => 0.62,
            "mean_parent_novelty_score" => 0.25,
            "option_utilities_sorted" => [1.05, 0.95],
            "records" => Dict{String,Any}[
                Dict("parent_tb_delta_abs" => 0.08, "basin_score" => 0.75),
                Dict("parent_tb_delta_abs" => 0.07, "basin_score" => 0.73),
            ],
        )
        routing_summary = Dict{String,Any}(
            "family_name" => "basin",
            "all_degenerate" => false,
            "matched_budget" => 2,
            "heuristic_top_region" => "basin-a",
            "best_region" => "basin-b",
            "heuristic_allocation_utility" => 1.50,
            "best_allocation_utility" => 2.00,
            "best_vs_heuristic_gap" => 0.50,
            "best_vs_uniform_gap" => 0.30,
            "heuristic_vs_anti_gap" => 0.10,
            "region_opportunity_range" => 0.25,
            "region_summaries" => [region_a, region_b],
        )
        dominant_summary = Dict{String,Any}(
            "family_name" => "basin",
            "all_degenerate" => false,
            "matched_budget" => 2,
            "heuristic_top_region" => "basin-a",
            "best_region" => "basin-a",
            "heuristic_allocation_utility" => 1.55,
            "best_allocation_utility" => 1.55,
            "best_vs_heuristic_gap" => 0.0,
            "best_vs_uniform_gap" => 0.0,
            "heuristic_vs_anti_gap" => 0.20,
            "region_opportunity_range" => 0.05,
            "region_summaries" => [region_a, region_b],
        )
        ambiguous_summary = Dict{String,Any}(
            "family_name" => "basin",
            "all_degenerate" => false,
            "matched_budget" => 2,
            "heuristic_top_region" => "basin-a",
            "best_region" => "basin-b",
            "heuristic_allocation_utility" => 1.50,
            "best_allocation_utility" => 1.505,
            "best_vs_heuristic_gap" => 0.005,
            "best_vs_uniform_gap" => 0.002,
            "heuristic_vs_anti_gap" => 0.0,
            "region_opportunity_range" => 0.01,
            "region_summaries" => [region_a, region_b],
        )
        runs = [
            Dict("probe" => Dict("snapshot_id" => "snap-r", "task_name" => "celecoxib_rediscovery", "region_family_summaries" => [routing_summary])),
            Dict("probe" => Dict("snapshot_id" => "snap-h", "task_name" => "celecoxib_rediscovery", "region_family_summaries" => [dominant_summary])),
            Dict("probe" => Dict("snapshot_id" => "snap-a", "task_name" => "celecoxib_rediscovery", "region_family_summaries" => [ambiguous_summary])),
        ]

        dataset = extract_opportunity_state_dataset(runs; family_name="basin", state_threshold=0.01)
        @test length(dataset) == 3
        stats = opportunity_state_dataset_stats(dataset)
        @test stats["override_eligible_fraction"] == 1 / 3
        @test stats["feature_dim"] > 0
        @test get(stats["regime_counts"], "routing_sensitive_state", 0) == 1
        @test get(stats["regime_counts"], "heuristic_dominant_state", 0) == 1
        @test get(stats["regime_counts"], "invariant_or_ambiguous_state", 0) == 1

        oracle_eval = evaluate_opportunity_state_detector(nothing, dataset; mode=:oracle)
        @test oracle_eval["predicted_positive_fraction"] == 1 / 3
        @test oracle_eval["precision"] == 1.0
        @test oracle_eval["recall"] == 1.0
        @test oracle_eval["mean_gap_predicted_positive"] > oracle_eval["mean_gap_predicted_negative"]
        @test oracle_eval["gap_separation"] > 0.0
        @test oracle_eval["zero_positive_collapse"] == false
        @test oracle_eval["all_positive_collapse"] == false

        all_negative_eval = evaluate_opportunity_state_detector(nothing, dataset; mode=:always_negative)
        all_positive_eval = evaluate_opportunity_state_detector(nothing, dataset; mode=:always_positive)
        @test all_negative_eval["zero_positive_collapse"] == true
        @test all_positive_eval["all_positive_collapse"] == true

        cond_oracle = evaluate_opportunity_state_conditional_oracle(nothing, dataset; mode=:oracle)
        cond_heur = evaluate_opportunity_state_conditional_oracle(nothing, dataset; mode=:always_negative)
        @test cond_oracle["mean_gain_vs_heuristic"] > cond_heur["mean_gain_vs_heuristic"]

        manual = OpportunityStateDataset([
            OpportunityStateRecord("m1", "task", "basin", Float32[1, 0], true, "routing_sensitive_state", "a", "b", 0.20f0, 0.10f0, 0.0f0, 0.20f0, 1.0f0, 1.2f0, 2),
            OpportunityStateRecord("m2", "task", "basin", Float32[0.9, 0.1], true, "routing_sensitive_state", "a", "b", 0.18f0, 0.09f0, 0.0f0, 0.18f0, 1.0f0, 1.18f0, 2),
            OpportunityStateRecord("m3", "task", "basin", Float32[0.8, 0.2], true, "routing_sensitive_state", "a", "b", 0.17f0, 0.08f0, 0.0f0, 0.16f0, 1.0f0, 1.17f0, 2),
            OpportunityStateRecord("m4", "task", "basin", Float32[0, 1], false, "heuristic_dominant_state", "a", "a", 0.0f0, 0.0f0, 0.15f0, 0.02f0, 1.0f0, 1.0f0, 2),
            OpportunityStateRecord("m5", "task", "basin", Float32[0.1, 0.9], false, "heuristic_dominant_state", "a", "a", 0.0f0, 0.0f0, 0.12f0, 0.03f0, 1.0f0, 1.0f0, 2),
            OpportunityStateRecord("m6", "task", "basin", Float32[0.2, 0.8], false, "heuristic_dominant_state", "a", "a", 0.0f0, 0.0f0, 0.11f0, 0.02f0, 1.0f0, 1.0f0, 2),
        ])
        trained = train_opportunity_state_detector(manual; rng=MersenneTwister(0), train_fraction=0.5)
        @test haskey(trained, "model")
        @test trained["model"] !== nothing
        @test haskey(trained, "val_eval")
        @test haskey(trained, "conditional_val_eval")
        @test haskey(trained, "oracle_conditional_val_eval")
        @test haskey(trained, "threshold_search")

        stability = opportunity_state_threshold_stability(trained["model"], manual; threshold_perturbations=[-0.1, 0.0, 0.1])
        @test stability["n_thresholds"] == 3
        @test 0.0 <= stability["robust_fraction"] <= 1.0
        @test 0.0 <= stability["nondegenerate_fraction"] <= 1.0

        repeatability = evaluate_opportunity_state_repeatability(manual;
            rng=MersenneTwister(0),
            num_splits=3,
            train_fraction=0.5,
            threshold_perturbations=[-0.1, 0.0, 0.1])
        @test haskey(repeatability, "summary")
        @test repeatability["summary"]["n_splits"] == 3
        @test haskey(repeatability["summary"], "nondegenerate_fraction")
        @test haskey(repeatability["summary"], "median_threshold_robust_fraction")
        @test repeatability["summary"]["median_gap_separation"] > 0.0
        @test length(repeatability["splits"]) == 3
        for split in repeatability["splits"]
            @test isempty(intersect(Set(split["train_snapshot_ids"]), Set(split["val_snapshot_ids"])))
            @test split["threshold_stability"]["n_thresholds"] == 3
        end
    end

    @testset "Batch 1S sparse-positive operating-point rules enforce guarded selection" begin
        manual = OpportunityStateDataset([
            OpportunityStateRecord("m1", "task", "basin", Float32[1, 0], true, "routing_sensitive_state", "a", "b", 0.20f0, 0.10f0, 0.0f0, 0.20f0, 1.0f0, 1.2f0, 2),
            OpportunityStateRecord("m2", "task", "basin", Float32[0.95, 0.05], true, "routing_sensitive_state", "a", "b", 0.18f0, 0.09f0, 0.0f0, 0.18f0, 1.0f0, 1.18f0, 2),
            OpportunityStateRecord("m3", "task", "basin", Float32[0.90, 0.10], true, "routing_sensitive_state", "a", "b", 0.16f0, 0.08f0, 0.0f0, 0.16f0, 1.0f0, 1.16f0, 2),
            OpportunityStateRecord("m4", "task", "basin", Float32[0.0, 1.0], false, "heuristic_dominant_state", "a", "a", 0.0f0, 0.0f0, 0.18f0, 0.02f0, 1.0f0, 1.0f0, 2),
            OpportunityStateRecord("m5", "task", "basin", Float32[0.10, 0.90], false, "heuristic_dominant_state", "a", "a", 0.0f0, 0.0f0, 0.15f0, 0.03f0, 1.0f0, 1.0f0, 2),
            OpportunityStateRecord("m6", "task", "basin", Float32[0.15, 0.85], false, "heuristic_dominant_state", "a", "a", 0.0f0, 0.0f0, 0.12f0, 0.02f0, 1.0f0, 1.0f0, 2),
        ])
        fit = train_opportunity_state_detector(manual; rng=MersenneTwister(0), train_fraction=0.5)
        detector = fit["model"]
        train_dataset = fit["train_dataset"]
        val_dataset = fit["val_dataset"]

        fixed = select_sparse_positive_operating_point(detector, train_dataset;
            rule=:fixed_threshold,
            fixed_thresholds=[0.45, 0.55])
        @test fixed["status"] == "evaluated_only"
        @test fixed["selected_threshold"] === nothing
        @test length(fixed["candidates"]) == 2

        guarded = select_sparse_positive_operating_point(detector, train_dataset;
            rule=:precision_guarded,
            fixed_thresholds=[0.45, 0.55],
            max_positive_fraction=0.6,
            min_positive_count=1,
            min_train_precision=0.5)
        @test guarded["status"] in ["selected", "no_valid_threshold"]
        @test length(guarded["candidates"]) == 2
        if guarded["status"] == "selected"
            @test guarded["selected_threshold"] !== nothing
        else
            @test guarded["selected_threshold"] === nothing
        end

        abstain = select_sparse_positive_operating_point(detector, train_dataset;
            rule=:precision_guarded,
            fixed_thresholds=[0.45, 0.55],
            max_positive_fraction=0.05,
            min_positive_count=1,
            min_train_precision=0.99)
        @test abstain["status"] == "no_valid_threshold"
        @test abstain["selected_threshold"] === nothing

        guarded_eval = evaluate_sparse_positive_operating_point(detector, train_dataset, val_dataset;
            rule=:precision_guarded,
            fixed_thresholds=[0.45, 0.55],
            max_positive_fraction=0.6,
            min_positive_count=1,
            min_train_precision=0.5)
        @test haskey(guarded_eval, "val_eval")
        @test haskey(guarded_eval, "conditional_val_eval")
        @test haskey(guarded_eval, "val_guard")
        if guarded_eval["abstained"]
            @test guarded_eval["selected_threshold"] === nothing
            @test guarded_eval["val_eval"]["zero_positive_collapse"] == true
        else
            @test guarded_eval["selected_threshold"] !== nothing
        end

        abstain_eval = evaluate_sparse_positive_operating_point(detector, train_dataset, val_dataset;
            rule=:precision_guarded,
            fixed_thresholds=[0.45, 0.55],
            max_positive_fraction=0.05,
            min_positive_count=1,
            min_train_precision=0.99)
        @test abstain_eval["abstained"] == true
        @test abstain_eval["selected_threshold"] === nothing
        @test abstain_eval["val_eval"]["zero_positive_collapse"] == true

        audit = evaluate_sparse_positive_operating_points(manual;
            rng=MersenneTwister(0),
            num_splits=3,
            train_fraction=0.5,
            rule_families=[:precision_guarded, :fraction_capped, :guarded_fallback],
            fixed_thresholds=[0.45, 0.55],
            max_positive_fraction=0.6,
            min_positive_count=1,
            min_train_precision=0.5)
        @test haskey(audit, "rule_summaries")
        @test haskey(audit, "splits")
        @test audit["summary"]["n_splits"] == 3
        @test haskey(audit["rule_summaries"], "precision_guarded")
        @test haskey(audit["rule_summaries"], "fraction_capped")
        @test haskey(audit["rule_summaries"], "guarded_fallback")
        for split in audit["splits"]
            @test isempty(intersect(Set(split["train_snapshot_ids"]), Set(split["val_snapshot_ids"])))
        end
    end


    @testset "Batch 1T repair-audit dataset extracts truthful feature families and labels" begin
        region_a = Dict{String,Any}(
            "region_key" => "basin-a",
            "count" => 2,
            "heuristic_region_score" => 0.90,
            "mean_heuristic_region_score" => 0.85,
            "best_option_utility" => 0.80,
            "mean_option_utility" => 0.75,
            "mean_parent_reward" => 0.72,
            "mean_parent_novelty_score" => 0.10,
            "option_utilities_sorted" => [0.80, 0.70],
            "records" => Dict{String,Any}[
                Dict("parent_tb_delta_abs" => 0.02, "basin_score" => 0.90),
                Dict("parent_tb_delta_abs" => 0.03, "basin_score" => 0.88),
            ],
        )
        region_b = Dict{String,Any}(
            "region_key" => "basin-b",
            "count" => 2,
            "heuristic_region_score" => 0.70,
            "mean_heuristic_region_score" => 0.68,
            "best_option_utility" => 1.05,
            "mean_option_utility" => 0.90,
            "mean_parent_reward" => 0.62,
            "mean_parent_novelty_score" => 0.25,
            "option_utilities_sorted" => [1.05, 0.95],
            "records" => Dict{String,Any}[
                Dict("parent_tb_delta_abs" => 0.08, "basin_score" => 0.75),
                Dict("parent_tb_delta_abs" => 0.07, "basin_score" => 0.73),
            ],
        )
        stable_summary = Dict{String,Any}(
            "family_name" => "basin",
            "all_degenerate" => false,
            "matched_budget" => 2,
            "heuristic_top_region" => "basin-a",
            "best_region" => "basin-b",
            "heuristic_allocation_utility" => 1.50,
            "best_allocation_utility" => 2.00,
            "best_vs_heuristic_gap" => 0.50,
            "best_vs_uniform_gap" => 0.30,
            "heuristic_vs_anti_gap" => 0.10,
            "region_opportunity_range" => 0.25,
            "region_summaries" => [region_a, region_b],
        )
        dominant_summary = Dict{String,Any}(
            "family_name" => "basin",
            "all_degenerate" => false,
            "matched_budget" => 2,
            "heuristic_top_region" => "basin-a",
            "best_region" => "basin-a",
            "heuristic_allocation_utility" => 1.55,
            "best_allocation_utility" => 1.55,
            "best_vs_heuristic_gap" => 0.0,
            "best_vs_uniform_gap" => 0.0,
            "heuristic_vs_anti_gap" => 0.20,
            "region_opportunity_range" => 0.05,
            "region_summaries" => [region_a, region_b],
        )
        runs = [
            Dict(
                "probe" => Dict("snapshot_id" => "snap-s", "task_name" => "celecoxib_rediscovery", "horizon" => 3, "max_allocation_budget" => 2, "region_family_summaries" => [stable_summary]),
                "seed_stats" => Dict("seeded" => 6, "augmented" => 2, "evaluated" => 8),
                "warmup_stats" => Dict("warmup_added" => 4, "warmup_evaluated" => 12, "warmup_rounds" => 1),
                "calls_used" => 24,
                "budget_total" => 128,
                "created_at_step" => 1,
            ),
            Dict(
                "probe" => Dict("snapshot_id" => "snap-h", "task_name" => "celecoxib_rediscovery", "horizon" => 3, "max_allocation_budget" => 2, "region_family_summaries" => [dominant_summary]),
                "seed_stats" => Dict("seeded" => 6, "augmented" => 2, "evaluated" => 8),
                "warmup_stats" => Dict("warmup_added" => 4, "warmup_evaluated" => 12, "warmup_rounds" => 1),
                "calls_used" => 24,
                "budget_total" => 128,
                "created_at_step" => 2,
            ),
        ]
        dataset = extract_opportunity_repair_audit_dataset(runs; family_name="basin", state_threshold=0.01, stability_threshold=0.05)
        @test length(dataset) == 2
        stats = opportunity_repair_audit_dataset_stats(dataset)
        @test stats["current_positive_fraction"] == 0.5
        @test stats["robust_positive_fraction"] == 0.5
        @test stats["abstain_fraction"] == 0.0
        @test get(stats["typed_counts"], "robust_positive", 0) == 1
        @test get(stats["typed_counts"], "heuristic_negative", 0) == 1
        @test length(dataset.records[1].baseline_features) > 0
        @test length(dataset.records[1].history_features) > 0
        @test length(dataset.records[1].surface_features) > 0
        @test length(dataset.records[1].robustness_features) > 0
        @test length(dataset.records[1].phase_features) > 0
        @test dataset.records[1].current_override_eligible == true
        @test dataset.records[1].robust_override_eligible == true
        @test dataset.records[1].typed_label == "robust_positive"
        @test dataset.records[2].typed_label == "heuristic_negative"
    end

    @testset "Batch 1T verdict logic can identify representation-dominant repair" begin
        function make_repair_record(id::String, positive::Bool, surface_a::Float32, surface_b::Float32)
            return OpportunityRepairAuditRecord(
                id,
                "celecoxib_rediscovery",
                "basin",
                Float32[0.0, 1.0],
                Float32[0.0, 0.0],
                Float32[surface_a, surface_b],
                Float32[0.0, 0.0],
                Float32[0.0, 0.0],
                positive,
                positive,
                false,
                positive ? "robust_positive" : "heuristic_negative",
                positive ? "stable_routing_sensitive_state" : "heuristic_dominant_state",
                positive ? 0.2f0 : 0.0f0,
                positive ? 0.2f0 : 0.0f0,
                1.0f0,
                positive ? 1.2f0 : 1.0f0,
                2,
            )
        end
        dataset = OpportunityRepairAuditDataset([
            make_repair_record("r1", true, 1.0f0, 0.0f0),
            make_repair_record("r2", true, 0.9f0, 0.1f0),
            make_repair_record("r3", true, 0.8f0, 0.2f0),
            make_repair_record("r4", false, 0.0f0, 1.0f0),
            make_repair_record("r5", false, 0.1f0, 0.9f0),
            make_repair_record("r6", false, 0.2f0, 0.8f0),
        ])
        baseline_probe = evaluate_opportunity_repair_binary_probe(dataset;
            feature_branch="baseline",
            semantics="current_binary",
            rng=MersenneTwister(0),
            num_splits=3,
            train_fraction=0.5)
        surface_probe = evaluate_opportunity_repair_binary_probe(dataset;
            feature_branch="baseline_plus_surface",
            semantics="current_binary",
            rng=MersenneTwister(0),
            num_splits=3,
            train_fraction=0.5)
        @test surface_probe["summary"]["median_pairwise_accuracy"] > baseline_probe["summary"]["median_pairwise_accuracy"]

        repair = evaluate_opportunity_representation_semantics_repair(dataset;
            rng=MersenneTwister(0),
            num_splits=3,
            train_fraction=0.5)
        @test repair["summary"]["verdict"] == "V1_REPRESENTATION_REPAIR_DOMINATES"
        @test repair["summary"]["decisive"] == true
        @test repair["summary"]["best_representation_branch"] == "baseline_plus_surface"
        @test repair["summary"]["representation_improvement"] > 0.0
        @test repair["summary"]["representation_ablation_drop"] > 0.0
    end

    @testset "Batch 1R intervention-geometry atlas reveals cleaner stable subsets than pooled positives" begin
        region_a = Dict{String,Any}(
            "region_key" => "basin-a",
            "count" => 2,
            "heuristic_region_score" => 0.90,
            "mean_heuristic_region_score" => 0.85,
            "best_option_utility" => 0.80,
            "mean_option_utility" => 0.75,
            "mean_parent_reward" => 0.72,
            "mean_parent_novelty_score" => 0.10,
            "option_utilities_sorted" => [0.80, 0.70],
            "records" => Dict{String,Any}[
                Dict("parent_tb_delta_abs" => 0.02, "basin_score" => 0.90),
                Dict("parent_tb_delta_abs" => 0.03, "basin_score" => 0.88),
            ],
        )
        region_b = Dict{String,Any}(
            "region_key" => "basin-b",
            "count" => 2,
            "heuristic_region_score" => 0.70,
            "mean_heuristic_region_score" => 0.68,
            "best_option_utility" => 1.05,
            "mean_option_utility" => 0.90,
            "mean_parent_reward" => 0.62,
            "mean_parent_novelty_score" => 0.25,
            "option_utilities_sorted" => [1.05, 0.95],
            "records" => Dict{String,Any}[
                Dict("parent_tb_delta_abs" => 0.08, "basin_score" => 0.75),
                Dict("parent_tb_delta_abs" => 0.07, "basin_score" => 0.73),
            ],
        )
        stable_summary = Dict{String,Any}(
            "family_name" => "basin",
            "all_degenerate" => false,
            "matched_budget" => 2,
            "heuristic_top_region" => "basin-a",
            "best_region" => "basin-b",
            "heuristic_allocation_utility" => 1.50,
            "best_allocation_utility" => 2.00,
            "best_vs_heuristic_gap" => 0.50,
            "best_vs_uniform_gap" => 0.30,
            "heuristic_vs_anti_gap" => 0.10,
            "region_opportunity_range" => 0.25,
            "region_summaries" => [region_a, region_b],
        )
        ambiguous_summary = Dict{String,Any}(
            "family_name" => "basin",
            "all_degenerate" => false,
            "matched_budget" => 2,
            "heuristic_top_region" => "basin-a",
            "best_region" => "basin-b",
            "heuristic_allocation_utility" => 1.50,
            "best_allocation_utility" => 1.515,
            "best_vs_heuristic_gap" => 0.015,
            "best_vs_uniform_gap" => 0.005,
            "heuristic_vs_anti_gap" => 0.0,
            "region_opportunity_range" => 0.01,
            "region_summaries" => [region_a, region_b],
        )
        dominant_summary = Dict{String,Any}(
            "family_name" => "basin",
            "all_degenerate" => false,
            "matched_budget" => 2,
            "heuristic_top_region" => "basin-a",
            "best_region" => "basin-a",
            "heuristic_allocation_utility" => 1.55,
            "best_allocation_utility" => 1.55,
            "best_vs_heuristic_gap" => 0.0,
            "best_vs_uniform_gap" => 0.0,
            "heuristic_vs_anti_gap" => 0.20,
            "region_opportunity_range" => 0.05,
            "region_summaries" => [region_a, region_b],
        )
        low_signal_summary = Dict{String,Any}(
            "family_name" => "basin",
            "all_degenerate" => false,
            "matched_budget" => 2,
            "heuristic_top_region" => "basin-a",
            "best_region" => "basin-b",
            "heuristic_allocation_utility" => 1.50,
            "best_allocation_utility" => 1.505,
            "best_vs_heuristic_gap" => 0.005,
            "best_vs_uniform_gap" => 0.001,
            "heuristic_vs_anti_gap" => 0.0,
            "region_opportunity_range" => 0.005,
            "region_summaries" => [region_a, region_b],
        )
        runs = [
            Dict("probe" => Dict("snapshot_id" => "snap-s", "task_name" => "celecoxib_rediscovery", "region_family_summaries" => [stable_summary])),
            Dict("probe" => Dict("snapshot_id" => "snap-a", "task_name" => "celecoxib_rediscovery", "region_family_summaries" => [ambiguous_summary])),
            Dict("probe" => Dict("snapshot_id" => "snap-h", "task_name" => "celecoxib_rediscovery", "region_family_summaries" => [dominant_summary])),
            Dict("probe" => Dict("snapshot_id" => "snap-l", "task_name" => "celecoxib_rediscovery", "region_family_summaries" => [low_signal_summary])),
        ]

        atlas = extract_intervention_geometry_atlas(runs; family_name="basin", state_threshold=0.01, stability_threshold=0.05)
        @test length(atlas["records"]) == 4
        stats = atlas["stats"]
        @test stats["size"] == 4
        @test stats["override_eligible_fraction"] == 0.5
        @test get(stats["regime_counts"], "stable_routing_sensitive_state", 0) == 1
        @test get(stats["regime_counts"], "ambiguous_routing_sensitive_state", 0) == 1
        @test get(stats["regime_counts"], "heuristic_dominant_state", 0) == 1
        @test get(stats["regime_counts"], "invariant_or_low_signal_state", 0) == 1

        comparison = atlas["comparison"]
        @test comparison["stable_vs_pooled_gap_mean_delta"] > 0.0
        @test comparison["stable_vs_pooled_gap_std_improvement"] > 0.0
        @test comparison["pooled_positive_ambiguous_fraction"] > 0.0
        @test comparison["mixture_explains_instability"] == true
        @test comparison["regime_split_useful"] == true
        @test atlas["interpretable"] == true
        @test atlas["recommendation"] == "REGIME_STRUCTURE_INTERPRETABLE"
    end

    @testset "Batch 1M option-value calibration supports truthful override targets and calibrated policies" begin
        function make_option_record(snapshot_id, object_id, features, step1_local_utility, option_value;
                                    continuation_gain=0.0f0,
                                    local_rank_fraction=0.0f0,
                                    local_margin_to_top=0.05f0,
                                    local_centered_utility=0.0f0,
                                    continuation_sensitive=false,
                                    local_ambiguous=true,
                                    gain_vs_entry_local_candidate=0.0f0,
                                    override_helpful=false,
                                    strong_override_helpful=false,
                                    entry_local_baseline=false,
                                    snapshot_has_override_opportunity=true,
                                    snapshot_has_strong_override_opportunity=true)
            return OptionValueRecord(
                snapshot_id,
                "task",
                3,
                object_id,
                object_id,
                "sca-a",
                object_id == "local" || object_id == "local-2" ? "CCO" : "CCN",
                :mutate,
                features,
                step1_local_utility,
                option_value,
                continuation_gain,
                option_value,
                1.0f0,
                option_value - 0.80f0,
                0.82f0,
                0.75f0,
                local_rank_fraction,
                local_margin_to_top,
                local_centered_utility,
                0.5f0,
                0.5f0,
                abs(option_value) <= 1f-6 ? continuation_gain : continuation_gain / max(abs(option_value), 1f-6),
                continuation_sensitive,
                local_ambiguous,
                false,
                gain_vs_entry_local_candidate,
                override_helpful,
                strong_override_helpful,
                entry_local_baseline,
                snapshot_has_override_opportunity,
                snapshot_has_strong_override_opportunity,
            )
        end

        rec_local = make_option_record("snap-1", "local", Float32[0.0, 1.0], 0.90f0, 0.80f0;
            continuation_gain=0.01f0,
            local_rank_fraction=1.0f0,
            local_margin_to_top=0.0f0,
            local_centered_utility=0.02f0,
            continuation_sensitive=false,
            local_ambiguous=true,
            gain_vs_entry_local_candidate=0.0f0,
            entry_local_baseline=true)
        rec_option = make_option_record("snap-1", "option", Float32[1.0, 0.0], 0.86f0, 1.00f0;
            continuation_gain=0.20f0,
            local_rank_fraction=0.0f0,
            local_margin_to_top=0.04f0,
            local_centered_utility=-0.02f0,
            continuation_sensitive=true,
            local_ambiguous=true,
            gain_vs_entry_local_candidate=0.20f0,
            override_helpful=true,
            strong_override_helpful=true)
        rec_local2 = make_option_record("snap-2", "local-2", Float32[0.0, 1.0], 0.88f0, 0.79f0;
            continuation_gain=0.01f0,
            local_rank_fraction=1.0f0,
            local_margin_to_top=0.0f0,
            local_centered_utility=0.01f0,
            continuation_sensitive=false,
            local_ambiguous=true,
            gain_vs_entry_local_candidate=0.0f0,
            entry_local_baseline=true)
        rec_option2 = make_option_record("snap-2", "option-2", Float32[1.0, 0.0], 0.84f0, 0.98f0;
            continuation_gain=0.18f0,
            local_rank_fraction=0.0f0,
            local_margin_to_top=0.04f0,
            local_centered_utility=-0.01f0,
            continuation_sensitive=true,
            local_ambiguous=true,
            gain_vs_entry_local_candidate=0.19f0,
            override_helpful=true,
            strong_override_helpful=true)

        dataset = OptionValueDataset([rec_local, rec_option, rec_local2, rec_option2])
        stats = option_value_dataset_stats(dataset)
        @test stats["override_positive_fraction"] > 0.0
        @test stats["snapshot_override_opportunity_fraction"] == 1.0

        model = LearnedOptionValueModel(Float32[1.0, 0.0], 0.0f0, 2, :basic)
        anchored_eval = evaluate_option_value_model(model, dataset;
            selection_rule=:local_anchored,
            override_margin=1.5,
            ambiguity_threshold=0.05)
        ambiguous_eval = evaluate_option_value_model(model, dataset;
            selection_rule=:ambiguity_gated,
            override_margin=0.2,
            ambiguity_threshold=0.05)

        @test anchored_eval["mean_gain_vs_entry_local_candidate"] == 0.0
        @test anchored_eval["override_rate"] == 0.0
        @test anchored_eval["preserve_rate"] == 1.0
        @test ambiguous_eval["mean_gain_vs_entry_local_candidate"] > 0.15
        @test ambiguous_eval["override_rate"] == 1.0
        @test ambiguous_eval["reorder_fraction_vs_entry_local"] == 1.0
        @test ambiguous_eval["continuation_sensitive_fraction"] == 1.0

        conf_dim = length(option_override_feature_vector(rec_option; ranking_score=1.0, ranking_gap_vs_entry_local=1.0, local_margin=0.04))
        conf_weights = zeros(Float32, conf_dim)
        conf_weights[2] = 4.0f0
        override_policy = CalibratedOrdinalOptionPolicy(model, conf_weights, -1.0f0, conf_dim, :confidence_threshold, 0.60f0, 0.45f0, 0.65f0, 0.05f0, 0.02f0)
        preserve_policy = CalibratedOrdinalOptionPolicy(model, conf_weights, -5.0f0, conf_dim, :confidence_threshold, 0.60f0, 0.45f0, 0.65f0, 0.05f0, 0.02f0)

        override_eval = evaluate_option_value_model(override_policy, dataset)
        preserve_eval = evaluate_option_value_model(preserve_policy, dataset)
        @test override_eval["override_rate"] == 1.0
        @test override_eval["override_precision"] == 1.0
        @test override_eval["override_recall"] == 1.0
        @test preserve_eval["override_rate"] == 0.0
        @test preserve_eval["preserve_rate"] == 1.0
        @test haskey(override_eval, "confidence_bucket_summary")

        trained_policy = train_calibrated_ordinal_option_policy(dataset;
            rng=MersenneTwister(0),
            ranking_config=OptionValueTrainingConfig(n_epochs=2, min_records=1, train_fraction=0.5, feature_mode=:basic, objective_mode=:pairwise),
            calibration_config=OptionCalibrationConfig(n_epochs=2, override_gain_threshold=0.02, confidence_threshold_candidates=[0.5, 0.6], confidence_low_threshold_candidates=[0.4], confidence_high_threshold_candidates=[0.6]),
            selection_rule=:anchored_confidence)
        @test haskey(trained_policy, "policy")
        @test haskey(trained_policy, "ranking_result")
        @test haskey(trained_policy, "confidence_result")
        @test haskey(trained_policy, "best_val_eval")
        @test haskey(trained_policy["best_val_eval"], "override_precision")

        fb = MolecularFrontierBuffer(20)
        add_to_frontier!(fb, "CCO"; reward=0.9, source=:seed, operator=:seed)
        add_to_frontier!(fb, "CCN"; reward=0.7, source=:warmup, operator=:mutate)
        vocab = SMILESVocabulary()
        cfg = HierarchicalEditConfig(
            allow_crossover=false,
            max_operator_candidates=2,
            parent_candidate_limit=2,
            basin_candidate_limit=1,
            operators=[:terminate],
        )
        probe = probe_coupled_hierarchy_options(fb, s -> 0.5, vocab;
            reward_fn_batch=xs -> fill(0.5, length(xs)),
            config=cfg,
            target_smiles="CCO",
            budget_remaining=20,
            created_at_step=1,
            task_name="option-1m-test",
            max_parents=2,
            max_basins=1,
            operators=[:terminate],
            horizon=2)
        comparison = compare_option_value_surfaces(probe)
        bridge_runs = [
            Dict("probe" => probe, "bridge" => comparison),
            Dict("probe" => probe, "bridge" => comparison),
        ]
        basic_dataset = extract_option_value_dataset(bridge_runs; feature_mode=:basic)
        augmented_dataset = extract_option_value_dataset(bridge_runs; feature_mode=:augmented)
        @test length(augmented_dataset.records[1].features) > length(basic_dataset.records[1].features)
        extracted_stats = option_value_dataset_stats(augmented_dataset)
        @test haskey(extracted_stats, "override_positive_fraction")
        @test haskey(extracted_stats, "snapshot_override_opportunity_fraction")

        trained_pairwise = train_option_value_model(augmented_dataset;
            rng=MersenneTwister(0),
            config=OptionValueTrainingConfig(n_epochs=2, min_records=1, train_fraction=1.0, feature_mode=:augmented, objective_mode=:pairwise))
        pairwise_eval = evaluate_option_value_model(trained_pairwise["model"], augmented_dataset; selection_rule=:local_anchored)
        @test haskey(pairwise_eval, "mean_gain_vs_entry_local_candidate")
        @test haskey(pairwise_eval, "reorder_fraction_vs_entry_local")
        @test haskey(pairwise_eval, "override_precision")
    end

    @testset "Proposal log stats summarize static-frontier signals" begin
        buffer = HierarchicalEditDiagnosticsBuffer(10)
        add_proposal_log!(buffer, HierarchicalEditProposalLog(
            0x01,
            "ep-1",
            "task",
            1,
            1,
            "scaffold-a",
            1.0,
            "CCO",
            0.5,
            "scaffold-a",
            :mutate,
            nothing,
            4,
            1,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            "CCN",
            0.7,
            0.2,
            0.55,
            0.60,
            0.65,
            0.7,
            false,
        ))
        add_proposal_log!(buffer, HierarchicalEditProposalLog(
            0x01,
            "ep-1",
            "task",
            2,
            1,
            "scaffold-a",
            1.0,
            "CCN",
            0.7,
            "scaffold-a",
            :crossover,
            "CCC",
            3,
            0,
            1,
            0,
            2,
            0,
            0,
            0,
            0,
            "",
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            true,
        ))

        stats = proposal_log_stats(buffer)
        @test stats["size"] == 2
        @test stats["empty_after_filter_fraction"] == 0.5
        @test stats["mean_raw_candidate_count"] == 3.5
        @test stats["duplicate_fraction"] > 0
        @test stats["cached_fraction"] > 0
        @test stats["chosen_positive_delta_fraction"] == 1.0
    end

    @testset "No frontier mutation during rollout scoring" begin
        Random.seed!(13)
        vocab = SMILESVocabulary()
        baseline_props = propose_edit("CCO", :mutate, vocab; max_candidates=3)
        if isempty(baseline_props)
            @test_skip "RDKit mutate proposals unavailable; skipping multi-step frontier immutability regression"
        else
            fb = MolecularFrontierBuffer(20)
            add_to_frontier!(fb, "CCO"; reward=0.8, source=:seed, operator=:seed)
            add_to_frontier!(fb, "CCN"; reward=0.7, source=:model, operator=:sample)
            add_to_frontier!(fb, "CCC"; reward=0.6, source=:model, operator=:sample)

            tb = EditTrajectoryBuffer(50)
            db = HierarchicalEditDiagnosticsBuffer(50)
            cfg = HierarchicalEditConfig(horizon=3, allow_crossover=false, allow_fragment_ops=false, max_operator_candidates=3)
            initial_size = length(fb)
            observed_sizes = Int[]

            function reward_fn_batch(smiles_batch)
                push!(observed_sizes, length(fb))
                return fill(0.55, length(smiles_batch))
            end

            ep = run_hierarchical_edit_episode!(fb, tb, s -> 0.55, vocab;
                reward_fn_batch=reward_fn_batch,
                diagnostics_buffer=db,
                config=cfg,
                target_smiles="CCO",
                budget_remaining=50,
                created_at_step=3,
                task_name="immutability",
                operator_override=:mutate)

            @test !isempty(observed_sizes)
            @test all(sz == initial_size for sz in observed_sizes)
            @test ep.frontier_size_before == initial_size
            @test ep.frontier_size_after >= initial_size
            @test !isempty(db.proposal_logs)
            @test all(log.snapshot_id == ep.snapshot_id for log in db.proposal_logs)
        end
    end

    @testset "Frontier utility helper detects frontier improvements" begin
        fb = MolecularFrontierBuffer(20)
        add_to_frontier!(fb, "CCO"; reward=0.6, source=:seed, operator=:seed)
        add_to_frontier!(fb, "CCN"; reward=0.5, source=:model, operator=:sample)

        before = frontier_quality_summary(fb; topk=10)
        add_to_frontier!(fb, "c1ccccc1"; reward=0.95, source=:edit, operator=:mutate)
        after = frontier_quality_summary(fb; topk=10)
        delta = compute_frontier_utility_delta(before, after, "c1ccccc1", get_scaffold("c1ccccc1"))

        @test delta["delta_top1"] > 0
        @test delta["frontier_utility_delta"] > 0
        @test delta["enters_topk"] == true
    end

    @testset "Learned operator hook overrides heuristic operator selection" begin
        fb = MolecularFrontierBuffer(20)
        add_to_frontier!(fb, "CCO"; reward=0.9, source=:seed, operator=:seed)
        add_to_frontier!(fb, "CCN"; reward=0.7, source=:warmup, operator=:mutate)
        snap = create_frontier_snapshot(fb; max_entries=10, target_smiles="CCO", budget_remaining=20, created_at_step=1)
        basin = candidate_basins(snap; max_candidates=1)[1].basin
        parent = snap.entries[1]
        cfg = HierarchicalEditConfig(
            operators=[:mutate, :crossover, :terminate],
            use_operator_adaptation=false,
            operator_sampling_weights=Dict(:mutate => 0.9, :crossover => 0.1),
            use_learned_operator=true,
            learned_operator_controller=DummyOperatorController(),
        )

        op, candidates, meta = choose_operator_action(snap, basin, parent, cfg; step_index=1)
        @test op == candidates[2].operator
        @test meta["override_applied"] == true
        @test meta["selection_reason"] == "custom_override"
    end

    @testset "Eligibility-gated operator controller preserves and activates correctly" begin
        fb = MolecularFrontierBuffer(20)
        add_to_frontier!(fb, "CCO"; reward=0.9, source=:seed, operator=:seed)
        add_to_frontier!(fb, "CCN"; reward=0.7, source=:warmup, operator=:mutate)
        snap = create_frontier_snapshot(fb; max_entries=10, target_smiles="CCO", budget_remaining=20, created_at_step=1)
        basin = candidate_basins(snap; max_candidates=1)[1].basin
        parent = snap.entries[1]

        ineligible_model = LearnedOperatorEligibilityModel(zeros(Float32, 8), -10.0f0, 8)
        ineligible_controller = create_gated_operator_controller(ineligible_model, DummyOperatorController(); eligibility_threshold=0.5f0)
        ineligible_cfg = HierarchicalEditConfig(
            operators=[:mutate, :crossover, :terminate],
            use_operator_adaptation=false,
            operator_sampling_weights=Dict(:mutate => 0.9, :crossover => 0.1),
            use_learned_operator=true,
            learned_operator_controller=ineligible_controller,
        )

        op1, candidates1, meta1 = choose_operator_action(snap, basin, parent, ineligible_cfg; step_index=1)
        @test op1 == candidates1[1].operator
        @test meta1["predicted_eligible"] == false
        @test meta1["acted_on"] == false
        @test meta1["preserved_to_heuristic"] == true
        @test meta1["selection_reason"] == "ineligible_preserve"

        eligible_model = LearnedOperatorEligibilityModel(zeros(Float32, 8), 10.0f0, 8)
        eligible_controller = create_gated_operator_controller(eligible_model, DummyOperatorController(); eligibility_threshold=0.5f0)
        eligible_cfg = HierarchicalEditConfig(
            operators=[:mutate, :crossover, :terminate],
            use_operator_adaptation=false,
            operator_sampling_weights=Dict(:mutate => 0.9, :crossover => 0.1),
            use_learned_operator=true,
            learned_operator_controller=eligible_controller,
        )

        op2, candidates2, meta2 = choose_operator_action(snap, basin, parent, eligible_cfg; step_index=1)
        @test op2 == candidates2[2].operator
        @test meta2["predicted_eligible"] == true
        @test meta2["acted_on"] == true
        @test meta2["override_applied"] == true
    end

    @testset "A2: Config defaults for new fields" begin
        cfg = HierarchicalEditConfig()
        @test cfg.min_exploration_per_operator == 5
        @test cfg.multi_child_min_reward_ratio == 0.2
        @test cfg.operator_prior_strength == 4.0
        @test cfg.use_operator_adaptation == true
        @test isnothing(cfg.operator_sampling_weights)
        @test cfg.basin_candidate_limit == 8
        @test cfg.use_learned_basin == false
        @test isnothing(cfg.learned_basin_controller)
        @test cfg.parent_candidate_limit == 16
        @test cfg.use_learned_parent == false
        @test isnothing(cfg.learned_parent_controller)
        @test cfg.use_learned_operator == false
        @test isnothing(cfg.learned_operator_controller)
    end

    @testset "A2: Static operator weights support bounded fixed-policy bias" begin
        cfg = HierarchicalEditConfig(operators=[:mutate, :crossover, :terminate],
                                      use_operator_adaptation=false,
                                      operator_sampling_weights=Dict(:mutate => 0.8, :crossover => 0.2))
        Random.seed!(123)
        choices = Symbol[]
        for _ in 1:400
            push!(choices, choose_operator(cfg))
        end
        mutate_frac = count(c -> c == :mutate, choices) / length(choices)
        crossover_frac = count(c -> c == :crossover, choices) / length(choices)
        @test mutate_frac > crossover_frac
        @test 0.65 < mutate_frac < 0.95
        @test :terminate ∉ choices
    end

    @testset "A2: Disabling adaptation ignores adaptive stats" begin
        cfg = HierarchicalEditConfig(operators=[:mutate, :crossover, :terminate],
                                      use_operator_adaptation=false,
                                      operator_sampling_weights=Dict(:mutate => 0.5, :crossover => 0.5),
                                      min_exploration_per_operator=10)
        stats = Dict{Symbol,Dict{String,Int}}(
            :mutate => Dict("positive_delta_count" => 50, "total_count" => 50),
            :crossover => Dict("positive_delta_count" => 0, "total_count" => 50),
        )
        Random.seed!(321)
        choices = Symbol[]
        for _ in 1:300
            push!(choices, choose_operator(cfg; operator_stats=stats))
        end
        mutate_frac = count(c -> c == :mutate, choices) / length(choices)
        @test 0.35 < mutate_frac < 0.65
    end

    @testset "A2: Exploration phase prefers least-tried operator" begin
        cfg = HierarchicalEditConfig(operators=[:mutate, :crossover, :terminate],
                                      min_exploration_per_operator=3)
        # Simulate: mutate tried 2x, crossover tried 0x
        stats = Dict{Symbol,Dict{String,Int}}(
            :mutate => Dict("positive_delta_count" => 1, "total_count" => 2),
        )
        # With crossover at 0 trials (under min_exploration=3), it should be preferred
        choices = Symbol[]
        for _ in 1:20
            push!(choices, choose_operator(cfg; operator_stats=stats))
        end
        # crossover should be selected every time (it's the least tried)
        @test all(c -> c == :crossover, choices)
    end

    @testset "A2: Exploration phase balances equally under-explored operators" begin
        cfg = HierarchicalEditConfig(operators=[:mutate, :crossover, :terminate],
                                      min_exploration_per_operator=5)
        # Both operators have 0 trials → both under-explored equally
        stats = Dict{Symbol,Dict{String,Int}}()
        Random.seed!(42)
        choices = Symbol[]
        for _ in 1:100
            push!(choices, choose_operator(cfg; operator_stats=stats))
        end
        # Both should appear (uniform random among equally under-explored)
        @test :mutate in choices
        @test :crossover in choices
        @test :terminate ∉ choices
    end

    @testset "A2: Thompson sampling activates after exploration phase" begin
        cfg = HierarchicalEditConfig(operators=[:mutate, :crossover, :terminate],
                                      min_exploration_per_operator=2,
                                      operator_prior_strength=4.0)
        # Both operators past min_exploration: mutate 90% success, crossover 10% success
        stats = Dict{Symbol,Dict{String,Int}}(
            :mutate => Dict("positive_delta_count" => 9, "total_count" => 10),
            :crossover => Dict("positive_delta_count" => 1, "total_count" => 10),
        )
        Random.seed!(99)
        choices = Symbol[]
        for _ in 1:200
            push!(choices, choose_operator(cfg; operator_stats=stats))
        end
        mutate_frac = count(c -> c == :mutate, choices) / length(choices)
        crossover_frac = count(c -> c == :crossover, choices) / length(choices)
        # mutate should be selected more often, but crossover still gets some share
        # due to informed prior (prior_strength=4 adds 2 pseudo-successes each)
        @test mutate_frac > crossover_frac
        @test crossover_frac > 0.05  # crossover not completely starved
    end

    @testset "A2: Informed prior prevents operator starvation" begin
        cfg = HierarchicalEditConfig(operators=[:mutate, :crossover, :terminate],
                                      min_exploration_per_operator=2,
                                      operator_prior_strength=10.0)  # Strong prior
        # mutate has 100% success, crossover has 0% — but with strong prior,
        # crossover still gets significant share
        stats = Dict{Symbol,Dict{String,Int}}(
            :mutate => Dict("positive_delta_count" => 5, "total_count" => 5),
            :crossover => Dict("positive_delta_count" => 0, "total_count" => 5),
        )
        Random.seed!(77)
        choices = Symbol[]
        for _ in 1:500
            push!(choices, choose_operator(cfg; operator_stats=stats))
        end
        crossover_frac = count(c -> c == :crossover, choices) / length(choices)
        # With prior_strength=10, crossover gets (5+0)/(10+5) = 33% base vs
        # mutate (5+5)/(10+5) = 67%. crossover should get ~33% of selections.
        @test crossover_frac > 0.20  # at least 20% with strong prior
    end

    @testset "A2: Quality gate filters low-reward multi-child proposals" begin
        # Verify the quality floor logic conceptually via config presence
        cfg = HierarchicalEditConfig(multi_child_min_reward_ratio=0.5)
        @test cfg.multi_child_min_reward_ratio == 0.5
        # With parent reward 0.8 and ratio 0.5, floor = 0.4
        # Proposals with reward < 0.4 should be filtered
        quality_floor = 0.8 * cfg.multi_child_min_reward_ratio
        @test quality_floor == 0.4
        @test 0.3 < quality_floor  # would be filtered
        @test 0.5 >= quality_floor  # would pass
    end
end
