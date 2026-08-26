#!/usr/bin/env julia
# PMO Validation: QED task with TB v10 config under standard 10K oracle budget
#
# This validates the CAFE-GFN pipeline end-to-end on the PMO benchmark protocol:
# - Budget: 10,000 unique oracle calls
# - Metric: AUC of top-10 average score (sampled every 100 calls)
# - Uses TB v10 config: freeze GRU + unfreeze gru_3 + constructive-only + reward-weighted
#
# Also runs RWMLE for head-to-head comparison under identical budget.

using Pkg; Pkg.activate(".")
using GFlowNet
using Serialization
using Statistics
using Printf
using Random
using Dates

flush(stdout)

# =============================================================================
# Budget-Tracked Oracle Wrapper
# =============================================================================

mutable struct BudgetOracle
    oracle_fn::Function
    cache::Dict{String, Float64}
    budget::Int
    calls_used::Int
    # AUC tracking: record top-10 average every `snapshot_interval` calls
    snapshot_interval::Int
    top10_snapshots::Vector{Float64}
    all_scores::Vector{Float64}  # sorted, for efficient top-10
end

function BudgetOracle(oracle_fn::Function; budget::Int=10000, snapshot_interval::Int=100)
    BudgetOracle(oracle_fn, Dict{String,Float64}(), budget, 0, snapshot_interval, Float64[], Float64[])
end

function budget_exhausted(bo::BudgetOracle)
    bo.calls_used >= bo.budget
end

function oracle_call!(bo::BudgetOracle, smiles::String)::Float64
    # Return cached result without spending budget
    if haskey(bo.cache, smiles)
        return bo.cache[smiles]
    end

    # Budget check
    if budget_exhausted(bo)
        return 0.0  # Return 0 instead of error (graceful degradation)
    end

    # New oracle call
    score = bo.oracle_fn(smiles)
    bo.cache[smiles] = score
    bo.calls_used += 1

    # Track scores for top-10
    if score > 0.0
        push!(bo.all_scores, score)
        sort!(bo.all_scores; rev=true)
        if length(bo.all_scores) > 1000
            resize!(bo.all_scores, 1000)  # Keep top 1000 for efficiency
        end
    end

    # Snapshot top-10 at intervals
    if bo.calls_used % bo.snapshot_interval == 0
        top10 = if length(bo.all_scores) >= 10
            mean(bo.all_scores[1:10])
        elseif !isempty(bo.all_scores)
            mean(bo.all_scores)
        else
            0.0
        end
        push!(bo.top10_snapshots, top10)
    end

    return score
end

function compute_auc_top10(bo::BudgetOracle)::Float64
    isempty(bo.top10_snapshots) && return 0.0
    return mean(bo.top10_snapshots)
end

function get_pmo_result(bo::BudgetOracle, task_name::String)
    top10 = length(bo.all_scores) >= 10 ? bo.all_scores[1:10] : bo.all_scores
    return (
        task_name = task_name,
        auc_top10 = compute_auc_top10(bo),
        top1 = isempty(bo.all_scores) ? 0.0 : bo.all_scores[1],
        top10_mean = isempty(top10) ? 0.0 : mean(top10),
        n_oracle_calls = bo.calls_used,
        unique_molecules = length(bo.cache),
        n_snapshots = length(bo.top10_snapshots),
    )
end

# =============================================================================
# Setup
# =============================================================================

println("=" ^ 70)
println("PMO Validation: QED Task — TB v10 vs RWMLE (10K Budget)")
println("=" ^ 70)
flush(stdout)

# Load pretrained checkpoint
println("\nLoading pretrained checkpoint...")
pretrained = load_pretrained_checkpoint("checkpoints/pretrain/final.jls")

println("Loading ZINC vocabulary...")
smiles_data = load_zinc_smiles("data/zinc/250k_rndm_zinc_drugs_clean_3.csv"; max_molecules=50000)
vocab = SMILESVocabulary()
prepare_zinc_dataset(vocab, smiles_data)

actual_vocab_size = size(pretrained.params.output.layer_2.weight, 1)
println("  Vocab size: $actual_vocab_size")

model, _, init_states = create_smiles_policy(;
    vocab_size=actual_vocab_size, hidden_dim=512, embed_dim=128, n_layers=3
)

# QED oracle (RDKit-based, no TDC needed)
using PythonCall
rdkit = pyimport("rdkit.Chem")
rdkit_qed = pyimport("rdkit.Chem.QED")
function qed_oracle(smi::String)::Float64
    mol = rdkit.MolFromSmiles(smi)
    pyis(mol, pybuiltins.None) && return 0.0
    try; return pyconvert(Float64, rdkit_qed.qed(mol)); catch; return 0.0; end
end

# Evaluation helper
function evaluate_model(model, ps, states, vocab, oracle_fn; n_samples=200, temp=0.8)
    qeds = Float64[]; valid = 0; unique_smiles = Set{String}()
    for _ in 1:n_samples
        smi, _, _ = sample_smiles_autoregressive(model, ps, states, vocab;
            max_length=150, temperature=temp, constrained=true)
        q = oracle_fn(smi)
        if q > 0.0; valid += 1; push!(qeds, q); push!(unique_smiles, smi); end
    end
    sort!(qeds; rev=true); n = length(qeds)
    return (valid=valid, total=n_samples, validity=100.0*valid/n_samples,
        qeds=qeds, n_unique=length(unique_smiles),
        mean_qed=n > 0 ? mean(qeds) : 0.0,
        top10_mean=n >= 10 ? mean(qeds[1:10]) : (n > 0 ? mean(qeds) : 0.0),
        geq_07=count(>=(0.7), qeds), geq_08=count(>=(0.8), qeds),
        geq_09=count(>=(0.9), qeds),
        top5=n >= 5 ? qeds[1:5] : qeds)
end

function print_eval(name, stats)
    @printf("  %-35s: valid=%d/%d (%.1f%%)  unique=%d  QED: mean=%.3f  top10=%.3f  >=0.9=%d\n",
        name, stats.valid, stats.total, stats.validity, stats.n_unique,
        stats.mean_qed, stats.top10_mean, stats.geq_09)
    length(stats.top5) > 0 && println("    Top 5: ", join([@sprintf("%.3f", q) for q in stats.top5], ", "))
    flush(stdout)
end

# Baseline
println("\n" * "=" ^ 70); println("BASELINE (pretrained)"); println("=" ^ 70); flush(stdout)
baseline = evaluate_model(model, pretrained.params, init_states, vocab, qed_oracle)
print_eval("pretrained", baseline)

# =============================================================================
# Experiment 1: TB v10 under PMO Budget
# =============================================================================

println("\n" * "=" ^ 70)
println("EXPERIMENT 1: TB v10 Fine-Tuning (10K Oracle Budget)")
println("=" ^ 70)
flush(stdout)

# Budget-tracked oracle
tb_oracle = BudgetOracle(qed_oracle; budget=10000, snapshot_interval=100)

# Wrap for fine-tuning (budget-tracked)
tb_reward_fn(smi) = oracle_call!(tb_oracle, smi)

ref_params = deepcopy(pretrained.params)
ref_states = deepcopy(init_states)

tb_config = FinetuningConfig(;
    n_iterations=25,        # Per segment
    sample_batch_size=32,
    learning_rate=3e-5,
    gradient_clip_norm=1.0,
    kl_weight=1.0,
    kl_decay_schedule=:none,
    loss_type=:shifted_cosh,
    cosh_threshold=2.0,
    max_length=150,
    temperature=1.0,
    epsilon=0.05,
    log_frequency=5,
    reward_exponent=8.0,
    min_reward=0.01,
    training_mode=:tb,
    constructive_only=true,
    freeze_gru=true,
    reward_weighted=true,
    unfreeze_top_gru=true,
)

println("  Config: TB v10 (freeze GRU + unfreeze gru_3 + constructive + reward-weighted)")
println("  Budget: $(tb_oracle.budget) oracle calls")
println("  Segments: 25 iters each, stop when budget exhausted")
flush(stdout)

tb_current_params = deepcopy(pretrained.params)
tb_best_qed = 0.0; tb_best_iter = 0; tb_best_params = nothing
segment = 0

while !budget_exhausted(tb_oracle)
    global segment += 1
    total_iter = segment * 25

    println("\n--- TB v10 Segment $segment (budget used: $(tb_oracle.calls_used)/$(tb_oracle.budget)) ---")
    flush(stdout)

    result = finetune_smiles_gflownet(
        model, vocab, tb_current_params, init_states,
        ref_params, ref_states, tb_reward_fn, tb_config; verbose=true)

    global tb_current_params = result.params

    # Quick evaluation (doesn't count toward PMO budget — uses raw oracle)
    stats = evaluate_model(model, tb_current_params, init_states, vocab, qed_oracle; n_samples=100)
    @printf("  Eval iter %d: QED=%.3f, validity=%.1f%%, budget=%d/%d\n",
        total_iter, stats.mean_qed, stats.validity, tb_oracle.calls_used, tb_oracle.budget)
    flush(stdout)

    if stats.mean_qed > tb_best_qed && stats.validity > 70.0
        global tb_best_qed = stats.mean_qed
        global tb_best_iter = total_iter
        global tb_best_params = deepcopy(tb_current_params)
    end

    # Safety: max 20 segments (500 iters)
    segment >= 20 && break
end

# Final TB evaluation
println("\n--- TB v10 FINAL Evaluation ---")
if tb_best_params !== nothing
    tb_final = evaluate_model(model, tb_best_params, init_states, vocab, qed_oracle)
    print_eval("TB v10 best (iter $tb_best_iter)", tb_final)
else
    tb_final = evaluate_model(model, tb_current_params, init_states, vocab, qed_oracle)
    print_eval("TB v10 final", tb_final)
end

tb_pmo = get_pmo_result(tb_oracle, "qed")
@printf("  PMO Metrics: AUC_top10=%.4f, top1=%.4f, top10_mean=%.4f, oracle_calls=%d, unique=%d\n",
    tb_pmo.auc_top10, tb_pmo.top1, tb_pmo.top10_mean, tb_pmo.n_oracle_calls, tb_pmo.unique_molecules)
flush(stdout)

# =============================================================================
# Experiment 2: RWMLE under PMO Budget (head-to-head comparison)
# =============================================================================

println("\n" * "=" ^ 70)
println("EXPERIMENT 2: RWMLE Fine-Tuning (10K Oracle Budget)")
println("=" ^ 70)
flush(stdout)

rwmle_oracle = BudgetOracle(qed_oracle; budget=10000, snapshot_interval=100)
rwmle_reward_fn(smi) = oracle_call!(rwmle_oracle, smi)

rwmle_config = FinetuningConfig(;
    n_iterations=25,
    sample_batch_size=32,
    learning_rate=3e-5,
    gradient_clip_norm=1.0,
    kl_weight=1.0,
    kl_decay_schedule=:none,
    loss_type=:shifted_cosh,
    cosh_threshold=2.0,
    max_length=150,
    temperature=1.0,
    epsilon=0.05,
    log_frequency=5,
    reward_exponent=4.0,        # RWMLE used β=4.0
    min_reward=0.01,
    training_mode=:rwmle,       # ← RWMLE mode
    constructive_only=false,    # N/A for RWMLE
    freeze_gru=false,           # RWMLE doesn't need frozen GRU
    reward_weighted=false,      # N/A for RWMLE
    unfreeze_top_gru=false,     # N/A
)

println("  Config: RWMLE (reward-weighted MLE)")
println("  Budget: $(rwmle_oracle.budget) oracle calls")
flush(stdout)

rwmle_current_params = deepcopy(pretrained.params)
rwmle_best_qed = 0.0; rwmle_best_iter = 0; rwmle_best_params = nothing
segment = 0

while !budget_exhausted(rwmle_oracle)
    global segment += 1
    total_iter = segment * 25

    println("\n--- RWMLE Segment $segment (budget used: $(rwmle_oracle.calls_used)/$(rwmle_oracle.budget)) ---")
    flush(stdout)

    result = finetune_smiles_gflownet(
        model, vocab, rwmle_current_params, init_states,
        ref_params, ref_states, rwmle_reward_fn, rwmle_config; verbose=true)

    global rwmle_current_params = result.params

    stats = evaluate_model(model, rwmle_current_params, init_states, vocab, qed_oracle; n_samples=100)
    @printf("  Eval iter %d: QED=%.3f, validity=%.1f%%, budget=%d/%d\n",
        total_iter, stats.mean_qed, stats.validity, rwmle_oracle.calls_used, rwmle_oracle.budget)
    flush(stdout)

    if stats.mean_qed > rwmle_best_qed && stats.validity > 70.0
        global rwmle_best_qed = stats.mean_qed
        global rwmle_best_iter = total_iter
        global rwmle_best_params = deepcopy(rwmle_current_params)
    end

    segment >= 20 && break
end

# Final RWMLE evaluation
println("\n--- RWMLE FINAL Evaluation ---")
if rwmle_best_params !== nothing
    rwmle_final = evaluate_model(model, rwmle_best_params, init_states, vocab, qed_oracle)
    print_eval("RWMLE best (iter $rwmle_best_iter)", rwmle_final)
else
    rwmle_final = evaluate_model(model, rwmle_current_params, init_states, vocab, qed_oracle)
    print_eval("RWMLE final", rwmle_final)
end

rwmle_pmo = get_pmo_result(rwmle_oracle, "qed")
@printf("  PMO Metrics: AUC_top10=%.4f, top1=%.4f, top10_mean=%.4f, oracle_calls=%d, unique=%d\n",
    rwmle_pmo.auc_top10, rwmle_pmo.top1, rwmle_pmo.top10_mean, rwmle_pmo.n_oracle_calls, rwmle_pmo.unique_molecules)
flush(stdout)

# =============================================================================
# Head-to-Head Comparison
# =============================================================================

println("\n" * "=" ^ 70)
println("HEAD-TO-HEAD: TB v10 vs RWMLE on PMO QED")
println("=" ^ 70)

@printf("  %-20s  AUC_top10  top1    top10   calls   unique  QED_mean  valid%%\n", "Method")
@printf("  %-20s  ---------  ------  ------  ------  ------  --------  ------\n", "-" ^ 20)

# TB v10
tb_eval_qed = tb_best_params !== nothing ? tb_best_qed : 0.0
tb_eval_val = tb_best_params !== nothing ? tb_final.validity : 0.0
@printf("  %-20s  %9.4f  %6.4f  %6.4f  %6d  %6d  %8.3f  %5.1f%%\n",
    "TB v10", tb_pmo.auc_top10, tb_pmo.top1, tb_pmo.top10_mean,
    tb_pmo.n_oracle_calls, tb_pmo.unique_molecules, tb_eval_qed, tb_eval_val)

# RWMLE
rwmle_eval_qed = rwmle_best_params !== nothing ? rwmle_best_qed : 0.0
rwmle_eval_val = rwmle_best_params !== nothing ? rwmle_final.validity : 0.0
@printf("  %-20s  %9.4f  %6.4f  %6.4f  %6d  %6d  %8.3f  %5.1f%%\n",
    "RWMLE", rwmle_pmo.auc_top10, rwmle_pmo.top1, rwmle_pmo.top10_mean,
    rwmle_pmo.n_oracle_calls, rwmle_pmo.unique_molecules, rwmle_eval_qed, rwmle_eval_val)

# SOTA reference
println("\n  SOTA Reference (sum across 23 tasks):")
println("    Genetic GFN:   16.2  (per-task avg: $(round(16.2/23, digits=3)))")
println("    REINVENT:      15.2  (per-task avg: $(round(15.2/23, digits=3)))")
println("    Mol GA:        15.7  (per-task avg: $(round(15.7/23, digits=3)))")

# Estimate where we'd land
println("\n  Our QED AUC estimates:")
@printf("    TB v10 QED AUC:  %.4f  (if all 23 tasks similar: %.1f total)\n",
    tb_pmo.auc_top10, tb_pmo.auc_top10 * 23)
@printf("    RWMLE QED AUC:   %.4f  (if all 23 tasks similar: %.1f total)\n",
    rwmle_pmo.auc_top10, rwmle_pmo.auc_top10 * 23)

# Save results
results_dir = "checkpoints/pmo_validation"
mkpath(results_dir)
open("$results_dir/qed_results.txt", "w") do f
    println(f, "PMO QED Validation Results")
    println(f, "Date: $(Dates.now())")
    println(f, "\nTB v10:")
    println(f, "  AUC_top10 = $(tb_pmo.auc_top10)")
    println(f, "  top1 = $(tb_pmo.top1)")
    println(f, "  top10_mean = $(tb_pmo.top10_mean)")
    println(f, "  oracle_calls = $(tb_pmo.n_oracle_calls)")
    println(f, "  unique_molecules = $(tb_pmo.unique_molecules)")
    println(f, "  best_qed = $tb_eval_qed (iter $tb_best_iter)")
    println(f, "\nRWMLE:")
    println(f, "  AUC_top10 = $(rwmle_pmo.auc_top10)")
    println(f, "  top1 = $(rwmle_pmo.top1)")
    println(f, "  top10_mean = $(rwmle_pmo.top10_mean)")
    println(f, "  oracle_calls = $(rwmle_pmo.n_oracle_calls)")
    println(f, "  unique_molecules = $(rwmle_pmo.unique_molecules)")
    println(f, "  best_qed = $rwmle_eval_qed (iter $rwmle_best_iter)")

    println(f, "\nAUC top-10 snapshots (TB v10):")
    for (i, s) in enumerate(tb_oracle.top10_snapshots)
        println(f, "  $(i*100) calls: $s")
    end
    println(f, "\nAUC top-10 snapshots (RWMLE):")
    for (i, s) in enumerate(rwmle_oracle.top10_snapshots)
        println(f, "  $(i*100) calls: $s")
    end
end
println("\nResults saved to $results_dir/qed_results.txt")

println("\nDone!"); flush(stdout)
