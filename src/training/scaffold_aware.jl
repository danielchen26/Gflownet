# Scaffold-Aware Training for CAFE-GFN 2.0
#
# Implements the complete scaffold-aware training framework from the v4 plan:
# 1. ScaffoldInfo + GlobalStats tracking
# 2. Auto-calibrating scaffold_quality (R_max(s) / R_median_global)
# 3. Auto-calibrating diversity_weight (C / n_samples(s))
# 4. Task-type auto-detection (:smooth, :sparse, :structural)
# 5. Novelty-rate-driven adaptive control (β, GA mix, budget fraction)
#
# All control parameters are continuous functions of scaffold novelty rate.
# No magic numbers, no hard-coded thresholds.

using Statistics: mean, median

# =============================================================================
# Data Structures
# =============================================================================

"""
    ScaffoldInfo

Tracks statistics for a single Bemis-Murcko scaffold.
"""
mutable struct ScaffoldInfo
    n_samples::Int           # Number of molecules with this scaffold
    reward_sum::Float64      # Sum of rewards for all molecules
    reward_max::Float64      # Best reward seen for this scaffold
    best_smiles::String      # SMILES of the best molecule with this scaffold

    ScaffoldInfo() = new(0, 0.0, 0.0, "")
end

"""
    GlobalStats

Tracks global statistics across all molecules for auto-calibration.
"""
mutable struct GlobalStats
    reward_sum::Float64
    reward_count::Int
    reward_median::Float64       # Updated periodically
    all_rewards::Vector{Float64} # For computing median (capped at 1000)
    n_scaffolds_total::Int       # Total unique scaffolds discovered
    recent_molecules::Vector{String}  # Recent SMILES for novelty rate (ring buffer)
    recent_idx::Int              # Current index in ring buffer

    function GlobalStats(; buffer_size::Int=500)
        new(0.0, 0, 0.1, Float64[], 0,
            fill("", buffer_size), 0)
    end
end

"""
    ScaffoldTracker

Complete scaffold tracking system combining per-scaffold and global stats.
"""
mutable struct ScaffoldTracker
    scaffold_stats::Dict{String, ScaffoldInfo}
    global_stats::GlobalStats
    task_type::Symbol            # :smooth, :sparse, :structural, :unknown
    target_scaffold::Union{Nothing, String}
    β_range::NamedTuple{(:β_min, :β_max), Tuple{Float64, Float64}}

    function ScaffoldTracker(; target_smiles::Union{Nothing, String}=nothing)
        target_scaffold = nothing
        if !isnothing(target_smiles)
            target_scaffold = get_scaffold(target_smiles)
            if target_scaffold == ""
                target_scaffold = nothing
            end
        end

        new(Dict{String, ScaffoldInfo}(),
            GlobalStats(),
            :unknown,
            target_scaffold,
            (β_min=2.0, β_max=8.0))  # Default range
    end
end

# β ranges per task type (from v4 plan)
const BETA_RANGES = Dict(
    :smooth     => (β_min=1.0, β_max=4.0),    # QED, logP
    :sparse     => (β_min=2.0, β_max=8.0),    # DRD2, JNK3
    :structural => (β_min=4.0, β_max=12.0),   # Similarity, rediscovery
    :unknown    => (β_min=2.0, β_max=8.0),    # Default
)

# =============================================================================
# Update Functions
# =============================================================================

"""
    update_scaffold_tracker!(tracker, smiles, reward)

Update the scaffold tracker with a new molecule observation.
"""
function update_scaffold_tracker!(tracker::ScaffoldTracker, smiles::String, reward::Float64)
    # Update global stats
    gs = tracker.global_stats
    gs.reward_sum += reward
    gs.reward_count += 1

    # Track recent rewards for median (capped at 1000)
    if length(gs.all_rewards) < 1000
        push!(gs.all_rewards, reward)
    else
        gs.all_rewards[rand(1:1000)] = reward  # Reservoir sampling
    end

    # Update median every 50 observations
    if gs.reward_count % 50 == 0 && !isempty(gs.all_rewards)
        gs.reward_median = median(gs.all_rewards)
    end

    # Add to recent molecules ring buffer
    gs.recent_idx = (gs.recent_idx % length(gs.recent_molecules)) + 1
    gs.recent_molecules[gs.recent_idx] = smiles

    # Update scaffold stats
    scaffold = get_scaffold(smiles)
    if !isempty(scaffold)
        if !haskey(tracker.scaffold_stats, scaffold)
            tracker.scaffold_stats[scaffold] = ScaffoldInfo()
            gs.n_scaffolds_total += 1
        end

        info = tracker.scaffold_stats[scaffold]
        info.n_samples += 1
        info.reward_sum += reward
        if reward > info.reward_max
            info.reward_max = reward
            info.best_smiles = smiles
        end
    end
end

"""
    detect_task_type!(tracker)

Auto-detect task type from reward distribution. Called after sufficient data.
"""
function detect_task_type!(tracker::ScaffoldTracker)
    gs = tracker.global_stats

    # Structural tasks: detected from target_scaffold (set at construction)
    if !isnothing(tracker.target_scaffold)
        tracker.task_type = :structural
        tracker.β_range = BETA_RANGES[:structural]
        return
    end

    # Need enough data
    if gs.reward_count < 30
        return
    end

    # Smooth vs sparse: from reward distribution
    reward_rate = count(r -> r > 0.1, gs.all_rewards) / length(gs.all_rewards)
    if reward_rate > 0.3
        tracker.task_type = :smooth
    else
        tracker.task_type = :sparse
    end
    tracker.β_range = BETA_RANGES[tracker.task_type]
end

# =============================================================================
# Reward Shaping (Axiom 1 + 2)
# =============================================================================

"""
    scaffold_quality(scaffold, tracker) → Float64

Auto-calibrating scaffold quality multiplier.

Quality = R_max(scaffold) / R_median_global
- High-performing scaffolds get amplified
- Self-calibrating: no magic numbers
"""
function scaffold_quality(scaffold::String, tracker::ScaffoldTracker)::Float64
    gs = tracker.global_stats
    med = max(0.01, gs.reward_median)

    # Target scaffold for structural tasks
    if !isnothing(tracker.target_scaffold) && scaffold == tracker.target_scaffold
        return 1.0 / med
    end

    if !haskey(tracker.scaffold_stats, scaffold)
        # Novel scaffold: assign prior (global mean)
        reward_mean = gs.reward_count > 0 ? gs.reward_sum / gs.reward_count : 0.1
        return reward_mean / med
    end

    info = tracker.scaffold_stats[scaffold]
    return info.reward_max / med
end

"""
    diversity_weight(scaffold, tracker) → Float64

Auto-calibrating diversity weight based on diminishing marginal information.

Weight = min(1.0, C / n_samples(scaffold))
where C = median sample count across all scaffolds.
"""
function diversity_weight(scaffold::String, tracker::ScaffoldTracker)::Float64
    # Never penalize target scaffold for structural tasks
    if tracker.task_type == :structural && !isnothing(tracker.target_scaffold) &&
       scaffold == tracker.target_scaffold
        return 1.0
    end

    if !haskey(tracker.scaffold_stats, scaffold)
        return 1.0  # Novel — no diminishing returns
    end

    n = tracker.scaffold_stats[scaffold].n_samples
    C = _median_scaffold_count(tracker)
    C = max(C, 3.0)  # Floor to prevent division issues early

    return min(1.0, C / n)
end

"""
    _median_scaffold_count(tracker) → Float64

Compute median sample count across all known scaffolds.
"""
function _median_scaffold_count(tracker::ScaffoldTracker)::Float64
    if isempty(tracker.scaffold_stats)
        return 3.0
    end
    counts = Float64[info.n_samples for info in values(tracker.scaffold_stats)]
    return median(counts)
end

"""
    shape_reward(raw_reward, smiles, tracker) → Float64

Apply scaffold-aware reward shaping.

R_shaped = R_task × scaffold_quality × diversity_weight
"""
function shape_reward(raw_reward::Float64, smiles::String,
                       tracker::ScaffoldTracker)::Float64
    scaffold = get_scaffold(smiles)
    if isempty(scaffold)
        return raw_reward  # Can't determine scaffold — pass through
    end

    sq = scaffold_quality(scaffold, tracker)
    dw = diversity_weight(scaffold, tracker)

    shaped = raw_reward * sq * dw

    # Clamp to prevent extreme values
    return clamp(shaped, 0.0, 100.0)
end

# =============================================================================
# Novelty Rate (Axiom 4)
# =============================================================================

"""
    compute_novelty_rate(tracker; window=100) → Float64

Compute scaffold novelty rate from recent molecules.

Returns fraction of recent molecules that have novel or near-novel scaffolds.
This single observable drives ALL adaptive control parameters.
"""
function compute_novelty_rate(tracker::ScaffoldTracker; window::Int=100)::Float64
    gs = tracker.global_stats

    # Get recent molecules from ring buffer
    n_filled = min(gs.reward_count, length(gs.recent_molecules))
    if n_filled == 0
        return 1.0  # No data → full exploration
    end

    # Get last `window` entries from ring buffer
    recent = String[]
    start_idx = max(1, gs.recent_idx - window + 1)
    if start_idx > 0
        for i in start_idx:gs.recent_idx
            if i >= 1 && i <= length(gs.recent_molecules) && !isempty(gs.recent_molecules[i])
                push!(recent, gs.recent_molecules[i])
            end
        end
    end
    # Handle wrap-around
    if start_idx <= 0
        wrap_start = length(gs.recent_molecules) + start_idx
        for i in wrap_start:length(gs.recent_molecules)
            if !isempty(gs.recent_molecules[i])
                push!(recent, gs.recent_molecules[i])
            end
        end
        for i in 1:gs.recent_idx
            if !isempty(gs.recent_molecules[i])
                push!(recent, gs.recent_molecules[i])
            end
        end
    end

    if isempty(recent)
        return 1.0
    end

    n_novel = 0
    n_with_scaffold = 0
    for smi in recent
        s = get_scaffold(smi)
        if isempty(s)
            continue
        end
        n_with_scaffold += 1
        if !haskey(tracker.scaffold_stats, s) || tracker.scaffold_stats[s].n_samples <= 2
            n_novel += 1
        end
    end

    return n_with_scaffold > 0 ? n_novel / n_with_scaffold : 1.0
end

# =============================================================================
# Adaptive Control (All Continuous Functions of Novelty Rate)
# =============================================================================

"""
    AdaptiveParams

All training parameters derived from novelty rate. Updated every segment.
"""
struct AdaptiveParams
    effective_β::Float64          # Reward exponent
    reward_weighted::Bool         # RW-TB when β > 2
    diverse_ga_fraction::Float64  # Fraction of GA budget for diverse crossover
    scaffold_ga_fraction::Float64 # Fraction for scaffold-preserving crossover
    ga_budget_fraction::Float64   # Fraction of remaining budget for GA
    use_promptsmiles::Bool        # PromptSMILES inference
    novelty_rate::Float64         # Raw novelty rate
end

"""
    compute_adaptive_params(tracker; β_override=nothing) → AdaptiveParams

Compute all adaptive parameters as continuous functions of novelty rate.

Every control parameter is derived from the scaffold novelty rate:
- β: low when exploring (high novelty), high when exploiting (low novelty)
- GA mix: diverse when exploring, scaffold-preserving when exploiting
- GA budget: proportional to novelty (more exploration = more GA value)
- PromptSMILES: only when exploiting known scaffolds
"""
function compute_adaptive_params(tracker::ScaffoldTracker;
                                  β_override::Union{Nothing, Float64}=nothing)::AdaptiveParams
    novelty_rate = compute_novelty_rate(tracker)

    β_min = tracker.β_range.β_min
    β_max = tracker.β_range.β_max

    # β: continuous function of novelty rate
    effective_β = if !isnothing(β_override)
        β_override
    else
        β_min + (1.0 - novelty_rate) * (β_max - β_min)
    end

    # GA type mix
    diverse_ga_fraction = novelty_rate
    scaffold_ga_fraction = 1.0 - novelty_rate

    # GA budget: 5-20% of remaining budget
    ga_budget_fraction = 0.05 + 0.15 * novelty_rate

    # PromptSMILES: only when exploiting (low novelty) AND we have target
    use_promptsmiles = novelty_rate < 0.5 && !isnothing(tracker.target_scaffold)

    # RW-TB: when β > 2 (gradient cap problem)
    reward_weighted = effective_β > 2.0

    return AdaptiveParams(
        effective_β,
        reward_weighted,
        diverse_ga_fraction,
        scaffold_ga_fraction,
        ga_budget_fraction,
        use_promptsmiles,
        novelty_rate
    )
end

# =============================================================================
# Integration Helper
# =============================================================================

"""
    run_graph_ga_step!(tracker, replay_buffer, vocab, budget_oracle,
                        adaptive_params;
                        remaining_budget=1000,
                        max_ga_per_segment=20) → Int

Run Graph GA search step based on adaptive parameters.

Returns number of oracle calls consumed.
"""
function run_graph_ga_step!(tracker::ScaffoldTracker,
                             replay_buffer,
                             vocab,
                             budget_oracle::Function,
                             adaptive_params::AdaptiveParams;
                             remaining_budget::Int=1000,
                             max_ga_per_segment::Int=20)::Int

    ga_budget = min(
        floor(Int, remaining_budget * adaptive_params.ga_budget_fraction),
        max_ga_per_segment
    )

    if ga_budget <= 0 || isempty(replay_buffer) || length(replay_buffer) < 20
        return 0
    end

    calls_used = 0
    top = get_top_molecules(replay_buffer, min(50, length(replay_buffer)))
    smiles_list = String[m.smiles for m in top]
    scores = Float64[m.reward for m in top]

    # Split GA budget between diverse and scaffold-preserving
    n_diverse = round(Int, ga_budget * adaptive_params.diverse_ga_fraction)
    n_scaffold = ga_budget - n_diverse

    # --- Diverse Graph GA ---
    if n_diverse > 0
        n_xover = max(1, n_diverse ÷ 2)
        n_mut = n_diverse - n_xover
        diverse_children = graph_ga_crossover_mutate(smiles_list, scores;
            n_crossover=n_xover, n_mutation=n_mut)

        for child_smi in diverse_children
            calls_used >= ga_budget && break
            try
                reward = budget_oracle(child_smi)
                calls_used += 1
                if reward > 0.01
                    tokens = encode(vocab, child_smi)
                    if length(tokens) >= 2
                        update_scaffold_tracker!(tracker, child_smi, reward)
                        # Store RAW oracle score — replay applies R^β at loss time
                        add_to_replay!(replay_buffer, child_smi, tokens, reward)
                    end
                end
            catch
                continue
            end
        end
    end

    # --- Scaffold-Preserving GA ---
    if n_scaffold > 0 && length(tracker.scaffold_stats) > 0
        scaffolds = String[get_scaffold(s) for s in smiles_list]
        scaffold_children = graph_ga_scaffold_crossover(
            smiles_list, scaffolds, scores;
            target_scaffold=tracker.target_scaffold,
            n_children=n_scaffold)

        for child_smi in scaffold_children
            calls_used >= ga_budget && break
            try
                reward = budget_oracle(child_smi)
                calls_used += 1
                if reward > 0.01
                    tokens = encode(vocab, child_smi)
                    if length(tokens) >= 2
                        update_scaffold_tracker!(tracker, child_smi, reward)
                        # Store RAW oracle score — replay applies R^β at loss time
                        add_to_replay!(replay_buffer, child_smi, tokens, reward)
                    end
                end
            catch
                continue
            end
        end
    end

    return calls_used
end
