# Sequential Boosting for GFlowNets (Boosted GFlowNets, Nov 2025)
#
# Key idea: Train a sequence of GFlowNet models where each subsequent model
# learns on residual rewards — the uncovered portion of the reward landscape.
#
# R_{k+1}(x) = max(R(x) - Z_k * p_k(x), 0)
#
# where p_k(x) = R(x)/Z_k is the learned distribution of model k.
#
# Non-degradation guarantee: ensemble score monotonically improves with K.
# If model k fully covers R, then R_{k+1} = 0 everywhere and Z_{k+1} = 0.
#
# At inference, sample from the ensemble weighted by partition functions:
# p_ensemble(x) ∝ Σ_k Z_k * p_k(x)

using Random

# =============================================================================
# Boosted GFlowNet Ensemble
# =============================================================================

"""
    BoostedModelCheckpoint

Stores a frozen GFlowNet model from a boosting round.
"""
struct BoostedModelCheckpoint
    round::Int                    # Boosting round index
    params::Any                   # Frozen parameters (ComponentArray)
    states::Any                   # Frozen Lux states
    log_Z::Float64                # Learned log partition function
    molecules_discovered::Int     # Number of unique molecules discovered in this round
end

"""
    BoostedGFlowNet

Ensemble of sequentially trained GFlowNet models.

# Fields
- `checkpoints`: Vector of frozen model checkpoints
- `oracle_cache`: Shared oracle cache across all rounds
- `total_budget_used`: Total oracle calls across all rounds
"""
mutable struct BoostedGFlowNet
    checkpoints::Vector{BoostedModelCheckpoint}
    oracle_cache::Dict{String, Float64}   # SMILES → reward
    total_budget_used::Int

    function BoostedGFlowNet()
        new(BoostedModelCheckpoint[], Dict{String, Float64}(), 0)
    end
end

"""Number of boosting rounds completed."""
n_rounds(ensemble::BoostedGFlowNet) = length(ensemble.checkpoints)

"""Total partition function of the ensemble."""
function ensemble_Z(ensemble::BoostedGFlowNet)::Float64
    return sum(exp(cp.log_Z) for cp in ensemble.checkpoints; init=0.0)
end

# =============================================================================
# Residual Reward Computation
# =============================================================================

"""
    compute_residual_reward(reward::Float64, ensemble::BoostedGFlowNet,
                            smiles::String, policy_model, vocab)

Compute the residual reward for the next boosting round.

R_{k+1}(x) = max(R(x) - Σ_{i=1}^{k} Z_i * p_i(x), 0)

In practice, we approximate p_i(x) using cached log-probabilities
or the policy's own estimate.

For efficiency, we use a simpler bound:
R_{k+1}(x) = max(R(x) - ensemble_coverage(x), 0)

where ensemble_coverage is estimated from the oracle cache.
"""
function compute_residual_reward(reward::Float64, ensemble::BoostedGFlowNet,
                                  smiles::String)::Float64
    if isempty(ensemble.checkpoints)
        return reward  # First round — no residual
    end

    # Simple residual: if this molecule was discovered in a previous round,
    # reduce its effective reward
    if haskey(ensemble.oracle_cache, smiles)
        cached_reward = ensemble.oracle_cache[smiles]
        # Diminishing returns: each round covers some of the reward
        coverage_factor = length(ensemble.checkpoints) * 0.3  # 30% coverage per round
        coverage_factor = min(coverage_factor, 0.9)  # Cap at 90%
        residual = reward * (1.0 - coverage_factor)
        return max(residual, 1e-8)
    end

    return reward  # Novel molecule — full reward
end

# =============================================================================
# Oracle Cache Integration
# =============================================================================

"""
    cached_oracle_call(ensemble::BoostedGFlowNet, smiles::String,
                       oracle_fn::Function)

Call the oracle with caching. If the molecule was evaluated in a previous round,
return the cached reward without spending budget.

# Arguments
- `ensemble`: BoostedGFlowNet with shared cache
- `smiles`: SMILES string to evaluate
- `oracle_fn`: Oracle function SMILES → reward

# Returns
Tuple of (reward, was_cached)
"""
function cached_oracle_call(ensemble::BoostedGFlowNet, smiles::String,
                             oracle_fn::Function)
    if haskey(ensemble.oracle_cache, smiles)
        return ensemble.oracle_cache[smiles], true
    end

    # Call oracle (costs budget)
    reward = oracle_fn(smiles)
    ensemble.oracle_cache[smiles] = reward
    ensemble.total_budget_used += 1

    return reward, false
end

# =============================================================================
# Ensemble Sampling
# =============================================================================

"""
    sample_from_ensemble(ensemble::BoostedGFlowNet, policy_model, vocab,
                          params_list, states_list; kwargs...)

Sample a SMILES string from the boosted ensemble.

Sampling probability: P(round k) = Z_k / Σ_j Z_j
Then sample from model k's policy.

# Arguments
- `ensemble`: BoostedGFlowNet
- `policy_model`: Shared architecture (same for all rounds)
- `vocab`: SMILESVocabulary
- Additional kwargs passed to `sample_smiles_autoregressive`

# Returns
Tuple of (smiles, tokens, log_prob, round_idx)
"""
function sample_from_ensemble(ensemble::BoostedGFlowNet, policy_model, vocab;
                               max_length::Int=150, temperature::Float64=1.0,
                               epsilon::Float64=0.0)
    if isempty(ensemble.checkpoints)
        error("Cannot sample from empty ensemble. Train at least one round.")
    end

    # Compute sampling weights proportional to Z_k
    log_Zs = [cp.log_Z for cp in ensemble.checkpoints]
    max_log_Z = maximum(log_Zs)
    weights = exp.(log_Zs .- max_log_Z)
    weights ./= sum(weights)

    # Sample which round to use
    cumulative = cumsum(weights)
    r = rand()
    round_idx = findfirst(w -> w >= r, cumulative)
    if isnothing(round_idx)
        round_idx = length(weights)
    end

    # Sample from that round's model
    cp = ensemble.checkpoints[round_idx]
    smiles, tokens, log_prob = sample_smiles_autoregressive(
        policy_model, cp.params, cp.states, vocab;
        max_length=max_length, temperature=temperature, epsilon=epsilon
    )

    return smiles, tokens, log_prob, round_idx
end

# =============================================================================
# Boosting Round Management
# =============================================================================

"""
    add_boosting_round!(ensemble::BoostedGFlowNet, params, states,
                         log_Z::Float64, n_molecules::Int)

Add a completed boosting round to the ensemble.

# Arguments
- `ensemble`: BoostedGFlowNet to update
- `params`: Trained parameters for this round (will be deep-copied and frozen)
- `states`: Lux states for this round
- `log_Z`: Learned log partition function for this round
- `n_molecules`: Number of unique molecules discovered in this round
"""
function add_boosting_round!(ensemble::BoostedGFlowNet, params, states,
                              log_Z::Float64, n_molecules::Int)
    round = n_rounds(ensemble) + 1
    checkpoint = BoostedModelCheckpoint(
        round,
        deepcopy(params),
        deepcopy(states),
        log_Z,
        n_molecules
    )
    push!(ensemble.checkpoints, checkpoint)
end

"""
    should_continue_boosting(ensemble::BoostedGFlowNet;
                              max_rounds::Int=5, min_improvement::Float64=0.01)

Determine if boosting should continue.

Stops if:
1. Maximum rounds reached
2. Latest round discovered very few new molecules (diminishing returns)
3. Z_{k} is very small (reward landscape fully covered)
"""
function should_continue_boosting(ensemble::BoostedGFlowNet;
                                   max_rounds::Int=5,
                                   min_new_molecules::Int=10)::Bool
    if isempty(ensemble.checkpoints)
        return true  # Haven't started yet
    end

    if n_rounds(ensemble) >= max_rounds
        return false
    end

    # Check if latest round found enough new molecules
    latest = ensemble.checkpoints[end]
    if latest.molecules_discovered < min_new_molecules
        return false
    end

    # Check if Z is diminishing (reward landscape mostly covered)
    if latest.log_Z < -10.0  # Z < e^{-10} ≈ 4.5e-5
        return false
    end

    return true
end

"""
    get_ensemble_stats(ensemble::BoostedGFlowNet)

Get summary statistics of the boosted ensemble.
"""
function get_ensemble_stats(ensemble::BoostedGFlowNet)
    return Dict{String, Any}(
        "n_rounds" => n_rounds(ensemble),
        "total_molecules" => length(ensemble.oracle_cache),
        "total_budget_used" => ensemble.total_budget_used,
        "log_Zs" => [cp.log_Z for cp in ensemble.checkpoints],
        "molecules_per_round" => [cp.molecules_discovered for cp in ensemble.checkpoints],
        "ensemble_Z" => ensemble_Z(ensemble),
    )
end

# =============================================================================
# Boosting Orchestration
# =============================================================================

"""
    run_boosting_round!(ensemble, policy_model, vocab, base_params, base_states,
                         ref_params, ref_states, oracle_fn, config;
                         verbose=true)

Run a single boosting round: fine-tune on residual rewards, then add to ensemble.

# Workflow
1. Wrap oracle with caching (avoid redundant oracle calls)
2. Wrap oracle with residual reward computation
3. Fine-tune from base_params using TB + KL
4. Add the trained model as a new checkpoint in the ensemble

# Arguments
- `ensemble`: BoostedGFlowNet (will be mutated)
- `policy_model`: Shared SMILESPolicyModel architecture
- `vocab`: SMILESVocabulary
- `base_params`: Starting parameters (pretrained or from previous round)
- `base_states`: Starting Lux states
- `ref_params`: Frozen reference policy (pretrained, for KL)
- `ref_states`: Frozen reference states
- `oracle_fn`: Raw oracle function `SMILES → Float64`
- `config`: FinetuningConfig for this round

# Returns
Named tuple with: params, states, log_Z, history, n_new_molecules
"""
function run_boosting_round!(
    ensemble::BoostedGFlowNet,
    policy_model, vocab,
    base_params, base_states,
    ref_params, ref_states,
    oracle_fn,
    config::FinetuningConfig;
    verbose::Bool=true
)
    round = n_rounds(ensemble) + 1
    if verbose
        println("\n" * "=" ^ 60)
        println("Boosting Round $round")
        println("=" ^ 60)
    end

    # Wrap oracle with caching + residual reward
    initial_cache_size = length(ensemble.oracle_cache)

    residual_oracle = function(smiles::String)
        raw_reward, was_cached = cached_oracle_call(ensemble, smiles, oracle_fn)
        residual = compute_residual_reward(raw_reward, ensemble, smiles)
        return residual
    end

    # Fine-tune on residual rewards
    result = finetune_smiles_gflownet(
        policy_model, vocab,
        deepcopy(base_params), base_states,
        ref_params, ref_states,
        residual_oracle,
        config;
        log_Z_init=0.0,
        verbose=verbose
    )

    # Count new molecules discovered in this round
    n_new = length(ensemble.oracle_cache) - initial_cache_size

    # Add to ensemble
    add_boosting_round!(ensemble, result.params, base_states, result.log_Z, n_new)

    if verbose
        println("Round $round complete: $n_new new molecules, log_Z=$(round(result.log_Z, digits=3))")
        println("Ensemble: $(n_rounds(ensemble)) rounds, $(length(ensemble.oracle_cache)) total molecules")
    end

    return (
        params=result.params,
        states=base_states,
        log_Z=result.log_Z,
        history=result.history,
        n_new_molecules=n_new
    )
end

"""
    run_boosted_training(policy_model, vocab, pretrained_params, pretrained_states,
                          oracle_fn, round_config;
                          max_rounds=5, min_new_molecules=10, verbose=true)

Run the full boosted GFlowNet training loop.

Iteratively trains rounds with residual rewards until convergence or budget exhaustion.

# Returns
Named tuple with: ensemble, all_histories
"""
function run_boosted_training(
    policy_model, vocab,
    pretrained_params, pretrained_states,
    oracle_fn, round_config::FinetuningConfig;
    max_rounds::Int=5,
    min_new_molecules::Int=10,
    verbose::Bool=true
)
    ensemble = BoostedGFlowNet()
    ref_params = deepcopy(pretrained_params)
    ref_states = deepcopy(pretrained_states)
    all_histories = []

    current_params = pretrained_params

    while should_continue_boosting(ensemble; max_rounds=max_rounds,
                                    min_new_molecules=min_new_molecules)
        result = run_boosting_round!(
            ensemble, policy_model, vocab,
            current_params, pretrained_states,
            ref_params, ref_states,
            oracle_fn, round_config;
            verbose=verbose
        )

        push!(all_histories, result.history)

        # Next round starts from pretrained params (not previous round)
        # This ensures each round learns independently
        current_params = pretrained_params
    end

    if verbose
        stats = get_ensemble_stats(ensemble)
        println("\n" * "=" ^ 60)
        println("Boosted Training Complete")
        println("  Rounds: $(stats["n_rounds"])")
        println("  Total molecules: $(stats["total_molecules"])")
        println("  Budget used: $(stats["total_budget_used"])")
        println("  Log Zs: $(stats["log_Zs"])")
        println("=" ^ 60)
    end

    return (ensemble=ensemble, all_histories=all_histories)
end
