# Comprehensive test for new features: replay buffer, genetic ops, config changes
# Run: julia --project=. test/smiles_gflownet/test_new_features.jl

println("=" ^ 60)
println("COMPREHENSIVE VERIFICATION: All New Features")
println("=" ^ 60)

include(joinpath(@__DIR__, "..", "..", "src", "GFlowNet.jl"))
using .GFlowNet

n_passed = 0
n_failed = 0

# ─── Test 1: FinetuningConfig new fields ───
print("Test 1: FinetuningConfig new fields... ")
try
    cfg = FinetuningConfig(
        n_iterations=50,
        use_replay=true,
        replay_ratio=8,
        use_qgfn_sampling=true,
        kl_weight=0.005,
        kl_decay_schedule=:none
    )
    @assert cfg.use_replay == true
    @assert cfg.replay_ratio == 8
    @assert cfg.use_qgfn_sampling == true
    @assert cfg.kl_weight == 0.005
    @assert cfg.kl_decay_schedule == :none
    println("PASSED")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

# ─── Test 2: FinetuningConfig defaults ───
print("Test 2: FinetuningConfig defaults... ")
try
    cfg = FinetuningConfig()
    @assert cfg.kl_weight == 0.01 "Expected 0.01, got $(cfg.kl_weight)"
    @assert cfg.kl_decay_schedule == :none "Expected :none, got $(cfg.kl_decay_schedule)"
    @assert cfg.use_replay == false
    @assert cfg.replay_ratio == 4
    @assert cfg.use_qgfn_sampling == false
    println("PASSED")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

# ─── Test 3: SMILESReplayBuffer basic operations ───
print("Test 3: SMILESReplayBuffer basic operations... ")
try
    buf = SMILESReplayBuffer(100)
    @assert isempty(buf) "Should start empty"
    @assert length(buf) == 0

    # Add entries
    add_to_replay!(buf, "CCO", [0, 5, 5, 10, 1], 0.7)
    add_to_replay!(buf, "CCCO", [0, 5, 5, 5, 10, 1], 0.8)
    add_to_replay!(buf, "CC(=O)O", [0, 5, 5, 20, 21, 10, 22, 1], 0.9)
    @assert length(buf) == 3 "Should have 3 entries"

    # Deduplication: higher reward should update
    add_to_replay!(buf, "CCO", [0, 5, 5, 10, 1], 0.95)
    @assert length(buf) == 3 "Should still be 3 (dedup)"

    # Deduplication: lower reward should NOT update
    add_to_replay!(buf, "CCO", [0, 5, 5, 10, 1], 0.1)
    @assert length(buf) == 3

    # Invalid entries should be rejected
    add_to_replay!(buf, "", [0, 1], 0.5)     # empty SMILES
    add_to_replay!(buf, "C", [0], -0.1)       # negative reward
    add_to_replay!(buf, "C", Int[], 0.5)       # too short tokens
    @assert length(buf) == 3 "Invalid entries should be rejected"

    println("PASSED")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

# ─── Test 4: Rank-weighted sampling with replacement ───
print("Test 4: Rank-weighted sampling (n > buffer size)... ")
try
    buf = SMILESReplayBuffer(100)
    add_to_replay!(buf, "CCO", [0, 5, 5, 10, 1], 0.7)
    add_to_replay!(buf, "CCCO", [0, 5, 5, 5, 10, 1], 0.8)
    add_to_replay!(buf, "CC(=O)O", [0, 5, 5, 20, 21, 10, 22, 1], 0.9)

    # Rank-weighted with replacement: n > buffer size should work
    sampled = sample_replay(buf, 10; rank_weighted=true)
    @assert length(sampled) == 10 "Should get 10 samples (with replacement), got $(length(sampled))"

    # All entries should be valid SMILESReplayEntry
    for entry in sampled
        @assert entry.smiles != "" "Each entry should have non-empty smiles"
        @assert entry.reward > 0 "Each entry should have positive reward"
    end

    println("PASSED")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

# ─── Test 5: Top molecules & stats ───
print("Test 5: get_top_molecules & replay_stats... ")
try
    buf = SMILESReplayBuffer(100)
    add_to_replay!(buf, "CCO", [0, 5, 5, 10, 1], 0.7)
    add_to_replay!(buf, "CCCO", [0, 5, 5, 5, 10, 1], 0.8)
    add_to_replay!(buf, "CC(=O)O", [0, 5, 5, 20, 21, 10, 22, 1], 0.9)

    top = get_top_molecules(buf, 2)
    @assert length(top) == 2
    @assert top[1].reward >= top[2].reward "Should be sorted by reward (descending)"
    @assert top[1].reward == 0.9

    stats = replay_stats(buf)
    @assert stats["size"] == 3
    @assert stats["max_reward"] == 0.9
    @assert stats["unique_smiles"] == 3

    println("PASSED")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

# ─── Test 6: Buffer eviction at capacity ───
print("Test 6: Buffer eviction at capacity... ")
try
    buf = SMILESReplayBuffer(3)  # small buffer
    add_to_replay!(buf, "A", [0, 1, 1], 0.1)
    add_to_replay!(buf, "B", [0, 2, 1], 0.5)
    add_to_replay!(buf, "C", [0, 3, 1], 0.9)
    add_to_replay!(buf, "D", [0, 4, 1], 0.3)  # should evict A (lowest 0.1)

    @assert length(buf) == 3 "Should stay at max capacity"
    top = get_top_molecules(buf, 3)
    smiles_in_buf = Set(e.smiles for e in top)
    @assert "A" ∉ smiles_in_buf "A (reward 0.1) should have been evicted"
    @assert "C" ∈ smiles_in_buf "C (reward 0.9) should remain"

    println("PASSED")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

# ─── Test 7: Batch add ───
print("Test 7: add_batch_to_replay!... ")
try
    buf = SMILESReplayBuffer(100)
    smiles = ["CCO", "CCCO", "CC(=O)O"]
    tokens = [[0, 5, 5, 10, 1], [0, 5, 5, 5, 10, 1], [0, 5, 5, 20, 21, 10, 22, 1]]
    rewards = [0.7, 0.8, 0.9]
    add_batch_to_replay!(buf, smiles, tokens, rewards)
    @assert length(buf) == 3
    @assert buf.needs_sort == false "Should be sorted after batch add"

    println("PASSED")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

# ─── Test 8: ScaffoldFilter ───
print("Test 8: ScaffoldFilter logic... ")
try
    sf = ScaffoldFilter(max_per_scaffold=2)
    @assert sf.max_per_scaffold == 2
    @assert isempty(sf.scaffold_counts)

    stats = scaffold_diversity_stats(sf)
    @assert stats["n_scaffolds"] == 0

    # Manually test filter logic (without RDKit)
    sf.scaffold_counts["scaffold_A"] = 1
    @assert sf.scaffold_counts["scaffold_A"] == 1
    sf.scaffold_counts["scaffold_A"] = 2

    stats2 = scaffold_diversity_stats(sf)
    @assert stats2["n_scaffolds"] == 1
    @assert stats2["max_count"] == 2
    @assert stats2["over_represented"] == 1

    println("PASSED")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

# ─── Test 9: Token-level mutation ───
print("Test 9: smiles_mutate_tokens... ")
try
    vocab = SMILESVocabulary()
    for tok in ["C", "N", "O", "S", "c", "n", "(", ")", "=", "#"]
        get_or_add_token!(vocab, tok)
    end

    tokens = [START_TOKEN, 3, 4, 5, 3, END_TOKEN]
    mutated = smiles_mutate_tokens(tokens, vocab; n_mutations=2)
    @assert length(mutated) == length(tokens) "Length should be preserved"
    @assert mutated[1] == START_TOKEN "START should be preserved"
    @assert mutated[end] == END_TOKEN "END should be preserved"

    println("PASSED")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

# ─── Test 10: compute_replay_loss exists ───
print("Test 10: compute_replay_loss exists... ")
try
    @assert isdefined(GFlowNet, :compute_replay_loss) "compute_replay_loss should be exported"
    println("PASSED")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

# ─── Test 11: Uniform sampling caps at buffer size ───
print("Test 11: Uniform sampling caps at buffer size... ")
try
    buf = SMILESReplayBuffer(100)
    add_to_replay!(buf, "CCO", [0, 5, 5, 10, 1], 0.7)
    add_to_replay!(buf, "CCCO", [0, 5, 5, 5, 10, 1], 0.8)

    sampled = sample_replay(buf, 10; rank_weighted=false)
    @assert length(sampled) == 2 "Uniform sampling should cap at buffer size, got $(length(sampled))"

    println("PASSED")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

# ─── Test 12: Exports check ───
print("Test 12: All new exports exist... ")
try
    # Replay buffer
    @assert isdefined(GFlowNet, :SMILESReplayEntry)
    @assert isdefined(GFlowNet, :SMILESReplayBuffer)
    @assert isdefined(GFlowNet, :add_to_replay!)
    @assert isdefined(GFlowNet, :add_batch_to_replay!)
    @assert isdefined(GFlowNet, :sample_replay)
    @assert isdefined(GFlowNet, :get_top_molecules)
    @assert isdefined(GFlowNet, :replay_stats)

    # Genetic operations
    @assert isdefined(GFlowNet, :augment_smiles_rdkit)
    @assert isdefined(GFlowNet, :create_augment_fn)
    @assert isdefined(GFlowNet, :smiles_crossover_rdkit)
    @assert isdefined(GFlowNet, :smiles_mutate_rdkit)
    @assert isdefined(GFlowNet, :smiles_mutate_tokens)
    @assert isdefined(GFlowNet, :ScaffoldFilter)
    @assert isdefined(GFlowNet, :get_scaffold)
    @assert isdefined(GFlowNet, :should_add_molecule)
    @assert isdefined(GFlowNet, :register_molecule!)
    @assert isdefined(GFlowNet, :scaffold_diversity_stats)
    @assert isdefined(GFlowNet, :generate_genetic_molecules)

    println("PASSED")
    global n_passed += 1
catch e
    println("FAILED: $e")
    global n_failed += 1
end

# ─── Summary ───
println()
println("=" ^ 60)
println("Results: $n_passed passed, $n_failed failed out of $(n_passed + n_failed)")
println("=" ^ 60)

if n_failed > 0
    exit(1)
end
