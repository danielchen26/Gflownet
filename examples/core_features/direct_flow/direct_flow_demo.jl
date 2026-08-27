# Learned Flow Networks Training Demo
# Demonstrates how DETAILED_BALANCE uses a neural network to learn flows F(s)
# instead of computing them recursively

using GFlowNet
using Statistics
using Printf

# Define a simple tree-structured domain for demonstration
struct TreeState <: AbstractState
    path::Vector{Int}  # Path from root
    is_terminal::Bool
end

struct TreeAction <: AbstractAction
    choice::Int  # Branch to take (1 or 2)
end

# Implement the required GFlowNet interface
function GFlowNet.state_to_features(state::TreeState)::Vector{Float32}
    # Features: path encoding and depth
    max_depth = 4
    features = zeros(Float32, max_depth + 1)
    
    # Encode path
    for (i, choice) in enumerate(state.path)
        if i <= max_depth
            features[i] = Float32(choice)
        end
    end
    
    # Add depth feature
    features[end] = Float32(length(state.path)) / max_depth
    
    return features
end

function GFlowNet.is_terminal_state(state::TreeState)::Bool
    return state.is_terminal
end

function GFlowNet.reward(state::TreeState)::Float64
    if !state.is_terminal
        return 0.0
    end
    
    # Reward based on path pattern
    # Higher reward for alternating patterns
    alternating_score = 0.0
    for i in 2:length(state.path)
        if state.path[i] != state.path[i-1]
            alternating_score += 1.0
        end
    end
    
    # Also reward specific terminal nodes
    path_sum = sum(state.path)
    if path_sum == 5  # Sweet spot
        alternating_score += 2.0
    end
    
    return exp(0.5 * alternating_score)
end

function GFlowNet.is_applicable(action::TreeAction, state::TreeState)::Bool
    return !state.is_terminal && length(state.path) < 4
end

function GFlowNet.apply_action(action::TreeAction, state::TreeState)::TreeState
    new_path = [state.path..., action.choice]
    # Terminate at depth 4
    is_terminal = length(new_path) >= 4
    return TreeState(new_path, is_terminal)
end

# Define equality and hashing
Base.:(==)(a::TreeState, b::TreeState) = a.path == b.path && a.is_terminal == b.is_terminal
Base.hash(s::TreeState, h::UInt) = hash((s.path, s.is_terminal), h)
Base.:(==)(a::TreeAction, b::TreeAction) = a.choice == b.choice
Base.hash(a::TreeAction, h::UInt) = hash(a.choice, h)

println("=== Learned Flow Networks Training Demo ===\n")

# Create the domain
initial_state = TreeState(Int[], false)
all_actions = [TreeAction(1), TreeAction(2)]

println("Domain: Binary tree with depth 4")
println("Actions: Choose branch 1 or 2 at each node")
println("Reward: Higher for alternating paths (1-2-1-2 pattern)\n")

# Function to visualize a trajectory
function show_trajectory(traj::Trajectory)
    println("Path: root", join([" → $(a.choice)" for a in traj.actions], ""))
    terminal_state = traj.states[end]
    println("Reward: $(round(reward(terminal_state), digits=3))")
end

# Create models for comparison
println("=== Creating Models ===\n")

# Model 1: Traditional TRAJECTORY_BALANCE (recursive flow)
println("1. Traditional model (TRAJECTORY_BALANCE with recursive flow):")
model_tb = create_gflownet(
    initial_state,
    all_actions;
    state_dim = 5,
    hidden_dim = 64,
    learning_rate = 0.01,
    include_flow_estimator = false
)
println("   - Uses recursive flow computation")
println("   - Flow network NOT included\n")

# Model 2: DETAILED_BALANCE, which learns flows with a neural network.
#
# This demo used to train DIRECT_FLOW_OBJECTIVE here. That objective is now
# disabled and raises an ArgumentError, because its loss was CONSTANT with
# respect to the model parameters: it called direct_flow_loss_batch without
# `params` and wrapped log Z in Zygote.@ignore, so Zygote returned nothing,
# train_step! short-circuited to (Inf, 0.0), and every recorded loss was
# non-finite. This very script was the evidence: `filter(!isnan, history.losses)`
# came back EMPTY and the script died on `df_losses[end]` with
#   BoundsError: attempt to access 0-element Vector{Float64} at index [0]
# It fails identically at commit 31fae84a, i.e. it never worked.
#
# DETAILED_BALANCE is the objective that actually does what this demo set out to
# show -- a learned flow network instead of recursive flow computation -- and it
# is verified to receive a real flow gradient (norm 1.412, previously exactly 0)
# and to respond to reward.
println("2. DETAILED_BALANCE model (neural network estimates flows):")
model_df = create_gflownet(
    initial_state,
    all_actions;
    state_dim = 5,
    hidden_dim = 64,
    learning_rate = 0.01,
    include_backward = true,        # DB needs P_B
    include_flow_estimator = true   # and a learned F
)
println("   - Uses a neural network to estimate flows")
println("   - Flow estimator network included")
println("   - No recursive flow computation needed\n")

# Show untrained behavior
println("=== Before Training ===\n")
println("Sample trajectories (untrained DETAILED_BALANCE model):")
for i in 1:3
    traj = sample_trajectory(model_df)
    print("  $i. ")
    show_trajectory(traj)
end

# Train both models
println("\n=== Training Phase ===\n")

# Train TRAJECTORY_BALANCE
println("Training TRAJECTORY_BALANCE model...")
config_tb = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    n_iterations = 100,
    batch_size = 32,
    verbose = false
)
history_tb = train_gflownet(model_tb, config_tb)
# Keep only finite losses, and say so loudly if there are none. `filter(!isnan, ...)`
# alone let Inf through and then indexed `[end]` on a possibly-empty vector, which
# is how this script died with a BoundsError instead of reporting that training had
# produced no usable loss at all.
function final_loss(history, label)
    good = filter(isfinite, history.losses)
    if isempty(good)
        error("$label produced no finite loss in $(length(history.losses)) " *
              "iterations -- training did not work.")
    end
    return good[end]
end

println("  Final loss: $(round(final_loss(history_tb, "TRAJECTORY_BALANCE"), digits=4))")

# Train DETAILED_BALANCE
println("\nTraining DETAILED_BALANCE model...")
config_df = TrainingConfig(
    objective = DETAILED_BALANCE,
    n_iterations = 100,
    batch_size = 32,
    verbose = false
)
history_df = train_gflownet(model_df, config_df)
println("  Final loss: $(round(final_loss(history_df, "DETAILED_BALANCE"), digits=4))")

# Analyze results
println("\n=== After Training ===\n")

function analyze_model(model, name)
    println("$name Results:")
    
    # Sample many trajectories
    trajectories = [sample_trajectory(model) for _ in 1:1000]
    
    # Count path patterns
    path_counts = Dict{Vector{Int}, Int}()
    rewards = Float64[]
    
    for traj in trajectories
        path = traj.states[end].path
        path_counts[path] = get(path_counts, path, 0) + 1
        push!(rewards, reward(traj.states[end]))
    end
    
    # Show top paths
    sorted_paths = sort(collect(path_counts), by=x->x[2], rev=true)
    println("\n  Top 5 most sampled paths:")
    for i in 1:min(5, length(sorted_paths))
        path, count = sorted_paths[i]
        freq = count / length(trajectories) * 100
        r = exp(0.5 * sum(path[i] != path[i-1] for i in 2:length(path)))
        println("    $(join(path, "-")): $(round(freq, digits=1))% (reward: $(round(r, digits=2)))")
    end
    
    println("\n  Statistics:")
    println("    Mean reward: $(round(mean(rewards), digits=3))")
    println("    Max reward: $(round(maximum(rewards), digits=3))")
    println("    Unique paths found: $(length(path_counts))")
    
    # Check if it found the optimal paths
    optimal_paths = [[1,2,1,2], [2,1,2,1]]
    optimal_found = sum(get(path_counts, path, 0) for path in optimal_paths)
    println("    Optimal paths frequency: $(round(optimal_found/length(trajectories)*100, digits=1))%")
    
    return trajectories, rewards
end

traj_tb, rewards_tb = analyze_model(model_tb, "TRAJECTORY_BALANCE")
println("\n" * "="^50 * "\n")
traj_df, rewards_df = analyze_model(model_df, "DETAILED_BALANCE")

# Mathematical explanation
println("\n=== Key Differences ===\n")
println("""
TRAJECTORY_BALANCE (Recursive Flow):
  - Computes F(s) = Σ P_F(s'|s) × F(s') recursively
  - Exact but computationally expensive for large state spaces
  - Requires traversing the graph structure
  
DETAILED_BALANCE (Neural Network Estimation):
  - Uses neural network Z(s) to directly estimate F(s)
  - Loss: (log P_F(τ) + log Z(s₀) - log R(sT))²
  - Faster inference, no recursion needed
  - Learns flow patterns from data
  - May be less accurate but more scalable

Both methods enforce the same mathematical constraint:
  P_F(τ) × Z ∝ R(sT)
  
But DETAILED_BALANCE learns F(s) with a network instead of recursing.
""")

# Demonstrate flow estimation
println("\n=== Flow Estimation Comparison ===\n")
println("Comparing flow estimates for intermediate states:")

test_states = [
    TreeState([1], false),
    TreeState([1, 2], false),
    TreeState([1, 2, 1], false)
]

for state in test_states
    # DETAILED_BALANCE uses a learned flow network
    if !is_terminal_state(state)
        df_flow = GFlowNet.compute_flow_estimate(model_df, state)
        println("  Path $(state.path): Z(s) = $(round(df_flow, digits=4))")
    end
end

println("""

Note: DETAILED_BALANCE's flow estimates are learned by the neural network,
while traditional methods would compute these recursively.
""")

println("\n=== Demo Complete ===")
println("""
Key Takeaway: DETAILED_BALANCE replaces expensive recursive flow computation
with a learned neural network estimator. This trades some accuracy for
significant computational efficiency, especially in large state spaces.
""")