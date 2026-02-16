# Domain-Agnostic Sub-Trajectory Balance Demo
# Shows how SUB_TRAJECTORY_BALANCE works with any domain implementing the GFlowNet interface

using GFlowNet
using Statistics
using Printf

# Define a simple counting domain for demonstration
struct CountingState <: AbstractState
    count::Int
    is_terminal::Bool
end

struct CountingAction <: AbstractAction
    increment::Int
end

# Implement the required GFlowNet interface
function GFlowNet.state_to_features(state::CountingState)::Vector{Float32}
    # Simple features: current count and terminal flag
    return Float32[
        state.count / 100.0,  # Normalize count
        state.is_terminal ? 1.0 : 0.0
    ]
end

function GFlowNet.is_terminal_state(state::CountingState)::Bool
    return state.is_terminal
end

function GFlowNet.reward(state::CountingState)::Float64
    if !state.is_terminal
        return 0.0
    end
    # Reward function: prefer counts around 50
    return exp(-0.01 * (state.count - 50)^2)
end

function GFlowNet.is_applicable(action::CountingAction, state::CountingState)::Bool
    return !state.is_terminal && state.count + action.increment <= 100
end

function GFlowNet.apply_action(action::CountingAction, state::CountingState)::CountingState
    new_count = state.count + action.increment
    # Terminate if we reach certain thresholds
    is_terminal = new_count >= 40 && new_count <= 60
    return CountingState(new_count, is_terminal)
end

# Define equality and hashing for proper caching
Base.:(==)(a::CountingState, b::CountingState) = a.count == b.count && a.is_terminal == b.is_terminal
Base.hash(s::CountingState, h::UInt) = hash((s.count, s.is_terminal), h)
Base.:(==)(a::CountingAction, b::CountingAction) = a.increment == b.increment
Base.hash(a::CountingAction, h::UInt) = hash(a.increment, h)

println("=== Domain-Agnostic Sub-Trajectory Balance Demo ===\n")

# Create the domain
initial_state = CountingState(0, false)
all_actions = [CountingAction(i) for i in [1, 2, 5, 10]]  # Different increment sizes

println("Domain: Counting from 0 to terminal states (40-60)")
println("Actions: Increment by $(join([a.increment for a in all_actions], ", "))")
println("Reward: Gaussian centered at count=50\n")

# Create model
model = create_gflownet(
    initial_state,
    all_actions;
    state_dim = 2,
    hidden_dim = 64,
    learning_rate = 0.01
)

# Demonstrate sub-trajectory extraction
println("=== Sub-Trajectory Extraction Example ===")

# Sample a trajectory
sample_traj = sample_trajectory(model)
println("\nSample trajectory:")
for (i, state) in enumerate(sample_traj.states)
    println("  State $i: count=$(state.count), terminal=$(state.is_terminal)")
end

# Show what sub-trajectories would be extracted
println("\nSub-trajectories (max length 3):")
sub_count = 0
for start_idx in 1:length(sample_traj.states)-1
    for end_idx in start_idx+1:min(start_idx+3, length(sample_traj.states))
        sub_states = sample_traj.states[start_idx:end_idx]
        sub_count += 1
        counts = [s.count for s in sub_states]
        println("  Sub-trajectory $sub_count: states $start_idx-$end_idx, counts: $counts")
        
        if sub_count >= 5  # Limit output
            println("  ... (and more)")
            break
        end
    end
    if sub_count >= 5
        break
    end
end

# Compare training objectives
println("\n=== Training Comparison ===")

# Train with TRAJECTORY_BALANCE
println("\n1. Training with TRAJECTORY_BALANCE...")
model_tb = create_gflownet(initial_state, all_actions; state_dim=2, hidden_dim=64)
config_tb = TrainingConfig(
    objective=TRAJECTORY_BALANCE,
    n_iterations=100,
    batch_size=32
)
history_tb = train_gflownet(model_tb, config_tb; verbose=false)

# Train with SUB_TRAJECTORY_BALANCE
println("\n2. Training with SUB_TRAJECTORY_BALANCE...")
model_stb = create_gflownet(initial_state, all_actions; state_dim=2, hidden_dim=64)
config_stb = TrainingConfig(
    objective=SUB_TRAJECTORY_BALANCE,
    n_iterations=100,
    batch_size=32,
    sub_trajectory_length=4
)
history_stb = train_gflownet(model_stb, config_stb; verbose=false)

# Analyze results
println("\n=== Training Results ===")

function analyze_model(model, name)
    # Sample many trajectories
    trajectories = [sample_trajectory(model) for _ in 1:1000]
    
    # Terminal state distribution
    terminal_counts = [t.states[end].count for t in trajectories]
    mean_count = mean(terminal_counts)
    std_count = std(terminal_counts)
    
    # Trajectory lengths
    lengths = [length(t.states) for t in trajectories]
    mean_length = mean(lengths)
    
    # Rewards
    rewards = [reward(t.states[end]) for t in trajectories]
    mean_reward = mean(rewards)
    
    println("\n$name Results:")
    println("  Mean terminal count: $(round(mean_count, digits=2)) ± $(round(std_count, digits=2))")
    println("  Mean trajectory length: $(round(mean_length, digits=2))")
    println("  Mean reward: $(round(mean_reward, digits=4))")
    
    # Show distribution of terminal states
    count_hist = Dict{Int,Int}()
    for c in terminal_counts
        count_hist[c] = get(count_hist, c, 0) + 1
    end
    
    println("  Terminal state distribution:")
    for count in sort(collect(keys(count_hist)))
        freq = count_hist[count] / length(trajectories) * 100
        if freq > 1.0  # Only show counts with >1% frequency
            println("    Count $count: $(round(freq, digits=1))%")
        end
    end
    
    return trajectories, rewards
end

traj_tb, rewards_tb = analyze_model(model_tb, "TRAJECTORY_BALANCE")
traj_stb, rewards_stb = analyze_model(model_stb, "SUB_TRAJECTORY_BALANCE")

# Loss analysis
println("\n=== Loss Analysis ===")

tb_losses = filter(!isnan, history_tb.losses)
stb_losses = filter(!isnan, history_stb.losses)

if !isempty(tb_losses) && !isempty(stb_losses)
    println("\nTRAJECTORY_BALANCE:")
    println("  Initial loss: $(round(tb_losses[1], digits=4))")
    println("  Final loss: $(round(tb_losses[end], digits=4))")
    println("  Loss variance: $(round(var(tb_losses), digits=6))")
    
    println("\nSUB_TRAJECTORY_BALANCE:")
    println("  Initial loss: $(round(stb_losses[1], digits=4))")
    println("  Final loss: $(round(stb_losses[end], digits=4))")
    println("  Loss variance: $(round(var(stb_losses), digits=6))")
end

# Mathematical explanation
println("\n=== Mathematical Insight ===")
println("""
Sub-Trajectory Balance decomposes the learning problem:

1. Full Trajectory Balance:
   P_F(s₀→s₁→...→sₜ) × Z = R(sₜ)
   
2. Sub-Trajectory Balance (for each sub-path i→j):
   P_F(sᵢ→...→sⱼ) × F(sᵢ) = F(sⱼ)

Benefits:
- More learning signals per trajectory (O(T²) vs O(T))
- Better credit assignment to intermediate states
- Lower variance gradients
- Faster convergence for long trajectories

Trade-offs:
- Higher computational cost per trajectory
- Requires accurate flow estimates at all states
""")

println("\n=== Demo Complete ===")

# Key takeaway
println("""
Key Takeaway: SUB_TRAJECTORY_BALANCE is domain-agnostic and works with
any state/action types that implement the GFlowNet interface. It provides
more stable training by learning from partial trajectories, making it
especially valuable for domains with long trajectories or sparse rewards.
""")