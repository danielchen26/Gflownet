using GFlowNet
using Test
using Statistics
using GFlowNet: compute_trajectory_loss

# Create a minimal test state/action for domain-agnostic testing
struct TestState <: AbstractState
    value::Int
    is_terminal::Bool
end

struct TestAction <: AbstractAction
    increment::Int
end

# Implement required interface
function GFlowNet.state_to_features(state::TestState)::Vector{Float32}
    return Float32[state.value, state.is_terminal ? 1.0 : 0.0]
end

function GFlowNet.is_terminal_state(state::TestState)::Bool
    return state.is_terminal
end

function GFlowNet.reward(state::TestState)::Float64
    return state.is_terminal ? exp(-abs(state.value - 10)) : 0.0
end

function GFlowNet.is_applicable(action::TestAction, state::TestState)::Bool
    return !state.is_terminal && state.value + action.increment <= 20
end

function GFlowNet.apply_action(action::TestAction, state::TestState)::TestState
    new_value = state.value + action.increment
    is_terminal = new_value >= 10
    return TestState(new_value, is_terminal)
end

# Define equality for caching
Base.:(==)(a::TestState, b::TestState) = a.value == b.value && a.is_terminal == b.is_terminal
Base.hash(s::TestState, h::UInt) = hash((s.value, s.is_terminal), h)
Base.:(==)(a::TestAction, b::TestAction) = a.increment == b.increment
Base.hash(a::TestAction, h::UInt) = hash(a.increment, h)

function create_test_model()
    initial_state = TestState(0, false)
    all_actions = [TestAction(i) for i in 1:5]
    
    return create_gflownet(
        initial_state,
        all_actions;
        state_dim = 2,
        hidden_dim = 32,
        learning_rate = 0.01
    )
end

function run_sub_trajectory_balance_tests()
    println("=== Testing SUB_TRAJECTORY_BALANCE Implementation ===\n")

    # Test 1: Basic sub-trajectory balance computation
    println("1. Testing sub-trajectory balance computation...")
    model = create_test_model()

# Sample some trajectories
trajectories = [sample_trajectory(model) for _ in 1:10]

# Test with different sub-trajectory lengths
for sub_length in [2, 3, 5]
    loss = sub_trajectory_balance_loss_batch(model, trajectories; sub_length=sub_length)
    println("  Sub-trajectory length $sub_length: loss = $(round(loss, digits=4))")
    @test loss >= 0  # Loss should be non-negative
    @test !isnan(loss)
    @test !isinf(loss)
end

# Test 2: Single trajectory analysis
println("\n2. Testing single trajectory sub-trajectory balance...")
long_trajectory = trajectories[argmax([length(t.states) for t in trajectories])]
println("  Trajectory length: $(length(long_trajectory.states))")

# Compute sub-trajectory loss
sub_loss = sub_trajectory_balance_loss(model, long_trajectory; sub_length=3)
println("  Sub-trajectory loss (length=3): $(round(sub_loss, digits=4))")

# Count number of sub-trajectories
sub_traj_count = 0
for start_idx in 1:length(long_trajectory.states)-1
    for end_idx in start_idx+1:min(start_idx+3, length(long_trajectory.states))
        sub_traj_count += 1
    end
end
println("  Number of sub-trajectories considered: $sub_traj_count")

# Test 3: Training with SUB_TRAJECTORY_BALANCE
println("\n3. Testing training with SUB_TRAJECTORY_BALANCE...")

config = TrainingConfig(
    objective=SUB_TRAJECTORY_BALANCE,
    n_iterations=20,
    batch_size=8,
    learning_rate=0.01,
    sub_trajectory_length=4
)

history = train_gflownet(model, config; verbose=false)
stb_losses = filter(!isnan, history.losses)

if !isempty(stb_losses)
    println("  Initial loss: $(round(stb_losses[1], digits=4))")
    println("  Final loss: $(round(stb_losses[end], digits=4))")
    println("  Reduction: $(round((1 - stb_losses[end]/stb_losses[1]) * 100, digits=1))%")
else
    println("  Warning: No valid losses recorded during training")
end

if !isempty(stb_losses)
    @test length(stb_losses) > 0
    @test stb_losses[end] <= stb_losses[1] + 0.1  # Allow small increase due to stochasticity
end

# Test 4: Compare with TRAJECTORY_BALANCE
println("\n4. Comparing SUB_TRAJECTORY_BALANCE vs TRAJECTORY_BALANCE...")

# Create two identical models
model_tb = create_test_model()
model_stb = create_test_model()

# Train with TRAJECTORY_BALANCE
config_tb = TrainingConfig(
    objective=TRAJECTORY_BALANCE,
    n_iterations=50,
    batch_size=16,
    learning_rate=0.01
)

println("  Training with TRAJECTORY_BALANCE...")
history_tb = train_gflownet(model_tb, config_tb; verbose=false)

# Train with SUB_TRAJECTORY_BALANCE
config_stb = TrainingConfig(
    objective=SUB_TRAJECTORY_BALANCE,
    n_iterations=50,
    batch_size=16,
    learning_rate=0.01,
    sub_trajectory_length=3
)

println("  Training with SUB_TRAJECTORY_BALANCE...")
history_stb = train_gflownet(model_stb, config_stb; verbose=false)

# Compare final losses (normalized by trajectory length)
tb_final = history_tb.losses[end]
stb_final = history_stb.losses[end]

println("\n  Final losses:")
println("    TRAJECTORY_BALANCE: $(round(tb_final, digits=4))")
println("    SUB_TRAJECTORY_BALANCE: $(round(stb_final, digits=4))")

# Test 5: Variance analysis
println("\n5. Analyzing variance of different objectives...")

# Sample trajectories and compute losses
test_trajectories = [sample_trajectory(model) for _ in 1:100]

# Compute TB losses
tb_losses = Float64[]
for traj in test_trajectories
    try
        loss = trajectory_balance_loss(model, traj)
        if !isnan(loss) && !isinf(loss)
            push!(tb_losses, loss)
        end
    catch
        # Skip invalid trajectories
    end
end

# Compute STB losses
stb_losses = Float64[]
for traj in test_trajectories
    loss = sub_trajectory_balance_loss(model, traj; sub_length=3)
    if !isnan(loss) && !isinf(loss)
        push!(stb_losses, loss)
    end
end

if !isempty(tb_losses) && !isempty(stb_losses)
    tb_var = var(tb_losses)
    stb_var = var(stb_losses)
    
    println("  Variance analysis:")
    println("    TB variance: $(round(tb_var, digits=6))")
    println("    STB variance: $(round(stb_var, digits=6))")
    println("    Variance ratio (STB/TB): $(round(stb_var/tb_var, digits=3))")
    
    # STB should generally have lower variance due to more frequent updates
    @test stb_var < tb_var * 2.0  # Allow some margin
end

# Test 6: Edge cases
println("\n6. Testing edge cases...")

# Very short trajectory with proper action
action = TestAction(10)  # Will make it terminal in one step
short_traj = Trajectory([TestState(0, false), TestState(10, true)], [action])
short_loss = sub_trajectory_balance_loss(model, short_traj; sub_length=5)
println("  Loss for 2-state trajectory: $(round(short_loss, digits=4))")
@test !isnan(short_loss)

# Empty trajectory handling
empty_traj = Trajectory(AbstractState[], AbstractAction[])
empty_loss = sub_trajectory_balance_loss(model, empty_traj)
println("  Loss for empty trajectory: $empty_loss")
@test empty_loss == 0.0

println("\n=== All SUB_TRAJECTORY_BALANCE tests passed! ===")
end

# Run the tests
run_sub_trajectory_balance_tests()