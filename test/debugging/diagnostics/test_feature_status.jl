# Test to Document Working vs Broken Features
# This test explicitly shows what works and what doesn't in GFlowNet.jl

using Test
using GFlowNet

@testset "Working vs Broken Features Documentation" begin
    
    @testset "Working Features (Safe to Use)" begin
        # Create a simple model
        model = create_grid_world_gflownet(
            grid_size=3,
            reward_positions=Dict((3,3) => 10.0),
            hidden_dim=16
        )
        
        @testset "Trajectory Balance Training (WORKS)" begin
            # This is what examples use - it works perfectly
            config = TrainingConfig(
                objective=TRAJECTORY_BALANCE,  # ✅ This works
                n_iterations=5,
                batch_size=4,
                learning_rate=0.01
            )
            
            # Training works fine
            @test_nowarn history = train_gflownet(model, config; verbose=false)
            
            # Sampling works fine
            @test_nowarn trajectory = sample_trajectory(model)
            
            # The trajectory balance loss computation only needs:
            # - forward_probability() ✅
            # - get_applicable_actions() ✅
            # - apply_action() ✅
            # - reward() ✅
            # It does NOT need flow(), get_next_states(), etc.
        end
        
        @testset "Core Interface Functions (WORK)" begin
            state = GridState(2, 2, false)
            
            # These all work correctly
            @test length(state_to_features(state)) == 3
            @test !is_terminal_state(state)
            @test is_applicable(MoveRight(), state)
            
            new_state = apply_action(MoveRight(), state)
            @test new_state.x == 3
            
            # On-demand computation works
            actions = get_applicable_actions(state, model.all_actions)
            @test length(actions) > 0
        end
    end
    
    @testset "Broken Features (Will Cause Errors)" begin
        model = create_grid_world_gflownet(grid_size=3, hidden_dim=16)
        state = GridState(2, 2, false)
        
        @testset "Detailed Balance Training (BROKEN)" begin
            # This would fail because model has no backward_policy field
            config = TrainingConfig(
                objective=DETAILED_BALANCE,  # ❌ This is broken
                n_iterations=5,
                batch_size=4
            )
            
            # This would throw an error about missing backward_policy
            @test_broken train_gflownet(model, config; verbose=false)
        end
        
        @testset "Flow Matching Training (BROKEN)" begin
            # This would fail because flow_matching_loss calls get_next_states()
            config = TrainingConfig(
                objective=FLOW_MATCHING,  # ❌ This is broken
                n_iterations=5,
                batch_size=4
            )
            
            # This would throw UndefVarError: get_next_states not defined
            @test_broken train_gflownet(model, config; verbose=false)
        end
        
        @testset "Flow Computation (BROKEN)" begin
            # These functions exist but call non-existent get_next_states()
            @test_throws UndefVarError flow(model, state)
            @test_throws UndefVarError compute_recursive_flow(model, state)
            @test_throws UndefVarError partition_function(model)
            @test_throws UndefVarError validate_flow_conservation(model, state)
        end
        
        @testset "Missing DAG Functions (DON'T EXIST)" begin
            # These functions are called but never defined
            @test_throws UndefVarError get_next_states(model.dag, state)
            @test_throws UndefVarError get_previous_states(model.dag, state)
            @test_throws UndefVarError get_root_state(model.dag)
        end
        
        @testset "Model Fields (DON'T EXIST)" begin
            # Model doesn't have these fields
            @test !hasproperty(model, :backward_policy)
            @test !hasproperty(model, :state_dim)
            @test !hasproperty(model, :dag)  # No explicit DAG object
        end
    end
    
    @testset "Why Examples Work" begin
        # Document the exact code path that makes examples work
        
        @testset "Trajectory Balance Loss Implementation" begin
            # The loss is computed in compute_single_trajectory_loss()
            # It only computes:
            # 1. Sum of log P_F(a|s) for each action in trajectory
            # 2. Log reward of terminal state
            # 3. Loss = (log_prob_sum - log_reward)²
            
            # No flow computation needed!
            # No backward policy needed!
            # No DAG traversal needed!
            
            @test true  # This is documentation
        end
        
        @testset "On-Demand Architecture Success" begin
            # Working functions use:
            # - get_applicable_actions() instead of get_next_states()
            # - apply_action() for state transitions
            # - Simple forward sampling
            
            # This avoids all the broken DAG infrastructure
            @test true  # This is documentation
        end
    end
end

println("""

=== COMPREHENSIVE ANALYSIS: Why Grid World Example Works ===

The grid world example (and other examples) work perfectly because they use 
a specific subset of GFlowNet functionality that has been successfully 
migrated to the new architecture.

1. TRAJECTORY BALANCE IS SELF-CONTAINED
   
   The loss computation in compute_single_trajectory_loss() only needs:
   - log_prob_sum = Σ log P_F(action_i | state_i)  [Forward policy]
   - log_reward = log(reward(terminal_state))       [Reward function]
   - loss = (log_prob_sum - log_reward)²           [Simple MSE]
   
   This enforces P_F(τ) ∝ R(s_T) without needing flow computation!

2. THE WORKING CODE PATH
   
   Training with TRAJECTORY_BALANCE uses:
   ✅ forward_probability() - Works perfectly
   ✅ get_applicable_actions() - On-demand computation
   ✅ apply_action() - Pure state transitions
   ✅ sample_forward_action() - Forward sampling
   ✅ reward() - Terminal rewards
   
   It completely avoids:
   ❌ flow() - Would call non-existent get_next_states()
   ❌ backward_policy - Model doesn't have this field
   ❌ partition_function() - Would call non-existent get_root_state()
   ❌ detailed_balance_loss() - Needs backward_policy
   ❌ flow_matching_loss() - Needs get_next_states()

3. TWO PARALLEL ARCHITECTURES
   
   OLD (Broken):
   - Explicit DAG: DirectedAcyclicGraph objects
   - Functions: get_next_states(), get_previous_states(), get_root_state()
   - Status: Called but never implemented
   
   NEW (Working):
   - On-demand: No explicit DAG construction
   - Functions: get_applicable_actions(), apply_action()
   - Status: Fully implemented and used by examples

4. WHY NO ERRORS IN EXAMPLES
   
   The examples specifically use:
   config = TrainingConfig(
       objective=TRAJECTORY_BALANCE,  # ← This is why it works!
       partition_function_method=SIMPLE_ESTIMATION,  # ← Not actually used
       n_iterations=50
   )
   
   If they used:
   objective=DETAILED_BALANCE  # ← Would fail immediately
   objective=FLOW_MATCHING     # ← Would fail immediately

5. MATHEMATICAL INSIGHT
   
   Trajectory Balance is a complete training objective by itself:
   - It learns P_F(τ) ∝ R(s_T) directly
   - No explicit flow computation needed
   - No backward policy needed
   - This is why GFlowNet training still works!

CONCLUSION: The core training loop has been successfully modernized to 
on-demand computation, but advanced mathematical features (flow computation, 
detailed balance) haven't been migrated yet. Users can train GFlowNets 
successfully as long as they stick to TRAJECTORY_BALANCE objective.
""")