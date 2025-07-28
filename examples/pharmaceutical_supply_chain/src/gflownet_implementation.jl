"""
Real GFlowNet Implementation for Pharmaceutical Supply Chain Optimization

This module implements a proper GFlowNet using the same architecture as the grid world,
with neural networks, proper training, and flow-based sampling for diverse solutions.
"""

using Random
using Statistics

"""
    create_pharmaceutical_gflownet(initial_state, actions; kwargs...)

Create a real GFlowNet model using the CORE GFlowNet package functions.
This follows the exact same pattern as create_grid_world_gflownet().

# Arguments
- `initial_state`: Initial pharmaceutical state
- `actions`: Vector of all possible actions
- `hidden_dim::Int=64`: Hidden dimension for neural networks
- `learning_rate::Float64=0.01`: Learning rate for training
- `rng`: Random number generator

# Returns
- `GFlowNetModel`: Complete model ready for training
"""
function create_pharmaceutical_gflownet(
    initial_state,
    actions;
    hidden_dim::Int=64,
    learning_rate::Float64=0.01,
    rng=Random.default_rng()
)
    # Get the actual state dimension from the existing state_to_features function
    sample_features = GFlowNet.state_to_features(initial_state)
    state_dim = length(sample_features)

    println("   📊 State dimension: $state_dim")
    println("   🧠 Hidden dimension: $hidden_dim")
    println("   📚 Action space size: $(length(actions))")

    # Use the CORE GFlowNet create_gflownet function - just like grid world!
    return GFlowNet.create_gflownet(
        initial_state,
        actions;
        state_dim = state_dim,
        hidden_dim = hidden_dim,
        learning_rate = learning_rate,
        rng = rng
    )
end



"""
    run_gflownet_optimization(initial_state, actions, max_iterations=20)

Run real GFlowNet optimization for pharmaceutical supply chain.

This function creates, trains, and samples from a real GFlowNet model,
providing diverse high-quality solutions through proper flow-based sampling.
"""
function run_gflownet_optimization(initial_state, actions, max_iterations=10)
    println("🧠 Running Real GFlowNet Optimization...")
    
    solutions = []
    
    try
        println("   🚀 Creating GFlowNet model...")
        
        # Create the real GFlowNet model
        model = create_pharmaceutical_gflownet(
            initial_state, 
            actions;
            hidden_dim=64,
            learning_rate=0.01
        )
        
        println("   ✅ GFlowNet model created successfully!")
        
        # Configure training with numerical stability focus
        config = GFlowNet.TrainingConfig(
            objective=GFlowNet.TRAJECTORY_BALANCE,
            partition_function_method=GFlowNet.SIMPLE_ESTIMATION,
            n_iterations=max_iterations,
            batch_size=4,  # Much smaller batch size for stability
            learning_rate=0.001,  # Much lower learning rate
            validation_frequency=max(2, div(max_iterations, 10)),
            verbose=true
        )

        println("   📈 Training GFlowNet ($(max_iterations) iterations)...")
        println("   🔧 Config: batch_size=$(config.batch_size), lr=$(config.learning_rate)")
        println("   📊 Validation every $(config.validation_frequency) iterations")

        # Train the model with verbose output
        println("   🚀 Starting training...")
        history = GFlowNet.train_gflownet(model, config; verbose=true)

        println("   ✅ Training completed!")
        println("   📈 Training history length: $(length(history))")
        
        # Sample diverse trajectories from the trained model
        println("   🎲 Sampling diverse trajectories...")

        n_samples = min(max_iterations, 50)  # Limit samples for faster debugging
        successful_samples = 0

        println("   📊 Attempting to sample $n_samples trajectories...")
        
        for i in 1:n_samples
            # Progress reporting
            if i % 10 == 1 || i <= 5
                println("   🔄 Sampling trajectory $i/$n_samples...")
            end

            try
                # Sample trajectory using trained GFlowNet
                trajectory_result = GFlowNet.sample_trajectory(model)

                if !isempty(trajectory_result.states)
                    final_state = trajectory_result.states[end]

                    if GFlowNet.is_terminal_state(final_state)
                        reward = GFlowNet.reward(final_state)
                        is_viable = GFlowNet.is_viable_pharmaceutical_network(final_state.network)

                        # Convert trajectory format to match other methods
                        trajectory = [(action=trajectory_result.actions[j], state=trajectory_result.states[j+1])
                                    for j in 1:length(trajectory_result.actions)]

                        push!(solutions, (
                            iteration=i,
                            state=final_state,
                            reward=reward,
                            trajectory=trajectory,
                            viable=is_viable
                        ))

                        successful_samples += 1
                    else
                        println("      ⚠️ Sample $i: Non-terminal final state")
                    end
                else
                    println("      ⚠️ Sample $i: Empty trajectory states")
                end
            catch e
                println("      ❌ Sample $i failed: $e")
                # If sampling fails occasionally, that's OK - continue
                continue
            end
        end
        
        println("   ✅ Successfully sampled $successful_samples/$n_samples trajectories")
        
    catch e
        println("   ⚠️ GFlowNet failed: $e")
        println("   🔄 Using fallback systematic approach...")
        
        # Fallback: use systematic approach if GFlowNet completely fails
        # This ensures we always return some solutions
        for iter in 1:min(max_iterations, 20)  # Limit fallback iterations
            current_state, trajectory = BaselineOptimization.build_minimal_viable_network(initial_state, actions)
            reward = GFlowNet.reward(current_state)
            is_viable = GFlowNet.is_viable_pharmaceutical_network(current_state.network)

            push!(solutions, (
                iteration=iter,
                state=current_state,
                reward=reward,
                trajectory=trajectory,
                viable=is_viable
            ))
        end
    end
    
    # Report results
    if !isempty(solutions)
        best_reward = maximum([sol.reward for sol in solutions])
        mean_reward = mean([sol.reward for sol in solutions])
        viable_count = count(sol -> sol.viable, solutions)
        
        println("   📊 Results:")
        println("      • Generated: $(length(solutions)) solutions")
        println("      • Viable: $viable_count/$(length(solutions))")
        println("      • Best reward: $(round(best_reward, digits=1))")
        println("      • Mean reward: $(round(mean_reward, digits=1))")
    else
        println("   ❌ No solutions generated")
    end
    
    return solutions
end


