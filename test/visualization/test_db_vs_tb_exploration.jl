# Compare TB vs DB for exploration and mode discovery
# DB has local balance constraints that may help exploration

using Test
using GFlowNet
using Statistics

println("=" ^ 70)
println("Comparing TB vs DB for Mode Discovery")
println("=" ^ 70)

@testset "TB vs DB Exploration Comparison" begin

    function train_and_evaluate(objective_name, objective, iterations; include_backward=false)
        model = GFlowNet.create_grid_world_gflownet(
            grid_size = 5,
            reward_positions = Dict(
                (5, 5) => 10.0,
                (1, 5) => 8.0
            ),
            partition_function_method = GFlowNet.LEARNABLE_ESTIMATION,
            hidden_dim = 64,
            include_backward = include_backward
        )

        config = GFlowNet.TrainingConfig(
            objective = objective,
            batch_size = 64,
            n_iterations = iterations,
            learning_rate = 0.005,
            temperature = 3.0
        )

        history = GFlowNet.train_gflownet(model, config; verbose=false)

        # Sample
        trajectories = [GFlowNet.sample_trajectory(model) for _ in 1:1000]

        position_counts = Dict{Tuple{Int,Int}, Int}()
        for traj in trajectories
            if !isempty(traj.states) && GFlowNet.is_terminal_state(traj.states[end])
                pos = (traj.states[end].x, traj.states[end].y)
                position_counts[pos] = get(position_counts, pos, 0) + 1
            end
        end

        peak1 = get(position_counts, (5, 5), 0)
        peak2 = get(position_counts, (1, 5), 0)
        others = 1000 - peak1 - peak2
        ratio = peak2 > 0 ? peak1 / peak2 : Inf

        valid_losses = filter(!isnan, history.losses)
        final_loss = isempty(valid_losses) ? NaN : valid_losses[end]

        learned_Z = haskey(model.parameters, :log_Z) ? exp(model.parameters.log_Z) : 1.0

        return Dict(
            "name" => objective_name,
            "peak1" => peak1,
            "peak2" => peak2,
            "others" => others,
            "ratio" => ratio,
            "final_loss" => final_loss,
            "learned_Z" => learned_Z,
            "modes_found" => (peak1 > 10 ? 1 : 0) + (peak2 > 10 ? 1 : 0)
        )
    end

    println("\n🎯 Expected Results:")
    println("   Peak1 (5,5) R=10: ~55.6% of samples to peaks")
    println("   Peak2 (1,5) R=8:  ~44.4% of samples to peaks")
    println("   Expected ratio: 1.25")
    println("   Expected Z ≈ 57 (sum of all rewards)")

    println("\n" * "-" ^ 70)
    println("Training with TRAJECTORY_BALANCE (2000 iterations)")
    println("-" ^ 70)

    tb_result = train_and_evaluate(
        "TRAJECTORY_BALANCE",
        GFlowNet.TRAJECTORY_BALANCE,
        2000,
        include_backward = false
    )

    println("TB Results:")
    println("   Peak1 (5,5): $(tb_result["peak1"]) samples ($(round(100*tb_result["peak1"]/1000, digits=1))%)")
    println("   Peak2 (1,5): $(tb_result["peak2"]) samples ($(round(100*tb_result["peak2"]/1000, digits=1))%)")
    println("   Other:       $(tb_result["others"]) samples")
    println("   Ratio:       $(round(tb_result["ratio"], digits=2)) (expected: 1.25)")
    println("   Final Loss:  $(round(tb_result["final_loss"], digits=6))")
    println("   Learned Z:   $(round(tb_result["learned_Z"], digits=2)) (expected: ~57)")
    println("   Modes found: $(tb_result["modes_found"])/2")

    println("\n" * "-" ^ 70)
    println("Training with DETAILED_BALANCE (2000 iterations)")
    println("-" ^ 70)

    db_result = train_and_evaluate(
        "DETAILED_BALANCE",
        GFlowNet.DETAILED_BALANCE,
        2000,
        include_backward = true
    )

    println("DB Results:")
    println("   Peak1 (5,5): $(db_result["peak1"]) samples ($(round(100*db_result["peak1"]/1000, digits=1))%)")
    println("   Peak2 (1,5): $(db_result["peak2"]) samples ($(round(100*db_result["peak2"]/1000, digits=1))%)")
    println("   Other:       $(db_result["others"]) samples")
    println("   Ratio:       $(round(db_result["ratio"], digits=2)) (expected: 1.25)")
    println("   Final Loss:  $(round(db_result["final_loss"], digits=6))")
    println("   Learned Z:   $(round(db_result["learned_Z"], digits=2))")
    println("   Modes found: $(db_result["modes_found"])/2")

    println("\n" * "=" ^ 70)
    println("📊 COMPARISON SUMMARY")
    println("=" ^ 70)
    println("\n| Metric          | TB           | DB           | Expected |")
    println("|-----------------|--------------|--------------|----------|")
    println("| Peak1 (5,5)     | $(lpad(tb_result["peak1"], 10)) | $(lpad(db_result["peak1"], 10)) | ~556     |")
    println("| Peak2 (1,5)     | $(lpad(tb_result["peak2"], 10)) | $(lpad(db_result["peak2"], 10)) | ~444     |")
    println("| Ratio           | $(lpad(round(tb_result["ratio"], digits=2), 10)) | $(lpad(round(db_result["ratio"], digits=2), 10)) | 1.25     |")
    println("| Modes found     | $(lpad(tb_result["modes_found"], 10))/2 | $(lpad(db_result["modes_found"], 10))/2 | 2/2      |")
    println("| Loss            | $(lpad(round(tb_result["final_loss"], digits=4), 10)) | $(lpad(round(db_result["final_loss"], digits=4), 10)) | low      |")

    # Tests
    @test db_result["modes_found"] >= tb_result["modes_found"]

    if db_result["peak2"] > 10 && tb_result["peak2"] > 10
        db_error = abs(db_result["ratio"] - 1.25)
        tb_error = abs(tb_result["ratio"] - 1.25)
        println("\nRatio error: TB=$(round(tb_error, digits=2)), DB=$(round(db_error, digits=2))")
        if db_error < tb_error
            println("✅ DB has better ratio convergence!")
        else
            println("⚠️ TB has better ratio convergence (unexpected)")
        end
    end

    println("\n" * "=" ^ 70)
    println("Analysis Complete")
    println("=" ^ 70)
end
