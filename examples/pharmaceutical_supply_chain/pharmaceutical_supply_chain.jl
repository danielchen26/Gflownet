"""
🏥 Pharmaceutical Supply Chain GFlowNet Example - Comprehensive Optimization Comparison
========================================================================================

This example demonstrates GFlowNet optimization for complex pharmaceutical supply chain problems,
comparing against traditional optimization methods with comprehensive analysis and reporting.

Key Features:
- Multi-objective optimization (cost, access, compliance, resilience)
- Comparison with 6 different optimization methods
- Mathematical problem formulation and fair comparison
- Professional HTML report generation with visualizations
- Real-world pharmaceutical industry scenarios

Methods Compared:
1. GFlowNets (Generative Flow Networks)
2. Nonlinear Programming (Ipopt)
3. Hill Climbing with restarts
4. Greedy optimization
5. Simulated Annealing
6. Random Search

Mathematical Problem:
- Decision Variables: Facility activation, production, transport, inventory
- Objective: Weighted sum of cost efficiency (30%), patient access (30%), 
           regulatory compliance (20%), supply chain resilience (20%)
- Constraints: Capacity, flow conservation, demand satisfaction, viability
"""

using GFlowNet
using Random
using Dates
using Statistics
using Printf

# Include the baseline methods
include("src/baseline_methods.jl")

# Include the real GFlowNet implementation
include("src/gflownet_implementation.jl")

# Include report generation if available
HAS_REPORTING = try
    include("report_generation.jl")
    true
catch
    println("📋 Note: Report generation not available - using basic output")
    false
end

println("🏥 Pharmaceutical Supply Chain GFlowNet Example")
println("="^70)
println("🕐 Started at: $(Dates.format(now(), "HH:MM:SS"))")
println()

# =============================================================================
# 1. Problem Setup and Configuration
# =============================================================================

println("1️⃣ Setting up Pharmaceutical Supply Chain Scenario")
println("="^55)

# Set random seed for reproducibility
Random.seed!(42)

# Create the pharmaceutical supply chain scenario using the actual GFlowNet package
println("   🏭 Creating realistic pharmaceutical network...")

# Create realistic pharmaceutical data using the actual package functions
facilities_original = create_global_facility_network()
drugs = create_realistic_drug_portfolio()
patient_populations = create_global_patient_populations()
connections = PharmaceuticalConnection[]  # Start with no connections
regulatory_environment = create_regulatory_environment()

# Modify facilities to start some as not established for optimization
facilities = []
for (i, facility) in enumerate(facilities_original)
    # Keep first 2 facilities established (research labs), make others not established
    established = i <= 2
    modified_facility = PharmaceuticalFacility(
        facility.id, facility.type, facility.region, facility.location,
        facility.capacity, facility.fixed_cost_annual, facility.variable_cost_per_unit,
        facility.regulatory_compliance, facility.storage_capabilities,
        facility.quality_rating, facility.lead_time_days, established
    )
    push!(facilities, modified_facility)
end

# Create adjacency matrix
n_facilities = length(facilities)
adjacency_matrix = zeros(Bool, n_facilities, n_facilities)

# Create the pharmaceutical network
network = PharmaceuticalNetwork(drugs, facilities, connections, patient_populations, adjacency_matrix, regulatory_environment)

# Create initial state
initial_state = PharmaceuticalState(
    network,
    Dict{Int, DrugPhase}(),  # drug_phase_status
    Dict{Int, Float64}(),    # facility_utilization
    Dict{Tuple{Int,Int}, Float64}(),  # inventory_levels
    Dict{Int, Set{RegulatoryRegion}}(),  # regulatory_approvals_pending
    0.0,  # total_cost
    0.0,  # patient_access_score
    0.8,  # regulatory_compliance_score
    0.0,  # supply_chain_resilience
    false,  # is_terminal
    36    # time_horizon_months (3 years)
)

# Create possible actions
actions = AbstractAction[]

# Facility establishment actions
println("   Debug: Checking $(length(facilities)) facilities for establishment actions")
for facility in facilities
    println("   Debug: Facility $(facility.id) established=$(facility.established)")
    if !facility.established
        # Create a new facility with established=true
        established_facility = PharmaceuticalFacility(
            facility.id, facility.type, facility.region, facility.location,
            facility.capacity, facility.fixed_cost_annual, facility.variable_cost_per_unit,
            facility.regulatory_compliance, facility.storage_capabilities,
            facility.quality_rating, facility.lead_time_days, true  # Set established=true
        )
        push!(actions, EstablishFacilityAction(established_facility, 50_000_000.0))
        println("   Debug: Added EstablishFacilityAction for facility $(facility.id)")
    end
end

# Capacity expansion actions
for facility in facilities
    for category in instances(DrugCategory)
        push!(actions, ExpandCapacityAction(facility.id, category, 50000.0, 10_000_000.0))
    end
end

# Connection establishment actions
for i in 1:length(facilities), j in 1:length(facilities)
    if i != j
        # Create connection with established=true so it will be active when added
        connection = PharmaceuticalConnection(i, j, Dict(ONCOLOGY => 10000.0), 1.0, 7, 0.95, true, true, true)
        push!(actions, EstablishConnectionAction(connection, 1_000_000.0))
    end
end

# Drug advancement actions
for drug in drugs
    if drug.phase != APPROVED && drug.phase != DISCONTINUED
        next_phase_map = Dict(
            PRECLINICAL => PHASE_I,
            PHASE_I => PHASE_II,
            PHASE_II => PHASE_III,
            PHASE_III => APPROVED
        )
        next_phase = get(next_phase_map, drug.phase, nothing)
        if next_phase !== nothing
            push!(actions, AdvanceDrugPhaseAction(drug.id, next_phase, 100_000_000.0, 0.7))
        end
    end
end

# Regulatory approval actions
for drug in drugs
    for region in instances(RegulatoryRegion)
        if region ∉ drug.regulatory_approvals
            push!(actions, SeekRegulatoryApprovalAction(drug.id, region, 50_000_000.0, 0.8, 12))
        end
    end
end

# Inventory optimization actions
for facility in facilities
    for drug in drugs
        push!(actions, OptimizeInventoryAction(facility.id, drug.id, 1000.0, 100.0))
    end
end

# Termination action
push!(actions, TerminatePharmaceuticalAction())

println("   ✅ Created scenario with:")
println("      • $(length(facilities)) facilities")
println("      • $(length(drugs)) drugs in pipeline")
println("      • $(length(actions)) possible actions")
println("      • Time horizon: $(initial_state.time_horizon_months) months")
println()

# =============================================================================
# 2. Optimization Methods Configuration
# =============================================================================

println("2️⃣ Configuring Optimization Methods")
println("="^40)

# Configuration for all methods
config = (
    max_iterations = 100,
    n_trials = 50,
    temperature_schedule = :exponential,
    n_restarts = 5,
    verbose = false
)

println("   📊 Methods to compare:")
println("      1. GFlowNets (Generative Flow Networks)")
println("      2. Nonlinear Programming (Ipopt solver)")
println("      3. Hill Climbing (with $(config.n_restarts) restarts)")
println("      4. Greedy Optimization")
println("      5. Simulated Annealing")
println("      6. Random Search")
println()

# =============================================================================
# 3. Run Comprehensive Comparison
# =============================================================================

println("3️⃣ Running Comprehensive Optimization Comparison")
println("="^50)

# Store all results
all_results = Dict{String, Any}()
method_names = [
    "GFlowNets",
    "Nonlinear Programming", 
    "Hill Climbing",
    "Greedy",
    "Simulated Annealing",
    "Random Search"
]

# Run each method
for (i, method_name) in enumerate(method_names)
    println("$(i)️⃣  $method_name")
    
    start_time = time()
    
    try
        # Run the actual optimization algorithms with systematic network building
        if method_name == "GFlowNets"
            solutions = run_gflownet_optimization(initial_state, actions, config.max_iterations)
        elseif method_name == "Nonlinear Programming"
            solutions = BaselineOptimization.nonlinear_programming_optimization(initial_state, actions, 1)
        elseif method_name == "Hill Climbing"
            solutions = BaselineOptimization.hill_climbing(initial_state, actions, config.max_iterations, config.n_restarts)
        elseif method_name == "Greedy"
            solutions = BaselineOptimization.greedy_optimization(initial_state, actions, config.max_iterations)
        elseif method_name == "Simulated Annealing"
            solutions = BaselineOptimization.simulated_annealing(initial_state, actions, config.max_iterations)
        elseif method_name == "Random Search"
            solutions = BaselineOptimization.random_search(initial_state, actions, config.max_iterations)
        end
        
        elapsed_time = time() - start_time
        
        # Extract metrics
        rewards = [sol.reward for sol in solutions]
        best_reward = isempty(rewards) ? 0.0 : maximum(rewards)
        mean_reward = isempty(rewards) ? 0.0 : mean(rewards)
        
        all_results[method_name] = (
            solutions = solutions,
            rewards = rewards,
            best_reward = best_reward,
            mean_reward = mean_reward,
            n_solutions = length(solutions),
            elapsed_time = elapsed_time
        )
        
        println("   Found $(length(solutions)) solutions")
        println("   Best reward: $(round(best_reward, digits=1))")
        println("   Time: $(round(elapsed_time, digits=2))s")
        
    catch e
        println("   ❌ Failed: $e")
        all_results[method_name] = (
            solutions = [],
            rewards = [0.0],
            best_reward = 0.0,
            mean_reward = 0.0,
            n_solutions = 0,
            elapsed_time = 0.0
        )
    end
    
    println()
end

# =============================================================================
# 4. Analysis and Results
# =============================================================================

println("4️⃣ Analyzing Results")
println("="^25)

# Calculate diversity and coverage metrics
for method_name in method_names
    result = all_results[method_name]
    if result.n_solutions > 1
        # Calculate solution diversity (simplified)
        rewards = result.rewards
        diversity = length(unique(round.(rewards, digits=1))) / length(rewards)
        coverage = std(rewards) / 100.0  # Normalized coverage metric
        
        all_results[method_name] = merge(result, (
            diversity = diversity,
            coverage = coverage
        ))
    else
        all_results[method_name] = merge(result, (
            diversity = 0.0,
            coverage = 0.2  # Default minimal coverage
        ))
    end
end

# Print summary table
println("📋 OPTIMIZATION COMPARISON RESULTS")
println("="^80)
println("Method              | Best Reward | Mean Reward | Std Dev | Diversity | Coverage")
println("-"^80)

for method_name in method_names
    result = all_results[method_name]
    std_dev = result.n_solutions > 1 ? std(result.rewards) : NaN
    
    @printf("%-18s | %-10.1f | %-10.1f | %-6.1f | %-8.3f | %-7.1f\n",
            method_name, result.best_reward, result.mean_reward, 
            std_dev, result.diversity, result.coverage)
end

println()

# Find best performing method
best_method = method_names[argmax([all_results[m].best_reward for m in method_names])]
most_diverse = method_names[argmax([all_results[m].diversity for m in method_names])]
best_explorer = method_names[argmax([all_results[m].coverage for m in method_names])]

println("🎯 KEY FINDINGS:")
println("="^50)
println("🏆 Best Overall Performance: $best_method (Reward: $(round(all_results[best_method].best_reward, digits=1)))")
println("🌈 Most Diverse Solutions: $most_diverse (Diversity: $(round(all_results[most_diverse].diversity, digits=3)))")
println("🔍 Best Exploration: $best_explorer (Coverage: $(round(all_results[best_explorer].coverage, digits=1)))")
println()

# =============================================================================
# 5. Generate Comprehensive Report
# =============================================================================

if HAS_REPORTING
    println("5️⃣ Generating Comprehensive Report")
    println("="^35)
    
    try
        generate_pharmaceutical_report(all_results, initial_state, method_names)
        println("✅ Comprehensive report generated successfully!")
    catch e
        println("❌ Report generation failed: $e")
        println("📋 Results summary saved to basic output")
    end
else
    println("5️⃣ Saving Basic Results")
    println("="^25)
    
    # Save basic CSV summary
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    results_dir = "results"
    mkpath(results_dir)
    
    summary_file = joinpath(results_dir, "optimization_summary_$timestamp.txt")
    open(summary_file, "w") do f
        println(f, "Pharmaceutical Supply Chain Optimization Results")
        println(f, "Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(f, "="^60)
        
        for method_name in method_names
            result = all_results[method_name]
            println(f, "$method_name:")
            println(f, "  Best Reward: $(result.best_reward)")
            println(f, "  Mean Reward: $(result.mean_reward)")
            println(f, "  Solutions: $(result.n_solutions)")
            println(f, "  Time: $(result.elapsed_time)s")
            println(f)
        end
    end
    
    println("📄 Basic summary saved to: $summary_file")
end

println()
println("✅ PHARMACEUTICAL SUPPLY CHAIN OPTIMIZATION COMPLETE!")
println("🕐 Finished at: $(Dates.format(now(), "HH:MM:SS"))")
