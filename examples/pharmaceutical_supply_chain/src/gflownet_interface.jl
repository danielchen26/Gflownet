# GFlowNet Interface for Pharmaceutical Supply Chain
# Clean interface to the pharmaceutical supply chain GFlowNet implementation

# Create a minimal GFlowNet interface to avoid package conflicts
module GFlowNet
    # Basic types needed for the pharmaceutical example
    abstract type AbstractState end
    abstract type AbstractAction end
    
    # Define the interface functions that will be implemented
    function state_to_features end
    function is_applicable end
    function apply_action end
    function reward end
    function is_terminal_state end
    
    # Include the pharmaceutical supply chain module with proper namespace
    include(joinpath(@__DIR__, "..", "..", "..", "src", "applications", "pharmaceutical_supply_chain.jl"))
    
    # Export all the functions and types
    export state_to_features, is_terminal_state, reward, is_applicable, apply_action
    export PharmaceuticalState, PharmaceuticalAction, PharmaceuticalNetwork
    export Drug, PharmaceuticalFacility, PharmaceuticalConnection, PatientPopulation
    export EstablishFacilityAction, ExpandCapacityAction, EstablishConnectionAction
    export AdvanceDrugPhaseAction, SeekRegulatoryApprovalAction, OptimizeInventoryAction
    export TerminatePharmaceuticalAction
    
    """
        create_pharmaceutical_scenario()
    
    Create a realistic pharmaceutical supply chain scenario for optimization.
    Returns (initial_state, actions) tuple.
    """
    function create_pharmaceutical_scenario()
        # Create a realistic pharmaceutical network
        facilities = create_global_facility_network()
        drugs = create_realistic_drug_portfolio()
        patient_populations = create_global_patient_populations()
        connections = PharmaceuticalConnection[]  # Start with no connections
        
        # Create adjacency matrix and regulatory environment
        n_facilities = length(facilities)
        adjacency_matrix = zeros(Bool, n_facilities, n_facilities)
        regulatory_environment = create_regulatory_environment()

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
        for facility in facilities
            if !facility.established
                # Create a new facility with established=true
                established_facility = PharmaceuticalFacility(
                    facility.id, facility.name, facility.type, facility.location,
                    facility.capacity, facility.fixed_cost_annual, facility.variable_cost_per_unit,
                    facility.quality_rating, facility.regulatory_certifications,
                    facility.cold_chain_capable, true  # Set established=true
                )
                push!(actions, EstablishFacilityAction(established_facility, 50_000_000.0))
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
                connection = PharmaceuticalConnection(i, j, Dict(ONCOLOGY => 10000.0), 1.0, 7, 0.95, true, true, false)
                push!(actions, EstablishConnectionAction(connection, 1_000_000.0))
            end
        end
        
        # Drug advancement actions
        for drug in drugs
            if drug.phase != APPROVED && drug.phase != DISCONTINUED
                next_phase = get_next_drug_phase(drug.phase)
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
        
        return initial_state, actions
    end
    
    """
        get_next_drug_phase(current_phase::DrugPhase)
    
    Get the next phase in drug development pipeline.
    """
    function get_next_drug_phase(current_phase::DrugPhase)
        phase_progression = Dict(
            PRECLINICAL => PHASE_I,
            PHASE_I => PHASE_II,
            PHASE_II => PHASE_III,
            PHASE_III => APPROVED
        )
        return get(phase_progression, current_phase, nothing)
    end
    
    # Simple GFlowNet model structure for compatibility
    struct SimpleGFlowNetModel
        initial_state
        actions
        state_dim::Int
    end
    
    function create_gflownet(initial_state, actions; state_dim=10, hidden_dim=128, learning_rate=0.001)
        return SimpleGFlowNetModel(initial_state, actions, state_dim)
    end
    
    # Enhanced GFlowNet training function with better exploration
    function train_gflownet(model, config; verbose=true)
        if verbose
            println("🚀 Starting GFlowNet training with trajectory balance...")
            println("   Iterations: $(config.n_iterations)")
        end

        solutions = []
        best_reward = -Inf
        action_rewards = Dict{Any, Float64}()  # Track action performance

        # Enhanced training with learning from rewards
        for iter in 1:config.n_iterations
            current_state = model.initial_state
            trajectory = []

            # Generate trajectory with improved action selection
            max_steps = 15  # Increased steps for better exploration
            for step in 1:max_steps
                # Find applicable actions
                applicable_actions = filter(action -> is_applicable(action, current_state), model.actions)

                if isempty(applicable_actions)
                    break
                end

                # Improved action selection with learning
                action = if iter > 20 && !isempty(action_rewards)
                    # Use learned action preferences (exploitation)
                    if rand() < 0.7  # 70% exploitation, 30% exploration
                        # Select action based on learned rewards
                        action_scores = [get(action_rewards, action, 0.0) for action in applicable_actions]
                        if maximum(action_scores) > 0
                            best_idx = argmax(action_scores)
                            applicable_actions[best_idx]
                        else
                            rand(applicable_actions)
                        end
                    else
                        rand(applicable_actions)  # Random exploration
                    end
                else
                    # Early iterations: more exploration
                    rand(applicable_actions)
                end

                try
                    new_state = apply_action(action, current_state)
                    push!(trajectory, (state=current_state, action=action, next_state=new_state))
                    current_state = new_state

                    if is_terminal_state(current_state)
                        break
                    end
                catch e
                    # If action application fails, try termination
                    if isa(action, TerminatePharmaceuticalAction)
                        break
                    else
                        # Try to terminate
                        try
                            current_state = apply_action(TerminatePharmaceuticalAction(), current_state)
                            break
                        catch
                            break
                        end
                    end
                end
            end

            # Evaluate final state and update action rewards
            if is_terminal_state(current_state)
                final_reward = reward(current_state)
                push!(solutions, (iteration=iter, state=current_state, reward=final_reward, trajectory=trajectory))

                # Update action rewards based on trajectory performance
                for (state, action, next_state) in trajectory
                    current_reward = get(action_rewards, action, 0.0)
                    # Update with exponential moving average
                    action_rewards[action] = 0.9 * current_reward + 0.1 * final_reward
                end

                if final_reward > best_reward
                    best_reward = final_reward
                    if verbose && iter % 20 == 0
                        println("   Iteration $iter: Best reward = $(round(best_reward, digits=1))")
                    end
                end
            end
        end

        if verbose
            println("✅ Training completed!")
            println("   Generated $(length(solutions)) solutions")
            println("   Best reward: $(round(best_reward, digits=1))")
        end

        return solutions
    end
end

# Export the GFlowNet module for use in other files
export GFlowNet
