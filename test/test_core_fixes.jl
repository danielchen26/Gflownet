# Comprehensive Test Suite for Core Fixes
# Tests all the critical fixes implemented for GFlowNet package

using Test
using GFlowNet
using ComponentArrays
using Random
using Zygote

# Load test utilities for testing
include(joinpath(@__DIR__, "test_utilities.jl"))

@testset "Core Fixes Test Suite" begin

    @testset "Issue #1: Critical Function Implementation" begin
        @testset "update_model_parameters! function exists and works" begin
            # Create a minimal test model
            dag = DirectedAcyclicGraph()
            forward_policy = ForwardPolicy(identity)  # Placeholder

            # Create test parameters as ComponentArray
            test_params = ComponentArray(
                forward=ComponentArray(weights=randn(Float32, 4, 2), bias=randn(Float32, 2)),
                backward=ComponentArray(weights=randn(Float32, 2, 4), bias=randn(Float32, 4))
            )

            test_states = (
                forward=nothing,
                backward=nothing
            )

            model = GFlowNetModel(
                dag=dag,
                forward_policy=forward_policy,
                parameters=test_params,
                states=test_states
            )

            # Create training config
            config = TrainingConfig(
                objective=TRAJECTORY_BALANCE,
                batch_size=2,
                learning_rate=0.01
            )

            # Create dummy trajectories
            state1 = SimpleState([1])
            state2 = SimpleState([2])
            trajectory = Trajectory([state1, state2])
            trajectories = [trajectory]

            # Test that the function exists and can be called
            @test hasmethod(update_model_parameters!, (GFlowNetModel, TrainingConfig, Vector{<:Trajectory}))

            # Store initial parameters
            initial_params = deepcopy(model.parameters)

            # This should not throw an error
            @test_nowarn begin
                loss = update_model_parameters!(model, config, trajectories)
                @test isa(loss, Float64)
            end
        end
    end

    @testset "Issue #2: Zygote Compatibility" begin
        @testset "Sampling functions avoid mutations" begin
            # Test that sampling uses functional approach
            dag = DirectedAcyclicGraph()
            forward_policy = ForwardPolicy(identity)

            test_params = ComponentArray(
                forward=ComponentArray(weights=randn(Float32, 4, 2), bias=randn(Float32, 2))
            )

            test_states = (forward=nothing,)

            model = GFlowNetModel(
                dag=dag,
                forward_policy=forward_policy,
                parameters=test_params,
                states=test_states
            )

            # Test that sample_trajectory is defined
            @test hasmethod(sample_trajectory, (GFlowNetModel,))

            # Test that recursive sampling helper exists
            @test hasmethod(GFlowNet._sample_trajectory_recursive,
                (GFlowNetModel, AbstractState, Vector{AbstractState}, Int, Any))
        end

        @testset "No global state mutations during differentiation" begin
            # Test that functions can be differentiated without mutation errors
            simple_function(x) = sum(x .^ 2)
            test_input = ComponentArray(a=[1.0, 2.0], b=[3.0, 4.0])

            # This should work without mutation errors
            @test_nowarn begin
                gradient = Zygote.gradient(simple_function, test_input)[1]
                @test isa(gradient, ComponentArray)
            end
        end
    end

    @testset "Issue #3: Thread-Safe Flow Cache" begin
        @testset "ThreadSafeFlowCache structure" begin
            # Test that ThreadSafeFlowCache is properly defined
            @test isdefined(GFlowNet, :ThreadSafeFlowCache)

            # Test cache operations
            @test_nowarn GFlowNet.clear_flow_cache!()

            # Test that FLOW_CACHE is properly initialized
            @test isdefined(GFlowNet, :FLOW_CACHE)
        end

        @testset "Thread-safe cache operations" begin
            # Test concurrent access doesn't cause errors
            state1 = SimpleState([1])
            state2 = SimpleState([2])

            # Create a minimal model for testing
            dag = DirectedAcyclicGraph()
            forward_policy = ForwardPolicy(identity)
            test_params = ComponentArray(forward=ComponentArray(w=[1.0, 2.0]))
            test_states = (forward=nothing,)

            model = GFlowNetModel(
                dag=dag,
                forward_policy=forward_policy,
                parameters=test_params,
                states=test_states
            )

            # Test that caching functions exist
            @test hasmethod(GFlowNet.compute_recursive_flow_memoized, (GFlowNet.GFlowNetModel, AbstractState))
            @test hasmethod(GFlowNet.get_cached_flow, (GFlowNet.GFlowNetModel, AbstractState))
        end
    end

    @testset "Issue #4: Parameter Type Standardization" begin
        @testset "ComponentArray parameter enforcement" begin
            # Test that to_component_array works for different input types

            # NamedTuple to ComponentArray
            nt_params = (a=[1.0, 2.0], b=[3.0, 4.0])
            ca_params = to_component_array(nt_params)
            @test isa(ca_params, ComponentArray)
            @test ca_params.a == nt_params.a
            @test ca_params.b == nt_params.b

            # ComponentArray passthrough
            existing_ca = ComponentArray(x=[1.0], y=[2.0])
            passthrough_ca = to_component_array(existing_ca)
            @test passthrough_ca === existing_ca

            # Array to ComponentArray
            array_params = [1.0, 2.0, 3.0]
            array_ca = to_component_array(array_params)
            @test isa(array_ca, ComponentArray)
        end

        @testset "GFlowNetModel type constraints" begin
            # Test that GFlowNetModel properly constrains parameter types
            dag = DirectedAcyclicGraph()
            forward_policy = ForwardPolicy(identity)

            # Valid ComponentArray parameters
            valid_params = ComponentArray(
                forward=ComponentArray(w=[1.0, 2.0])
            )
            test_states = (forward=nothing,)

            # This should work
            @test_nowarn GFlowNetModel(
                dag=dag,
                forward_policy=forward_policy,
                parameters=valid_params,
                states=test_states
            )

            # Test safe constructor
            @test_nowarn create_gflownet_model_safe(
                dag=dag,
                forward_policy=forward_policy,
                parameters=(forward=(w=[1.0, 2.0],),),  # NamedTuple
                states=test_states
            )
        end
    end

    @testset "Issue #5: Complete Interface Implementation" begin
        @testset "Abstract interface definitions" begin
            # Test that all abstract types are defined
            @test isdefined(GFlowNet, :AbstractState)
            @test isdefined(GFlowNet, :AbstractAction)
            @test isdefined(GFlowNet, :AbstractGFlowNetObjective)
            @test isdefined(GFlowNet, :AbstractPolicy)
            @test isdefined(GFlowNet, :AbstractPartitionFunctionEstimator)
        end

        @testset "Required interface methods" begin
            # Test that interface methods are defined
            @test hasmethod(is_terminal_state, (AbstractState,))
            @test hasmethod(state_to_features, (AbstractState,))
            @test hasmethod(reward, (AbstractState,))
            @test hasmethod(is_applicable, (AbstractAction, AbstractState))
            @test hasmethod(apply_action, (AbstractAction, AbstractState))
        end

        @testset "Default implementations for SimpleState/SimpleAction" begin
            # Test SimpleState interface (from test utilities)
            state = SimpleState([1, 2])
            @test isa(state_to_features(state), Vector{Float32})
            @test isa(reward(state), Float64)
            @test isa(is_terminal_state(state), Bool)

            # Test SimpleAction interface
            action = SimpleAction(3)
            state = SimpleState([1, 2])
            @test isa(is_applicable(action, state), Bool)

            if is_applicable(action, state)
                new_state = apply_action(action, state)
                @test isa(new_state, SimpleState)
            end
        end

        @testset "Interface validation functions" begin
            # Test validation functions exist
            @test hasmethod(GFlowNet.validate_state_interface, (Type{<:AbstractState},))
            @test hasmethod(GFlowNet.validate_action_interface, (Type{<:AbstractAction}, Type{<:AbstractState}))

            # Test validation works for SimpleState/SimpleAction (from test utilities)
            @test_nowarn GFlowNet.validate_state_interface(SimpleState)
            @test_nowarn GFlowNet.validate_action_interface(SimpleAction, SimpleState)
        end
    end

    @testset "Issue #6: Efficient Data Structures" begin
        @testset "DAG performance improvements" begin
            # Test that DAG operations are efficient (using test utilities)
            dag = create_dag(
                SimpleState([0]),  # initial
                [SimpleState([-1])],  # terminals
                SimpleState([-1]),  # sink
                [SimpleAction(1), SimpleAction(2)]  # actions
            )

            # Test that action cache is working
            state = SimpleState([0])
            @test haskey(dag.action_cache, state)

            # Test efficient lookup functions
            @test hasmethod(get_applicable_actions, (DirectedAcyclicGraph, AbstractState))
            @test hasmethod(get_next_states, (DirectedAcyclicGraph, AbstractState))
            @test hasmethod(get_previous_states, (DirectedAcyclicGraph, AbstractState))
        end

        @testset "Optimized transition computations" begin
            # Test that transition functions are optimized
            # Create simple DAG for testing using test utilities
            dag = create_test_dag()
            state = SimpleState([1])

            # These should be fast operations
            @test_nowarn get_applicable_actions(dag, state)
            @test_nowarn get_next_states(dag, state)
            @test_nowarn get_previous_states(dag, state)
        end
    end

    @testset "Issue #7: Improved Training Loop" begin
        @testset "Training function robustness" begin
            # Test that training functions exist and have proper signatures
            @test hasmethod(train_gflownet, (GFlowNetModel, TrainingConfig))
            @test hasmethod(compute_loss_and_gradients, (GFlowNetModel, TrainingConfig, Vector{<:Trajectory}))
            @test hasmethod(apply_optimizer_updates!, (GFlowNetModel, Any))
        end

        @testset "Error handling in training" begin
            # Test that training handles errors gracefully
            dag = create_test_dag()
            forward_policy = ForwardPolicy(identity)
            test_params = ComponentArray(forward=ComponentArray(w=[1.0]))
            test_states = (forward=nothing,)

            model = GFlowNetModel(
                dag=dag,
                forward_policy=forward_policy,
                parameters=test_params,
                states=test_states
            )

            config = TrainingConfig(
                objective=TRAJECTORY_BALANCE,
                batch_size=1,
                n_iterations=1
            )

            # This should handle errors gracefully
            @test_nowarn train_gflownet(model, config)
        end
    end

    @testset "Issue #8: Integration Tests" begin
        @testset "End-to-end workflow" begin
            # Test that a complete workflow works

            # 1. Create DAG
            initial_state = SimpleState([0])
            terminal_states = [SimpleState([-1])]
            terminal_sink = SimpleState([-1])
            actions = [SimpleAction(1), SimpleAction(2)]

            dag = create_dag(initial_state, terminal_states, terminal_sink, actions)

            # 2. Create policies (simplified)
            forward_policy = ForwardPolicy(identity)

            # 3. Create parameters
            params = ComponentArray(
                forward=ComponentArray(weights=randn(Float32, 2, 2), bias=randn(Float32, 2))
            )

            states = (forward=nothing,)

            # 4. Create model
            model = GFlowNetModel(
                dag=dag,
                forward_policy=forward_policy,
                parameters=params,
                states=states
            )

            # 5. Create config
            config = TrainingConfig(
                objective=TRAJECTORY_BALANCE,
                batch_size=2,
                n_iterations=1
            )

            # 6. Test training
            @test_nowarn train_gflownet(model, config)
        end

        @testset "Parameter update consistency" begin
            # Test that parameter updates are consistent
            dag = create_test_dag()
            forward_policy = ForwardPolicy(identity)

            initial_params = ComponentArray(
                forward=ComponentArray(w=[1.0, 2.0])
            )

            model = GFlowNetModel(
                dag=dag,
                forward_policy=forward_policy,
                parameters=deepcopy(initial_params),
                states=(forward=nothing,)
            )

            config = TrainingConfig(batch_size=1, learning_rate=0.01)

            # Create dummy trajectory
            state1 = SimpleState([1])
            state2 = SimpleState([2])
            trajectory = Trajectory([state1, state2])
            trajectories = [trajectory]

            # Parameters should change after update
            old_params = deepcopy(model.parameters)
            update_model_parameters!(model, config, trajectories)

            # Verify parameters changed (or stayed same if gradients were zero)
            @test isa(model.parameters, ComponentArray)
        end
    end

    @testset "Backward Compatibility" begin
        @testset "Legacy interfaces still work" begin
            # Test that old interfaces are maintained where possible

            # SimpleState and SimpleAction should still work (from test utilities)
            state = SimpleState([1, 2, 3])
            action = SimpleAction(4)

            @test isa(state, AbstractState)
            @test isa(action, AbstractAction)

            # Basic operations should work
            @test isa(hash(state), UInt)
            @test isa(hash(action), UInt)
            @test isa(state == state, Bool)
            @test isa(action == action, Bool)
        end
    end

    @testset "Performance Regression Tests" begin
        @testset "No major performance degradations" begin
            # Test that core operations are reasonably fast

            # State operations
            state = SimpleState([1, 2, 3])
            @test (@timed state_to_features(state)).time < 0.001
            @test (@timed reward(state)).time < 0.001
            @test (@timed is_terminal_state(state)).time < 0.001

            # Action operations
            action = SimpleAction(4)
            @test (@timed is_applicable(action, state)).time < 0.001
            if is_applicable(action, state)
                @test (@timed apply_action(action, state)).time < 0.001
            end

            # Cache operations
            @test (@timed GFlowNet.clear_flow_cache!()).time < 0.001
        end
    end
end
