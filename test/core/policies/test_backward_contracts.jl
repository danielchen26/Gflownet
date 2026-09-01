"""
Backward-policy CONTRACTS, as distinct from test_backward_policy.jl which checks that
the numbers come out right for a well-behaved grid-world state.

Three contracts, each of which used to be silently violated:

1. `validate_backward_policy_normalization` reports the TRUE parent-sum. It passed its
   arguments to `compute_backward_probability` swapped -- (policy, parent, child) where
   the signature is (policy, child, parent) -- so it summed P_B(child | parent) over a
   set of unrelated distributions. Measured before the repair on a freshly initialised
   3x3 grid-world backward policy: at GridState(2,2,false), whose true parent-sum is
   1.0000000298, it reported 2.0 with is_normalized = false; at GridState(3,2,false),
   whose true sum is 0.9999999404, the same swap happened to report exactly 1.0 and
   passed. It could fail a correct policy and certify a broken one.

2. A domain with no parent enumeration is reported as UNVERIFIABLE, not as normalised.
   `find_parent_for_action` defaults to `nothing` (core/interface.jl:874), so the empty
   parent set is the normal case for any domain that does not override it, and the
   validator has no distribution to sum. It used to return is_normalized = true for
   that case.

3. `compute_backward_probability` returns exactly 0.0 for a state that is not a parent
   of the child. The non-parent test used to sit below the unique-parent shortcut, so a
   child with exactly one parent returned 1.0 for ANY source state -- a probability of 1
   for an impossible transition, which corrupts every residual it enters.

The oracle for the parent sets here is the grid geometry itself (rung 2 of the sourcing
ladder): with the action set [MoveRight, MoveUp, Terminate], the parents of a
non-terminal (x,y) are (x-1,y) and (x,y-1) where those are on the grid, and the parent of
a terminal (x,y) is (x,y,false). The parent-sum target is 1 by the definition of a
backward policy, NOT by measurement; 1.0000000298 is quoted only to show the scale of
Float32 softmax round-off against the 1e-6 tolerance asserted below.
"""

using Test
using GFlowNet
using Random

# A domain that does not override `find_parent_for_action`, i.e. three of the five
# domains in this repo and every user-defined one. Deliberately NOT grid world.
struct NoParentHookState <: AbstractState
    value::Int
    is_terminal::Bool
end

struct StepAction <: AbstractAction
    delta::Int
end

GFlowNet.state_to_features(s::NoParentHookState)::Vector{Float32} =
    Float32[s.value, s.is_terminal ? 1.0f0 : 0.0f0]
GFlowNet.is_terminal_state(s::NoParentHookState)::Bool = s.is_terminal
GFlowNet.reward(s::NoParentHookState)::Float64 = s.is_terminal ? Float64(s.value) : 0.0
GFlowNet.is_applicable(a::StepAction, s::NoParentHookState)::Bool =
    !s.is_terminal && s.value + a.delta <= 4
GFlowNet.apply_action(a::StepAction, s::NoParentHookState)::NoParentHookState =
    NoParentHookState(s.value + a.delta, s.value + a.delta >= 4)
Base.:(==)(a::NoParentHookState, b::NoParentHookState) =
    a.value == b.value && a.is_terminal == b.is_terminal
Base.hash(s::NoParentHookState, h::UInt) = hash((s.value, s.is_terminal), h)

@testset "Backward Policy Contracts" begin
    Random.seed!(7)

    model = create_grid_world_gflownet(
        grid_size = 3,
        hidden_dim = 32,
        learning_rate = 0.01,
        include_backward = true
    )
    bw = model.backward_policy

    @testset "validator reports the true parent-sum" begin
        # (2,2) has two parents, so P_B there is a genuine softmax and the sum is the
        # only thing that can be wrong. (3,2) is the coincidence point: the swapped
        # implementation reported exactly 1.0 there, so testing (2,2) alone would leave
        # a validator that fails good policies, and testing (3,2) alone would pass a
        # broken one.
        for child in (GridState(2, 2, false), GridState(3, 2, false))
            parents = GFlowNet.backward_parent_states(child, model.all_actions)
            @test length(parents) == 2

            result = GFlowNet.validate_backward_policy_normalization(
                model, child, model.all_actions
            )

            # 1e-6, against a measured Float32 softmax round-off of 3.0e-8 on this model.
            # It rejects the swapped-argument sum of 2.0 (error 1.0) and the pre-repair
            # independent-sigmoid sums of 1.1967-1.2922 by four to seven orders of
            # magnitude, so passing is not an artefact of a loose bar.
            @test result.is_normalized
            @test result.status === :normalized
            @test isapprox(result.total_prob, 1.0; atol = 1e-6)
            @test Set(result.parent_states) == Set(parents)

            # The validator must agree with a hand-rolled sum in the documented argument
            # order, which is the quantity a GFlowNet requires to be 1.
            direct = sum(
                GFlowNet.compute_backward_probability(
                    bw, child, p, model.parameters.backward,
                    model.states.backward, model.all_actions
                ) for p in parents
            )
            @test result.total_prob ≈ direct
        end

        # Terminal states are checked rather than skipped with a fake (true, 0.0).
        terminal = GridState(3, 3, true)
        term_result = GFlowNet.validate_backward_policy_normalization(
            model, terminal, model.all_actions
        )
        @test term_result.parent_states == [GridState(3, 3, false)]
        @test term_result.status === :normalized
        @test term_result.total_prob == 1.0

        # The source state has no parents by construction, and that is KNOWN, so it is
        # its own verdict rather than either a violation or a blind pass.
        source_result = GFlowNet.validate_backward_policy_normalization(
            model, model.initial_state, model.all_actions
        )
        @test source_result.status === :source_state
        @test isempty(source_result.parent_states)

        # The three-element positional destructuring every existing caller uses still
        # works, and `is_normalized` is still a Bool.
        is_normalized, total_prob, parent_states =
            GFlowNet.validate_backward_policy_normalization(
                model, GridState(2, 2, false), model.all_actions
            )
        @test isa(is_normalized, Bool)
        @test isa(total_prob, Float64)
        @test isa(parent_states, Vector)
    end

    @testset "no parent enumeration is not success" begin
        hookless = create_gflownet(
            NoParentHookState(0, false),
            AbstractAction[StepAction(1), StepAction(2)];
            state_dim = 2,
            hidden_dim = 16,
            include_backward = true
        )
        state = NoParentHookState(2, false)

        # The premise: this domain cannot enumerate parents, so there is no set to
        # normalise over even though the state genuinely has two parents (0 and 1).
        @test isempty(GFlowNet.backward_parent_states(state, hookless.all_actions))

        result = GFlowNet.validate_backward_policy_normalization(
            hookless, state, hookless.all_actions
        )
        @test !result.is_normalized
        @test result.status === :no_parent_enumeration
        @test isnan(result.total_prob)          # no sum exists; 0.0 or 1.0 would read as one
        @test isempty(result.parent_states)

        # And it propagates: the batch validator refuses, naming the missing hook,
        # instead of averaging an unchecked state into a pass.
        trajectories = [sample_trajectory(hookless) for _ in 1:3]
        consistency = GFlowNet.validate_backward_policy_consistency(hookless, trajectories)
        @test !consistency.is_valid
        @test occursin("find_parent_for_action", consistency.message)
        @test consistency.stats.n_unverifiable > 0

        # The monitor reports NaN rather than a zero error it never measured.
        metrics = GFlowNet.monitor_backward_policy_learning(
            hookless, [state]; verbose = false
        )
        @test isnan(metrics["mean_normalization_error"])
        @test metrics["states_unverifiable"] == 1

        # Grid world, by contrast, is verifiable end to end.
        grid_consistency = GFlowNet.validate_backward_policy_consistency(
            model, [sample_trajectory(model) for _ in 1:3]
        )
        @test grid_consistency.stats.n_unverifiable == 0
        @test grid_consistency.is_valid
    end

    @testset "P_B is zero for a non-parent" begin
        # (2,1) is reachable only from (1,1) under [MoveRight, MoveUp, Terminate]: this
        # is the single-parent case where the non-parent answer used to be 1.0.
        child = GridState(2, 1, false)
        @test GFlowNet.backward_parent_states(child, model.all_actions) ==
              [GridState(1, 1, false)]

        for non_parent in (GridState(3, 3, false), GridState(2, 2, false),
                           GridState(1, 2, false), GridState(2, 1, false))
            @test GFlowNet.compute_backward_probability(
                bw, child, non_parent, model.parameters.backward,
                model.states.backward, model.all_actions
            ) == 0.0
        end

        # The one real parent still carries all the mass, so the distribution is intact.
        @test GFlowNet.compute_backward_probability(
            bw, child, GridState(1, 1, false), model.parameters.backward,
            model.states.backward, model.all_actions
        ) == 1.0

        # A terminal child has exactly one parent too, and the same contract holds.
        @test GFlowNet.compute_backward_probability(
            bw, GridState(3, 3, true), GridState(2, 3, false), model.parameters.backward,
            model.states.backward, model.all_actions
        ) == 0.0

        # The model-level wrapper inherits the contract.
        @test backward_transition_probability(model, child, GridState(3, 3, false)) == 0.0
        @test backward_transition_probability(model, child, GridState(1, 1, false)) == 1.0
    end
end
