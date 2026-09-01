"""
Causal-discovery parent enumeration, and the partition function it makes reachable.

WHY THIS FILE EXISTS

The trajectory-balance repair -- subtracting `sum log P_B` with P_B uniform over parents when
no backward policy exists -- was INERT in this domain, and nothing noticed.

The repaired losses read

    n_parents = length(backward_parent_states(child, model.all_actions))
    n_parents > 1 && (log_backward_sum += -log(n_parents))

and `backward_parent_states` enumerates parents by calling `find_parent_for_action`, whose
DEFAULT (src/core/interface.jl) returns `nothing`. Only grid_world and molecular_generation
defined overrides, so for causal discovery the parent set came back EMPTY, the guard never
fired, log P_B stayed identically 0, and the loss was the pre-repair one.

MEASURED, before the parent methods existed, 1000 iterations at batch 32, lr 0.005,
z multiplier 10, seed 20260828:

    Z_true   = sum over terminals of R(x)        = 13.667470
    Z_biased = sum over terminals of n(x) R(x)   = 30.864859
    learned Z                                    = 30.881821

0.055% from the BIASED value and 126% from the true one -- the grid's pre-repair signature
reproduced in a second domain AFTER the repair. The sampled terminal law was 0.0271 in total
variation from the biased law and 0.2727 from R(x)/Z.

After the parent methods:

    seed 20260828  learned Z = 13.6435   0.18% from true, 55.80% from biased
    seed 7         learned Z = 13.6679   0.00% from true, 55.72% from biased

WHAT THIS TEST PINS

Both directions. Closeness to Z_true alone would pass a model that learned nothing if the
tolerance were loose, and distance from Z_biased alone is weak. The pair is what distinguishes
"converged to the right value" from "converged to the wrong one" -- and it is exactly the
assertion whose absence let the inert repair look fine.
"""

using Test
using GFlowNet
using Random
using Printf

const _CD_NODES = ["A", "B", "C"]

_cd_state(edges) = begin
    adj = falses(3, 3)
    for (i, j) in edges
        adj[i, j] = true
    end
    GFlowNet.DAGState(adj, _CD_NODES, false)
end

@testset "Causal discovery parent enumeration" begin
    actions = GFlowNet.create_dag_actions(3)
    s0 = _cd_state(Tuple{Int,Int}[])

    @testset "a DAG with k edges has k parents" begin
        # The defining structure: edges commute, so a state with k edges is reached by
        # removing any one of them. This is what makes P_B == 1 wrong here -- a k-edge state
        # has k parents, not one -- and it is the reason the domain needed the override at all.
        @test length(GFlowNet.backward_parent_states(s0, actions)) == 0
        @test length(GFlowNet.backward_parent_states(_cd_state([(1, 2)]), actions)) == 1
        @test length(GFlowNet.backward_parent_states(_cd_state([(1, 2), (2, 3)]), actions)) == 2
        @test length(GFlowNet.backward_parent_states(_cd_state([(1, 2), (2, 3), (1, 3)]),
                                                     actions)) == 3
    end

    # Enumerated oracle. Built here rather than trusted from the implementation: reachable
    # states by walk, distinct action sequences per terminal, then both candidate partition
    # functions. Mirrors what test/theory/enumerate.jl does for the grid.
    paths = Dict{GFlowNet.DAGState,Int}()
    paths[s0] = 1
    function _walk(s, depth)
        depth > 8 && return
        for a in actions
            GFlowNet.is_applicable(a, s) || continue
            c = GFlowNet.apply_action(a, s)
            c == s && continue
            paths[c] = get(paths, c, 0) + 1
            GFlowNet.is_terminal_state(c) || _walk(c, depth + 1)
        end
    end
    _walk(s0, 0)

    terminals = [s for s in keys(paths) if GFlowNet.is_terminal_state(s)]
    Z_true = sum(GFlowNet.reward(s) for s in terminals)
    Z_biased = sum(paths[s] * GFlowNet.reward(s) for s in terminals)

    @testset "the two candidate partition functions" begin
        # 50 reachable states, 25 terminal. Pinned so a domain change that alters the state
        # space announces itself here rather than as a mysterious shift in the Z assertions.
        @test length(paths) == 50
        @test length(terminals) == 25
        @test Z_true ≈ 13.667470 atol = 1e-5
        @test Z_biased ≈ 30.864859 atol = 1e-5
        # They must be far apart, otherwise the assertion below cannot distinguish them.
        @test Z_biased / Z_true > 2.0
    end

    @testset "TB converges to sum R(x), not the path-weighted sum" begin
        # 1000 iterations is the measured budget: seeds 20260828 and 7 give 13.6435 and
        # 13.6679, i.e. 0.18% and 0.00% from Z_true. Two seeds because one is not evidence --
        # the sampling-accuracy work in this repo already found an objective that converged on
        # 1 of 3 seeds and looked fine on the one that was tried first.
        for seed in (20260828, 7)
            Random.seed!(seed)
            model = GFlowNet.create_gflownet(
                s0, actions;
                state_dim = 14,          # matches state_to_features for a 3-node DAG
                hidden_dim = 64,
                partition_function_method = LEARNABLE_ESTIMATION,
            )
            config = TrainingConfig(
                objective = TRAJECTORY_BALANCE,
                partition_function_method = LEARNABLE_ESTIMATION,
                n_iterations = 1000, batch_size = 32, learning_rate = 0.005,
                z_learning_rate_multiplier = 10.0,
                entropy_weight = 0.0,    # the 0.01 default shifts the optimum by ~1e-6
            )
            for _ in 1:1000
                GFlowNet.train_step!(model,
                    [GFlowNet.sample_trajectory(model) for _ in 1:32], config)
            end

            Z = exp(model.parameters.log_Z)
            err_true = abs(Z - Z_true) / Z_true
            err_biased = abs(Z - Z_biased) / Z_biased

            # 0.03 against a measured worst of 0.0018. Loose enough for optimiser noise,
            # tight enough that the biased value (55.7% away) cannot pass.
            @test err_true < 0.03
            # And far from the biased fixed point. Before the parent methods the learned Z sat
            # 0.00055 from it.
            @test err_biased > 0.3
        end
    end
end
