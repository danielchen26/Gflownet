"""
Tree-structure invariant for the fragment-based molecular DAG.

WHY THIS FILE EXISTS

The corrected Trajectory Balance objective is

    (log Z + sum log P_F(tau) - log R(x) - sum log P_B(tau))^2

and at its optimum, summing over every trajectory reaching x,

    Z * p(x) = R(x) * sum_{tau -> x} P_B(tau | x).

So the sampler gives p(x) proportional to R(x) if and only if P_B is a genuine
distribution over parents, making that sum equal 1. If P_B is instead a constant 1
per edge, the sum becomes n(x), the number of distinct paths reaching x, and the
sampler converges to p(x) proportional to n(x) R(x) -- the path-count bias that was
measured on the 3x3 grid as Z = 78 instead of 19 with per-terminal sampling ratios
from 0.2436 to 1.4615.

`find_parent_for_action` returns `nothing` for every MolState
(molecular_generation.jl:568), because a fragment join cannot be cheaply reversed.
`compute_backward_probability` therefore falls through to its empty-parent-set case
and returns 1.0.

THAT IS CORRECT HERE, AND ONLY BECAUSE THE DAG IS A TREE.

`apply_action(::FragmentAction, ::MolState)` always joins at
`state.attachment_points[1]` -- a fixed slot, never a choice. The fragment sequence
therefore determines both the SMILES and the remaining attachment list uniquely, so
every state has exactly one parent and n(x) = 1.

This property is load-bearing and was completely untested. Making the join site a
choice -- an obvious feature request, since real fragment GFlowNets do pick the
attachment point -- turns the DAG into a lattice and silently reintroduces the bias:
training would still run, losses would still fall, and the sampled distribution would
be wrong by a factor of n(x). These tests make that failure loud.

MEASURED WHEN WRITTEN: 8 fragments, depth <= 4, 65 reachable states, 0 states
reached by more than one path, 0 SMILES aliased across distinct states.
"""

using Test

include(joinpath(@__DIR__, "test_setup.jl"))

@testset "Fragment DAG is a tree (P_B = 1 is exact)" begin
    # Deliberately includes the symmetric linkers, which are the fragments most
    # likely to admit commuting build orders: attaching A then B to a 1,3- or 1,4-
    # disubstituted ring is, chemically, the same molecule as B then A.
    fragment_indices = [1, 2, 5, 41, 42, 43, 44, 45]
    acts = AbstractAction[FRAGMENT_LIBRARY[i] for i in fragment_indices]

    @test length(acts) == 8

    # Without RDKit every join fails, so the enumeration would explore nothing and
    # the invariant would pass VACUOUSLY. Skip loudly instead of banking a
    # meaningless green -- the same reason test/fixtures/molecular.jl exists. Set
    # GFLOWNET_TEST_RDKIT=true to actually check this.
    if !RDKIT_AVAILABLE
        @warn "Fragment DAG tree invariant NOT CHECKED -- no RDKit" reason = rdkit_reason()
    else

    s0 = MolState("", Int[], Int[], 0, false, zeros(Float32, FINGERPRINT_DIM))
    max_depth = 4

    # state -> every action sequence that reaches it
    paths = Dict{MolState, Vector{Vector{Int}}}()

    function walk!(s, seq, depth)
        depth > 0 && push!(get!(paths, s, Vector{Int}[]), copy(seq))
        depth >= max_depth && return
        for (i, a) in enumerate(acts)
            GFlowNet.is_applicable(a, s) || continue
            child = try
                GFlowNet.apply_action(a, s)
            catch
                nothing
            end
            child === nothing && continue

            # An action that does not change the state is NOT an edge, so it must not
            # be walked. `apply_action` returns the state UNCHANGED when a join fails
            # (molecular_generation.jl:545 and :552), so without this filter every
            # failed join looks like a self-loop and every state becomes reachable by
            # many "paths" -- which would fail the invariant below for a reason that
            # has nothing to do with DAG shape. `sample_trajectory` applies the same
            # rule, so this mirrors what training actually traverses.
            child == s && continue

            push!(seq, i)
            walk!(child, seq, depth + 1)
            pop!(seq)
        end
    end
    walk!(s0, Int[], 0)

    @test !isempty(paths)

    multi_path = [(s, ps) for (s, ps) in paths if length(ps) > 1]

    # THE INVARIANT. Every reachable state has exactly one parent, so n(x) = 1 and
    # the constant P_B = 1 returned for MolState is the exact unique-parent value,
    # not an approximation.
    @test isempty(multi_path)

    if !isempty(multi_path)
        # Actionable rather than mysterious: name a colliding state and two paths.
        s, ps = first(multi_path)
        @info """
        FRAGMENT DAG IS NO LONGER A TREE.

        State $(s.smiles) (n_fragments = $(s.n_fragments)) is reachable by
        $(length(ps)) distinct action sequences, e.g. $(ps[1]) and $(ps[2]).

        P_B = 1 per edge is now WRONG: sum over parents of P_B is $(length(ps)),
        not 1, so Trajectory Balance will converge to p(x) proportional to
        n(x) R(x) instead of R(x). Implement find_parent_for_action for MolState,
        or return 1/(number of parents) from compute_backward_probability.
        """
        @info "colliding paths" state_count = length(paths) collisions = length(multi_path)
    end

    # A distinct failure mode with the same consequence: if two states that ARE
    # different DAG positions collapse to one under ==/hash, they merge into a
    # single multi-parent node. MolState equality deliberately excludes
    # n_fragments, so this is worth pinning separately.
    by_smiles = Dict{String, Set{MolState}}()
    for s in keys(paths)
        push!(get!(by_smiles, s.smiles, Set{MolState}()), s)
    end
    aliased = [(smi, states) for (smi, states) in by_smiles if length(states) > 1]
    @test isempty(aliased)
    end  # RDKIT_AVAILABLE
end

@testset "P_B is exactly 1 for every molecular transition" begin
    # The code path, not just the theory: whatever compute_backward_probability
    # returns for a MolState edge must be exactly 1, since the unique parent carries
    # all the backward mass. Anything else biases TB.
    acts = AbstractAction[FRAGMENT_LIBRARY[1], FRAGMENT_LIBRARY[2], FRAGMENT_LIBRARY[41]]
    all_actions = AbstractAction[acts..., TerminateMolAction()]

    s0 = MolState("", Int[], Int[], 0, false, zeros(Float32, FINGERPRINT_DIM))

    # Walk a few transitions and check the parent enumeration agrees with "no
    # parents found", which is what makes the 1.0 fallback fire.
    if !RDKIT_AVAILABLE
        @warn "Molecular P_B check NOT RUN -- no RDKit" reason = rdkit_reason()
    else
        s = s0
        checked = 0
        for a in acts
            GFlowNet.is_applicable(a, s) || continue
            child = try
                GFlowNet.apply_action(a, s)
            catch
                nothing
            end
            child === nothing && continue
            child == s && continue      # not an edge; see the walk in the testset above

            parents = GFlowNet.backward_parent_states(child, all_actions)
            # Documented consequence of find_parent_for_action returning nothing.
            @test isempty(parents)

            s = child
            checked += 1
        end
        @test checked >= 1
    end
end
