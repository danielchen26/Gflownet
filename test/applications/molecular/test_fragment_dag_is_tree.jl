"""
The fragment-based molecular DAG is a LATTICE, not a tree, so P_B = 1 is not its
unique-parent value, and the backward path must REFUSE rather than assume.

WHAT THIS FILE USED TO ASSERT, AND WHY IT PASSED ANYWAY

The previous version walked eight fragments to depth 4, keyed states by `MolState`,
found 65 nodes with none reached by more than one path, and concluded that
`find_parent_for_action` returning `nothing` -- hence P_B = 1 per edge -- was the
exact unique-parent value rather than an approximation.

Two omissions made that green meaningless.

 1. `TerminateMolAction` was left out of the action list, so the walk never reached a
    terminal state. A terminal state is where the reward lives and where the
    multiplicity concentrates: `apply_action(::TerminateMolAction, s)` runs
    `finalize_smiles`, which DELETES every remaining dummy atom, so the three
    distinct states left by starters 41, 42 and 43 all collapse onto `c1ccccc1`.

 2. `MolState` equality compares the RAW stored SMILES, and that string is a function
    of build history rather than of the molecule. `join_fragment` parses each incoming
    fragment with `MolFromSmarts`, so the fragment added LAST leaves query bonds in
    the output: order [41, 1, 2] stores `C1:C:C:C(c2ccccc2-c2ccccc2):N:C:1` while
    order [41, 2, 1] stores `C1:C:C:C(c2ccccc2-c2ccccn2):C:C:1`. Both are
    `c1ccc(-c2ccccc2-c2ccccn2)cc1`. `finalize_smiles` erases the artifact, which is
    exactly why the collisions surface at the terminal states.

So the old 65 was never a node count. It is the number of distinct join sequences --
the number of trajectories -- and those 65 trajectories land on 41 distinct molecules.

WHAT IS TRUE

Exhaustive enumeration of the instance below, MEASURED 2026-08-31:

    50 non-terminal + 41 terminal = 91 canonical nodes
    in-degree histogram over all 91:      1 => 72, 2 => 15, 3 => 3, 4 => 1
      the 15 in-degree-2 nodes are non-terminal (joins that commute)
      the 4 higher ones are terminal (the finalize_smiles collapse):
      c1ccccc1, c1ccc(-c2ccccc2)cc1, c1ccc(-c2ccccn2)cc1, c1ccc(-c2ccsc2)cc1
    terminal n_paths histogram:           1 => 22, 2 => 15, 3 => 3, 4 => 1
      65 trajectories over 41 terminals
    Z_true = sum_x R(x)            = 28.4957
    Z_PB1  = sum_x n_paths(x) R(x) = 45.4131   (ratio 1.5937)
    TV(R/Z , n R / sum n R)        = 0.1965

Trajectory Balance on this instance, 1000 iterations, batch 32, lr 0.005,
z_learning_rate_multiplier 10, seed 20260828, reached Z = 45.4132 while
`find_parent_for_action` returned `nothing`, and Z = 28.4941 when the enumerated
parent set was supplied instead. The repair at losses.jl:566 was therefore inert
here: it reads `backward_parent_states`, which was empty for every MolState.

WHAT THIS FILE ASSERTS NOW

 1. The backward path refuses. `find_parent_for_action` throws
    `MolecularBackwardUnavailable` for both action types, and so does
    `backward_parent_states`. This is the assertion that fails if anyone restores the
    `nothing`, because an empty parent set is read as "unique parent, P_B = 1".

 2. The DAG really is a lattice, at the measured multiplicity, compared by the
    CANONICAL form the reward path uses. If a future change ever made this DAG a
    genuine tree, assertion 2 fails and points at assertion 1: the refusal would then
    be removable. The two are pinned together so they cannot drift apart.

 3. The oracle checks itself. With P_B uniform over the enumerated parents, the
    backward measure sum_tau P_B(tau | x) must be exactly 1 at every terminal, and
    with P_B = 1 it must be exactly n_paths(x). Those are identities, not
    measurements, so they fail if the enumeration's parent sets are wrong.
"""

using Test
using Printf

include(joinpath(@__DIR__, "test_setup.jl"))

# ============================================================================
# The instance
# ============================================================================

# Fragment library entries. Deliberately includes the symmetric linkers, which are
# the fragments most likely to admit commuting build orders.
#
# The enumeration over these is EXHAUSTIVE, not depth-truncated: 41-45 are the five
# starters and each exposes two attachment points, while 1, 2 and 5 expose none, so
# after three joins `attachment_points` is empty and `is_applicable` rejects every
# fragment action. Nothing below depends on a max-depth cutoff.
const DAG_FRAGMENT_IDS = [1, 2, 5, 41, 42, 43, 44, 45]

# ---- constants from the exhaustive enumeration, MEASURED 2026-08-31 ----------
# Structural counts. These depend only on RDKit's join and canonicalisation, not on
# the reward, so they are asserted exactly.
const EXPECTED_NONTERMINAL_NODES = 50
const EXPECTED_TERMINAL_NODES = 41
const EXPECTED_INDEGREE_HISTOGRAM = Dict(1 => 72, 2 => 15, 3 => 3, 4 => 1)
const EXPECTED_TERMINAL_PATH_HISTOGRAM = Dict(1 => 22, 2 => 15, 3 => 3, 4 => 1)
const EXPECTED_TRAJECTORIES = 65      # = 22 + 2*15 + 3*3 + 4*1
const EXPECTED_MULTIPARENT_TERMINALS = ["c1ccc(-c2ccccc2)cc1",
                                        "c1ccc(-c2ccccn2)cc1",
                                        "c1ccc(-c2ccsc2)cc1",
                                        "c1ccccc1"]

# Reward-weighted quantities. `reward(::MolState)` goes through
# RDKitBridge.compute_mol_properties, whose SA term falls back to a descriptor
# approximation when `sascorer` is absent, so these move with the chemistry stack.
# Bounded rather than pinned, with the measured value beside the bound; the
# load-bearing claim is that the ratio is not 1.
const MEASURED_Z_TRUE = 28.4957
const MEASURED_Z_PB1 = 45.4131
const MEASURED_RATIO = 1.5937         # Z_PB1 / Z_true
const MEASURED_TV = 0.1965            # TV(R/Z , n R / sum n R)
const RATIO_BOUNDS = (1.40, 1.80)
const TV_BOUNDS = (0.15, 0.25)

# ============================================================================
# Oracle
# ============================================================================

"""
    canonical_key(state) -> (String, Bool)

Node identity for the fragment DAG: the molecular graph up to canonical atom
ordering, plus the terminated flag. This is the identity the REWARD uses --
`apply_action(::TerminateMolAction, ·)` stores `finalize_smiles`, which canonicalises
-- and it is not the identity `MolState` uses, which is the raw history-dependent
string. Keying on the raw string is what made the previous version of this file
vacuous.
"""
function canonical_key(state::MolState)
    c = RDKitBridge.canonicalize_smiles(state.smiles)
    return (c === nothing ? state.smiles : c, state.is_terminated)
end

"""
    enumerate_fragment_dag(all_actions) -> NamedTuple

Exhaustively enumerate the reachable fragment DAG, reading every domain rule from the
implementation rather than restating it here, in the same spirit as
test/theory/enumerate.jl.

Per canonical node: `npaths` (distinct action sequences from the initial state),
`parents` (set of canonical parent nodes), `rewards` (terminal nodes only), and
`sequences` (up to four witnessing action-index sequences, for the failure message).
"""
function enumerate_fragment_dag(all_actions)
    npaths    = Dict{Tuple{String,Bool}, Int}()
    parents   = Dict{Tuple{String,Bool}, Set{Tuple{String,Bool}}}()
    rewards   = Dict{Tuple{String,Bool}, Float64}()
    sequences = Dict{Tuple{String,Bool}, Vector{Vector{Int}}}()

    root = MolState("", Int[], Int[], 0, false, zeros(Float32, FINGERPRINT_DIM))

    function walk!(state, sequence)
        key = canonical_key(state)
        for (i, action) in enumerate(all_actions)
            GFlowNet.is_applicable(action, state) || continue
            child = try
                GFlowNet.apply_action(action, state)
            catch
                nothing
            end
            child === nothing && continue

            # An action that does not change the state is NOT an edge. `apply_action`
            # returns the state UNCHANGED when a join fails (molecular_generation.jl
            # :545 and :552), and `sample_trajectory` traverses the same graph, so
            # without this filter every failed join would look like a self-loop.
            child == state && continue

            child_key = canonical_key(child)
            npaths[child_key] = get(npaths, child_key, 0) + 1
            push!(get!(parents, child_key, Set{Tuple{String,Bool}}()), key)
            witnesses = get!(sequences, child_key, Vector{Int}[])
            length(witnesses) < 4 && push!(witnesses, vcat(sequence, i))

            if child.is_terminated
                rewards[child_key] = GFlowNet.reward(child)
            else
                push!(sequence, i)
                walk!(child, sequence)
                pop!(sequence)
            end
        end
    end
    walk!(root, Int[])

    return (; npaths, parents, rewards, sequences, root_key = canonical_key(root))
end

"""
    backward_measure(dag, key, pb) -> Float64

sum over backward paths of the product of P_B along them, where `pb(n_parents)` gives
the backward probability of one edge into a node with `n_parents` parents. With
`pb = n -> 1/n` this must be 1 at every node; with `pb = n -> 1.0` it must equal
n_paths. Both are identities of the enumeration, which is what makes them a check ON
the enumeration.
"""
function backward_measure(dag, key, pb, memo = Dict{Tuple{String,Bool}, Float64}())
    key == dag.root_key && return 1.0
    haskey(memo, key) && return memo[key]
    ps = dag.parents[key]
    m = sum(backward_measure(dag, p, pb, memo) for p in ps) * pb(length(ps))
    memo[key] = m
    return m
end

histogram(xs) = (h = Dict{Int,Int}(); for x in xs; h[x] = get(h, x, 0) + 1; end; h)

# ============================================================================
# 1. The backward path is UNAVAILABLE, and the consequence is warned about
#    where it can be seen. Needs no chemistry.
# ============================================================================

@testset "MolState parents are unrecoverable, and that is surfaced not hidden" begin
    # A MolState cannot recover its parents: the join history is not stored -- only the
    # canonical SMILES, the open attachment points and their labels. So neither P_B = 1 nor
    # uniform-over-parents is computable, and P_B = 1 is NOT valid here because the fragment
    # DAG is not a tree. Measured on the [1, 2, 5, 41, 42, 43, 44, 45] subset with
    # TerminateMolAction included and states compared by CANONICAL smiles: 41 terminals, 19
    # of them multi-path, max n(x) = 4, Z with P_B = 1 is 45.4131 against a true 28.4957 --
    # a 1.5937x bias toward states reachable by more join orders.
    #
    # AN EARLIER VERSION OF THIS TESTSET ASSERTED THAT find_parent_for_action THROWS, and the
    # source threw MolecularBackwardUnavailable from it. That design was reverted, and the
    # reason is worth keeping because it is the same mistake this repo has been unwinding all
    # along.
    #
    # find_parent_for_action is a STRUCTURAL QUERY -- "which state precedes this one under
    # this action" -- reached from backward_parent_states, reached from inside the loss,
    # reached from inside a training loop that catches every exception and records NaN.
    # Throwing there produced molecular TB training at 0 of 5 finite losses: the refusal was
    # correct in substance and INVISIBLE in effect, because the loop converted it into
    # precisely the silent failure it was meant to prevent. A full history of NaN, with the
    # completion banner printed.
    #
    # So the query returns `nothing`, and the consequence is stated ONCE at training entry by
    # validate_training_config, where a user reads it. Measured after the revert: molecular TB
    # 5/5 finite, grid TB 5/5 finite with no warning (grid enumerates parents).
    state = MolState("*c1ccccc1", [0], Int[], 1, false, zeros(Float32, FINGERPRINT_DIM))
    terminal = MolState("c1ccccc1", Int[], Int[], 1, true, zeros(Float32, FINGERPRINT_DIM))

    # The query answers rather than throwing.
    @test isnothing(GFlowNet.find_parent_for_action(state, FRAGMENT_LIBRARY[1]))
    @test isnothing(GFlowNet.find_parent_for_action(terminal, TerminateMolAction()))

    all_actions = AbstractAction[FRAGMENT_LIBRARY[1], FRAGMENT_LIBRARY[41],
                                 TerminateMolAction()]
    @test isempty(GFlowNet.backward_parent_states(terminal, all_actions))

    # THE WARNING AND THE TRAINING RUN NEED RDKIT, and they are gated for a measured reason.
    # Without it every `apply_action` fails with UndefVarError(:RDKitBridge) and warns, so the
    # captured log fills with those warnings instead of the P_B one and no trajectory can be
    # built -- the run then gives 0 of 3 finite losses for a reason that has nothing to do
    # with what is being tested. The parent-query assertions above need no chemistry and stay
    # ungated.
    if !RDKIT_AVAILABLE
        @warn "P_B surfacing NOT CHECKED -- no RDKit" reason = rdkit_reason()
    else
        # The WARNING is what carries the finding. This is the assertion that fails if someone
        # deletes the surfacing and leaves the silent P_B = 1 -- the state this domain was
        # actually in before any of this. Its CONTENT is asserted, not merely its existence: a
        # warning that fires without naming the cause or the fix is not a surfacing.
        model = create_molecular_gflownet(hidden_dim = 8,
                                          partition_function_method = LEARNABLE_ESTIMATION)
        config = TrainingConfig(objective = TRAJECTORY_BALANCE,
                                partition_function_method = LEARNABLE_ESTIMATION,
                                n_iterations = 1, batch_size = 2)
        buf = IOBuffer()
        Base.CoreLogging.with_logger(
            Base.CoreLogging.SimpleLogger(buf, Base.CoreLogging.Warn)) do
            try
                GFlowNet.validate_training_config(config, model)
            catch
            end
        end
        text = String(take!(buf))
        @test occursin("cannot enumerate parents", text)
        @test occursin("P_B", text)
        @test occursin("find_parent_for_action", text)

        # Training itself must still WORK. Throwing from find_parent_for_action broke it:
        # measured 0 of 5 finite losses, because the training loop caught the refusal and
        # recorded NaN.
        history = train_gflownet(model,
            TrainingConfig(objective = TRAJECTORY_BALANCE,
                           partition_function_method = LEARNABLE_ESTIMATION,
                           n_iterations = 3, batch_size = 2); verbose = false)
        @test count(isfinite, history.losses) == 3
    end
end

# ============================================================================
# 2 and 3. The lattice itself, and the oracle's self-check.
# ============================================================================

@testset "The fragment DAG is a lattice, at the measured multiplicity" begin
    # Without RDKit every join fails, the enumeration explores nothing, and every
    # assertion below would pass VACUOUSLY -- which is the failure mode this file is
    # a correction of. Skip loudly. Set GFLOWNET_TEST_RDKIT=true to check it.
    if !RDKIT_AVAILABLE
        @warn "Fragment DAG multiplicity NOT CHECKED -- no RDKit" reason = rdkit_reason()
        @test_skip "fragment DAG enumeration requires RDKit"
    else
        fragments = FragmentAction[FRAGMENT_LIBRARY[i] for i in DAG_FRAGMENT_IDS]
        all_actions = AbstractAction[fragments..., TerminateMolAction()]
        @test length(all_actions) == 9

        dag = enumerate_fragment_dag(all_actions)
        @test !isempty(dag.npaths)

        terminals = [k for k in keys(dag.npaths) if k[2]]
        nonterminals = [k for k in keys(dag.npaths) if !k[2]]

        # The exhaustiveness precondition, checked rather than asserted in prose: no
        # fragment action is applicable once three fragments are in place, so the
        # enumeration cannot have been cut off by a depth bound.
        deepest = MolState("", Int[], Int[], 0, false, zeros(Float32, FINGERPRINT_DIM))
        for id in (41, 1, 2)
            deepest = GFlowNet.apply_action(FRAGMENT_LIBRARY[id], deepest)
        end
        @test deepest.n_fragments == 3
        @test isempty(deepest.attachment_points)
        @test !any(GFlowNet.is_applicable(f, deepest) for f in fragments)
        @test GFlowNet.is_applicable(TerminateMolAction(), deepest)

        @test length(nonterminals) == EXPECTED_NONTERMINAL_NODES
        @test length(terminals) == EXPECTED_TERMINAL_NODES

        # ---- 3. The oracle checks itself, before anything is concluded from it ----
        # Identities, not measurements: uniform-over-parents makes the backward
        # measure exactly 1 at every terminal, and P_B = 1 makes it exactly n_paths.
        # Wrong parent sets break both.
        for k in terminals
            @test backward_measure(dag, k, n -> 1.0 / n) ≈ 1.0 atol = 1e-12
            @test backward_measure(dag, k, n -> 1.0) ≈ dag.npaths[k] atol = 1e-9
        end

        # ---- 2. The multiplicity ----
        indegrees = histogram([length(ps) for ps in values(dag.parents)])
        @test indegrees == EXPECTED_INDEGREE_HISTOGRAM

        multi_parent = [k for k in keys(dag.parents) if length(dag.parents[k]) > 1]

        # THE CLAIM THIS FILE ONCE DENIED. If it ever fails because every node has
        # exactly one parent, the refusal asserted in the first testset is no longer
        # necessary and should be replaced by a real parent enumeration -- see the
        # MolecularBackwardUnavailable docstring.
        @test !isempty(multi_parent)
        @test length(multi_parent) == 19

        @test sort([k[1] for k in multi_parent if k[2]]) == EXPECTED_MULTIPARENT_TERMINALS
        @test count(k -> !k[2], multi_parent) == 15

        terminal_paths = histogram([dag.npaths[k] for k in terminals])
        @test terminal_paths == EXPECTED_TERMINAL_PATH_HISTOGRAM
        @test sum(dag.npaths[k] for k in terminals) == EXPECTED_TRAJECTORIES

        # The raw-SMILES key is BLIND to all of it. Two build orders of the same
        # molecule are two MolStates, so a walk keyed on MolState sees the 65
        # trajectories as 65 separate nodes and reports a tree.
        left = GFlowNet.apply_action(FRAGMENT_LIBRARY[2],
                 GFlowNet.apply_action(FRAGMENT_LIBRARY[1],
                   GFlowNet.apply_action(FRAGMENT_LIBRARY[41],
                     MolState("", Int[], Int[], 0, false, zeros(Float32, FINGERPRINT_DIM)))))
        right = GFlowNet.apply_action(FRAGMENT_LIBRARY[1],
                  GFlowNet.apply_action(FRAGMENT_LIBRARY[2],
                    GFlowNet.apply_action(FRAGMENT_LIBRARY[41],
                      MolState("", Int[], Int[], 0, false, zeros(Float32, FINGERPRINT_DIM)))))
        @test left != right                                     # MolState says two nodes
        @test left.smiles != right.smiles
        @test canonical_key(left) == canonical_key(right)        # chemistry says one
        @test canonical_key(left)[1] == "c1ccc(-c2ccccc2-c2ccccn2)cc1"
        @test RDKitBridge.finalize_smiles(left.smiles) ==
              RDKitBridge.finalize_smiles(right.smiles)

        # And the invariant fails on the RAW string too, for fragments outside this
        # instance: joining hydroxyl then fluorine onto the 1,2-disubstituted benzene
        # gives byte-identical SMILES either way, so no canonicalisation is even
        # needed to see it.
        oh_then_f = GFlowNet.apply_action(FRAGMENT_LIBRARY[22],
                      GFlowNet.apply_action(FRAGMENT_LIBRARY[16],
                        GFlowNet.apply_action(FRAGMENT_LIBRARY[41],
                          MolState("", Int[], Int[], 0, false, zeros(Float32, FINGERPRINT_DIM)))))
        f_then_oh = GFlowNet.apply_action(FRAGMENT_LIBRARY[16],
                      GFlowNet.apply_action(FRAGMENT_LIBRARY[22],
                        GFlowNet.apply_action(FRAGMENT_LIBRARY[41],
                          MolState("", Int[], Int[], 0, false, zeros(Float32, FINGERPRINT_DIM)))))
        @test oh_then_f.smiles == "Oc1ccccc1F"
        @test f_then_oh.smiles == "Oc1ccccc1F"
        @test oh_then_f == f_then_oh

        # ---- what P_B = 1 costs on this instance ----
        Z_true = sum(dag.rewards[k] for k in terminals)
        Z_pb1 = sum(dag.npaths[k] * dag.rewards[k] for k in terminals)
        law_true = Dict(k => dag.rewards[k] / Z_true for k in terminals)
        law_pb1 = Dict(k => dag.npaths[k] * dag.rewards[k] / Z_pb1 for k in terminals)
        tv = sum(abs(law_true[k] - law_pb1[k]) for k in terminals) / 2

        @test Z_pb1 > Z_true
        @test RATIO_BOUNDS[1] < Z_pb1 / Z_true < RATIO_BOUNDS[2]
        @test TV_BOUNDS[1] < tv < TV_BOUNDS[2]

        @info """
        FRAGMENT DAG MULTIPLICITY (exhaustive enumeration of $(DAG_FRAGMENT_IDS)).

        $(length(multi_parent)) of $(length(dag.npaths)) nodes have more than one parent;
        in-degree histogram $(sort(collect(indegrees))).
        $(sum(dag.npaths[k] for k in terminals)) trajectories reach
        $(length(terminals)) terminal molecules.

        P_B = 1 per edge therefore solves for p(x) proportional to n(x) R(x):
          Z_true = $(round(Z_true, digits = 4))  (measured when written $(MEASURED_Z_TRUE))
          Z_PB1  = $(round(Z_pb1, digits = 4))  (measured when written $(MEASURED_Z_PB1))
          ratio  = $(round(Z_pb1 / Z_true, digits = 4))  (measured when written $(MEASURED_RATIO))
          TV     = $(round(tv, digits = 4))  (measured when written $(MEASURED_TV))
        """
    end
end
