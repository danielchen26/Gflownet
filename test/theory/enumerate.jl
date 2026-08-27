# test/theory/enumerate.jl
#
# Exact enumeration helpers for the grid-world DAG. These exist to give the test
# suite a GROUND TRUTH that does not depend on training, on sampling noise, or on
# any of the implementation's own (previously stubbed) validators.
#
# Grid moves are MoveRight (x+1) and MoveUp (y+1) plus Terminate, so the state
# graph is a lattice: (x,y) has parents {(x-1,y), (x,y-1)} within bounds, and the
# number of distinct (1,1)->(x,y) paths is binomial(x+y-2, x-1).
#
# Every domain rule is READ FROM THE IMPLEMENTATION rather than restated here, so
# these helpers cannot silently drift from the code they are checking.

using GFlowNet

const GS = GFlowNet.GridState

"""
    can_terminate(x, y) -> Bool

Whether the implementation allows Terminate from (x,y), via
`GFlowNet.is_applicable`. Grid world forbids terminating at the start state
(`src/applications/grid_world.jl:107-108`), so (1,1) is NOT a terminal state and
must be excluded from Z. Getting this wrong shifts Z by R(1,1)=0.1.
"""
can_terminate(x::Int, y::Int) =
    GFlowNet.is_applicable(GFlowNet.Terminate(), GS(x, y, false))

"""
    path_counts(n) -> Dict{Tuple{Int,Int},Int}

Number of distinct (1,1) -> (x,y) paths, by dynamic programming, cross-checked
against the lattice closed form. n(x) is the factor by which a TB loss missing
its `sum log P_B` term biases the terminal law.
"""
function path_counts(n::Int)
    u = Dict{Tuple{Int,Int},Int}()
    for x in 1:n, y in 1:n
        u[(x, y)] = (x == 1 && y == 1) ? 1 : get(u, (x - 1, y), 0) + get(u, (x, y - 1), 0)
    end
    for x in 1:n, y in 1:n
        @assert u[(x, y)] == binomial(x + y - 2, x - 1) "path count mismatch at ($x,$y)"
    end
    return u
end

"""Parents of a grid state, within bounds."""
function parents_of(s)
    ps = GS[]
    s.x > 1 && push!(ps, GS(s.x - 1, s.y, false))
    s.y > 1 && push!(ps, GS(s.x, s.y - 1, false))
    return ps
end

"""
    reward_table(n) -> Dict{Tuple{Int,Int},Float64}

R(x) for terminal-capable cells, 0.0 elsewhere, so the mass recursion cannot
credit reward to a cell that is not allowed to terminate.
"""
reward_table(n::Int) = Dict((x, y) => (can_terminate(x, y) ?
                                       GFlowNet.reward(GS(x, y, true)) : 0.0)
                            for x in 1:n, y in 1:n)

"""
    exact_Z(n) -> Float64

Z = sum over reachable terminal states of R(x). This is the denominator of the
defining theorem p(x) = R(x)/Z.
"""
exact_Z(n::Int) = sum(values(reward_table(n)))

"""
    backward_mass(n) -> Dict{Tuple{Int,Int},Float64}

m(s) = R(s) + sum over children of m(child): the total unnormalised reward mass
reachable from s, counting every path separately. m((1,1)) equals
sum_x n(x) R(x) -- the Z that an uncorrected TB loss converges to.
"""
function backward_mass(n::Int)
    R = reward_table(n)
    m = Dict{Tuple{Int,Int},Float64}()
    for x in n:-1:1, y in n:-1:1
        acc = R[(x, y)]
        x < n && (acc += m[(x + 1, y)])
        y < n && (acc += m[(x, y + 1)])
        m[(x, y)] = acc
    end
    return m
end

"""
    analytic_optimum_terminal_law(n) -> Dict{Tuple{Int,Int},Float64}

Terminal distribution induced by the policy that EXACTLY zeroes the coded TB loss
`(log Z + sum log P_F - log R)^2` -- the loss with no `sum log P_B` term.

That zero is attained by P_F(s->s') = m(s')/m(s) and P_terminate(s) = R(s)/m(s),
with log Z = log m(s0). The resulting law is n(x)R(x) / sum_y n(y)R(y): the
path-count bias. Once the backward term is restored this should no longer
describe the optimum, which is precisely how the fix is verified.
"""
function analytic_optimum_terminal_law(n::Int)
    R = reward_table(n)
    m = backward_mass(n)
    reach = Dict{Tuple{Int,Int},Float64}((1, 1) => 1.0)
    law = Dict{Tuple{Int,Int},Float64}()
    for x in 1:n, y in 1:n
        p = get(reach, (x, y), 0.0)
        p == 0.0 && continue
        mm = m[(x, y)]
        mm == 0.0 && continue
        R[(x, y)] > 0 && (law[(x, y)] = p * R[(x, y)] / mm)
        x < n && (reach[(x + 1, y)] = get(reach, (x + 1, y), 0.0) + p * m[(x + 1, y)] / mm)
        y < n && (reach[(x, y + 1)] = get(reach, (x, y + 1), 0.0) + p * m[(x, y + 1)] / mm)
    end
    return law
end

"""
    analytic_optimum_terminal_law_corrected(n) -> Dict{Tuple{Int,Int},Float64}

Terminal law at the zero of the CORRECTED TB loss
`(log Z + sum log P_F - sum log P_B - log R)^2`, using a fixed uniform backward
policy `P_B(parent|child) = 1/|parents(child)|`.

Trajectory Balance is valid for ANY fixed P_B, so at the optimum the terminal law
must be exactly `R(x)/Z` regardless of which P_B was chosen. That invariance is
what this function checks: the reward-mass recursion is reweighted by P_B so that
the path multiplicity cancels instead of accumulating.

Concretely, with `w(child->parent) = 1/|parents(child)|` the forward mass becomes
`m(s) = R(s) + sum_{children c} m(c) * P_B(s|c)`, and `m((1,1))` equals Z rather
than `sum_x n(x)R(x)`.
"""
function analytic_optimum_terminal_law_corrected(n::Int)
    R = reward_table(n)
    nparents(x, y) = length(parents_of(GS(x, y, false)))

    m = Dict{Tuple{Int,Int},Float64}()
    for x in n:-1:1, y in n:-1:1
        acc = R[(x, y)]
        if x < n
            acc += m[(x + 1, y)] / max(nparents(x + 1, y), 1)
        end
        if y < n
            acc += m[(x, y + 1)] / max(nparents(x, y + 1), 1)
        end
        m[(x, y)] = acc
    end

    reach = Dict{Tuple{Int,Int},Float64}((1, 1) => 1.0)
    law = Dict{Tuple{Int,Int},Float64}()
    for x in 1:n, y in 1:n
        p = get(reach, (x, y), 0.0)
        (p == 0.0 || m[(x, y)] == 0.0) && continue
        mm = m[(x, y)]
        R[(x, y)] > 0 && (law[(x, y)] = p * R[(x, y)] / mm)
        if x < n
            share = m[(x + 1, y)] / max(nparents(x + 1, y), 1) / mm
            reach[(x + 1, y)] = get(reach, (x + 1, y), 0.0) + p * share
        end
        if y < n
            share = m[(x, y + 1)] / max(nparents(x, y + 1), 1) / mm
            reach[(x, y + 1)] = get(reach, (x, y + 1), 0.0) + p * share
        end
    end
    return law
end

"""
    set_grid!(n, rewards)

Set the process-global GRID_CONFIG that `GFlowNet.reward(::GridState)` reads.
"""
function set_grid!(n::Int, rewards::Dict{Tuple{Int,Int},Float64})
    GFlowNet.GRID_CONFIG[] = (grid_size = n, reward_positions = rewards)
    return nothing
end
