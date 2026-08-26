# ============================================================================
# Edit-TB Model — 3-Head Factored Scoring Model
# ============================================================================
#
# Factored within-HE edit policy pilot.
# Three independent scoring heads for basin, parent, and operator decisions.
# Each head scores candidates via softmax over candidate sets.
# Total ~6K parameters without task conditioning; slightly larger with it.

using Lux
using Random

# ============================================================================
# Model Creation
# ============================================================================

"""
Create the 3-head factored edit policy model.

Each head is a small MLP that scores (context, candidate) pairs:
- Basin head:    task(optional) + frontier_features(8) + basin_candidate(5 or 6) → score
- Parent head:   task(optional) + basin_context(4) + parent_candidate(4 or 5) → score
- Operator head: task(optional) + parent_context(6) + operator_candidate(3 or 4) → score
"""
function create_edit_policy(rng::AbstractRNG;
                            config::EditTBConfig=EditTBConfig(),
                            task_names::Vector{String}=String[])
    task_dim = _edit_tb_task_feature_dim(task_names; enabled=config.task_conditioning)
    basin_in = task_dim + (config.include_heuristic_scores ? 14 : 13)   # frontier(8) + basin(6 or 5)
    parent_in = task_dim + (config.include_heuristic_scores ? 9 : 8)    # basin_ctx(4) + parent(5 or 4)
    operator_in = task_dim + (config.include_heuristic_scores ? 10 : 9) # parent_ctx(6) + operator(4 or 3)
    hidden = 64

    basin_head = Lux.Chain(
        Lux.Dense(basin_in, hidden, Lux.relu),
        Lux.Dense(hidden, 1)
    )
    parent_head = Lux.Chain(
        Lux.Dense(parent_in, hidden, Lux.relu),
        Lux.Dense(hidden, 1)
    )
    operator_head = Lux.Chain(
        Lux.Dense(operator_in, hidden, Lux.relu),
        Lux.Dense(hidden, 1)
    )

    return (basin=basin_head, parent=parent_head, operator=operator_head)
end

"""
Initialize the edit policy: create model + parameters + states.
Returns (model, params, states).
"""
function init_edit_policy(rng::AbstractRNG;
                          config::EditTBConfig=EditTBConfig(),
                          task_names::Vector{String}=String[])
    model = create_edit_policy(rng; config=config, task_names=task_names)
    ps_b, st_b = Lux.setup(rng, model.basin)
    ps_p, st_p = Lux.setup(rng, model.parent)
    ps_o, st_o = Lux.setup(rng, model.operator)
    params = (basin=ps_b, parent=ps_p, operator=ps_o)
    states = (basin=st_b, parent=st_p, operator=st_o)
    return model, params, states
end

"""Count total parameters in the edit policy."""
function count_edit_policy_params(params)
    function _count(x)
        if x isa AbstractArray
            return length(x)
        elseif x isa NamedTuple
            return sum(_count(v) for v in values(x); init=0)
        elseif x isa Tuple
            return sum(_count(v) for v in x; init=0)
        else
            return 0
        end
    end
    return _count(params)
end
