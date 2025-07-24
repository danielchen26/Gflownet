# 🔴 Phase 1: Critical Fixes

## 📋 Overview

This phase addresses critical blockers that prevent the examples from working properly with the modern GFlowNet framework. These are must-fix issues that break core functionality.

## 🎯 Tasks

### **Task P1.1: Fix Grid World Reward Function**
**Priority**: CRITICAL  
**Estimated Time**: 30 minutes  
**Status**: [ ] PENDING

**Problem**: Grid world returns `0.0` rewards for non-terminal states, violating GFlowNet requirements.

**Current Code**:
```julia
function GFlowNet.reward(state::GridState)
    if !state.is_terminal
        return 0.0  # ❌ BREAKS GFLOWNET
    end
    # ... reward logic
end
```

**Fix Required**:
```julia
function GFlowNet.reward(state::GridState)
    if !state.is_terminal
        return 0.01  # ✅ Small positive reward
    end
    # Return actual rewards for terminal states
    for (pos, reward_value) in REWARD_POSITIONS
        if (state.x, state.y) == pos
            return reward_value
        end
    end
    return 0.1  # ✅ Default positive reward
end
```

**Files to Update**:
- `examples/grid_world/grid_world.jl` (lines ~95-105)

---

### **Task P1.2: Modernize Grid World Training Interface**
**Priority**: CRITICAL  
**Estimated Time**: 1 hour  
**Status**: [ ] PENDING

**Problem**: Grid world uses legacy manual training loop instead of modern `train_gflownet()` interface.

**Current Approach**:
```julia
# ❌ Manual training loop
for iter in 1:n_iterations
    trajectories = [sample_trajectory(...) for _ in 1:batch_size]
    total_loss, total_grad = compute_loss_and_grad(...)
    apply_optimizer!(...)
end
```

**Modern Approach**:
```julia
# ✅ Modern training interface
config = TrainingConfig(
    objective = TRAJECTORY_BALANCE,
    partition_function_method = ADAPTIVE_ESTIMATION,
    batch_size = 32,
    n_iterations = 1000
)
model = create_grid_world_model()
train_gflownet(model, config)
```

**Implementation Steps**:
1. Create `create_grid_world_model()` function
2. Replace manual training with `TrainingConfig`
3. Use `train_gflownet()` function
4. Remove custom gradient computation
5. Remove custom trajectory sampling

**Files to Update**:
- `examples/grid_world/grid_world.jl` (major refactoring)

---

### **Task P1.3: Create Grid World Model Factory**
**Priority**: CRITICAL  
**Estimated Time**: 45 minutes  
**Status**: [ ] PENDING

**Problem**: Grid world doesn't create a proper `GFlowNetModel` instance.

**Required Implementation**:
```julia
function create_grid_world_model()
    # 1. Create states and actions
    initial_state = GridState(1, 1, false)
    terminal_states = [GridState(x, y, true) for x in 1:GRID_SIZE for y in 1:GRID_SIZE]
    terminal_sink = GridState(0, 0, true)
    actions = create_grid_actions()
    
    # 2. Create DAG using core function
    dag = create_dag(initial_state, terminal_states, terminal_sink, actions)
    
    # 3. Create policies using core functions
    input_dim = GRID_SIZE * GRID_SIZE + 1
    output_dim = length(dag.states)
    rng = Random.default_rng()
    
    forward_policy, forward_ps, forward_st = create_forward_policy(input_dim, 64, output_dim, rng)
    flow_estimator, flow_ps, flow_st = create_flow_estimator(input_dim, 64, rng)
    
    # 4. Create optimizers
    opt = Optimisers.Adam(0.001)
    forward_opt_state = Optimisers.setup(opt, forward_ps)
    flow_opt_state = Optimisers.setup(opt, flow_ps)
    optimizer = (forward = forward_opt_state, flow = flow_opt_state)
    
    # 5. Create GFlowNetModel
    return GFlowNetModel(
        dag = dag,
        forward_policy = forward_policy,
        backward_policy = nothing,
        flow_estimator = flow_estimator,
        partition_function = nothing,
        objectives = [TrajectoryBalanceObjective(1.0)],
        optimizer = optimizer,
        parameters = (forward = forward_ps, backward = nothing, flow = flow_ps),
        states = (forward = forward_st, backward = nothing, flow = flow_st)
    )
end
```

**Files to Create/Update**:
- `examples/grid_world/grid_world.jl` (add function)

---

### **Task P1.4: Fix Grid World State Feature Representation**
**Priority**: HIGH  
**Estimated Time**: 15 minutes  
**Status**: [ ] PENDING

**Problem**: Current feature representation may have indexing issues.

**Current Code Review**:
```julia
function GFlowNet.state_to_features(state::GridState)
    pos = zeros(Float32, GRID_SIZE * GRID_SIZE)
    idx = (state.y - 1) * GRID_SIZE + state.x  # Check if this is correct
    pos[idx] = 1.0f0
    return vcat(pos, [Float32(state.is_terminal)])
end
```

**Validation Required**:
- Ensure indexing is correct for all grid positions
- Verify output dimension matches model expectations
- Test edge cases (boundaries)

**Files to Update**:
- `examples/grid_world/grid_world.jl` (validation and potential fix)

---

### **Task P1.5: Remove Legacy Training Code**
**Priority**: HIGH  
**Estimated Time**: 30 minutes  
**Status**: [ ] PENDING

**Problem**: Grid world contains extensive legacy training code that should be removed.

**Code to Remove**:
- `train_grid_gflownet()` function
- `compute_trajectory_balance_loss_and_gradients()` function  
- `sample_grid_trajectory()` function (if using core version)
- Custom logging infrastructure (use core logging)

**Files to Update**:
- `examples/grid_world/grid_world.jl` (remove ~200 lines)

---

## 📊 Phase 1 Completion Criteria

### Functional Tests
- [ ] Grid world example runs without errors
- [ ] Uses modern training interface (`train_gflownet`)
- [ ] All rewards are positive
- [ ] Model training converges
- [ ] Visualizations work correctly

### Code Quality
- [ ] No legacy training loops remain
- [ ] Uses core DAG, policies, and training
- [ ] Follows modern patterns consistently
- [ ] Code is clean and well-documented

### Integration Tests
- [ ] Compatible with current GFlowNet core
- [ ] Uses exported functions correctly
- [ ] No custom reimplementations of core functionality

## 📝 Implementation Order

1. **P1.1**: Fix reward function (quickest fix)
2. **P1.3**: Create model factory (foundation)
3. **P1.2**: Modernize training interface (main refactor)
4. **P1.4**: Validate state features (testing)
5. **P1.5**: Remove legacy code (cleanup)

## 🧪 Testing Strategy

After each task:
1. Run the grid world example
2. Check for errors and warnings
3. Verify training proceeds normally
4. Validate outputs are reasonable

## 📁 Backup Strategy

Before making changes:
```bash
cp examples/grid_world/grid_world.jl examples/grid_world/grid_world_backup.jl
```

## 📈 Success Metrics

- Example runs without errors
- Training loss decreases over time
- Agent learns to reach high-reward positions
- Visualizations show meaningful trajectories
- Code uses modern GFlowNet patterns

---

*Phase 1 Status: PLANNING → READY TO IMPLEMENT* 