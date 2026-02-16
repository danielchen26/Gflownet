# Flow Matching Objective Example

This example demonstrates the **FLOW_MATCHING** training objective, which directly optimizes the flow conservation equation.

## Mathematical Foundation

Flow Matching enforces:
```
F(s) = Σ_{s'∈children(s)} P_F(s'|s) * F(s')
```

The neural network learns to estimate F(s) directly, minimizing:
```
L_FM(s) = (Z(s) - Σ_{s'} P_F(s'|s) * F(s'))²
```

where:
- Z(s) is the neural network's flow estimate
- The sum represents the expected flow computed recursively

## Key Features

1. **Direct Flow Learning**: Neural network learns F(s) values explicitly
2. **No Backward Policy**: Unlike DETAILED_BALANCE, only needs forward policy
3. **Flow Conservation**: Optimizes conservation equation directly
4. **Explicit Estimates**: Provides direct access to flow values

## Running the Example

```bash
cd examples/core_features/flow_matching
julia --project=. flow_matching_demo.jl
```

## What You'll See

1. **Training Progress**: Loss decreasing as flow conservation improves
2. **Conservation Analysis**: How well the learned flows satisfy conservation
3. **Flow Heatmap**: Visualization of learned F(s) values
4. **Comparison**: Neural network flows vs recursive computation

## When to Use FLOW_MATCHING

- When you need explicit flow estimates
- For debugging flow conservation issues
- When backward policy is not available/needed
- For understanding flow dynamics in your domain

## Output Files

- `flow_matching_results.png`: Comprehensive visualization of results
  - Training loss curve
  - Conservation error over time
  - Terminal state distribution
  - Learned flow value heatmap