# Multi-Start GFlowNet Example

This example demonstrates **multi-start GFlowNets** where the model learns from multiple initial states, each with its own partition function Z(s₀).

## Key Concepts

### Multiple Initial States
Instead of a single starting point, we have:
- S₀ = {s₀¹, s₀², ..., s₀ᵏ}
- Each initial state has its own partition function: Z(s₀ⁱ)

### Initial State Distribution
The probability of starting from state s₀ⁱ is:
```
P(s₀ⁱ) = Z(s₀ⁱ) / Σⱼ Z(s₀ʲ)
```

### Learning Process
The model learns:
1. Which initial states lead to high-reward regions
2. Appropriate Z values for each initial state
3. Policies conditioned on different starting points

## Running the Example

```bash
cd examples/core_features/multi_start
julia --project=. multi_start_demo.jl
```

## What You'll See

1. **Training Progress**: Shows how different initial states are being used
2. **Log Z Evolution**: How partition functions change during training
3. **Initial State Distribution**: Which starting points the model prefers
4. **Reward Analysis**: Average rewards from each initial state

## Example Output

```
Initial State Statistics:
  State 1 (1,1):
    - Learned P(s₀): 0.182
    - Actual usage: 18.3%
    - Avg reward: 0.245
    - Final log Z: -0.523
    
  State 2 (5,1):
    - Learned P(s₀): 0.412
    - Actual usage: 41.5%
    - Avg reward: 0.687
    - Final log Z: 0.294
```

## Use Cases

1. **Multi-Modal Distributions**: Different initial states specialize in different modes
2. **Exploration**: Better coverage of the state space
3. **Transfer Learning**: Pre-trained models for different scenarios
4. **Hierarchical Generation**: Start from different abstraction levels

## Visualizations

The example creates `multi_start_results.png` with:
- Log Z evolution over training
- Learned initial state distribution
- Average rewards per initial state
- Terminal state heatmap

## Key Insights

- Initial states leading to better rewards get higher Z values
- The model automatically balances exploration from different starts
- Useful for problems with multiple natural starting points
- Can discover unexpected advantageous starting positions