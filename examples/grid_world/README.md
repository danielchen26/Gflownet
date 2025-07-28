# Grid World GFlowNet Example - Proportional Sampling & Acyclic Control

A comprehensive demonstration of GFlowNet training on a 5×5 grid world with all actions enabled and cycle prevention.

## 🎯 **What This Example Demonstrates**

This example showcases **optimal GFlowNet usage** with key insights:

### **Core GFlowNet Principles:**
- **Proportional Sampling**: P_F(τ) ∝ R(s_T) - samples trajectories proportional to rewards
- **NOT Greedy Optimization**: GFlowNet explores diverse high-reward trajectories
- **Acyclic Control**: Prevents wasteful cycles while preserving exploration benefits

### **Key Features:**
- All 5 actions enabled (Right, Left, Up, Down, Terminate) for maximum exploration
- Strategic reward placement requiring diverse movement directions
- Acyclic rate control (80% cycle prevention) for optimal performance
- Proportional sampling analysis showing theoretically correct behavior

## 🚀 **How to Run**

From this directory (`examples/grid_world/`):

```bash
# Main optimized example
julia grid_world.jl

# Proportional sampling analysis
julia gflownet_analysis.jl

# Acyclic control demonstration  
julia acyclic_example.jl
```

## 📊 **Expected Results & Interpretation**

### **Main Example Output:**
- **~23% optimal reward rate** (reward=50.0) - **This is CORRECT!** 
- **Mean reward: ~15-20** - Shows balanced exploration
- **21 unique positions** - Demonstrates diverse exploration
- **No significant cycles** - Acyclic control working

### **Why 23% is Optimal (Not 100%):**

**GFlowNet Theory**: Sampling frequency ∝ Reward value

| Reward | Expected % | Why Not 100%? |
|--------|------------|----------------|
| 50.0 | ~24% | Proportional sampling, not greedy |
| 40.0 | ~19% | High rewards get high probability |
| 30.0 | ~15% | Medium rewards get medium probability |
| 20.0 | ~10% | Lower rewards get lower probability |

**Key Insight**: 100% optimal would violate GFlowNet principles and prevent exploration!

## ⚙️ **Optimal Configuration**

```julia
# 1. Enable all actions for full exploration
model = create_grid_world_gflownet(
    grid_size=5,
    allow_all_moves=true,  # All 5 actions
    reward_positions=Dict(
        (5, 5) => 50.0,  # Requires optimal navigation
        (1, 5) => 40.0,  # Requires left movement  
        (5, 1) => 40.0,  # Requires down movement
        (3, 3) => 30.0   # Center position
    )
)

# 2. Use acyclic control to prevent cycles
config = create_default_sampling_config(acyclic_rate=0.8)
trajectories = [sample_trajectory(model; config=config) for _ in 1:100]
```

**Why This Works:**
- **All 5 actions**: Can reach positions requiring any movement direction
- **Acyclic control**: Eliminates ~80% of wasteful cycles
- **Strategic rewards**: Encourage optimal navigation patterns
- **Proportional sampling**: Achieves theoretically correct reward distribution

## 🔧 **Acyclic Rate Control**

### **Key Parameter: `acyclic_rate`**

```julia
# No cycle control (baseline - many cycles)
config = SamplingConfig(acyclic_rate=0.0)

# Light cycle prevention (50% reduction)
config = create_default_sampling_config(acyclic_rate=0.5)

# Recommended setting (80% cycle prevention)
config = create_default_sampling_config(acyclic_rate=0.8)

# Strict DAG enforcement (100% cycle prevention)
config = create_default_sampling_config(acyclic_rate=1.0)
```

### **Performance Impact:**

| acyclic_rate | Cycles | Efficiency | Trade-off |
|--------------|--------|------------|-----------|
| 0.0 | Many | Low | Full exploration, wasteful |
| 0.8 | Few | High | **Optimal balance** |
| 1.0 | None | Highest | May limit exploration |

## 📈 **Learning Objectives**

This example teaches:

### **1. GFlowNet vs Optimization:**
- **GFlowNet**: Samples diverse trajectories ∝ rewards
- **Optimization**: Finds single best solution
- **Use GFlowNet when**: You want diverse high-quality solutions

### **2. Proportional Sampling:**
- **23% optimal rate is CORRECT** - shows proper GFlowNet behavior
- **100% would be WRONG** - violates proportional sampling principle
- **Higher rewards → higher probability** (not certainty)

### **3. Acyclic Control:**
- **Problem**: 5-action models can waste computation on cycles
- **Solution**: `acyclic_rate` parameter prevents cycles
- **Benefit**: Exploration + efficiency in one framework

### **4. Production Guidelines:**
- **Start with**: `acyclic_rate=0.8` for most grid-like problems
- **Monitor**: Cycle statistics during development
- **Expect**: Proportional reward distribution, not greedy behavior

## 🔬 **Advanced Analysis**

### **Files Included:**

1. **`grid_world.jl`** - Main optimized example with acyclic control
2. **`gflownet_analysis.jl`** - Theoretical proportional sampling analysis  
3. **`acyclic_example.jl`** - Demonstrates acyclic rate effects
4. **`report_generation.jl`** - Visualization and analysis tools

### **Key Metrics to Monitor:**

- **Proportional sampling**: Reward distribution matches theory
- **Cycle statistics**: Reduced with acyclic control
- **Exploration diversity**: Number of unique positions reached
- **Training stability**: Consistent convergence across runs

## 💡 **Common Misconceptions Addressed**

### **❌ "GFlowNet should achieve 100% optimal"**
**✅ Reality**: GFlowNet samples proportionally to rewards (P_F(τ) ∝ R(s_T))

### **❌ "More actions always help"** 
**✅ Reality**: More actions help IF cycles are controlled via acyclic_rate

### **❌ "Cycles don't matter much"**
**✅ Reality**: Cycles can waste 80%+ of computation without control

### **❌ "Acyclic control limits exploration"**
**✅ Reality**: Smart acyclic control (rate=0.8) improves both efficiency AND exploration

## 🎯 **Success Criteria**

A well-trained model should show:

✅ **Proportional reward distribution** (not greedy optimization)  
✅ **~20-25% optimal reward rate** (for reward=50.0 in this setup)  
✅ **Diverse exploration** (15+ unique terminal positions)  
✅ **Minimal cycles** (with acyclic_rate=0.8)  
✅ **Stable training** (consistent convergence)  

Perfect for understanding GFlowNet theory and building intuition for more complex applications!