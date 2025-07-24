# GFlowNet.jl Examples

*Fully modernized examples showcasing production-ready GFlowNet implementations with cutting-edge 2024 research capabilities.*

## 🚀 **Modern Features in All Examples**

Every example now uses:
- **Modern Training Interface**: `TrainingConfig` + `train_gflownet()` 
- **Advanced Objectives**: Domain-optimized training objectives
- **Intelligent Z Estimation**: Automatic partition function learning
- **Robust Error Handling**: Graceful fallbacks and comprehensive monitoring
- **Rich Visualization**: Training progress and results automatically generated

## 📋 **Available Examples**

### **🎯 Grid World** *(Perfect for Beginners)*
```bash
cd examples/grid_world && julia grid_world.jl
```
**Configuration**: `TRAJECTORY_BALANCE` + `SIMPLE_ESTIMATION`

**Features**:
- Clean introduction to GFlowNet concepts
- 5×5 grid navigation with multiple reward positions
- Automatic path visualization and training progress plots
- Modern interface demonstration

**Why this configuration**: Deterministic backward paths, enumerable terminal states

---

### **🧪 Active Learning** *(Intermediate)*
```bash
cd examples/active_learning && julia active_learning.jl
```
**Configuration**: `SUB_TRAJECTORY_BALANCE` + `ADAPTIVE_ESTIMATION`

**Features**:
- Intelligent experiment selection strategy
- Sequential decision making with credit assignment
- Dynamic experiment value handling
- Robust training with fallback mechanisms

**Why this configuration**: Sequential decisions benefit from sub-trajectory balance, dynamic values need adaptive Z estimation

---

### **🔬 Feature Acquisition** *(Most Advanced)*
```bash
cd examples/feature_acquisition && julia main.jl
```
**Configuration**: `ADAPTIVE_SUB_TB` + `ADAPTIVE_ESTIMATION`

**Features**:
- Strategic experimental design with budget constraints
- Non-deterministic path handling (multiple ways to reach same feature sets)
- Sparse important decision focusing with difficulty targeting
- Real-world drug discovery application scenarios

**Why this configuration**: Most complex example needing cutting-edge techniques for non-deterministic environments with sparse critical decisions

---

### **📊 Causal Discovery** *(Research-Grade)*
```bash
cd examples/causal_discovery && julia causal_discovery.jl
```
**Configuration**: `GENERAL_TRAJECTORY_BALANCE` + `SAMPLING_BASED`

**Features**:
- DAG structure learning from observational data
- Complete theoretical formulation with P_B term
- Complex graph space navigation
- BIC scoring and structural Hamming distance evaluation

**Why this configuration**: Non-deterministic graph construction requires general TB, complex spaces need sampling-based Z estimation

---

### **⚛️ Molecular Design** *(Specialized)*
```bash
cd examples/molecule_design && julia molecule_example.jl
```
**Configuration**: `HIERARCHICAL_SUB_TB` + `SIMPLE_ESTIMATION`

**Features**:
- Sequential molecule construction (atom-by-atom)
- Multi-scale molecular structure learning
- Hierarchical balance at different scales (bonds, rings, motifs, molecules)
- Chemical space optimization

**Why this configuration**: Molecular construction has natural hierarchical structure, chemical spaces are often enumerable

## 🔧 **Quick Setup**

### **One-Time Setup**
```bash
# Install all dependencies for all examples
julia examples/setup_examples.jl
```

### **Run Any Example**
```bash
# Choose your experience level
cd examples/grid_world && julia grid_world.jl           # Beginner
cd examples/active_learning && julia active_learning.jl # Intermediate  
cd examples/feature_acquisition && julia main.jl        # Advanced
cd examples/causal_discovery && julia causal_discovery.jl # Research
cd examples/molecule_design && julia molecule_example.jl  # Specialized
```

## 📊 **Configuration Comparison**

| **Example** | **Objective** | **Z Method** | **Complexity** | **Key Feature** |
|-------------|---------------|--------------|----------------|-----------------|
| Grid World | `TRAJECTORY_BALANCE` | `SIMPLE_ESTIMATION` | Beginner | Clean introduction |
| Active Learning | `SUB_TRAJECTORY_BALANCE` | `ADAPTIVE_ESTIMATION` | Intermediate | Sequential decisions |
| Feature Acquisition | `ADAPTIVE_SUB_TB` | `ADAPTIVE_ESTIMATION` | Advanced | Non-deterministic handling |
| Causal Discovery | `GENERAL_TRAJECTORY_BALANCE` | `SAMPLING_BASED` | Research | Complete theory |
| Molecular Design | `HIERARCHICAL_SUB_TB` | `SIMPLE_ESTIMATION` | Specialized | Multi-scale structure |

## 🎓 **Learning Progression**

### **1. Start Here: Grid World**
- Understand basic GFlowNet concepts
- Learn state/action definitions
- See flow conservation in action
- Modern training interface introduction

### **2. Intermediate: Active Learning**
- Sequential decision making
- Credit assignment importance
- Dynamic environments
- Sub-trajectory balance benefits

### **3. Advanced: Feature Acquisition**
- Non-deterministic path handling
- Sparse decision importance
- Real-world complexity
- Cutting-edge techniques

### **4. Research: Causal Discovery**
- Complete theoretical formulation
- Complex space navigation
- Advanced evaluation metrics
- Production research code

### **5. Specialized: Molecular Design**
- Domain-specific optimization
- Hierarchical structures
- Scale-aware learning
- Chemical space expertise

## 🛠️ **What You'll See**

All examples provide **rich, modern output**:

```
Training GFlowNet with modern training interface...

Training configuration:
  Objective: TRAJECTORY_BALANCE (standard GFlowNet theory)
  Partition function method: SIMPLE_ESTIMATION (enumerable terminal states)
  Batch size: 32
  Iterations: 1000

Starting training with configuration:
Iteration 100: Loss = 0.245, Z estimate = 17.5
Iteration 200: Loss = 0.156, Z estimate = 16.8
...

Training completed!
  Final loss: 0.0234
  Final Z estimate: 15.2
  Total training iterations: 1000

Results saved to [example]_*.png
```

## 📈 **Modern vs Legacy**

### **❌ OLD (What we had)**
```julia
# Manual training loops in every example
for iter in 1:n_iterations
    trajectories = [sample_trajectory(model) for _ in 1:batch_size]
    loss, grad = compute_loss_and_grad(model, trajectories)
    apply_optimizer!(model, grad)
    # ... manual logging, Z estimation, error handling
end
```

### **✅ NEW (What we have now)**
```julia
# Clean, consistent, optimized training
config = TrainingConfig(
    objective=TRAJECTORY_BALANCE,
    partition_function_method=SIMPLE_ESTIMATION,
    batch_size=32,
    n_iterations=1000
)

history = train_gflownet(model, config; verbose=true)
```

## 🎯 **Domain-Specific Quick Config**

Each domain now has optimized default configurations:

```julia
# Automatic optimal setup for each domain
config_grid = create_modern_training_config(:grid_world)
config_active = create_modern_training_config(:active_learning)  
config_feature = create_modern_training_config(:feature_acquisition)
config_causal = create_modern_training_config(:causal_discovery)
config_molecular = create_modern_training_config(:molecular_design)
```

## 📁 **Example Structure**

Each example is **self-contained** with:
- `Project.toml` - Isolated dependencies
- `Manifest.toml` - Exact versions
- `[example].jl` - Main executable
- `README.md` - Specific documentation
- Generated outputs (`.png`, `.csv` files)

## 🔄 **Migration from Legacy**

If you have old GFlowNet code, the **modernization is straightforward**:

1. **Replace training loops** with `TrainingConfig` + `train_gflownet()`
2. **Select appropriate objective** for your domain type
3. **Choose Z estimation method** based on space characteristics
4. **Add error handling** with try-catch blocks
5. **Use training history** for metrics and visualization

See the [Migration Guide](../docs/MIGRATION_GUIDE.md) for detailed instructions.

## ✅ **Validation**

All examples have been **thoroughly validated**:
- ✅ Modern interface used throughout
- ✅ Domain-optimized configurations selected
- ✅ Comprehensive error handling implemented
- ✅ Results quality preserved or improved
- ✅ Documentation updated and accurate
- ✅ Backward compatibility maintained

## 🚀 **Ready for Production**

These examples demonstrate **production-ready patterns**:
- **Robust error handling** with graceful degradation
- **Comprehensive monitoring** and logging
- **Scalable architecture** for deployment
- **Research-grade techniques** in practical applications
- **Clear documentation** for maintenance and extension

## 📚 **Next Steps**

After exploring the examples:
- Read the [Core Concepts](../docs/src/guide/core_concepts.md) for theoretical background
- Study [Training Objectives](../docs/src/guide/training_objectives.md) for advanced techniques
- Check [API Reference](../docs/src/api/) for implementation details
- Contribute new examples using the modern patterns!

---

*These examples showcase the full power of modern GFlowNet.jl while maintaining clarity and educational value.* 🎯 