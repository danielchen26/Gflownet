# Grid World GFlowNet Example

A simple demonstration of GFlowNet training on a 5×5 grid world environment.

## 🎯 **What This Example Demonstrates**

This example shows how to:
- Set up a discrete grid world environment for GFlowNet training
- Define reward positions to guide trajectory generation  
- Train a GFlowNet using trajectory balance objective
- Visualize training progress and learned policies
- Sample diverse trajectories from the trained model

## 🚀 **How to Run**

From this directory (`examples/grid_world/`):

```bash
julia grid_world.jl
```

The script will:
1. Create a 5×5 grid environment with rewards at positions (3,3), (1,5), and (5,1)
2. Set up forward policy and flow estimator neural networks
3. Train for 1000 iterations using trajectory balance
4. Generate visualizations and save all results to `results/`

## 📊 **Expected Output**

The example generates several files in the `results/` folder:

- **Training curve** (`grid_world_loss.png`): Shows loss evolution over iterations
- **Trajectory visualization** (`grid_world_paths.png`): Shows sampled paths on the grid
- **Reward distribution** (`grid_world_rewards.png`): Histogram of trajectory rewards
- **Training data** (`grid_world_training.csv`): Raw training metrics for analysis

## ⚙️ **Configuration**

Current training setup:
- **Grid**: 5×5 with rewards at (3,3)=10.0, (1,5)=5.0, (5,1)=5.0
- **Neural Networks**: 3-layer MLPs with 64 hidden units
- **Training**: 1000 iterations, batch size 32, learning rate 0.001
- **Objective**: Trajectory Balance with simple partition function estimation

## 🔧 **Customization**

To modify the example:

1. **Change grid size**: Edit grid dimensions in `create_grid_world_dag()`
2. **Adjust rewards**: Modify reward positions and values 
3. **Training config**: Update `TrainingConfig` parameters
4. **Network architecture**: Change hidden dimensions in neural network creation
5. **Visualizations**: Customize plots in the visualization functions

## 📖 **Learning Objectives**

This example demonstrates:
- **Environment setup**: How to define states, actions, and rewards
- **Model creation**: Setting up policies and flow estimators  
- **Training process**: Using the modern `train_gflownet()` interface
- **Result analysis**: Visualizing training progress and model behavior

Perfect for understanding the basic GFlowNet training workflow before moving to more complex examples like molecular design or causal discovery. 