# Main training loop and infrastructure for GFlowNet
# Moved from core/interface.jl for better organization

using Zygote
using Statistics
using Optimisers
using ComponentArrays
using Random

using ..GFlowNet: AbstractState, AbstractAction, GFlowNetModel, Trajectory
using ..GFlowNet: TrainingConfig, TrainingHistory, TrainingObjective
using ..GFlowNet: TRAJECTORY_BALANCE, DETAILED_BALANCE, FLOW_MATCHING
using ..GFlowNet: sample_trajectory, is_valid_trajectory
using ..GFlowNet: state_to_features, reward, is_terminal_state
using ..GFlowNet: get_applicable_actions, apply_action
using ..GFlowNet: forward_action_probabilities, compute_backward_probability
using ..GFlowNet: flow, flow_estimate, clear_flow_cache!

# Import the compute_trajectory_loss function that will be moved here
# For now, we'll define the structure and move the actual implementation later