# GFlowNet Visualization Module
# Modern web-based visualization system

using ..GFlowNet: GFlowNetModel, Trajectory, TrainingHistory, sample_trajectory, reward, AbstractState

"""
    export_trajectory_data(trajectory::Trajectory)

Export trajectory data to JSON-compatible format for web visualization.
"""
function export_trajectory_data(trajectory::Trajectory)
    states = []
    for (i, state) in enumerate(trajectory.states)
        push!(states, Dict(
            "index" => i,
            "features" => state_to_features(state),
            "reward" => reward(state),
            "action" => i < length(trajectory.states) ? trajectory.actions[i] : nothing
        ))
    end
    return Dict("states" => states, "total_reward" => trajectory.total_reward)
end

"""
    export_training_history(history::TrainingHistory)

Export training history to JSON-compatible format for web visualization.
"""
function export_training_history(history::TrainingHistory)
    return Dict(
        "losses" => history.losses,
        "rewards" => history.mean_rewards,
        "episodes" => history.episodes,
        "timestamps" => history.timestamps
    )
end

# Export functions for web API
export export_trajectory_data, export_training_history

@info """
GFlowNet Web Visualization System
================================
Visualization is now handled by a modern web stack:
- React + TypeScript for the UI
- D3.js + Three.js for beautiful visualizations
- WebSocket for real-time updates
- Oxygen.jl API for data export

To start the visualization dashboard:
1. Start the API server: `julia --project -e 'include("src/utils/visualization/api/server.jl")'`
2. Start the web dashboard: `cd src/utils/visualization/web && npm start`
"""