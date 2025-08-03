# Visualization Utilities - On-Demand Approach Compatible
# Simple visualization functions that work without explicit DAG construction

using Plots
using ..GFlowNet: GFlowNetModel, Trajectory, TrainingHistory, sample_trajectory, reward, AbstractState

"""
    plot_trajectory(trajectory::Trajectory; title="GFlowNet Trajectory")

Plot a single trajectory showing the sequence of states and actions.
Works with any domain by using state_to_features for coordinates.
"""
function plot_trajectory(trajectory::Trajectory; title="GFlowNet Trajectory")
    if isempty(trajectory.states)
        return plot(title=title, xlabel="No trajectory to plot")
    end

    # Extract coordinates from state features
    coords = []
    for state in trajectory.states
        features = state_to_features(state)
        if length(features) >= 2
            push!(coords, (features[1], features[2]))
        else
            # Fallback for 1D or other representations
            push!(coords, (length(coords), features[1]))
        end
    end

    # Extract x and y coordinates
    x_coords = [c[1] for c in coords]
    y_coords = [c[2] for c in coords]

    # Create plot
    p = plot(x_coords, y_coords,
             marker=:circle,
             markersize=6,
             linewidth=2,
             title=title,
             xlabel="State Feature 1",
             ylabel="State Feature 2",
             legend=false)

    # Mark start and end
    if length(coords) > 0
        plot!(p, [x_coords[1]], [y_coords[1]],
              marker=:star, markersize=10, markercolor=:green,
              label="Start")

        if length(coords) > 1
            plot!(p, [x_coords[end]], [y_coords[end]],
                  marker=:diamond, markersize=10, markercolor=:red,
                  label="End")
        end
    end

    return p
end

"""
    plot_multiple_trajectories(trajectories::Vector{Trajectory}; max_trajectories=10)

Plot multiple trajectories on the same plot for comparison.
"""
function plot_multiple_trajectories(trajectories::Vector{Trajectory}; max_trajectories=10)
    if isempty(trajectories)
        return plot(title="No trajectories to plot")
    end

    # Limit number of trajectories for readability
    trajectories_to_plot = trajectories[1:min(length(trajectories), max_trajectories)]

    p = plot(title="Multiple GFlowNet Trajectories",
             xlabel="State Feature 1",
             ylabel="State Feature 2")

    for (i, traj) in enumerate(trajectories_to_plot)
        if !isempty(traj.states)
            # Extract coordinates
            coords = []
            for state in traj.states
                features = state_to_features(state)
                if length(features) >= 2
                    push!(coords, (features[1], features[2]))
                else
                    push!(coords, (length(coords), features[1]))
                end
            end

            x_coords = [c[1] for c in coords]
            y_coords = [c[2] for c in coords]

            # Plot trajectory with different colors
            plot!(p, x_coords, y_coords,
                  alpha=0.7,
                  linewidth=1,
                  marker=:circle,
                  markersize=3,
                  label="Trajectory $i")
        end
    end

    return p
end

"""
    plot_training_progress(history::TrainingHistory)

Plot training progress including loss and gradient norms.
"""
function plot_training_progress(history::TrainingHistory)
    if isempty(history.losses)
        return plot(title="No training history available")
    end

    # Create subplots
    p1 = plot(history.losses,
              title="Training Loss",
              xlabel="Iteration",
              ylabel="Loss",
              linewidth=2,
              color=:blue)

    p2 = plot(history.gradient_norms,
              title="Gradient Norms",
              xlabel="Iteration",
              ylabel="Gradient Norm",
              linewidth=2,
              color=:red)

    p3 = plot(history.iteration_times,
              title="Iteration Times",
              xlabel="Iteration",
              ylabel="Time (s)",
              linewidth=2,
              color=:green)

    # Combine plots
    return plot(p1, p2, p3, layout=(3, 1), size=(800, 600))
end

"""
    plot_reward_distribution(trajectories::Vector{Trajectory})

Plot the distribution of rewards from trajectories.
"""
function plot_reward_distribution(trajectories::Vector{Trajectory})
    rewards = Float64[]

    for traj in trajectories
        if !isempty(traj.states)
            terminal_state = traj.states[end]
            if is_terminal_state(terminal_state)
                push!(rewards, reward(terminal_state))
            end
        end
    end

    if isempty(rewards)
        return plot(title="No rewards to plot")
    end

    # Create histogram
    p = histogram(rewards,
                  title="Reward Distribution",
                  xlabel="Reward Value",
                  ylabel="Frequency",
                  bins=20,
                  alpha=0.7,
                  color=:lightblue)

    # Add statistics
    mean_reward = mean(rewards)
    vline!(p, [mean_reward],
           linewidth=3,
           color=:red,
           linestyle=:dash,
           label="Mean: $(round(mean_reward, digits=2))")

    return p
end

"""
    plot_state_visitation(trajectories::Vector{Trajectory})

Plot state visitation frequency for 2D state spaces.
Works best with grid worlds or other 2D domains.
"""
function plot_state_visitation(trajectories::Vector{Trajectory})
    # Collect all state coordinates
    state_coords = Dict{Tuple{Float64,Float64}, Int}()

    for traj in trajectories
        for state in traj.states
            features = state_to_features(state)
            if length(features) >= 2
                coord = (round(features[1], digits=3), round(features[2], digits=3))
                state_coords[coord] = get(state_coords, coord, 0) + 1
            end
        end
    end

    if isempty(state_coords)
        return plot(title="No state coordinates to plot")
    end

    # Extract coordinates and counts
    x_coords = [coord[1] for coord in keys(state_coords)]
    y_coords = [coord[2] for coord in keys(state_coords)]
    counts = [state_coords[coord] for coord in keys(state_coords)]

    # Create scatter plot with size proportional to visitation
    p = scatter(x_coords, y_coords,
                markersize=counts .* 2,
                alpha=0.7,
                title="State Visitation Frequency",
                xlabel="State Feature 1",
                ylabel="State Feature 2",
                colorbar=true,
                markercolor=:viridis,
                markerstrokewidth=0)

    return p
end

"""
    create_training_dashboard(model::GFlowNetModel, history::TrainingHistory, trajectories::Vector{Trajectory})

Create a comprehensive dashboard showing training progress and results.
"""
function create_training_dashboard(model::GFlowNetModel, history::TrainingHistory, trajectories::Vector{Trajectory})
    # Create individual plots
    p1 = plot_training_progress(history)
    p2 = plot_reward_distribution(trajectories)
    p3 = plot_multiple_trajectories(trajectories; max_trajectories=5)
    p4 = plot_state_visitation(trajectories)

    # Combine into dashboard
    dashboard = plot(p1, p2, p3, p4,
                    layout=(2, 2),
                    size=(1200, 800),
                    plot_title="GFlowNet Training Dashboard")

    return dashboard
end

"""
    analyze_trajectories_visual(trajectories::Vector{Trajectory})

Create visual analysis of trajectory properties.
"""
function analyze_trajectories_visual(trajectories::Vector{Trajectory})
    if isempty(trajectories)
        return plot(title="No trajectories to analyze")
    end

    # Analyze trajectory lengths
    lengths = [length(traj.actions) for traj in trajectories if !isempty(traj.actions)]

    # Analyze rewards
    rewards = Float64[]
    for traj in trajectories
        if !isempty(traj.states)
            terminal_state = traj.states[end]
            if is_terminal_state(terminal_state)
                push!(rewards, reward(terminal_state))
            end
        end
    end

    # Create plots
    p1 = histogram(lengths,
                   title="Trajectory Lengths",
                   xlabel="Length",
                   ylabel="Frequency",
                   bins=min(20, maximum(lengths) - minimum(lengths) + 1),
                   alpha=0.7)

    p2 = histogram(rewards,
                   title="Terminal Rewards",
                   xlabel="Reward",
                   ylabel="Frequency",
                   bins=20,
                   alpha=0.7)

    p3 = scatter(lengths[1:min(length(lengths), length(rewards))],
                 rewards[1:min(length(lengths), length(rewards))],
                 title="Length vs Reward",
                 xlabel="Trajectory Length",
                 ylabel="Terminal Reward",
                 alpha=0.7)

    return plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))
end

# Export visualization functions
export plot_trajectory, plot_multiple_trajectories, plot_training_progress
export plot_reward_distribution, plot_state_visitation, create_training_dashboard
export analyze_trajectories_visual
