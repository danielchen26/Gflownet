# HTML Report Generation for GFlowNet Examples
# This module provides a general template system for generating comprehensive HTML reports

using Dates
using Plots
using Statistics
using ..GFlowNet: AbstractState, Trajectory, reward

export generate_html_report, save_html_report
export ReportData, add_section!, add_plot!, add_table!, add_metrics!

"""
    ReportData

Structure to hold all data needed for generating an HTML report.
"""
mutable struct ReportData
    title::String
    description::String
    timestamp::String
    sections::Vector{Dict{String, Any}}
    plots::Vector{Dict{String, Any}}
    metrics::Dict{String, Any}
    tables::Vector{Dict{String, Any}}
    
    function ReportData(title::String, description::String="")
        new(title, description, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), 
            Dict{String, Any}[], Dict{String, Any}[], Dict{String, Any}(), Dict{String, Any}[])
    end
end

"""
    add_section!(report::ReportData, title::String, content::String)

Add a text section to the report.
"""
function add_section!(report::ReportData, title::String, content::String)
    push!(report.sections, Dict("title" => title, "content" => content))
end

"""
    add_plot!(report::ReportData, title::String, plot_path::String, description::String="")

Add a plot to the report.
"""
function add_plot!(report::ReportData, title::String, plot_path::String, description::String="")
    push!(report.plots, Dict("title" => title, "path" => plot_path, "description" => description))
end

"""
    add_table!(report::ReportData, title::String, headers::Vector{String}, rows::Vector{Vector{String}})

Add a table to the report.
"""
function add_table!(report::ReportData, title::String, headers::Vector{String}, rows::Vector{Vector{String}})
    push!(report.tables, Dict("title" => title, "headers" => headers, "rows" => rows))
end

"""
    add_metrics!(report::ReportData, metrics::Dict{String, Any})

Add metrics to the report.
"""
function add_metrics!(report::ReportData, metrics::Dict{String, Any})
    merge!(report.metrics, metrics)
end

"""
    generate_html_template()

Generate the base HTML template with CSS styling.
"""
function generate_html_template()
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}} - GFlowNet Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .timestamp {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
            font-style: italic;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9rem;
        }
        
        .section {
            background: white;
            margin-bottom: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .section-header {
            background: #f8f9fa;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid #dee2e6;
        }
        
        .section-header h2 {
            color: #333;
            font-size: 1.4rem;
        }
        
        .section-content {
            padding: 1.5rem;
        }
        
        .plot-container {
            text-align: center;
            margin: 1rem 0;
        }
        
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .plot-description {
            margin-top: 0.5rem;
            color: #666;
            font-style: italic;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        
        table th,
        table td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        
        table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #333;
        }
        
        table tr:hover {
            background-color: #f8f9fa;
        }
        
        .highlight {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 5px;
        }
        
        .success {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 5px;
        }
        
        .info {
            background-color: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 5px;
        }
        
        .footer {
            text-align: center;
            color: #666;
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid #dee2e6;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .metrics-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{TITLE}}</h1>
            <p>{{DESCRIPTION}}</p>
        </div>
        
        <div class="timestamp">
            Generated on {{TIMESTAMP}}
        </div>
        
        {{METRICS_SECTION}}
        
        {{SECTIONS}}
        
        <div class="footer">
            <p>Generated by GFlowNet.jl Report System</p>
        </div>
    </div>
</body>
</html>
"""
end

"""
    generate_metrics_html(metrics::Dict{String, Any})

Generate HTML for the metrics section.
"""
function generate_metrics_html(metrics::Dict{String, Any})
    if isempty(metrics)
        return ""
    end
    
    metrics_html = "<div class=\"metrics-grid\">\n"
    
    for (key, value) in metrics
        formatted_value = if isa(value, AbstractFloat)
            string(round(value, digits=3))
        else
            string(value)
        end
        
        # Format key for display
        display_key = replace(titlecase(string(key)), "_" => " ")
        
        metrics_html *= """
        <div class="metric-card">
            <div class="metric-value">$formatted_value</div>
            <div class="metric-label">$display_key</div>
        </div>
        """
    end
    
    metrics_html *= "</div>\n"
    return metrics_html
end

"""
    generate_sections_html(report::ReportData)

Generate HTML for all sections, plots, and tables.
"""
function generate_sections_html(report::ReportData)
    sections_html = ""
    
    # Add text sections
    for section in report.sections
        sections_html *= """
        <div class="section">
            <div class="section-header">
                <h2>$(section["title"])</h2>
            </div>
            <div class="section-content">
                $(section["content"])
            </div>
        </div>
        """
    end
    
    # Add plots section
    if !isempty(report.plots)
        sections_html *= """
        <div class="section">
            <div class="section-header">
                <h2>Visualizations</h2>
            </div>
            <div class="section-content">
        """
        
        for plot in report.plots
            sections_html *= """
            <div class="plot-container">
                <h3>$(plot["title"])</h3>
                <img src="$(plot["path"])" alt="$(plot["title"])">
                <div class="plot-description">$(plot["description"])</div>
            </div>
            """
        end
        
        sections_html *= """
            </div>
        </div>
        """
    end
    
    # Add tables section
    if !isempty(report.tables)
        for table in report.tables
            sections_html *= """
            <div class="section">
                <div class="section-header">
                    <h2>$(table["title"])</h2>
                </div>
                <div class="section-content">
                    <table>
                        <thead>
                            <tr>
            """
            
            for header in table["headers"]
                sections_html *= "<th>$header</th>"
            end
            
            sections_html *= """
                            </tr>
                        </thead>
                        <tbody>
            """
            
            for row in table["rows"]
                sections_html *= "<tr>"
                for cell in row
                    sections_html *= "<td>$cell</td>"
                end
                sections_html *= "</tr>"
            end
            
            sections_html *= """
                        </tbody>
                    </table>
                </div>
            </div>
            """
        end
    end
    
    return sections_html
end

"""
    generate_html_report(report::ReportData)

Generate the complete HTML report.
"""
function generate_html_report(report::ReportData)
    template = generate_html_template()
    
    # Replace placeholders
    html = replace(template, "{{TITLE}}" => report.title)
    html = replace(html, "{{DESCRIPTION}}" => report.description)
    html = replace(html, "{{TIMESTAMP}}" => report.timestamp)
    html = replace(html, "{{METRICS_SECTION}}" => generate_metrics_html(report.metrics))
    html = replace(html, "{{SECTIONS}}" => generate_sections_html(report))
    
    return html
end

"""
    save_html_report(report::ReportData, filepath::String)

Save the HTML report to a file.
"""
function save_html_report(report::ReportData, filepath::String)
    html = generate_html_report(report)
    
    # Ensure directory exists
    dir = dirname(filepath)
    if !isdir(dir)
        mkpath(dir)
    end
    
    # Write HTML file
    open(filepath, "w") do f
        write(f, html)
    end
    
    println("📄 HTML report saved to: $filepath")
    return filepath
end

"""
    create_grid_visualization(trajectories::Vector{<:Trajectory}, 
                             reward_positions::Dict{Tuple{Int,Int}, Float64},
                             grid_size::Int; 
                             save_path::String="grid_visualization.png")

Create a 2D grid visualization showing reward positions and trajectory paths.
"""
function create_grid_visualization(trajectories::Vector{<:Trajectory}, 
                                  reward_positions::Dict{Tuple{Int,Int}, Float64},
                                  grid_size::Int; 
                                  save_path::String="grid_visualization.png")
    
    # Create base grid
    grid_plot = plot(aspect_ratio=:equal, 
                    xlims=(0.5, grid_size + 0.5), 
                    ylims=(0.5, grid_size + 0.5),
                    title="Grid World - Rewards and Trajectories",
                    xlabel="X Position", 
                    ylabel="Y Position",
                    legend=:outertopright,
                    size=(600, 600))
    
    # Draw grid lines
    for i in 1:grid_size+1
        plot!(grid_plot, [i-0.5, i-0.5], [0.5, grid_size+0.5], color=:lightgray, alpha=0.5, label="")
        plot!(grid_plot, [0.5, grid_size+0.5], [i-0.5, i-0.5], color=:lightgray, alpha=0.5, label="")
    end
    
    # Plot reward positions
    for ((x, y), reward_val) in reward_positions
        color = reward_val >= 5.0 ? :red : :orange
        size = reward_val >= 5.0 ? 150 : 100
        scatter!(grid_plot, [x], [y], 
                marker=:star, 
                markersize=size/10, 
                color=color, 
                alpha=0.8,
                label=reward_val >= 5.0 ? "High Reward ($reward_val)" : "Medium Reward ($reward_val)")
    end
    
    # Plot trajectory paths
    path_colors = [:blue, :green, :purple, :brown, :pink]
    plotted_paths = 0
    
    for (i, traj) in enumerate(trajectories)
        if plotted_paths >= 10  # Limit number of paths shown
            break
        end
        
        if length(traj.states) > 1
            x_path = [state.x for state in traj.states]
            y_path = [state.y for state in traj.states]
            
            color = path_colors[mod(i-1, length(path_colors)) + 1]
            plot!(grid_plot, x_path, y_path, 
                 color=color, 
                 alpha=0.6, 
                 linewidth=2,
                 label=plotted_paths == 0 ? "Trajectory Paths" : "")
            
            # Mark start and end
            scatter!(grid_plot, [x_path[1]], [y_path[1]], 
                    marker=:circle, color=:green, markersize=6, alpha=0.8,
                    label=plotted_paths == 0 ? "Start" : "")
            scatter!(grid_plot, [x_path[end]], [y_path[end]], 
                    marker=:square, color=:red, markersize=6, alpha=0.8,
                    label=plotted_paths == 0 ? "End" : "")
            
            plotted_paths += 1
        end
    end
    
    # Save plot
    savefig(grid_plot, save_path)
    println("📊 Grid visualization saved to: $save_path")
    
    return grid_plot
end

"""
    create_reward_distribution_plot(rewards::Vector{Float64}; save_path::String="reward_distribution.png")

Create a reward distribution histogram.
"""
function create_reward_distribution_plot(rewards::Vector{Float64}; save_path::String="reward_distribution.png")
    p = histogram(rewards, 
                 bins=min(20, length(unique(rewards))),
                 title="Reward Distribution",
                 xlabel="Reward Value",
                 ylabel="Frequency",
                 color=:blue,
                 alpha=0.7,
                 legend=false,
                 size=(500, 400))
    
    # Add statistics
    mean_reward = mean(rewards)
    max_reward = maximum(rewards)
    vline!(p, [mean_reward], color=:red, linewidth=2, label="Mean: $(round(mean_reward, digits=2))")
    vline!(p, [max_reward], color=:green, linewidth=2, label="Max: $max_reward")
    
    savefig(p, save_path)
    println("📊 Reward distribution plot saved to: $save_path")
    
    return p
end

"""
    create_training_progress_plot(training_history::Dict; save_path::String="training_progress.png")

Create training progress visualization.
"""
function create_training_progress_plot(training_history::Dict; save_path::String="training_progress.png")
    if haskey(training_history, :losses) && !isempty(training_history[:losses])
        losses = training_history[:losses]
        p = plot(1:length(losses), losses,
                title="Training Progress",
                xlabel="Iteration",
                ylabel="Loss",
                color=:blue,
                linewidth=2,
                legend=false,
                size=(500, 400))
        
        savefig(p, save_path)
        println("📊 Training progress plot saved to: $save_path")
        return p
    else
        println("⚠️  No training loss data available for plotting")
        return nothing
    end
end 