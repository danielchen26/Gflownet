using Dates
using Statistics

"""
    GFlowNetLogger

A logger for tracking and recording GFlowNet training progress and performance.
"""
struct GFlowNetLogger
    log_file::Union{String, Nothing}
    log_frequency::Int
    verbose::Bool
    metrics::Dict{String, Vector{Float64}}
    start_time::DateTime
    iteration::Ref{Int}
end

"""
    GFlowNetLogger(log_file=nothing; log_frequency=10, verbose=true)

Create a new GFlowNet logger.
"""
function GFlowNetLogger(log_file=nothing; log_frequency=10, verbose=true)
    if !isnothing(log_file)
        # Create log file with header
        open(log_file, "w") do io
            println(io, "timestamp,iteration,loss,reward_mean,reward_std,flow_consistency,time_elapsed")
        end
    end
    
    return GFlowNetLogger(
        log_file,
        log_frequency,
        verbose,
        Dict{String, Vector{Float64}}(),
        now(),
        Ref(0)
    )
end

"""
    log_metric!(logger::GFlowNetLogger, metric_name::String, value::Real)

Log a metric value.
"""
function log_metric!(logger::GFlowNetLogger, metric_name::String, value::Real)
    if !haskey(logger.metrics, metric_name)
        logger.metrics[metric_name] = Float64[]
    end
    
    push!(logger.metrics[metric_name], Float64(value))
    
    return logger
end

"""
    log_iteration!(logger::GFlowNetLogger, loss::Real; 
                  reward_mean=nothing, reward_std=nothing, flow_consistency=nothing)

Log metrics for a training iteration.
"""
function log_iteration!(logger::GFlowNetLogger, loss::Real; 
                       reward_mean=nothing, reward_std=nothing, flow_consistency=nothing)
    # Increment iteration counter
    logger.iteration[] += 1
    
    # Log metrics
    log_metric!(logger, "loss", loss)
    
    if !isnothing(reward_mean)
        log_metric!(logger, "reward_mean", reward_mean)
    end
    
    if !isnothing(reward_std)
        log_metric!(logger, "reward_std", reward_std)
    end
    
    if !isnothing(flow_consistency)
        log_metric!(logger, "flow_consistency", flow_consistency)
    end
    
    # Log to file if needed
    if !isnothing(logger.log_file) && (logger.iteration[] % logger.log_frequency == 0)
        open(logger.log_file, "a") do io
            time_elapsed = (now() - logger.start_time).value / 1000.0  # in seconds
            
            # Write log line
            println(io, string(now()), ",",
                   logger.iteration[], ",",
                   get(logger.metrics, "loss", [NaN])[end], ",",
                   get(logger.metrics, "reward_mean", [NaN])[end], ",",
                   get(logger.metrics, "reward_std", [NaN])[end], ",",
                   get(logger.metrics, "flow_consistency", [NaN])[end], ",",
                   time_elapsed)
        end
    end
    
    # Print to console if verbose
    if logger.verbose && (logger.iteration[] % logger.log_frequency == 0)
        time_elapsed = (now() - logger.start_time).value / 1000.0  # in seconds
        
        @info "Iteration $(logger.iteration[]) ($(round(time_elapsed, digits=2))s)" loss reward_mean reward_std flow_consistency
    end
    
    return logger
end

"""
    get_metric(logger::GFlowNetLogger, metric_name::String)

Get the full history of a metric.
"""
function get_metric(logger::GFlowNetLogger, metric_name::String)
    return get(logger.metrics, metric_name, Float64[])
end

"""
    get_last_metric(logger::GFlowNetLogger, metric_name::String)

Get the most recent value of a metric.
"""
function get_last_metric(logger::GFlowNetLogger, metric_name::String)
    values = get_metric(logger, metric_name)
    return isempty(values) ? NaN : values[end]
end

"""
    reset!(logger::GFlowNetLogger)

Reset the logger, clearing all stored metrics.
"""
function reset!(logger::GFlowNetLogger)
    empty!(logger.metrics)
    logger.iteration[] = 0
    logger.start_time = now()
    
    return logger
end

"""
    save_metrics(logger::GFlowNetLogger, filename::String)

Save all metrics to a CSV file.
"""
function save_metrics(logger::GFlowNetLogger, filename::String)
    open(filename, "w") do io
        # Write header
        metric_names = keys(logger.metrics)
        println(io, "iteration,", join(metric_names, ","))
        
        # Write data rows
        max_length = maximum(length(vals) for vals in values(logger.metrics))
        
        for i in 1:max_length
            row = ["$i"]
            for name in metric_names
                vals = logger.metrics[name]
                val = i <= length(vals) ? string(vals[i]) : "NaN"
                push!(row, val)
            end
            println(io, join(row, ","))
        end
    end
end

"""
    summarize_performance(model, n_samples=100)

Compute and return various performance metrics for a GFlowNet model.
"""
function summarize_performance(model, n_samples=100)
    # Sample trajectories
    trajectories = [sample_trajectory(model) for _ in 1:n_samples]
    
    # Extract terminal states
    terminal_states = [trajectory.states[end] for trajectory in trajectories]
    
    # Compute rewards
    rewards = [reward(state) for state in terminal_states]
    
    # Compute statistics
    reward_mean = mean(rewards)
    reward_std = std(rewards)
    reward_min = minimum(rewards)
    reward_max = maximum(rewards)
    
    # Estimate partition function
    Z_estimate = sum(rewards) / n_samples
    
    # Check flow consistency (if flow estimator exists)
    flow_consistency = NaN
    if !isnothing(model.flow_estimator)
        # Calculate flow consistency as relative error between
        # estimated total flow and sum of terminal rewards
        total_flow = flow(model, model.dag.initial_state)
        total_reward = sum(reward(s) for s in model.dag.terminal_states)
        flow_consistency = abs(total_flow - total_reward) / total_reward
    end
    
    # Return metrics as a dictionary
    return Dict(
        "reward_mean" => reward_mean,
        "reward_std" => reward_std,
        "reward_min" => reward_min,
        "reward_max" => reward_max,
        "Z_estimate" => Z_estimate,
        "flow_consistency" => flow_consistency,
        "n_samples" => n_samples
    )
end

"""
    time_execution(f::Function)

Time the execution of a function and return the result and elapsed time.
"""
function time_execution(f::Function)
    start_time = now()
    result = f()
    elapsed_time = (now() - start_time).value / 1000.0  # in seconds
    return result, elapsed_time
end

"""
    benchmark_sampling(model, n_samples=100)

Benchmark the sampling performance of a GFlowNet model.
"""
function benchmark_sampling(model, n_samples=100)
    # Measure time to sample trajectories
    _, sampling_time = time_execution() do
        [sample_trajectory(model) for _ in 1:n_samples]
    end
    
    # Average time per sample
    avg_time_per_sample = sampling_time / n_samples
    
    return Dict(
        "total_sampling_time" => sampling_time,
        "avg_time_per_sample" => avg_time_per_sample,
        "samples_per_second" => n_samples / sampling_time,
        "n_samples" => n_samples
    )
end 