using ..GFlowNet: GFlowNetModel, AbstractState, sample_trajectory, flow
using Statistics

"""
    entropy_estimator(model::GFlowNetModel, n_samples::Int=1000)

Estimate the entropy of the distribution represented by the GFlowNet.
"""
function entropy_estimator(model::GFlowNetModel, n_samples::Int=1000)
    total_log_prob = 0.0
    
    for _ in 1:n_samples
        # Sample a trajectory
        trajectory = sample_trajectory(model)
        
        # Get the final state
        final_state = trajectory.states[end]
        
        # Compute log probability of this state
        log_prob = log(reward(final_state)) - log(model.partition_function)
        
        total_log_prob += log_prob
    end
    
    # Estimate entropy: H(p) = -E[log p(x)]
    return -total_log_prob / n_samples
end

"""
    kl_divergence(model::GFlowNetModel, target_function, n_samples::Int=1000)

Estimate the KL divergence between the GFlowNet distribution and a target distribution.
"""
function kl_divergence(model::GFlowNetModel, target_function, n_samples::Int=1000)
    total_kl = 0.0
    
    for _ in 1:n_samples
        # Sample a trajectory
        trajectory = sample_trajectory(model)
        
        # Get the final state
        final_state = trajectory.states[end]
        
        # Compute log probability of this state under GFlowNet
        log_prob_model = log(reward(final_state)) - log(model.partition_function)
        
        # Compute log probability under target
        log_prob_target = log(target_function(final_state))
        
        # Add to KL estimate: KL(p||q) = E_p[log(p(x)/q(x))]
        total_kl += log_prob_model - log_prob_target
    end
    
    return total_kl / n_samples
end

"""
    mutual_information(model::GFlowNetModel, variable_indices1, variable_indices2, n_samples::Int=1000)

Estimate the mutual information between two subsets of variables in the GFlowNet.
"""
function mutual_information(model::GFlowNetModel, variable_indices1, variable_indices2, n_samples::Int=1000)
    # Sample states from GFlowNet
    samples = []
    for _ in 1:n_samples
        trajectory = sample_trajectory(model)
        final_state = trajectory.states[end]
        push!(samples, final_state)
    end
    
    # Extract variables
    X = [state_to_features(s)[variable_indices1] for s in samples]
    Y = [state_to_features(s)[variable_indices2] for s in samples]
    
    # Estimate mutual information using a simple binning approach
    # For continuous variables, a more sophisticated estimator would be needed
    
    # Bin the data
    X_bins = bin_data(X, 10)
    Y_bins = bin_data(Y, 10)
    
    # Compute empirical probabilities
    p_x = count_frequencies(X_bins)
    p_y = count_frequencies(Y_bins)
    p_xy = count_joint_frequencies(X_bins, Y_bins)
    
    # Compute mutual information: I(X;Y) = sum_x,y p(x,y) log(p(x,y)/(p(x)p(y)))
    mi = 0.0
    for (xy, p_joint) in p_xy
        x, y = xy
        p_marginal_x = p_x[x]
        p_marginal_y = p_y[y]
        
        mi += p_joint * log(p_joint / (p_marginal_x * p_marginal_y))
    end
    
    return mi
end

"""
    bin_data(data, n_bins)

Bin data into discrete categories for mutual information estimation.
"""
function bin_data(data, n_bins)
    result = []
    for x in data
        # Convert each data point to a bin index
        if length(x) == 1
            # 1D case
            push!(result, floor(Int, x[1] * n_bins))
        else
            # Multi-dimensional case
            push!(result, tuple([floor(Int, xi * n_bins) for xi in x]...))
        end
    end
    return result
end

"""
    count_frequencies(data)

Count the frequencies of each value in the data.
"""
function count_frequencies(data)
    counts = Dict()
    for x in data
        counts[x] = get(counts, x, 0) + 1
    end
    
    # Normalize to probabilities
    n = length(data)
    for (k, v) in counts
        counts[k] = v / n
    end
    
    return counts
end

"""
    count_joint_frequencies(data1, data2)

Count the joint frequencies of values in two datasets.
"""
function count_joint_frequencies(data1, data2)
    counts = Dict()
    for (x, y) in zip(data1, data2)
        counts[(x, y)] = get(counts, (x, y), 0) + 1
    end
    
    # Normalize to probabilities
    n = length(data1)
    for (k, v) in counts
        counts[k] = v / n
    end
    
    return counts
end 