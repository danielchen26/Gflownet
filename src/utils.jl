# src/utils.jl

# This file contains utility functions for the GFlowNet project.

"""
    split_data(data, ratio::Float64)

Split the data into training and validation sets based on the given ratio.

# Arguments
- `data`: The dataset to be split.
- `ratio::Float64`: The ratio of the training set size to the total data size.

# Returns
- `train_data`: The training set.
- `val_data`: The validation set.

# Examples
```julia
data = [(rand(10), rand(2)) for _ in 1:100]
train_data, val_data = split_data(data, 0.8)
```
"""
function split_data(data, ratio::Float64)
    train_size = floor(Int, length(data) * ratio)
    return data[1:train_size], data[(train_size + 1):end]
end

"""
    evaluate_model(model::Models.GFlowNet, data)

Evaluate the model on the given data.

# Arguments
- `model::Models.GFlowNet`: The GFlowNet model to be evaluated.
- `data`: The evaluation data.

# Returns
- `loss`: The average loss over the data.

# Examples
```julia
model = Models.GFlowNet(10, 64, 2)
val_data = [(rand(10), rand(2))]
loss = evaluate_model(model, val_data)
```
"""
function evaluate_model(model::Models.GFlowNet, data)
    loss_sum = 0.0
    for (x, y) in data
        prediction = model(x)
        loss_sum += Flux.mse(prediction, y)
    end
    return loss_sum / length(data)
end

# You can add additional utility functions or structures if needed
