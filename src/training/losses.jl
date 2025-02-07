module Losses

export trajectory_balance_loss

using Flux: logsoftmax

function trajectory_balance_loss(forward_logits, backward_logits, rewards)
    # Implementation from original training.jl
    log_Z = 0.0  # Placeholder for partition function
    log_pf = sum(logsoftmax(forward_logits))
    log_pb = sum(logsoftmax(backward_logits))
    loss = (log_Z + log_pf - log_pb - log.(rewards)).^2
    mean(loss)
end

end
