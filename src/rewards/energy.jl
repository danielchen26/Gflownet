module EnergyRewards

export EnergyBasedReward

using Flux

struct EnergyBasedReward
    energy_network::Chain
end

function (r::EnergyBasedReward)(state)
    -logsumexp(r.energy_network(state))
end

end
