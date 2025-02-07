## ============= activate environment =============
cd(@__DIR__)
using Pkg
Pkg.activate(".")



## ============= Load packages =============
using Flux, Random, Distributions

## ============= Simulated Dataset Generation =============
function simulate_data(num_experiments)
    data = Dict()
    for i in 1:num_experiments
        exp_name = "Experiment_" * string(i)
        data[exp_name] = (rand(Uniform(0.5, 10.0)), rand(["Feature_" * string(j) for j in 1:rand(1:5)]))
    end
    return data
end

# Experiments and initial state
experiments = simulate_data(10)

# State representation
struct State
    evidence::Dict{String, Bool}
    cost::Float64
end

# Initialize the state
initial_state = State(Dict(exp => false for exp in keys(experiments)), 0.0)

# GFlowNet Model
struct GFlowNet
    layer1::Dense
    layer2::Dense
end
# Define how Flux finds the trainable parameters in your model
Flux.trainable(m::GFlowNet) = (m.layer1, m.layer2)

struct GFlowNetModel
    layer1::Flux.Dense
    layer2::Flux.Dense

    # Inner constructor
    function GFlowNetModel(input_dim::Int, hidden_dim::Int, output_dim::Int)
        layer1 = Flux.Dense(input_dim, hidden_dim, Flux.relu)
        layer2 = Flux.Dense(hidden_dim, output_dim)
        new(layer1, layer2)
    end
end


input_dim = length(keys(initial_state.evidence)) * 2  # Assuming binary feature representation
hidden_dim = 128  # Example hidden dimension size
output_dim = length(experiments)  # Number of possible actions
policy_network = GFlowNetModel(input_dim, hidden_dim, output_dim)
# Flow Calculation Function

function calculate_flow(current_state::State, policy_network)
    # Convert current state to input format for policy network
    network_input = state_to_input(current_state)  # You need to define this function

    # Get probabilities for each experiment
    experiment_probabilities = policy_network(network_input)

    # Calculate flow for each possible next state
    flows = Dict()
    for (experiment, (cost, features)) in experiments
        # Assume each experiment leads to a distinct next state
        next_state = update_state(current_state, experiment, features)  # Define this function
        flow_to_next_state = experiment_probabilities[experiment]
        flows[next_state] = flow_to_next_state
    end

    return flows
end

# TD Loss Function
function td_loss(trajectory, model, Z)
    loss = 0.0
    for t in 1:length(trajectory)-1
        current_state = trajectory[t]
        next_state = trajectory[t+1]
        prob = model([current_state, next_state])
        r = reward(next_state)
        loss += sum((calculate_flow(current_state, model) * prob - r ./ Z) .^ 2)
    end
    return loss
end

# Reward Function
function reward(state::State)
    info_gain = sum([state.evidence[exp] ? 1.0 : 0.0 for exp in keys(state.evidence)])
    cost = state.cost
    return [info_gain, -cost]
end

# Trajectory Generation
function generate_trajectory(policy_network, initial_state, experiments)
    state = initial_state
    trajectory = [state]
    while !all(state.evidence)
        possible_actions = [exp for exp in keys(experiments) if !state.evidence[exp]]
        chosen_action = rand(possible_actions)  # Improved action selection based on policy network
        cost, assays = experiments[chosen_action]
        state = State(Dict(exp => state.evidence[exp] || exp == chosen_action for exp in keys(experiments)), state.cost + cost)
        push!(trajectory, state)
    end
    return trajectory
end

# Training Loop
optimizer = ADAM(params(policy_network))
number_of_episodes = 100
recent_rewards = []
initial = initial_state
Z = 1.0  # Initial estimate of Z

for episode in 1:number_of_episodes
    trajectory = generate_trajectory(policy_network, initial, experiments)
    traj_reward = reward(trajectory[end])
    push!(recent_rewards, traj_reward)

    if episode % 10 == 0
        Z = mean(recent_rewards)
        recent_rewards = []
    end

    loss = td_loss(trajectory, policy_network, Z)
    Flux.train!(loss, params(policy_network), [(trajectory,)], optimizer)
end

println("Training complete")
