#!/usr/bin/env julia

# Test script for the composition-based type structure

println("Loading packages...")
using Pkg
Pkg.activate(".")

# This should load all our components
using GFlowNet
using Graphs

println("Creating initial state...")
# Create a molecule data structure
molecule_data = MoleculeData(Symbol[], Tuple{Int, Int, Int}[])
initial_state = MoleculeState(molecule_data, false)

println("Creating terminal state...")
# Create a terminal state
terminal_data = MoleculeData(Symbol[:C, :H, :O], [(1, 2, 1), (1, 3, 1)])
terminal_state = MoleculeState(terminal_data, true)

println("Creating sink state...")
# Create a terminal sink
sink_data = MoleculeData(Symbol[:SINK], Tuple{Int, Int, Int}[])
sink_state = MoleculeState(sink_data, true)

println("Creating actions...")
# Create some actions
actions = AbstractAction[
    AddAtomAction(:C, (1.0, 1.0, 1.0)),
    AddAtomAction(:H, (2.0, 1.0, 1.0)),
    AddAtomAction(:O, (1.0, 2.0, 1.0)),
    AddBondAction(1, 2, 1),
    AddBondAction(1, 3, 1),
    TerminateMoleculeAction()
]

println("Testing is_applicable...")
# Test if actions are applicable
for action in actions
    println("  $(typeof(action)) is applicable: $(is_applicable(action, initial_state))")
end

println("Testing apply_action...")
# Test applying an action
if length(actions) > 0
    action = actions[1]
    if is_applicable(action, initial_state)
        new_state = apply_action(action, initial_state)
        println("  Applied $(typeof(action)) to initial state")
        println("  New state: $(new_state)")
    else
        println("  Action not applicable")
    end
end

println("Testing DAG creation...")
# Try to create a DAG
try
    dag = create_dag(initial_state, [terminal_state], sink_state, actions)
    println("  DAG created successfully with $(nv(dag.graph)) vertices and $(ne(dag.graph)) edges")
catch e
    println("  Error creating DAG: $(e)")
end

println("Tests complete") 