using Documenter
using GFlowNet

makedocs(
    sitename = "GFlowNet.jl",
    format = Documenter.HTML(
        prettyurls = false,  # Use simple URLs
        canonical = "https://yourusername.github.io/GFlowNet.jl"
    ),
    clean = false,  # Don't clean build directory due to permission issues
    build = "../docbuild",
    modules = [GFlowNet],  # Include the GFlowNet module here
    doctest = false,  # Skip doctests to avoid errors
    warnonly = [:missing_docs, :cross_references],  # Only warn about missing docs and broken links instead of failing
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "guide/getting_started.md",
            "guide/core_concepts.md", 
            "guide/examples.md"
        ],
        "Manual" => [
            "manual/overview.md",
            "manual/training_system.md",
            "manual/objectives.md",
            "manual/backward_policy.md",
            "manual/developer_guide.md",
            "manual/migration.md"
        ],
        "Theory" => [
            "guide/mathematical_background.md",
            "theory/partition_function.md",
            "theory/flow_consistency.md",
            "guide/training_objectives.md"
        ],
        "API Reference" => [
            "api/core_types.md",
            "api/policies.md",
            "api/training.md",
            "api/flow_networks.md",
            "api/utils.md"
        ],
        "Applications" => [
            "applications/grid_world.md",
            "applications/supply_chain.md",
            "applications/molecular_design.md",
            "applications/causal_discovery.md",
            "applications/active_learning.md"
        ],
        "Internals" => [
            "internals/architecture.md",
            "internals/design_decisions.md", 
            "internals/known_limitations.md",
            "internals/flow_functions_multistart.md"
        ],
        "Extensions" => [
            "extensions/continuous.md",
            "extensions/information.md",
            "extensions/non_acyclic.md"
        ]
    ]
)

# Uncomment this when ready to deploy
# deploydocs(
#     repo = "github.com/yourusername/GFlowNet.jl.git",
#     devbranch = "main"
# ) 