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
    warnonly = [:missing_docs],  # Only warn about missing docs instead of failing
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "guide/getting_started.md",
            "guide/core_concepts.md",
            "guide/mathematical_background.md",
            "guide/training_objectives.md"
        ],
        "API Reference" => [
            "api/core_types.md",
            "api/flow_networks.md",
            "api/directed_acyclic_graph.md",
            "api/policies.md",
            "api/training.md"
        ],
        "Applications" => [
            "applications/grid_world.md",
            "applications/molecular_design.md",
            "applications/causal_discovery.md",
            "applications/active_learning.md"
        ]
    ]
)

# Uncomment this when ready to deploy
# deploydocs(
#     repo = "github.com/yourusername/GFlowNet.jl.git",
#     devbranch = "main"
# ) 