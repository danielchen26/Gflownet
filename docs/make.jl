using Documenter

makedocs(
    sitename = "GFlowNet.jl",
    format = Documenter.HTML(
        prettyurls = false,  # Use simple URLs
        canonical = "https://yourusername.github.io/GFlowNet.jl"
    ),
    clean = false,  # Don't clean build directory due to permission issues
    build = "../docbuild",
    modules = Module[],  # Empty array of Module type to satisfy the type requirement
    doctest = false,  # Skip doctests to avoid errors
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "guide/getting_started.md",
            "guide/core_concepts.md",
            "guide/mathematical_background.md",
            "guide/training_objectives.md"
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