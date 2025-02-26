using Documenter

makedocs(
    sitename = "GFlowNet.jl",
    format = Documenter.HTML(
        prettyurls = false,  # Use simple URLs
        canonical = "https://yourusername.github.io/GFlowNet.jl"
    ),
    clean = false,  # Don't clean build directory due to permission issues
    build = "../docbuild",
    modules = [Main],  # We'll change this to your module name once package is importable
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "guide/getting_started.md",
            "guide/core_concepts.md",
            "guide/mathematical_background.md",
            "guide/training_objectives.md"
        ]
    ]
)

# Uncomment this when ready to deploy
# deploydocs(
#     repo = "github.com/yourusername/GFlowNet.jl.git",
#     devbranch = "main"
# ) 