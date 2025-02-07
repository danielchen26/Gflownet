using Documenter, GFlowNet

makedocs(
    sitename = "GFlowNet",
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md"
    ]
)

deploydocs(
    repo = "github.com/yourusername/GFlowNet.jl.git"
) 