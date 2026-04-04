using Documenter
using CosmoNEXUS

makedocs(;
    modules=[CosmoNEXUS],
    sitename="CosmoNEXUS.jl",
    format=Documenter.HTML(;
        canonical="https://ispirovjr.github.io/CosmoNEXUS.jl",
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/ispirovjr/CosmoNEXUS.jl",
    devbranch="main",
)
