# see documentation at https://juliadocs.github.io/Documenter.jl/stable/

using Documenter, LogDensityTestSuite

makedocs(
    modules = [LogDensityTestSuite],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Tamás K. Papp",
    sitename = "LogDensityTestSuite.jl",
    pages = Any["index.md"],
    clean = true,
    checkdocs = :exports,
)

# Some setup is needed for documentation deployment, see “Hosting Documentation” and
# deploydocs() in the Documenter manual for more information.
deploydocs(
    repo = "github.com/tpapp/LogDensityTestSuite.jl.git",
    push_preview = true
)
