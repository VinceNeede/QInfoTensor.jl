using Dates
using PkgBenchmark
using QInfoTensor

# Runs benchmark/benchmarks.jl (PkgBenchmark's default script location)
# and writes a timestamped markdown report to results/. Run from anywhere
# — @__DIR__ resolves relative to this file, not the caller's pwd.

results = benchmarkpkg(QInfoTensor)

results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)

timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
outfile = joinpath(results_dir, "benchmark_$timestamp.md")

export_markdown(outfile, results)
println("Wrote $outfile")