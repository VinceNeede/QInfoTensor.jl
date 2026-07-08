# suites/dmrg.jl
# ------------------------------------------------------------------------
# DMRG benchmarks using parameters provisioned by DMRG_PROBLEMS. Populates:
#   SUITE["dmrg"][name]["nsite=$nsite"]["maxdim=$χ"]
# ------------------------------------------------------------------------

using Logging: with_logger, NullLogger

SUITE["dmrg"] = BenchmarkGroup()

for problem in DMRG_PROBLEMS
    SUITE["dmrg"][problem.name] = BenchmarkGroup()
    sites, H, ψ0 = build_dmrg_inputs(problem)

    for nsite in (1, 2)
        SUITE["dmrg"][problem.name]["nsite=$nsite"] = BenchmarkGroup()

        for χ in problem.maxdim_values
            # dmrg! mutates ψ in-place; setup cleanly refreshes the initial state on every trial.
            # with_logger(NullLogger()) suppresses the internal @info logs to eliminate I/O overhead.
            SUITE["dmrg"][problem.name]["nsite=$nsite"]["maxdim=$χ"] =
                @benchmarkable(
                    with_logger(NullLogger()) do
                        dmrg!(ψ, $H, $(problem.nsweeps); nsite=$nsite, maxdim=$χ, cutoff=$(problem.cutoff))
                    end,
                    setup=(ψ = copy($ψ0))
                )
        end
    end
end