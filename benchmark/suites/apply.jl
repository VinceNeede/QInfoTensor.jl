# ------------------------------------------------------------------------
# apply benchmarks. Populates SUITE["apply"]["circuit"][name]["maxdim=$χ"]
# and SUITE["apply"]["hamiltonian"][name]["maxdim=$χ"].
#
# Inputs (sites/ψ0/H_odd/H_even, or sites/H/ψ0) are built once, outside
# the timed benchmark, matching the pattern used throughout this suite.
#
# Speed/memory regression tracking only — the accuracy-vs-maxdim curve is
# a separate, non-blocking report (not written yet).
# ------------------------------------------------------------------------

SUITE["apply"] = BenchmarkGroup()
SUITE["apply"]["circuit"] = BenchmarkGroup()
SUITE["apply"]["hamiltonian"] = BenchmarkGroup()

for problem in CIRCUIT_PROBLEMS
    SUITE["apply"]["circuit"][problem.name] = BenchmarkGroup()
    sites, ψ0, H_odd, H_even = build_circuit_inputs(problem)

    for χ in problem.maxdim_values
        SUITE["apply"]["circuit"][problem.name]["maxdim=$χ"] =
            @benchmarkable(
                run_circuit_trajectory($ψ0, $H_odd, $H_even, $(problem.n_steps);
                    maxdim=$χ, cutoff=$(problem.cutoff))
            )
    end
end

for problem in HAMAPPLY_PROBLEMS
    SUITE["apply"]["hamiltonian"][problem.name] = BenchmarkGroup()
    sites, H, ψ0 = build_hamapply_inputs(problem)

    for χ in problem.maxdim_values
        SUITE["apply"]["hamiltonian"][problem.name]["maxdim=$χ"] =
            @benchmarkable(
                run_hamapply($H, $ψ0; maxdim=$χ, cutoff=$(problem.cutoff))
            )
    end
end