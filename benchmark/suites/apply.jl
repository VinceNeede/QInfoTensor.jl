# ------------------------------------------------------------------------
# apply benchmarks. Populates
#   SUITE["apply"]["circuit"][name]["alg=$alg"]["maxdim=$χ"]
# and
#   SUITE["apply"]["hamiltonian"][name]["alg=$alg"]["maxdim=$χ"]
# for alg in APPLY_ALGS (:zipup, :src — see benchmarks.jl).
#
# Inputs (sites/ψ0/H_odd/H_even, or sites/H/ψ0) are built once per problem,
# OUTSIDE the timed benchmark and shared across both algs — matching the
# pattern used throughout this suite, and valid here since the same input
# state/operator pair is exactly what a apples-to-apples :zipup-vs-:src
# comparison needs.
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

    for alg in APPLY_ALGS
        SUITE["apply"]["circuit"][problem.name]["alg=$alg"] = BenchmarkGroup()

        for χ in problem.maxdim_values
            SUITE["apply"]["circuit"][problem.name]["alg=$alg"]["maxdim=$χ"] =
                @benchmarkable(
                    run_circuit_trajectory($ψ0, $H_odd, $H_even, $(problem.n_steps);
                        alg=$alg, maxdim=$χ, cutoff=$(problem.cutoff))
                )
        end
    end
end

for problem in HAMAPPLY_PROBLEMS
    SUITE["apply"]["hamiltonian"][problem.name] = BenchmarkGroup()
    sites, H, ψ0 = build_hamapply_inputs(problem)

    for alg in APPLY_ALGS
        SUITE["apply"]["hamiltonian"][problem.name]["alg=$alg"] = BenchmarkGroup()

        for χ in problem.maxdim_values
            SUITE["apply"]["hamiltonian"][problem.name]["alg=$alg"]["maxdim=$χ"] =
                @benchmarkable(
                    run_hamapply($H, $ψ0; alg=$alg, maxdim=$χ, cutoff=$(problem.cutoff))
                )
        end
    end
end