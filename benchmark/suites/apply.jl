# ------------------------------------------------------------------------
# apply benchmarks. Populates
#   SUITE["apply"]["circuit"][name]["alg=$alg"]["maxdim=$χ"]
#   SUITE["apply"]["hamiltonian"][name]["alg=$alg"]["maxdim=$χ"]
#   SUITE["apply"]["random"][name]["alg=$alg"]["maxdim=$χ"]
# for alg in APPLY_ALGS (:zipup, :src, :densitymatrix — see benchmarks.jl).
#
# "random" uses RANDOM_APPLY_PROBLEMS (random_apply.jl): synthetic
# random MPO/MPS with large D=χ=50, mirroring the SRC paper's own Figure
# 1/section 4.4.2 setup. This is the ONLY problem family here with
# D,χ ≫ 1 and steep compression (χ̄ ≪ D·χ) — the regime the paper's own
# results (and ours) show SRC's asymptotic advantage over the OTHER
# algorithms actually shows up in. circuit/hamiltonian both have MPO bond
# dimension D≤4, too small to see a difference (see chat) — kept for
# their own regression-tracking value (now across all three algs), not
# as a test of SRC's claimed speed advantage.
#
# Inputs (sites/ψ0/H_odd/H_even, or sites/H/ψ0, or H/ψ0) are built once
# per problem, OUTSIDE the timed benchmark and shared across both algs —
# matching the pattern used throughout this suite, and valid here since
# the same input state/operator pair is exactly what an apples-to-apples
# :zipup-vs-:src comparison needs.
#
# Speed/memory regression tracking only — the accuracy-vs-maxdim curve is
# a separate, non-blocking report (not written yet).
# ------------------------------------------------------------------------

SUITE["apply"] = BenchmarkGroup()
SUITE["apply"]["circuit"] = BenchmarkGroup()
SUITE["apply"]["hamiltonian"] = BenchmarkGroup()
SUITE["apply"]["random"] = BenchmarkGroup()

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

for problem in RANDOM_APPLY_PROBLEMS
    SUITE["apply"]["random"][problem.name] = BenchmarkGroup()
    H, ψ0 = build_random_apply_inputs(problem)

    for alg in APPLY_ALGS
        # CRITICAL: Skip densitymatrix for large random problems to avoid fundamental OOM.
        # This regime is kept to measure the speed scaling between :zipup and :src.
        alg == :densitymatrix && continue

        SUITE["apply"]["random"][problem.name]["alg=$alg"] = BenchmarkGroup()

        for χ in problem.maxdim_values
            @benchmarkable(
                run_random_apply($H, $ψ0; alg=$alg, maxdim=$χ)
            )
        end
    end
end