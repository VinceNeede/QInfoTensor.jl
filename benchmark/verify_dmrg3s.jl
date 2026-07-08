# ---------------------------------------------------------------------------
# verify_dmrg3s.jl — nsite=1 (:dmrg3s) vs nsite=2 (:twosite) energy/bond-dim
# agreement at the DMRG_PROBLEMS scale (L=20), not the tiny L=6 ED check.
# ---------------------------------------------------------------------------
#
# compare_itensor.jl's timing comparison only ever exercises nsite=2 against
# ITensor — it says nothing about whether nsite=1 (specifically its
# ER/right-environment-based backward branch, the least-verified piece of
# this whole DMRG implementation) is actually converging to the same answer
# at this scale, only that it's faster. test_dmrg.jl's ED-based check
# covers correctness but only at L=6 — small enough that it might not
# stress whatever L=6 doesn't stress (larger bond dimension, more
# sweeps-to-converge, many more backward-branch invocations).
#
# This is a correctness check, not a timing one, so it deliberately does
# NOT use `@benchmarkable`/SUITE — there's no natural way to assert on a
# return value there. One untimed run per problem/maxdim instead, same
# shape as compare_itensor.jl's own untimed "verification pass" alongside
# its timed @benchmark calls, and the same "separate correctness script,
# not entangled with timing" spirit as verify_circuit_equivalence.jl.
#
# Also exercises `_default_noise` for the first time outside test_dmrg.jl's
# L=6 tests: `noise` is left unspecified below, so nsite=1 uses the new
# default schedule rather than a hand-picked one.

using TensorKit
using QInfoTensor
using Printf

include(joinpath(@__DIR__, "problems.jl"))

const FIXED_TOL = 1e-12
const ENERGY_ATOL = 1e-6

struct DMRG3SVerifyRow
    name::String
    maxdim::Int
    twosite_energy::Float64
    twosite_maxlinkdim::Int
    dmrg3s_energy::Float64
    dmrg3s_maxlinkdim::Int
    energy_diff::Float64
    passed::Bool
end

function _verify_dmrg3s(problem::DMRGProblem)
    println("Verifying dmrg3s vs twosite: $(problem.name) ...")
    sites, H, ψ0 = build_dmrg_inputs(problem)

    rows = DMRG3SVerifyRow[]
    for χ in problem.maxdim_values
        @info "  maxdim=$χ: running :twosite ..."
        ψ_twosite = copy(ψ0)
        _, _, sd_twosite = dmrg!(ψ_twosite, H, problem.nsweeps;
            nsite=2, maxdim=χ, cutoff=problem.cutoff, eigsolve_tol=FIXED_TOL)
        E_twosite = sd_twosite[end].energies[end]
        dim_twosite = QInfoTensor.maxlinkdim(ψ_twosite)

        @info "  maxdim=$χ: running :dmrg3s (default noise schedule) ..."
        ψ_dmrg3s = copy(ψ0)
        _, _, sd_dmrg3s = dmrg!(ψ_dmrg3s, H, problem.nsweeps;
            nsite=1, maxdim=χ, cutoff=problem.cutoff, eigsolve_tol=FIXED_TOL)
        E_dmrg3s = sd_dmrg3s[end].energies[end]
        dim_dmrg3s = QInfoTensor.maxlinkdim(ψ_dmrg3s)

        diff = abs(E_twosite - E_dmrg3s)
        passed = diff < ENERGY_ATOL

        push!(rows, DMRG3SVerifyRow(problem.name, χ, E_twosite, dim_twosite,
                                     E_dmrg3s, dim_dmrg3s, diff, passed))
    end
    return rows
end

function _print_table(rows)
    println("\n=== :dmrg3s vs :twosite — energy/bond-dim agreement (L=20 scale) ===")
    @printf("%-24s %6s %14s %8s %14s %8s %12s %6s\n",
        "problem", "maxdim", "twosite E", "ts dim", "dmrg3s E", "d3s dim", "|ΔE|", "pass")
    for r in rows
        @printf("%-24s %6d %14.6f %8d %14.6f %8d %12.2e %6s\n",
            r.name, r.maxdim, r.twosite_energy, r.twosite_maxlinkdim,
            r.dmrg3s_energy, r.dmrg3s_maxlinkdim, r.energy_diff, r.passed ? "PASS" : "FAIL")
    end
end

function _write_markdown_report(path, rows)
    open(path, "w") do io
        println(io, "# dmrg3s vs twosite verification report\n")
        println(io, "| problem | maxdim | twosite E | twosite dim | dmrg3s E | dmrg3s dim | \\|ΔE\\| | pass |")
        println(io, "|---|---|---|---|---|---|---|---|")
        for r in rows
            @printf(io, "| %s | %d | %.6f | %d | %.6f | %d | %.2e | %s |\n",
                r.name, r.maxdim, r.twosite_energy, r.twosite_maxlinkdim,
                r.dmrg3s_energy, r.dmrg3s_maxlinkdim, r.energy_diff, r.passed ? "PASS" : "FAIL")
        end
    end
end

# ---------------------------------------------------------------------------

all_rows = reduce(vcat, (_verify_dmrg3s(p) for p in DMRG_PROBLEMS))
_print_table(all_rows)

mkpath(joinpath(@__DIR__, "results"))
_write_markdown_report(joinpath(@__DIR__, "results", "verify_dmrg3s.md"), all_rows)
println("\nSaved: benchmark/results/verify_dmrg3s.md")

n_failed = count(!(r.passed) for r in all_rows)
if n_failed > 0
    error("$n_failed/$(length(all_rows)) dmrg3s-vs-twosite comparisons disagree beyond atol=$ENERGY_ATOL — " *
          "do not trust :dmrg3s timing numbers (or its energies) at this scale until this is resolved.")
end
println("\nAll $(length(all_rows)) comparisons agree within atol=$ENERGY_ATOL — PASS")
