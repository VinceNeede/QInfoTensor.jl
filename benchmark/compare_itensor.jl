# ---------------------------------------------------------------------------
# QInfoTensor vs ITensor comparison — circuit/apply only, now across BOTH
# implemented apply! algorithms (:zipup, :densitymatrix). Standalone
# script, NOT part of the PkgBenchmark SUITE (ITensor doesn't change
# across QInfoTensor commits, so PkgBenchmark's judge/compare isn't the
# right mechanism for this).
#
# :src is deliberately excluded here: it has no ITensor-side equivalent
# to compare against (Camaño, Epperly & Tropp 2025 is not implemented in
# ITensorMPS), so there's nothing to run this comparison against for that
# algorithm.
#
# No DMRG section: QInfoTensor has no DMRG implementation yet (DMRG3S is
# still on the roadmap, see design_notes.md) — the original reference
# script's DMRG comparison is dropped entirely rather than ported
# speculatively.
#
# Qualified names: QInfoTensor and ITensorMPS both export `siteinds`/
# `MPO`/`random_mps`-like names — with both packages loaded these would be
# ambiguous at global scope, so ITensor-side calls stay qualified
# (`ITensorMPS.X`) throughout, even inside included files.
#
# Usage:
#   julia --project=benchmark benchmark/compare_itensor.jl
# ---------------------------------------------------------------------------
using MKL
MKL.set_num_threads(1)
using Strided
Strided.set_num_threads(1)
using BenchmarkTools
using TensorKit
using QInfoTensor
using ITensors, ITensorMPS
using Printf

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "circuit.jl"))
include(joinpath(@__DIR__, "itensor_circuit.jl"))

const COMPARISON_ALGS = (:zipup, :densitymatrix)

struct CircuitComparisonRow
    name::String
    alg::Symbol
    maxdim::Int
    qit_time_ms::Float64
    qit_memory_mib::Float64
    qit_maxlinkdim::Int
    it_time_ms::Float64
    it_memory_mib::Float64
    it_maxlinkdim::Int
end

function _compare_circuit(problem::CircuitProblem)
    println("Benchmarking $(problem.name) ...")
    sites, ψ0, H_odd, H_even = build_circuit_inputs(problem)
    it_sites, it_ψ0, it_H_odd, it_H_even = build_itensor_circuit_inputs(problem)

    rows = CircuitComparisonRow[]
    for alg in COMPARISON_ALGS, χ in problem.maxdim_values
        # QInfoTensor side: run_circuit_trajectory's current signature takes
        # only alg/maxdim/cutoff — no sweep_maxdim/sweep_cutoff (removed
        # since an earlier version of this file; :zipup now computes its
        # own internal sweep defaults, and :densitymatrix never had such a
        # concept — passing them here would be a MethodError).
        qit_trial = @benchmark(
            run_circuit_trajectory($ψ0, $H_odd, $H_even, $(problem.n_steps);
                alg=$alg, maxdim=$χ, cutoff=$(problem.cutoff)),
        )
        # ITensor side: sweep_maxdim/sweep_cutoff ARE still meaningful here
        # (ITensor's own zipup genuinely has the two-tier split — see
        # run_itensor_circuit_trajectory's docstring) and are simply
        # ignored internally when alg=:densitymatrix, so passing them
        # unconditionally for both algs is safe.
        it_trial = @benchmark(
            run_itensor_circuit_trajectory($it_ψ0, $it_H_odd, $it_H_even, $(problem.n_steps);
                alg=$alg, maxdim=$χ, cutoff=$(problem.cutoff),
                sweep_maxdim=$(2χ), sweep_cutoff=$(problem.cutoff / 10)),
        )
        qit_m, it_m = median(qit_trial), median(it_trial)

        # @benchmark discards the function's return value, so the actual
        # OUTPUT state needs one extra, untimed call per row to inspect its
        # bond dimension — negligible cost next to the benchmark itself.
        # This is exactly the check that caught the p=2/p=1 truncerror bug
        # (see chat): making it a permanent column means a future
        # regression like that shows up directly in this table instead of
        # needing a separate diagnostic script.
        ψ_qit = run_circuit_trajectory(ψ0, H_odd, H_even, problem.n_steps;
            alg, maxdim=χ, cutoff=problem.cutoff)
        ψ_it = run_itensor_circuit_trajectory(it_ψ0, it_H_odd, it_H_even, problem.n_steps;
            alg, maxdim=χ, cutoff=problem.cutoff, sweep_maxdim=2χ, sweep_cutoff=problem.cutoff / 10)

        # Qualified: both QInfoTensor and ITensorMPS export `maxlinkdim`,
        # ambiguous unqualified with both packages loaded (same reasoning
        # as this file's other qualified ITensor-side calls — see header).
        qit_dim = QInfoTensor.maxlinkdim(ψ_qit)
        it_dim = ITensorMPS.maxlinkdim(ψ_it)

        push!(rows, CircuitComparisonRow(
            problem.name, alg, χ,
            time(qit_m) / 1e6, memory(qit_m) / 2^20, qit_dim,
            time(it_m) / 1e6, memory(it_m) / 2^20, it_dim,
        ))
    end
    return rows
end

circuit_rows = reduce(vcat, (_compare_circuit(p) for p in CIRCUIT_PROBLEMS))

function _print_circuit_table(rows)
    @printf("%-16s %-14s %8s %12s %12s %10s %12s %12s %10s\n",
        "problem", "alg", "maxdim", "QIT time(ms)", "QIT mem(MiB)", "QIT dim",
        "IT time(ms)", "IT mem(MiB)", "IT dim")
    for r in rows
        @printf("%-16s %-14s %8d %12.2f %12.2f %10d %12.2f %12.2f %10d\n",
            r.name, string(r.alg), r.maxdim, r.qit_time_ms, r.qit_memory_mib, r.qit_maxlinkdim,
            r.it_time_ms, r.it_memory_mib, r.it_maxlinkdim)
    end
end

_print_circuit_table(circuit_rows)

function _write_markdown(path, rows)
    open(path, "w") do io
        println(io, "| problem | alg | maxdim | QIT time (ms) | QIT mem (MiB) | QIT dim | IT time (ms) | IT mem (MiB) | IT dim | speedup (IT/QIT) |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|")
        for r in rows
            speedup = r.it_time_ms / r.qit_time_ms
            @printf(io, "| %s | %s | %d | %.2f | %.2f | %d | %.2f | %.2f | %d | %.2fx |\n",
                r.name, string(r.alg), r.maxdim, r.qit_time_ms, r.qit_memory_mib, r.qit_maxlinkdim,
                r.it_time_ms, r.it_memory_mib, r.it_maxlinkdim, speedup)
        end
    end
end

function _write_csv(path, rows)
    open(path, "w") do io
        println(io, "problem,alg,maxdim,qit_time_ms,qit_memory_mib,qit_maxlinkdim,it_time_ms,it_memory_mib,it_maxlinkdim")
        for r in rows
            println(io, join((r.name, r.alg, r.maxdim, r.qit_time_ms, r.qit_memory_mib, r.qit_maxlinkdim,
                r.it_time_ms, r.it_memory_mib, r.it_maxlinkdim), ","))
        end
    end
end

mkpath(joinpath(@__DIR__, "results"))
_write_markdown(joinpath(@__DIR__, "results", "compare_itensor.md"), circuit_rows)
_write_csv(joinpath(@__DIR__, "results", "compare_itensor.csv"), circuit_rows)

println("\nSaved: benchmark/results/compare_itensor.md, benchmark/results/compare_itensor.csv")