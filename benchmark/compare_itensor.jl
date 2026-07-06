# ---------------------------------------------------------------------------
# QInfoTensor vs ITensor comparison — circuit/apply only. Standalone script,
# NOT part of the PkgBenchmark SUITE (ITensor doesn't change across
# QInfoTensor commits, so PkgBenchmark's judge/compare isn't the right
# mechanism for this).
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

struct CircuitComparisonRow
    name::String
    maxdim::Int
    qit_time_ms::Float64
    qit_memory_mib::Float64
    it_time_ms::Float64
    it_memory_mib::Float64
end

function _compare_circuit(problem::CircuitProblem)
    println("Benchmarking $(problem.name) ...")
    sites, ψ0, H_odd, H_even = build_circuit_inputs(problem)
    it_sites, it_ψ0, it_H_odd, it_H_even = build_itensor_circuit_inputs(problem)

    rows = CircuitComparisonRow[]
    for χ in problem.maxdim_values
        qit_trial = @benchmark(
            run_circuit_trajectory($ψ0, $H_odd, $H_even, $(problem.n_steps);
                maxdim=$χ, cutoff=$(problem.cutoff),
                sweep_maxdim=$(2χ), sweep_cutoff=$(problem.cutoff / 10)),
        )
        it_trial = @benchmark(
            run_itensor_circuit_trajectory($it_ψ0, $it_H_odd, $it_H_even, $(problem.n_steps);
                maxdim=$χ, cutoff=$(problem.cutoff),
                sweep_maxdim=$(2χ), sweep_cutoff=$(problem.cutoff / 10)),
        )
        qit_m, it_m = median(qit_trial), median(it_trial)
        push!(rows, CircuitComparisonRow(
            problem.name, χ,
            time(qit_m) / 1e6, memory(qit_m) / 2^20,
            time(it_m) / 1e6, memory(it_m) / 2^20,
        ))
    end
    return rows
end

circuit_rows = reduce(vcat, (_compare_circuit(p) for p in CIRCUIT_PROBLEMS))

function _print_circuit_table(rows)
    @printf("%-16s %8s %12s %12s %12s %12s\n",
        "problem", "maxdim", "QIT time(ms)", "QIT mem(MiB)", "IT time(ms)", "IT mem(MiB)")
    for r in rows
        @printf("%-16s %8d %12.2f %12.2f %12.2f %12.2f\n",
            r.name, r.maxdim, r.qit_time_ms, r.qit_memory_mib, r.it_time_ms, r.it_memory_mib)
    end
end

_print_circuit_table(circuit_rows)

function _write_markdown(path, rows)
    open(path, "w") do io
        println(io, "| problem | maxdim | QIT time (ms) | QIT mem (MiB) | IT time (ms) | IT mem (MiB) | speedup (IT/QIT) |")
        println(io, "|---|---|---|---|---|---|---|")
        for r in rows
            speedup = r.it_time_ms / r.qit_time_ms
            @printf(io, "| %s | %d | %.2f | %.2f | %.2f | %.2f | %.2fx |\n",
                r.name, r.maxdim, r.qit_time_ms, r.qit_memory_mib,
                r.it_time_ms, r.it_memory_mib, speedup)
        end
    end
end

function _write_csv(path, rows)
    open(path, "w") do io
        println(io, "problem,maxdim,qit_time_ms,qit_memory_mib,it_time_ms,it_memory_mib")
        for r in rows
            println(io, join((r.name, r.maxdim, r.qit_time_ms, r.qit_memory_mib,
                r.it_time_ms, r.it_memory_mib), ","))
        end
    end
end

mkpath(joinpath(@__DIR__, "results"))
_write_markdown(joinpath(@__DIR__, "results", "compare_itensor.md"), circuit_rows)
_write_csv(joinpath(@__DIR__, "results", "compare_itensor.csv"), circuit_rows)

println("\nSaved: benchmark/results/compare_itensor.md, benchmark/results/compare_itensor.csv")
