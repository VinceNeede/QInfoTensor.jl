# ---------------------------------------------------------------------------
# QInfoTensor vs ITensor comparison — circuit/apply and 2-site DMRG.
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
using Logging: with_logger, NullLogger

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "circuit.jl"))
include(joinpath(@__DIR__, "itensor_circuit.jl"))
include(joinpath(@__DIR__, "itensor_dmrg.jl"))

const COMPARISON_ALGS = (:zipup, :densitymatrix)
const FIXED_TOL = 1e-12
const IT_KRYLOVDIM = QInfoTensor._DEFAULT_EIGSOLVE_KWARGS.krylovdim  # Synchronized dynamically [cite: 118]
const IT_MAXITER   = QInfoTensor._DEFAULT_EIGSOLVE_KWARGS.maxiter    # Synchronized dynamically [cite: 118]

# Existing schema for quantum circuit runs
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

# New dedicated schema including final variational energy metrics [cite: 216]
struct DMRGComparisonRow
    name::String
    maxdim::Int
    qit_time_ms::Float64
    qit_memory_mib::Float64
    qit_energy::Float64
    qit_maxlinkdim::Int
    it_time_ms::Float64
    it_memory_mib::Float64
    it_energy::Float64
    it_maxlinkdim::Int
end

function _compare_circuit(problem::CircuitProblem)
    println("Benchmarking Circuit $(problem.name) ...")
    sites, ψ0, H_odd, H_even = build_circuit_inputs(problem)
    it_sites, it_ψ0, it_H_odd, it_H_even = build_itensor_circuit_inputs(problem)

    rows = CircuitComparisonRow[]
    for alg in COMPARISON_ALGS, χ in problem.maxdim_values
        qit_trial = @benchmark(
            run_circuit_trajectory($ψ0, $H_odd, $H_even, $(problem.n_steps);
                alg=$alg, maxdim=$χ, cutoff=$(problem.cutoff)),
        )
        
        it_trial = @benchmark(
            run_itensor_circuit_trajectory($it_ψ0, $it_H_odd, $it_H_even, $(problem.n_steps);
                alg=$alg, maxdim=$χ, cutoff=$(problem.cutoff),
                sweep_maxdim=$(2χ), sweep_cutoff=$(problem.cutoff / 10)),
        )
        qit_m, it_m = median(qit_trial), median(it_trial)

        ψ_qit = run_circuit_trajectory(ψ0, H_odd, H_even, problem.n_steps;
            alg, maxdim=χ, cutoff=problem.cutoff)
        ψ_it = run_itensor_circuit_trajectory(it_ψ0, it_H_odd, it_H_even, problem.n_steps;
            alg, maxdim=χ, cutoff=problem.cutoff, sweep_maxdim=2χ, sweep_cutoff=problem.cutoff / 10)

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

function _compare_dmrg(problem::DMRGProblem)
    println("Benchmarking DMRG $(problem.name) ...")
    sites, H, ψ0 = build_dmrg_inputs(problem)
    it_sites, it_H, it_ψ0 = build_itensor_dmrg_inputs(problem)

    rows = DMRGComparisonRow[]
    for χ in problem.maxdim_values
        @info "benchmarking QInfoTensor DMRG χ=$χ ..."
        # QInfoTensor side: Timed execution
        t0 = time() 
        qit_trial = @benchmark(
            with_logger(NullLogger()) do
                dmrg!(ψ, $H, $(problem.nsweeps); nsite=2, maxdim=$χ, cutoff=$(problem.cutoff), eigsolve_tol=FIXED_TOL)
            end,
            setup=(ψ = copy($ψ0))
        )
        @info "  done" elapsed_s = round(time() - t0; digits=2)

        @info "benchmarking ITensor DMRG χ=$χ ..."
        t0 = time()
        # ITensor side: Timed execution with matched solver shapes
        it_trial = @benchmark(
            ITensorMPS.dmrg($it_H, ψ; 
                nsweeps=$(problem.nsweeps), 
                maxdim=$χ, 
                cutoff=$(problem.cutoff), 
                eigsolve_krylovdim=IT_KRYLOVDIM, 
                eigsolve_maxiter=IT_MAXITER, 
                eigsolve_tol=FIXED_TOL,
                outputlevel=0),
            setup=(ψ = copy($it_ψ0))
        )
        @info "  done" elapsed_s = round(time() - t0; digits=2)

        qit_m, it_m = median(qit_trial), median(it_trial)

        # Untimed verification pass to harvest final energy metrics and link profiles safely
        @info "Run to get energy"
        ψ_qit = copy(ψ0)
        _, _, sd_qit = with_logger(NullLogger()) do
            dmrg!(ψ_qit, H, problem.nsweeps; nsite=2, maxdim=χ, cutoff=problem.cutoff, eigsolve_tol=FIXED_TOL)
        end
        qit_energy = sd_qit[end].energies[end] # [cite: 216]
        qit_dim = QInfoTensor.maxlinkdim(ψ_qit)
        
        ψ_it = copy(it_ψ0)
        it_energy, ψ_it = ITensorMPS.dmrg(it_H, ψ_it; 
            nsweeps=problem.nsweeps, maxdim=χ, cutoff=problem.cutoff, 
            eigsolve_krylovdim=IT_KRYLOVDIM, eigsolve_maxiter=IT_MAXITER, eigsolve_tol=FIXED_TOL, 
            outputlevel=0)
        it_dim = ITensorMPS.maxlinkdim(ψ_it)

        push!(rows, DMRGComparisonRow(
            problem.name, χ,
            time(qit_m) / 1e6, memory(qit_m) / 2^20, qit_energy, qit_dim,
            time(it_m) / 1e6, memory(it_m) / 2^20, it_energy, it_dim,
        ))
    end
    return rows
end

# Executions
circuit_rows = reduce(vcat, (_compare_circuit(p) for p in CIRCUIT_PROBLEMS))
dmrg_rows = reduce(vcat, (_compare_dmrg(p) for p in DMRG_PROBLEMS))

# Separate terminal printing formatters
function _print_circuit_table(rows)
    println("\n=== Circuit Performance Comparison ===")
    @printf("%-16s %-14s %8s %12s %12s %10s %12s %12s %10s\n",
        "problem", "alg", "maxdim", "QIT time(ms)", "QIT mem(MiB)", "QIT dim",
        "IT time(ms)", "IT mem(MiB)", "IT dim")
    for r in rows
        @printf("%-16s %-14s %8d %12.2f %12.2f %10d %12.2f %12.2f %10d\n",
            r.name, string(r.alg), r.maxdim, r.qit_time_ms, r.qit_memory_mib, r.qit_maxlinkdim,
            r.it_time_ms, r.it_memory_mib, r.it_maxlinkdim)
    end
end

function _print_dmrg_table(rows)
    println("\n=== DMRG Performance & Energy Verification ===")
    @printf("%-24s %6s %12s %12s %14s %8s %12s %12s %14s %8s\n",
        "problem", "maxdim", "QIT time(ms)", "QIT mem(MiB)", "QIT Energy", "QIT dim",
        "IT time(ms)", "IT mem(MiB)", "IT Energy", "IT dim")
    for r in rows
        @printf("%-24s %6d %12.2f %12.2f %14.6f %8d %12.2f %12.2f %14.6f %8d\n",
            r.name, r.maxdim, r.qit_time_ms, r.qit_memory_mib, r.qit_energy, r.qit_maxlinkdim,
            r.it_time_ms, r.it_memory_mib, r.it_energy, r.it_maxlinkdim)
    end
end

_print_circuit_table(circuit_rows)
_print_dmrg_table(dmrg_rows)

# Combined Sequential Markdown Writer
function _write_markdown_report(path, c_rows, d_rows)
    open(path, "w") do io
        println(io, "# QInfoTensor vs ITensor Performance Report\n")
        
        # Table 1: Circuits
        println(io, "## 1. Circuit Trajectory Compression")
        println(io, "| problem | alg | maxdim | QIT time (ms) | QIT mem (MiB) | QIT dim | IT time (ms) | IT mem (MiB) | IT dim | speedup (IT/QIT) |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|")
        for r in c_rows
            speedup = r.it_time_ms / r.qit_time_ms
            @printf(io, "| %s | %s | %d | %.2f | %.2f | %d | %.2f | %.2f | %d | %.2fx |\n",
                r.name, string(r.alg), r.maxdim, r.qit_time_ms, r.qit_memory_mib, r.qit_maxlinkdim,
                r.it_time_ms, r.it_memory_mib, r.it_maxlinkdim, speedup)
        end
        println(io, "\n")
        
        # Table 2: DMRG
        println(io, "## 2. Two-Site DMRG Ground State Optimization")
        println(io, "| problem | maxdim | QIT time (ms) | QIT mem (MiB) | QIT Energy | QIT dim | IT time (ms) | IT mem (MiB) | IT Energy | IT dim | speedup (IT/QIT) |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|---|")
        for r in d_rows
            speedup = r.it_time_ms / r.qit_time_ms
            @printf(io, "| %s | %d | %.2f | %.2f | %.6f | %d | %.2f | %.2f | %.6f | %d | %.2fx |\n",
                r.name, r.maxdim, r.qit_time_ms, r.qit_memory_mib, r.qit_energy, r.qit_maxlinkdim,
                r.it_time_ms, r.it_memory_mib, r.it_energy, r.it_maxlinkdim, speedup)
        end
    end
end

mkpath(joinpath(@__DIR__, "results"))
_write_markdown_report(joinpath(@__DIR__, "results", "compare_itensor.md"), circuit_rows, dmrg_rows)

println("\nSaved: benchmark/results/compare_itensor.md")