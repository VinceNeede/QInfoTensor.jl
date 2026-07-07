using MKL, LinearAlgebra
using BenchmarkTools
using TensorKit
using QInfoTensor

BLAS.set_num_threads(1)
MKL.set_num_threads(1)

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "circuit.jl"))

# ---------------------------------------------------------------------------
# Blocking correctness checks: one untimed run per problem PER ALGORITHM, to
# make sure apply is actually right before trusting the timings SUITE
# measures. If either check fails, this errors out before SUITE is even
# built.
#
# Checked for both APPLY_ALGS (:zipup and :src) — :src is a randomized
# algorithm (Camaño, Epperly & Tropp 2025), so its own correctness at
# near-exact bond dimension is exactly as worth re-confirming here as
# :zipup's, not something safe to assume by analogy.
#
# Two DIFFERENT checks are needed, because the two problem types have
# different notions of "correct":
#   - CircuitProblem: the gate is orthogonal, so exact unitary evolution
#     preserves norm exactly. Run near-exact (maxdim = 4^n_steps, the true
#     final bond dimension — no truncation at all) and check norm(ψ)≈1.
#   - HamiltonianApplyProblem: H is not unitary, so norm isn't preserved —
#     that check doesn't apply here. Instead, cross-check apply's result
#     against inner(ψ0,H,ψ0) directly (the same consistency already
#     validated in the package's own test suite, re-run here against this
#     benchmark's specific problem instance rather than just trusting the
#     generic test elsewhere).
# ---------------------------------------------------------------------------

const APPLY_ALGS = (:zipup, :src)

function _check_circuit_correctness(problem::CircuitProblem, alg::Symbol)
    _, ψ0, H_odd, H_even = build_circuit_inputs(problem)
    χ_exact = 4^problem.n_steps
    ψ = run_circuit_trajectory(ψ0, H_odd, H_even, problem.n_steps; alg, maxdim=χ_exact, cutoff=1e-14)
    n = norm(ψ)
    @assert isapprox(n, 1.0; atol=1e-6) """
        Circuit correctness check failed for $(problem.name) (alg=$alg):
        got norm(ψ)=$n, expected ≈1 (orthogonal gates preserve norm exactly)
        """
    return nothing
end

function _check_hamapply_correctness(problem::HamiltonianApplyProblem, alg::Symbol)
    _, H, ψ0 = build_hamapply_inputs(problem)
    Hψ0 = run_hamapply(H, ψ0; alg, maxdim=1000, cutoff=1e-14)  # near-exact
    lhs = inner(ψ0, Hψ0)
    rhs = inner(ψ0, H, ψ0)
    @assert isapprox(lhs, rhs; atol=1e-8) """
        Hamiltonian apply correctness check failed for $(problem.name) (alg=$alg):
        got ⟨ψ0|Hψ0⟩=$lhs, expected ⟨ψ0|H|ψ0⟩=$rhs
        """
    return nothing
end

for alg in APPLY_ALGS, problem in CIRCUIT_PROBLEMS
    _check_circuit_correctness(problem, alg)
end

for alg in APPLY_ALGS, problem in HAMAPPLY_PROBLEMS
    _check_hamapply_correctness(problem, alg)
end

# ---------------------------------------------------------------------------
# SUITE
# ---------------------------------------------------------------------------

const SUITE = BenchmarkGroup()

include(joinpath(@__DIR__, "suites", "apply.jl"))