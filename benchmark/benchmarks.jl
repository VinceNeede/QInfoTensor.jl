using MKL, LinearAlgebra
using BenchmarkTools
using TensorKit
using QInfoTensor

BLAS.set_num_threads(1)
MKL.set_num_threads(1)

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "circuit.jl"))
include(joinpath(@__DIR__, "random_apply.jl"))

# ---------------------------------------------------------------------------
# Blocking correctness checks: one untimed run per CHECK problem PER
# ALGORITHM, to make sure apply is actually right before trusting the
# timings SUITE measures. If either check fails, this errors out before
# SUITE is even built.
#
# Deliberately run against CIRCUIT_CHECK_PROBLEMS/HAMAPPLY_CHECK_PROBLEMS
# (small, fast, defined in circuit.jl/problems.jl) rather than the real
# CIRCUIT_PROBLEMS/HAMAPPLY_PROBLEMS used by SUITE below. These checks
# exist to catch construction bugs (wrong leg convention, wrong OpSum
# term, wrong gate placement, ...) which show up identically regardless
# of L/n_steps/maxdim — there's nothing gained in bug-catching power by
# running the check at the same L=50, n_steps=6, maxdim=1000 scale used
# for timing, only cost (an exact, untruncated bond-4096 trajectory,
# doubled for every alg in APPLY_ALGS, doubled again by judge() running
# target+baseline — this is what silently ate 20+ minutes with zero
# visible progress before this file had any logging in it).
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
#   - HamiltonianApplyProblem/RandomApplyProblem: neither is unitary
#     (RandomApplyProblem's MPO isn't even meant to be physical — it's
#     raw random data, see random_apply.jl), so norm isn't preserved —
#     that check doesn't apply. Instead, cross-check apply's result
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

function _check_random_apply_correctness(problem::RandomApplyProblem, alg::Symbol)
    H, ψ0 = build_random_apply_inputs(problem)
    χ_exact = problem.D * problem.χ
    Hψ0 = run_random_apply(H, ψ0; alg, maxdim=χ_exact, cutoff=nothing)  # exact, no truncation
    lhs = inner(ψ0, Hψ0)
    rhs = inner(ψ0, H, ψ0)
    # rtol, not atol: random_apply's data isn't guaranteed O(1) even after
    # per-site normalization (see build_random_apply_inputs) — an absolute
    # tolerance is meaningless once the compared quantities aren't O(1)
    # themselves, since it stops bounding anything relative to their actual
    # size. This was a real, separate bug from the cutoff/truncation issue
    # discussed in chat: atol=1e-8 against inner products that could be
    # arbitrarily large or small would pass/fail almost arbitrarily,
    # independent of whether the computation was actually correct.
    @assert isapprox(lhs, rhs; rtol=1e-6) """
        Random apply correctness check failed for $(problem.name) (alg=$alg):
        got ⟨ψ0|Hψ0⟩=$lhs, expected ⟨ψ0|H|ψ0⟩=$rhs
        """
    return nothing
end

for alg in APPLY_ALGS, problem in CIRCUIT_CHECK_PROBLEMS
    t0 = time()
    @info "Checking circuit correctness" problem = problem.name alg
    flush(stderr)
    _check_circuit_correctness(problem, alg)
    @info "  done" elapsed_s = round(time() - t0; digits=2)
    flush(stderr)
end

for alg in APPLY_ALGS, problem in HAMAPPLY_CHECK_PROBLEMS
    t0 = time()
    @info "Checking hamapply correctness" problem = problem.name alg
    flush(stderr)
    _check_hamapply_correctness(problem, alg)
    @info "  done" elapsed_s = round(time() - t0; digits=2)
    flush(stderr)
end

for alg in APPLY_ALGS, problem in RANDOM_APPLY_CHECK_PROBLEMS
    t0 = time()
    @info "Checking random-apply correctness" problem = problem.name alg
    flush(stderr)
    _check_random_apply_correctness(problem, alg)
    @info "  done" elapsed_s = round(time() - t0; digits=2)
    flush(stderr)
end

# ---------------------------------------------------------------------------
# SUITE
# ---------------------------------------------------------------------------

const SUITE = BenchmarkGroup()

include(joinpath(@__DIR__, "suites", "apply.jl"))