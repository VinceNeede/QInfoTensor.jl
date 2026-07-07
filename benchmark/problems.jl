# ------------------------------------------------------------------------
# Shared physical problem definitions for the apply benchmarks. Created
# now (not earlier, as speculative DMRG scaffolding) because
# HamiltonianApplyProblem genuinely needs HamiltonianSpec — see chat.
#
# TFIM convention: Sz-Sz coupling + Sx field, matching QInfoTensor's own
# test suite (test_opsum.jl/test_mpo.jl), not the old prototype
# benchmark's Sx-Sx+Sz convention — a deliberate choice for consistency
# with what's already tested here, not an oversight (see chat).
# ------------------------------------------------------------------------

struct HamiltonianSpec
    name::String
    L::Int
    periodic::Bool
    build_sites::Function   # () -> sites
    build_opsum::Function   # () -> OpSum
end

function _tfim_opsum(L::Int; J::Real=1.0, h::Real=1.0, periodic::Bool)
    os = OpSum()
    range_ = periodic ? (1:L) : (1:L-1)
    for i in range_
        j = periodic ? mod1(i + 1, L) : i + 1
        os += (-J, :Sz, i, :Sz, j)
    end
    for i in 1:L
        os += (-h, :Sx, i)
    end
    return os
end

const tfim_L20_open = HamiltonianSpec(
    "tfim_L20_open", 20, false,
    () -> sitetypes(:SpinHalf, 20),
    () -> _tfim_opsum(20; periodic=false),
)

const tfim_L20_periodic = HamiltonianSpec(
    "tfim_L20_periodic", 20, true,
    () -> sitetypes(:SpinHalf, 20),
    () -> _tfim_opsum(20; periodic=true),
)

# ------------------------------------------------------------------------
# HamiltonianApplyProblem: apply(H,ψ) ONCE, to a random_mps with a
# non-trivial STARTING bond dimension, swept over maxdim. NOT repeated
# application of H — see chat / circuit.jl's own comment on why that
# isn't physically meaningful (that's what the circuit-trajectory
# benchmark is for instead). This benchmark specifically stresses a
# Hamiltonian MPO's own bond dimension w, which can be much larger than
# the circuit's thin (w=4) gate-layer MPO.
# ------------------------------------------------------------------------

struct HamiltonianApplyProblem
    name::String
    hamiltonian::HamiltonianSpec
    initial_maxdim::Int          # bond dimension of the random_mps input state
    maxdim_values::Vector{Int}   # swept for the benchmark
    cutoff::Float64
end

const _HAMAPPLY_MAXDIM_VALUES = [10, 20, 40, 80]
const _HAMAPPLY_CUTOFF = 1e-12

const hamapply_tfim_L20_open = HamiltonianApplyProblem(
    "hamapply_tfim_L20_open", tfim_L20_open, 20, _HAMAPPLY_MAXDIM_VALUES, _HAMAPPLY_CUTOFF,
)

const hamapply_tfim_L20_periodic = HamiltonianApplyProblem(
    "hamapply_tfim_L20_periodic", tfim_L20_periodic, 20, _HAMAPPLY_MAXDIM_VALUES, _HAMAPPLY_CUTOFF,
)

const HAMAPPLY_PROBLEMS = (hamapply_tfim_L20_open, hamapply_tfim_L20_periodic)

# ------------------------------------------------------------------------
# Small "smoke-test" Hamiltonian problems, used ONLY by the correctness
# checks in benchmarks.jl — deliberately NOT part of HAMAPPLY_PROBLEMS/
# SUITE. Same reasoning as circuit_check in circuit.jl: a bug in
# _tfim_opsum/MPO construction (open vs. periodic wraparound term, sign
# convention, etc.) shows up identically at L=6 as at L=20, so there's
# nothing gained in bug-catching power from checking at the full L=20,
# maxdim=1000 benchmark scale — only cost.
# ------------------------------------------------------------------------

const _HAMAPPLY_CHECK_L = 6
const _HAMAPPLY_CHECK_INITIAL_MAXDIM = 4
const _HAMAPPLY_CHECK_MAXDIM_VALUES = [8]

const tfim_check_open = HamiltonianSpec(
    "tfim_check_open", _HAMAPPLY_CHECK_L, false,
    () -> sitetypes(:SpinHalf, _HAMAPPLY_CHECK_L),
    () -> _tfim_opsum(_HAMAPPLY_CHECK_L; periodic=false),
)

const tfim_check_periodic = HamiltonianSpec(
    "tfim_check_periodic", _HAMAPPLY_CHECK_L, true,
    () -> sitetypes(:SpinHalf, _HAMAPPLY_CHECK_L),
    () -> _tfim_opsum(_HAMAPPLY_CHECK_L; periodic=true),
)

const hamapply_check_open = HamiltonianApplyProblem(
    "hamapply_check_open", tfim_check_open,
    _HAMAPPLY_CHECK_INITIAL_MAXDIM, _HAMAPPLY_CHECK_MAXDIM_VALUES, _HAMAPPLY_CUTOFF,
)

const hamapply_check_periodic = HamiltonianApplyProblem(
    "hamapply_check_periodic", tfim_check_periodic,
    _HAMAPPLY_CHECK_INITIAL_MAXDIM, _HAMAPPLY_CHECK_MAXDIM_VALUES, _HAMAPPLY_CUTOFF,
)

const HAMAPPLY_CHECK_PROBLEMS = (hamapply_check_open, hamapply_check_periodic)

"""
    build_hamapply_inputs(problem::HamiltonianApplyProblem) -> (sites, H, ψ0)

Build `sites`, the left-canonicalized `MPO`, and the initial (non-trivial
bond dimension) random `MPS`. Not part of the timed benchmark.
"""
function build_hamapply_inputs(problem::HamiltonianApplyProblem)
    sites = problem.hamiltonian.build_sites()
    H = MPO(problem.hamiltonian.build_opsum(), sites)
    orthogonalize!(H, 1)
    ψ0 = random_mps(sites, problem.initial_maxdim)
    return sites, H, ψ0
end

"""
    run_hamapply(H, ψ0; alg=:zipup, maxdim, cutoff) -> MPS

Compute `H|ψ0⟩` via `apply`. This is the part actually timed by
`@benchmarkable`.

`cutoff` is only forwarded when `alg != :src`. `:src`'s `apply!` method
(see apply.jl) only accepts `maxdim` — it has no `cutoff` keyword, since
SRC's accuracy/speed tradeoff comes from oversampling (paper, section
3.4) targeting a fixed output bond dimension, not a truncation tolerance.
Passing `cutoff` through to `:src` raises a `MethodError`/keyword error,
so it's dropped for that one case. `:zipup` and `:densitymatrix` both
accept `cutoff` (an SVD truncation tolerance for `:zipup`, an
eigendecomposition truncation tolerance for `:densitymatrix`), so both
get it forwarded via the same `else` branch.
"""
function run_hamapply(H, ψ0; alg::Symbol=:zipup, maxdim::Int, cutoff::Real)
    if alg == :src
        return apply(H, ψ0; alg, maxdim)
    else
        return apply(H, ψ0; alg, maxdim, cutoff)
    end
end