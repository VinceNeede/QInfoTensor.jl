using Random

# ------------------------------------------------------------------------
# RandomApplyProblem: a synthetic random MPO applied to a synthetic
# random MPS, mirroring the SRC paper's own benchmark setup as closely as
# possible (Camaño, Epperly & Tropp 2025, Figure 1 and section 4.4.2):
#   - n=100 sites, physical dimension d=2
#   - MPO bond dimension D=50, MPS bond dimension χ=50
#   - i.i.d. entries, real-valued, uniform on [α,1] with α=-0.5 (the
#     paper's own "intermediate difficulty" choice — see Figure 1's
#     caption and section 1's discussion of how α controls difficulty)
#   - stored as complex, per the paper's own convention ("the data...is
#     real, but we store it using complex data types, which is the
#     required data format for most quantum tensor network problems")
#
# This is DELIBERATELY non-physical (the paper's own words) — its only
# purpose is to reach D,χ ≫ 1 with a genuinely small requested output
# bond dimension χ̄ ≪ D·χ (steep compression), which is the regime the
# paper's own comparisons (and ours, see chat) show SRC's asymptotic
# advantage over contract-then-compress/density-matrix actually
# materializes. CircuitProblem/HamiltonianApplyProblem above both have
# MPO bond dimension D≤4, far too small to show a difference — this
# problem exists specifically to fill that gap.
# ------------------------------------------------------------------------

struct RandomApplyProblem
    name::String
    n::Int
    D::Int              # MPO bond dimension
    χ::Int               # MPS bond dimension
    d::Int               # physical dimension
    α::Float64           # entries ~ Uniform(α, 1); α=-0.5 is the paper's own choice
    maxdim_values::Vector{Int}   # requested output bond dimension χ̄, swept
    seed::Int            # pins the random draw — see build_random_apply_inputs
end

# Paper's own χ̄ sweep in Figure 1 goes from 5 to 100; reused here verbatim.
const _RANDOM_ALPHA = -0.5
const _RANDOM_MAXDIM_VALUES = [5, 10, 20, 40, 80, 100]

# Paper's own Figure 1 / section 4.4.2 setup exactly: n=100, D=χ=50, d=2.
const random_apply_paper = RandomApplyProblem(
    "random_apply_paper_n100_D50_chi50", 100, 50, 50, 2, _RANDOM_ALPHA, _RANDOM_MAXDIM_VALUES, 1234,
)

const RANDOM_APPLY_PROBLEMS = (random_apply_paper,)

# Small correctness-check version: same D≈χ shape, tiny n so an exact
# (untruncated, maxdim=D*χ) apply is cheap to cross-check against
# inner(ψ0,H,ψ0) directly. See _check_random_apply_correctness in
# benchmarks.jl.
const random_apply_check = RandomApplyProblem(
    "random_apply_check", 6, 3, 3, 2, _RANDOM_ALPHA, [9], 4321,  # maxdim=D*χ=9 → exact, no truncation
)

const RANDOM_APPLY_CHECK_PROBLEMS = (random_apply_check,)

"""
    _random_dense(dims...; α, rng) -> Array{ComplexF64}

Dense array of the given shape with i.i.d. real entries uniform on
`[α,1]`, stored as complex — matching the paper's own storage convention
(see module docstring above).
"""
function _random_dense(dims::Int...; α::Real, rng::AbstractRNG)
    lo, hi = float(α), 1.0
    return complex.(lo .+ (hi - lo) .* rand(rng, dims...))
end

"""
    build_random_mps_tensor(χl, χr, d; α, rng) -> MPSTensor

`(left,phys)→right` leg convention (codomain=`(left,phys)`,
domain=`(right,)`), matching QInfoTensor's own `MPSTensor` convention
(see design_notes.md).

Normalized to unit Frobenius norm before returning — see the module-level
note above `_random_dense` for why: without this, `n` unnormalized
random tensors compound multiplicatively across the chain, and
`norm(ψ0)` was confirmed to reach ~1e123 for the paper-scale problem
(n=100,D=χ=50) — a root numerical pathology, not just a truncation-
criterion mismatch. Per-site normalization keeps each factor's individual
contribution O(1), preventing that compounding at its source.
"""
function build_random_mps_tensor(χl::Int, χr::Int, d::Int; α::Real, rng::AbstractRNG)
    arr = _random_dense(χl, d, χr; α, rng)
    t = TensorMap(arr, (ℂ^χl ⊗ ℂ^d) ← ℂ^χr)
    return t / norm(t)
end

"""
    build_random_mpo_tensor(Dl, Dr, d; α, rng) -> MPOTensor

`(left,site_out)→(site_in,right)` leg convention: codomain=`(left,
site_out)`, domain=`(site_in,right)`.

NOTE: this is `(site_in, right)`, NOT `(right, site_in)` as
design_notes.md's prose describes — the domain order there doesn't match
the actual implementation. Confirmed empirically from circuit.jl's
`_identity_mpo_tensor`, a known-working MPOTensor construction:
`TensorMap(Id, (ℂ^1 ⊗ V) ← (V ⊗ ℂ^1))` has domain `(V, ℂ^1)` =
`(site_in, right)`. Using the design doc's stated order here produced a
`SpaceMismatch` (physical leg colliding with the bond leg) the first time
this was run — see chat.

Also normalized to unit Frobenius norm before returning, for the same
reason as `build_random_mps_tensor` — see its docstring.
"""
function build_random_mpo_tensor(Dl::Int, Dr::Int, d::Int; α::Real, rng::AbstractRNG)
    arr = _random_dense(Dl, d, d, Dr; α, rng)  # (left, site_out, site_in, right)
    t = TensorMap(arr, (ℂ^Dl ⊗ ℂ^d) ← (ℂ^d ⊗ ℂ^Dr))
    return t / norm(t)
end

"""
    build_random_apply_inputs(problem::RandomApplyProblem) -> (H, ψ0)

Build a random open-boundary MPO `H` (bond dimension `D`) and random
open-boundary MPS `ψ0` (bond dimension `χ`), both with `n` sites and
physical dimension `d`. Boundary bonds are trivial (dimension 1); every
interior bond is exactly `D` (MPO) or `χ` (MPS). Each individual site
tensor is normalized to unit Frobenius norm at construction (see
`build_random_mps_tensor`/`build_random_mpo_tensor`) — this is NOT full
MPS canonicalization/orthogonalization (still applied separately below,
via `orthogonalize!`), just a guard against the overall network's norm
compounding to an absurd scale across `n` sites. `ψ0` itself is still
not meant to be read as a physically normalized quantum state in any
deeper sense — it's raw random data, matching the paper's own Figure 1
methodology, just without the numerical landmine of astronomical norms.

`problem.seed` pins the random draw via a dedicated `MersenneTwister`,
so target/baseline runs under `judge` see IDENTICAL data — otherwise
time/memory comparisons would carry extra noise from run-to-run data
variation, and the exact-recovery correctness check for
`random_apply_check` would be flaky (SRC's exact recovery, Theorem 3 in
the paper, holds "with probability one" for a given random draw, not
universally across all possible draws).

NOTE: assumes `QInfoTensor.MPS` has a raw-tensor-vector constructor
analogous to the `QInfoTensor.MPO(::Vector{<:MPOTensor})` constructor
already used successfully in circuit.jl's `build_gate_layer_mpo`. Adjust
the two constructor calls below if the actual API differs.

Both `H` and `ψ0` are left-canonicalized here, ONCE, before returning —
outside the timed benchmark. `ψ0` is built as raw, independent random
tensors with NO orthogonality relationship between adjacent sites
(unlike `build_hamapply_inputs`'s `random_mps`, which already returns a
canonical-form state), so it needs this step explicitly. Skipping it was
tried first and produced a `:zipup` runtime that was essentially FLAT
across the entire `maxdim` sweep (5 through 100): `apply!`'s internal
canonicalization cost, `O(n·[dD³+dχ³])` per the paper, is independent of
the requested output bond dimension, and at `D=χ=50, n=100` it's large
enough to swamp the part of the cost that actually depends on `maxdim`.
Paying that cost fresh on every `@benchmarkable` trial (rather than once,
upfront, as done here) hid the very scaling behavior this problem exists
to measure. See chat.
"""
function build_random_apply_inputs(problem::RandomApplyProblem)
    rng = MersenneTwister(problem.seed)
    n, D, χ, d, α = problem.n, problem.D, problem.χ, problem.d, problem.α
    T = ComplexF64

    mpo_tensors = Vector{MPOTensor{T,ComplexSpace,Vector{T}}}(undef, n)
    mps_tensors = Vector{MPSTensor{T,ComplexSpace,Vector{T}}}(undef, n)

    for i in 1:n
        Dl = i == 1 ? 1 : D
        Dr = i == n ? 1 : D
        χl = i == 1 ? 1 : χ
        χr = i == n ? 1 : χ
        mpo_tensors[i] = build_random_mpo_tensor(Dl, Dr, d; α, rng)
        mps_tensors[i] = build_random_mps_tensor(χl, χr, d; α, rng)
    end

    H = QInfoTensor.MPO(mpo_tensors)
    QInfoTensor.orthogonalize!(H, 1)
    ψ0 = QInfoTensor.MPS(mps_tensors)
    QInfoTensor.orthogonalize!(ψ0, 1)
    return H, ψ0
end

"""
    run_random_apply(H, ψ0; alg=:zipup, maxdim, cutoff=nothing) -> MPS

Compute `H|ψ0⟩` via `apply`. Same alg-dependent kwarg filtering as
`run_hamapply` (see problems.jl): `:src`'s `apply!` method only accepts
`maxdim`, so `cutoff` is only forwarded for `alg != :src` (i.e. for
`:zipup` and `:densitymatrix`).

`cutoff` defaults to `nothing` here — NOT the small-but-nonzero `1e-14`
used elsewhere in the benchmark suite (`run_hamapply`,
`run_circuit_trajectory`), and not `0.0` either. `nothing` is `apply!`'s
own idiom for "this criterion is off" (the same idiom already used for
`sweep_maxdim=nothing`/`sweep_cutoff=nothing` in `run_hamapply`) — a
cleaner way to fully disable the relative-tolerance truncation path than
passing a numeric `0.0`, which is still a real value that gets compared/
computed against rather than skipping the check outright.

This stands on its own merits independent of numerical scale: the
paper's own Figure 1/section 4.4.2 never specifies a cutoff at all — it
always requests a fixed target bond dimension χ̄ directly, precisely
because a tolerance-based criterion is close to meaningless on i.i.d.
random data with no natural entanglement decay. `cutoff=nothing`
reproduces that protocol exactly: truncation is governed by `maxdim`
alone, matching `:src`'s own (cutoff-free) behavior.

(Separately, `build_random_apply_inputs` now also normalizes each site
tensor at construction, addressing the numerical pathology — norm(ψ0)/
norm(H) reaching ~1e123/~1e138 before that fix — that FIRST exposed this
issue, by making `cutoff=1e-14` collapse `:zipup`'s output to bond
dimension 1 regardless of `maxdim`. That was a real, separate bug fixed
at its source; `cutoff=nothing` here is the correct choice regardless of
whether that numerical pathology is present, since it matches the
paper's actual comparison protocol either way. See chat.)
"""
function run_random_apply(H, ψ0; alg::Symbol=:zipup, maxdim::Int, cutoff::Union{Real,Nothing}=nothing)
    if alg == :src
        return apply(H, ψ0; alg, maxdim)
    else
        return apply(H, ψ0; alg, maxdim, cutoff)
    end
end