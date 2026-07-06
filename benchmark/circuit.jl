# ------------------------------------------------------------------------
# Brickwork random circuit — ported from the prototype's circuit.jl.
#
# Boundary-leg insertion went through several rounds of discussion (see
# chat) before landing on a CONFIRMED-correct approach:
#   - manual dense permutedims+svd+Diagonal-multiply -> simplified to
#     permute+left_orth for the regroup-then-SVD core (safe, no boundary
#     legs involved in that part).
#   - block()/dense-round-trip boundary insertion -> tried replacing with
#     a pure outer-product @tensor construction (no shared labels) to
#     avoid leaving TensorKit's representation -> CONFIRMED BROKEN: an
#     outer-product placing a tensor's native codomain leg (e_r, since
#     Tensor(...) is always codomain-only by definition) into an output
#     DOMAIN position is itself a boundary crossing, introducing a dual —
#     but the SAME construction for e_l (codomain->codomain, no crossing)
#     doesn't. This asymmetry (left boundary plain, right boundary dual)
#     was confirmed empirically on a freshly-built layer MPO (printed
#     codomain/domain, before orthogonalize! even runs) and is exactly
#     what broke orthogonalize!/right_orth downstream. Reverted to the
#     dense round-trip, which structurally can't produce this asymmetry —
#     same uniform-plain-legs guarantee as _fsm_site_tensor (OpSum's
#     known-working MPO construction, validated via passing tests).
#   - _gate_to_mpo_tensors/_identity_mpo_tensor are computed ONCE outside
#     the per-layer loop in build_gate_layer_mpo (gate/V identical for
#     every application within a layer) and reused — free, correct reuse,
#     not an approximation.
#
# This file needs NO LinearAlgebra dependency (svd/Diagonal/I are all
# gone, replaced by left_orth and op(:Id) respectively).
# ------------------------------------------------------------------------

# Fixed gate, seed=42, real orthogonal 4×4 (generated once offline, reused
# verbatim from the prototype for continuity — this is a plain dense
# array, entirely backend-independent).
const _CIRCUIT_GATE = [
    0.154219769592948  -0.661326125434211   0.307225002353379  -0.666690945201626
    -0.987434631181200  -0.079398578030011   0.069569904996381  -0.117595677087443
    -0.008503201203209  -0.456837201326657   0.545528812124499   0.702585071144668
    0.033418668136880   0.589612918030985   0.776640934660152  -0.219245657017761
]

"""
    _gate_to_mpo_tensors(gate, V) -> (MPOTensor, MPOTensor)

Decompose a 2-site gate (dense `d²×d²` matrix, combined-index convention
`k=(i1-1)*d+i2`, `i2` faster) into two `MPOTensor`s via SVD, with an
internal bond of dimension `rank(gate)` (typically `d²`, since
`_CIRCUIT_GATE` is generic orthogonal). `V` is the (single, shared)
physical space — both sites must have the same physical space for this
gate to apply.

Boundary legs are inserted via a DENSE ROUND-TRIP (`block(...)` out,
`TensorMap(...)` back in), NOT a pure outer-product `@tensor` construction
— confirmed necessary, not a style choice. An earlier outer-product
version (`e_l[bl]*Gate0[...]*e_r[br]`, no shared labels) was tried and
CONFIRMED BROKEN: `e_r`'s own native leg is codomain-only (`Tensor(...)`
is always rank-(N,0) by definition), so placing it into the output's
DOMAIN side is itself a crossing, which introduces a dual — while `e_l`'s
leg enters in a codomain position (no crossing, stays plain). This
asymmetry (left boundary plain, right boundary dual) was confirmed
empirically (printed `codomain`/`domain` on a freshly-built layer MPO,
before `orthogonalize!` even runs) and is exactly what broke
`orthogonalize!`/`right_orth` downstream — `_fsm_site_tensor`
(`OpSum`→`MPO`, known-working) builds every leg uniformly plain via a
single direct `TensorMap(matrix,cod←dom)` call with no crossing
anywhere; the dense round-trip here restores that same uniform-plain
convention structurally, since a direct construction can't introduce
this kind of asymmetry.
"""
function _gate_to_mpo_tensors(gate::AbstractMatrix, V::ElementarySpace)
    Gate0 = TensorMap(gate, (V ⊗ V) ← (V ⊗ V))  # legs (o2,o1;i2,i1) per TensorKit's own flattening
    Gate_regrouped = TensorKit.permute(Gate0, ((2, 4), (1, 3)))  # (o1,i1) | (o2,i2)
    V_factor, C_factor = left_orth(Gate_regrouped)

    L_dense = insertleftunit(V_factor, Val(1))
    R_dense = insertrightunit(C_factor, Val(3))

    left_tensor = TensorKit.permute(L_dense, ((1, 2), (3, 4)))
    right_tensor = TensorKit.permute(R_dense, ((1, 2), (3, 4)))
    return left_tensor, right_tensor
end

"""
    _identity_mpo_tensor(st::SiteType) -> MPOTensor

Identity operator on a single site, trivial (dimension-1) bonds on both
sides — used for sites a circuit layer doesn't cover. Built via a direct
`TensorMap` construction (dense identity matrix already has exactly the
right shape for a rank-(2,2) tensor's dense constructor, `(d,d)`, once
both boundary legs are `ℂ^1` — no reshape needed at all), for the same
uniform-plain-legs reason as `_gate_to_mpo_tensors`.
"""
function _identity_mpo_tensor(st::QInfoTensor.SiteType)
    V = TensorKit.space(st)
    Id = QInfoTensor.op(st, QInfoTensor.OpName(:Id))  # dense (d,d) matrix, already the right shape
    return TensorMap(Id, (ℂ^1 ⊗ V) ← (V ⊗ ℂ^1))
end

"""
    build_gate_layer_mpo(sites, gate; start=1) -> MPO

One layer of the brickwork circuit: `gate` applied to non-overlapping
bonds starting at `start` (`start=1` → bonds `(1,2),(3,4),...`; `start=2`
→ bonds `(2,3),(4,5),...`, with sites 1 and/or `L` left as identity if
uncovered). `length(sites)` must be even.

`_gate_to_mpo_tensors`/`_identity_mpo_tensor` are computed once and
reused across every bond/site in the layer — valid since `gate` and the
physical space are identical throughout (assumes uniform physical space
across `sites`), not an approximation.
"""
function build_gate_layer_mpo(sites::Vector{<:QInfoTensor.SiteType{<:Any,Trivial}}, gate::AbstractMatrix; start::Int=1)
    L = length(sites)
    iseven(L) || throw(ArgumentError("build_gate_layer_mpo requires even L"))
    start in (1, 2) || throw(ArgumentError("start must be 1 or 2"))

    V = TensorKit.space(sites[1])
    T = eltype(gate)
    tensors = Vector{MPOTensor{T,ComplexSpace,Vector{T}}}(undef, L)
    i = 1

    id_tens = _identity_mpo_tensor(sites[1])
    left_tens, right_tens = _gate_to_mpo_tensors(gate, V)

    if start == 2
        tensors[1] = id_tens
        i = 2
    end

    while i <= L
        if i == L
            tensors[i] = id_tens
            i += 1
        else
            tensors[i], tensors[i+1] = left_tens, right_tens
            i += 2
        end
    end

    return QInfoTensor.MPO(tensors)
end

# ------------------------------------------------------------------------
# CircuitProblem: benchmark apply() over a trajectory of N circuit steps,
# rather than a single apply or repeated application of a Hamiltonian
# (which isn't physically meaningful — see problems.jl's
# HamiltonianApplyProblem for the deliberately-different single-apply
# design used there instead).
# ------------------------------------------------------------------------

struct CircuitProblem
    name::String
    L::Int
    n_steps::Int
    maxdim_values::Vector{Int}
    cutoff::Float64
end

const _CIRCUIT_MAXDIM_VALUES = [5, 10, 20, 40, 80, 160, 320]
const _CIRCUIT_CUTOFF = 1e-12

# n_steps chosen so the exact final bond dimension (4^n_steps) stays
# manageable while still generating enough entanglement to make a
# maxdim-vs-error curve interesting.
const circuit_L20 = CircuitProblem("circuit_L20", 20, 6, _CIRCUIT_MAXDIM_VALUES, _CIRCUIT_CUTOFF)
const circuit_L50 = CircuitProblem("circuit_L50", 50, 6, _CIRCUIT_MAXDIM_VALUES, _CIRCUIT_CUTOFF)

const CIRCUIT_PROBLEMS = (circuit_L20, circuit_L50)

"""
    build_quench_state(sites) -> MPS

Fully z-polarized product state (all "Up"), bond dimension 1 — the
standard initial state for a quench.
"""
build_quench_state(sites) = QInfoTensor.MPS(sites, fill("Up", length(sites)))

"""
    build_circuit_inputs(problem::CircuitProblem) -> (sites, ψ0, H_odd, H_even)

Build sites, the quench initial state, and the two layer MPOs
(canonicalized once — same fixed gates every step, "Floquet"). Not part
of the timed benchmark.
"""
function build_circuit_inputs(problem::CircuitProblem)
    sites = sitetypes(:SpinHalf, problem.L)
    ψ0 = build_quench_state(sites)
    H_odd = build_gate_layer_mpo(sites, _CIRCUIT_GATE; start=1)
    QInfoTensor.orthogonalize!(H_odd, 1)
    H_even = build_gate_layer_mpo(sites, _CIRCUIT_GATE; start=2)
    QInfoTensor.orthogonalize!(H_even, 1)
    return sites, ψ0, H_odd, H_even
end

"""
    run_circuit_trajectory(ψ0, H_odd, H_even, n_steps; kwargs...) -> MPS

Apply `n_steps` circuit steps (odd layer then even layer each step).
`kwargs` (`maxdim`, `cutoff`, etc.) are forwarded directly to `apply` —
deliberately NOT given separate `sweep_maxdim=nothing`/`sweep_cutoff=
nothing` defaults here, since that would override `apply`'s own smart
defaults (`2*maxdim`, `cutoff/10`) with `nothing` whenever the caller
only specifies `maxdim`/`cutoff`. This is the part actually timed by
`@benchmarkable`.
"""
function run_circuit_trajectory(ψ0, H_odd, H_even, n_steps; kwargs...)
    ψ = ψ0
    for _ in 1:n_steps
        ψ = QInfoTensor.apply(H_odd, ψ; kwargs...)
        ψ = QInfoTensor.apply(H_even, ψ; kwargs...)
    end
    return ψ
end