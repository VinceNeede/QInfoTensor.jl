# ---------------------------------------------------------------------------
# Direct check: do ψ_qit (QInfoTensor) and ψ_it (ITensor) represent the same
# physical state?
#
# Materializes both MPS to a dense vector (2^L complex/real numbers) and
# compares via overlap (invariant to global phase, which has no physical
# meaning). Intended for small L (4-8), NOT the real benchmark problems
# (2^20, 2^50 are intractable as dense vectors).
#
# Flattening convention used for BOTH materializations, to keep them
# comparable: site 1 is the fastest-varying component in the flattened
# vector (site1 fastest, siteL slowest).
#
# Usage:
#   julia --project=benchmark benchmark/verify_circuit_equivalence.jl
# ---------------------------------------------------------------------------

using LinearAlgebra
using TensorKit
using QInfoTensor
using ITensors, ITensorMPS

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "circuit.jl"))
include(joinpath(@__DIR__, "itensor_circuit.jl"))

"""
    mps_to_dense(ψ::MPS) -> Vector

Materialize a QInfoTensor `MPS` to a dense vector (site1 fastest-varying).

`block(t, sector)` gives a `(dim(left)*dim(phys), dim(right))` matrix for
each `MPSTensor` `t` — `left` is listed first in the codomain, so it
varies fastest in that row index (same convention confirmed via the
OpSum FSM reshape validation). For site 1 (`left=ℂ^1`) this collapses
directly to `(d₁,χ₁)`, no `dropdims` needed.
"""
function mps_to_dense(ψ)
    L = length(ψ)
    E = ψ[1]
    for i in 2:L
        E = E * repartition(ψ[i], 1, 2)
        E = repartition(E, numind(E) - 1, 1)
    end
    return vec(block(E, Trivial()))
end

"""
    itensor_mps_to_dense(ψ, sites) -> Vector

Materialize an ITensorMPS.MPS to a dense vector, same convention as
`mps_to_dense`.
"""
function itensor_mps_to_dense(ψ, sites)
    T = prod(ψ)                  # single ITensor with all site indices
    A = Array(T, sites...)       # axis order = sites[1],...,sites[L]
    return vec(A)
end

"""
    compare_states(v_qit, v_it; label="") -> NamedTuple

Compare two vectors up to global phase (physically irrelevant): overlap
≈ 1 ⟹ same physical state. Also prints the raw ‖v_qit - v_it‖, useful
only once the overlap is already ≈1 (otherwise phase makes it uninformative).
"""
function compare_states(v_qit, v_it; label="")
    overlap = abs(dot(v_qit, v_it)) / (norm(v_qit) * norm(v_it))
    println("── $label ──")
    println("  ‖v_qit‖ = ", norm(v_qit), "   ‖v_it‖ = ", norm(v_it))
    println("  |⟨v_qit|v_it⟩| / (‖v_qit‖‖v_it‖) = ", overlap, "   (expected ≈ 1 if same physical state)")
    println("  ‖v_qit - v_it‖ (raw, phase-sensitive) = ", norm(v_qit - v_it))
    return (; overlap, raw_diff=norm(v_qit - v_it))
end

# ---------------------------------------------------------------------------
# Test 1: initial state ψ0 (no gates applied) — checks "Up" means the same
# physical thing in both libraries, before introducing any circuit complexity.
# ---------------------------------------------------------------------------

L = 4
sites_qit = sitetypes(:SpinHalf, L)
sites_it = ITensorMPS.siteinds("S=1/2", L)

ψ0_qit = build_quench_state(sites_qit)
ψ0_it = ITensorMPS.MPS(sites_it, fill("Up", L))

compare_states(mps_to_dense(ψ0_qit), itensor_mps_to_dense(ψ0_it, sites_it); label="ψ0 (initial state)")

# ---------------------------------------------------------------------------
# Test 2: after a single layer (start=1), no truncation — isolates the
# gate/MPO construction's correctness from the rest of the trajectory.
# ---------------------------------------------------------------------------

H_odd_qit = build_gate_layer_mpo(sites_qit, _CIRCUIT_GATE; start=1)
QInfoTensor.orthogonalize!(H_odd_qit, 1)
ψ1_qit = QInfoTensor.apply(H_odd_qit, ψ0_qit)

H_odd_it = _itensor_gate_layer_mpo(sites_it, _CIRCUIT_GATE; start=1)
ψ1_it = ITensorMPS.apply(H_odd_it, ψ0_it; alg="zipup", cutoff=0.0, maxdim=typemax(Int), mindim=1)

compare_states(mps_to_dense(ψ1_qit), itensor_mps_to_dense(ψ1_it, sites_it); label="after 1 layer (start=1)")

# ---------------------------------------------------------------------------
# Test 3: after one full step (odd then even), no truncation.
# ---------------------------------------------------------------------------

ψ2_qit = run_circuit_trajectory(ψ0_qit, H_odd_qit, let
    H = build_gate_layer_mpo(sites_qit, _CIRCUIT_GATE; start=2)
    QInfoTensor.orthogonalize!(H, 1)
    H
end, 1)

ψ2_it = run_itensor_circuit_trajectory(ψ0_it, H_odd_it, let
    H = _itensor_gate_layer_mpo(sites_it, _CIRCUIT_GATE; start=2)
    ITensorMPS.orthogonalize!(H, 1)
    H
end, 1)

compare_states(mps_to_dense(ψ2_qit), itensor_mps_to_dense(ψ2_it, sites_it); label="after 1 full step (odd+even)")
