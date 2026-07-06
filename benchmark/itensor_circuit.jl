# ---------------------------------------------------------------------------
# ITensor side of the brickwork circuit — analogous to circuit.jl, shared
# between compare_itensor.jl (speed/memory benchmark) and
# verify_circuit_equivalence.jl (physical correctness check).
#
# ITensorMPS.apply(H::MPO, ψ::MPS; ...) defaults to alg="densitymatrix" —
# must explicitly request alg="zipup" to match QInfoTensor's apply
# algorithm (Stoudenmire & White 2010).
# ---------------------------------------------------------------------------

"""
    _itensor_gate_layer_mpo(sites, gate; start=1) -> MPO

ITensor analog of `build_gate_layer_mpo` (circuit.jl): same fixed gate,
decomposed via `ITensors.svd`. Unlike QInfoTensor (where every
`MPOTensor` is rank-4, boundaries included), in ITensor the first/last
tensor of the chain are rank-3 — no link to the left of the first tensor,
none to the right of the last. Fictitious (dim-1) links only appear
between consecutive INTERNAL segments, never at the two absolute ends.
`length(sites)` must be even.
"""
function _itensor_gate_layer_mpo(sites, gate::AbstractMatrix; start::Int=1)
    L = length(sites)
    T = permutedims(reshape(gate, 2, 2, 2, 2), (2, 1, 4, 3))   # (out1,out2,in1,in2)

    A = ITensorMPS.MPO(sites)
    i = 1

    if start == 2
        A[1] = ITensors.op("Id", sites[1])
        i = 2
    end

    while i <= L
        if i == L
            A[i] = ITensors.op("Id", sites[i])
            i += 1
        else
            gate_tensor = ITensors.ITensor(T, sites[i]', sites[i+1]', sites[i], sites[i+1])
            A[i], A[i+1] = ITensorMPS.factorize(gate_tensor, sites[i]', sites[i])
            i += 2
        end
    end

    return A
end

"""
    build_itensor_circuit_inputs(problem::CircuitProblem) -> (sites, ψ0, H_odd, H_even)

ITensor analog of `build_circuit_inputs` (circuit.jl): same structure,
same quench state, MPOs canonicalized once — outside the timed benchmark,
exactly as on the QInfoTensor side.
"""
function build_itensor_circuit_inputs(problem::CircuitProblem)
    sites = ITensorMPS.siteinds("S=1/2", problem.L)
    ψ0 = ITensorMPS.MPS(sites, fill("Up", problem.L))
    H_odd = _itensor_gate_layer_mpo(sites, _CIRCUIT_GATE; start=1)
    ITensorMPS.orthogonalize!(H_odd, 1)
    H_even = _itensor_gate_layer_mpo(sites, _CIRCUIT_GATE; start=2)
    ITensorMPS.orthogonalize!(H_even, 1)
    return sites, ψ0, H_odd, H_even
end

"""
    run_itensor_circuit_trajectory(ψ0, H_odd, H_even, n_steps; maxdim=nothing, cutoff=nothing,
                                    sweep_maxdim=nothing, sweep_cutoff=nothing) -> MPS

ITensor analog of `run_circuit_trajectory` (circuit.jl).

ITensor's zipup method has two distinct truncation levels (see
`ITensors.contract(::Algorithm"zipup", A::MPO, B::AbstractMPS; cutoff,
maxdim, mindim, truncate_kwargs=(;cutoff,maxdim,mindim), kwargs...)`):
- direct `cutoff`/`maxdim`/`mindim`: truncation during the zip-up
  contraction sweep itself — corresponds to QInfoTensor's
  `sweep_cutoff`/`sweep_maxdim`.
- `truncate_kwargs`: final compression pass — corresponds to
  QInfoTensor's `maxdim`/`cutoff`.

With `sweep_maxdim=nothing, sweep_cutoff=nothing` (default, matching
circuit.jl's usage), the direct parameters are ITensor's own "no
truncation" defaults, and all real truncation happens in
`truncate_kwargs` — exactly matching QInfoTensor's side.
"""
function run_itensor_circuit_trajectory(ψ0, H_odd, H_even, n_steps;
                                         maxdim=nothing, cutoff=nothing,
                                         sweep_maxdim=nothing, sweep_cutoff=nothing)
    apply_kwargs = (
        alg="zipup",
        cutoff=sweep_cutoff,
        maxdim=sweep_maxdim,
        mindim=1,
        truncate_kwargs=(cutoff=cutoff, maxdim=maxdim, mindim=1),
    )

    ψ = ψ0
    for _ in 1:n_steps
        ψ = ITensorMPS.apply(H_odd, ψ; apply_kwargs...)
        ψ = ITensorMPS.apply(H_even, ψ; apply_kwargs...)
    end
    return ψ
end
