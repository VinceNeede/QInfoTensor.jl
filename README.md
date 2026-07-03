# QInfoTensor.jl

*A finite, symmetry-aware matrix product state / matrix product operator library for quantum information, built on [TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl).*

**Status:** early design phase. Core types and conventions are being worked out interactively before implementation begins — nothing here is stable, and no code has been written yet. See `design_notes.md` for the current state of decided vs. open questions.

## What this is

QInfoTensor.jl combines two things that currently live in separate ecosystems:

- **ITensor's ergonomics for finite systems** — automatic Hamiltonian construction from sums of operator terms, sweep-based DMRG, gate/MPO application to circuits, entanglement/fidelity measures. This is the natural fit for quantum-information use cases: finite qubit registers, circuit simulation, specific-state questions — not the thermodynamic limit.
- **TensorKit.jl's tensor backend** — `TensorMap`s with native abelian *and* non-abelian symmetry support (a heavier lift to bolt onto ITensor's `Index`-based design), plus GPU support via `Adapt.jl`/`CuArray`, and a natural home for an eventual open-quantum-system extension.

MPSKit.jl already sits on TensorKit.jl, but its design center is the infinite/VUMPS case — fixed-point iteration, lazy multi-gauge caching — which serves different priorities than a finite, sequential, sweep-based library needs. QInfoTensor.jl uses MPSKit as a reference to learn from, not a dependency to build on.

## Design stance

- Custom `MPSTensor`/`MPOTensor`/MPS/MPO container types, built directly on `TensorKit.TensorMap`.
- Single-orthogonality-center canonical-form bookkeeping (`llim`/`rlim`), matching ITensor's convention — not MPSKit's lazy `AL`/`AR`/`AC`/`CL`/`CR` multi-gauge cache, which exists to serve VUMPS's need for simultaneous gauges that this library doesn't have.
- Contraction primarily via the `@tensor` macro (`TensorOperations.jl`, implemented natively by `TensorMap`) rather than `TensorMap`'s `*` composition operator — this gives labeled-index ergonomics close to ITensor's implicit matching, without requiring domain/codomain to line up for every contraction.
- Per-site physical spaces and operator dispatch via a `SiteType{tag, Sector}` parametric tag struct, extending the existing `OpName`/`op` dispatch convention to also determine the `TensorKit` space (trivial, abelian, or non-abelian symmetry) a site carries.

## Planned algorithms

Ported from an existing dense (non-symmetric) prototype library, roughly in order of expected difficulty on this new backend:

1. **Zip-up** MPO–MPS contraction — single-pass, sequential. Doesn't fit MPSKit's iterative `approximate` framework at all, so this is a genuinely new addition rather than a reimplementation of something MPSKit already offers.
2. **DMRG3S** (Hubig, McCulloch, Schollwöck, Wolf 2015, arXiv:1501.05504) — strictly single-site DMRG with subspace expansion. MPSKit's closest analog (`changebonds` with `OptimalExpand`) is a separate step, not fused into the local update the way DMRG3S needs.
3. **SRC** (Successive Randomized Compression; Camaño, Epperly & Tropp 2025, arXiv:2504.06475) — not yet designed even in the prototype library; needs first-principles design work from the paper before porting.

## Longer-term direction

Extension toward open quantum system simulation (Lindblad/GKSL dynamics via vectorized density operators or a Lindbladian-as-MPO), once the closed-system algorithms above are working. Reference points to check first: ITensor ecosystem's `TensorMixedStates.jl` and `LindbladMPO`.

## Installation

Not yet registered.

```julia
] add https://github.com/<your-org>/QInfoTensor.jl
```

## Dependencies

- [`TensorKit.jl`](https://github.com/QuantumKitHub/TensorKit.jl) — tensor backend, symmetry
- [`TensorOperations.jl`](https://github.com/Jutho/TensorOperations.jl) — `@tensor` contraction macro

`MPSKit.jl` is used only in a development/benchmarking environment (for comparison scripts), never as a package dependency.

## Related projects

- [ITensor / ITensorMPS.jl](https://itensor.org) — the ergonomics/finite-system reference point
- [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl) — the symmetric-backend reference point (VUMPS-first design)
- [TensorMixedStates.jl](https://itensor.org/codes/) — open-quantum-system MPS on ITensor, worth revisiting once OQS work begins

## Contributing / status

This project is in active interactive design. See `design_notes.md` for context before opening issues or PRs.
