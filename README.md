# QInfoTensor.jl

*A finite, symmetry-aware matrix product state / matrix product operator library for quantum information, built on [TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl).*

**Status:** core functionality implemented, tested, and benchmarked
(including a speed/memory comparison against ITensor) — `SiteType`/`op`/`state`
dispatch, `MPS`/`MPO` containers, orthogonalization, compression,
`OpSum`→`MPO` construction, `inner`/`norm`/`normalize`, and two `H|ψ⟩`
algorithms (zip-up and SRC, see "Planned algorithms" below).
**Currently restricted to `Trivial` (unsymmetrized)
sites** — see "Current scope" below. See `design_notes.md` for the full
implementation history, resolved design decisions, and open questions.

## What this is

QInfoTensor.jl combines two things that currently live in separate ecosystems:

- **ITensor's ergonomics for finite systems** — automatic Hamiltonian construction from sums of operator terms, sweep-based DMRG, gate/MPO application to circuits, entanglement/fidelity measures. This is the natural fit for quantum-information use cases: finite qubit registers, circuit simulation, specific-state questions — not the thermodynamic limit.
- **TensorKit.jl's tensor backend** — `TensorMap`s with native abelian *and* non-abelian symmetry support (a heavier lift to bolt onto ITensor's `Index`-based design), plus GPU support via `Adapt.jl`/`CuArray`, and a natural home for an eventual open-quantum-system extension.

MPSKit.jl already sits on TensorKit.jl, but its design center is the infinite/VUMPS case — fixed-point iteration, lazy multi-gauge caching — which serves different priorities than a finite, sequential, sweep-based library needs. QInfoTensor.jl uses MPSKit as a reference to learn from, not a dependency to build on.

## Current scope: `Trivial` sites only

Symmetric (`U1Irrep`/`SU2Irrep`) tensor support is a first-class design
goal (see "Design stance" below), and `SiteType`/`space` already carry a
symmetry-sector type parameter throughout the codebase. But `random_mps`,
the product-state `MPS` constructor, and `OpSum`→`MPO` construction are
currently restricted to `Trivial` sites at the type level, pending a
solved design for how a nonzero-charge object (a charged operator like
`S+`, a charge-definite basis state, an FSM auxiliary bond) gets a
compensating auxiliary leg. See `design_notes.md`, "Current scope," for
the full explanation — this is the single biggest blocker to symmetric
support and the next major piece of design work.

## Design stance

- Custom `MPSTensor`/`MPOTensor`/MPS/MPO container types, built directly on `TensorKit.TensorMap` (concrete type aliases, not wrapper structs).
- Single-orthogonality-center canonical-form bookkeeping (`llim`/`rlim`), matching ITensor's convention — not MPSKit's lazy `AL`/`AR`/`AC`/`CL`/`CR` multi-gauge cache, which exists to serve VUMPS's need for simultaneous gauges that this library doesn't have.
- Contraction primarily via the `@tensor`/`@tensoropt` macros (`TensorOperations.jl`) rather than `TensorMap`'s `*` composition operator — this gives labeled-index ergonomics close to ITensor's implicit matching, without requiring domain/codomain to line up for every contraction.
- Per-site physical spaces and operator dispatch via a `SiteType{tag, Sym}` parametric tag struct (`Sym` a `TensorKit.Sector`, defaulting to `Trivial`), extending the existing `OpName`/`op` dispatch convention to also determine the `TensorKit` space a site carries.
- A project-wide `!`/`!!` mutation convention: `f!` mutates only container structure (safe under a shallow `copy`), `f!!` mutates a `TensorMap`'s underlying data in place (faster, unsafe if storage is shared). See `design_notes.md` for the full rationale and how it interacts with TensorKit's own, differently-meaning `!` convention on factorization functions.

## Implemented so far

- **Core types**: `SiteType`/`OpName`/`StateName`, `space`/`op`/`state`/`optensor`/`statetensor`, `MPSTensor`/`MPOTensor`, `AbstractTensorTrain`/`MPS`/`MPO`.
- **Orthogonalization & compression**: `orthogonalize!`/`orthogonalize!!`/`orthogonalize`, `compress!`/`compress!!`/`compress` (ITensor-compatible `cutoff` semantics), for both `MPS` and `MPO`.
- **Construction**: `random_mps`, product-state `MPS(sites, states)` (accepts `StateName`, `String`, or `Symbol` state labels), `sitetypes` (bulk `SiteType` construction).
- **Hamiltonian construction**: `OpSum`/`add!`, FSM-based `MPO(::OpSum, sites)`.
- **Expectation values & overlaps**: `inner(ψ,φ)`, `inner(ψ,H,φ)`, `norm`/`normalize`/`normalize!`/`normalize!!` (MPS, requires an orthogonality center), `maxlinkdim`/`linkdims`/`linkinds`/`linkind` (bond-dimension introspection, generic over `AbstractTensorTrain`).
- **Operator application**: `apply!`/`apply` for `H|ψ⟩`, two algorithms — zip-up (`alg=:zipup`, following Paeckel et al. 2019 for default truncation parameters) and SRC (`alg=:src`, Successive Randomized Compression, Camaño, Epperly & Tropp 2025); `Trivial`-sector only for both, `:src` additionally has no `cutoff` (rank-only truncation, matching the paper's own methodology).
- **Benchmarking**: a `PkgBenchmark`-based suite (comparing `:zipup` vs. `:src` directly) and a speed/memory comparison against ITensor (see below).

## Benchmarking

A `PkgBenchmark` suite in `benchmark/` covers `apply!` over a brickwork
circuit trajectory, over single Hamiltonian application, and over a
synthetic random MPO/MPS problem (mirroring Camaño, Epperly & Tropp
2025's own Figure 1 setup — large, controllable `D`/`χ`, specifically to
test the regime where SRC's asymptotic advantage is claimed to show up,
which neither of the other two problems reaches). All three sweep both
`alg=:zipup` and `alg=:src`. Run via `julia benchmark/run_and_export.jl`;
results are written as timestamped markdown to `benchmark/results/`.
`benchmark/compare_git.jl` compares two git refs directly via
`PkgBenchmark.judge`.

On the random-tensor problem (`n=100`, `D=χ=50`), SRC's time advantage
over zip-up grows with the requested output bond dimension — roughly at
par at `maxdim=5`, ~1.7× faster at `maxdim=100` — qualitatively matching
the paper's own reported scaling. See `design_notes.md`'s "Benchmark
suite" section for the numerical pitfalls this uncovered (unnormalized
synthetic data causing genuine, non-bug bond-dimension collapse under a
relative truncation cutoff) before this comparison became meaningful.

`benchmark/compare_itensor.jl` compares against ITensor on the same
circuit benchmark (`Trivial` sites, single-threaded, controlled BLAS/
`Strided.jl` thread counts on both sides). Current result, reproduced
twice: QInfoTensor is consistently faster (~1.3×–2.6×, largest at small
bond dimension, shrinking as `maxdim` grows) and consistently lower
memory (~2.2×–2.6×, flat across the whole tested range). This is a
`Trivial`-sites comparison — see `design_notes.md` for caveats and for
where a symmetric-tensor comparison could show a more fundamental
advantage once that support lands.

## Planned algorithms

Ported from an existing dense (non-symmetric) prototype library, roughly in order of expected difficulty on this new backend:

1. ~~**Zip-up** MPO–MPS contraction — single-pass, sequential.~~ Implemented as `apply!(...,Val(:zipup))`.
2. ~~**SRC** (Successive Randomized Compression; Camaño, Epperly & Tropp 2025, arXiv:2504.06475).~~ Implemented as `apply!(...,Val(:src))` and benchmarked directly against zip-up (see "Benchmarking" above); not symmetry-preserving, consistent with current `Trivial`-only scope and the paper's own stated limitation (§3.7).
3. **DMRG3S** (Hubig, McCulloch, Schollwöck, Wolf 2015, arXiv:1501.05504) — strictly single-site DMRG with subspace expansion. Not yet started; MPSKit's closest analog (`changebonds` with `OptimalExpand`) is a separate step, not fused into the local update the way DMRG3S needs. Listed here out of the original expected-difficulty order — SRC (above) turned out to be tractable sooner — and now has a real `apply!`/`inner` foundation (both algorithms) to build the local update on top of.

## Longer-term direction

Extension toward open quantum system simulation (Lindblad/GKSL dynamics via vectorized density operators or a Lindbladian-as-MPO), once the closed-system algorithms above are working and symmetric-tensor support is unblocked. Reference points to check first: ITensor ecosystem's `TensorMixedStates.jl` and `LindbladMPO`.

## Installation

Not yet registered.

```julia
] add https://github.com/<your-org>/QInfoTensor.jl
```

## Dependencies

- [`TensorKit.jl`](https://github.com/QuantumKitHub/TensorKit.jl) — tensor backend, symmetry
- [`TensorOperations.jl`](https://github.com/Jutho/TensorOperations.jl) — `@tensor`/`@tensoropt` contraction macros

`MPSKit.jl` is used only in a development/benchmarking environment (for comparison scripts), never as a package dependency. No `LinearAlgebra` dependency — everything needed (`norm`, `normalize`, `rmul!`, etc.) is already re-exported by `TensorKit.jl`.

## Related projects

- [ITensor / ITensorMPS.jl](https://itensor.org) — the ergonomics/finite-system reference point
- [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl) — the symmetric-backend reference point (VUMPS-first design)
- [TensorMixedStates.jl](https://itensor.org/codes/) — open-quantum-system MPS on ITensor, worth revisiting once OQS work begins

## AI usage transparency

**Design direction, priorities, and final decisions belong to the
project's maintainer(s).** Claude (Anthropic's AI assistant) has been
used throughout this project's development, across many separate
sessions, but its role has been that of an interactive
collaborator/pair-programmer executing under the maintainer's direction —
not an independent author. Every design decision recorded in
`design_notes.md` reflects a choice the maintainer made and directed,
even where Claude helped draft the reasoning or the code that followed
from it.

Concretely, within that relationship, Claude's contributions have included:

- drafting implementations of core types and algorithms, iterated on
  interactively rather than generated once and accepted as-is;
- debugging real, specific failures — e.g. tracing a `SpaceMismatch`
  error back to an incorrect leg-convention assumption, or diagnosing a
  randomized-compression benchmark anomaly (a bond dimension silently
  collapsing to 1) down to floating-point behavior under a relative
  truncation cutoff applied to unnormalized synthetic data;
- writing and maintaining this README and `design_notes.md`, the latter
  specifically designed to capture *empirical* reasoning (confirmed
  behavior, resolved bugs, syntax gotchas) rather than just decisions;
- reviewing and optimizing performance-sensitive code, such as the SRC
  algorithm's prepass step.

A few things worth being explicit about:

- **Claude has no memory between sessions** unless prior context is
  explicitly re-supplied. This project's history spans many disconnected
  conversations, so this note describes the general *nature* of the
  collaboration rather than an exhaustive log of who-decided-what-when —
  `design_notes.md`'s own convention (one commit per resolved design
  decision, with rationale in the message, per "Repository / workflow
  conventions" there) is the closest thing to that detailed record, and
  reflects the maintainer's decisions at each step, not Claude's.
- Claims phrased as "confirmed empirically" or "confirmed via a real
  session" throughout `design_notes.md` reflect actual code that was
  actually run and checked, not just a plausible-sounding assertion —
  Claude proposes, implements, and explains; an actual execution whose
  output gets inspected (by the maintainer, or within a session via a
  real `bash`/Julia run) is what turns a claim into a confirmed one.

## Contributing / status

This project is in active interactive design. See `design_notes.md` for context before opening issues or PRs — it contains not just decisions but the empirical reasoning (confirmed TensorKit behavior, resolved bugs, syntax gotchas) behind them, much of which took real effort to pin down and is worth reading before re-deriving.