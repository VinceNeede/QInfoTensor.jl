# QInfoTensor.jl — design notes

Working notes for designing and implementing `QInfoTensor.jl`. Intended as
project knowledge so future chats start from this context instead of
re-deriving it. Written in the same style as the prototype (dense,
non-symmetric) library's own `opsum_mpo_design_notes.md` /
`dmrg_status_notes.md` / `apply_benchmark_notes.md` — this file should be
kept up to date the same way, alongside the code it documents.

## Why this project exists

The prototype library (dense `DenseTensor`, hand-rolled `Index`, `OpSum`,
FSM-based `MPO` construction, working DMRG and zip-up `apply!`) is
functionally complete and numerically validated, but has no path to
symmetric (charge-conserving, non-abelian) tensors. Rather than retrofit
block-sparse storage into the existing design, `QInfoTensor.jl` is a
ground-up rewrite on top of `TensorKit.jl`, which already solves symmetric
tensor storage and provides a `TensorOperations.jl`-compatible `@tensor`
macro that works identically on dense and symmetric tensors.

The goal is specifically **ITensor's finite-system ergonomics and
quantum-information framing, on TensorKit's symmetric backend** — a
combination that doesn't currently exist as a single library. See the
"Relationship to existing libraries" section below for how this positions
against ITensor and MPSKit specifically.

## Relationship to existing libraries

- **ITensor / ITensorMPS.jl**: finite-system-first, general-purpose,
  mature. `OpSum`/`AutoMPO` for arbitrary Hamiltonian construction, `apply`
  for circuits, well-tested DMRG/TDVP. Abelian QN-`Index` support is solid;
  non-abelian symmetry is not natively supported the way TensorKit's
  category-theoretic domain/codomain structure supports it. This is the
  ergonomics and use-case reference point.
- **MPSKit.jl**: built on TensorKit.jl already, but its design center is
  the infinite/VUMPS case — `find_groundstate`, `leading_boundary`, and
  `approximate` are all "iterate a fixed-point map," because that's what
  translationally-invariant infinite systems require. Finite `DMRG` exists
  within that same framework, and `FiniteMPS` caches all of
  `AL`/`AR`/`AC`/`CL`/`CR` lazily (multi-gauge) because VUMPS needs
  simultaneous left/right gauges at a site — machinery this library's
  finite, single-orthogonality-center design doesn't need. **Zip-up
  specifically does not fit MPSKit's philosophy at all**: it's a one-shot
  sequential pass with no notion of "walk to a boundary" in the infinite
  case, so MPSKit has no equivalent and isn't likely to add one. MPSKit is
  used here as a reference for TensorKit usage patterns and algorithm
  ideas, never as a dependency.
- **TensorKit.jl**: the actual backend. `TensorMap{T,S,N1,N2}` is a
  morphism domain (N2 legs) → codomain (N1 legs); symmetry is encoded via
  the `Sector` type parameter of `GradedSpace` (`Trivial`, `U1Irrep`,
  `SU2Irrep`, ...). `@tensor` (via `TensorOperations.jl`) contracts by
  label, like ITensor's implicit matching, and does not require
  domain/codomain to align — domain/codomain mainly matters for
  `leftorth`/`rightorth`/`tsvd` and for `adjoint`/`dag` semantics.

## Decisions made

### `SiteType{tag, Sector}`

Extends the prototype library's existing `OpName{N}` zero-cost dispatch
tag convention up one level. `tag` plays the same role `SiteType` already
plays for `op` dispatch in the prototype library (e.g. `:SpinHalf`).
`Sector` is TensorKit's own sector type parameter (`Trivial`, `U1Irrep`,
`SU2Irrep`, ...), baked in as a **type parameter**, not stored instance
data — chosen for two reasons:

1. Zero-cost, dispatchable, consistent with the existing `OpName{N}`
   pattern.
2. Determinism: unlike `op`, which can legitimately return a different
   matrix depending on call-time kwargs (e.g. a rotation angle), `space`
   must return the *same* space every time for a given site, or bond
   tensors across the chain will develop `SpaceMismatch` errors (or worse,
   silently coincide in dimension and contract incorrectly). Baking the
   symmetry choice into the type prevents this class of bug structurally.

Planned counterpart to the existing `op` function:

```
space(st::SiteType{tag,Sector}; kwargs...) -> ElementarySpace
```

analogous to `op(st::SiteType, ::OpName{...}; kwargs...) -> Matrix`.

### `MPSTensor` leg convention

`(left, phys) → right`, i.e. codomain = `(left, phys)` (`N1=2`), domain =
`(right,)` (`N2=1`). Matches MPSKit's own convention. Chosen for
consistency with `llim`/`rlim` sweep-direction bookkeeping (see below),
not because containers are shared with MPSKit.

### `MPOTensor` leg convention

`(left, site_out) → (right, site_in)`, i.e. codomain = `(left, site_out)`
(`N1=2`), domain = `(right, site_in)` (`N2=2`). `site_out` is the bra/output
leg (the prototype library's "primed" leg), `site_in` the ket/input leg.
Chosen to mirror the MPS grouping (left + "outgoing" together, right +
"incoming" together). This mainly affects `dag`/`adjoint` and
`leftorth`/`rightorth` behavior, not everyday contraction, since `@tensor`
contracts by label regardless of domain/codomain assignment.

### Canonical form / orthogonality bookkeeping

**Single orthogonality center via `llim`/`rlim`**, matching the prototype
library's existing `DenseTensor`-based MPS and ITensor's convention.
Explicitly *not* MPSKit's lazy `AL`/`AR`/`AC`/`CL`/`CR` cache — that exists
to serve VUMPS's simultaneous-gauge requirement, which finite
DMRG/zip-up/DMRG3S/SRC don't share. Revisit only if/when a VUMPS-style
algorithm is ever added.

### Contraction idiom

Primarily `@tensor` (from `TensorOperations.jl`, which `TensorMap`
implements natively as a backend), not `TensorMap`'s `*` composition
operator. `*` requires domain(A) to structurally match codomain(B);
`@tensor` contracts by explicit label, closer to ITensor's ergonomics.
`@tensor`'s output still needs a domain/codomain split assigned at
construction (TensorKit has dedicated syntax for this) — confirm exact
syntax against current docs when contraction code is first written, rather
than assuming it from memory.

## Open questions (not yet decided)

1. **`space(::SiteType{tag,Sector}; kwargs...)` dispatch table.** Start
   with `Trivial` (plain `ComplexSpace(d)`, matches ITensor's default
   un-symmetrized case), then `U1Irrep` (graded, charge sectors), then
   `SU2Irrep` (non-abelian, fusion-rule-based — the biggest jump in mental
   model, since multiplets are stored as single blocks rather than
   individual `Sz` eigenstates).
2. **How `op` should behave when an operator isn't representable as a
   legal symmetric `TensorMap` under a site's `Sector`.** E.g. under
   `U1Irrep` (`Sz`-conservation), `Sx`/`Sy` individually aren't legal
   charge-conserving operators — only `S+`/`S-` (which carry nonzero
   charge themselves, domain/codomain in different sectors) and `Sz` are.
   Options: error at dispatch time for illegal combinations, or only
   define the legal operator subset per `Sector` and let missing methods
   be the error signal. Needs resolving before extending `op`'s dispatch
   table past `Trivial`.
3. **Exact `@tensor` output domain/codomain syntax** — confirm against
   current TensorKit docs when the first real contraction is written.
4. **`MPOTensor` index-order/cache-friendliness tradeoffs**, inherited
   from the prototype library's own unresolved question in
   `apply_benchmark_notes.md` (`(1,2,4,3)` vs `(1,3,4,2)`-style ordering
   for the `H*ψ` contraction). Revisit once benchmarking is possible on
   this backend.
5. **Open quantum systems extension** (deferred until closed-system
   algorithms are working): vectorized-ρ (doubled physical leg) vs.
   Lindbladian-as-MPO. No design work done yet. Reference points to check
   first: ITensor ecosystem's `TensorMixedStates.jl` and `LindbladMPO`.

## Roadmap

1. Core types: `SiteType`, `space`, `MPSTensor`, `MPOTensor`, MPS/MPO
   containers with `llim`/`rlim` bookkeeping.
2. Port `OpSum` → `MPO` FSM construction from the prototype library,
   adapted for `TensorMap`-valued operators instead of raw `Matrix`.
3. **Zip-up** MPO–MPS contraction (easiest port — algorithm already fully
   designed and validated in the prototype library; the new work is
   entirely "how does TensorKit want this expressed," not new algorithm
   design).
4. **DMRG3S** (harder — needs to fuse into a single-site local update and
   handle bond-dimension growth mid-sweep; no direct MPSKit equivalent to
   lean on).
5. **SRC** (hardest — no existing design even in the prototype library;
   start from Camaño, Epperly & Tropp 2025, arXiv:2504.06475, Algorithm 1).
6. Open quantum systems extension (see open question 5 above).

## Repository / workflow conventions

- New standalone package, not a branch of the prototype library — no
  shared dependency graph or core types, and the prototype library should
  stay stable/deployable while this is unstable.
- `MPSKit.jl` lives only in a dev/benchmark environment, never in
  `Project.toml` as a package dependency — keeps "reference only" true
  structurally, not just by intention.
- One commit per resolved design decision where practical; commit messages
  should carry the *rationale* (why this convention over the alternative),
  not just the diff — multiple-dispatch Julia code doesn't make that
  obvious from the diff alone.
- Feature branches for larger units (e.g. "zip-up working end-to-end"),
  squash-merged onto `main` once numerically validated against a known
  reference (ITensor output or exact diagonalization), the same way the
  prototype library's DMRG/`apply!` work was validated.
- This file lives in the repo and is updated alongside the code it
  documents, not maintained separately.
