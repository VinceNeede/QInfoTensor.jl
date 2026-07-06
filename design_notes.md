# QInfoTensor.jl — design notes

Working notes for designing and implementing `QInfoTensor.jl`. Intended as
project knowledge so future chats start from this context instead of
re-deriving it. Written in the same style as the prototype (dense,
non-symmetric) library's own `opsum_mpo_design_notes.md` /
`dmrg_status_notes.md` / `apply_benchmark_notes.md` — this file should be
kept up to date the same way, alongside the code it documents.

**Status update**: core types, `SiteType`/`op`/`state`, MPS/MPO
containers, orthogonalization, compression, `OpSum`→MPO construction,
`inner`/`norm`/`normalize`, and zip-up `apply!` are all implemented and
tested (`Trivial` sites only — see "Current scope" below). This file has
been rewritten to reflect the actual, tested implementation rather than
the original pre-implementation plan; see git history for the original
version if the historical reasoning is needed.

## Why this project exists

The prototype library (dense `DenseTensor`, hand-rolled `Index`, `OpSum`,
FSM-based `MPO` construction, working DMRG and zip-up `apply!`) is
functionally complete and numerically validated, but has no path to
symmetric (charge-conserving, non-abelian) tensors. Rather than retrofit
block-sparse storage into the existing design, `QInfoTensor.jl` is a
ground-up rewrite on top of `TensorKit.jl`, which already solves symmetric
tensor storage and provides a `TensorOperations.jl`-compatible `@tensor`
macro that works identically on dense and symmetric tensors.

## Current scope: `Trivial` sites only

Everything implemented so far — `random_mps`, the product-state `MPS`
constructor, `OpSum`→`MPO` FSM construction — is restricted to `Trivial`
(unsymmetrized) sites at the **type level** (`SiteType{<:Any,Trivial}` in
argument signatures), not just by convention. Extending to `U1Irrep`/
`SU2Irrep` requires solving a real, currently-open design problem: a
nonzero-charge object (a charged operator like `S+`/`S-`, or a
charge-definite basis state like `:Up` under `U1Irrep`) cannot be embedded
as a bare, unadorned tensor — it needs a compensating/auxiliary leg
carrying the missing charge (concretely: an MPO virtual bond charge for
operators, an MPS boundary charge for states). This is the same
underlying issue in all three cases:

- `op(::SiteType{:SpinHalf,U1Irrep}, ::OpName{"S+"})` — not defined; the
  dense matrix would fail TensorKit's own `TensorMap` charge-conservation
  check if naively embedded (confirmed empirically).
- `statetensor(SiteType(:SpinHalf;sym=U1Irrep), StateName(:Up))` — throws
  `ArgumentError` (confirmed empirically): `:Up` carries nonzero `Sz`
  charge, but a bare rank-(1,0) tensor's domain is the charge-0 unit
  space, so there's no legal block for the data.
- `OpSum`'s FSM construction under symmetry would need each auxiliary FSM
  state to carry its own charge sector, determined by the net charge of
  the partial operator string it represents — undesigned.

Solve this once (likely: a general "charged tensor needs an auxiliary
leg" utility) and all three unlock together. Not attempted yet.

## Relationship to existing libraries

(Unchanged from original — see ITensor/MPSKit/TensorKit positioning.)
`OpSum`'s FSM construction and the `AbstractTensorTrain`/`MPS`/`MPO`
container split are the two biggest departures from a naive TensorKit
port, and both are documented in detail below.

## Core types — as actually implemented

### `SiteType{tag,Sym}` / `OpName{name}` / `StateName{name}`

Ported from the prototype's `tags.jl` almost unchanged, with one addition:
`SiteType` gained a second type parameter, `Sym<:TensorKit.Sector`,
defaulting to `Trivial` via a **keyword**, not a positional argument:

```julia
SiteType(:SpinHalf)                    # tag=:SpinHalf, Sym=Trivial
SiteType(:SpinHalf; sym=U1Irrep)       # tag=:SpinHalf, Sym=U1Irrep
SiteType(:Boson, 4)                    # tag=(:Boson,4), Sym=Trivial
SiteType(:Boson, 4; sym=U1Irrep)       # tag=(:Boson,4), Sym=U1Irrep
```

`sym` is a keyword specifically because a positional argument would
collide with `params...` (used for parametric tags like `:Boson`) —
discovered as a real bug in an earlier draft, fixed before merging.

`OpName`/`StateName` are unchanged from the prototype — symmetry is a
`SiteType`-only concern; `op`/`state` still return plain dense
`Matrix`/`Vector`.

A convenience plural constructor exists, mirroring ITensors.jl's
`siteinds`:
```julia
sitetypes(:SpinHalf, 10)                  # Vector{SiteType{:SpinHalf,Trivial}}
sitetypes(:SpinHalf, 10; sym=U1Irrep)
sitetypes(:Boson, 10, 4)                  # parametric tag, params after L
```

### `space(::SiteType)` and the `op`/`state` ↔ `TensorMap` boundary

`TensorKit.space(st::SiteType)` is extended per `(tag,Sym)` combination,
no `kwargs` (must be a pure function of `st` alone — this determinism
requirement is why `Sym` is baked into the type rather than passed at
call time, matching how `op`'s legitimate call-time variability, e.g. a
rotation angle, is a fundamentally different kind of parameter).

`optensor`/`statetensor` are the conversion layer: they take `op`/
`state`'s dense output and wrap it into a `TensorMap` via `space(st)`.
This is also where illegal-under-symmetry operators get rejected — **for
free**, via TensorKit's own `TensorMap` constructor validating charge
conservation, not bespoke logic. Confirmed empirically: an
off-diagonal-in-`Sz` matrix (like `Sx`) fed into `TensorMap(M, V←V)`
against a `U1Irrep`-graded `V` throws `ArgumentError: Data has non-zero
elements at incompatible positions`. This resolved the original open
question about `op`'s behavior under illegal symmetric combinations.

`:SpinHalf` (`qubit.jl`) is generic over `Sym ∈ {Trivial,U1Irrep}` for
every operator that's actually charge-conserving in the `Sz` basis (`Id`,
`Sz`, `S2`, projectors, `Rz`) — one dense matrix serves both symmetries,
confirmed to embed validly in both. Charge-mixing operators (`Sx`, `Sy`,
`Rx`, `Ry`, `S+`, `S-`) are `Trivial`-only, no `U1Irrep` method defined at
all (not just undefined-and-erroring — genuinely not expressible without
the "needs an auxiliary charge leg" mechanism from "Current scope" above).
`SU2Irrep` is not addressed anywhere — open question 1 from the original
design notes (multiplet/basis-change issue) remains unsolved.

### `MPSTensor{T,S,A}` / `MPOTensor{T,S,A}`

Concrete type aliases over `TensorMap` (not `AbstractTensorMap`, no
wrapper struct — TensorKit's `TensorMap{T,S,N1,N2,A}` already has
everything needed):

```julia
const MPSTensor{T,S<:ElementarySpace,A<:DenseVector{T}} = TensorMap{T,S,2,1,A}
const MPOTensor{T,S<:ElementarySpace,A<:DenseVector{T}} = TensorMap{T,S,2,2,A}
```

Confirmed via a real session: `typeof(t)` on both a `Trivial`- and a
`U1Irrep`-sector tensor shows exactly this 5-parameter shape, with `A`
(storage backend, e.g. `Vector{Float64}` or `CuVector{ComplexF64}`) fully
independent of `T` (scalar type) and `S` (symmetry, carried by the space
type). GPU support (if/when pursued) is expected to be transient/local to
specific operations (`adapt` in, compute, `adapt` back out) rather than a
container-wide storage swap — `A<:DenseVector{T}` means containers stay
CPU-resident always.

**MPSTensor leg convention**: `(left,phys) → right`, i.e. codomain
`(left,phys)`, domain `(right,)`. Unchanged from the original plan.

**MPOTensor leg convention — CHANGED from the original plan**: codomain
`(left,site_out)`, domain `(site_in,right)` — **not** `(right,site_in)` as
originally written. Flattened native order: `(left,site_out,site_in,right)`
— bond legs on the outside, both physical legs adjacent in the middle.
Changed specifically because it makes orthogonalization's per-site steps
plain contiguous `repartition` calls in both directions (the original
`(right,site_in)` order needed a genuine `permute` for one direction,
since `site_in` and `right` weren't adjacent). Unplanned bonus: this also
turned out to be exactly the natural flattening order for `OpSum`'s FSM
dense-array construction (`(S_prev,site_out,site_in,S_curr)` maps
directly onto it), confirming the change was right for reasons beyond the
original motivation.

`site_out`/`site_in` are **both plain `phys`-typed**, not one dualized —
this was deliberately checked (see "Duality rules learned the hard way"
below): duality bookkeeping belongs at the point of contraction (explicit
`conj()`), not baked into storage, or applying an operator would silently
change the physical leg's type on every application.

### `AbstractTensorTrain{T,S,A}` / `MPS` / `MPO`

Shared abstract supertype carries only structurally-universal,
semantics-free accessors (`length`, `getindex`, `eltype`, `spacetype`,
`storagetype`, `isortho`/`orthocenter` — the latter two are pure formulas
over `llim`/`rlim` so they're safe to share generically). **No stub
methods for `tensors`/`llim`/`rlim` on the abstract type** — Julia already
raises `MethodError` when a concrete type hasn't implemented one; an
explicit throwing stub adds nothing. `tensors`/`llim`/`rlim`/
`set_ortho_lims!` are defined separately on `MPS` and `MPO` even though
today the bodies coincide — deliberately not shared, specifically because
whether MPO's `llim`/`rlim` (used by zip-up) means the same thing as
MPS's is still an open question (see below); if MPS/MPO ever diverge here,
that's a one-method override, not an un-sharing of generic code.

Plain `MPS`/`MPO`, not `FiniteMPS`/`FiniteMPO` — no infinite-system
concept anywhere in this library.

`llim`/`rlim` semantics (still the original working definition, **not
independently re-verified against ITensorMPS.jl source** — flagged as
unverified from the start, still unverified): sites `1:llim` are
left-orthogonal, sites `rlim:L` are right-orthogonal, unique orthogonality
center at `llim+1` when `llim+1==rlim-1`, non-orthogonalized state is
`llim=0, rlim=L+1`.

**Open, unresolved**: whether MPO's `llim`/`rlim` bookkeeping (as used by
zip-up's `apply!`) means the same thing as MPS's, or needs different
semantics. `isortho`/`orthocenter` are inherited generically by MPO from
`AbstractTensorTrain` for now; revisit if zip-up ever needs something
different (see `apply!` notes — it currently just marks `set_ortho_lims!`
by hand rather than relying on the generic machinery meaning anything
during the sweep itself).

## Mutation convention (project-wide)

- **`f!(x)`** — mutates only the *container*: rebinds `tensors[i]` to a
  new `TensorMap`, updates `llim`/`rlim`/etc. Never reaches into an
  existing `TensorMap`'s storage. Safe to wrap with a shallow `copy()` for
  a non-mutating variant.
- **`f!!(x)`** — mutates a `TensorMap`'s *data* in place (`rmul!`, etc.).
  Not safe around a shallow `copy()`.
- **`copy(ψ)`** is shallow (new `Vector`, same `TensorMap` objects inside)
  and is only correct *because* every `f!` in the library holds to the
  above. No custom `deepcopy` — relies on Julia's automatic recursive
  behavior, unconfirmed whether that's clean for `TensorMap`'s internals
  (cached fusion-tree data etc.) but not yet exercised in a way that
  would reveal a problem.
- `orthogonalize`/`compress` (no bang) always go through the single-`!`
  path + `copy()`, never `!!`+`copy()` (unsafe) or `!!`+`deepcopy()` (safe
  but defeats the purpose).

Separate, unrelated fact worth not confusing with the above: TensorKit's
own `!` on factorization functions (`left_orth!`, `qr_compact!`, etc.)
means something different again — BLAS/LAPACK-style "may destroy the
input as scratch, always use the return value," not our project's `!`/`!!`
distinction. `orthogonalize!`/`compress!` use the **non-destructive**
factorizations (`left_orth`, `right_orth`, no trailing `!`) to preserve
safety under shallow `copy`; `orthogonalize!!`/`compress!!` use the
destructive ones for speed, correct only when the caller owns `ψ`
outright.

## Orthogonalization

`orthogonalize!(ψ, j)` / `orthogonalize!!(ψ, j)`: sweep from wherever
`llim`/`rlim` currently are toward site `j`. Implementation is two plain,
self-contained step functions per concrete type (`_leftstep!`/
`_rightstep!` for `MPS`, separately for `MPO`) called from one shared
`_orthogonalize!` skeleton — an earlier version tried to unify the two
directions via `Left()`/`Right()` dispatch-tag types, but the two
absorption `@tensor` calls still had to differ by hand regardless, so the
abstraction bought nothing and was reverted in favor of directness.

**MPS steps**: left step needs no `repartition` (native `(2,1)` split
already matches `left_orth`'s expectation); right step needs
`repartition(t,1,2)` first (bending `phys` from codomain into domain),
then `right_orth`, then `repartition` back. This asymmetry is inherent to
the leg convention, not a code-quality issue.

**MPO steps**: thanks to the corrected leg order (see above), *both*
directions are now plain `repartition` — left step wants
codomain=`(left,site_out,site_in)`/domain=`(right,)` (contiguous under
the native order), right step wants codomain=`(left,)`/domain=
`(site_out,site_in,right)` (also contiguous). Neither needs `permute`.

**Confirmed empirically** (a genuine risk given how much of this is
hand-derived leg bookkeeping): building a toy MPS tensor, running it
through the right-step's `repartition`/`right_orth`/`repartition`
sequence, and checking the result's bond leg is `===` the original space
object *and* satisfies the isometry condition numerically — both held.
This validated not just this one step but the general principle that
**bending a leg out and back across the codomain/domain boundary (same
leg, both directions) cancels the dual it introduces exactly** — a fact
relied on throughout `orthogonalize.jl` and `random_mps`.

## Compression

`compress!`/`compress!!(ψ; maxdim=nothing, cutoff=nothing, center=length(ψ))`
reuses the exact same `_orthogonalize!` skeleton — the only difference is
which factorize function gets passed in (SVD-based with a truncation
strategy, instead of plain QR). `reset_ortho_lims!` forces a full-chain
resweep first, since every bond needs revisiting to be truncated
regardless of `ψ`'s starting orthogonality.

`cutoff` is defined to match **ITensor's own convention exactly**
(confirmed against ITensor's docs): relative discarded-squared-weight,
`Σ(discarded σᵢ²)/Σ(all σᵢ²) < cutoff`. Built as
`truncrank(maxdim) & truncerror(rtol=sqrt(cutoff))`, deliberately
bypassing the convenience `trunc::NamedTuple` interface — confirmed from
`MatrixAlgebraKit` source that the `NamedTuple`'s bare `rtol`/`atol` route
to `trunctol` (a **per-value** threshold, wrong semantics) and its
`maxerror` routes to `truncerror(atol=...)` (**absolute**, also wrong) —
neither gives the relative-squared-weight quantity `cutoff` is documented
to mean. `truncrank`/`truncerror` are built directly and combined via `&`
(`TruncationIntersection`) instead. `alg=:svd` is not passed explicitly —
confirmed that presence of a non-trivial `trunc` alone makes
`left_orth`/`right_orth` auto-select SVD.

## `random_mps` / product-state `MPS` construction

Both `Trivial`-only (see "Current scope"). `random_mps` builds
mixed-canonical directly (no post-hoc `orthogonalize!` needed): left half
via `randn`+`left_orth` (no bending needed, matches the left-step
convention), right half via `randn`+`repartition`+`right_orth`+
`repartition` (matches the validated right-step pattern exactly, on
purpose — reusing a confirmed-correct sequence rather than inventing a
new one). Deliberately does **not** query the resulting bond space back
out of a factorization result — since `maxdim` caps every bond at
`≤ dim(left)*dim(phys)` *before* construction, the result's bond space is
provably identical to the space already chosen going in, sidestepping an
entire category of "does this accessor work the way I think" risk.
`TensorKit.randisometry` exists and was tried as a shortcut for the left
loop, then reverted in favor of the explicit `randn`+`left_orth` form for
code-style consistency between the two loops (randisometry is presumably
just sugar for the same operation anyway).

Product-state `MPS(sites, states)` sidesteps `statetensor` (which can't
carry bond legs) and goes through the dense `state(...)` vector directly,
reshaped into a `(d,1)` matrix and wrapped via the confirmed
`TensorMap(denseMatrix, cod←dom)` pattern. String/Symbol state-name entry
points (`MPS(sites, ["Up","Dn",...])`) are supported via a
`StateName.(states)` conversion wrapper.

## `OpSum` → `MPO` (FSM construction)

`OpTerm`/`OpSum`/`add!`/`+` ported from the prototype almost unchanged —
pure bookkeeping, no tensor-backend dependency. One real design
adjustment made under pressure from Julia's type system: `OpTerm.ops` is
stored as `Vector{Pair{Int,Any}}`, **not** the seemingly-more-precise
`Vector{Pair{Int,Tuple{OpName,NamedTuple}}}` — both `OpName` and
`NamedTuple` are `UnionAll`-backed, and `Vector`/`Pair` are *invariant* in
their type parameters (unlike `Tuple`, which is covariant) — a precisely
parametrized field type meant fighting Julia's dispatch/construction
rules at every call site for no actual benefit, since the precise types
only matter once, at `op(...)` dispatch time. `add!`'s low-level method
signature needed `Pair{Int,<:Tuple{OpName,NamedTuple}}...` (the `<:` is
load-bearing — bare `Pair{Int,Tuple{OpName,NamedTuple}}` fails to match
concrete literals like `1 => (OpName(:Sz), (;))`, a `Pair`-invariance
gotcha, not a bug in the concept).

`_fsm_states` is pure combinatorics on site indices (unchanged from the
prototype). `_fsm_site_tensor` builds a dense `(|S_prev|,d,d,|S_curr|)`
array exactly like the prototype (this part is backend-agnostic), then
`reshape`s and wraps it into an `MPOTensor`. **The reshape convention —
does Julia's natural column-major `reshape` of that dense array match
TensorKit's own multi-leg matrix-flattening convention — was explicitly
verified**, not assumed: an asymmetric operator placed at a known FSM
block was independently extracted back out via `@tensor` and matched
exactly (`scratch_opsum_reshape.jl`). `Trivial`-only (see "Current scope"
for why symmetric FSM-state charge assignment is deferred).

## `inner`/`norm`/`normalize` family

`inner(ψ,φ)` and `inner(ψ,H,φ)` extend `TensorKit.inner` (not a fresh
name) — convention is conjugate-linear in the first argument, linear in
the second (physics/Dirac convention), matching what a real session
confirmed for `dot`/`inner` on bare tensors. `norm`/`normalize`/
`normalize!`/`normalize!!` are **MPS-only, and require `isortho(ψ)`**
(throws otherwise, via `orthocenter`'s own existing check — no separate
check needed) — always the cheap O(1) orthocenter-based shortcut, no
O(L) fallback for non-orthogonal states. `normalize!!` uses `rmul!`
in-place, correctly flagged as `!!`-class.

MPO deliberately not covered by any of these yet: `inner(::MPO,::MPO)`
would need a different contraction (both `site_out`/`site_in` legs), and
"norm of an operator" is ambiguous (operator vs. Frobenius norm) — a
separate design discussion, not attempted here.

### Duality rules learned the hard way (worth not re-deriving from scratch)

Both confirmed via real scratch-script runs, not assumed:

1. **`conj()` on a tensor reference inside `@tensor` flips the effective
   duality of *every* leg of that reference**, including legs untouched
   by the contraction happening at that call site — carried through into
   whatever output references them. This is why `inner`'s bra tensor gets
   a *whole-reference* `conj(ψi[...])`, not a per-leg one.
2. **A domain leg pairs validly against a fresh/untouched plain leg
   without needing `conj`; a codomain leg doesn't** (confirmed via the
   very first scratch check, placing a known operator at a known FSM
   block and extracting it back out with one-hot selectors — the
   codomain-pairing selector needed `conj`, the domain-pairing one
   didn't).

These two facts fully explain a `SpaceMismatch` bug that took several
rounds to pin down in `inner(ψ,H,φ)`'s boundary environment `E`: `E`'s
MPO-bond leg was built into `E`'s *codomain*, but it needed to pair
(unconjugated on both sides) against `H`'s plain codomain leg — invalid,
since neither side is dual. Moving that leg into `E`'s *domain* (per
fact 2) fixed it with **plain `@tensor`, no `@tensoropt` needed** — the
earlier appearance that `@tensoropt` specifically was required (tried and
seemingly confirmed across multiple contraction orders before the real
fix was found) was a red herring: the bug was never about which order got
chosen, only about this one leg's placement. The same "read the actual
post-contraction domain spaces off the tensor itself, don't assume them"
principle applies to `apply!`'s right boundary combiner, built from
`domain(θ)` after the contraction rather than pre-computed.

Also confirmed: `space(t, i)` (single integer) *does* return the
individual `ElementarySpace` of leg `i` — usable and more readable than
`codomain(t)[j]`/`domain(t)[j]`, contrary to an earlier misdiagnosis in
this project's own history where a `HomSpace`-returning call was mistaken
for a per-leg accessor (that was actually `space(t)` with no index, a
different call entirely).

### `@tensor`/`@tensoropt` syntax notes

- **Semicolon-separated output with a single index before the `;`**
  requires **space**, not comma, between multiple indices after it (e.g.
  `Enew[rp; rh r]`, not `Enew[rp; rh, r]`) — likely because Julia's own
  array-literal grammar (`[1 2; 3 4]`, space-separated within a row)
  takes precedence when there's exactly one index before the semicolon,
  before `@tensor`'s own comma convention gets a chance to apply. With
  **multiple** indices before the `;` (e.g. `a, b; c`), no such ambiguity
  arises and comma works fine — confirmed both ways empirically. Not
  re-verified for the *multiple-before-multiple-after* case
  (`a, b; c, d`, used in the MPO orthogonalization steps) — flagged as an
  untested risk, not yet exercised by any test.
- **`@tensoropt`'s cost-hint system supports only one symbolic scaling
  parameter at a time** — multiple independent symbols (e.g. separate `χ`
  and `w`) crash deep in its cost-polynomial code
  (`MethodError: one(::Type{Power{_A,Int64} where _A})`). Confirmed by
  elimination across several variants. Workaround when genuinely needed:
  express a secondary scaling class as a fraction of the same symbol
  (`χ/8`), or mark only the large indices unlabeled (`(a, b, c)` — implicit
  "these are expensive, everything else is cheap") which sidesteps the
  issue entirely and is what `inner`/`apply!` actually use.
- Binding a Julia variable to a concrete integer value with the *same
  name* as a symbolic cost tag used elsewhere in the same scope is safe
  (tested, ruled out as a cause of the above) — the cost-hint symbols are
  processed as opaque tags at macro-expansion time, not evaluated.
- `@test_warn` cannot test `@warn`-generated warnings (a stdlib
  limitation) — use `@test_logs (:warn, r"...")` instead, where a `Regex`
  matches via `occursin` (substring semantics).

## `apply!`/`apply` (zip-up)

`apply!(H, ψ, Val(:zipup); maxdim, cutoff, sweep_maxdim=2*maxdim,
sweep_cutoff=cutoff/10)` — defaults follow Paeckel et al. 2019's
rationale (loose intermediate truncation during the sweep, refined by a
final `compress!` pass). Warns (via `@test_logs`-testable `@warn`, not
`@test_warn`) if `H` isn't left-canonicalized at site 1 first.

Loop structure: a running left-context tensor `R_left` (initially a
combiner fusing the MPO and MPS left-boundary bonds via `isomorphism`,
then the QR remainder from the previous site) is contracted with
`H[i]`/`ψ[i]` at every site — confirmed via a real `left_orth` run that
the QR remainder's shape (`(bond)←(rh_prev,r_prev)`, `N1=1,N2=2`) matches
the initial combiner's shape exactly, so the loop body is genuinely
identical at every site, not just similar. At the last site, the
combined `(rh,r)` domain legs get fused into `new_right` via a **second**
combiner — built from `domain(θ)` *after* the contraction, not
pre-computed from `H`/`ψ`'s original spaces, for the same duality reason
described above (`θ`'s domain legs come out dual; a combiner built from
the pre-contraction plain spaces would mismatch).

`compress!(ψ; maxdim, cutoff, center=1)` finishes the sweep, and its own
`reset_ortho_lims!` makes the preceding `set_ortho_lims!` call inside
`apply!` mostly a formality (marking state before the compress pass takes
over) rather than something `compress!` actually depends on being
correct.

## Open questions (updated)

1. **`SU2Irrep` dispatch table** — untouched, original open question 1
   stands as originally written.
2. **`op`'s illegal-combination behavior** — resolved: TensorKit's own
   `TensorMap` constructor rejects illegal (non-charge-conserving)
   dense matrices automatically; no method is defined for those
   `(SiteType,OpName)` pairs in the first place, so the rejection happens
   even earlier, at `op(...)` dispatch, with a plain `MethodError`.
3. **`@tensor`/factorization exact syntax** — resolved through extensive
   empirical work; see "Duality rules" and "syntax notes" above for the
   accumulated, hard-won facts. No longer a single open item; folded into
   documented project knowledge instead.
4. **`MPOTensor` leg order** — resolved, changed from the original plan
   (see "Core types" above), for reasons that turned out to generalize
   beyond the original motivation.
5. **Open quantum systems extension** — still fully deferred, no design
   work done, unchanged from the original plan.
6. **Charged operators / charged states / symmetric FSM construction** —
   the single biggest open item blocking any non-`Trivial` symmetry work;
   see "Current scope" above for the unified statement of the problem.
7. **MPO's `llim`/`rlim` semantics under zip-up** — still open (see
   "Core types" above); `apply!` currently works around this by
   `set_ortho_lims!`-ing directly rather than relying on the shared
   generic bookkeeping meaning anything mid-sweep.
8. **`a, b; c, d` (multiple-before-multiple-after) semicolon syntax in
   MPO orthogonalization steps** — written, never actually exercised by a
   test (no MPO orthogonalization test exists yet). Real, currently
   untested risk given the comma/space subtlety discovered elsewhere.

## Roadmap (actual order, for reference)

1. ~~Core types: `SiteType`, `space`, `MPSTensor`, `MPOTensor`, MPS/MPO
   containers with `llim`/`rlim` bookkeeping.~~ Done.
2. ~~`orthogonalize!`/`compress!` (not in the original roadmap's stated
   order, but built before `OpSum` since `random_mps` needed something to
   validate against).~~ Done.
3. ~~Port `OpSum` → `MPO` FSM construction.~~ Done, `Trivial`-only.
4. ~~`inner`/`norm`/`normalize`, zip-up `apply!`.~~ Done.
5. **DMRG3S** — not started. No direct MPSKit equivalent to lean on, per
   the original notes; now also has a real `apply!`/`inner` foundation to
   build the local update on top of.
6. **SRC** — not started, no design work done, start from Camaño,
   Epperly & Tropp 2025 Algorithm 1 as originally planned.
7. Open quantum systems extension — not started.
8. Symmetric (`U1Irrep`/`SU2Irrep`) support across the board — blocked on
   the charged-operator/charged-state problem in "Current scope" above.

## Repository / workflow conventions

(Unchanged from original — new standalone package, `MPSKit.jl` dev-only,
one commit per resolved design decision with rationale in the message,
feature branches squash-merged after numerical validation, this file kept
alongside the code it documents.)