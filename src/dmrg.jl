# Default KrylovKit knobs when a sweep doesn't override eigsolve_kwargs;
# merged in `dmrg!` below (matches the prototype's own default).
const _DEFAULT_EIGSOLVE_KWARGS = (krylovdim=6, maxiter=5)

"""
    ProjMPO{ET,N}
 
Windowed effective-Hamiltonian cache for DMRG. `N ∈ {1,2}` is the number of
active sites per local update. `env[i]` holds the left environment through
site `i` while `i ≤ lpos`, or the right environment through site `i` while
`i ≥ rpos`. Construct with `ProjMPO(H, nsite)`, advance with [`position!`](@ref).
"""
mutable struct ProjMPO{ET,N}
    H::MPO
    env::Vector{ET}
    lpos::Int
    rpos::Int
    function ProjMPO(H::MPO, env::Vector{ET}, nsite::Int, lpos::Int, rpos::Int) where {ET}
        nsite ∉ (1, 2) && error("nsite must be 1 or 2")
        return new{ET,nsite}(H, env, lpos, rpos)
    end
end
 
nsite(::ProjMPO{ET,N}) where {ET,N} = N
Base.length(P::ProjMPO) = length(P.H)
 
# --- Environment tensor type -------------------------------------------
#
# CONFIRMED — this is literally `inner(ψ,H,φ)`'s own boundary environment,
# specialized to bra==ket==ψ (DMRG wants ⟨ψ|H_eff|ψ⟩-shaped environments,
# not two different states). Split, taken directly from the real
# `TensorKit.inner(ψ,H,φ)` source:
#   codomain(E) = (bra_bond,)            — N1 = 1
#   domain(E)   = (mpo_bond, ket_bond)   — N2 = 2
const EnvTensor{T,S,A} = TensorMap{T,S,1,2,A}
const TwoSiteTensor{T,S,A} = TensorMap{T,S,3,1,A}

function ProjMPO(H::MPO{T, S, A}, nsite::Int=2) where {T, S, A}
    L = length(H)
    ET = EnvTensor{T, S, A}
    return ProjMPO(H, Vector{ET}(undef, L), nsite, 0, L + 1)
end
 
function _rangeP(P::ProjMPO{ET,1}, forward::Bool) where {ET}
    L = length(P)
    return forward ? (1:(L-1)) : (L:-1:2)
end
function _rangeP(P::ProjMPO{ET,2}, forward::Bool) where {ET}
    L = length(P)
    return forward ? (1:(L-1)) : ((L-1):-1:1)
end
 
# --- Seeding / extending the environment --------------------------------
#
# CONFIRMED (left case) — lifted directly from the real `inner(ψ,H,φ)`
# source, specialized to bra==ket==ψ. Boundary MPS/MPO tensors carry an
# explicit dim-1 leg, so the seed is just a trivial `ones` map between
# those three dim-1 spaces — no `isomorphism`/edge-branch needed, and no
# special-casing inside the sweep loop itself: `E_0` (this seed) is fed
# into the *same* `_extend_env` as every other site, it doesn't produce
# `P.env[1]` directly (that was a bug in the first draft — fixed in
# `position!` below).
#
# Right case (`edge=:right`) is a mirror guess, NOT lifted from working
# code — `inner` only ever sweeps left-to-right into a scalar, so there's
# no precedent for a right-moving seed in your codebase yet. Needs a real
# Hermiticity check before trusting it, same bar as everything else new
# here — mirror-symmetry of the diagram doesn't by itself guarantee the
# domain/codomain assignment stays legal (per the "domain pairs without
# conj, codomain doesn't" asymmetry already documented for E elsewhere).
function _seed_env(ψ::MPS, H::MPO, ::Val{:left})
    T = promote_type(eltype(ψ), eltype(H))
    Vbra = space(ψ[1], 1)     # leg 1 = codomain leg 1 = left bond
    Vmpo = space(H[1], 1)
    Vket = Vbra                # same state on both sides for DMRG
    return ones(T, Vbra ← (Vmpo ⊗ Vket))
end
 
function _seed_env(ψ::MPS, H::MPO, ::Val{:right})
    T = promote_type(eltype(ψ), eltype(H))
    Vbra = space(ψ[end], 3)   # leg 3 = domain leg 1 = right bond
    Vmpo = space(H[end], 4)   # leg 4 = domain leg 2 = right bond
    Vket = Vbra
    return ones(T, Vbra ← (Vmpo ⊗ Vket))
end
 
"""
    _extend_env(E, ψi, Hi, ::Val{:left}) -> EnvTensor
 
Contract one more site into a left-moving environment. CONFIRMED leg
convention (matches the real `inner(ψ,H,φ)` source exactly, with
bra==ket==ψi): `E`'s legs are `(bra=lp; mpo=lh, ket=l)`; `Hi`'s codomain
carries `(lh, sp)` — `sp` is the bra/site_out leg, paired against
`conj(ψi)` — and its domain carries `(s, rh)` — `s` is the ket/site_in
leg, paired against the plain `ψi` (domain-pairs-without-conj, per the
established rule). This was backwards in the first draft of this file.
"""
function _extend_env(E::EnvTensor, ψi::MPSTensor, Hi::MPOTensor, ::Val{:left})
    @tensoropt (lp, rp, l, r) begin
        Enew[rp; rh r] := E[lp; lh l] * conj(ψi[lp, sp, rp]) * Hi[lh, sp, s, rh] * ψi[l, s, r]
    end
    return Enew
end
 
"""
    _extend_env(E, ψi, Hi, ::Val{:right}) -> EnvTensor
 
Mirror of the `:left` method for a right-moving environment (absorbing
site `i` from the right, i.e. `ψi`'s right bond is the already-accumulated
side, its left bond is fresh). UNVERIFIED — see the note on `_seed_env`'s
`:right` method above; same Hermiticity check needed before trusting this.
"""
function _extend_env(E::EnvTensor, ψi::MPSTensor, Hi::MPOTensor, ::Val{:right})
    @tensoropt (lp, rp, l, r) begin
        Enew[lp; lh l] := E[rp; rh r] * conj(ψi[lp, sp, rp]) * Hi[lh, sp, s, rh] * ψi[l, s, r]
    end
    return Enew
end
 
"""
    position!(P::ProjMPO, ψ::MPS, pos::Int) -> ProjMPO
 
Extend/shrink cached environments so the active window covers
`pos : pos + nsite(P) - 1`. Direct port of the prototype's incremental
lpos/rpos logic — no TensorKit-specific change expected here beyond
`_extend_env`'s own internals and the boundary seeding above.
"""
function position!(P::ProjMPO, ψ::MPS, pos::Int)
    L = length(ψ)
    lpos_target = pos - 1
    rpos_target = pos + nsite(P)
 
    for i in (P.lpos+1):lpos_target
        Eprev = i == 1 ? _seed_env(ψ, P.H, Val(:left)) : P.env[i-1]
        P.env[i] = _extend_env(Eprev, ψ[i], P.H[i], Val(:left))
    end
    P.lpos = lpos_target
 
    for i in (P.rpos-1):-1:rpos_target
        Eprev = i == L ? _seed_env(ψ, P.H, Val(:right)) : P.env[i+1]
        P.env[i] = _extend_env(Eprev, ψ[i], P.H[i], Val(:right))
    end
    P.rpos = rpos_target
 
    return P
end
 
# --- Local tensor extraction ---------------------------------------------
 
_local_tensor(::ProjMPO{ET,1}, ψ::MPS, pos::Int) where {ET} = ψ[pos]
 
function _local_tensor(::ProjMPO{ET,2}, ψ::MPS, pos::Int) where {ET}
    # Merge two adjacent MPSTensors into one rank-4 object:
    #   codomain = (left, site1, site2), domain = (right,)
    # generalizing MPSTensor's own "everything but right goes in codomain"
    # convention to two sites. CONFIRM this is the split you actually want
    # before building _update_local_tensors! around it — the alternative,
    # codomain=(left,site1)/domain=(site2,right), avoids a repartition on
    # write-back for the right-orthogonal branch, mirroring the exact
    # asymmetry orthogonalize! already had to solve once for MPSTensor.
    @tensor phi2[l s1 s2; r] := ψ[pos][l s1; b] * ψ[pos+1][b s2; r]
    return phi2
end
 
# --- Effective Hamiltonian application (the `product` function) ---------
 
"""
    _env(P, ψ, boundarypos, ::Val{:left})  -> EL (cached, or trivial seed at the edge)
    _env(P, ψ, boundarypos, ::Val{:right}) -> ER (cached, or trivial seed at the edge)

Shared "seed-or-cached" fetch used by `heff` and `_dmrg3s_perturbation` —
`boundarypos` is whichever site index actually sits at the chain edge for
the caller (`pos` for a left environment, the window's last site for a
right one), so nsite=2's `heff` can pass `pos+1` for its `ER` lookup.
"""
_env(P::ProjMPO, ψ::MPS, boundarypos::Int, ::Val{:left}) =
    boundarypos == 1 ? _seed_env(ψ, P.H, Val(:left)) : P.env[P.lpos]
_env(P::ProjMPO, ψ::MPS, boundarypos::Int, ::Val{:right}) =
    boundarypos == length(P) ? _seed_env(ψ, P.H, Val(:right)) : P.env[P.rpos]

"""
    heff(P::ProjMPO, pos::Int, ψ::MPS, ϕ) -> ϕ'
 
Applies the effective Hamiltonian to the local tensor `ϕ` (matching
`nsite(P)` sites, window starting at `pos`), returning a tensor in the
SAME space as `ϕ` — required for `eigsolve` to see a well-defined linear
map. Unlike the prototype, no `_align`/vec-reshape step is expected to be
needed: writing the output labels to literally match `ϕ`'s own labels
should be enough for TensorKit to produce a tensor of the right space
directly.
 
`ψ` is only needed to build the boundary seed at `pos ∈ {1,L}` (via
`_env`, same object `_extend_env`/`position!` use) — this replaces
the first draft's hand-written, buggy boundary contractions (which
dropped/misassigned `Hi`'s own left/right leg) with a single uniform
formula, same principle as the `position!` fix above.
"""
function heff(P::ProjMPO{ET,1}, pos::Int, ψ::MPS, phi::MPSTensor) where {ET}
    Hi = P.H[pos]
    EL, ER = _env(P, ψ, pos, Val(:left)), _env(P, ψ, pos, Val(:right))
    @tensoropt (l, r, li, ri) begin
        Hphi[l s; r] := EL[l; lh li] * Hi[lh s; si rh] * phi[li si; ri] * ER[r; rh ri]
    end
    return Hphi
end
 
function heff(P::ProjMPO{ET,2}, pos::Int, ψ::MPS, phi2::TwoSiteTensor) where {ET}
    H1, H2 = P.H[pos], P.H[pos+1]
    EL, ER = _env(P, ψ, pos, Val(:left)), _env(P, ψ, pos+1, Val(:right))
    @tensoropt (l, r, li, ri) begin
        Hphi2[l s1 s2; r] := EL[l; lh li] * H1[lh s1; si1 mid] * H2[mid s2; si2 rh] *
                              phi2[li si1 si2; ri] * ER[r; rh ri]
    end
    return Hphi2
end
 
# ----------------------------------------------------------------------
# Local update / write-back
# ----------------------------------------------------------------------
 
# nsite=2: plain tsvd, reusing the existing `_svd_truncation_strategy` —
# lowest-risk part of this whole file, structurally identical to a single
# orthogonalize!/compress! step.
function _update_local_tensors!(::ProjMPO{ET,2}, ψ::MPS, phi2::TwoSiteTensor, pos::Int, forward::Bool;
                                 maxdim=nothing, cutoff=nothing) where {ET}
    strategy = _svd_truncation_strategy(maxdim, cutoff)  # existing helper, per compress.jl
    phi2p = permute(phi2, ((1, 2), (3, 4)))
    U, S, Vh, truncerr = svd_trunc(phi2p; trunc=strategy)
 
    if forward
        ψ.tensors[pos] = U                              # already legal MPSTensor shape, no repartition needed
        SVh = S * Vh                                     # still codomain=(bond,), domain=(s2,r)
        ψ.tensors[pos+1] = permute(SVh, ((1, 2), (3,)))  # -> codomain=(bond,s2), domain=(r,)
        set_ortho_lims!(ψ, pos, pos+2)
    else
        ψ.tensors[pos] = U * S                           # still codomain=(l,s1), domain=(bond,) — already legal
        ψ.tensors[pos+1] = permute(Vh, ((1, 2), (3,)))    # same shape fix, no S involved on this side
        set_ortho_lims!(ψ, pos-1, pos+1)
    end
    return truncerr
end
 
# nsite=1 (DMRG3S): eigensolve result written straight back, no local
# truncation — bond dimension only ever grows via subspace expansion.
# THE NOISE / SUBSPACE-EXPANSION STEP IS NOT DRAFTED HERE.
#
# It's the one genuinely open piece: the prototype's `hcat`/`vcat` of a
# noise-scaled H-perturbation onto the local matrix has no direct
# TensorMap equivalent I'd trust without checking. Planned approach,
# following the exact precedent SRC already set for an unrepresentable
# batched contraction:
#   1. Compute the perturbation tensor via `heff`-style partial contraction
#      (env on one side only, matching the prototype's `P.env[P.lpos]*ϕ*Hi`).
#      This has an extra MPO-bond leg (`rh`/`wr`) beyond ϕ's own shape.
#   2. Drop to plain `Array` for ϕ and the perturbation (safe: Trivial-only
#      scope, same justification SRC used for its batched outer product).
#   3. `cat` along the matching bond axis, same as the prototype's
#      `hcat`/`vcat`.
#   4. Re-wrap into a TensorMap with an ENLARGED bond space — needs
#      TensorKit's direct-sum space combinator (confirm exact name/call;
#      I believe this exists but haven't verified it against current docs).
#   5. `tsvd`/`leftorth`-style factorize back down to `maxdim`, same
#      truncation strategy as :twosite.
#
# This needs a real interactive session against TensorKit before being
# written for real — flagging rather than guessing at step 4's API.

"""
    _dmrg3s_perturbation(P, ψ, pos, ϕ, ::Val{:left})  -> Pert   # grows ϕ's right bond
    _dmrg3s_perturbation(P, ψ, pos, ϕ, ::Val{:right}) -> Pert   # grows ϕ's left bond

Builds the DMRG3S/noise-term perturbation tensor: H acting on the local
site with only a partial environment, leaving one extra MPO-bond leg
beyond ϕ's own shape, to be fused into a new candidate bond direction in
`_dmrg3s_noise`.

Genuinely different contractions, not a relabeling: `phys` always sits in
ϕ's codomain (with `left`), so growing the right bond (`:left`, using
`EL`) puts the extra leg and `phys` on opposite sides "for free" — no leg
here ever crosses the codomain/domain boundary — while growing the left
bond (`:right`, using `ER`) puts them on the SAME side, forcing `phys`
into an intermediate, non-MPSTensor-legal domain placement that
`_dmrg3s_noise` corrects afterward via a second, canceling crossing (same
"bend out, bend back in" pattern used everywhere else in this file).

`:left` is a close cousin of `heff`'s own confirmed contraction pattern.
`:right` is genuine new derivation AND depends on the still-unverified
right-moving environment (`_seed_env`/`_extend_env` `Val{:right}`) —
wants a real numerical check before being trusted, more so than anything
else here.
"""
function _dmrg3s_perturbation(P::ProjMPO, ψ::MPS, pos::Int, ϕ::MPSTensor, ::Val{:left})
    Hi = P.H[pos]
    EL = _env(P, ψ, pos, Val(:left))
    @tensoropt Pert[lp, s'; rh, r] := EL[lp; lh l] * Hi[lh, s'; s, rh] * ϕ[l, s; r]
    return Pert
end
function _dmrg3s_perturbation(P::ProjMPO, ψ::MPS, pos::Int, ϕ::MPSTensor, ::Val{:right})
    Hi = P.H[pos]
    ER = _env(P, ψ, pos, Val(:right))
    @tensoropt Pert[lh, li; s, r] := Hi[lh, s; si, rh] * ϕ[li, si; ri] * ER[r; rh ri]
    return Pert
end

"""
    _expansion_injectors(T, V1, Vpert) -> (inj1, inj2, combiner)

Shared piece of `_dmrg3s_noise`'s two directions: fuses the perturbation's
extra legs (`Vpert`) into a single new space, direct-sums it against `V1`
(ϕ's own existing bond, whichever side is being grown), and returns the
isometric injections into the combined space plus the fusing combiner.
"""
function _expansion_injectors(T, V1, Vpert)
    V2 = fuse(Vpert)
    V_sum = V1 ⊕ V2
    return isometry(T, V_sum ← V1), isometry(T, V_sum ← V2), isomorphism(T, V2 ← Vpert)
end

"""
    _dmrg3s_noise(P, ψ, ϕ, pos, α, ::Val{:left})  -> (ϕ', inj1)   # grows the right bond
    _dmrg3s_noise(P, ψ, ϕ, pos, α, ::Val{:right}) -> (ϕ', inj1)   # grows the left bond

Subspace-expansion write: embeds `ϕ` and the (fused, α-scaled)
perturbation into a common enlarged bond space via a pair of isometric
injections, then sums — this is what actually grows the bond dimension,
since a plain single-site update alone cannot. Also returns `inj1` — the
caller needs it to reconcile the NEIGHBORING site's still-old-sized bond
before absorbing the SVD remainder into it (see `_dmrg3s_residual`).

`:left` can embed `ϕ` via plain composition (`* inj1'`) since the growing
leg is ϕ's WHOLE domain; `:right` needs `@tensor` (the growing leg is
only part of a composite codomain) plus a shape-fixing `permute` at the
end — the second of two deliberate, canceling crossings on `phys` (the
first happened inside `_dmrg3s_perturbation` itself).
"""
function _dmrg3s_noise(P::ProjMPO, ψ::MPS, ϕ::MPSTensor, pos::Int, α::Real, ::Val{:left})
    T = promote_type(eltype(P.H), eltype(ψ), eltype(ϕ))
    perturbation = lmul!(α, _dmrg3s_perturbation(P, ψ, pos, ϕ, Val(:left)))
    inj1, inj2, combiner = _expansion_injectors(T, domain(ϕ)[1], domain(perturbation))
    return ϕ * inj1' + perturbation * combiner' * inj2', inj1
end
function _dmrg3s_noise(P::ProjMPO, ψ::MPS, ϕ::MPSTensor, pos::Int, α::Real, ::Val{:right})
    T = promote_type(eltype(P.H), eltype(ψ), eltype(ϕ))
    perturbation = lmul!(α, _dmrg3s_perturbation(P, ψ, pos, ϕ, Val(:right)))
    inj1, inj2, combiner = _expansion_injectors(T, codomain(ϕ)[1], codomain(perturbation))
    @tensoropt ϕnew[a s; r] := inj1[a, l] * ϕ[l, s; r]
    return ϕnew + permute(inj2 * combiner * perturbation, ((1, 2), (3,))), inj1
end

"""
    _dmrg3s_residual(::Val{:left}, S, Vh, inj1)   -> S*Vh(*inj1)   # absorbed into ψ[pos+1]
    _dmrg3s_residual(::Val{:right}, U, S, inj1)   -> (inj1'*)U*S   # absorbed into ψ[pos-1]

Builds the tensor absorbed into the neighboring site. The factor whose
domain/codomain sits in the possibly-enlarged `V_sum` gets restricted back
down onto `V1` via `inj1`/`inj1'` BEFORE touching the neighbor at all —
equivalent to lifting the neighbor into `V_sum` first and contracting the
full factor against it, by associativity, but cheaper (never materializes
a `V_sum`-sized copy of the neighbor). `inj1 === nothing` (no noise this
step) skips the restriction entirely, since dimensions already match.
"""
_dmrg3s_residual(::Val{:left}, S, Vh, ::Nothing) = S * Vh
_dmrg3s_residual(::Val{:left}, S, Vh, inj1) = S * Vh * inj1
_dmrg3s_residual(::Val{:right}, U, S, ::Nothing) = U * S
_dmrg3s_residual(::Val{:right}, U, S, inj1) = inj1' * U * S

function _update_local_tensors!(P::ProjMPO{ET,1}, ψ::MPS, ϕ::MPSTensor, pos::Int, forward::Bool;
                                 maxdim=nothing, cutoff=nothing, noise=nothing) where {ET}
    strategy = _svd_truncation_strategy(maxdim, cutoff)
    dir = forward ? Val(:left) : Val(:right)
    inj1 = nothing
    if !isnothing(noise)
        ϕ, inj1 = _dmrg3s_noise(P, ψ, ϕ, pos, noise, dir)
    end
    if forward
        U, S, Vh, ϵ = svd_trunc(ϕ; trunc=strategy)
        ψ.tensors[pos] = U
        R = _dmrg3s_residual(dir, S, Vh, inj1)
        @tensoropt ψnext[l s; r] := R[l, l'] * ψ[pos+1][l', s; r]
        ψ.tensors[pos+1] = ψnext
        set_ortho_lims!(ψ, pos, pos+2)
    else
        # phys lives in ϕ's codomain here (with the growing left bond), so
        # it needs pulling into the domain before svd_trunc splits at the
        # left/(phys,right) boundary — mirrors :twosite's own
        # partition-then-fix-shape pattern, not something new.
        U, S, Vh, ϵ = svd_trunc(permute(ϕ, ((1,), (2, 3))); trunc=strategy)
        ψ.tensors[pos] = permute(Vh, ((1, 2), (3,)))  # -> codomain=(newbond,s), domain=(r,)
        ψ.tensors[pos-1] = ψ[pos-1] * _dmrg3s_residual(dir, U, S, inj1)
        set_ortho_lims!(ψ, pos-2, pos)
    end
    return ϵ
end
 
# ----------------------------------------------------------------------
# Sweep / driver — direct port, minus the tolerance ratchet
# ----------------------------------------------------------------------
# Your existing scalar/vector hooks
_sweep_param(p::AbstractVector, sw::Int) = p[min(sw, length(p))]
_sweep_param(p, ::Int) = p

function _sweep_param(kwargs::Base.Pairs, sw::Int)
    return NamedTuple(k => _sweep_param(v, sw) for (k, v) in pairs(kwargs))
end

# Scalar type of a ProjMPO's cached environments -- needed to scale the
# default eigsolve_tol schedule by machine precision. EnvTensor{T,S,A}'s
# T is exactly the scalar type of the underlying MPS/MPO, so this is the
# real scalar type in play (F32 vs F64) without a separate lookup on ψ/H.
_scalartype(::Type{EnvTensor{T,S,A}}) where {T,S,A} = T
_realtype(::ProjMPO{ET}) where {ET} = real(_scalartype(ET))

"""
    _default_noise(nsweeps::Int; start=1e-2, decades_per_sweep=1, zero_tail=2) -> Vector

Default per-sweep noise schedule: geometric decay from `start` for
`nsweeps - zero_tail` sweeps, then exactly `nothing` (not `0.0`) for the
last `zero_tail` sweeps.

`nothing`, not `0.0`, matters here: `_update_local_tensors!` only skips
the whole perturbation/fuse/⊕/inject path via `!isnothing(noise)` — a
literal `0.0` would still run all of it (α-scaling a perturbation to
exactly zero doesn't stop `_expansion_injectors` from building a bigger
`V_sum` and growing the bond with zero-weight directions, wastefully
relying on `svd_trunc` to discard them after the fact). `nothing` for the
tail actually skips the work, not just zeroes its numerical contribution
— and matches the standard advice to turn noise off entirely for the
final sweeps rather than merely shrink it, since any nonzero mixing
biases the state away from the true single-site variational optimum.
"""
function _default_noise(nsweeps::Int; start::Real=1e-2, decades_per_sweep::Real=1, zero_tail::Int=2)
    ramp = max(nsweeps - zero_tail, 0)
    return [sw <= ramp ? start * 10.0^(-decades_per_sweep*(sw-1)) : nothing for sw in 1:nsweeps]
end

"""
    _default_eigsolve_tol(F::Type{<:Real}, nsweeps::Int; start=1e-6, decades_per_sweep=1) -> Vector{F}

Default per-sweep `eigsolve` tolerance schedule: starts at `start` and
tightens by `decades_per_sweep` decades per sweep, floored at `10*eps(F)`
so it never asks for more precision than the scalar type can represent
(the factor of 10 is a safety margin, not literal machine epsilon). `F32`
and `F64` diverge here since the floor differs by ~8 orders of magnitude
-- that's what makes the schedule type-aware, unlike the single hardcoded
`1e-12` (already tighter than Float32 can resolve at all). This is a
plain geometric default, not a derived quantity -- unlike the dropped
truncation-error ratchet, it doesn't claim to be more than a reasonable
starting point.
"""
function _default_eigsolve_tol(F::Type{<:Real}, nsweeps::Int; start::Real=1e-6, decades_per_sweep::Real=1)
    floor = 10 * eps(F)
    return F[max(F(start) * F(10.0)^(-decades_per_sweep*(sw-1)), floor) for sw in 1:nsweeps]
end
 
"""
    dmrg_sweep!(ψ, P, forward::Bool; maxdim, cutoff, noise, eigsolve_tol, eigsolve_kwargs)
        -> (energies, truncerrs, converged)
 
One sweep (`forward=true` left-to-right, else right-to-left). Direct port
of the prototype's `dmrg_sweep!`, with two changes:
  - `eigsolve` is called directly on the local `TensorMap` (no vec/reshape).
  - `eigsolve_tol` is a plain per-sweep schedule value (see `dmrg!` below),
    not derived from the previous sweep's truncation error.
"""
function dmrg_sweep!(ψ::MPS, P::ProjMPO, forward::Bool;
                      eigsolve_tol=10*eps(_realtype(P)), eigsolve_kwargs=(;),
                      kwargs...)
    energies, truncerrs, converged = Float64[], Float64[], Bool[]
 
    for pos in _rangeP(P, forward)
        orthogonalize!(ψ, pos)
        position!(P, ψ, pos)
        phi0 = _local_tensor(P, ψ, pos)
 
        vals, vecs, info = eigsolve(x -> heff(P, pos, ψ, x), phi0, 1, :SR;
                                     ishermitian=true, tol=eigsolve_tol, verbosity=0,
                                     eigsolve_kwargs...)

        truncerr = _update_local_tensors!(P, ψ, vecs[1], pos, forward; kwargs...)
        push!(energies, real(vals[1]))
        push!(truncerrs, truncerr)
        push!(converged, info.converged >= 1)
    end
    return energies, truncerrs, converged
end
 
"""
    dmrg!(ψ, P, nsweeps; kwargs...) -> (ψ, P, sweep_data)
    dmrg!(ψ, H, nsweeps; nsite=2, kwargs...) -> (ψ, P, sweep_data)
 
## Keyword arguments
 
| Argument          | Description |
|-------------------|-------------|
| `maxdim`          | Max bond dimension per sweep (scalar or `Vector` schedule). |
| `cutoff`          | SVD truncation cutoff (scalar or `Vector`), `:twosite` only. |
| `noise`           | Subspace-expansion mixing (`:dmrg3s`/nsite=1 only; scalar/`Vector`). Default: `_default_noise(nsweeps)`, geometric decay then exactly `nothing` (not `0.0`) for the last 2 sweeps. Pass `noise=nothing` explicitly to disable expansion for every sweep instead. |
| `eigsolve_tol`    | Per-sweep `eigsolve` tolerance (scalar or `Vector`). Default: `_default_eigsolve_tol`, a geometric schedule tightening by a decade per sweep, floored at `10*eps(F)` for the actual scalar type `F` (`F32`/`F64` get genuinely different floors). No longer derived from truncation error — see design discussion for why the previous adaptive ratchet was dropped. |
| `eigsolve_kwargs` | Forwarded to `KrylovKit.eigsolve` (e.g. `krylovdim`, `maxiter`); merged with `_DEFAULT_EIGSOLVE_KWARGS`, per-sweep values win. |
| `start_forward`   | Sweep direction for sweep 1 (default `true`, left-to-right). |
| `nsite`           | (MPO overload only) 1 (`:dmrg3s`) or 2 (`:twosite`). |
"""
function dmrg!(ψ::MPS, P::ProjMPO{ET, 1}, nsweeps::Int;
               eigsolve_tol=nothing, eigsolve_kwargs=(;),
               start_forward::Bool=true,
               maxdim=nothing, cutoff=nothing, noise=_default_noise(nsweeps)) where {ET}
    return _dmrg!(ψ, P, nsweeps;
               eigsolve_tol=eigsolve_tol, eigsolve_kwargs=eigsolve_kwargs,
               start_forward=start_forward,
               maxdim=maxdim, cutoff=cutoff, noise=noise)
end
function dmrg!(ψ::MPS, P::ProjMPO{ET, 2}, nsweeps::Int;
               eigsolve_tol=nothing, eigsolve_kwargs=(;),
               start_forward::Bool=true,
               maxdim=nothing, cutoff=nothing) where {ET}
    return _dmrg!(ψ, P, nsweeps;
               eigsolve_tol=eigsolve_tol, eigsolve_kwargs=eigsolve_kwargs,
               start_forward=start_forward,
               maxdim=maxdim, cutoff=cutoff)
end

function _dmrg!(ψ::MPS, P::ProjMPO, nsweeps::Int;
               eigsolve_tol=nothing, eigsolve_kwargs=(;),
               start_forward::Bool=true,
               kwargs...)
    sweep_data = NamedTuple{(:energies, :truncerrs, :converged),
                             Tuple{Vector{Float64},Vector{Float64},Vector{Bool}}}[]

    tol_schedule = isnothing(eigsolve_tol) ? _default_eigsolve_tol(_realtype(P), nsweeps) : eigsolve_tol

    for sw in 0:(nsweeps-1)
        forward = iseven(sw) ? start_forward : !start_forward
        kw = merge(_DEFAULT_EIGSOLVE_KWARGS, _sweep_param(eigsolve_kwargs, sw+1))
        energies, truncerrs, converged = dmrg_sweep!(ψ, P, forward;
                        eigsolve_tol=_sweep_param(tol_schedule, sw+1),
                        eigsolve_kwargs=kw,
                        _sweep_param(kwargs, sw+1)...)
        push!(sweep_data, (energies=energies, truncerrs=truncerrs, converged=converged))
 
        @info "dmrg sweep $(sw+1)/$nsweeps" forward energy=energies[end] max_truncerr=maximum(truncerrs; init=0.0) n_unconverged=count(!, converged) maxlinkdim=maxlinkdim(ψ)
    end
    return ψ, P, sweep_data
end
 
dmrg!(ψ::MPS, H::MPO, nsweeps::Int; nsite::Int=2, kwargs...) =
    dmrg!(ψ, ProjMPO(H, nsite), nsweeps; kwargs...)