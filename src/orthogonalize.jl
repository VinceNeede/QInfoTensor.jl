# ------------------------------------------------------------------------
# orthogonalize!/orthogonalize!!
#
# Two plain step functions (_leftstep!/_rightstep!), each self-contained,
# no dispatch-tag indirection — an earlier version tried to unify them
# via Left()/Right() marker-type dispatch, but the two @tensor absorption
# lines still had to differ by hand regardless, so the abstraction bought
# nothing beyond extra pieces to read through. Sharing is kept to what
# actually IS identical between the two: the sweep-bounds loop and the
# choice of destructive-vs-non-destructive factorize function (that's
# _orthogonalize!, the one thing orthogonalize!/orthogonalize!! differ on).
#
# Confirmed facts this relies on (see chat):
#   - left_orth(t)  -> V, C   using t's OWN (2,1) split — no repartition
#     needed for the left step.
#   - right_orth(t) -> C, Vᴴ  (bond factor FIRST) — needs t repartitioned
#     to (1,2) first (bending phys from codomain into domain), since
#     right_orth/left_orth always factor along the tensor's own stored
#     split; there is no leftind/rightind argument to pick one ad hoc.
#   - repartition(t, N1, N2) -> tdst, shares data when possible.
#   - @tensor Out[a,b;c] := ...  — `;` marks the OUTPUT's codomain/domain
#     split explicitly; confirmed via a real session call.
# ------------------------------------------------------------------------

"""
    _leftstep!(ψ::MPS, i::Int; factorize) -> ψ

Left-orthogonalize site `i`, absorbing the bond factor into site `i+1`.
`factorize` is `left_orth` or `left_orth!`.
"""
function _leftstep!(ψ::MPS, i::Int; factorize)
    V, C = factorize(ψ[i])
    ψ.tensors[i] = V
    @tensor ψ.tensors[i+1][a, b; c] := C[a, d] * ψ.tensors[i+1][d, b, c]
    return ψ
end

"""
    _rightstep!(ψ::MPS, i::Int; factorize) -> ψ

Right-orthogonalize site `i`, absorbing the bond factor into site `i-1`.
`factorize` is `right_orth` or `right_orth!`.
"""
function _rightstep!(ψ::MPS, i::Int; factorize)
    t = repartition(ψ[i], 1, 2)
    C, Vd = factorize(t)
    ψ.tensors[i] = repartition(Vd, 2, 1)
    @tensor ψ.tensors[i-1][a, b; c] := ψ.tensors[i-1][a, b, d] * C[d, c]
    return ψ
end

# ── MPO ──────────────────────────────────────────────────────────────────
#
# LEG ORDER CHANGED from design_notes.md's original text: domain is now
# (site_in,right), not (right,site_in) — flattened native order
# (left,site_out,site_in,right), bond legs on the outside, both physical
# legs adjacent in the middle. Pure documentation/convention choice, not
# forced by contraction correctness (@tensor contracts by label
# regardless of position) — changed specifically because it makes BOTH
# orthogonalization steps plain contiguous repartition calls, no permute
# needed at all (the original (right,site_in) order needed a genuine
# permute for the left step, since site_in and right weren't adjacent).
# design_notes.md's MPOTensor convention section needs updating to match.
#
# Still open (unchanged from before): whether this (site_out,site_in)
# bundling, and llim/rlim's meaning for an MPO generally, is actually
# what zip-up needs — flagged in mpo.jl, not resolved here.

function _leftstep!(ψ::MPO, i::Int; factorize)
    t = repartition(ψ[i], 3, 1)                     # (left,site_out,site_in) | (right)
    V, C = factorize(t)
    ψ.tensors[i] = repartition(V, 2, 2)              # (left,site_out) | (site_in,right) — native shape
    @tensor ψ.tensors[i+1][a, b; c, d] := C[a, e] * ψ.tensors[i+1][e, b, c, d]
    return ψ
end

function _rightstep!(ψ::MPO, i::Int; factorize)
    t = repartition(ψ[i], 1, 3)                      # (left) | (site_out,site_in,right)
    C, Vd = factorize(t)
    ψ.tensors[i] = repartition(Vd, 2, 2)             # (left,site_out) | (site_in,right) — native shape
    @tensor ψ.tensors[i-1][a, b; c, d] := ψ.tensors[i-1][a, b, c, e] * C[e, d]
    return ψ
end

"""
    _orthogonalize!(ψ::MPS, j::Int; leftfactorize, rightfactorize) -> ψ

Shared sweep skeleton behind both `orthogonalize!`/`orthogonalize!!` —
only `leftfactorize`/`rightfactorize` (non-destructive vs. destructive
TensorKit factorizations) differ between the two.
"""
function _orthogonalize!(ψ::AbstractTensorTrain, j::Int; leftfactorize, rightfactorize)
    for i in (llim(ψ)+1):(j-1)
        _leftstep!(ψ, i; factorize=leftfactorize)
    end
    for i in (rlim(ψ)-1):-1:(j+1)
        _rightstep!(ψ, i; factorize=rightfactorize)
    end
    set_ortho_lims!(ψ, j - 1, j + 1)
    return ψ
end

"""
    orthogonalize!(ψ::AbstractTensorTrain, j::Int)

Move `ψ`'s orthogonality center to site `j`, sweeping from wherever
`llim(ψ)`/`rlim(ψ)` currently place it. Uses TensorKit's non-destructive
factorizations (see mutation-convention note above) — safe on a shallow
`copy(ψ)`.
"""
function orthogonalize!(ψ::AbstractTensorTrain, j::Int)
    return _orthogonalize!(ψ, j; leftfactorize=left_orth, rightfactorize=right_orth)
end

"""
    orthogonalize!!(ψ::AbstractTensorTrain, j::Int)

Same as [`orthogonalize!`](@ref), but uses TensorKit's destructive
(bang) factorizations internally for speed. Only correct if `ψ`'s tensors
aren't shared with anything else — see mutation-convention note above.
"""
function orthogonalize!!(ψ::AbstractTensorTrain, j::Int)
    return _orthogonalize!(ψ, j; leftfactorize=left_orth!, rightfactorize=right_orth!)
end

"""
    orthogonalize(ψ::AbstractTensorTrain, args...; kwargs...)

Non-mutating version of `orthogonalize!`, via `orthogonalize!(copy(ψ), ...)`.
Deliberately calls `orthogonalize!` (not `orthogonalize!!`) — see the
mutation convention above; `copy()` is only safe to combine with `!`.
"""
orthogonalize(ψ::AbstractTensorTrain, args...; kwargs...) = orthogonalize!(copy(ψ), args...; kwargs...)

# ------------------------------------------------------------------------
# compress!/compress!!
#
# Reuses _orthogonalize!'s exact sweep skeleton and both MPS/MPO
# _leftstep!/_rightstep! methods unchanged — the only difference from
# orthogonalize! is WHICH factorize function gets passed in: SVD-based
# with an explicit truncation strategy, instead of plain QR. No explicit
# alg=:svd needed — left_orth/right_orth auto-select SVD whenever a
# non-trivial `trunc` is present.
#
"""
    compress(ψ::AbstractTensorTrain, args...; kwargs...)

Non-mutating version of `compress!`, via `compress!(copy(ψ), ...)`.
"""
compress(ψ::AbstractTensorTrain, args...; kwargs...) = compress!(copy(ψ), args...; kwargs...)


"""
    compress!(ψ::AbstractTensorTrain; maxdim=nothing, cutoff=nothing, center::Int=length(ψ))

Sweep the whole chain, truncating every bond, leaving the orthogonality
center at site `center` when done (default: the last site — the natural
resting point of a single left-to-right sweep; other `center` values may
require the sweep to double back).

- `maxdim`: hard cap on any bond's dimension (`nothing` = no cap).
- `cutoff`: truncate the smallest singular values at a bond whose
  discarded squared weight, `sum(σ_discarded.^2) / sum(σ_all.^2)`, stays
  below `cutoff` — i.e. `cutoff` bounds relative discarded probability
  weight, NOT raw singular value magnitude (`nothing` = no cutoff).

Uses TensorKit's non-destructive factorizations — safe on a shallow
`copy(ψ)`.
"""
function compress!(ψ::AbstractTensorTrain; maxdim::Union{Int,Nothing}=nothing,
                    cutoff::Union{Real,Nothing}=nothing, center::Int=length(ψ))
    reset_ortho_lims!(ψ)
    trunc = _truncation_strategy(maxdim, cutoff)
    # no alg=:svd needed — passing a non-trivial `trunc` alone makes
    # left_orth/right_orth select an SVD-based decomposition automatically
    return _orthogonalize!(ψ, center;
        leftfactorize=t -> left_orth(t; trunc=trunc),
        rightfactorize=t -> right_orth(t; trunc=trunc))
end

"""
    compress!!(ψ::AbstractTensorTrain; maxdim=nothing, cutoff=nothing, center::Int=length(ψ))

Same as [`compress!`](@ref), but uses TensorKit's destructive (bang)
factorizations internally for speed. Only correct if `ψ`'s tensors aren't
shared with anything else — see mutation-convention note above.
"""
function compress!!(ψ::AbstractTensorTrain; maxdim::Union{Int,Nothing}=nothing,
                     cutoff::Union{Real,Nothing}=nothing, center::Int=length(ψ))
    reset_ortho_lims!(ψ)
    trunc = _truncation_strategy(maxdim, cutoff)
    return _orthogonalize!(ψ, center;
        leftfactorize=t -> left_orth!(t; trunc=trunc),
        rightfactorize=t -> right_orth!(t; trunc=trunc))
end