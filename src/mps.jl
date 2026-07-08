# ------------------------------------------------------------------------
# MPS
#
# Finite chain of MPSTensors with single-orthogonality-center bookkeeping
# via llim/rlim (design_notes.md, matching ITensor's convention). No
# "infinite" concept in this library, so plain `MPS`, not `FiniteMPS`.
#
# llim/rlim semantics (working definition — flagged as unverified against
# ITensorMPS.jl source, confirm if/when we have a reference to check
# against):
#   - sites 1:llim       are left-orthogonal
#   - sites rlim:length(ψ) are right-orthogonal
#   - a unique orthogonality center exists only when llim+1 == rlim-1,
#     at site llim+1 (isortho/orthocenter: see tensortrain.jl, defined
#     generically there)
#   - a freshly constructed, non-orthogonalized MPS: llim=0, rlim=L+1
#     (confirmed bounds)
# ------------------------------------------------------------------------

"""
    MPS{T,S,A} <: AbstractTensorTrain{T,S,A}

Finite matrix product state: a chain of [`MPSTensor`](@ref)s plus
`llim`/`rlim` orthogonality-center bookkeeping.
"""
mutable struct MPS{T,S<:ElementarySpace,A<:DenseVector{T}} <: AbstractTensorTrain{T,S,A}
    tensors::Vector{MPSTensor{T,S,A}}
    llim::Int
    rlim::Int
end

"""
    MPS(ts::Vector{<:MPSTensor})

Construct an `MPS` from a vector of site tensors, marked as
non-orthogonalized (`llim=0`, `rlim=length(ts)+1`).
"""
function MPS(ts::Vector{MPSTensor{T,S,A}}) where {T,S,A}
    return MPS{T,S,A}(ts, 0, length(ts) + 1)
end

tensors(ψ::MPS) = ψ.tensors

"""
    llim(ψ::MPS) -> Int

Rightmost site such that sites `1:llim(ψ)` are left-orthogonal.
"""
llim(ψ::MPS) = ψ.llim

"""
    rlim(ψ::MPS) -> Int

Leftmost site such that sites `rlim(ψ):length(ψ)` are right-orthogonal.
"""
rlim(ψ::MPS) = ψ.rlim

"""
    MPS(ts::Vector{<:MPSTensor}, center::Int)

Construct an `MPS` already in mixed-canonical form with orthogonality
center at site `center` (`llim=center-1`, `rlim=center+1`). Caller is
responsible for `ts` actually being in that form — no check is performed.
"""
function MPS(ts::Vector{MPSTensor{T,S,A}}, center::Int) where {T,S,A}
    return MPS{T,S,A}(ts, center - 1, center + 1)
end

"""
    MPS([T=Float64,] sites::Vector{<:SiteType}, states::Vector{<:StateName})

Product-state MPS (all bonds dimension 1), built from dense state vectors
via [`state`](@ref). `Trivial` sites only for now — a nonzero-charge basis
state (e.g. `:Up` under `U1Irrep`) can't be embedded as a bare rank-(1,0)
tensor at all (see chat/qubit.jl: same "needs an auxiliary charge leg"
issue as S+/S-), so a symmetric version of this constructor needs that
solved first, not attempted here.

Sidesteps `statetensor` itself (which only builds a bare rank-(1,0)
tensor, no bond legs) and goes through the dense `state(...)` vector
directly, reshaped into a `(d,1)` matrix — this only relies on the
already-confirmed `TensorMap(denseMatrix, codomain <- domain)`
construction pattern, generalized from a single leg to a codomain that's
`(ℂ^1 ⊗ physspace)`; using `ℂ^1` in that codomain rather than `physspace`
alone doesn't add real leg-order risk since a 1-dimensional factor can't
affect how the flattening reads.
"""
function MPS(::Type{T}, sites::Vector{<:SiteType{<:Any,Trivial}}, states::Vector{<:StateName}) where {T<:Number}
    L = length(sites)
    @assert length(states) == L "Number of states doesn't match number of sites"
    tensors = Vector{MPSTensor{T,ComplexSpace,Vector{T}}}(undef, L)
    for i in 1:L
        physspace = TensorKit.space(sites[i])
        v = T.(state(sites[i], states[i]))
        data = reshape(v, length(v), 1)
        tensors[i] = TensorMap(data, (ℂ^1 ⊗ physspace) ← ℂ^1)
    end
    return MPS(tensors)
end

function MPS(sites::Vector{<:SiteType{<:Any,Trivial}}, states::Vector{<:StateName})
    return MPS(Float64, sites, states)
end

function MPS(::Type{T}, sites::Vector{<:SiteType{<:Any,Trivial}}, states::Vector{<:Union{AbstractString,Symbol}}) where {T<:Number}
    return MPS(T, sites, StateName.(states))
end

function MPS(sites::Vector{<:SiteType{<:Any,Trivial}}, states::Vector{<:Union{AbstractString,Symbol}})
    return MPS(Float64, sites, StateName.(states))
end

"""
Directly set `llim`/`rlim`. Low-level bookkeeping — callers (e.g. an
orthogonalization sweep) are responsible for `l`/`r` actually reflecting
which tensors are left-/right-orthogonal; this does not check or enforce it.
"""
function set_ortho_lims!(ψ::MPS, l::Int, r::Int)
    ψ.llim = l
    ψ.rlim = r
    return ψ
end

# ------------------------------------------------------------------------
# random_mps
#
# Trivial sites only (see MPS(sites,states) above for why symmetric
# bond-space selection is deferred). Built already in mixed-canonical
# form (llim=mid, rlim=mid+2): left half via randn+left_orth, right half
# via randn+repartition+right_orth+repartition (the "QR of Gaussian data"
# trick for a Haar-random isometry, discarding the bond factor). Both
# loops use the same explicit idiom on purpose — see comment further
# below for why randisometry was tried and dropped as an inconsistent
# shortcut.
#
# Deliberately does NOT query the resulting bond space back out of a
# factorization result (e.g. via domain(Q)[1]) -- since maxdim caps every
# bond at <= dim(left)*dim(phys) BEFORE construction, the result's bond
# space is provably identical to the space already picked going in, so we
# just reuse that variable instead.
# ------------------------------------------------------------------------

"""
    random_mps([T=Float64,] sites::Vector{<:SiteType}; maxdim::Int)

Random `MPS` over `sites` (`Trivial` sites only, see mps.jl's
`MPS(sites,states)` docstring for why) with maximum bond dimension
`maxdim`, already in mixed-canonical form with the orthogonality center
at site `L÷2 + 1`.
"""
function random_mps(::Type{T}, sites::Vector{<:SiteType{<:Any,Trivial}}; maxdim::Int=1) where {T<:Number}
    L = length(sites)
    tensors = Vector{MPSTensor{T,ComplexSpace,Vector{T}}}(undef, L)
    mid = L ÷ 2

    # Both loops use the same explicit randn+factorize idiom on purpose
    # (not randisometry for the left loop, randn+repartition+right_orth
    # for the right) — randisometry is presumably just sugar for exactly
    # this (Q,_ = left_orth(randn(...))) anyway, so using it only on one
    # side made the code look more asymmetric than it structurally is.
    # The real, unavoidable asymmetry is leg order: left needs no
    # repartition (MPSTensor's native (2,1) split already matches),
    # right does (validated pattern, see chat: same leg crosses the
    # boundary out and back, dual cancels exactly, confirmed via a real
    # script run — matches _rightstep! in orthogonalize.jl on purpose).

    left = ℂ^1
    for i in 1:mid
        physspace = TensorKit.space(sites[i])
        right = ℂ^(min(maxdim, dim(left) * dim(physspace)))
        t = randn(T, (left ⊗ physspace) ← right)
        Q, _ = left_orth(t)
        tensors[i] = Q
        left = right   # no rank deficiency by construction, so this holds exactly
    end

    right = ℂ^1
    for i in L:-1:(mid+2)
        physspace = TensorKit.space(sites[i])
        left_i = ℂ^(min(maxdim, dim(right) * dim(physspace)))
        t = randn(T, (left_i ⊗ physspace) ← right)
        t2 = repartition(t, 1, 2)
        _, Vd = right_orth(t2)
        tensors[i] = repartition(Vd, 2, 1)
        right = left_i   # same reasoning, mirrored
    end

    physspace_c = TensorKit.space(sites[mid+1])
    t_c = randn(T, (left ⊗ physspace_c) ← right)   # left,right = bond spaces left over from the two loops above
    tensors[mid+1] = t_c / norm(t_c)

    return MPS(tensors, mid + 1)
end

function random_mps(sites::Vector{<:SiteType{<:Any,Trivial}}; maxdim::Int=1)
    return random_mps(Float64, sites; maxdim)
end
# backward compatibility
random_mps(::Type{T}, sites::Vector{<:SiteType{<:Any,Trivial}}, maxdim::Int) = random_mps(T, sites; maxdim)
random_mps(sites::Vector{<:SiteType{<:Any,Trivial}}, maxdim::Int) = random_mps(sites; maxdim)


# ------------------------------------------------------------------------
# norm(ψ), normalize(ψ)/normalize!(ψ)/normalize!!(ψ) — MPS only, and only
# defined when isortho(ψ) (throws otherwise). No O(L) fallback for the
# non-orthogonal case: norm/normalize are ALWAYS the cheap O(1)
# orthocenter-based shortcut (isometric flanks preserve norm by
# construction) -- if you need the norm of a non-orthogonal MPS, call
# orthogonalize! first, or use inner(ψ,ψ) directly.
#
# This falls out almost for free: orthocenter(ψ) itself already throws
# ArgumentError when !isortho(ψ), so no separate check is needed here.
#
# inner(ψ,φ) is NOT restricted to isortho — for two DIFFERENT states
# there's no O(1) shortcut regardless of orthogonality (that trick only
# applies to self-overlap via the orthocenter), so it stays the general
# full O(L) contraction, usable on any pair.
#
# Extended via TensorKit. (not LinearAlgebra.) since TensorKit already
# re-exports norm/normalize/normalize!/rmul! (originally LinearAlgebra's)
# -- no need for a fresh explicit LinearAlgebra dependency, same as how
# TensorKit.space is extended in space.jl rather than defining a new name.
#
# MPO deliberately not covered here: inner(::MPO,::MPO) would need a
# different contraction (both site_out/site_in legs), and "norm of an
# operator" is ambiguous (operator vs. Frobenius norm) — left for a
# separate, later design discussion rather than guessed at here.
#
# inner's contraction logic is NOT new: it's test_mps.jl's _mps_inner
# helper, which already ran successfully in the passing test suite,
# promoted here essentially unchanged.
#
# Convention: inner(ψ,φ) = ⟨ψ|φ⟩, conjugate-linear in the FIRST argument
# — CHECK this matches TensorKit's own inner(::AbstractTensorMap,...)
# convention before trusting it's consistent across the codebase (see
# chat for the specific REPL check; still pending confirmation).
# ------------------------------------------------------------------------

"""
    inner(ψ::MPS, φ::MPS) -> Number

`⟨ψ|φ⟩`, via the standard transfer-matrix/environment contraction.
Conjugate-linear in `ψ`, linear in `φ`. `ψ`/`φ` must have the same length
and matching physical spaces per site; bond dimensions may differ (e.g.
comparing a compressed state to the original). Assumes a `ℂ^1` left
boundary on both, true for everything this package currently constructs.
"""
function TensorKit.inner(ψ::MPS, φ::MPS)
    L = length(ψ)
    length(φ) == L || throw(ArgumentError("MPS must have the same length"))
    E = id(ℂ^1)
    for i in 1:L
        @tensor Enew[b; c] := conj(ψ[i][a, s, b]) * E[a; a'] * φ[i][a', s, c]
        E = Enew
    end
    return tr(E)
end

"""
    norm(ψ::MPS) -> Real

Norm of the orthogonality center. Throws `ArgumentError` if `!isortho(ψ)`
(via `orthocenter(ψ)`) — call `orthogonalize!` first if needed.
"""
TensorKit.norm(ψ::MPS) = TensorKit.norm(ψ[orthocenter(ψ)])

"""
    normalize!(ψ::MPS) -> ψ

Rescale `ψ` to unit norm by rebinding the orthogonality center to a
freshly-scaled tensor — container-only mutation, safe on a shallow
`copy(ψ)`. Throws if `!isortho(ψ)`.
"""
function TensorKit.normalize!(ψ::MPS)
    c = orthocenter(ψ)
    ψ.tensors[c] = TensorKit.normalize(ψ[c])
    return ψ
end

"""
    normalize!!(ψ::MPS) -> ψ

Same as [`normalize!`](@ref), but rescales the orthogonality center's
storage IN PLACE (`rmul!`) rather than rebinding to a freshly allocated
tensor — faster, but only correct if `ψ`'s tensors aren't shared with
anything else (see mutation-convention note in tensortrain.jl — this is
squarely a `!!`-class operation, not a `!` one). Throws if `!isortho(ψ)`.
"""
function normalize!!(ψ::MPS)
    c = orthocenter(ψ)
    rmul!(ψ.tensors[c], inv(TensorKit.norm(ψ[c])))
    return ψ
end

"""
    normalize(ψ::MPS) -> MPS

Non-mutating version of `normalize!`, via `normalize!(copy(ψ))`. Throws
if `!isortho(ψ)`.
"""
TensorKit.normalize(ψ::MPS) = TensorKit.normalize!(copy(ψ))