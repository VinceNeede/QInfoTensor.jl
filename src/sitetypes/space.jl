"""
    space(st::SiteType) -> TensorKit.ElementarySpace

Return the physical `ElementarySpace` for a given [`SiteType`](@ref).
Extends `TensorKit.space`. No `kwargs` — must be a pure function of `st`
alone (see [`SiteType`](@ref) docstring for why: bond tensors across a
chain require every site to report a space of consistent structure).

No general fallback is defined; each concrete `(tag,Sym)` combination adds
its own method, e.g.

    TensorKit.space(::SiteType{:SpinHalf,Trivial}) = ℂ^2
    TensorKit.space(::SiteType{:SpinHalf,U1Irrep}) = U1Space(1//2=>1, -1//2=>1)
"""
function TensorKit.space(st::SiteType)
    throw(MethodError(TensorKit.space, (st,)))
end

"""
    op(st::SiteType, on::OpName; kwargs...) :: Matrix

Return the local operator matrix for site type `st` and operator `on`, as
a plain dense `Matrix` — ported unchanged from the prototype library.
Basis ordering must match the charge grading `space(st)` declares once a
non-`Trivial` `Sym` is used, or [`optensor`](@ref) will throw.

    op(SiteType(:SpinHalf), OpName(:Rz); θ=π/2)
    op(SiteType(:SpinHalf), "Rz"; θ=π/2)   # string entry point
"""
function op end

op(st::SiteType, name::AbstractString; kwargs...) = op(st, OpName(name); kwargs...)
op(tag::Union{Symbol,AbstractString}, name::AbstractString; kwargs...) =
    op(SiteType(tag), OpName(name); kwargs...)

"""
    state(st::SiteType, sn::StateName; kwargs...) :: Vector

Return the basis-state vector for state name `sn` on site type `st`, as a
plain dense `Vector` — ported unchanged from the prototype library.

    state(SiteType(:SpinHalf), StateName(:Up))
    state(SiteType(:SpinHalf), "Up")   # string entry point
"""
function state end

state(st::SiteType, name::AbstractString; kwargs...) = state(st, StateName(name); kwargs...)
state(tag::Union{Symbol,AbstractString}, name::AbstractString; kwargs...) =
    state(SiteType(tag), StateName(name); kwargs...)

# ── dense -> TensorMap conversion ────────────────────────────────────────────

"""
    optensor(st::SiteType, on::OpName; kwargs...) -> TensorMap

Embed the dense operator `op(st, on; kwargs...)` into a rank-(1,1)
`TensorMap` over `space(st) ← space(st)`. The resulting scalar type `T` is
whatever `op` itself returns (e.g. `Float64` for `Sz`, `ComplexF64` for
`Sy`) — real-vs-complex is entirely the dense operator's choice, unrelated
to `space(st)`'s symmetry.

Throws if `on`'s matrix isn't a legal operator under `st`'s symmetry
(i.e. doesn't respect the charge blocks `space(st)` declares) — this is
TensorKit's own `TensorMap` constructor validating charge conservation,
not bespoke logic here. This resolves design_notes.md's open question 2:
illegal operators error at construction time, automatically.
"""
function optensor(st::SiteType, on::OpName; kwargs...)
    M = op(st, on; kwargs...)
    V = TensorKit.space(st)
    return TensorMap(M, V ← V)
end

"""
    statetensor([T,] st::SiteType, sn::StateName; kwargs...) -> AbstractTensor

Embed the dense state vector `state(st, sn; kwargs...)` into a rank-(1,0)
tensor over `space(st)`. `T` defaults to the dense vector's own eltype if
not given explicitly.
"""
function statetensor(::Type{T}, st::SiteType, sn::StateName; kwargs...) where {T<:Number}
    v = state(st, sn; kwargs...)
    V = TensorKit.space(st)
    return Tensor(T.(v), V)
end
function statetensor(st::SiteType, sn::StateName; kwargs...)
    v = state(st, sn; kwargs...)
    return statetensor(eltype(v), st, sn; kwargs...)
end