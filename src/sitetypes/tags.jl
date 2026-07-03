# ── SiteType ─────────────────────────────────────────────────────────────────

"""
    SiteType{tag,Sym}

Zero-cost tag struct identifying a local Hilbert space for dispatch,
parametrized by:

- `tag` — a `Symbol` (e.g. `:SpinHalf`) or a `Tuple` (e.g. `(:Boson, 4)` for
  a site type with a structural parameter such as a truncated boson
  dimension), exactly as in the prototype library.
- `Sym<:TensorKit.Sector` — the symmetry sector used to build this site's
  physical space via [`space`](@ref). Defaults to `Trivial` (no symmetry),
  so existing un-symmetrized code is unaffected; opt into symmetry with the
  `sym` keyword, e.g. `SiteType(:SpinHalf; sym=U1Irrep)`.

Construct with `SiteType(:SpinHalf)`, `SiteType("SpinHalf")`,
`SiteType(:Boson, 4)`, or `SiteType(:SpinHalf; sym=U1Irrep)`.
"""
struct SiteType{tag,Sym<:Sector} end

_sitetag(tag::Symbol)            = tag
_sitetag(tag::Symbol, params...) = (tag, params...)

SiteType(tag::Symbol, params...; sym::Type{<:Sector}=Trivial) =
    SiteType{_sitetag(tag, params...),sym}()
SiteType(tag::AbstractString, params...; sym::Type{<:Sector}=Trivial) =
    SiteType(Symbol(tag), params...; sym=sym)
SiteType(t::Tuple; sym::Type{<:Sector}=Trivial) = SiteType{t,sym}()

Base.show(io::IO, ::SiteType{tag,Sym}) where {tag<:Symbol,Sym} =
    print(io, "SiteType(:$tag", Sym === Trivial ? "" : "; sym=$Sym", ")")
Base.show(io::IO, ::SiteType{tag,Sym}) where {tag<:Tuple,Sym} =
    print(io, "SiteType$tag", Sym === Trivial ? "" : "; sym=$Sym")

# ── OpName ───────────────────────────────────────────────────────────────────

"""
    OpName{N}

Zero-cost tag struct identifying a local operator for dispatch.
Construct with `OpName(:Sz)` or `OpName("Sz")`. Unchanged from the
prototype library — symmetry is a `SiteType`-only concern, `op` methods
still just return plain dense matrices (see `space.jl`).

Define new operators by adding methods:

    op(::SiteType{:MyType}, ::OpName{:MyOp}) = ...
    op(::SiteType{:MyType}, ::OpName{:MyOp}; param::Real) = ...
"""
struct OpName{N} end

OpName(s::Symbol)         = OpName{s}()
OpName(s::AbstractString) = OpName(Symbol(s))

Base.show(io::IO, ::OpName{N}) where {N} = print(io, "OpName(:$N)")

# ── StateName ─────────────────────────────────────────────────────────────────

"""
    StateName{N}

Zero-cost tag struct identifying a basis state for dispatch.
Construct with `StateName(:Up)` or `StateName("Up")`. Unchanged from the
prototype library.

Define new states by adding methods:

    state(::SiteType{:MyType}, ::StateName{:MyState}) = [1.0, 0.0, 0.0]
    state(::SiteType{:MyType}, ::StateName{:Coherent}; θ::Real) = [cos(θ/2), sin(θ/2)]
"""
struct StateName{N} end

StateName(s::Symbol)         = StateName{s}()
StateName(s::AbstractString) = StateName(Symbol(s))

Base.show(io::IO, ::StateName{N}) where {N} = print(io, "StateName(:$N)")

# ── @alias_sitetype ──────────────────────────────────────────────────────────

"""
    @alias_sitetype Alias => Canonical

Make `SiteType(Alias)` forward `space`, `op`, and `state` to
`SiteType(Canonical)`, for any symmetry `Sym` the caller picks — e.g.
`@alias_sitetype Qubit => SpinHalf` makes both `SiteType(:Qubit)` and
`SiteType(:Qubit; sym=U1Irrep)` forward to the corresponding `:SpinHalf`
methods. Ported from the prototype library; generalized over `Sym` (the
original only handled the un-symmetrized case, since `SiteType` had no
`Sym` parameter yet).
"""
macro alias_sitetype(expr)
    @assert expr.head == :call && expr.args[1] == :(=>) "Usage: @alias_sitetype Alias => Canonical"

    function to_sym(arg)
        if arg isa String
            QuoteNode(Symbol(arg))   # "S=1/2" → literal :S=1/2, not a variable lookup
        elseif arg isa Expr && arg.head == :tuple
            Expr(:tuple, map(a -> a isa QuoteNode ? a : QuoteNode(a), arg.args)...)
        else
            QuoteNode(arg)
        end
    end
    alias     = to_sym(expr.args[2])
    canonical = to_sym(expr.args[3])

    return esc(quote
        TensorKit.space(::SiteType{$(alias),Sym}) where {Sym} =
            TensorKit.space(SiteType{$(canonical),Sym}())
        op(::SiteType{$(alias),Sym}, on::OpName; kwargs...) where {Sym} =
            op(SiteType{$(canonical),Sym}(), on; kwargs...)
        state(::SiteType{$(alias),Sym}, sn::StateName; kwargs...) where {Sym} =
            state(SiteType{$(canonical),Sym}(), sn; kwargs...)
    end)
end