# ------------------------------------------------------------------------
# OpTerm / OpSum / add! / +
#
# Ported from the prototype library's opsum.jl. Pure bookkeeping (sites,
# op names, coefficients) — no tensor-backend dependency at all, so this
# part needs no TensorKit-specific redesign. QInfoTensor's own OpName is
# used at every call boundary (add!'s signature, the FSM/dispatch code),
# but OpTerm's actual STORAGE is loose (Pair{Int,Any}) rather than
# precisely Pair{Int,Tuple{OpName,NamedTuple}} — see OpTerm's docstring
# for why (two rounds of Vector/Pair invariance bugs, worked through in
# chat, resolved by not fighting Julia's type system over a field whose
# precise type never actually needs to be statically known).
# ------------------------------------------------------------------------

"""
    OpTerm{C<:Number}

A single term in an [`OpSum`](@ref): a scalar coefficient `coeff` and an
ordered list of `site => (opname, params)` pairs (sorted by site).

`ops` is stored loosely (`Pair{Int,Any}`, each value actually an
`(OpName,NamedTuple)` tuple at runtime) deliberately — both `OpName` and
`NamedTuple` are UnionAll-backed, and `Vector` is invariant in its
element type, so a precisely-parametrized field type here just means
fighting Julia's type system at every construction site for no real
benefit (the precise types only matter once, at `op(...)` dispatch time
in `_opsum_cache_and_eltype`/`_fsm_site_tensor` — not for storage). See
chat for the two rounds of invariance bugs this replaced.

Not normally constructed directly — use [`add!`](@ref) or `opsum + (...)`.
"""
struct OpTerm{C<:Number}
    coeff::C
    ops::Vector{Pair{Int,Any}}
end

"""
    OpSum

A sum of operator terms used to build an [`MPO`](@ref) (`Trivial` sites
only for now — see the `MPO(::OpSum,...)` docstring).

```julia
H = OpSum()
H += (J, :Sz, 1, :Sz, 2)
H += (h, :Sx, 1)
mpo = MPO(H, sites)
```

Each term is a `Tuple` whose first element is the numeric coefficient,
followed by alternating `opname, site` pairs. Parametric operators use a
`(name, NamedTuple)` pair in place of the bare name.

See also: [`add!`](@ref), [`MPO`](@ref).
"""
struct OpSum
    terms::Vector{OpTerm}
end

OpSum() = OpSum(OpTerm[])

"""
    add!(opsum::OpSum, coeff, ops::Pair{Int,Tuple{OpName,NamedTuple}}...) -> OpSum
    add!(opsum::OpSum, term::Tuple) -> OpSum

Append a new term to `opsum` (mutating).

The low-level form takes a numeric `coeff` and explicit `site =>
(OpName, params)` pairs. The high-level tuple form parses `(coeff, name,
site, name, site, …)`, accepting bare `Symbol`/`String`/`OpName` (no
params) or `(name, NamedTuple)` pairs for parametric operators.

Note: this extends TensorKit's own `add!` (a `VectorInterface.jl`
primitive for in-place tensor addition, re-exported by `TensorKit`) via
multiple dispatch — different argument types, no actual collision, same
pattern as extending `TensorKit.norm`/`TensorKit.inner` elsewhere in this
package.
"""
function TensorKit.add!(opsum::OpSum, coeff::Number, ops::Pair{Int,<:Tuple{OpName,NamedTuple}}...)
    sorted = Vector{Pair{Int,Any}}(sort(collect(ops); by=first))
    push!(opsum.terms, OpTerm(coeff, sorted))
    return opsum
end

function TensorKit.add!(opsum::OpSum, term::Tuple{Vararg})
    isempty(term) && return opsum

    coeff = first(term)
    coeff isa Number || throw(ArgumentError("The first element of an OpSum tuple must be a numeric coefficient."))

    ops = Pair{Int,Tuple{OpName,NamedTuple}}[]
    i = 2
    while i <= length(term)
        opdata = term[i]
        local opn, params
        if opdata isa Union{Symbol,AbstractString}
            opn, params = OpName(opdata), (;)
        elseif opdata isa OpName
            opn, params = opdata, (;)
        elseif opdata isa Tuple && length(opdata) == 2 &&
               opdata[1] isa Union{Symbol,AbstractString,OpName} && opdata[2] isa NamedTuple
            opn = opdata[1] isa OpName ? opdata[1] : OpName(opdata[1])
            params = opdata[2]
        else
            error("Expected an operator name (Symbol/String/OpName) or a (name, NamedTuple) pair at position $i, got: $opdata")
        end

        i += 1
        i > length(term) && error("Malformed term: missing a physical site index for operator '$opn'.")

        site = term[i]
        site isa Int || throw(ArgumentError("Expected an integer site index at position $i, got: $site"))

        push!(ops, site => (opn, params))
        i += 1
    end

    return TensorKit.add!(opsum, coeff, ops...)
end

"""
    opsum + term -> OpSum

Syntactic sugar for `add!(opsum, term)`. Mutates `opsum` in place and
returns it (despite the `+` spelling) so the idiom `H += (coeff, ...)` works.
"""
Base.:+(opsum::OpSum, term::Tuple{Vararg}) = TensorKit.add!(opsum, term)

# ------------------------------------------------------------------------
# MPO(opsum::OpSum, sites) — FSM construction.
#
# Trivial sites only for now: each FSM auxiliary state would need its own
# charge sector under symmetry (determined by the net charge of the
# partial operator string it represents), which is a real, undecided
# design problem — same treatment as S+/S- and the product-state MPS
# constructor. Not attempted here.
#
# _fsm_states is pure combinatorics on site indices (no tensor backend
# involved) — ported from the prototype essentially unchanged.
#
# _fsm_site_tensor is the genuinely new part: builds a dense
# (|S_prev|,d,d,|S_curr|) array exactly like the prototype (this part
# doesn't care about the tensor backend at all), then reshapes+wraps it
# into an MPOTensor. The reshape convention — does Julia's natural
# column-major reshape of that dense array match TensorKit's own
# multi-leg matrix-flattening convention — was explicitly verified via a
# real script run (scratch_opsum_reshape.jl): an asymmetric operator
# placed at a known FSM block was independently extracted back out via
# @tensor and matched exactly, confirming this is not just assumed.
#
# Leg order (S_prev,site_out,site_in,S_curr) maps directly onto
# MPOTensor's (left,site_out,site_in,right) convention with NO
# reordering needed — this is exactly why that convention was chosen
# earlier (to make orthogonalize!'s repartition calls simple); turns out
# to also be the natural order for FSM construction, unplanned but
# convenient confirmation the earlier choice was right.
# ------------------------------------------------------------------------

const FSMLabel = Union{Symbol,Tuple{Int,Int}}

function _fsm_states(opsum::OpSum, N::Int)
    states = [FSMLabel[] for _ in 0:N]   # states[n+1] = label set at bond n
    states[1] = FSMLabel[:I]
    states[end] = FSMLabel[:F]
    for n in 1:(N-1)
        push!(states[n+1], :I, :F)
    end

    for (α, term) in enumerate(opsum.terms)
        ops = term.ops
        for j in 1:(length(ops)-1)
            site_j, site_j1 = ops[j][1], ops[j+1][1]
            for n in site_j:(site_j1-1)
                push!(states[n+1], (α, j))
            end
        end
    end
    return states
end

function _fsm_site_tensor(opsum::OpSum, n::Int, st::SiteType{<:Any,Trivial},
                           S_prev::Vector{FSMLabel}, S_curr::Vector{FSMLabel},
                           op_cache, ::Type{C}) where {C<:Number}
    V = TensorKit.space(st)
    d = dim(V)
    Tarr = zeros(C, length(S_prev), d, d, length(S_curr))
    ip = Dict(l => i for (i, l) in enumerate(S_prev))
    ic = Dict(l => i for (i, l) in enumerate(S_curr))
    Id = [i == j ? one(C) : zero(C) for i in 1:d, j in 1:d]

    for (label, i) in ip
        j = get(ic, label, nothing)
        isnothing(j) || (Tarr[i, :, :, j] .+= Id)
    end

    for (α, term) in enumerate(opsum.terms)
        ops = term.ops
        for (j, (site_j, opdata)) in enumerate(ops)
            site_j == n || continue
            r = j == 1 ? :I : (α, j - 1)
            c = j == length(ops) ? :F : (α, j)
            opn, params = opdata
            opmat = op_cache[(site_j, opn, params)]
            coeff = j == 1 ? term.coeff : one(C)
            Tarr[ip[r], :, :, ic[c]] .+= coeff .* opmat
        end
    end

    Lspace = ℂ^length(S_prev)
    Rspace = ℂ^length(S_curr)
    M = reshape(Tarr, length(S_prev) * d, d * length(S_curr))
    return TensorMap(M, (Lspace ⊗ V) ← (V ⊗ Rspace))
end

function _opsum_cache_and_eltype(opsum::OpSum, sites::Vector{<:SiteType{<:Any,Trivial}})
    cache = Dict{Tuple{Int,OpName,NamedTuple},Matrix}()
    # Bool, not Float64: Bool is the true identity element for numeric
    # promotion (promote_type(Bool,T)==T for any numeric T), so it never
    # forces unwanted precision the way starting at Float64 would (e.g.
    # an all-Float32 OpSum would otherwise get silently promoted to
    # Float64 just because of the seed, not because anything in it needs
    # double precision). Falls back to Float64 below if opsum turns out
    # to have no terms at all (C would otherwise stay Bool, nonsensical).
    C = Bool
    for term in opsum.terms
        C = promote_type(C, typeof(term.coeff))
        for (site_j, opdata) in term.ops
            opn, params = opdata
            key = (site_j, opn, params)
            mat = get!(cache, key) do
                op(sites[site_j], opn; params...)
            end
            C = promote_type(C, eltype(mat))
        end
    end
    C = C === Bool ? Float64 : C  # empty OpSum: no terms ever seen, fall back to a sensible default
    return cache, C
end

"""
    MPO(opsum::OpSum, sites::Vector{<:SiteType}) -> MPO

Build the finite-state-machine MPO representation of `opsum` over
`sites`. `Trivial` sites only — see module-level note above.
"""
function MPO(opsum::OpSum, sites::Vector{<:SiteType{<:Any,Trivial}})
    N = length(sites)
    states = _fsm_states(opsum, N)
    cache, C = _opsum_cache_and_eltype(opsum, sites)

    tensors = Vector{MPOTensor{C,ComplexSpace,Vector{C}}}(undef, N)
    for n in 1:N
        tensors[n] = _fsm_site_tensor(opsum, n, sites[n], states[n], states[n+1], cache, C)
    end
    return MPO(tensors)
end