# ------------------------------------------------------------------------
# MPO
#
# Finite chain of MPOTensors. Also carries llim/rlim bookkeeping — unlike
# a generic MPO, this one is used in zip-up (single-pass MPO-MPS
# contraction), which needs some notion of "how far the sweep has
# progressed."
#
# OPEN QUESTION, not resolved here: whether "left-orthogonal"/
# "right-orthogonal" for an MPO tensor means the same thing zip-up needs.
# isortho/orthocenter are inherited generically from AbstractTensorTrain
# (tensortrain.jl) — if that turns out wrong for MPO, override
# isortho(ψ::MPO)/orthocenter(ψ::MPO) here specifically once zip-up is
# actually designed; nothing else needs to change.
# ------------------------------------------------------------------------

"""
    MPO{T,S,A} <: AbstractTensorTrain{T,S,A}

Finite matrix product operator: a chain of [`MPOTensor`](@ref)s plus
`llim`/`rlim` bookkeeping (used by zip-up — see module-level note above
about the open question on what this bookkeeping means for an MPO).
"""
mutable struct MPO{T,S<:ElementarySpace,A<:DenseVector{T}} <: AbstractTensorTrain{T,S,A}
    tensors::Vector{MPOTensor{T,S,A}}
    llim::Int
    rlim::Int
end

"""
    MPO(ts::Vector{<:MPOTensor})

Construct an `MPO` from a vector of site tensors, marked as
non-orthogonalized (`llim=0`, `rlim=length(ts)+1`).
"""
function MPO(ts::Vector{MPOTensor{T,S,A}}) where {T,S,A}
    return MPO{T,S,A}(ts, 0, length(ts) + 1)
end

tensors(ψ::MPO) = ψ.tensors

"""
    llim(ψ::MPO) -> Int
"""
llim(ψ::MPO) = ψ.llim

"""
    rlim(ψ::MPO) -> Int
"""
rlim(ψ::MPO) = ψ.rlim

"""
    set_ortho_lims!(ψ::MPO, l::Int, r::Int) -> ψ
"""
function set_ortho_lims!(ψ::MPO, l::Int, r::Int)
    ψ.llim = l
    ψ.rlim = r
    return ψ
end

"""
    inner(ψ::MPS, H::MPO, φ::MPS) -> Number

Compute the matrix element `⟨ψ|H|φ⟩` by contracting the bra, MPO, and ket
left-to-right in a single sweep.  `ψ` is automatically conjugated.
"""
function TensorKit.inner(ψ::MPS, H::MPO, φ::MPS)
    L = length(H)
    length(ψ) == L && length(φ) == L || throw(ArgumentError("MPS/MPO length mismatch"))
 
    T = promote_type(eltype(ψ), eltype(H), eltype(φ))


    V_H_left = space(H[1], 1)
    V_ψ_left = space(ψ[1], 1)
    V_ϕ_left = space(φ[1], 1)
    E = TensorMap(ones(T, 1, 1), V_ψ_left ← (V_H_left ⊗ V_ϕ_left))

    for i in 1:L
        ψi, Hi, φi = ψ[i], H[i], φ[i]
        @tensoropt (lp, rp, l, r) begin
            Enew[rp; rh r] := E[lp; lh l] * conj(ψi[lp, sp, rp]) * Hi[lh, sp, s, rh] * φi[l, s, r]
        end
        E = Enew
    end
    return block(E, first(blocksectors(E)))[1]
end
 
function _expect(ψi::MPSTensor, O::AbstractTensorMap)
    @tensor res[:] := conj(ψi[l, sp, r]) * O[sp, s] * ψi[l, s, r]
    return real(scalar(res))
end

"""
    expect!(ψ::MPS, O::AbstractTensorMap, pos::Int)

In-place expectation value. Modifies the gauge/orthogonalization center of `ψ`.
"""
function expect!(ψ::MPS, O::AbstractTensorMap, pos::Int)
    ψ = orthogonalize!(ψ, pos)
    return _expect(ψ[pos], O)
end

"""
    expect(ψ::MPS, O::AbstractTensorMap, pos::Int)

Out-of-place expectation value. Safe to use; does not modify the input `ψ`.
"""
function expect(ψ::MPS, O::AbstractTensorMap, pos::Int)
    ψc = orthogonalize(ψ, pos) # Assuming this creates a safe view/copy
    return _expect(ψc[pos], O)
end


_get_operator(site::SiteType, on::OpName; kwargs...) = optensor(site, on; kwargs...)

# Convenience: Convert AbstractString to OpName
_get_operator(site::SiteType, name::AbstractString; kwargs...) = _get_operator(site, OpName(name); kwargs...)

# Convenience: Extract the correct site from a Vector of sites
_get_operator(sites::Vector{<:SiteType}, op, pos::Int; kwargs...) = _get_operator(sites[pos], op; kwargs...)

# --- Single Position Dispatch ---
function expect(ψ::MPS, sites::Vector{<:SiteType}, op, pos::Int; kwargs...)
    O = _get_operator(sites, op, pos; kwargs...)
    return expect(ψ, O, pos)
end

# --- Multiple Positions Dispatch (The Single Source of Truth) ---
# This single method handles ALL operator types (String, OpName) 
# because it delegates operator construction to `_get_operator`.
function expect(ψ::MPS, sites::Vector{<:SiteType}, op, positions::AbstractVector{Int}=1:length(ψ); kwargs...)
    ψc = copy(ψ) # Clone once for the lifetime of the loop
    
    # We sort positions to minimize orthogonalization movement overhead, 
    # and use expect! internally because we are mutating our local clone ψc.
    return [expect!(ψc, _get_operator(sites, op, pos; kwargs...), pos) for pos in sort(positions)]
end


# ==============================================================================
# 4. UNIFORM LATTICE CATCH-ALLS (Single SiteType Shortcuts)
# ==============================================================================

# Convert single SiteType to a Vector and re-dispatch for multiple positions
function expect(ψ::MPS, siteT::SiteType, op, positions::AbstractVector{Int}; kwargs...)
    sites = fill(siteT, length(ψ))
    return expect(ψ, sites, op, positions; kwargs...)
end

# Fallback: If given a single SiteType and no positions, default to all positions
function expect(ψ::MPS, siteT::SiteType, op; kwargs...)
    return expect(ψ, siteT, op, 1:length(ψ); kwargs...)
end

"""
    apply!(H::MPO, ψ::MPS, ::Val{:zipup};
           maxdim=nothing, cutoff=nothing,
           sweep_maxdim=2*maxdim, sweep_cutoff=cutoff/10) -> MPS

Compute `H|ψ⟩` in-place using the zip-up algorithm, overwriting `ψ`.

The left-to-right pass contracts `H[i] * ψ[i]` site by site, fusing the
left and right link indices via `Combiner` and factorizing via QR (or SVD
if `sweep_maxdim`/`sweep_cutoff` are provided). The right-to-left pass
compresses the result with `compress!` using `maxdim`/`cutoff`.

For best numerical accuracy, `H` should be in left-canonical form
(`orthogonalize!(H, 1)` called beforehand). A warning is issued if `H`
is not in canonical form.

Returns `ψ` (now representing `H|ψ⟩`) in left-canonical form
(orthogonality center at site 1).

# Arguments
- `maxdim`: maximum bond dimension for the right-to-left compression pass.
- `cutoff`: singular value cutoff for the right-to-left compression pass.
- `sweep_maxdim`: maximum bond dimension during the left-to-right pass
  (default: `2*maxdim`, following Paeckel et al. 2019 — a loose intermediate
  truncation, refined by the final compression pass. `nothing` if `maxdim`
  is `nothing` — no truncation anywhere, exact contract-then-compress).
- `sweep_cutoff`: singular value cutoff during the left-to-right pass
  (default: `cutoff/10`, same rationale; `nothing` if `cutoff` is `nothing`).

See also: Paeckel et al., *Time-evolution methods for matrix-product states*,
Ann. Phys. 411, 167998 (2019), [arXiv:1901.05824](https://arxiv.org/abs/1901.05824)
— section on the zip-up algorithm, for the rationale behind the default
`sweep_maxdim`/`sweep_cutoff` values.

# Example
```julia
sites = siteinds(:SpinHalf, 10)
H = MPO(opsum, sites)
orthogonalize!(H, 1)
ψ = random_mps(Float64, sites, 8)
apply!(H, ψ; maxdim=16, cutoff=1e-10)
```
"""
function apply!(H::MPO, ψ::MPS, ::Val{:zipup};
                maxdim=nothing, cutoff=nothing,
                sweep_maxdim=isnothing(maxdim) ? nothing : 2 * maxdim,
                sweep_cutoff=isnothing(cutoff) ? nothing : cutoff / 10)
    L = length(H)
    length(ψ) == L || throw(ArgumentError("MPO/MPS length mismatch"))
 
    if !isortho(H) || orthocenter(H) != 1
        @warn "apply!(H, ψ, Val(:zipup)): H is not left-canonicalized at site 1. " *
              "Call orthogonalize!(H, 1) before apply! for best numerical accuracy."
    end
    orthogonalize!(ψ, 1)
 
    T = promote_type(eltype(H), eltype(ψ))
 
    V_mpo_left = space(H[1], 1)
    V_mps_left = space(ψ[1], 1)
    V_left_fused = fuse(V_mpo_left ⊗ V_mps_left)
    R_left = isomorphism(T, V_left_fused ← (V_mpo_left ⊗ V_mps_left))
 
    strategy = _truncation_strategy(sweep_maxdim, sweep_cutoff)
 
    for i in 1:L
        Hi, ψi = H[i], ψ[i]
        @tensoropt (new_left, l, r) begin
            θ[new_left, sp; rh, r] := R_left[new_left; lh l] * Hi[lh, sp, s, rh] * ψi[l, s, r]
        end
 
        if i == L
            V_rh = domain(θ)[1] # already encodes duality
            V_r  = domain(θ)[2]
            V_right_fused = fuse(V_rh ⊗ V_r)
            R_right = isomorphism(T, (V_rh ⊗ V_r) ← V_right_fused)

            @tensor θ_fused[new_left, sp; new_right] :=
                θ[new_left, sp, rh, r] * R_right[rh, r, new_right]
            ψ.tensors[L] = θ_fused
        else
            Q, R_left = left_orth(θ; trunc=strategy)
            ψ.tensors[i] = Q
        end
    end
  
    compress!(ψ; maxdim, cutoff, center=1)
 
    return ψ
end

"""
    apply!(H::MPO, ψ::MPS; alg=:zipup, kwargs...) -> MPS

Dispatch to the algorithm selected by `alg`. Currently supported: `:zipup`.
See [`apply!(H, ψ, Val(:zipup))`](@ref) for keyword arguments.
"""
apply!(H::MPO, ψ::MPS; alg=:zipup, kwargs...) = apply!(H, ψ, Val(alg); kwargs...)

"""
    apply(H::MPO, ψ::MPS; kwargs...) -> MPS

Non-mutating version of [`apply!`](@ref): returns a new MPS representing
`H|ψ⟩` without modifying `ψ`. See `apply!` for keyword arguments.
"""
apply(H::MPO,  ψ::MPS; kwargs...) = apply!(H, copy(ψ); kwargs...)