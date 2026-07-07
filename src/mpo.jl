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

function _src_prepass(H::MPO, ψ::MPS, χ̄::Int, T::Type)
    L = length(H)
    C = Vector{TensorMap}(undef, L-1)

    V_chi = ℂ^χ̄

    # --- Site 1 Boundary ---
    H_1 = removeunit(H[1], Val(1))
    ψ_1 = removeunit(ψ[1], Val(1))
    physd = codomain(H_1)[1]    
    Ω_1 = TensorMap(randn(T, χ̄, dim(physd)), V_chi ← physd)
    
    @tensoropt (a, c) C[1][a; b c] := Ω_1[a, d] * H_1[d, e, b] * ψ_1[e, c]

    for i in 2:L-1
        H_i, ψ_i = H[i], ψ[i]
        physd = codomain(H_i)[2] # Top physical leg
        Ω_i = TensorMap(randn(T, χ̄, dim(physd)), V_chi ← physd)
        
        # --- batched outer product, replacing the δ copy-tensor ---
        Carr = convert(Array, C[i-1])      # (a, d, e)
        Ωarr = convert(Array, Ω_i)          # (a, f)
        Z = Carr .* reshape(Ωarr, χ̄, 1, 1, size(Ωarr, 2))  # (a, d, e, f), O(χ̄·d·e·f)

        Vd, Ve = domain(C[i-1])
        Zmap = TensorMap(Z, V_chi ← Vd ⊗ Ve ⊗ physd)

        @tensoropt (a, e, c) C[i][a; b c] := Zmap[a; d e f] * H_i[d, f, g, b] * ψ_i[e, g, c]
    end
    
    return C
end

"""
    apply!(H::MPO, ψ::MPS, ::Val{:src}; maxdim::Int = maxlinkdim(H) * maxlinkdim(ψ)) -> MPS

Compute the compressed MPO-MPS product ``H|ψ⟩`` using the Successive Randomized Compression (SRC)
algorithm.

The algorithm computes a compressed MPS representation ``|η⟩ \\approx H|ψ⟩`` targeting a maximum 
bond dimension `maxdim`. It operates via a single-pass framework consisting of a left-to-right 
randomized sketching prepass followed by a right-to-left compression pass using randomized 
matrix factorizations. 

The input state `ψ` is modified **in-place** to store the compressed result and is returned 
in right-canonical form.

# Arguments
- `H::MPO`: The Matrix Product Operator to apply.
- `ψ::MPS`: The Matrix Product State to be multiplied. Overwritten in-place.
- `maxdim::Int`: The maximum target bond dimension (sketch dimension ``\\bar{\\chi}``). Defaults to 
  the exact, untruncated product of the MPO and MPS bond dimensions, which guarantees exact recovery 
  with probability one (Theorem 3).

# Errors
- Throws an `ArgumentError` if `ψ` or `H` contains non-trivial symmetry sectors, as the randomized 
  sketching step does not natively preserve block-sparse quantum numbers.

# References
- Camaño, C., Epperly, E. N., & Tropp, J. A. (2026). *Successive randomized compression: A 
  randomized algorithm for the compressed MPO-MPS product*. arXiv:2504.06475.
"""
function apply!(H::MPO, ψ::MPS, ::Val{:src}; maxdim::Int=maxlinkdim(H)*maxlinkdim(ψ))
    if sectortype(ψ[1]) != Trivial || sectortype(H[1]) != Trivial
        throw(ArgumentError("The Successive Randomized Compression (SRC) algorithm does not support physical symmetries. " *
                            "Both the MPO and MPS must be defined over plain `ComplexSpace` (sectortype must be `Trivial`)."))
    end

    T = promote_type(eltype(H), eltype(ψ))
    C = _src_prepass(H, ψ, maxdim, T)
    L = length(H)

    H_L = removeunit(H[L], Val(4))
    ψ_L = removeunit(ψ[L], Val(3))
    @tensoropt Y_L[a; c] := C[L-1][a, b, d] * H_L[b, c, e] * ψ_L[d, e]
    _, η_L = right_orth(Y_L)
    @tensoropt S_L[a, b; c] := conj(η_L[a, d]) * H_L[b, d, e] * ψ_L[c, e] # S: (new_bond_left, bond left H, bond left ψ)
    ψ.tensors[L] = insertleftunit(repartition(η_L, 2, 0), 3)

    S_ip = S_L
    for i in reverse(2:L-1)
        H_i, ψ_i = H[i], ψ[i]
        @tensoropt Y_i[a; b c] := C[i-1][a, d, e] * H_i[d, b, g, f] * ψ_i[e, g, h] * S_ip[c, f, h]
        _, η_i = right_orth(Y_i)
        @tensoropt S_i[c, b; a] := conj(η_i[c, d, e]) * H_i[b, d, g, f] * ψ_i[a, g, h] * S_ip[e, f, h]
        ψ.tensors[i] = permute(η_i, ((1, 2), (3, )))
        S_ip = S_i
    end

    H_1 = removeunit(H[1], Val(1))
    ψ_1 = removeunit(ψ[1], Val(1))
    @tensoropt η_1[a, b] := H_1[a, c, d] * ψ_1[c, e] * S_ip[b, d, e]
    ψ.tensors[1] = repartition(insertleftunit(η_1, 1), 2, 1)
    set_ortho_lims!(ψ, 0, 2) # State is right-canonicalized by construction
    return ψ
end


"""
    apply!(H::MPO, ψ::MPS; alg = :zipup, kwargs...) -> MPS

Dispatch to the MPO-MPS multiplication algorithm selected by `alg`.

# Supported Algorithms
- `:zipup`: Standard zip-up method using sequential deterministic truncations. 
  See [`apply!(H, ψ, Val(:zipup))`](@ref) for specific keyword arguments.
- `:src`: Successive Randomized Compression (SRC) using single-pass randomized sketching. 
  See [`apply!(H, ψ, Val(:src))`](@ref) for specific keyword arguments.

# Returns
- The compressed result matrix product state, modified in place.
"""
apply!(H::MPO, ψ::MPS; alg=:zipup, kwargs...) = apply!(H, ψ, Val(alg); kwargs...)

"""
    apply(H::MPO, ψ::MPS; kwargs...) -> MPS

Non-mutating version of [`apply!`](@ref): returns a new MPS representing
`H|ψ⟩` without modifying `ψ`. See `apply!` for keyword arguments.
"""
apply(H::MPO,  ψ::MPS; kwargs...) = apply!(H, copy(ψ); kwargs...)