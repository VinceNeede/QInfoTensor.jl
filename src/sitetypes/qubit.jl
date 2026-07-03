# ------------------------------------------------------------------------
# :SpinHalf site type
#
# Ported from the prototype library's qubit.jl. Reused verbatim wherever
# possible; generic over Sym ∈ {Trivial, U1Irrep} (both confirmed
# abelian/basis-change-free) ONLY for operators/states that are actually
# charge-conserving in the Sz basis. SU2Irrep is not addressed at all here
# — design_notes.md open question 1, still deferred.
#
# Charge convention: U1Irrep charge label = Sz eigenvalue directly
# (fractional charges confirmed to work), with basis order (Up, Dn) =
# (+1/2, -1/2) matching the dense matrices' row/column order below.
# ------------------------------------------------------------------------

const _AbelianSym = Union{Trivial,U1Irrep}

@alias_sitetype Qubit => SpinHalf
@alias_sitetype "S=1/2" => SpinHalf

# ── space ────────────────────────────────────────────────────────────────

TensorKit.space(::SiteType{:SpinHalf,Trivial}) = ℂ^2
TensorKit.space(::SiteType{:SpinHalf,U1Irrep}) = U1Space(1 // 2 => 1, -1 // 2 => 1)

# ── state ────────────────────────────────────────────────────────────────
#
# Up/Dn are individually charge-definite (Sz eigenstates) -> legal under
# both symmetries. :Coherent is a superposition of Sz eigenstates -> NOT
# charge-definite -> Trivial only (no U1Irrep method defined; calling it
# under sym=U1Irrep is a MethodError, which is correct, not a gap).

state(::SiteType{:SpinHalf,Sym}, ::StateName{:Up}) where {Sym<:_AbelianSym} = [1.0, 0.0]
state(::SiteType{:SpinHalf,Sym}, ::StateName{:Dn}) where {Sym<:_AbelianSym} = [0.0, 1.0]

# Unicode aliases
const _Sp_Up = StateName{Symbol("↑")}
const _Sp_Dn = StateName{Symbol("↓")}
state(st::SiteType{:SpinHalf,Sym}, ::_Sp_Up) where {Sym<:_AbelianSym} = state(st, StateName(:Up))
state(st::SiteType{:SpinHalf,Sym}, ::_Sp_Dn) where {Sym<:_AbelianSym} = state(st, StateName(:Dn))

# Bit-label aliases
state(st::SiteType{:SpinHalf,Sym}, ::StateName{Symbol("0")}) where {Sym<:_AbelianSym} = state(st, StateName(:Up))
state(st::SiteType{:SpinHalf,Sym}, ::StateName{Symbol("1")}) where {Sym<:_AbelianSym} = state(st, StateName(:Dn))

state(::SiteType{:SpinHalf,Trivial}, ::StateName{:Coherent}; θ::Real, ϕ::Real=0.0) =
    ComplexF64[cos(θ / 2), exp(im * ϕ) * sin(θ / 2)]

# ── op ───────────────────────────────────────────────────────────────────
#
# Diagonal-in-Sz (charge-conserving) -> legal under both symmetries:
# Id, Sz, S², the projectors, Rz.

op(::SiteType{:SpinHalf,Sym}, ::OpName{:Id}) where {Sym<:_AbelianSym} = Float64[1 0; 0 1]
op(::SiteType{:SpinHalf,Sym}, ::OpName{:Sz}) where {Sym<:_AbelianSym} = Float64[0.5 0; 0 -0.5]
op(::SiteType{:SpinHalf,Sym}, ::OpName{:S2}) where {Sym<:_AbelianSym} = Float64[0.75 0; 0 0.75]

op(::SiteType{:SpinHalf,Sym}, ::OpName{:ProjUp}) where {Sym<:_AbelianSym} = Float64[1 0; 0 0]
op(::SiteType{:SpinHalf,Sym}, ::OpName{:ProjDn}) where {Sym<:_AbelianSym} = Float64[0 0; 0 1]
const _OpProjUp = OpName{Symbol("Proj↑")}
const _OpProjDn = OpName{Symbol("Proj↓")}
op(st::SiteType{:SpinHalf,Sym}, ::_OpProjUp) where {Sym<:_AbelianSym} = op(st, OpName(:ProjUp))
op(st::SiteType{:SpinHalf,Sym}, ::_OpProjDn) where {Sym<:_AbelianSym} = op(st, OpName(:ProjDn))

"""
    op(::SiteType{:SpinHalf,Sym}, ::OpName{:Rz}; θ::Real) where {Sym<:Union{Trivial,U1Irrep}}

Rotation by `θ` around z: exp(-iθSz) = diag(e^{-iθ/2}, e^{iθ/2}). Diagonal
in Sz -> charge-conserving -> legal under U1Irrep too (unlike Rx/Ry below).
"""
op(::SiteType{:SpinHalf,Sym}, ::OpName{:Rz}; θ::Real) where {Sym<:_AbelianSym} =
    ComplexF64[exp(-im * θ / 2) 0; 0 exp(im * θ / 2)]

# Off-diagonal-in-Sz (charge-mixing) -> Trivial ONLY. Calling optensor with
# these under sym=U1Irrep will throw ArgumentError at the TensorMap(M, V←V)
# step (confirmed behavior) — no method is defined here for U1Irrep, so
# calling op(...) itself already fails cleanly before that point.

op(::SiteType{:SpinHalf,Trivial}, ::OpName{:Sx}) = Float64[0 0.5; 0.5 0]
op(::SiteType{:SpinHalf,Trivial}, ::OpName{:Sy}) = ComplexF64[0 -0.5im; 0.5im 0]

"""
    op(::SiteType{:SpinHalf,Trivial}, ::OpName{:Rx}; θ::Real)

Rotation by `θ` around x: exp(-iθSx) = cos(θ/2)𝟙 - i sin(θ/2)σx.
"""
op(::SiteType{:SpinHalf,Trivial}, ::OpName{:Rx}; θ::Real) =
    ComplexF64[cos(θ / 2) -im*sin(θ / 2)
        -im*sin(θ / 2) cos(θ / 2)]

"""
    op(::SiteType{:SpinHalf,Trivial}, ::OpName{:Ry}; θ::Real)

Rotation by `θ` around y: exp(-iθSy) = cos(θ/2)𝟙 - i sin(θ/2)σy.
"""
op(::SiteType{:SpinHalf,Trivial}, ::OpName{:Ry}; θ::Real) =
    Float64[cos(θ / 2) -sin(θ / 2)
        sin(θ / 2) cos(θ / 2)]

# S+, S- are OFF-DIAGONAL in Sz (Sz=-1/2 <-> Sz=+1/2), exactly like Sx/Sy
# above — they are NOT invariant tensors under U(1) at all (under a U(1)
# rotation, S+ picks up a phase, S+ -> e^{iα}S+, rather than staying
# fixed), so TensorMap(Sp_dense, V←V) would throw the same ArgumentError
# Sx does. This is not a limitation of optensor's V←V signature that a
# wider signature would fix.
#
# The actual symmetric representation needs an auxiliary/virtual leg
# carrying the missing charge (+1 for S+, -1 for S-), so that e.g. an
# MPO tensor (site_out, bond_out) ← (site_in) is invariant overall, with
# that bond charge threading across the chain to cancel against a
# compensating S- elsewhere (e.g. in an Sp_i * Sm_j Hamiltonian term).
# That's an MPO-construction-level concept (roadmap step 2, OpSum -> MPO
# FSM porting) — a bare per-site optensor(st, on) has no virtual leg to
# put that charge on. Left dense-only, Trivial-only, until that's
# designed. OPEN ITEM, not solved here.
const _OpSp = OpName{Symbol("S+")}
const _OpSm = OpName{Symbol("S-")}
op(::SiteType{:SpinHalf,Trivial}, ::_OpSp) = Float64[0 1; 0 0]
op(::SiteType{:SpinHalf,Trivial}, ::_OpSm) = Float64[0 0; 1 0]