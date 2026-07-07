# ------------------------------------------------------------------------
# AbstractTensorTrain{T,S,A}
#
# Shared supertype for MPS and MPO — both are, structurally, a finite
# chain of tensors with matching bond spaces; the only difference is the
# tensor rank (MPSTensor vs MPOTensor).
#
# No stub/interface methods declared here for tensors/llim/rlim — Julia
# already raises MethodError when a concrete type hasn't implemented one;
# an explicit throwing stub on the abstract type adds nothing and can
# only get in the way. Each concrete type (MPS, MPO) defines its own
# tensors(ψ), llim(ψ), rlim(ψ), set_ortho_lims!(ψ,l,r) directly.
#
# isortho/orthocenter ARE defined here, generically, since they're pure
# formulas over llim(ψ)/rlim(ψ) — dispatch reaches whichever concrete
# implementation is in play. If MPO's llim/rlim bookkeeping (used in
# zip-up) ever turns out to need different isortho/orthocenter semantics
# than MPS's, a specific `isortho(ψ::MPO)` method overrides this generic
# one — so nothing is lost by not duplicating it up front.
# ------------------------------------------------------------------------

abstract type AbstractTensorTrain{T,S<:ElementarySpace,A<:DenseVector{T}} end

Base.length(ψ::AbstractTensorTrain) = length(tensors(ψ))
Base.getindex(ψ::AbstractTensorTrain, i::Int) = tensors(ψ)[i]
Base.eachindex(ψ::AbstractTensorTrain) = eachindex(tensors(ψ))
Base.iterate(ψ::AbstractTensorTrain, args...) = iterate(tensors(ψ), args...)

Base.eltype(::AbstractTensorTrain{T}) where {T} = T
TensorKit.spacetype(::AbstractTensorTrain{T,S}) where {T,S} = S
TensorKit.storagetype(::AbstractTensorTrain{T,S,A}) where {T,S,A} = A
linkind(x::AbstractTensorTrain, pos::Int) = (dom = domain(x[pos]); dom[length(dom)])
linkinds(x::AbstractTensorTrain) = [linkind(x, i) for i in eachindex(x)]
linkdims(x::AbstractTensorTrain) = dim.(linkinds(x))
maxlinkdim(x::AbstractTensorTrain) = maximum(linkdims(x))

"""
    isortho(ψ::AbstractTensorTrain) -> Bool

Whether `ψ` currently has a unique, well-defined orthogonality center.
"""
isortho(ψ::AbstractTensorTrain) = llim(ψ) + 1 == rlim(ψ) - 1

"""
    orthocenter(ψ::AbstractTensorTrain) -> Int

Site index of the unique orthogonality center. Throws if `!isortho(ψ)`.
"""
function orthocenter(ψ::AbstractTensorTrain)
    isortho(ψ) || throw(ArgumentError(
        "$(typeof(ψ).name.wrapper) has no unique orthogonality center (llim=$(llim(ψ)), rlim=$(rlim(ψ)))"
    ))
    return llim(ψ) + 1
end

"""
    reset_ortho_lims!(ψ::AbstractTensorTrain) -> ψ

Mark `ψ` as fully non-orthogonalized (`llim=0`, `rlim=length(ψ)+1`).
Used by `compress!`/`compress!!` since every bond needs revisiting to be
truncated, regardless of `ψ`'s orthogonality status going in.
"""
reset_ortho_lims!(ψ::AbstractTensorTrain) = set_ortho_lims!(ψ, 0, length(ψ) + 1)

Base.copy(ψ::T) where {T<:AbstractTensorTrain} = T(copy(tensors(ψ)), llim(ψ), rlim(ψ))
