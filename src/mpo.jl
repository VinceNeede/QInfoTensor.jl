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