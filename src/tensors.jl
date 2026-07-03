# ------------------------------------------------------------------------
# MPSTensor / MPOTensor — leg-shape aliases, agreed in chat:
#   - concrete TensorMap, not AbstractTensorMap (no wrapper struct either)
#   - T/S/A independent: scalar type, space (carries symmetry via its
#     Sector), storage backend — confirmed via TensorMap{T,S,N1,N2,A},
#     5 params, from a real session (`typeof(t)` on both a Trivial- and a
#     U1Irrep-sector tensor).
#   - A<:DenseVector{T}: containers stay CPU-resident (Vector{T}) always;
#     GPU acceleration, if/when added, is transient inside a specific
#     operation (adapt in, compute, adapt back out) — not a container-wide
#     storage swap. See chat discussion.
#
# MPSTensor: (left,phys) -> right       i.e. codomain=(left,phys), domain=(right,)
# MPOTensor: (left,site_out) -> (site_in,right)
#   NOTE: changed from an earlier (right,site_in) domain order — this one
#   (bond legs on the outside, both physical legs adjacent in the middle)
#   makes orthogonalization's per-site steps plain contiguous repartition
#   calls in both directions, no permute needed; see orthogonalize.jl.
# ------------------------------------------------------------------------

const MPSTensor{T,S<:ElementarySpace,A<:DenseVector{T}} = TensorMap{T,S,2,1,A}
const MPOTensor{T,S<:ElementarySpace,A<:DenseVector{T}} = TensorMap{T,S,2,2,A}