module QInfoTensor

using TensorKit
using TensorOperations

# ------------------------------------------------------------------------
# Core types, built up incrementally and agreed step-by-step (see chat /
# design_notes.md). Current state: SiteType/OpName/StateName + space +
# the op/state <-> TensorMap boundary. MPSTensor/MPOTensor and the
# FiniteMPS/FiniteMPO containers are agreed in design but not yet written
# here — next step.
# ------------------------------------------------------------------------

include("sitetypes/tags.jl")
include("sitetypes/space.jl")
include("sitetypes/qubit.jl")

include("tensors.jl")
include("tensortrain.jl")
include("mps.jl")
include("mpo.jl")
include("orthogonalize.jl")
 
export SiteType, OpName, StateName, @alias_sitetype, sitetypes
export op, state, optensor, statetensor
 
export MPSTensor, MPOTensor
export AbstractTensorTrain, MPS, MPO, random_mps
export tensors, llim, rlim, isortho, orthocenter, set_ortho_lims!
export orthogonalize, orthogonalize!, orthogonalize!!, compress, compress!, compress!!, normalize!!
 
end # module QInfoTensor