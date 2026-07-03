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

export SiteType, OpName, StateName, @alias_sitetype
export op, state, optensor, statetensor

end # module QInfoTensor