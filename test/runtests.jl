using Test
using TensorKit
using QInfoTensor
using LinearAlgebra

@testset "QInfoTensor.jl" begin
    include("test_sitetype.jl")
    include("test_mps.jl")
    include("test_opsum.jl")
    include("test_mpo.jl")
    include("test_dmrg.jl")
end