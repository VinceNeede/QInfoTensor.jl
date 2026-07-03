using Test
using TensorKit
using QInfoTensor

@testset "QInfoTensor.jl" begin
    include("test_sitetype.jl")
    include("test_mps.jl")
end