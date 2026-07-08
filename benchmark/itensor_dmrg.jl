function _itensor_tfim_opsum(L::Int; J::Real=1.0, h::Real=1.0, periodic::Bool)
    os = ITensorMPS.OpSum()
    range_ = periodic ? (1:L) : (1:L-1)
    for i in range_
        j = periodic ? mod1(i + 1, L) : i + 1
        os += -J, "Sz", i, "Sz", j
    end
    for i in 1:L
        os += -h, "Sx", i
    end
    return os
end

"""
    build_itensor_dmrg_inputs(problem::DMRGProblem) -> (sites, H, ψ0)

ITensor analog of `build_dmrg_inputs`. Constructs an identical Hamiltonian MPO 
and a random starting state fixed to link dimension 2.
"""
function build_itensor_dmrg_inputs(problem::DMRGProblem)
    L = problem.hamiltonian.L
    periodic = problem.hamiltonian.periodic
    sites = ITensorMPS.siteinds("S=1/2", L)
    
    os = _itensor_tfim_opsum(L; J=1.0, h=1.0, periodic=periodic)
    H = ITensorMPS.MPO(os, sites)
    
    # Generate an initial state matching the starting bond dimension 2
    ψ0 = ITensorMPS.random_mps(sites; linkdims=2)
    return sites, H, ψ0
end