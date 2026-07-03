# usings (Test, TensorKit, QInfoTensor) live in runtests.jl.

@testset "OpSum / add! / + bookkeeping" begin
    H = OpSum()
    @test isempty(H.terms)

    add!(H, 2.0, 1 => (OpName(:Sz), (;)))
    @test length(H.terms) == 1
    @test H.terms[1].coeff == 2.0

    H += (1.5, :Sz, 3, :Sz, 1)   # deliberately out of site order
    @test length(H.terms) == 2
    # ops within a term get sorted by site
    @test first.(H.terms[2].ops) == [1, 3]

    H += (0.5, "Sx", 2)  # string operator name, no params
    @test length(H.terms) == 3
    @test H.terms[3].ops[1][2][1] == OpName(:Sx)

    @test_throws ArgumentError add!(OpSum(), ("not a number", "Sz", 1))
end

@testset "TFIM OpSum -> MPO matches an independently-built Hamiltonian" begin
    L = 4
    J = 1.0
    h = 0.5
    sites = sitetypes(:SpinHalf, L)

    H = OpSum()
    for i in 1:(L-1)
        H += (-J, :Sz, i, :Sz, i + 1)
    end
    for i in 1:L
        H += (-h, :Sx, i)
    end

    mpo = MPO(H, sites)
    @test length(mpo) == L

    # Bond dimensions, traced through _fsm_states for this specific
    # Hamiltonian: leftmost/rightmost bonds are {:I}/{:F} only (dim 1);
    # the bond after site i carries {:I,:F,(α,1)} (dim 3) whenever a
    # 2-site Sz_i*Sz_{i+1} term is "in flight" across it — true for
    # bonds after sites 1,2,3 here (each is exactly one such term's span).
    @test dim(codomain(mpo[1])[1]) == 1  # leftmost: {:I}
    @test dim(domain(mpo[1])[2]) == 3    # after site 1: {:I,:F,(1,1)} from Sz1*Sz2
    @test dim(codomain(mpo[L])[1]) == 3  # before site 4: {:I,:F,(3,1)} from Sz3*Sz4
    @test dim(domain(mpo[L])[2]) == 1    # rightmost: {:F}

    # Contract the whole chain into one big (L,L)-rank operator tensor.
    # Both FSM boundary bonds are dimension-1 by construction (leftmost
    # = {:I}, rightmost = {:F}), selected via explicit one-hot tensors.
    # conj on e_I (pairs against a CODOMAIN leg) but not e_F (pairs
    # against a DOMAIN leg) — established rule from the earlier scratch
    # script check (scratch_opsum_reshape.jl).
    e_I = Tensor(ComplexF64[1], codomain(mpo[1])[1])
    e_F = Tensor(ComplexF64[1], domain(mpo[L])[2])

    @tensor Hfull[a, b, c, d; e, f, g, h] :=
        conj(e_I[l1]) * mpo[1][l1, a, e, l2] * mpo[2][l2, b, f, l3] *
        mpo[3][l3, c, g, l4] * mpo[4][l4, d, h, l5] * e_F[l5]

    # Independent reference, built entirely separately from the FSM/reshape
    # machinery: single-site operators combined via @tensor directly (a
    # mechanism already validated extensively elsewhere), summed with the
    # right coefficients. Compared to Hfull via ≈ as genuine TensorKit
    # tensors — sidesteps ever needing to know/match TensorKit's internal
    # multi-leg flattening convention against some external kron
    # convention, since both sides use the same (whatever it is) convention.
    Idop = [optensor(sites[i], OpName(:Id)) for i in 1:L]
    Szop = [optensor(sites[i], OpName(:Sz)) for i in 1:L]
    Sxop = [optensor(sites[i], OpName(:Sx)) for i in 1:L]

    function _term_tensor(op1, op2, op3, op4)
        @tensor t[a, b, c, d; e, f, g, h] := op1[a, e] * op2[b, f] * op3[c, g] * op4[d, h]
        return t
    end
    _pick(k, special) = get(special, k, Idop[k])

    H_ref = nothing
    for i in 1:(L-1)
        special = Dict(i => Szop[i], i + 1 => Szop[i+1])
        t = _term_tensor(_pick(1, special), _pick(2, special), _pick(3, special), _pick(4, special))
        H_ref = H_ref === nothing ? (-J) * t : H_ref + (-J) * t
    end
    for i in 1:L
        special = Dict(i => Sxop[i])
        t = _term_tensor(_pick(1, special), _pick(2, special), _pick(3, special), _pick(4, special))
        H_ref = H_ref + (-h) * t
    end

    @test Hfull ≈ H_ref
end