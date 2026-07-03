# usings (Test, TensorKit, QInfoTensor) live in runtests.jl.

@testset "inner(ψ,H,φ): basic error handling" begin
    sites3 = sitetypes(:SpinHalf, 3)
    sites4 = sitetypes(:SpinHalf, 4)
    H3 = OpSum()
    H3 += (1.0, :Sz, 1, :Sz, 2)
    mpo3 = MPO(H3, sites3)

    ψ3 = MPS(sites3, fill(StateName(:Up), 3))
    ψ4 = MPS(sites4, fill(StateName(:Up), 4))

    @test_throws ArgumentError inner(ψ4, mpo3, ψ4)  # MPS/MPO length mismatch
    @test_throws ArgumentError inner(ψ3, mpo3, ψ4)  # ψ/φ length mismatch
end

@testset "inner(ψ,H,φ): TFIM expectation values on product states (analytic)" begin
    # Hand-computable check: for a Sz-product-eigenstate, ⟨Sz_i Sz_{i+1}⟩
    # is just the product of the two eigenvalues (±0.5 each), and
    # ⟨Sx_i⟩=0 always (Sx is purely off-diagonal in the Sz basis). So
    # ⟨ψ|H_TFIM|ψ⟩ = -J*Σ⟨Sz_i Sz_{i+1}⟩ - h*0, computable by hand,
    # independent of the MPO/FSM machinery entirely.
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

    # |↑↑↑↑⟩: every Sz_i*Sz_{i+1} = (+0.5)*(+0.5) = 0.25, 3 bonds
    ψ_up = MPS(sites, fill(StateName(:Up), L))
    @test real(inner(ψ_up, mpo, ψ_up)) ≈ -J * 3 * 0.25

    # |↑↓↑↓⟩: every adjacent pair has opposite sign, Sz_i*Sz_{i+1} = -0.25
    ψ_alt = MPS(sites, [StateName(:Up), StateName(:Dn), StateName(:Up), StateName(:Dn)])
    @test real(inner(ψ_alt, mpo, ψ_alt)) ≈ -J * 3 * (-0.25)

    # cross term: different states, no simple hand formula, but should at
    # least be finite and NOT equal either diagonal value (basic sanity
    # check that this isn't accidentally returning one of the diagonal
    # cases due to some indexing bug)
    cross = inner(ψ_up, mpo, ψ_alt)
    @test !(cross ≈ real(inner(ψ_up, mpo, ψ_up)))
    @test !(cross ≈ real(inner(ψ_alt, mpo, ψ_alt)))
end

@testset "inner(ψ,H,φ): cross-check against an independently-built dense operator, random MPS" begin
    # Stronger check than the product-state case above: exercises real,
    # non-trivial bond dimensions on BOTH the MPS and MPO sides, not just
    # dimension-1 bonds. Two completely independent constructions are
    # compared: (1) inner(ψ,H,φ) itself (MPO/FSM machinery), and (2) a
    # dense reference operator (built via optensor+@tensor, no FSM
    # involved at all) sandwiched between fully-contracted dense state
    # tensors (built via boundary selectors, no inner() involved at all).
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

    # Full dense-ish state tensor for a (possibly non-trivial-bond) MPS,
    # hardcoded for L=4. Boundary-selector conj rule matches the earlier
    # (validated) Hfull construction in test_opsum.jl: conj on the
    # codomain-leg pairing (e_l, vs ψ[1]'s left/codomain leg), not on the
    # domain-leg one (e_r, vs ψ[4]'s right/domain leg).
    function _mps_dense(ψ::MPS)
        e_l = Tensor(ComplexF64[1], codomain(ψ[1])[1])
        e_r = Tensor(ComplexF64[1], domain(ψ[4])[1])
        @tensor full[a, b, c, d] :=
            conj(e_l[l1]) * ψ[1][l1, a, l2] * ψ[2][l2, b, l3] *
            ψ[3][l3, c, l4] * ψ[4][l4, d, l5] * e_r[l5]
        return full
    end

    ψ = random_mps(sites, 4)
    φ = random_mps(sites, 4)
    ψfull = _mps_dense(ψ)
    φfull = _mps_dense(φ)

    @tensor val_ref = conj(ψfull[a, b, c, d]) * H_ref[a, b, c, d, e, f, g, h] * φfull[e, f, g, h]

    val_inner = inner(ψ, mpo, φ)
    @test val_inner ≈ val_ref
end

@testset "MPS Expectation Values" begin
    # Setup a deterministic 4-site state: |↑↓↑↓⟩
    sites = sitetypes(:SpinHalf, 4)
    ψ = MPS(sites, ["Up", "Dn", "Up", "Dn"])
    
    # Adjust these values based on your library's convention (e.g., 0.5 vs 1.0)
    expected_sz = [0.5, -0.5, 0.5, -0.5] 

    @testset "Operator Resolution Layer" begin
        # Test that _get_operator successfully resolves types identically
        op_str = QInfoTensor._get_operator(sites[1], "Sz")
        op_name = QInfoTensor._get_operator(sites[1], OpName("Sz"))
        
        @test op_str isa AbstractTensorMap
        @test op_str == op_name
    end

    @testset "Single Position Dispatch" begin
        # Test out-of-place single site
        @test expect(ψ, sites, "Sz", 1) ≈ expected_sz[1]
        @test expect(ψ, sites, "Sz", 2) ≈ expected_sz[2]
        
        # Test passing an explicit AbstractTensorMap directly
        sz_tensor = QInfoTensor._get_operator(sites[1], "Sz")
        @test expect(ψ, sz_tensor, 3) ≈ expected_sz[3]
    end

    @testset "Mutating vs Non-Mutating" begin
        ψ_copy = copy(ψ)
        sz_tensor = QInfoTensor._get_operator(sites[1], "Sz")
        
        # expect! is allowed to change the internal gauge/orthogonalization center
        val_mutating = expect!(ψ_copy, sz_tensor, 2)
        
        # expect should leave the original state's gauge or data unharmed
        val_safe = expect(ψ, sz_tensor, 2)
        
        @test val_mutating ≈ val_safe ≈ expected_sz[2]
    end

    @testset "Multiple Positions Dispatch" begin
        # Test the default tracking behavior (all positions)
        @test expect(ψ, sites, "Sz") ≈ expected_sz
        
        # Test a subset of positions, out of order (verifies the internal `sort`)
        subset = [4, 2]
        @test expect(ψ, sites, "Sz", subset) ≈ expected_sz[subset]
    end

    @testset "Uniform Lattice Shortcuts" begin
        # Test when passing a single SiteType instead of a Vector
        single_site = sites[1] # :SpinHalf instance
        
        # Should automatically fill the lattice and compute all sites
        @test expect(ψ, single_site, "Sz") ≈ expected_sz
        
        # Should work for specific positions with a single SiteType
        @test expect(ψ, single_site, "Sz", [1, 3]) ≈ [expected_sz[1], expected_sz[3]]
    end
end


@testset "apply!/apply: zip-up MPO-MPS contraction" begin
    L = 6
    J = 1.0
    h = 0.5
    sites = sitetypes(:SpinHalf, L)
 
    H_os = OpSum()
    for i in 1:(L-1)
        H_os += (-J, :Sz, i, :Sz, i + 1)
    end
    for i in 1:L
        H_os += (-h, :Sx, i)
    end
    H = MPO(H_os, sites)
    orthogonalize!(H, 1)   # required for zip-up accuracy; suppresses warning
 
    @testset "apply: ⟨ψ|Hφ⟩ == ⟨ψ|H|φ⟩ for random states" begin
        ψ = random_mps(sites, 4)
        φ = random_mps(sites, 4)
        Hφ = apply(H, φ; maxdim=16, cutoff=1e-10)
        @test inner(ψ, Hφ) ≈ inner(ψ, H, φ) atol=1e-6
    end
 
    @testset "apply: TFIM energy on |↑↑...↑⟩ matches analytic value" begin
        # ⟨↑...↑|H_TFIM|↑...↑⟩ = -J*(L-1)*0.25, ⟨Sx⟩=0 for Sz eigenstates
        ψ_up = MPS(sites, fill(StateName(:Up), L))
        Hψ = apply(H, ψ_up; maxdim=16, cutoff=1e-10)
        @test inner(ψ_up, Hψ) ≈ -J * (L - 1) * 0.25 atol=1e-8
    end
 
    @testset "apply!: mutates ψ in place, apply: leaves original unchanged" begin
        φ = random_mps(sites, 4)
        φ_copy = copy(φ)
        Hφ = apply(H, φ; maxdim=16, cutoff=1e-10)
 
        # φ unchanged after non-mutating apply
        @test inner(φ, φ_copy) ≈ sqrt(real(inner(φ, φ)) * real(inner(φ_copy, φ_copy))) atol=1e-10
 
        # apply! result matches non-mutating version
        apply!(H, φ_copy; maxdim=16, cutoff=1e-10)
        ψ = random_mps(sites, 4)
        @test inner(ψ, φ_copy) ≈ inner(ψ, Hφ) atol=1e-6
    end
 
    @testset "apply: result is orthogonalized after compress!" begin
        φ = random_mps(sites, 4)
        Hφ = apply(H, φ; maxdim=16, cutoff=1e-10)
        @test isortho(Hφ)
        @test orthocenter(Hφ) == 1
    end
 
    @testset "apply: warning issued when H is not left-canonicalized" begin
        H_unortho = MPO(H_os, sites)
        φ = random_mps(sites, 4)
        @test_logs (:warn, r"not left-canonicalized") apply(H_unortho, φ; maxdim=16, cutoff=1e-10)
    end
end