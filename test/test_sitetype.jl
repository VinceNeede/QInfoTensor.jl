@testset "SiteType construction" begin
    # default Sym = Trivial
    st = SiteType(:SpinHalf)
    @test st isa SiteType{:SpinHalf,Trivial}

    # opt-in symmetry via keyword, not positional
    stU1 = SiteType(:SpinHalf; sym=U1Irrep)
    @test stU1 isa SiteType{:SpinHalf,U1Irrep}

    # string entry point
    @test SiteType("SpinHalf") isa SiteType{:SpinHalf,Trivial}

    # parametric tag (tuple), with and without symmetry — this is the
    # combination that was previously impossible (params... vs Sym
    # colliding positionally)
    stBoson = SiteType(:Boson, 4)
    @test stBoson isa SiteType{(:Boson, 4),Trivial}
    stBosonU1 = SiteType(:Boson, 4; sym=U1Irrep)
    @test stBosonU1 isa SiteType{(:Boson, 4),U1Irrep}
end

@testset "@alias_sitetype forwarding" begin
    for sym in (Trivial, U1Irrep)
        st_canon = SiteType(:SpinHalf; sym=sym)
        st_alias1 = SiteType(:Qubit; sym=sym)
        st_alias2 = SiteType("S=1/2"; sym=sym)

        @test TensorKit.space(st_alias1) == TensorKit.space(st_canon)
        @test TensorKit.space(st_alias2) == TensorKit.space(st_canon)
        @test op(st_alias1, OpName(:Sz)) == op(st_canon, OpName(:Sz))
        @test state(st_alias1, StateName(:Up)) == state(st_canon, StateName(:Up))
    end
end

@testset "space(::SiteType{:SpinHalf,...})" begin
    @test TensorKit.space(SiteType(:SpinHalf)) == ℂ^2
    @test TensorKit.space(SiteType(:SpinHalf; sym=U1Irrep)) ==
          U1Space(1 // 2 => 1, -1 // 2 => 1)

    # no fallback for an undeclared SiteType — must MethodError, not
    # silently return something
    @test_throws MethodError TensorKit.space(SiteType(:NotASite))
end

@testset "optensor: charge-conserving ops, legal under Trivial and U1Irrep" begin
    # NOTE: domain(t)/codomain(t) return a ProductSpace, not the bare
    # ElementarySpace V — compare domain==codomain for shape, and use
    # space(t, i) (the per-leg accessor) to check against V itself.
    for sym in (Trivial, U1Irrep)
        st = SiteType(:SpinHalf; sym=sym)
        V = TensorKit.space(st)

        tId = optensor(st, OpName(:Id))
        @test domain(tId) == codomain(tId)
        @test space(tId, 1) == V
        @test tId ≈ id(V)

        tSz = optensor(st, OpName(:Sz))
        @test domain(tSz) == codomain(tSz)
        @test space(tSz, 1) == V
        @test eltype(tSz) == Float64  # Sz is real; optensor shouldn't force complex

        tS2 = optensor(st, OpName(:S2))
        @test domain(tS2) == codomain(tS2)
        @test space(tS2, 1) == V

        tProjUp = optensor(st, OpName(:ProjUp))
        tProjDn = optensor(st, OpName(:ProjDn))
        @test tProjUp + tProjDn ≈ id(V)

        # Rz: complex-valued AND symmetric at once — checks T and Sym are
        # genuinely independent axes, not accidentally coupled
        tRz = optensor(st, OpName(:Rz); θ=0.3)
        @test eltype(tRz) == ComplexF64
        @test domain(tRz) == codomain(tRz)
        @test space(tRz, 1) == V
    end
end

@testset "optensor: charge-mixing ops, Trivial-only by construction" begin
    stTrivial = SiteType(:SpinHalf)
    stU1 = SiteType(:SpinHalf; sym=U1Irrep)

    # Trivial: works fine
    tSx = optensor(stTrivial, OpName(:Sx))
    @test domain(tSx) == codomain(tSx)
    @test space(tSx, 1) == ℂ^2
    tSy = optensor(stTrivial, OpName(:Sy))
    @test eltype(tSy) == ComplexF64
    tRx = optensor(stTrivial, OpName(:Rx); θ=0.7)
    tRy = optensor(stTrivial, OpName(:Ry); θ=0.7)

    # U1Irrep: no op(...) method defined at all for Sx/Sy/Rx/Ry under
    # U1Irrep, so this must fail at `op` itself, before optensor even
    # gets to construct a (charge-violating) TensorMap.
    @test_throws MethodError op(stU1, OpName(:Sx))
    @test_throws MethodError op(stU1, OpName(:Sy))
    @test_throws MethodError op(stU1, OpName(:Rx); θ=0.7)
    @test_throws MethodError op(stU1, OpName(:Ry); θ=0.7)
    @test_throws MethodError optensor(stU1, OpName(:Sx))

    # Sanity check on the *mechanism* itself, independent of our dispatch
    # choices: feeding Sx's dense matrix directly into TensorMap against a
    # U1Space manually should throw ArgumentError (TensorKit's own charge
    # check), confirming *why* we chose not to define op(...) there.
    V = U1Space(1 // 2 => 1, -1 // 2 => 1)
    Sx_dense = op(stTrivial, OpName(:Sx))
    @test_throws ArgumentError TensorMap(Sx_dense, V ← V)
end

@testset "optensor: S+/S- are Trivial-only, dense op(...) still works" begin
    stTrivial = SiteType(:SpinHalf)
    stU1 = SiteType(:SpinHalf; sym=U1Irrep)

    Sp = op(stTrivial, OpName("S+"))
    Sm = op(stTrivial, OpName("S-"))
    @test Sp == Float64[0 1; 0 0]
    @test Sm == Float64[0 0; 1 0]

    tSp = optensor(stTrivial, OpName("S+"))
    @test domain(tSp) == codomain(tSp)
    @test space(tSp, 1) == ℂ^2

    # no U1Irrep method defined (S+/S- are not invariant under U(1) at
    # all — they pick up a phase, not stay fixed — see qubit.jl comment)
    # — must MethodError
    @test_throws MethodError op(stU1, OpName("S+"))
    @test_throws MethodError op(stU1, OpName("S-"))

    # and confirm the underlying reason, same as Sx above
    V = U1Space(1 // 2 => 1, -1 // 2 => 1)
    @test_throws ArgumentError TensorMap(Sp, V ← V)
end

@testset "statetensor: Trivial" begin
    st = SiteType(:SpinHalf)
    V = TensorKit.space(st)

    tUp = statetensor(st, StateName(:Up))
    @test space(tUp, 1) == V
    @test eltype(tUp) == Float64  # inferred from state(...)'s own eltype

    tDn = statetensor(st, StateName(:Dn))
    @test space(tDn, 1) == V

    # explicit T override
    tUpC = statetensor(ComplexF64, st, StateName(:Up))
    @test eltype(tUpC) == ComplexF64

    # unicode / bit-label aliases agree with canonical names
    @test statetensor(st, StateName("↑")) ≈ tUp
    @test statetensor(st, StateName("0")) ≈ tUp
    @test statetensor(st, StateName("↓")) ≈ tDn
    @test statetensor(st, StateName("1")) ≈ tDn

    tCoh = statetensor(st, StateName(:Coherent); θ=π / 3)
    @test space(tCoh, 1) == V
end

@testset "statetensor: nonzero-charge states aren't representable as bare (1,0) tensors under U1Irrep" begin
    # Same structural issue as S+/S- in qubit.jl: |Up>/|Dn> individually
    # carry nonzero Sz charge, so they can't be embedded as an invariant
    # rank-(1,0) tensor with domain=one(V) (the charge-0 unit space) — V's
    # U1Irrep grading (charges ±1/2) has no charge-0 sector at all, so the
    # blocksectors intersection is empty regardless of which entry is
    # nonzero. This needs a compensating/auxiliary leg to solve properly
    # (MPS-boundary-style), same open item as S+/S-, not solved here.
    stU1 = SiteType(:SpinHalf; sym=U1Irrep)
    @test_throws ArgumentError statetensor(stU1, StateName(:Up))
    @test_throws ArgumentError statetensor(stU1, StateName(:Dn))

    # :Coherent has no U1Irrep method at all (not charge-definite even in
    # principle) — MethodError, before even reaching the ArgumentError case
    @test_throws MethodError statetensor(stU1, StateName(:Coherent); θ=π / 3)
end