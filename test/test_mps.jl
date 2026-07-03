# usings (Test, TensorKit, QInfoTensor) live in runtests.jl.

# ------------------------------------------------------------------------
# Test-only helpers — NOT part of the package.
#
# _mps_inner used to duplicate inner(ψ,φ)'s logic by hand; now that
# inner/norm are real package functions (normalize.jl, promoted directly
# from this exact logic), tests below call inner(...) / real(inner(ψ,ψ))
# directly instead. Only the isometry-check helpers remain — those don't
# have a package-level equivalent (isisometric wasn't confirmed to exist
# in this session, see chat).
# ------------------------------------------------------------------------

function _is_left_isometric(t)
    @tensor tt[a; b] := conj(t[c, s, a]) * t[c, s, b]
    return tt ≈ id(domain(t)[1])
end
function _is_right_isometric(t)
    @tensor tt[a; b] := t[a, s, c] * conj(t[b, s, c])
    return tt ≈ id(codomain(t)[1])
end

# ------------------------------------------------------------------------

@testset "MPS product-state constructor" begin
    L = 5
    sites = sitetypes(:SpinHalf, L)
    states = [StateName(:Up), StateName(:Dn), StateName(:Up), StateName(:Up), StateName(:Dn)]
    ψ = MPS(sites, states)

    @test length(ψ) == L
    @test llim(ψ) == 0
    @test rlim(ψ) == L + 1  # unorthogonalized by construction (conservative default)

    # every bond is trivial (dimension 1)
    for i in 1:L
        @test dim(domain(ψ[i])[1]) == 1
        @test dim(codomain(ψ[i])[1]) == 1
    end

    # normalized (product of unit-norm single-site states). Can't use the
    # real norm(ψ) here — it requires isortho(ψ), and this constructor
    # deliberately marks itself unorthogonalized (llim=0,rlim=L+1) even
    # though a product state trivially could be marked otherwise. Use
    # inner directly instead, which is unrestricted.
    @test real(inner(ψ, ψ)) ≈ 1.0

    # spot-check actual data on one site against the raw state vector
    v = state(sites[2], states[2])  # :Dn -> [0.0, 1.0]
    arr = block(ψ[2], Trivial())    # dense (d,1) matrix — block() confirmed
    # working earlier in this session (the very first SiteType/space REPL
    # check); Array(t) does NOT exist as a method, confirmed by running this.
    @test vec(arr) ≈ v

    # default T=Float64 for real-valued states (:Up/:Dn are real)
    @test eltype(ψ[1]) == Float64
end

@testset "random_mps" begin
    L = 8
    maxdim = 4
    sites = sitetypes(:SpinHalf, L)
    ψ = random_mps(sites, maxdim)

    @test length(ψ) == L
    mid = L ÷ 2
    @test llim(ψ) == mid
    @test rlim(ψ) == mid + 2
    @test isortho(ψ)
    @test orthocenter(ψ) == mid + 1

    # left-orthogonal region: sites 1:llim
    for i in 1:llim(ψ)
        @test _is_left_isometric(ψ[i])
    end
    # right-orthogonal region: sites rlim:L
    for i in rlim(ψ):L
        @test _is_right_isometric(ψ[i])
    end

    # bond dimensions never exceed maxdim
    for i in 1:(L-1)
        @test dim(domain(ψ[i])[1]) <= maxdim
    end

    # normalized by construction (center tensor explicitly normalized,
    # flanks isometric) — ψ IS orthogonal here, so the real norm(ψ) applies.
    @test norm(ψ)^2 ≈ 1.0 atol = 1e-10
    # cross-check against the independent full-contraction path — these
    # use genuinely different code paths (orthocenter shortcut vs. full
    # transfer-matrix contraction), so agreement is a real, non-circular
    # validation of the orthocenter-shortcut assumption itself.
    @test norm(ψ)^2 ≈ real(inner(ψ, ψ)) atol = 1e-10

    # T defaults to Float64
    @test eltype(ψ[1]) == Float64
    ψc = random_mps(ComplexF64, sites, maxdim)
    @test eltype(ψc[1]) == ComplexF64
end

@testset "orthogonalize! moves the center and preserves the state" begin
    L = 8
    sites = sitetypes(:SpinHalf, L)
    ψ0 = random_mps(sites, 4)
    nrm0 = real(inner(ψ0, ψ0))

    for target in (1, 3, L ÷ 2 + 1, L)
        ψ = copy(ψ0)
        orthogonalize!(ψ, target)

        @test isortho(ψ)
        @test orthocenter(ψ) == target

        for i in 1:(target-1)
            @test _is_left_isometric(ψ[i])
        end
        for i in (target+1):L
            @test _is_right_isometric(ψ[i])
        end

        # state unchanged: compare |overlap| to norms rather than the raw
        # complex overlap, since QR-based orthogonalization is only
        # unique up to a per-bond phase in general (no QRpos-style
        # positive-diagonal convention assumed here) — magnitude match is
        # the correct invariant to check, not exact phase.
        ov = inner(ψ0, ψ)
        @test abs(ov) ≈ sqrt(nrm0 * real(inner(ψ, ψ))) atol = 1e-8
    end
end

@testset "orthogonalize!! matches orthogonalize! (destructive vs non-destructive)" begin
    L = 6
    sites = sitetypes(:SpinHalf, L)
    ψ0 = random_mps(sites, 4)

    ψ1 = orthogonalize!(copy(ψ0), 2)
    ψ2 = orthogonalize!!(copy(ψ0), 2)  # copy() + !! is technically unsafe in
    # general (see tensortrain.jl mutation-convention note) but ψ0 here is
    # only read from afterward via inner on ψ1 (a SEPARATE copy), so
    # ψ2's destructive factorization corrupting ψ0's shared storage isn't
    # actually exercised as a bug by this particular test — flagging so
    # this isn't mistaken for a blessed usage pattern elsewhere.

    @test isortho(ψ2)
    @test orthocenter(ψ2) == 2
    ov = inner(ψ1, ψ2)
    @test abs(ov) ≈ sqrt(real(inner(ψ1, ψ1)) * real(inner(ψ2, ψ2))) atol = 1e-8
end

@testset "compress!: lossless (no truncation) preserves the state exactly" begin
    L = 8
    sites = sitetypes(:SpinHalf, L)
    ψ0 = random_mps(sites, 4)
    nrm0 = real(inner(ψ0, ψ0))

    ψ = copy(ψ0)
    compress!(ψ; maxdim=100)  # generously large — should truncate nothing

    @test isortho(ψ)
    @test orthocenter(ψ) == length(ψ)  # default center

    ov = inner(ψ0, ψ)
    @test abs(ov) ≈ sqrt(nrm0 * real(inner(ψ, ψ))) atol = 1e-8
end

@testset "compress!: maxdim actually caps bond dimension" begin
    L = 8
    sites = sitetypes(:SpinHalf, L)
    ψ0 = random_mps(sites, 8)  # bigger bonds than we'll allow after compress!

    ψ = copy(ψ0)
    compress!(ψ; maxdim=2)

    for i in 1:(L-1)
        @test dim(domain(ψ[i])[1]) <= 2
    end
    @test isortho(ψ)

    # Lossy truncation: norm DECREASES (discarded weight is dropped, not
    # renormalized back to 1) — a valid truncation is a projection, so it
    # can only lose weight, never gain it.
    n2 = norm(ψ)^2
    @test 0 < n2 <= 1.0 + 1e-8
    @test n2 < 1.0 - 1e-6  # confirms maxdim=2 genuinely constrained this random state

    # Cauchy-Schwarz sanity bound — holds regardless of how much was discarded
    ov = abs(inner(ψ0, ψ))
    @test ov <= sqrt(real(inner(ψ0, ψ0)) * n2) + 1e-8
end

@testset "compress!: cutoff with generous maxdim truncates negligible weight only" begin
    L = 8
    sites = sitetypes(:SpinHalf, L)
    ψ0 = random_mps(sites, 4)
    nrm0 = real(inner(ψ0, ψ0))

    ψ = copy(ψ0)
    compress!(ψ; maxdim=100, cutoff=1e-12)  # essentially lossless at this cutoff

    ov = abs(inner(ψ0, ψ))
    @test ov ≈ sqrt(nrm0 * real(inner(ψ, ψ))) atol = 1e-6
end

@testset "norm/normalize family: require isortho, throw otherwise" begin
    L = 5
    sites = sitetypes(:SpinHalf, L)
    states = fill(StateName(:Up), L)
    ψ_unortho = MPS(sites, states)  # llim=0,rlim=L+1, deliberately not orthogonal

    @test !isortho(ψ_unortho)
    @test_throws ArgumentError norm(ψ_unortho)
    @test_throws ArgumentError normalize(ψ_unortho)
    @test_throws ArgumentError normalize!(copy(ψ_unortho))
    @test_throws ArgumentError normalize!!(copy(ψ_unortho))
end

@testset "norm/normalize family: correctness when isortho" begin
    L = 6
    sites = sitetypes(:SpinHalf, L)
    ψ0 = random_mps(sites, 4)  # already normalized, orthogonal by construction

    @test isortho(ψ0)
    @test norm(ψ0) ≈ 1.0 atol = 1e-10

    # scale ψ0 up (still orthogonal — rescaling the center alone doesn't
    # disturb the isometric flanks) so normalize actually has work to do
    ψ_big = copy(ψ0)
    ψ_big.tensors[orthocenter(ψ_big)] = ψ_big[orthocenter(ψ_big)] * 3.0
    @test norm(ψ_big) ≈ 3.0 atol = 1e-8

    # normalize (non-mutating)
    ψ_n = normalize(ψ_big)
    @test norm(ψ_n) ≈ 1.0 atol = 1e-8
    @test norm(ψ_big) ≈ 3.0 atol = 1e-8  # original untouched

    # normalize! (mutating, container-only)
    ψ1 = copy(ψ_big)
    normalize!(ψ1)
    @test norm(ψ1) ≈ 1.0 atol = 1e-8

    # normalize!! (mutating, in-place storage) — same numerical result
    ψ2 = copy(ψ_big)
    normalize!!(ψ2)
    @test norm(ψ2) ≈ 1.0 atol = 1e-8

    # all three agree on the resulting state (same target direction/scale)
    @test abs(inner(ψ_n, ψ1)) ≈ 1.0 atol = 1e-8
    @test abs(inner(ψ_n, ψ2)) ≈ 1.0 atol = 1e-8

    # normalizing preserves direction: overlap with the pre-scaled original
    # should equal exactly 1/3 in magnitude (ψ_big = 3*ψ0, ψ0 already unit norm)
    @test abs(inner(ψ0, ψ_n)) ≈ 1.0 atol = 1e-8
end