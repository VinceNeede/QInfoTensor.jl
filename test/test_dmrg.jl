# usings (Test, TensorKit, QInfoTensor) live in runtests.jl.
#
# This file additionally needs LinearAlgebra (kron/eigvals/Hermitian, for
# the exact-diagonalization reference below) and Random (fixed seed) —
# neither is a dependency of the package itself (see design_notes' "No
# LinearAlgebra dependency" note), so both need to be test-only deps in
# test/Project.toml, not the main Project.toml.
#
# Unlike the random-state checks elsewhere in this test suite (which hold
# for ANY random draw — isometry properties, bond-dimension caps, etc.),
# DMRG's energy convergence from a random initial state genuinely can
# depend on the draw, especially the nsite=1 noise tests starting from
# bond dimension 2 within only 10 sweeps — so this file fixes the seed,
# unlike its siblings.
using LinearAlgebra
using Random

Random.seed!(1234)

# ------------------------------------------------------------------------
# Test-only helpers — NOT part of the package.
# ------------------------------------------------------------------------

# TFIM Hamiltonian: H = -J Σ Sz_i Sz_{i+1} - h Σ Sx_i
function _tfim(L::Int, J::Float64, h::Float64)
    sites = sitetypes(:SpinHalf, L)
    H = OpSum()
    for i in 1:(L-1)
        H += (-J, :Sz, i, :Sz, i + 1)
    end
    for i in 1:L
        H += (-h, :Sx, i)
    end
    return sites, MPO(H, sites)
end

# Independent ED reference — plain dense Kronecker construction, no
# QInfoTensor.jl involved, so it can't share a bug with the code under test.
function _ed_tfim_energy(L::Int, J::Float64, h::Float64)
    Id2 = Matrix{Float64}(I, 2, 2)
    Sz = [1.0 0.0; 0.0 -1.0] ./ 2
    Sx = [0.0 1.0; 1.0 0.0] ./ 2
    embed(op, i, L) = foldl(kron, [k == i ? op : Id2 for k in 1:L])
    Hdense = zeros(Float64, 2^L, 2^L)
    for i in 1:(L-1)
        Hdense .+= -J .* embed(Sz, i, L) * embed(Sz, i + 1, L)
    end
    for i in 1:L
        Hdense .+= -h .* embed(Sx, i, L)
    end
    return minimum(eigvals(Hermitian(Hdense)))
end

const DMRG_MAXDIM_SCHEDULE = [10, 10, 20, 20, 40, 40, 40, 40, 40, 40]
const DMRG3S_NOISE_SCHEDULE = [1e-2, 1e-2, 1e-3, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5, 0.0, 0.0]
const DMRG3S_TOL_SCHEDULE = [1e-4, 1e-4, 1e-6, 1e-6, 1e-8, 1e-8, 1e-10, 1e-10, 1e-12, 1e-12]

# ------------------------------------------------------------------------

@testset "dmrg!: :twosite matches exact diagonalization" begin
    L, J, h = 6, 1.0, 0.5
    sites, H = _tfim(L, J, h)
    E_ed = _ed_tfim_energy(L, J, h)

    ψ = random_mps(sites, 2)
    ψ, _, sd = dmrg!(ψ, H, 10; nsite=2, maxdim=DMRG_MAXDIM_SCHEDULE, cutoff=1e-10)
    E = sd[end].energies[end]
    @test E ≈ E_ed atol=1e-6
end

@testset "dmrg!: nsite=2 has no noise kwarg — rejects it structurally, doesn't silently ignore it" begin
    # ProjMPO{T,2}'s dmrg! method has no `noise` keyword at all (and no
    # `kwargs...` catch-all), unlike ProjMPO{T,1} — subspace expansion is
    # only meaningful for the single-site algorithm. Passing `noise` with
    # nsite=2 should fail at dispatch (MethodError), not get silently
    # dropped, same "no fallback, must MethodError" bar test_sitetype.jl
    # applies to illegal SiteType/op combinations.
    L, J, h = 6, 1.0, 0.5
    sites, H = _tfim(L, J, h)
    ψ = random_mps(sites, 2)
    @test_throws MethodError dmrg!(ψ, H, 2; nsite=2, noise=0.1)
end

@testset "dmrg!: :dmrg3s (nsite=1), staged by risk" begin
    # :dmrg3s is much newer than :twosite, and its backward-sweep branch
    # depends on the still-unverified right-moving environment
    # (_seed_env/_extend_env Val(:right)). Staged so a failure points at
    # roughly where the bug is instead of "somewhere in dmrg3s".
    L, J, h = 6, 1.0, 0.5
    sites, H = _tfim(L, J, h)
    E_ed = _ed_tfim_energy(L, J, h)

    @testset "no noise: plain eigensolve/SVD/write-back, both sweep directions" begin
        # Generous fixed bond dimension, since single-site DMRG can't grow
        # it without noise — exercises BOTH forward and backward branches
        # without touching _dmrg3s_noise/_dmrg3s_perturbation or the
        # right-environment seed/extend machinery at all.
        ψ = random_mps(sites, 16)
        ψ, _, sd = dmrg!(ψ, H, 6; nsite=1, noise=nothing,
                          eigsolve_tol=[1e-6, 1e-6, 1e-8, 1e-8, 1e-10, 1e-10])
        E = sd[end].energies[end]
        @test E ≈ E_ed atol=1e-6
    end

    @testset "noise, forward-first: EL-based (Val(:left)) expansion" begin
        ψ = random_mps(sites, 2)
        ψ, _, sd = dmrg!(ψ, H, 10; nsite=1, noise=DMRG3S_NOISE_SCHEDULE,
                          start_forward=true, eigsolve_tol=DMRG3S_TOL_SCHEDULE)
        E = sd[end].energies[end]
        @test maxlinkdim(ψ) > 2  # subspace expansion actually grew the bond
        @test E ≈ E_ed atol=1e-6
    end

    @testset "noise, backward-first: ER-based (Val(:right)) expansion" begin
        # Highest-risk stage: forces the FIRST sweep backward, so
        # _dmrg3s_noise(...,Val(:right)) / _seed_env(Val(:right)) actually run.
        ψ = random_mps(sites, 2)
        ψ, _, sd = dmrg!(ψ, H, 10; nsite=1, noise=DMRG3S_NOISE_SCHEDULE,
                          start_forward=false, eigsolve_tol=DMRG3S_TOL_SCHEDULE)
        E = sd[end].energies[end]
        @test maxlinkdim(ψ) > 2
        @test E ≈ E_ed atol=1e-6
    end
end

@testset "dmrg!: :dmrg3s vs :twosite cross-check, same Hamiltonian" begin
    L, J, h = 6, 1.0, 0.5
    sites, H = _tfim(L, J, h)

    ψ_twosite = random_mps(sites, 2)
    _, _, sd_twosite = dmrg!(ψ_twosite, H, 10; nsite=2, maxdim=DMRG_MAXDIM_SCHEDULE, cutoff=1e-10)
    E_twosite = sd_twosite[end].energies[end]

    ψ_dmrg3s = random_mps(sites, 2)
    _, _, sd_dmrg3s = dmrg!(ψ_dmrg3s, H, 10; nsite=1, noise=DMRG3S_NOISE_SCHEDULE,
                            start_forward=true, eigsolve_tol=DMRG3S_TOL_SCHEDULE)
    E_dmrg3s = sd_dmrg3s[end].energies[end]

    @test E_twosite ≈ E_dmrg3s atol=1e-6
end

@testset "heff is Hermitian when the right-moving environment is genuinely exercised" begin
    # Closes a real gap, not just adding coverage: the right-moving
    # environment (_seed_env/_extend_env, Val(:right)) was previously only
    # INDIRECTLY confirmed via correct end-to-end dmrg3s energies — never
    # given a direct check the way _densitymatrix_envs's own E was
    # (‖E-E'‖/‖E‖). A literal E≈E' doesn't even type-check here, though:
    # EnvTensor is (1,2)-shaped (codomain=(bra,), domain=(mpo,ket)), not
    # square like _densitymatrix_envs's (2,2)-shaped E, so E' has a
    # DIFFERENT shape than E, not just swapped bra/ket. The actual
    # invariant that matters — and the one eigsolve's `ishermitian=true`
    # is implicitly relying on — is the bilinear-form condition
    # ⟨x|heff(y)⟩ = conj(⟨y|heff(x)⟩) for arbitrary x,y in the same space
    # as the local tensor, checked directly via `inner`, no assumption
    # about EnvTensor's own shape needed.
    L = 5
    sites = sitetypes(:SpinHalf, L)
    H_os = OpSum()
    for i in 1:(L-1)
        H_os += (-1.0, :Sz, i, :Sz, i + 1)
    end
    for i in 1:L
        H_os += (-0.5, :Sx, i)
    end
    H = MPO(H_os, sites)
    ψ = random_mps(sites, 4)
    orthogonalize!(ψ, 1)

    P = QInfoTensor.ProjMPO(H, 1)
    pos = 2  # NOT a boundary site: forces position! to chain multiple real
    # _extend_env(...,Val(:right)) calls (seed -> real -> real) to build
    # P.env[P.rpos], not just a single seed-based extend — the same "many
    # invocations, not just one" exercise verify_dmrg3s.jl's energy
    # agreement already implied indirectly, now checked directly.
    QInfoTensor.position!(P, ψ, pos)

    Vc, Vd = codomain(ψ[pos]), domain(ψ[pos])
    x = randn(ComplexF64, Vc ← Vd)
    y = randn(ComplexF64, Vc ← Vd)

    Hx = QInfoTensor.heff(P, pos, ψ, x)
    Hy = QInfoTensor.heff(P, pos, ψ, y)

    lhs = inner(x, Hy)
    rhs = conj(inner(y, Hx))
    @test lhs ≈ rhs atol=1e-10
end