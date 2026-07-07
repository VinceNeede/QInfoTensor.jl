"""
    _svd_truncation_strategy(maxdim, cutoff) -> TruncationStrategy

Build a `truncrank(maxdim) & truncerror(rtol=sqrt(cutoff))` strategy for
SVD-based truncation (`tsvd`, used by `:zipup`), where `values` are
singular values `σᵢ`.

`rtol=sqrt(cutoff)`, NOT `cutoff` directly: `truncerror`'s default `p=2`
bounds `Σ(discarded σᵢ²)/Σ(all σᵢ²) < rtol²`. Since `Σσᵢ²` is exactly the
quantity ITensor's own `cutoff` convention bounds (relative discarded
squared-weight), matching that convention exactly requires `rtol² =
cutoff`, i.e. `rtol = sqrt(cutoff)`. Confirmed via MatrixAlgebraKit
source (`src/implementations/truncation.jl` + `src/interface/
truncation.jl`) and cross-checked against ITensor's own documented
`cutoff` definition.

Either `maxdim`/`cutoff` may be `nothing`, in which case that criterion
is omitted entirely (not passed as `nothing` to `truncrank`/`truncerror`,
which don't accept it) rather than going through `TruncationStrategy(;
kwargs...)`'s own nothing-skipping constructor — that convenience path
was deliberately avoided: its bare `rtol`/`atol` kwargs route to
`trunctol`'s PER-VALUE criterion, not `truncerror`'s cumulative one, and
only its `maxerror` kwarg reaches `truncerror` — as `atol`, not `rtol`.
Neither matches what `cutoff` is documented to mean here, so
`truncrank`/`truncerror` are built directly and combined via `&`
(`TruncationIntersection`) instead.
"""
function _svd_truncation_strategy(maxdim::Union{Int,Nothing}, cutoff::Union{Real,Nothing})
    strategy = notrunc()
    isnothing(maxdim) || (strategy &= truncrank(maxdim))
    isnothing(cutoff) || (strategy &= truncerror(; rtol=sqrt(cutoff)))
    return strategy
end

"""
    _eig_truncation_strategy(maxdim, cutoff) -> TruncationStrategy

Build a `truncrank(maxdim) & truncerror(rtol=cutoff, p=1)` strategy for
Hermitian-eigendecomposition-based truncation (`eigh_trunc`, used by
`:densitymatrix`), where `values` are density-matrix eigenvalues `λᵢ`.

`p=1`, NOT the default `p=2`: `λᵢ = σᵢ²` already sits on the "squared"
weight scale (`Σλᵢ = Tr(ρ) = Σσᵢ²`) — the same quantity `Σσᵢ²` that
`_svd_truncation_strategy`'s `p=2`/`rtol=sqrt(cutoff)` combination has to
manufacture from singular values via squaring. Reusing `p=2` here would
apply that squaring a SECOND time (bounding `Σλᵢ²`, i.e. `Σσᵢ⁴` — not
`cutoff`'s intended meaning), which is what an earlier version of this
function did, discarding far more of the spectrum than `cutoff`
requested; confirmed via `_truncerr_impl`'s source (fixed `by =
abs(v)^p`), not by inference. With `p=1`, `rtol=cutoff` bounds
`Σ(discarded λᵢ)/Σ(all λᵢ) < cutoff` directly — no transform needed,
since `λᵢ` is already the right scale. Empirically confirmed: at
`maxdim=320`, a real (non-`nothing`) `cutoff`, this now reproduces
`:zipup`'s achieved bond dimension exactly (320), where the `p=2`
version capped at 64 and a naive `rtol=cutoff²` guess landed at 289 —
see chat.

Same `nothing`-handling and `TruncationStrategy(;kwargs...)`-avoidance
rationale as `_svd_truncation_strategy` — see its docstring.
"""
function _eig_truncation_strategy(maxdim::Union{Int,Nothing}, cutoff::Union{Real,Nothing})
    strategy = notrunc()
    isnothing(maxdim) || (strategy &= truncrank(maxdim))
    isnothing(cutoff) || (strategy &= truncerror(; rtol=cutoff, p=1))
    return strategy
end