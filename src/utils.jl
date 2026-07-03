# cutoff <-> TruncationByError mapping (confirmed via MatrixAlgebraKit
# source, src/implementations/truncation.jl + src/interface/truncation.jl,
# AND cross-checked directly against ITensor's own documented cutoff
# definition, sum(discarded λ²)/sum(all λ²) < ε, relative by default):
# with p=2 (default), truncerror(rtol=r) discards the smallest singular
# values whose cumulative Σσᵢ² stays ≤ r² * Σ(all σᵢ²) — i.e. r² is
# EXACTLY the relative discarded-squared-weight fraction, matching
# ITensor's `cutoff` exactly. So our `cutoff` maps as rtol = sqrt(cutoff).
# Going through the trunc::NamedTuple interface was deliberately avoided:
# its `maxerror` key maps to `truncerror(atol=maxerror)` — ABSOLUTE, not
# relative — confirmed from TruncationStrategy(...)'s source, so it does
# not give us what `cutoff` is documented to mean. `truncrank`/
# `truncerror` are built directly and combined via `&` (TruncationIntersection).
# ------------------------------------------------------------------------

# Bypasses TruncationStrategy(; kwargs...) entirely (its bare rtol/atol
# route to trunctol's PER-VALUE criterion, not truncerror's cumulative
# one — confirmed from source; only ITS `maxerror` reaches truncerror,
# and even then as atol). truncrank/truncerror don't accept `nothing`
# themselves, so this replicates just the nothing-skipping/`&`-combining
# behavior TruncationStrategy's own constructor uses internally.
function _truncation_strategy(maxdim::Union{Int,Nothing}, cutoff::Union{Real,Nothing})
    strategy = notrunc()
    isnothing(maxdim) || (strategy &= truncrank(maxdim))
    isnothing(cutoff) || (strategy &= truncerror(; rtol=sqrt(cutoff)))
    return strategy
end

