"""
The fix: a multivariate proper scoring rule sees the dependence the sum of
marginal CRPS is blind to.

Companion to `summed_marginal_crps_is_blind_to_dependence.py`. That script shows
ensemble A (volatility clustering) and ensemble B (A with each column permuted)
receive an identical summed-marginal CRPS despite different joint structure.

Here we score the same A and B with the ENERGY SCORE, a strictly proper scoring
rule for multivariate distributions:

    ES(P, y) = E_P || X - y ||  -  (1/2) E_P || X - X' ||

where X, X' are independent draws from the forecast P (here the empirical path
ensemble), y is the realized path vector, and ||.|| is Euclidean norm over the
whole path. Because the norm couples all time-points at once, the energy score
depends on the joint law, not just the marginals -- so it separates A from B.

To make the point sharp we let the realized outcome itself be a clustered path
(drawn from the same GARCH process as A). A good forecaster of a clustered
world should be rewarded for producing clustered paths; the energy score does
exactly that, while the summed-marginal CRPS is indifferent.

Run:
    python energy_score_sees_dependence.py

Requires: numpy.
"""
import numpy as np

from summed_marginal_crps_is_blind_to_dependence import (
    build_clustered_ensemble,
    scramble_dependence,
    summed_marginal_crps,
    volatility_clustering,
)


def energy_score(paths: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                 n_pairs: int = 4000) -> float:
    """Energy score of an empirical path ensemble against a realized path y.

    First term: mean Euclidean distance from ensemble paths to the outcome.
    Second term: half the mean Euclidean distance between independent pairs of
    ensemble paths (estimated by sampling n_pairs pairs). Lower is better.
    """
    m = paths.shape[0]
    term1 = np.linalg.norm(paths - y[None, :], axis=1).mean()
    i = rng.integers(0, m, size=n_pairs)
    j = rng.integers(0, m, size=n_pairs)
    term2 = np.linalg.norm(paths[i] - paths[j], axis=1).mean()
    return float(term1 - 0.5 * term2)


def main() -> None:
    rng = np.random.default_rng(0)
    M, T = 1000, 48

    A = build_clustered_ensemble(M, T, rng)
    B = scramble_dependence(A, rng)

    # Realized world is itself clustered: many independent clustered outcomes,
    # so both scores are averaged over outcomes for a stable comparison.
    n_out = 400
    outcomes = build_clustered_ensemble(n_out, T, rng)

    marg_a = np.mean([summed_marginal_crps(A, outcomes[k]) for k in range(n_out)])
    marg_b = np.mean([summed_marginal_crps(B, outcomes[k]) for k in range(n_out)])
    es_a = np.mean([energy_score(A, outcomes[k], rng) for k in range(n_out)])
    es_b = np.mean([energy_score(B, outcomes[k], rng) for k in range(n_out)])

    print(f"volatility clustering, A (clustered): {volatility_clustering(A):+.4f}")
    print(f"volatility clustering, B (scrambled): {volatility_clustering(B):+.4f}")
    print()
    print("Summed-marginal CRPS (lower = better) -- BLIND to the difference:")
    print(f"    A: {marg_a:.6f}")
    print(f"    B: {marg_b:.6f}")
    print(f"    B - A: {marg_b - marg_a:+.6f}   (indistinguishable up to MC noise)")
    print()
    print("Energy score (lower = better) -- SEES the difference:")
    print(f"    A: {es_a:.6f}")
    print(f"    B: {es_b:.6f}")
    print(f"    B - A: {es_b - es_a:+.6f}   (B is scored strictly worse)")
    print()
    print("Against a clustered world, the energy score rewards the clustered")
    print("ensemble A over the scrambled ensemble B; the summed-marginal CRPS")
    print("does not. That is why adding a multivariate proper scoring term is")
    print("the remedy for the joint-blindness of a sum of marginal CRPS.")

    assert es_b - es_a > 0  # energy score prefers the realistic ensemble


if __name__ == "__main__":
    main()
