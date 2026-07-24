"""
A sum of per-marginal CRPS is blind to the joint dependence structure.

This is a self-contained, reproducible demonstration of a standard fact about
proper scoring rules, using the exact scoring shape of an ensemble forecasting
competition: the total score is a sum of univariate CRPS values, one per scored
time-gridpoint functional (e.g. the return over each interval), summed across
all intervals.

CRPS is strictly proper for each univariate marginal. But a *sum* of marginal
CRPS values is NOT proper for the joint distribution of the path cloud: the
total depends only on the per-column marginals and is invariant to the copula
(how values are coupled across time within a path).

We show this by constructing two 1000-path ensembles:

  A : paths with genuine volatility clustering (a GARCH(1,1) process).
  B : ensemble A with each time-column independently permuted across paths.
      This preserves every column's marginal EXACTLY (each column holds the
      same 1000 values, reassigned to different paths) while destroying the
      within-path temporal dependence.

Because every marginal is identical, the summed CRPS is identical to floating
point. Yet a genuine joint property -- the autocorrelation of absolute returns,
i.e. volatility clustering -- is materially different. The score cannot tell a
realistic path ensemble apart from a dependence-scrambled one.

Run:
    python summed_marginal_crps_is_blind_to_dependence.py

Requires: numpy.
"""
import numpy as np


def crps_ensemble_column(x: np.ndarray, y: float) -> float:
    """Energy-form (NRG) ensemble CRPS for one column of M values vs a scalar y:

        CRPS = (1/M) sum_i |x_i - y|  -  (1/(2 M^2)) sum_{i,j} |x_i - x_j|

    The pairwise dispersion term is computed in O(M log M) via the sorted
    identity sum_{i,j}|x_i - x_j| = 2 sum_k (2k - 1 - M) * x_(k).
    This is a symmetric function of the ensemble members: it depends on the
    column only through its multiset of values, not through their ordering
    across paths. That symmetry is the entire reason the total is blind to
    the copula.
    """
    m = x.shape[0]
    mean_abs_error = np.abs(x - y).mean()
    xs = np.sort(x)
    k = np.arange(1, m + 1)
    mean_pairwise = (2.0 * np.sum((2 * k - 1 - m) * xs)) / (m * m)
    return mean_abs_error - 0.5 * mean_pairwise


def summed_marginal_crps(returns: np.ndarray, outcome: np.ndarray) -> float:
    """Total score = sum of per-column CRPS, matching a competition scorer that
    sums univariate CRPS across every scored interval."""
    return sum(
        crps_ensemble_column(returns[:, t], outcome[t])
        for t in range(returns.shape[1])
    )


def volatility_clustering(returns: np.ndarray) -> float:
    """Mean within-path lag-1 autocorrelation of absolute returns.
    A pure joint/dependence functional: it depends on how consecutive returns
    are coupled within a path, and has no univariate-marginal footprint."""
    a = np.abs(returns)
    a = a - a.mean(axis=1, keepdims=True)
    num = (a[:, :-1] * a[:, 1:]).sum(axis=1)
    den = (a * a).sum(axis=1)
    return float(np.mean(num / den))


def build_clustered_ensemble(m: int, t: int, rng: np.random.Generator) -> np.ndarray:
    """M paths of T returns from a GARCH(1,1) process (volatility clusters)."""
    omega, alpha, beta = 0.02, 0.12, 0.86
    ret = np.empty((m, t))
    sig2 = np.full(m, omega / (1.0 - alpha - beta))
    for step in range(t):
        r = np.sqrt(sig2) * rng.standard_normal(m)
        ret[:, step] = r
        sig2 = omega + alpha * r**2 + beta * sig2
    return ret


def scramble_dependence(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently permute each column across paths.
    Preserves every column's marginal exactly; destroys temporal dependence."""
    out = np.empty_like(returns)
    for t in range(returns.shape[1]):
        out[:, t] = returns[rng.permutation(returns.shape[0]), t]
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    M, T = 1000, 48

    A = build_clustered_ensemble(M, T, rng)
    B = scramble_dependence(A, rng)

    # One fixed realized outcome to score both ensembles against.
    outcome = rng.standard_normal(T) * 0.15

    score_a = summed_marginal_crps(A, outcome)
    score_b = summed_marginal_crps(B, outcome)
    marginals_identical = all(
        np.allclose(np.sort(A[:, t]), np.sort(B[:, t])) for t in range(T)
    )

    print("Per-column marginals identical between A and B:", marginals_identical)
    print()
    print(f"summed per-column CRPS, A (clustered): {score_a:.10f}")
    print(f"summed per-column CRPS, B (scrambled): {score_b:.10f}")
    print(f"absolute score difference            : {abs(score_a - score_b):.2e}")
    print()
    print(f"volatility clustering, A (|ret| lag-1 autocorr): {volatility_clustering(A):+.4f}")
    print(f"volatility clustering, B (|ret| lag-1 autocorr): {volatility_clustering(B):+.4f}")
    print()
    print("Identical marginals -> identical summed-marginal score, while the")
    print("joint dependence (volatility clustering) differs materially. A sum of")
    print("marginal CRPS is proper for the marginals but not for the joint law.")

    assert marginals_identical
    assert abs(score_a - score_b) < 1e-9


if __name__ == "__main__":
    main()
