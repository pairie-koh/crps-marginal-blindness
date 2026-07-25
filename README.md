# A sum of marginal CRPS is blind to path dependence

A small, self-contained, reproducible demonstration of a standard fact about
proper scoring rules — with a worked example in the exact scoring shape used by
ensemble price-path forecasting competitions.

## The point

Many forecasting competitions score an ensemble of predicted paths by summing a
**univariate CRPS** over every scored time interval (the 5-minute return, the
30-minute return, and so on) and adding the results into one number.

CRPS is *strictly proper* for each univariate marginal. But a **sum** of
marginal CRPS values is **not proper for the joint distribution** of the path
cloud. The total depends only on the per-interval marginals and is invariant to
the *copula* — how values are coupled across time within a path. Volatility
clustering, jump-time ordering, sample-path continuity and cross-horizon
dependence are all copula properties, so the score cannot see them.

Concretely: two ensembles with identical per-interval marginals receive an
identical score, no matter how differently their individual paths behave.

## The demonstration

`summed_marginal_crps_is_blind_to_dependence.py` builds two 1000-path ensembles:

- **A** — paths with genuine volatility clustering (a GARCH(1,1) process).
- **B** — ensemble A with each time-column independently permuted across paths.
  This preserves every column's marginal *exactly* while destroying the
  within-path temporal dependence.

```
Per-column marginals identical between A and B: True

summed per-column CRPS, A (clustered): 10.7420746563
summed per-column CRPS, B (scrambled): 10.7420746563
absolute score difference            : 0.00e+00

volatility clustering, A (|ret| lag-1 autocorr): +0.0418
volatility clustering, B (|ret| lag-1 autocorr): -0.0214
```

Identical marginals → identical summed-marginal score, to floating point, while
the joint dependence differs materially (opposite-signed clustering). A
realistic ensemble and a dependence-scrambled one are indistinguishable to the
score.

## Why this matters

A scoring rule determines what forecasters build. If the rule is blind to path
realism, nothing in the objective pushes forecasters toward it — so the joint
structure drifts wherever the marginal fit happens to leave it, and the
best-*scoring* model and the best-*forecasting* model become different objects.

The mechanism is indifference rather than perverse reward, and the distinction
matters. A sum of marginal CRPS is strictly proper *for the marginals*, so it
does not pay to misstate them. It simply assigns **zero** weight to the
dependence structure, as the demonstration above shows exactly. Unrealistic path
structure is therefore not rewarded — it is unpriced, which is enough for a
field of optimizers to drift away from realism at no cost.

## The fix

`energy_score_sees_dependence.py` scores the same A and B with the **energy
score**, a strictly proper scoring rule for multivariate distributions:

    ES(P, y) = E_P || X - y ||  -  (1/2) E_P || X - X' ||

Because the norm couples the whole path at once, the energy score depends on the
joint law. Against a clustered world it rewards the clustered ensemble A over
the scrambled ensemble B, while the summed-marginal CRPS stays indifferent:

```
Summed-marginal CRPS (lower = better) -- BLIND to the difference:
    B - A: +0.000000   (indistinguishable up to MC noise)

Energy score (lower = better) -- SEES the difference:
    B - A: +0.005835   (B is scored strictly worse)
```

The **variogram score** (Scheuerer & Hamill, 2015) is a robust alternative
that targets pairwise dependence directly. Adding a multivariate proper scoring
term — even at a small weight — is the standard remedy for the joint-blindness
of a sum of marginal CRPS.

## Run it

```bash
pip install -r requirements.txt
python summed_marginal_crps_is_blind_to_dependence.py
python energy_score_sees_dependence.py
```

Pure NumPy, runs in seconds, fixed seeds so the printed numbers reproduce
exactly. No data files, no network.

## References

- Gneiting, T. & Raftery, A. E. (2007), *Strictly Proper Scoring Rules,
  Prediction, and Estimation*, Journal of the American Statistical Association,
  102(477), 359–378 — CRPS, the energy score, and propriety.
- Scheuerer, M. & Hamill, T. M. (2015), *Variogram-Based Proper Scoring Rules
  for Probabilistic Forecasts of Multivariate Quantities*, Monthly Weather
  Review, 143(4), 1321–1334, doi:10.1175/MWR-D-14-00269.1.

## License

MIT. See `LICENSE`.
