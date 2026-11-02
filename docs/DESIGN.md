# persuadables: design

## Goals

1. A trustworthy readout of a randomized email experiment: is the randomization sound, what did
   each email do, how precise is that, and does the effect differ across segments fixed in advance.
2. Uplift models that rank customers by how much an email changes their behaviour, validated on
   data with a known answer before they are trusted on real data.
3. A decision: what a budget-constrained targeting policy is worth, with an interval, compared with
   the policies a team would otherwise use.
4. Reproducibility: one config file, seeded steps, deterministic outputs, a test for each claim.

## Non-goals

- Observational causal inference. Every estimator assumes randomized assignment with known
  probabilities; none of them adjusts for confounding.
- Real-time serving. Models are refit by the pipeline; predictions are the artifact, not pickled
  models.
- Hyperparameter search. One regularized XGBoost configuration is shared by all base models (see
  "Alternatives").

## Data model

The raw CSV has one row per customer. Four SQL migrations in `src/persuadables/data/sql/` build:

| Table | Built by | Contents |
|---|---|---|
| `raw` | V001 | the CSV with an explicit schema (`auto_detect = false`: unparseable values fail) |
| `customers` | V002 | `customer_id` (1-based file row), integer `arm` (0 none, 1 Mens, 2 Womens), cleaned levels |
| `features` | V003 | model inputs (all pre-period) and six pre-registered segment columns |
| `experiment` | V004 | `features` plus `is_test`, a seeded stratified split |
| `schema_migrations` | runner | version, name and sha256 of every applied SQL file |

The split deals rows into ten folds within each (arm, visit, conversion) stratum, ordered by
`md5(customer_id || ':' || seed)`; folds 0-2 are the test set. md5 (not DuckDB's `hash`) keeps the
split identical across DuckDB versions.

## Invariants (each enforced by validation or a test)

- 64,000 rows, unique ids, no NULL in any column, every column of the declared type.
- `arm` in {0, 1, 2}; recency 1-12; history > 0; spend >= 0; flags are 0/1; categorical levels from
  a closed list; exactly one zip and one channel indicator per row.
- `conversion <= visit` and `spend > 0` if and only if `conversion = 1`.
- Test share within 1 point of 30% in every arm.
- No sample-ratio mismatch (p >= 0.001) and every |SMD| < 0.1.
- Rerunning any stage with the same config reproduces `results/*.json` byte for byte.
- A policy's action probabilities form a distribution per customer; top-k emails exactly
  round(k * n) customers.

## Assumptions

1. **Randomization.** Assignment to the three arms was random with equal probability, independent
   of every covariate and potential outcome. Checked, not assumed blindly: SRM and balance.
2. **SUTVA / no interference.** One customer's email does not change another's outcome, and "the
   Mens email" is one well-defined treatment.
3. **Pre-period covariates.** Recency, history, the merchandise flags, zip area, newbie status and
   channel describe the 12 months before the campaign, so the email cannot have changed them. This
   is what makes them valid for CUPED, balance checks, segments and model inputs.
4. **Known propensities.** For off-policy evaluation the logging policy is the design (1/3 per arm).
   SNIPW additionally absorbs small realised deviations from 1/3.
5. **Transportability.** The policy values describe this customer base in this two-week window. A
   new campaign, season or list would need a new experiment to confirm them.

## Threats to validity

- **Proxy outcome.** Both ranked policies rank on visits (pre-registered for density): the uplift
  policy by predicted visit uplift, the response baseline by P(visit | best single email). They
  tie on visits and differ on conversions and revenue, where the response baseline wins at 30% and
  50%. That compares two visit rankings; it does not show what a conversion- or revenue-ranked
  policy (uplift or response) would earn, which would be a new, separately reported analysis.
  Visits bought by the Womens email convert less, so a visit-optimal email choice is not
  conversion-optimal either.
- **Small test split for conversions.** 38, 81 and 57 conversions (no email, Mens, Womens) in the
  test split put wide intervals on every conversion and revenue value per 1,000 emails.
- **Multiple comparisons.** Holm within each family (6 effects; 36 heterogeneity tests). Policy
  comparisons are reported with intervals, not as a family of tests; the README calls out which
  intervals exclude zero and does not claim more.
- **Small budgets are noisy.** Per-1,000-email values divide by the budget, so values at 1-5% are
  dominated by noise; figures start at 5%.
- **Selection on the training split.** Learner selection by CV Qini uses only the training split,
  but CV Qini on rare outcomes is itself noisy (fold SDs comparable to the means for conversion).
- **Spend outliers.** Spend is capped at $499 in the source data, which bounds the influence of any
  single buyer on the mean.

## Why each test

| Question | Method | Why this one |
|---|---|---|
| Is the split broken? | Pearson chi-square on arm counts (SRM) | The cheapest, most decisive check; a failure invalidates everything else |
| Are the arms comparable? | Standardized mean differences | Scale-free and independent of n, unlike a t-test on covariates |
| Effect on a rate | Two-proportion z-test (pooled SE for the test, unpooled for the CI) | n ~ 21,000 per arm; standard and matches statsmodels exactly |
| Effect on spend | Welch's t-test + stratified percentile bootstrap | Unequal variances; 99% zeros make the bootstrap the check on the normal approximation |
| More precision | CUPED (multivariate, pooled theta) | Unbiased under randomization, equals regression adjustment; the achieved reduction is reported |
| Many tests | Holm step-down | Controls family-wise error under any dependence; uniformly better than Bonferroni |
| Could we have seen it? | MDE at 80% power from the realised arm sizes; power for, and arm size needed for, a fixed +10% planning lift | Post-hoc "observed power" is a function of the p-value; the MDE and power at a lift fixed in advance are the informative quantities |
| Does the effect differ by segment? | Cochran's Q over independent level estimates | Levels are disjoint customers, so the estimates are independent and Q ~ chi2(k-1) |
| Does the model rank persuadables? | Qini coefficient (bootstrap CI), AUUC, uplift@k | Individual effects are never observed; ranking metrics on randomized data are |
| Are predicted uplifts the right size? | Observed vs predicted by decile, WLS slope | A model can rank well and still be miscalibrated, which matters once uplift meets a margin |
| What is a policy worth? | SNIPW with a paired Poisson bootstrap | Unbiased up to normalization on randomized logs; pairing tightens policy differences |

## Failure modes and how they surface

- **The source file changes or the download is truncated:** sha256 mismatch, the file is
  discarded, `DataIntegrityError` names both hashes.
- **A value no longer parses:** V001 fails inside its transaction with DuckDB's line number.
- **A row breaks a rule:** validation names the rule, the count and example `customer_id`s, and
  the pipeline stops before any analysis.
- **A stage runs before its inputs exist:** a `FileNotFoundError` that names the command to run.
- **A rare outcome has one class in a fold:** the base model becomes a constant predictor instead
  of an XGBoost error.
- **A policy is unidentified** (no logged customer received any action it takes, so its total
  importance weight is zero): `snipw_bootstrap` raises a `ValueError` naming the policy and budget
  instead of writing a NaN (which the JSON writer would turn into `null`). A bootstrap resample in
  which that happens by chance is a NaN draw, skipped by the quantile intervals.

## Alternatives considered

- **Streamlit dashboard.** Rejected for a static Plotly page: nothing to keep running, nothing to
  stop, opens from disk, and it cannot show numbers that are not in `results/`.
- **pandas throughout.** Rejected for numpy arrays at module boundaries: strict typing stays
  meaningful and the SQL layer owns the tabular logic.
- **causalml / scikit-uplift.** Rejected as dependencies: the learners and metrics are a few hundred
  lines here, each tested against a reference or a known-effect simulation, and the metric
  definitions follow scikit-uplift so the numbers are comparable.
- **The `shap` package.** XGBoost's built-in `pred_contribs` computes the same exact TreeSHAP values
  without a numba dependency.
- **Per-learner tuning.** Tuning on CV Qini with roughly 200-400 conversions per training pair would
  mostly fit noise; a shared, regularized configuration keeps the comparison fair.

## Pre-registration record

Fixed in `configs/default.toml` and `V003__features.sql` before the readout, training or
evaluation stages were first run:
segments (purchase category, recency band, history tier, channel, zip area, tenure), CUPED
covariates, alpha 0.05, power 0.8, the ranking outcome (visit), the headline budget (30%), the
budget grid and the XGBoost configuration. No analysis setting was changed after the test split
was first scored. Later edits touched only figures, wording, policy labels, the synthetic test
data, and added reported quantities that change no estimate: the power for and arm size needed
to detect a +10% lift (`reference_relative_lift`), the share of zero spend, and the per-split,
per-arm outcome counts in `training.json`.

The response baseline has always been P(ranking outcome | best single email), i.e. P(visit | Mens
email): a visit-response model, not a purchase model. It is labelled by what it scores everywhere,
and no P(buy) baseline was added after seeing the results (it is on the roadmap as a separate
analysis).
