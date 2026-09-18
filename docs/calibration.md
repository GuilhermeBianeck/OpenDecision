# Calibration and abstention

Every backend returns raw scores. The runtime applies softmax to produce
`normalized_probabilities`. Without task-specific evidence these are relative
model scores, not a reliable frequency of correct answers.

Temperature scaling fits one positive scalar T to minimize negative log likelihood
on a held-out calibration split, then uses `softmax(scores / T)`. It changes score
sharpness, not the ranking of candidates. OpenDecision searches inverse temperature
over T in [0.05, 100]. A boundary optimum is allowed and should prompt a review of
model quality and calibration data. The implementation has no SciPy dependency.

`calibrate` scores the calibration split in batches, fits the profile, then
scores the validation split and reports calibration metrics before and after so
the held-out effect is visible. Temperature scaling never reorders candidates,
so accuracy is identical on both sides; ECE, NLL and Brier are what move.

```bash
opendecision calibrate --model tiny \
  --dataset benchmarks/datasets/core.jsonl --output calibration/tiny.json
opendecision benchmark --model tiny --calibration calibration/tiny.json \
  --dataset benchmarks/datasets/core.jsonl --split validation \
  --output local-reports/tiny-validation
```

The CLI accepts only labeled objective / agent-control rows in the `calibration`
split. It rejects an existing profile during fitting. The low-level Python
`fit_temperature` function accepts scores and indexes, so its caller must enforce
the same split discipline. Profiles store the data hash, sample count, task family,
model and checkpoint revision, serialization template, precision, maximum sequence
length, timestamp, and before/after calibration NLL. Precision resolves with the
device, so the repository ships `<name>.json` fitted at the GPU default and
`<name>-float32.json` at the CPU default; loading the wrong one raises a
`CalibrationError` naming `precision` rather than silently rescaling. The MLX
backends are the exception: their weights are quantised for Apple silicon and
have no CPU counterpart, so they ship a single profile and no `-float32`
variant.

### A profile is not always worth loading

Temperature scaling is fitted to minimize NLL on the calibration split, and a
backend that is already well calibrated has nothing to gain. Measured on the
held-out validation split:

| Backend | Uncalibrated ECE | After a fitted profile | Shipped? |
| --- | ---: | ---: | --- |
| `qwen35_4b` | 0.195 | 0.073 | yes |
| `lfm25` | 0.210 | 0.163 | yes |
| `qwen35` | **0.061** | 0.115 | **no** |

`qwen35` arrives better calibrated than any other backend and a fitted profile
makes it worse on ECE, NLL and Brier at once, so none is committed for it. Fit
one against your own data if your distribution differs, and read the held-out
report the `calibrate` command prints before adopting it. A profile that
regresses is a result, not a failure to fit. Incompatible model identities
are rejected. The same identity can still encounter a different task distribution.

## One temperature per candidate count

How sharp a scorer is depends strongly on how many candidates a request has.
The same checkpoint can be overconfident on yes/no rows and underconfident when
its mass is spread across twelve options, so a single pooled temperature is a
compromise that fits neither. `fit_temperature` therefore fits one temperature
per candidate count by default, for every count with at least
`min_rows_per_count` (25) calibration rows; the pooled temperature stays in
`temperature` as the fallback for counts that were not fitted. Pass
`per_choice_count=False` (or `opendecision calibrate --pooled`) for the single
pooled value.

`temperature_for(n)` returns the temperature a request with `n` candidates
receives, and each result reports it as `metadata.calibration_temperature`.
`covers(n)` is exact for a per-count profile and reports range membership for a
pooled one (`null` for profiles that predate the fields). A profile also records
`choice_count_min` and `choice_count_max`.

Fitting per count does not make a profile transfer across task distributions.
Fit separate profiles per task family when the decisions differ in kind, not
only in width.

### A per-count profile breaks a single margin threshold

Scaling each candidate count by its own temperature makes the distributions
better calibrated within a count and **not comparable between counts**. A margin
of 0.5 after a temperature of 9.55 is far stronger evidence than the same margin
after 0.39, so one global `margin_threshold` silently selects whichever counts
were sharpened. Measured for `base` on the test split:

| | k=2 clears margin ≥ 0.5 | k=4 | k=12 | Pooled accuracy when answered |
| --- | ---: | ---: | ---: | ---: |
| uncalibrated | 73 % at 0.863 accuracy | 13 % at 0.429 | 0 % | 0.830 |
| calibrated | 0 % | 43 % at 0.602 | 8 % at 0.417 | 0.586 |

Nothing got worse inside a count; the answered set changed composition, from
mostly high-accuracy binary rows to mostly four-way rows. Choose
`abstain_threshold` and `margin_threshold` **per candidate count** when a
per-count profile is loaded, and read `objective_by_choice_count` in a benchmark
report rather than the pooled coverage table. `--pooled` keeps one temperature
and one comparable threshold, at the cost of within-count calibration.

### Validate on your own distribution

A profile fitted and validated on one split can still regress on another whose
candidate-count mix differs. The committed corpus is 56 / 28 / 16 percent
(k=2 / k=4 / k=12) in the calibration split, 64 / 14 / 21 in validation and
42 / 39 / 19 in test, and `base` improved on validation while regressing on
test. Temperature scaling is also fitted to minimize NLL, which can move ECE the
other way: on validation `multilingual` improved NLL from 1.136 to 1.007 while
its ECE rose from 0.164 to 0.217, because at twelve candidates it is already
underconfident (accuracy 0.718 against mean confidence 0.594) and further
smoothing is exactly the wrong correction. Check the held-out report the
`calibrate` command prints, on data that resembles your workload, before
adopting a profile.

`probabilities` exposes the effective distribution. `calibrated_probabilities`
is null without a profile; `normalized_probabilities` always retains the original
softmax. For the effective distribution:

- `top_probability` is the highest candidate probability.
- `confidence` is the highest minus second-highest probability.
- `abstain_threshold` is the minimum top probability.
- `margin_threshold` is the minimum margin.
- Either failed threshold causes `choice=null` and `abstained=true`.

Thresholds use `<`: a value exactly equal to its threshold passes. Booleans expose
the yes probability even when abstained; their `value` then becomes null. Multi-label
questions receive independent binary distributions. A ranking is still returned
when its top-choice decision abstains.

A rubric `score` is a decision over its level descriptions. Its `decision`
carries the same normalized and calibrated distributions, `confidence`, and
abstention behaviour as any choice; `score` re-keys that distribution by level
index and reports its expectation. Calibrating the underlying choice therefore
calibrates the level distribution, but not the numeric magnitude of `score`,
which depends on how the rubric was written.

With statement scoring (`method: "statement"`), a boolean's yes/no distribution
is the softmax of the entailment and contradiction logits, so calibration and
the two thresholds above apply to it unchanged. `unsupported` is the neutral
share of the three-way softmax and is not part of that distribution; it is an
absolute signal that the state neither supports nor contradicts the statement.
`unsupported_threshold` abstains when `unsupported` exceeds it (equality passes),
independently of how decisive the yes/no margin looks. Backends without statement
scoring reject `unsupported_threshold` rather than ignoring it.

Choose thresholds on validation according to the consequences of error and
abstention. Report coverage and accuracy on answered objective examples together;
perfect accuracy at tiny coverage can be operationally useless. Moral dilemmas
without a specified policy have no accuracy target. Calibration on simple support
routing is not evidence of calibrated moral or medical judgment.

The benchmark JSON includes NLL, Brier score, ECE, and reliability bins. ECE uses
top-choice correctness and fixed probability bins. It depends on the sample and
binning and should not be used alone. Platt scaling, isotonic regression, and
out-of-distribution guarantees are outside this alpha.
