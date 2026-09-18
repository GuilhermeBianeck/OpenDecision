# Calibration and abstention

Every backend returns raw scores. The runtime applies softmax to produce
`normalized_probabilities`. Without task-specific evidence these are relative
model scores, not a reliable frequency of correct answers.

Temperature scaling fits one positive scalar T to minimize negative log likelihood
on a held-out calibration split, then uses `softmax(scores / T)`. It changes score
sharpness, not the ranking of candidates. OpenDecision searches inverse temperature
over T in [0.05, 100]. A boundary optimum is allowed and should prompt a review of
model quality and calibration data. The implementation has no SciPy dependency.

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
length, timestamp, and before/after calibration NLL. Incompatible model identities
are rejected. The same identity can still encounter a different task distribution.

A profile also records the smallest and largest number of candidates it was
fitted on (`choice_count_min`, `choice_count_max`). One temperature is shared
by every choice count, but it was only observed on that range: a temperature
fitted on yes/no rows says nothing about a ten-way decision. Each result reports
`metadata.calibration_covers_choice_count` (`true`, `false`, or `null` when the
profile predates this field) so callers can treat out-of-range decisions as
uncalibrated. Fit separate profiles per task family when candidate counts differ.

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

Choose thresholds on validation according to the consequences of error and
abstention. Report coverage and accuracy on answered objective examples together;
perfect accuracy at tiny coverage can be operationally useless. Moral dilemmas
without a specified policy have no accuracy target. Calibration on simple support
routing is not evidence of calibrated moral or medical judgment.

The benchmark JSON includes NLL, Brier score, ECE, and reliability bins. ECE uses
top-choice correctness and fixed probability bins. It depends on the sample and
binning and should not be used alone. Platt scaling, isotonic regression, and
out-of-distribution guarantees are outside this alpha.
