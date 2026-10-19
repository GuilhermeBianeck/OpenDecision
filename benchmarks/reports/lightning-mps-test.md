# OpenDecision benchmark report

Status: **measured**. Generated 2026-10-19T09:50:17.004345+00:00.

Model `netprojects/h2o-lightning-4b-mlx-q4`; revision `2bda86b6f1ecbb5e6055f9fe8986f2a6e06849e4`; backend `lightning`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.9395509499136442 |
| balanced_accuracy | 0.9299500702137619 |
| macro_f1 | 0.9426127803644386 |
| negative_log_likelihood | 0.25906120694512996 |
| brier_score | 0.11536269432800349 |
| ece | 0.10696845122822486 |
| coverage | 1.0 |
| selective_accuracy | 0.9395509499136442 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.9560 | 0.1474 |
| agent_control | 156 | 0.9679 | 0.1791 |
| verification | 352 | 0.9716 | 0.1235 |
| robustness | 286 | 0.8636 | 0.1044 |

## Objective accuracy by candidate count

A margin threshold is only comparable within one candidate count once a
per-count calibration profile is applied. Read coverage per row, not pooled.

| Candidates | Count | Accuracy | ECE | Coverage at margin >= 0.5 | Accuracy when answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 690 | 0.9710 | 0.1141 | 0.8638 | 0.9983 |
| 4 | 312 | 0.8686 | 0.0787 | 0.8686 | 0.9004 |
| 12 | 156 | 0.9423 | 0.1570 | 0.8846 | 0.9855 |

## Separate evaluation families

Policy agreement (not moral accuracy): `1` across 44 policy-labeled cases.
Ranking NDCG: `1.0`. Ambiguous-case abstention rate: `0.25`.
Ordinal rubrics: 216 cases, exact level `0.8518518518518519`, within one level `0.9861111111111112`, mean absolute expected error `0.34596397653340366` level steps.
Verification statements: 352 cases, 352 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `2459.079958000075` ms. First batch: `2622.616` ms.
Warm batch latency: `{'p50': 114.8590000011609, 'p95': 399.80824119920726, 'p99': 439.11292700111517, 'samples': 1725}`. Amortized per decision: `{'p50': 114.8590000011609, 'p95': 399.80824119920726, 'p99': 439.11292700111517}`.
Throughput including first batch: `6.403` decisions/s.
Process lifetime peak RSS: `3285876736` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.9395509499136442 |
| 0.25 | 0.9370 | 0.9612903225806452 |
| 0.5 | 0.8679 | 0.9701492537313433 |
| 0.75 | 0.6174 | 0.9944055944055944 |
| 0.8 | 0.3877 | 0.9955456570155902 |
| 0.9 | 0.0009 | 1 |
| 0.95 | 0.0000 | None |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.8333333333333334,
      "mean_total_variation": 0.17715216217284271,
      "objective_accuracy_delta": -0.1595744680851064
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.8548387096774194,
      "mean_total_variation": 0.10875983129037096,
      "objective_accuracy_delta": -0.010638297872340425
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.9112903225806451,
      "mean_total_variation": 0.06260090564866529,
      "objective_accuracy_delta": 0.0
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.9722222222222222,
      "mean_total_variation": 0.024373406635243535,
      "objective_accuracy_delta": 0.02127659574468085
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.9305555555555556,
      "mean_total_variation": 0.0653846287006293,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.9930555555555556,
      "mean_total_variation": 0.011516407216742745,
      "objective_accuracy_delta": 0.0
    },
    "option_rotation": {
      "pairs": 94,
      "choice_stability": 0.8829787234042553,
      "mean_total_variation": 0.03954540023073267,
      "objective_accuracy_delta": -0.016129032258064516
    },
    "question_rewording": {
      "pairs": 112,
      "choice_stability": 1,
      "mean_total_variation": 0.02091375128408937,
      "objective_accuracy_delta": 0.0
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.9583333333333334,
      "mean_total_variation": 0.029528072518856446,
      "objective_accuracy_delta": 0.031914893617021274
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.9791666666666666,
      "mean_total_variation": 0.025185899061304026,
      "objective_accuracy_delta": 0.0
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.9444444444444444,
      "mean_total_variation": 0.03503896092826666,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.9722222222222222,
      "mean_total_variation": 0.04548053372699601,
      "objective_accuracy_delta": 0.010638297872340425
    }
  }
}
```

## Limitations

- Synthetic seed scenarios and deterministic perturbations are correlated, not independently human-validated samples.
- Held-out semantic groups reduce template leakage but do not establish real-world generalization.
- Abstention threshold is a test configuration, not a validated safety guarantee.
- A limited run can omit labels and robustness pairs; AUROC is null where undefined.

The companion JSON contains probabilities, raw scores, per-case choice-order probes, reliability bins, confusion matrices, valid-vocabulary AUROC and full runtime metadata.
