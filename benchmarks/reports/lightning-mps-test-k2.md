# OpenDecision benchmark report

Status: **measured**. Generated 2026-10-19T11:02:51.083779+00:00.

Model `netprojects/h2o-lightning-4b-mlx-q4`; revision `2bda86b6f1ecbb5e6055f9fe8986f2a6e06849e4`; backend `lightning`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.9447322970639033 |
| balanced_accuracy | 0.9358315563183717 |
| macro_f1 | 0.9452257991851158 |
| negative_log_likelihood | 0.26604577751665515 |
| brier_score | 0.11792532312402414 |
| ece | 0.11249152341796 |
| coverage | 1.0 |
| selective_accuracy | 0.9447322970639033 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.9725 | 0.1670 |
| agent_control | 156 | 0.9744 | 0.1860 |
| verification | 352 | 0.9716 | 0.1235 |
| robustness | 286 | 0.8601 | 0.1135 |

## Objective accuracy by candidate count

A margin threshold is only comparable within one candidate count once a
per-count calibration profile is applied. Read coverage per row, not pooled.

| Candidates | Count | Accuracy | ECE | Coverage at margin >= 0.5 | Accuracy when answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 690 | 0.9725 | 0.1136 | 0.8623 | 0.9983 |
| 4 | 312 | 0.8750 | 0.0832 | 0.8429 | 0.9011 |
| 12 | 156 | 0.9615 | 0.1767 | 0.8590 | 0.9851 |

## Separate evaluation families

Policy agreement (not moral accuracy): `1` across 44 policy-labeled cases.
Ranking NDCG: `1.0`. Ambiguous-case abstention rate: `0.9375`.
Ordinal rubrics: 216 cases, exact level `0.8518518518518519`, within one level `0.9814814814814815`, mean absolute expected error `0.34565453704894944` level steps.
Verification statements: 352 cases, 352 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `2494.118458002049` ms. First batch: `2900.378` ms.
Warm batch latency: `{'p50': 267.1379160019569, 'p95': 493.36379139713245, 'p99': 582.2073318395995, 'samples': 1725}`. Amortized per decision: `{'p50': 267.1379160019569, 'p95': 493.36379139713245, 'p99': 582.2073318395995}`.
Throughput including first batch: `3.504` decisions/s.
Process lifetime peak RSS: `3285647360` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.9447322970639033 |
| 0.25 | 0.9344 | 0.9593345656192237 |
| 0.5 | 0.8566 | 0.9707661290322581 |
| 0.75 | 0.6105 | 0.9858557284299858 |
| 0.8 | 0.3722 | 0.988399071925754 |
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
      "choice_stability": 0.8583333333333333,
      "mean_total_variation": 0.1627069709548537,
      "objective_accuracy_delta": -0.1276595744680851
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.8951612903225806,
      "mean_total_variation": 0.10319404574124932,
      "objective_accuracy_delta": 0.0
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.9435483870967742,
      "mean_total_variation": 0.05784565122690279,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.9861111111111112,
      "mean_total_variation": 0.021799719518015747,
      "objective_accuracy_delta": 0.02127659574468085
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.9513888888888888,
      "mean_total_variation": 0.06096991858379349,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.9930555555555556,
      "mean_total_variation": 0.009609238064273756,
      "objective_accuracy_delta": 0.0
    },
    "option_rotation": {
      "pairs": 94,
      "choice_stability": 1,
      "mean_total_variation": 0.013950451431542311,
      "objective_accuracy_delta": 0.0
    },
    "question_rewording": {
      "pairs": 112,
      "choice_stability": 1,
      "mean_total_variation": 0.019102858258673125,
      "objective_accuracy_delta": 0.0
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.9583333333333334,
      "mean_total_variation": 0.027335071840963433,
      "objective_accuracy_delta": 0.031914893617021274
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.9930555555555556,
      "mean_total_variation": 0.02399026418773209,
      "objective_accuracy_delta": 0.0
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.9583333333333334,
      "mean_total_variation": 0.03151415319776191,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.9791666666666666,
      "mean_total_variation": 0.043277792180461044,
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
