# OpenDecision benchmark report

Status: **measured**. Generated 2026-09-18T19:40:09.216511+00:00.

Model `mlx-community/Qwen3.5-4B-4bit`; revision `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`; backend `qwen35_4b`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.853195164075993 |
| balanced_accuracy | 0.8186361193460585 |
| macro_f1 | 0.8396075027035714 |
| negative_log_likelihood | 0.7851449805575886 |
| brier_score | 0.24454463436490304 |
| ece | 0.09644119121562135 |
| coverage | 1.0 |
| selective_accuracy | 0.853195164075993 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.8544 | 0.0891 |
| agent_control | 156 | 0.8013 | 0.1379 |
| verification | 352 | 0.9830 | 0.0070 |
| robustness | 286 | 0.7203 | 0.2271 |

## Objective accuracy by candidate count

A margin threshold is only comparable within one candidate count once a
per-count calibration profile is applied. Read coverage per row, not pooled.

| Candidates | Count | Accuracy | ECE | Coverage at margin >= 0.5 | Accuracy when answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 690 | 0.9116 | 0.0465 | 0.9420 | 0.9385 |
| 4 | 312 | 0.7436 | 0.2115 | 0.9295 | 0.7621 |
| 12 | 156 | 0.8141 | 0.1025 | 0.8462 | 0.8712 |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.8409090909090909` across 44 policy-labeled cases.
Ranking NDCG: `0.9418654887386008`. Ambiguous-case abstention rate: `0.08333333333333333`.
Ordinal rubrics: 216 cases, exact level `0.7685185185185185`, within one level `0.9537037037037037`, mean absolute expected error `0.31303053339185677` level steps.
Verification statements: 352 cases, 352 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `2941.166542004794` ms. First batch: `3106.920` ms.
Warm batch latency: `{'p50': 135.38212497951463, 'p95': 1002.0622993935821, 'p99': 1667.4392261379398, 'samples': 1725}`. Amortized per decision: `{'p50': 135.38212497951463, 'p95': 1002.0622993935821, 'p99': 1667.4392261379398}`.
Throughput including first batch: `4.078` decisions/s.
Process lifetime peak RSS: `3266052096` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.853195164075993 |
| 0.25 | 0.9585 | 0.8693693693693694 |
| 0.5 | 0.9257 | 0.8824626865671642 |
| 0.75 | 0.8541 | 0.8988877654196158 |
| 0.8 | 0.8428 | 0.9006147540983607 |
| 0.9 | 0.8074 | 0.9016042780748663 |
| 0.95 | 0.7513 | 0.9080459770114943 |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.325,
      "mean_total_variation": 0.6525342409199886,
      "objective_accuracy_delta": -0.574468085106383
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.7903225806451613,
      "mean_total_variation": 0.21408173404801573,
      "objective_accuracy_delta": -0.10638297872340426
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.8306451612903226,
      "mean_total_variation": 0.15654601648404895,
      "objective_accuracy_delta": -0.0425531914893617
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.9166666666666666,
      "mean_total_variation": 0.06563218544600287,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.8333333333333334,
      "mean_total_variation": 0.1297993712946949,
      "objective_accuracy_delta": -0.010638297872340425
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.9375,
      "mean_total_variation": 0.029695315488731333,
      "objective_accuracy_delta": 0.02127659574468085
    },
    "option_rotation": {
      "pairs": 94,
      "choice_stability": 0.7978723404255319,
      "mean_total_variation": 0.1549738852401007,
      "objective_accuracy_delta": 0.016129032258064516
    },
    "question_rewording": {
      "pairs": 112,
      "choice_stability": 0.9107142857142857,
      "mean_total_variation": 0.05918521736884534,
      "objective_accuracy_delta": 0.016129032258064516
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.9097222222222222,
      "mean_total_variation": 0.06894340215902052,
      "objective_accuracy_delta": -0.010638297872340425
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.8125,
      "mean_total_variation": 0.1839972259972059,
      "objective_accuracy_delta": -0.05319148936170213
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.8680555555555556,
      "mean_total_variation": 0.08825874625439985,
      "objective_accuracy_delta": 0.031914893617021274
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.8819444444444444,
      "mean_total_variation": 0.09560938850444994,
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
