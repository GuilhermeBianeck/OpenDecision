# OpenDecision benchmark report

Status: **measured**. Generated 2026-09-18T11:47:16.875477+00:00.

Model `BAAI/bge-reranker-v2-m3`; revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`; backend `multilingual`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.5120898100172712 |
| balanced_accuracy | 0.5270171286905567 |
| macro_f1 | 0.5169633424963841 |
| negative_log_likelihood | 1.4554735211005252 |
| brier_score | 0.6976891801633519 |
| ece | 0.2010475240161329 |
| coverage | 1.0 |
| selective_accuracy | 0.5120898100172712 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.5852 | 0.1375 |
| agent_control | 156 | 0.6731 | 0.0592 |
| verification | 352 | 0.4517 | 0.1965 |
| robustness | 286 | 0.4056 | 0.4919 |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.7045454545454546` across 44 policy-labeled cases.
Ranking NDCG: `0.9829752152037564`. Ambiguous-case abstention rate: `0.17708333333333334`.
Ordinal rubrics: 216 cases, exact level `0.36574074074074076`, within one level `0.7546296296296297`, mean absolute expected error `0.8278311117434569` level steps.
Verification statements: 352 cases, 0 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `5305.319582985248` ms. First batch: `5637.921` ms.
Warm batch latency: `{'p50': 58.760292013175786, 'p95': 499.7030252008699, 'p99': 960.2440913137979, 'samples': 1725}`. Amortized per decision: `{'p50': 58.760292013175786, 'p95': 499.7030252008699, 'p99': 960.2440913137979}`.
Throughput including first batch: `6.335` decisions/s.
Process lifetime peak RSS: `763101184` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.5120898100172712 |
| 0.25 | 0.6330 | 0.5429740791268759 |
| 0.5 | 0.4508 | 0.5306513409961686 |
| 0.75 | 0.3385 | 0.5255102040816326 |
| 0.8 | 0.3083 | 0.5070028011204482 |
| 0.9 | 0.2314 | 0.44776119402985076 |
| 0.95 | 0.1917 | 0.42792792792792794 |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.35833333333333334,
      "mean_total_variation": 0.5781728127498896,
      "objective_accuracy_delta": -0.574468085106383
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.5403225806451613,
      "mean_total_variation": 0.2773916564219158,
      "objective_accuracy_delta": -0.13829787234042554
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.6935483870967742,
      "mean_total_variation": 0.19751334772878365,
      "objective_accuracy_delta": -0.031914893617021274
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.8125,
      "mean_total_variation": 0.138362895769573,
      "objective_accuracy_delta": 0.031914893617021274
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.6041666666666666,
      "mean_total_variation": 0.28530101292533083,
      "objective_accuracy_delta": -0.20212765957446807
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.9027777777777778,
      "mean_total_variation": 0.04260028225267695,
      "objective_accuracy_delta": 0.031914893617021274
    },
    "option_rotation": {
      "pairs": 94,
      "choice_stability": 1,
      "mean_total_variation": 0.0,
      "objective_accuracy_delta": 0.0
    },
    "question_rewording": {
      "pairs": 112,
      "choice_stability": 0.8571428571428571,
      "mean_total_variation": 0.13140322572700117,
      "objective_accuracy_delta": 0.0
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.8333333333333334,
      "mean_total_variation": 0.08311441003567434,
      "objective_accuracy_delta": 0.0
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.75,
      "mean_total_variation": 0.12632164463913742,
      "objective_accuracy_delta": 0.0
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.8541666666666666,
      "mean_total_variation": 0.09077673414783244,
      "objective_accuracy_delta": 0.07446808510638298
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.7777777777777778,
      "mean_total_variation": 0.12110267082841762,
      "objective_accuracy_delta": -0.05319148936170213
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
