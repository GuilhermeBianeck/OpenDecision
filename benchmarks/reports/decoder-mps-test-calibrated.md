# OpenDecision benchmark report

Status: **measured**. Generated 2026-09-18T13:38:14.520870+00:00.

Model `Qwen/Qwen3-0.6B`; revision `c1899de289a04d12100db370d81485cdf75e47ca`; backend `decoder`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.6044905008635578 |
| balanced_accuracy | 0.6263613668278982 |
| macro_f1 | 0.6477455796743808 |
| negative_log_likelihood | 0.8743049214196227 |
| brier_score | 0.5279603392638766 |
| ece | 0.07717079600547198 |
| coverage | 1.0 |
| selective_accuracy | 0.6044905008635578 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.6209 | 0.1228 |
| agent_control | 156 | 0.4808 | 0.0769 |
| verification | 352 | 0.6818 | 0.1057 |
| robustness | 286 | 0.5559 | 0.1870 |

## Objective accuracy by candidate count

A margin threshold is only comparable within one candidate count once a
per-count calibration profile is applied. Read coverage per row, not pooled.

| Candidates | Count | Accuracy | ECE | Coverage at margin >= 0.5 | Accuracy when answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 690 | 0.6261 | 0.0527 | 0.0449 | 0.7419 |
| 4 | 312 | 0.5096 | 0.1878 | 0.4071 | 0.4882 |
| 12 | 156 | 0.6987 | 0.1457 | 0.3590 | 0.7679 |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.75` across 44 policy-labeled cases.
Ranking NDCG: `0.9815883087928584`. Ambiguous-case abstention rate: `0.8020833333333334`.
Ordinal rubrics: 216 cases, exact level `0.5231481481481481`, within one level `0.8518518518518519`, mean absolute expected error `0.6462340760811789` level steps.
Verification statements: 352 cases, 352 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `3721.701124974061` ms. First batch: `3952.612` ms.
Warm batch latency: `{'p50': 118.37958300020546, 'p95': 301.82434177258983, 'p99': 311.0514232167043, 'samples': 1725}`. Amortized per decision: `{'p50': 118.37958300020546, 'p95': 301.82434177258983, 'p99': 311.0514232167043}`.
Throughput including first batch: `6.990` decisions/s.
Process lifetime peak RSS: `4985782272` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.6044905008635578 |
| 0.25 | 0.4870 | 0.650709219858156 |
| 0.5 | 0.1848 | 0.5981308411214953 |
| 0.75 | 0.0613 | 0.6338028169014085 |
| 0.8 | 0.0466 | 0.6111111111111112 |
| 0.9 | 0.0138 | 0.5625 |
| 0.95 | 0.0000 | None |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.38333333333333336,
      "mean_total_variation": 0.34842264735470935,
      "objective_accuracy_delta": -0.5531914893617021
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.7016129032258065,
      "mean_total_variation": 0.16219995632378406,
      "objective_accuracy_delta": -0.10638297872340426
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.7258064516129032,
      "mean_total_variation": 0.1367091968858424,
      "objective_accuracy_delta": -0.031914893617021274
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.9166666666666666,
      "mean_total_variation": 0.059098852086693784,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.8194444444444444,
      "mean_total_variation": 0.09116587602626264,
      "objective_accuracy_delta": 0.06382978723404255
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.9444444444444444,
      "mean_total_variation": 0.02601042997770614,
      "objective_accuracy_delta": 0.02127659574468085
    },
    "option_rotation": {
      "pairs": 94,
      "choice_stability": 0.8723404255319149,
      "mean_total_variation": 0.1314281693090875,
      "objective_accuracy_delta": 0.04838709677419355
    },
    "question_rewording": {
      "pairs": 112,
      "choice_stability": 0.8660714285714286,
      "mean_total_variation": 0.051428862805517325,
      "objective_accuracy_delta": 0.06451612903225806
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.8958333333333334,
      "mean_total_variation": 0.06556807052187058,
      "objective_accuracy_delta": 0.05319148936170213
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.8958333333333334,
      "mean_total_variation": 0.07298822601965249,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.9305555555555556,
      "mean_total_variation": 0.04192761797156151,
      "objective_accuracy_delta": 0.05319148936170213
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.9166666666666666,
      "mean_total_variation": 0.05952235023087351,
      "objective_accuracy_delta": -0.02127659574468085
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
