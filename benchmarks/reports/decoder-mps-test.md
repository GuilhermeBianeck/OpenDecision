# OpenDecision benchmark report

Status: **measured**. Generated 2026-09-18T11:42:43.218875+00:00.

Model `Qwen/Qwen3-0.6B`; revision `c1899de289a04d12100db370d81485cdf75e47ca`; backend `decoder`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.6044905008635578 |
| balanced_accuracy | 0.6263613668278982 |
| macro_f1 | 0.6477455796743808 |
| negative_log_likelihood | 1.891814982834431 |
| brier_score | 0.672563493090619 |
| ece | 0.27539721411718915 |
| coverage | 1.0 |
| selective_accuracy | 0.6044905008635578 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.6209 | 0.2703 |
| agent_control | 156 | 0.4808 | 0.3229 |
| verification | 352 | 0.6818 | 0.2010 |
| robustness | 286 | 0.5559 | 0.3892 |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.75` across 44 policy-labeled cases.
Ranking NDCG: `0.9815883087928584`. Ambiguous-case abstention rate: `0.07291666666666667`.
Ordinal rubrics: 216 cases, exact level `0.5231481481481481`, within one level `0.8518518518518519`, mean absolute expected error `0.6377899690376465` level steps.
Verification statements: 352 cases, 352 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `6466.5704580256715` ms. First batch: `6817.525` ms.
Warm batch latency: `{'p50': 158.1232919706963, 'p95': 419.39634979935363, 'p99': 444.5553019223735, 'samples': 1725}`. Amortized per decision: `{'p50': 158.1232919706963, 'p95': 419.39634979935363, 'p99': 444.5553019223735}`.
Throughput including first batch: `5.153` decisions/s.
Process lifetime peak RSS: `4985815040` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.6044905008635578 |
| 0.25 | 0.9067 | 0.6114285714285714 |
| 0.5 | 0.7858 | 0.6076923076923076 |
| 0.75 | 0.6511 | 0.6140583554376657 |
| 0.8 | 0.6166 | 0.6176470588235294 |
| 0.9 | 0.5363 | 0.6135265700483091 |
| 0.95 | 0.4508 | 0.6398467432950191 |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.38333333333333336,
      "mean_total_variation": 0.5872132204749979,
      "objective_accuracy_delta": -0.5531914893617021
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.6935483870967742,
      "mean_total_variation": 0.2725965950316105,
      "objective_accuracy_delta": -0.10638297872340426
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.7096774193548387,
      "mean_total_variation": 0.24924098413139176,
      "objective_accuracy_delta": -0.031914893617021274
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.9236111111111112,
      "mean_total_variation": 0.09299004918744005,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.8194444444444444,
      "mean_total_variation": 0.16771124550279903,
      "objective_accuracy_delta": 0.06382978723404255
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.9444444444444444,
      "mean_total_variation": 0.047205346154529274,
      "objective_accuracy_delta": 0.02127659574468085
    },
    "option_rotation": {
      "pairs": 94,
      "choice_stability": 0.8297872340425532,
      "mean_total_variation": 0.15776912196769702,
      "objective_accuracy_delta": 0.04838709677419355
    },
    "question_rewording": {
      "pairs": 112,
      "choice_stability": 0.875,
      "mean_total_variation": 0.09535074810922227,
      "objective_accuracy_delta": 0.06451612903225806
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.9097222222222222,
      "mean_total_variation": 0.10345763146700258,
      "objective_accuracy_delta": 0.05319148936170213
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.9027777777777778,
      "mean_total_variation": 0.10198987125432928,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.9305555555555556,
      "mean_total_variation": 0.07557425962118199,
      "objective_accuracy_delta": 0.05319148936170213
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.9236111111111112,
      "mean_total_variation": 0.09491852589607928,
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
