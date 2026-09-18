# OpenDecision benchmark report

Status: **measured**. Generated 2026-09-18T15:04:51.620731+00:00.

Model `Skywork/Skywork-Reward-V2-Qwen3-0.6B`; revision `8c14a4e9e6321deaf572544339b16b8d6bbe8886`; backend `smart`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.6174438687392055 |
| balanced_accuracy | 0.551474488999844 |
| macro_f1 | 0.5518511406007531 |
| negative_log_likelihood | 0.9597959319646028 |
| brier_score | 0.5215006186074894 |
| ece | 0.10205289762993167 |
| coverage | 1.0 |
| selective_accuracy | 0.6174438687392055 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.5385 | 0.1028 |
| agent_control | 156 | 0.6282 | 0.1785 |
| verification | 352 | 0.8125 | 0.1214 |
| robustness | 286 | 0.4720 | 0.3190 |

## Objective accuracy by candidate count

A margin threshold is only comparable within one candidate count once a
per-count calibration profile is applied. Read coverage per row, not pooled.

| Candidates | Count | Accuracy | ECE | Coverage at margin >= 0.5 | Accuracy when answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 690 | 0.6710 | 0.1298 | 0.4986 | 0.6802 |
| 4 | 312 | 0.6763 | 0.0802 | 0.5192 | 0.8333 |
| 12 | 156 | 0.2628 | 0.1532 | 0.0064 | 0.0000 |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.7272727272727273` across 44 policy-labeled cases.
Ranking NDCG: `0.9264185642859892`. Ambiguous-case abstention rate: `0.08333333333333333`.
Ordinal rubrics: 216 cases, exact level `0.6111111111111112`, within one level `0.8981481481481481`, mean absolute expected error `0.4948191695952552` level steps.
Verification statements: 352 cases, 0 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `2264.1304999706335` ms. First batch: `2455.812` ms.
Warm batch latency: `{'p50': 42.102209001313895, 'p95': 216.72675039153546, 'p99': 476.28288892097765, 'samples': 1725}`. Amortized per decision: `{'p50': 42.102209001313895, 'p95': 216.72675039153546, 'p99': 476.28288892097765}`.
Throughput including first batch: `11.695` decisions/s.
Process lifetime peak RSS: `895614976` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.6174438687392055 |
| 0.25 | 0.6528 | 0.7116402116402116 |
| 0.5 | 0.4378 | 0.727810650887574 |
| 0.75 | 0.2073 | 0.7208333333333333 |
| 0.8 | 0.1744 | 0.7227722772277227 |
| 0.9 | 0.1123 | 0.8076923076923077 |
| 0.95 | 0.0812 | 0.8617021276595744 |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.4666666666666667,
      "mean_total_variation": 0.3840902265856089,
      "objective_accuracy_delta": -0.3829787234042553
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.717741935483871,
      "mean_total_variation": 0.2212808987566786,
      "objective_accuracy_delta": -0.0851063829787234
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.717741935483871,
      "mean_total_variation": 0.19491914862852192,
      "objective_accuracy_delta": -0.09574468085106383
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.8472222222222222,
      "mean_total_variation": 0.0966940046411129,
      "objective_accuracy_delta": 0.0
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.7708333333333334,
      "mean_total_variation": 0.1797540981556659,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.8958333333333334,
      "mean_total_variation": 0.07750794222072992,
      "objective_accuracy_delta": 0.02127659574468085
    },
    "option_rotation": {
      "pairs": 94,
      "choice_stability": 1,
      "mean_total_variation": 0.0,
      "objective_accuracy_delta": 0.0
    },
    "question_rewording": {
      "pairs": 112,
      "choice_stability": 0.7857142857142857,
      "mean_total_variation": 0.10929132761141963,
      "objective_accuracy_delta": 0.016129032258064516
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.7986111111111112,
      "mean_total_variation": 0.13044940465148042,
      "objective_accuracy_delta": 0.0851063829787234
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.7430555555555556,
      "mean_total_variation": 0.14889310972779943,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.7916666666666666,
      "mean_total_variation": 0.12325782748134391,
      "objective_accuracy_delta": 0.07446808510638298
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.8194444444444444,
      "mean_total_variation": 0.12115565648385608,
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
