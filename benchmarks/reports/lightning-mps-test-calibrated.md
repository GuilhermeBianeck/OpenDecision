# OpenDecision benchmark report

Status: **measured**. Generated 2026-10-19T09:58:06.411212+00:00.

Model `netprojects/h2o-lightning-4b-mlx-q4`; revision `2bda86b6f1ecbb5e6055f9fe8986f2a6e06849e4`; backend `lightning`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.9395509499136442 |
| balanced_accuracy | 0.9299500702137619 |
| macro_f1 | 0.9426127803644386 |
| negative_log_likelihood | 0.16641615836044815 |
| brier_score | 0.09134820242705755 |
| ece | 0.008493567322382607 |
| coverage | 1.0 |
| selective_accuracy | 0.9395509499136442 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.9560 | 0.0320 |
| agent_control | 156 | 0.9679 | 0.0695 |
| verification | 352 | 0.9716 | 0.0160 |
| robustness | 286 | 0.8636 | 0.0890 |

## Objective accuracy by candidate count

A margin threshold is only comparable within one candidate count once a
per-count calibration profile is applied. Read coverage per row, not pooled.

| Candidates | Count | Accuracy | ECE | Coverage at margin >= 0.5 | Accuracy when answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 690 | 0.9710 | 0.0088 | 0.9362 | 0.9938 |
| 4 | 312 | 0.8686 | 0.0244 | 0.9135 | 0.8982 |
| 12 | 156 | 0.9423 | 0.0192 | 0.9679 | 0.9603 |

## Separate evaluation families

Policy agreement (not moral accuracy): `1` across 44 policy-labeled cases.
Ranking NDCG: `1.0`. Ambiguous-case abstention rate: `0.0625`.
Ordinal rubrics: 216 cases, exact level `0.8518518518518519`, within one level `0.9861111111111112`, mean absolute expected error `0.24805560515158112` level steps.
Verification statements: 352 cases, 352 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `2800.5248330009636` ms. First batch: `3091.173` ms.
Warm batch latency: `{'p50': 107.38104200208909, 'p95': 392.3441006016219, 'p99': 430.9160529206565, 'samples': 1725}`. Amortized per decision: `{'p50': 107.38104200208909, 'p95': 392.3441006016219, 'p99': 430.9160529206565}`.
Throughput including first batch: `6.497` decisions/s.
Process lifetime peak RSS: `2774515712` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.9395509499136442 |
| 0.25 | 0.9775 | 0.950530035335689 |
| 0.5 | 0.9344 | 0.9639556377079482 |
| 0.75 | 0.8705 | 0.9712301587301587 |
| 0.8 | 0.8333 | 0.9792746113989638 |
| 0.9 | 0.7487 | 0.9930795847750865 |
| 0.95 | 0.6157 | 0.9957924263674615 |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.8333333333333334,
      "mean_total_variation": 0.18781566084163612,
      "objective_accuracy_delta": -0.1595744680851064
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.8709677419354839,
      "mean_total_variation": 0.12144731945973464,
      "objective_accuracy_delta": -0.010638297872340425
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.9193548387096774,
      "mean_total_variation": 0.06615120898493636,
      "objective_accuracy_delta": 0.0
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.9722222222222222,
      "mean_total_variation": 0.022911026067480376,
      "objective_accuracy_delta": 0.02127659574468085
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.9375,
      "mean_total_variation": 0.06316734032897517,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.9930555555555556,
      "mean_total_variation": 0.01085393498077568,
      "objective_accuracy_delta": 0.0
    },
    "option_rotation": {
      "pairs": 94,
      "choice_stability": 0.8723404255319149,
      "mean_total_variation": 0.05176495562489706,
      "objective_accuracy_delta": -0.016129032258064516
    },
    "question_rewording": {
      "pairs": 112,
      "choice_stability": 0.9910714285714286,
      "mean_total_variation": 0.02064119119817566,
      "objective_accuracy_delta": 0.0
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.9513888888888888,
      "mean_total_variation": 0.034127126088803354,
      "objective_accuracy_delta": 0.031914893617021274
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.9861111111111112,
      "mean_total_variation": 0.02396238511588642,
      "objective_accuracy_delta": 0.0
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.9513888888888888,
      "mean_total_variation": 0.043289701788568356,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.9652777777777778,
      "mean_total_variation": 0.035407501115016726,
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
