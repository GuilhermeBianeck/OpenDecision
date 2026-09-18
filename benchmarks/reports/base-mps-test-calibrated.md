# OpenDecision benchmark report

Status: **measured**. Generated 2026-09-18T15:09:00.870779+00:00.

Model `tasksource/ModernBERT-base-nli`; revision `de4ab7e77845098b7fab7f6ab9d370ddff27b19c`; backend `base`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.6753022452504318 |
| balanced_accuracy | 0.5686200828695758 |
| macro_f1 | 0.5933338993270614 |
| negative_log_likelihood | 1.0378678565108685 |
| brier_score | 0.5216415060963433 |
| ece | 0.1829944946759007 |
| coverage | 1.0 |
| selective_accuracy | 0.6753022452504318 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.5687 | 0.1141 |
| agent_control | 156 | 0.7500 | 0.1677 |
| verification | 352 | 0.9631 | 0.3642 |
| robustness | 286 | 0.4161 | 0.1804 |

## Objective accuracy by candidate count

A margin threshold is only comparable within one candidate count once a
per-count calibration profile is applied. Read coverage per row, not pooled.

| Candidates | Count | Accuracy | ECE | Coverage at margin >= 0.5 | Accuracy when answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 690 | 0.7652 | 0.1950 | 0.0000 |  |
| 4 | 312 | 0.5962 | 0.1862 | 0.4135 | 0.5969 |
| 12 | 156 | 0.4359 | 0.1942 | 0.0641 | 0.5000 |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.9090909090909091` across 44 policy-labeled cases.
Ranking NDCG: `0.8673343167718516`. Ambiguous-case abstention rate: `0.9895833333333334`.
Ordinal rubrics: 216 cases, exact level `0.4212962962962963`, within one level `0.8611111111111112`, mean absolute expected error `0.6832366823012231` level steps.
Verification statements: 352 cases, 352 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `2698.3960829675198` ms. First batch: `115.321` ms.
Warm batch latency: `{'p50': 13.99095804663375, 'p95': 74.613199417945, 'p99': 134.54650008818135, 'samples': 1725}`. Amortized per decision: `{'p50': 13.99095804663375, 'p95': 74.613199417945, 'p99': 134.54650008818135}`.
Throughput including first batch: `39.855` decisions/s.
Process lifetime peak RSS: `1329840128` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.6753022452504318 |
| 0.25 | 0.3109 | 0.7611111111111111 |
| 0.5 | 0.1200 | 0.5899280575539568 |
| 0.75 | 0.0674 | 0.5256410256410257 |
| 0.8 | 0.0613 | 0.5352112676056338 |
| 0.9 | 0.0449 | 0.4807692307692308 |
| 0.95 | 0.0363 | 0.42857142857142855 |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.5416666666666666,
      "mean_total_variation": 0.17607299776963797,
      "objective_accuracy_delta": -0.32978723404255317
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.7016129032258065,
      "mean_total_variation": 0.12737404304408204,
      "objective_accuracy_delta": -0.1276595744680851
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.717741935483871,
      "mean_total_variation": 0.10676942365103774,
      "objective_accuracy_delta": -0.11702127659574468
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.9027777777777778,
      "mean_total_variation": 0.05360744314810685,
      "objective_accuracy_delta": -0.031914893617021274
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.7222222222222222,
      "mean_total_variation": 0.11907634301104633,
      "objective_accuracy_delta": -0.19148936170212766
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.9166666666666666,
      "mean_total_variation": 0.03032565920559491,
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
      "choice_stability": 0.8660714285714286,
      "mean_total_variation": 0.07059129356505332,
      "objective_accuracy_delta": -0.016129032258064516
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.8402777777777778,
      "mean_total_variation": 0.05079712790852458,
      "objective_accuracy_delta": -0.031914893617021274
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.7986111111111112,
      "mean_total_variation": 0.07132251186951566,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.875,
      "mean_total_variation": 0.04441219609402948,
      "objective_accuracy_delta": -0.02127659574468085
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.7430555555555556,
      "mean_total_variation": 0.10109567126784935,
      "objective_accuracy_delta": -0.13829787234042554
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
