# OpenDecision benchmark report

Status: **measured**. Generated 2026-09-18T15:02:23.494509+00:00.

Model `tasksource/ModernBERT-base-nli`; revision `de4ab7e77845098b7fab7f6ab9d370ddff27b19c`; backend `base`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.6753022452504318 |
| balanced_accuracy | 0.5686200828695758 |
| macro_f1 | 0.5933338993270614 |
| negative_log_likelihood | 0.9124425863927647 |
| brier_score | 0.48072430981891684 |
| ece | 0.1537403426211059 |
| coverage | 1.0 |
| selective_accuracy | 0.6753022452504318 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.5687 | 0.2861 |
| agent_control | 156 | 0.7500 | 0.2638 |
| verification | 352 | 0.9631 | 0.0407 |
| robustness | 286 | 0.4161 | 0.3060 |

## Objective accuracy by candidate count

A margin threshold is only comparable within one candidate count once a
per-count calibration profile is applied. Read coverage per row, not pooled.

| Candidates | Count | Accuracy | ECE | Coverage at margin >= 0.5 | Accuracy when answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 690 | 0.7652 | 0.0817 | 0.7290 | 0.8608 |
| 4 | 312 | 0.5962 | 0.2586 | 0.1346 | 0.4286 |
| 12 | 156 | 0.4359 | 0.3022 | 0.0000 |  |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.9090909090909091` across 44 policy-labeled cases.
Ranking NDCG: `0.8673343167718516`. Ambiguous-case abstention rate: `0.5625`.
Ordinal rubrics: 216 cases, exact level `0.4212962962962963`, within one level `0.8611111111111112`, mean absolute expected error `0.6939865543361508` level steps.
Verification statements: 352 cases, 352 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `2077.2317920345813` ms. First batch: `2192.551` ms.
Warm batch latency: `{'p50': 13.937417010311037, 'p95': 75.65094177844001, 'p99': 133.7364100664854, 'samples': 1725}`. Amortized per decision: `{'p50': 13.937417010311037, 'p95': 75.65094177844001, 'p99': 133.7364100664854}`.
Throughput including first batch: `38.293` decisions/s.
Process lifetime peak RSS: `1330577408` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.6753022452504318 |
| 0.25 | 0.5812 | 0.787518573551263 |
| 0.5 | 0.4706 | 0.8275229357798165 |
| 0.75 | 0.3601 | 0.8752997601918465 |
| 0.8 | 0.3290 | 0.8923884514435696 |
| 0.9 | 0.2297 | 0.9774436090225563 |
| 0.95 | 0.1649 | 1 |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.5416666666666666,
      "mean_total_variation": 0.22112441744168523,
      "objective_accuracy_delta": -0.32978723404255317
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.6774193548387096,
      "mean_total_variation": 0.13367058540822865,
      "objective_accuracy_delta": -0.1276595744680851
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.6774193548387096,
      "mean_total_variation": 0.1140744257896668,
      "objective_accuracy_delta": -0.11702127659574468
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.8958333333333334,
      "mean_total_variation": 0.05239960988512391,
      "objective_accuracy_delta": -0.031914893617021274
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.7083333333333334,
      "mean_total_variation": 0.16445383212254358,
      "objective_accuracy_delta": -0.19148936170212766
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.9027777777777778,
      "mean_total_variation": 0.04119379744987474,
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
      "choice_stability": 0.8303571428571429,
      "mean_total_variation": 0.07752742455595843,
      "objective_accuracy_delta": -0.016129032258064516
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.8333333333333334,
      "mean_total_variation": 0.05211979709940042,
      "objective_accuracy_delta": -0.031914893617021274
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.7847222222222222,
      "mean_total_variation": 0.10181472255247874,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.8472222222222222,
      "mean_total_variation": 0.04941506477228829,
      "objective_accuracy_delta": -0.02127659574468085
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.7152777777777778,
      "mean_total_variation": 0.1202731970911662,
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
