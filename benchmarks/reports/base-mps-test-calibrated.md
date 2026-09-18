# OpenDecision benchmark report

Status: **measured**. Generated 2026-09-18T13:34:07.034797+00:00.

Model `tasksource/ModernBERT-base-nli`; revision `de4ab7e77845098b7fab7f6ab9d370ddff27b19c`; backend `base`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.6753022452504318 |
| balanced_accuracy | 0.5679790572285501 |
| macro_f1 | 0.5933748600731071 |
| negative_log_likelihood | 1.038926631533914 |
| brier_score | 0.5232759473716049 |
| ece | 0.1844504882921696 |
| coverage | 1.0 |
| selective_accuracy | 0.6753022452504318 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.5632 | 0.1056 |
| agent_control | 156 | 0.7564 | 0.1746 |
| verification | 352 | 0.9659 | 0.3713 |
| robustness | 286 | 0.4161 | 0.1788 |

## Objective accuracy by candidate count

A margin threshold is only comparable within one candidate count once a
per-count calibration profile is applied. Read coverage per row, not pooled.

| Candidates | Count | Accuracy | ECE | Coverage at margin >= 0.5 | Accuracy when answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 690 | 0.7667 | 0.1995 | 0.0000 |  |
| 4 | 312 | 0.5962 | 0.1774 | 0.4263 | 0.6015 |
| 12 | 156 | 0.4295 | 0.1867 | 0.0769 | 0.4167 |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.9090909090909091` across 44 policy-labeled cases.
Ranking NDCG: `0.8738919743718051`. Ambiguous-case abstention rate: `1`.
Ordinal rubrics: 216 cases, exact level `0.4166666666666667`, within one level `0.8518518518518519`, mean absolute expected error `0.6834810233267786` level steps.
Verification statements: 352 cases, 352 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `2489.1714159748517` ms. First batch: `2632.040` ms.
Warm batch latency: `{'p50': 22.310958011075854, 'p95': 132.206733035855, 'p99': 267.32204815838486, 'samples': 1725}`. Amortized per decision: `{'p50': 22.310958011075854, 'p95': 132.206733035855, 'p99': 267.32204815838486}`.
Throughput including first batch: `21.933` decisions/s.
Process lifetime peak RSS: `649265152` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.6753022452504318 |
| 0.25 | 0.2953 | 0.7485380116959064 |
| 0.5 | 0.1252 | 0.5862068965517241 |
| 0.75 | 0.0682 | 0.5316455696202531 |
| 0.8 | 0.0596 | 0.5072463768115942 |
| 0.9 | 0.0449 | 0.4807692307692308 |
| 0.95 | 0.0354 | 0.4146341463414634 |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.5333333333333333,
      "mean_total_variation": 0.1763475696572011,
      "objective_accuracy_delta": -0.35106382978723405
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.7016129032258065,
      "mean_total_variation": 0.12815291546084398,
      "objective_accuracy_delta": -0.14893617021276595
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.7338709677419355,
      "mean_total_variation": 0.10599972712418826,
      "objective_accuracy_delta": -0.10638297872340426
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.8958333333333334,
      "mean_total_variation": 0.0539700769920131,
      "objective_accuracy_delta": -0.031914893617021274
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.7361111111111112,
      "mean_total_variation": 0.11826154339311265,
      "objective_accuracy_delta": -0.19148936170212766
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.9305555555555556,
      "mean_total_variation": 0.02994746622383068,
      "objective_accuracy_delta": 0.010638297872340425
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
      "mean_total_variation": 0.0692837449353961,
      "objective_accuracy_delta": -0.08064516129032258
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.8541666666666666,
      "mean_total_variation": 0.05016880302899517,
      "objective_accuracy_delta": -0.031914893617021274
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.8055555555555556,
      "mean_total_variation": 0.07103878438085648,
      "objective_accuracy_delta": 0.0
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.875,
      "mean_total_variation": 0.04469614455416705,
      "objective_accuracy_delta": -0.0425531914893617
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.7361111111111112,
      "mean_total_variation": 0.10094794727358505,
      "objective_accuracy_delta": -0.14893617021276595
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
