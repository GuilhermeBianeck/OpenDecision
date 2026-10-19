# OpenDecision benchmark report

Status: **measured**. Generated 2026-10-19T10:20:05.337755+00:00.

Model `mlx-community/Qwen3.5-2B-4bit`; revision `674aaa7240b91e8012fcad5d791b7dfe5ba90207`; backend `qwen35`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.802245250431779 |
| balanced_accuracy | 0.7463631871846882 |
| macro_f1 | 0.7562424876606938 |
| negative_log_likelihood | 0.6083247184693816 |
| brier_score | 0.31290688834237645 |
| ece | 0.09183843866973673 |
| coverage | 1.0 |
| selective_accuracy | 0.802245250431779 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.8022 | 0.1244 |
| agent_control | 156 | 0.7756 | 0.1143 |
| verification | 352 | 0.9886 | 0.0197 |
| robustness | 286 | 0.5874 | 0.2615 |

## Objective accuracy by candidate count

A margin threshold is only comparable within one candidate count once a
per-count calibration profile is applied. Read coverage per row, not pooled.

| Candidates | Count | Accuracy | ECE | Coverage at margin >= 0.5 | Accuracy when answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 690 | 0.8536 | 0.0625 | 0.7913 | 0.9011 |
| 4 | 312 | 0.7147 | 0.1927 | 0.8237 | 0.7354 |
| 12 | 156 | 0.7500 | 0.1101 | 0.6474 | 0.9010 |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.9090909090909091` across 44 policy-labeled cases.
Ranking NDCG: `0.9808345598837861`. Ambiguous-case abstention rate: `0.6458333333333334`.
Ordinal rubrics: 216 cases, exact level `0.6296296296296297`, within one level `0.8425925925925926`, mean absolute expected error `0.5420913014615281` level steps.
Verification statements: 352 cases, 352 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `2322.2263749994454` ms. First batch: `2518.541` ms.
Warm batch latency: `{'p50': 142.19412499733153, 'p95': 240.65305839831126, 'p99': 253.35646316118073, 'samples': 1725}`. Amortized per decision: `{'p50': 142.19412499733153, 'p95': 240.65305839831126, 'p99': 253.35646316118073}`.
Throughput including first batch: `7.124` decisions/s.
Process lifetime peak RSS: `1955856384` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.802245250431779 |
| 0.25 | 0.8748 | 0.8321816386969397 |
| 0.5 | 0.7807 | 0.8539823008849557 |
| 0.75 | 0.6589 | 0.8859764089121888 |
| 0.8 | 0.6183 | 0.9008379888268156 |
| 0.9 | 0.4819 | 0.9265232974910395 |
| 0.95 | 0.3834 | 0.9436936936936937 |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.375,
      "mean_total_variation": 0.5361052134683671,
      "objective_accuracy_delta": -0.5319148936170213
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.8064516129032258,
      "mean_total_variation": 0.18240924799025146,
      "objective_accuracy_delta": -0.10638297872340426
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.7983870967741935,
      "mean_total_variation": 0.16332177302496748,
      "objective_accuracy_delta": -0.06382978723404255
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.9027777777777778,
      "mean_total_variation": 0.08868839202926308,
      "objective_accuracy_delta": -0.02127659574468085
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.8819444444444444,
      "mean_total_variation": 0.10794413842883804,
      "objective_accuracy_delta": -0.02127659574468085
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.9444444444444444,
      "mean_total_variation": 0.026893576578380544,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "option_rotation": {
      "pairs": 94,
      "choice_stability": 0.9680851063829787,
      "mean_total_variation": 0.044314428846168666,
      "objective_accuracy_delta": 0.016129032258064516
    },
    "question_rewording": {
      "pairs": 112,
      "choice_stability": 0.9553571428571429,
      "mean_total_variation": 0.04661436462252817,
      "objective_accuracy_delta": 0.016129032258064516
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.9097222222222222,
      "mean_total_variation": 0.05754622641941352,
      "objective_accuracy_delta": -0.02127659574468085
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.875,
      "mean_total_variation": 0.1417646689279568,
      "objective_accuracy_delta": -0.031914893617021274
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.9027777777777778,
      "mean_total_variation": 0.058071523531586955,
      "objective_accuracy_delta": 0.0
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.9375,
      "mean_total_variation": 0.059904781097569054,
      "objective_accuracy_delta": -0.010638297872340425
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
