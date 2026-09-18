# OpenDecision benchmark report

Status: **measured**. Generated 2026-09-18T15:01:37.897721+00:00.

Model `cross-encoder/nli-deberta-v3-xsmall`; revision `a150876415327c80daeff35ca6f68f5ed8cf5c24`; backend `tiny`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.5708117443868739 |
| balanced_accuracy | 0.4316993464052287 |
| macro_f1 | 0.46625869888678345 |
| negative_log_likelihood | 0.9984990209557462 |
| brier_score | 0.5314232869460475 |
| ece | 0.12392090637161823 |
| coverage | 1.0 |
| selective_accuracy | 0.5708117443868739 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.3462 | 0.1027 |
| agent_control | 156 | 0.6346 | 0.1879 |
| verification | 352 | 0.9318 | 0.0300 |
| robustness | 286 | 0.3776 | 0.3176 |

## Objective accuracy by candidate count

A margin threshold is only comparable within one candidate count once a
per-count calibration profile is applied. Read coverage per row, not pooled.

| Candidates | Count | Accuracy | ECE | Coverage at margin >= 0.5 | Accuracy when answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 690 | 0.7203 | 0.1149 | 0.6710 | 0.8229 |
| 4 | 312 | 0.4038 | 0.1438 | 0.2051 | 0.4062 |
| 12 | 156 | 0.2436 | 0.1599 | 0.0513 | 0.0000 |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.9090909090909091` across 44 policy-labeled cases.
Ranking NDCG: `0.9402876283949687`. Ambiguous-case abstention rate: `0.3333333333333333`.
Ordinal rubrics: 216 cases, exact level `0.3611111111111111`, within one level `0.8611111111111112`, mean absolute expected error `0.7256348930159818` level steps.
Verification statements: 352 cases, 352 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `2511.777332983911` ms. First batch: `2663.112` ms.
Warm batch latency: `{'p50': 10.968958027660847, 'p95': 101.49617480346933, 'p99': 178.97439807187763, 'samples': 1725}`. Amortized per decision: `{'p50': 10.968958027660847, 'p95': 101.49617480346933, 'p99': 178.97439807187763}`.
Throughput including first batch: `36.502` decisions/s.
Process lifetime peak RSS: `1213759488` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.5708117443868739 |
| 0.25 | 0.5924 | 0.7099125364431487 |
| 0.5 | 0.4620 | 0.7607476635514019 |
| 0.75 | 0.3472 | 0.8407960199004975 |
| 0.8 | 0.3135 | 0.8677685950413223 |
| 0.9 | 0.2582 | 0.9297658862876255 |
| 0.95 | 0.2288 | 0.9471698113207547 |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.475,
      "mean_total_variation": 0.3790968242479328,
      "objective_accuracy_delta": -0.3191489361702128
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.6129032258064516,
      "mean_total_variation": 0.14205221047640795,
      "objective_accuracy_delta": -0.13829787234042554
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.6048387096774194,
      "mean_total_variation": 0.1343142069688801,
      "objective_accuracy_delta": -0.06382978723404255
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.8541666666666666,
      "mean_total_variation": 0.06373480791053798,
      "objective_accuracy_delta": 0.0
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.6736111111111112,
      "mean_total_variation": 0.14273689497457723,
      "objective_accuracy_delta": -0.09574468085106383
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.9375,
      "mean_total_variation": 0.04258249194407025,
      "objective_accuracy_delta": -0.0425531914893617
    },
    "option_rotation": {
      "pairs": 94,
      "choice_stability": 1,
      "mean_total_variation": 0.0,
      "objective_accuracy_delta": 0.0
    },
    "question_rewording": {
      "pairs": 112,
      "choice_stability": 0.6696428571428571,
      "mean_total_variation": 0.10238283574971321,
      "objective_accuracy_delta": -0.08064516129032258
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.8819444444444444,
      "mean_total_variation": 0.058191039780856754,
      "objective_accuracy_delta": -0.06382978723404255
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.6388888888888888,
      "mean_total_variation": 0.16408960311750384,
      "objective_accuracy_delta": -0.031914893617021274
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.8402777777777778,
      "mean_total_variation": 0.07105553047988349,
      "objective_accuracy_delta": -0.0851063829787234
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.8263888888888888,
      "mean_total_variation": 0.09531552956767507,
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
