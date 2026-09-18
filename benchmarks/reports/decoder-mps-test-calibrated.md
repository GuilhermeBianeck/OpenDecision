# OpenDecision benchmark report

Status: **measured**. Generated 2026-09-18T15:11:08.791709+00:00.

Model `Qwen/Qwen3-0.6B`; revision `c1899de289a04d12100db370d81485cdf75e47ca`; backend `decoder`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.6079447322970639 |
| balanced_accuracy | 0.6272225689568489 |
| macro_f1 | 0.6471292938977479 |
| negative_log_likelihood | 0.8720672957453456 |
| brier_score | 0.5268349267116271 |
| ece | 0.08313716865937397 |
| coverage | 1.0 |
| selective_accuracy | 0.6079447322970639 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.6181 | 0.1162 |
| agent_control | 156 | 0.4872 | 0.0946 |
| verification | 352 | 0.6960 | 0.1095 |
| robustness | 286 | 0.5524 | 0.1596 |

## Objective accuracy by candidate count

A margin threshold is only comparable within one candidate count once a
per-count calibration profile is applied. Read coverage per row, not pooled.

| Candidates | Count | Accuracy | ECE | Coverage at margin >= 0.5 | Accuracy when answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 690 | 0.6304 | 0.0578 | 0.0565 | 0.7436 |
| 4 | 312 | 0.5128 | 0.1839 | 0.4038 | 0.4762 |
| 12 | 156 | 0.6987 | 0.1448 | 0.3526 | 0.7818 |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.75` across 44 policy-labeled cases.
Ranking NDCG: `0.9815883087928584`. Ambiguous-case abstention rate: `0.7708333333333334`.
Ordinal rubrics: 216 cases, exact level `0.5092592592592593`, within one level `0.8425925925925926`, mean absolute expected error `0.6489622944628005` level steps.
Verification statements: 352 cases, 352 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `2499.8893750016578` ms. First batch: `177.438` ms.
Warm batch latency: `{'p50': 65.82995899952948, 'p95': 127.02463318128137, 'p99': 131.28198814811185, 'samples': 1725}`. Amortized per decision: `{'p50': 65.82995899952948, 'p95': 127.02463318128137, 'p99': 131.28198814811185}`.
Throughput including first batch: `13.828` decisions/s.
Process lifetime peak RSS: `1133363200` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.6079447322970639 |
| 0.25 | 0.4905 | 0.6461267605633803 |
| 0.5 | 0.1900 | 0.6 |
| 0.75 | 0.0570 | 0.6363636363636364 |
| 0.8 | 0.0449 | 0.6153846153846154 |
| 0.9 | 0.0104 | 0.5833333333333334 |
| 0.95 | 0.0000 | None |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.35833333333333334,
      "mean_total_variation": 0.3490091218061844,
      "objective_accuracy_delta": -0.5957446808510638
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.717741935483871,
      "mean_total_variation": 0.16202959689370125,
      "objective_accuracy_delta": -0.10638297872340426
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.7258064516129032,
      "mean_total_variation": 0.1369519582960976,
      "objective_accuracy_delta": -0.0425531914893617
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.9236111111111112,
      "mean_total_variation": 0.05959273797453324,
      "objective_accuracy_delta": -0.010638297872340425
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.7986111111111112,
      "mean_total_variation": 0.09280705397138687,
      "objective_accuracy_delta": 0.031914893617021274
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.9375,
      "mean_total_variation": 0.0277117964402248,
      "objective_accuracy_delta": 0.0
    },
    "option_rotation": {
      "pairs": 94,
      "choice_stability": 0.8723404255319149,
      "mean_total_variation": 0.13343674936103356,
      "objective_accuracy_delta": 0.04838709677419355
    },
    "question_rewording": {
      "pairs": 112,
      "choice_stability": 0.8839285714285714,
      "mean_total_variation": 0.050519659113471574,
      "objective_accuracy_delta": 0.06451612903225806
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.9027777777777778,
      "mean_total_variation": 0.06584388393205957,
      "objective_accuracy_delta": 0.031914893617021274
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.9027777777777778,
      "mean_total_variation": 0.07524368038248112,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.9236111111111112,
      "mean_total_variation": 0.04373354924316988,
      "objective_accuracy_delta": 0.031914893617021274
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.9097222222222222,
      "mean_total_variation": 0.06078575995300307,
      "objective_accuracy_delta": -0.0425531914893617
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
