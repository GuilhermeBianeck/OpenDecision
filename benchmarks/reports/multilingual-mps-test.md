# OpenDecision benchmark report

Status: **measured**. Generated 2026-09-18T15:06:06.120076+00:00.

Model `BAAI/bge-reranker-v2-m3`; revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`; backend `multilingual`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.5094991364421416 |
| balanced_accuracy | 0.5236225966956191 |
| macro_f1 | 0.5143209730288568 |
| negative_log_likelihood | 1.452737405486551 |
| brier_score | 0.6980592970138257 |
| ece | 0.2043623753908504 |
| coverage | 1.0 |
| selective_accuracy | 0.5094991364421416 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.5824 | 0.1407 |
| agent_control | 156 | 0.6795 | 0.0689 |
| verification | 352 | 0.4460 | 0.2085 |
| robustness | 286 | 0.4021 | 0.4750 |

## Objective accuracy by candidate count

A margin threshold is only comparable within one candidate count once a
per-count calibration profile is applied. Read coverage per row, not pooled.

| Candidates | Count | Accuracy | ECE | Coverage at margin >= 0.5 | Accuracy when answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 690 | 0.4826 | 0.2372 | 0.3623 | 0.3960 |
| 4 | 312 | 0.5705 | 0.2503 | 0.7244 | 0.6283 |
| 12 | 156 | 0.5064 | 0.2121 | 0.3526 | 0.7818 |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.7045454545454546` across 44 policy-labeled cases.
Ranking NDCG: `0.9829752152037564`. Ambiguous-case abstention rate: `0.17708333333333334`.
Ordinal rubrics: 216 cases, exact level `0.37037037037037035`, within one level `0.7546296296296297`, mean absolute expected error `0.8270862784364522` level steps.
Verification statements: 352 cases, 0 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `3641.9167080312036` ms. First batch: `3759.682` ms.
Warm batch latency: `{'p50': 19.606708956416696, 'p95': 120.73410000884905, 'p99': 224.82652701437473, 'samples': 1725}`. Amortized per decision: `{'p50': 19.606708956416696, 'p95': 120.73410000884905, 'p99': 224.82652701437473}`.
Throughput including first batch: `23.342` decisions/s.
Process lifetime peak RSS: `4129226752` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.5094991364421416 |
| 0.25 | 0.6364 | 0.5468113975576662 |
| 0.5 | 0.4585 | 0.5348399246704332 |
| 0.75 | 0.3394 | 0.5292620865139949 |
| 0.8 | 0.3040 | 0.5056818181818182 |
| 0.9 | 0.2314 | 0.4552238805970149 |
| 0.95 | 0.1926 | 0.4304932735426009 |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.35,
      "mean_total_variation": 0.5761075647524047,
      "objective_accuracy_delta": -0.5851063829787234
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.4838709677419355,
      "mean_total_variation": 0.2767940642510034,
      "objective_accuracy_delta": -0.1595744680851064
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.6774193548387096,
      "mean_total_variation": 0.19830394202663715,
      "objective_accuracy_delta": -0.05319148936170213
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.7986111111111112,
      "mean_total_variation": 0.1395238544391133,
      "objective_accuracy_delta": 0.02127659574468085
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.6041666666666666,
      "mean_total_variation": 0.2827028809409505,
      "objective_accuracy_delta": -0.2127659574468085
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.8958333333333334,
      "mean_total_variation": 0.04381167630936292,
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
      "choice_stability": 0.8571428571428571,
      "mean_total_variation": 0.1318140185837026,
      "objective_accuracy_delta": 0.0
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.8402777777777778,
      "mean_total_variation": 0.0857902654642144,
      "objective_accuracy_delta": -0.010638297872340425
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.7708333333333334,
      "mean_total_variation": 0.12491071443149457,
      "objective_accuracy_delta": -0.02127659574468085
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.8402777777777778,
      "mean_total_variation": 0.09086671131958515,
      "objective_accuracy_delta": 0.0425531914893617
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.7777777777777778,
      "mean_total_variation": 0.12183877768590785,
      "objective_accuracy_delta": -0.05319148936170213
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
