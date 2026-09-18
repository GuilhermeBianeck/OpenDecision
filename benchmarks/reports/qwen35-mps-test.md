# OpenDecision benchmark report

Status: **measured**. Generated 2026-09-18T19:29:58.060628+00:00.

Model `mlx-community/Qwen3.5-2B-4bit`; revision `674aaa7240b91e8012fcad5d791b7dfe5ba90207`; backend `qwen35`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.7987910189982729 |
| balanced_accuracy | 0.7291985229104905 |
| macro_f1 | 0.7341705724392923 |
| negative_log_likelihood | 0.580265711988507 |
| brier_score | 0.29858852391591817 |
| ece | 0.08436448681730109 |
| coverage | 1.0 |
| selective_accuracy | 0.7987910189982729 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.8132 | 0.1333 |
| agent_control | 156 | 0.8141 | 0.0871 |
| verification | 352 | 0.9886 | 0.0197 |
| robustness | 286 | 0.5385 | 0.2689 |

## Objective accuracy by candidate count

A margin threshold is only comparable within one candidate count once a
per-count calibration profile is applied. Read coverage per row, not pooled.

| Candidates | Count | Accuracy | ECE | Coverage at margin >= 0.5 | Accuracy when answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 690 | 0.8406 | 0.0612 | 0.8188 | 0.9027 |
| 4 | 312 | 0.7276 | 0.1542 | 0.8269 | 0.7481 |
| 12 | 156 | 0.7564 | 0.1670 | 0.6603 | 0.9126 |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.8863636363636364` across 44 policy-labeled cases.
Ranking NDCG: `0.9843621216146543`. Ambiguous-case abstention rate: `0.09375`.
Ordinal rubrics: 216 cases, exact level `0.48148148148148145`, within one level `0.8564814814814815`, mean absolute expected error `0.6550521290322862` level steps.
Verification statements: 352 cases, 352 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `2296.789665997494` ms. First batch: `2379.523` ms.
Warm batch latency: `{'p50': 69.46983304806054, 'p95': 166.33781655691564, 'p99': 280.8026361628434, 'samples': 1725}`. Amortized per decision: `{'p50': 69.46983304806054, 'p95': 166.33781655691564, 'p99': 280.8026361628434}`.
Throughput including first batch: `11.857` decisions/s.
Process lifetime peak RSS: `1956642816` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.7987910189982729 |
| 0.25 | 0.8843 | 0.837890625 |
| 0.5 | 0.7997 | 0.8606911447084233 |
| 0.75 | 0.6442 | 0.8994638069705094 |
| 0.8 | 0.6002 | 0.9093525179856116 |
| 0.9 | 0.4957 | 0.9407665505226481 |
| 0.95 | 0.4162 | 0.950207468879668 |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.39166666666666666,
      "mean_total_variation": 0.5295412080518594,
      "objective_accuracy_delta": -0.48936170212765956
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.7741935483870968,
      "mean_total_variation": 0.2005801484062824,
      "objective_accuracy_delta": -0.13829787234042554
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.8145161290322581,
      "mean_total_variation": 0.14616328174959073,
      "objective_accuracy_delta": -0.05319148936170213
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.875,
      "mean_total_variation": 0.08555769022553383,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.8472222222222222,
      "mean_total_variation": 0.10763605319412525,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.9652777777777778,
      "mean_total_variation": 0.026723939112685762,
      "objective_accuracy_delta": 0.0
    },
    "option_rotation": {
      "pairs": 94,
      "choice_stability": 0.8404255319148937,
      "mean_total_variation": 0.129097677000883,
      "objective_accuracy_delta": 0.0
    },
    "question_rewording": {
      "pairs": 112,
      "choice_stability": 0.9375,
      "mean_total_variation": 0.05673736406977693,
      "objective_accuracy_delta": 0.0
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.9166666666666666,
      "mean_total_variation": 0.06099099024736614,
      "objective_accuracy_delta": 0.0
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.8541666666666666,
      "mean_total_variation": 0.16246613512891808,
      "objective_accuracy_delta": -0.0425531914893617
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.8958333333333334,
      "mean_total_variation": 0.0661600297692458,
      "objective_accuracy_delta": 0.0
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.8958333333333334,
      "mean_total_variation": 0.06813276215848713,
      "objective_accuracy_delta": 0.02127659574468085
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
