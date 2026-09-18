# OpenDecision benchmark report

Status: **measured**. Generated 2026-09-18T11:28:24.765223+00:00.

Model `cross-encoder/nli-deberta-v3-xsmall`; revision `a150876415327c80daeff35ca6f68f5ed8cf5c24`; backend `tiny`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.575993091537133 |
| balanced_accuracy | 0.4377325289089995 |
| macro_f1 | 0.47244331208111184 |
| negative_log_likelihood | 0.9979642949731785 |
| brier_score | 0.5312360446861116 |
| ece | 0.12674646567262748 |
| coverage | 1.0 |
| selective_accuracy | 0.575993091537133 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.3626 | 0.1036 |
| agent_control | 156 | 0.6282 | 0.1844 |
| verification | 352 | 0.9318 | 0.0375 |
| robustness | 286 | 0.3811 | 0.3094 |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.9090909090909091` across 44 policy-labeled cases.
Ranking NDCG: `0.9402876283949687`. Ambiguous-case abstention rate: `0.3333333333333333`.
Ordinal rubrics: 216 cases, exact level `0.35185185185185186`, within one level `0.8703703703703703`, mean absolute expected error `0.7259795579221939` level steps.
Verification statements: 352 cases, 352 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `3635.728334018495` ms. First batch: `3880.893` ms.
Warm batch latency: `{'p50': 24.49024998350069, 'p95': 195.3833997715264, 'p99': 381.5758800157345, 'samples': 1725}`. Amortized per decision: `{'p50': 24.49024998350069, 'p95': 195.3833997715264, 'p99': 381.5758800157345}`.
Throughput including first batch: `19.466` decisions/s.
Process lifetime peak RSS: `1066680320` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.575993091537133 |
| 0.25 | 0.5924 | 0.7113702623906706 |
| 0.5 | 0.4620 | 0.7663551401869159 |
| 0.75 | 0.3446 | 0.8446115288220551 |
| 0.8 | 0.3135 | 0.8787878787878788 |
| 0.9 | 0.2565 | 0.936026936026936 |
| 0.95 | 0.2280 | 0.946969696969697 |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.4666666666666667,
      "mean_total_variation": 0.3797307622618999,
      "objective_accuracy_delta": -0.3404255319148936
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.5967741935483871,
      "mean_total_variation": 0.1421906912838165,
      "objective_accuracy_delta": -0.14893617021276595
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.5967741935483871,
      "mean_total_variation": 0.13414986985889849,
      "objective_accuracy_delta": -0.09574468085106383
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.875,
      "mean_total_variation": 0.06388681190234197,
      "objective_accuracy_delta": -0.02127659574468085
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.6597222222222222,
      "mean_total_variation": 0.14302060912909248,
      "objective_accuracy_delta": -0.11702127659574468
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.9444444444444444,
      "mean_total_variation": 0.0419853293782042,
      "objective_accuracy_delta": -0.05319148936170213
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
      "mean_total_variation": 0.10222630706861323,
      "objective_accuracy_delta": -0.11290322580645161
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.8680555555555556,
      "mean_total_variation": 0.05813148665409073,
      "objective_accuracy_delta": -0.09574468085106383
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.6805555555555556,
      "mean_total_variation": 0.16361837543288768,
      "objective_accuracy_delta": -0.0425531914893617
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.8541666666666666,
      "mean_total_variation": 0.07130544046744604,
      "objective_accuracy_delta": -0.09574468085106383
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.8402777777777778,
      "mean_total_variation": 0.09539822953186308,
      "objective_accuracy_delta": -0.031914893617021274
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
