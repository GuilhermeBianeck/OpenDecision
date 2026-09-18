# OpenDecision benchmark report

Status: **measured**. Generated 2026-09-18T13:25:31.074198+00:00.

Model `Skywork/Skywork-Reward-V2-Qwen3-0.6B`; revision `8c14a4e9e6321deaf572544339b16b8d6bbe8886`; backend `smart`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.6157167530224525 |
| balanced_accuracy | 0.5504195488982507 |
| macro_f1 | 0.5489673101729242 |
| negative_log_likelihood | 0.9632259216881456 |
| brier_score | 0.5231506339886972 |
| ece | 0.08588530342327463 |
| coverage | 1.0 |
| selective_accuracy | 0.6157167530224525 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.5330 | 0.0623 |
| agent_control | 156 | 0.6346 | 0.1676 |
| verification | 352 | 0.8097 | 0.1229 |
| robustness | 286 | 0.4720 | 0.3196 |

## Objective accuracy by candidate count

A margin threshold is only comparable within one candidate count once a
per-count calibration profile is applied. Read coverage per row, not pooled.

| Candidates | Count | Accuracy | ECE | Coverage at margin >= 0.5 | Accuracy when answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 690 | 0.6710 | 0.1287 | 0.4986 | 0.6831 |
| 4 | 312 | 0.6699 | 0.0584 | 0.5288 | 0.8364 |
| 12 | 156 | 0.2628 | 0.0645 | 0.0128 | 0.0000 |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.7272727272727273` across 44 policy-labeled cases.
Ranking NDCG: `0.9257251110805402`. Ambiguous-case abstention rate: `0.08333333333333333`.
Ordinal rubrics: 216 cases, exact level `0.6064814814814815`, within one level `0.8981481481481481`, mean absolute expected error `0.49086262667878566` level steps.
Verification statements: 352 cases, 0 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `4258.13879200723` ms. First batch: `4586.260` ms.
Warm batch latency: `{'p50': 77.82404101453722, 'p95': 483.8485503802076, 'p99': 996.6331449733116, 'samples': 1725}`. Amortized per decision: `{'p50': 77.82404101453722, 'p95': 483.8485503802076, 'p99': 996.6331449733116}`.
Throughput including first batch: `5.681` decisions/s.
Process lifetime peak RSS: `3681386496` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.6157167530224525 |
| 0.25 | 0.6520 | 0.7112582781456953 |
| 0.5 | 0.4413 | 0.7299412915851272 |
| 0.75 | 0.2142 | 0.7137096774193549 |
| 0.8 | 0.1805 | 0.722488038277512 |
| 0.9 | 0.1105 | 0.8046875 |
| 0.95 | 0.0864 | 0.86 |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.43333333333333335,
      "mean_total_variation": 0.3866267216694318,
      "objective_accuracy_delta": -0.4148936170212766
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.7338709677419355,
      "mean_total_variation": 0.2230705813479885,
      "objective_accuracy_delta": -0.10638297872340426
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.7419354838709677,
      "mean_total_variation": 0.19400356656026727,
      "objective_accuracy_delta": -0.09574468085106383
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.8055555555555556,
      "mean_total_variation": 0.09487465822910308,
      "objective_accuracy_delta": 0.0
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.7777777777777778,
      "mean_total_variation": 0.17939466409862526,
      "objective_accuracy_delta": -0.02127659574468085
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.9097222222222222,
      "mean_total_variation": 0.07076370220766404,
      "objective_accuracy_delta": 0.0
    },
    "option_rotation": {
      "pairs": 94,
      "choice_stability": 1,
      "mean_total_variation": 0.0,
      "objective_accuracy_delta": 0.0
    },
    "question_rewording": {
      "pairs": 112,
      "choice_stability": 0.8035714285714286,
      "mean_total_variation": 0.10826512097880868,
      "objective_accuracy_delta": 0.016129032258064516
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.7916666666666666,
      "mean_total_variation": 0.1280875072768248,
      "objective_accuracy_delta": 0.07446808510638298
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.7638888888888888,
      "mean_total_variation": 0.14416273709321512,
      "objective_accuracy_delta": -0.010638297872340425
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.8194444444444444,
      "mean_total_variation": 0.11818940858727536,
      "objective_accuracy_delta": 0.05319148936170213
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.8055555555555556,
      "mean_total_variation": 0.1180088206195507,
      "objective_accuracy_delta": -0.02127659574468085
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
