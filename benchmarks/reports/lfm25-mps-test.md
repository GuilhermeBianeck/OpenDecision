# OpenDecision benchmark report

Status: **measured**. Generated 2026-09-18T19:31:40.078687+00:00.

Model `LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit`; revision `7ccafdb04c36936f4f1c4685198c6c9a40275932`; backend `lfm25`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.5958549222797928 |
| balanced_accuracy | 0.5917095750767151 |
| macro_f1 | 0.6009289163901813 |
| negative_log_likelihood | 3.0374796559156514 |
| brier_score | 0.7483842604402018 |
| ece | 0.35069165677470493 |
| coverage | 1.0 |
| selective_accuracy | 0.5958549222797928 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.4615 | 0.4307 |
| agent_control | 156 | 0.8397 | 0.1038 |
| verification | 352 | 0.5938 | 0.3979 |
| robustness | 286 | 0.6364 | 0.3313 |

## Objective accuracy by candidate count

A margin threshold is only comparable within one candidate count once a
per-count calibration profile is applied. Read coverage per row, not pooled.

| Candidates | Count | Accuracy | ECE | Coverage at margin >= 0.5 | Accuracy when answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 690 | 0.6855 | 0.2881 | 0.9623 | 0.6852 |
| 4 | 312 | 0.5224 | 0.4361 | 0.9263 | 0.5190 |
| 12 | 156 | 0.3462 | 0.4880 | 0.7436 | 0.3966 |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.9090909090909091` across 44 policy-labeled cases.
Ranking NDCG: `0.9517698209585524`. Ambiguous-case abstention rate: `0.010416666666666666`.
Ordinal rubrics: 216 cases, exact level `0.5092592592592593`, within one level `0.7777777777777778`, mean absolute expected error `0.6958749771514164` level steps.
Verification statements: 352 cases, 352 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `2385.3400830412284` ms. First batch: `2443.970` ms.
Warm batch latency: `{'p50': 46.06254200916737, 'p95': 105.71904160315171, 'p99': 255.83461318630714, 'samples': 1725}`. Amortized per decision: `{'p50': 46.06254200916737, 'p95': 105.71904160315171, 'p99': 255.83461318630714}`.
Throughput including first batch: `17.377` decisions/s.
Process lifetime peak RSS: `1154236416` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.5958549222797928 |
| 0.25 | 0.9637 | 0.6021505376344086 |
| 0.5 | 0.9231 | 0.6089803554724041 |
| 0.75 | 0.8575 | 0.6233635448136958 |
| 0.8 | 0.8411 | 0.6262833675564682 |
| 0.9 | 0.8040 | 0.6326530612244898 |
| 0.95 | 0.7573 | 0.637400228050171 |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.44166666666666665,
      "mean_total_variation": 0.5514507421011924,
      "objective_accuracy_delta": -0.40425531914893614
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.7096774193548387,
      "mean_total_variation": 0.2722924519439702,
      "objective_accuracy_delta": -0.22340425531914893
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.782258064516129,
      "mean_total_variation": 0.21695477083566897,
      "objective_accuracy_delta": -0.1276595744680851
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.9097222222222222,
      "mean_total_variation": 0.08363145024795969,
      "objective_accuracy_delta": -0.0425531914893617
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.7847222222222222,
      "mean_total_variation": 0.19733352542547541,
      "objective_accuracy_delta": -0.11702127659574468
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.9652777777777778,
      "mean_total_variation": 0.03417214068381201,
      "objective_accuracy_delta": -0.02127659574468085
    },
    "option_rotation": {
      "pairs": 94,
      "choice_stability": 0.7978723404255319,
      "mean_total_variation": 0.17774114094049406,
      "objective_accuracy_delta": 0.03225806451612903
    },
    "question_rewording": {
      "pairs": 112,
      "choice_stability": 0.9375,
      "mean_total_variation": 0.07764620337572745,
      "objective_accuracy_delta": -0.03225806451612903
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.9305555555555556,
      "mean_total_variation": 0.06289674443744026,
      "objective_accuracy_delta": 0.031914893617021274
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.8472222222222222,
      "mean_total_variation": 0.16415877384231192,
      "objective_accuracy_delta": -0.010638297872340425
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.9166666666666666,
      "mean_total_variation": 0.07058749508276495,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.8611111111111112,
      "mean_total_variation": 0.11214513489657497,
      "objective_accuracy_delta": -0.06382978723404255
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
