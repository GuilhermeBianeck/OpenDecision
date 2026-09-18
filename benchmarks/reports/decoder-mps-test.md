# OpenDecision benchmark report

Status: **measured**. Generated 2026-09-18T15:08:14.276984+00:00.

Model `Qwen/Qwen3-0.6B`; revision `c1899de289a04d12100db370d81485cdf75e47ca`; backend `decoder`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.6079447322970639 |
| balanced_accuracy | 0.6272225689568489 |
| macro_f1 | 0.6471292938977479 |
| negative_log_likelihood | 1.8881659100423716 |
| brier_score | 0.6697710198468554 |
| ece | 0.27709015624402056 |
| coverage | 1.0 |
| selective_accuracy | 0.6079447322970639 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.6181 | 0.2696 |
| agent_control | 156 | 0.4872 | 0.3474 |
| verification | 352 | 0.6960 | 0.1896 |
| robustness | 286 | 0.5524 | 0.3903 |

## Objective accuracy by candidate count

A margin threshold is only comparable within one candidate count once a
per-count calibration profile is applied. Read coverage per row, not pooled.

| Candidates | Count | Accuracy | ECE | Coverage at margin >= 0.5 | Accuracy when answered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 690 | 0.6304 | 0.2647 | 0.7957 | 0.6284 |
| 4 | 312 | 0.5128 | 0.3907 | 0.7917 | 0.5061 |
| 12 | 156 | 0.6987 | 0.1453 | 0.7756 | 0.7769 |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.75` across 44 policy-labeled cases.
Ranking NDCG: `0.9815883087928584`. Ambiguous-case abstention rate: `0.09375`.
Ordinal rubrics: 216 cases, exact level `0.5092592592592593`, within one level `0.8425925925925926`, mean absolute expected error `0.64130579938408` level steps.
Verification statements: 352 cases, 352 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `2663.147665967699` ms. First batch: `2838.806` ms.
Warm batch latency: `{'p50': 65.8712909789756, 'p95': 127.07796636968851, 'p99': 131.29254790954292, 'samples': 1725}`. Amortized per decision: `{'p50': 65.8712909789756, 'p95': 127.07796636968851, 'p99': 131.29254790954292}`.
Throughput including first batch: `13.536` decisions/s.
Process lifetime peak RSS: `1143095296` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.6079447322970639 |
| 0.25 | 0.8964 | 0.617533718689788 |
| 0.5 | 0.7919 | 0.6150490730643402 |
| 0.75 | 0.6563 | 0.6131578947368421 |
| 0.8 | 0.6209 | 0.6175243393602226 |
| 0.9 | 0.5363 | 0.6215780998389694 |
| 0.95 | 0.4568 | 0.6370510396975425 |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.35833333333333334,
      "mean_total_variation": 0.59382485997782,
      "objective_accuracy_delta": -0.5957446808510638
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.7096774193548387,
      "mean_total_variation": 0.27373511201592243,
      "objective_accuracy_delta": -0.10638297872340426
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.717741935483871,
      "mean_total_variation": 0.245841383745178,
      "objective_accuracy_delta": -0.0425531914893617
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.9305555555555556,
      "mean_total_variation": 0.09212820067086859,
      "objective_accuracy_delta": -0.010638297872340425
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.8055555555555556,
      "mean_total_variation": 0.17027193218074552,
      "objective_accuracy_delta": 0.031914893617021274
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.9375,
      "mean_total_variation": 0.049745531847710835,
      "objective_accuracy_delta": 0.0
    },
    "option_rotation": {
      "pairs": 94,
      "choice_stability": 0.8297872340425532,
      "mean_total_variation": 0.15794343096531635,
      "objective_accuracy_delta": 0.04838709677419355
    },
    "question_rewording": {
      "pairs": 112,
      "choice_stability": 0.8928571428571429,
      "mean_total_variation": 0.09255481513218379,
      "objective_accuracy_delta": 0.06451612903225806
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.9166666666666666,
      "mean_total_variation": 0.10234420912871577,
      "objective_accuracy_delta": 0.031914893617021274
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.8958333333333334,
      "mean_total_variation": 0.10282234912401761,
      "objective_accuracy_delta": 0.010638297872340425
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.9236111111111112,
      "mean_total_variation": 0.07584934066895449,
      "objective_accuracy_delta": 0.031914893617021274
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.9097222222222222,
      "mean_total_variation": 0.09320826704611748,
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
