# OpenDecision benchmark report

Status: **measured**. Generated 2026-09-18T11:30:13.344623+00:00.

Model `tasksource/ModernBERT-base-nli`; revision `de4ab7e77845098b7fab7f6ab9d370ddff27b19c`; backend `base`; device `mps`; batch 1.

Dataset: 1726 decisions, 144 underlying scenarios, 25 groups, split `test`. SHA-256 `43c4923ea72329e0f060e88cfb10f7418e89551e29deee5edb1fef7bbeaed5a4`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 1158 |
| accuracy | 0.6753022452504318 |
| balanced_accuracy | 0.5679790572285501 |
| macro_f1 | 0.5933748600731071 |
| negative_log_likelihood | 0.9123701243922143 |
| brier_score | 0.4806206279535609 |
| ece | 0.14881328992442416 |
| coverage | 1.0 |
| selective_accuracy | 0.6753022452504318 |

## Objective accuracy by family

| Family | Count | Accuracy | ECE |
| --- | ---: | ---: | ---: |
| objective | 364 | 0.5632 | 0.2809 |
| agent_control | 156 | 0.7564 | 0.2696 |
| verification | 352 | 0.9659 | 0.0413 |
| robustness | 286 | 0.4161 | 0.3060 |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.9090909090909091` across 44 policy-labeled cases.
Ranking NDCG: `0.8738919743718051`. Ambiguous-case abstention rate: `0.5729166666666666`.
Ordinal rubrics: 216 cases, exact level `0.4166666666666667`, within one level `0.8518518518518519`, mean absolute expected error `0.6947327910896055` level steps.
Verification statements: 352 cases, 352 scored as statements.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `3551.501707988791` ms. First batch: `3770.941` ms.
Warm batch latency: `{'p50': 28.720042027998716, 'p95': 183.38452499592677, 'p99': 383.4154679486528, 'samples': 1725}`. Amortized per decision: `{'p50': 28.720042027998716, 'p95': 183.38452499592677, 'p99': 383.4154679486528}`.
Throughput including first batch: `16.041` decisions/s.
Process lifetime peak RSS: `652263424` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.6753022452504318 |
| 0.25 | 0.5846 | 0.7872968980797637 |
| 0.5 | 0.4715 | 0.8296703296703297 |
| 0.75 | 0.3601 | 0.8729016786570744 |
| 0.8 | 0.3307 | 0.8929503916449086 |
| 0.9 | 0.2280 | 0.9772727272727273 |
| 0.95 | 0.1641 | 1 |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {
    "authority_injection": {
      "pairs": 120,
      "choice_stability": 0.5333333333333333,
      "mean_total_variation": 0.22091226791390722,
      "objective_accuracy_delta": -0.35106382978723405
    },
    "distractor_middle": {
      "pairs": 124,
      "choice_stability": 0.6774193548387096,
      "mean_total_variation": 0.13338783282802677,
      "objective_accuracy_delta": -0.14893617021276595
    },
    "distractor_start": {
      "pairs": 124,
      "choice_stability": 0.7016129032258065,
      "mean_total_variation": 0.11242930670868335,
      "objective_accuracy_delta": -0.10638297872340426
    },
    "irrelevant_context": {
      "pairs": 144,
      "choice_stability": 0.8958333333333334,
      "mean_total_variation": 0.052139555736247,
      "objective_accuracy_delta": -0.031914893617021274
    },
    "long_context": {
      "pairs": 144,
      "choice_stability": 0.7222222222222222,
      "mean_total_variation": 0.16339125501126942,
      "objective_accuracy_delta": -0.19148936170212766
    },
    "lowercase": {
      "pairs": 144,
      "choice_stability": 0.9236111111111112,
      "mean_total_variation": 0.04085500723367151,
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
      "choice_stability": 0.8303571428571429,
      "mean_total_variation": 0.07811201036314543,
      "objective_accuracy_delta": -0.08064516129032258
    },
    "quoted_state": {
      "pairs": 144,
      "choice_stability": 0.8541666666666666,
      "mean_total_variation": 0.052043008888945474,
      "objective_accuracy_delta": -0.031914893617021274
    },
    "state_injection": {
      "pairs": 144,
      "choice_stability": 0.7986111111111112,
      "mean_total_variation": 0.10181660230778931,
      "objective_accuracy_delta": 0.0
    },
    "unicode_context": {
      "pairs": 144,
      "choice_stability": 0.8541666666666666,
      "mean_total_variation": 0.04924470832947313,
      "objective_accuracy_delta": -0.0425531914893617
    },
    "uppercase": {
      "pairs": 144,
      "choice_stability": 0.7222222222222222,
      "mean_total_variation": 0.12041277000434467,
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
