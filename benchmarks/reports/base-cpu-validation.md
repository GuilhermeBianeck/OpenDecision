# OpenDecision benchmark report

Status: **measured**. Generated 2026-09-18T09:28:37.676395+00:00.

Model `tasksource/ModernBERT-base-nli`; revision `de4ab7e77845098b7fab7f6ab9d370ddff27b19c`; backend `base`; device `cpu`; batch 1.

Dataset: 24 decisions, 24 underlying scenarios, 7 groups, split `validation`. SHA-256 `7b25ab46b108bf7098e83271d4c973f39d3b6f9f2f19237c7e62711bf3bf8541`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 14 |
| accuracy | 0.9285714285714286 |
| balanced_accuracy | 0.9166666666666666 |
| macro_f1 | 0.9523809523809523 |
| negative_log_likelihood | 0.4395989080804411 |
| brier_score | 0.22055264012520973 |
| ece | 0.22712455559277114 |
| coverage | 1.0 |
| selective_accuracy | 0.9285714285714286 |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.6666666666666666` across 3 policy-labeled cases.
Ranking NDCG: `0.7704903730198444`. Ambiguous-case abstention rate: `0.5`.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `3529.4529999955557` ms. First batch: `3636.932` ms.
Warm batch latency: `{'p50': 63.51541599724442, 'p95': 88.36274999193847, 'p99': 89.0592224872671, 'samples': 23}`. Amortized per decision: `{'p50': 63.51541599724442, 'p95': 88.36274999193847, 'p99': 89.0592224872671}`.
Throughput including first batch: `4.596` decisions/s.
Process lifetime peak RSS: `885669888` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 0.9285714285714286 |
| 0.25 | 0.7143 | 1 |
| 0.5 | 0.5714 | 1 |
| 0.75 | 0.2143 | 1 |
| 0.8 | 0.1429 | 1 |
| 0.9 | 0.0000 | None |
| 0.95 | 0.0000 | None |

## Robustness

```json
{
  "choice_order": null,
  "self_consistency": null,
  "perturbations": {}
}
```

## Limitations

- Synthetic seed scenarios and deterministic perturbations are correlated, not independently human-validated samples.
- Held-out semantic groups reduce template leakage but do not establish real-world generalization.
- Abstention threshold is a test configuration, not a validated safety guarantee.
- A limited run can omit labels and robustness pairs; AUROC is null where undefined.

The companion JSON contains probabilities, raw scores, per-case choice-order probes, reliability bins, confusion matrices, valid-vocabulary AUROC and full runtime metadata.
