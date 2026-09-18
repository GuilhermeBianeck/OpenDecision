# OpenDecision benchmark report

Status: **measured**. Generated 2026-09-18T09:23:19.448752+00:00.

Model `cross-encoder/nli-deberta-v3-xsmall`; revision `a150876415327c80daeff35ca6f68f5ed8cf5c24`; backend `tiny`; device `cpu`; batch 1.

Dataset: 24 decisions, 24 underlying scenarios, 6 groups, split `validation`. SHA-256 `5e6c6705697e6614ae0b0acc157621a88adc434311aa15d5821fcefac8299813`.

## Objective tasks

| Metric | Value |
| --- | ---: |
| count | 12 |
| accuracy | 1 |
| balanced_accuracy | 1.0 |
| macro_f1 | 1.0 |
| negative_log_likelihood | 0.19435083085435698 |
| brier_score | 0.07493977497714212 |
| ece | 0.17164114661316188 |
| coverage | 1.0 |
| selective_accuracy | 1 |

## Separate evaluation families

Policy agreement (not moral accuracy): `0.75` across 4 policy-labeled cases.
Ranking NDCG: `0.9729553249874899`. Ambiguous-case abstention rate: `0.5`.

## Timing

Measurement: **in-process Python SDK; includes serialization and scoring**.
Cold model load: `3108.0371669959277` ms. First batch: `3178.486` ms.
Warm batch latency: `{'p50': 29.159083031117916, 'p95': 37.61255783610977, 'p99': 38.488249523798004, 'samples': 23}`. Amortized per decision: `{'p50': 29.159083031117916, 'p95': 37.61255783610977, 'p99': 38.488249523798004}`.
Throughput including first batch: `6.186` decisions/s.
Process lifetime peak RSS: `597180416` bytes; this is not isolated model memory.

## Selective answering

Thresholds are top-one minus top-two margins.

| Minimum margin | Coverage | Accuracy when answered |
| --- | ---: | ---: |
| 0.0 | 1.0000 | 1 |
| 0.25 | 1.0000 | 1 |
| 0.5 | 0.8333 | 1 |
| 0.75 | 0.5000 | 1 |
| 0.8 | 0.3333 | 1 |
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
