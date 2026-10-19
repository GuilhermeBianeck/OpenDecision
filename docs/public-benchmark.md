# Public decision benchmark

A public, MIT-licensed decision benchmark publishes 231 of its items in three
tiers: `original` (72 short hand-written decisions), `easy` (48) and `hard`
(111 multi-step policy, numeric, temporal and probability decisions, some with
states near 4,000 tokens). Its harness speaks `POST /v1/systemone`, so it runs
against `opendecision serve` unchanged. This page records OpenDecision's
results on those 231 items. Reports with every prediction are in
[`benchmarks/reports/public/`](../benchmarks/reports/public/).

## Method

- The harness's own `/v1/systemone` adapter, unmodified, at harness commit
  `609a664`; each report records the SHA-256 of every item file.
- A local server, one request at a time, on a 16 GB Apple-silicon Mac.
  Latency is the harness's caller-side wall time.
- An invalid or failed answer counts as wrong. None occurred.
- *Chance-corrected* accuracy is `(accuracy − chance) / (1 − chance)`, with
  chance the mean of `1 / options` over a tier's items, so tiers with
  different option counts compare fairly. ECE is top-label, 10 equal bins.
- `scripts/public_benchmark.py` turns the harness's per-tier results into the
  committed reports.

## Results

`qwen35`, the general 2B backend and the Apple-silicon fallback, with its
default 2,048-token context:

| Tier | Correct | Accuracy | Chance | Chance-corrected | ECE | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 51 / 72 | 70.8 % | 31.1 % | 57.7 % | 0.125 | 62 ms | 64 ms |
| easy | 47 / 48 | 97.9 % | 28.4 % | 97.1 % | 0.043 | 63 ms | 75 ms |
| hard | 41 / 111 | 36.9 % | 33.6 % | 5.0 % | 0.311 | 355 ms | 740 ms |
| **all** | **139 / 231** | **60.2 %** | 31.8 % | 41.6 % | 0.177 | 75 ms | 636 ms |

The same backend with `--max-length 8192`, so that no hard-tier state is
truncated, scores 137 / 231 (59.3 %): hard drops to 39 / 111 and its p50 rises
to 799 ms. Truncation is not what limits the hard tier.

`lightning`, the decision-trained 4B backend and the Apple-silicon default, at
its default 8,192-token context and without a calibration profile:

| Tier | Correct | Accuracy | Chance-corrected | ECE | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 72 / 72 | 100 % | 100 % | 0.135 | 154 ms | 167 ms |
| easy | 48 / 48 | 100 % | 100 % | 0.087 | 150 ms | 161 ms |
| hard | 84 / 111 | 75.7 % | 63.4 % | 0.107 | 508 ms | 2,756 ms |
| **all** | **204 / 231** | **88.3 %** | 82.9 % | 0.105 | 166 ms | 2,082 ms |

At 2,048 tokens it scores 197 / 231 (hard 77 / 111): unlike `qwen35`, it uses
the rest of a long state, which is why its default context is 8,192 tokens.
Peak memory was 4.46 GB and free memory on the 16 GB machine never fell below
46 % during the run.

## Reading the result

- On short decisions `qwen35` is fast and well calibrated: the easy
  tier is near ceiling with ECE 0.043.
- On the hard tier it is barely above chance (5 % chance-corrected) and
  confidently wrong (ECE 0.311). Multi-hop, temporal-numeric and probability
  items score 22 %, 20 % and 10 %. A 2B general model read off one forward
  pass does not do this arithmetic and rule-chaining.
- Giving it the whole state does not help, so the remedy is a stronger
  decision-trained checkpoint, not a longer context.
- `lightning` is that checkpoint. It clears milestone M3's bar of 85 % on these
  231 items with 88.3 %, answers 63 % of the hard tier above chance where
  `qwen35` answers 5 %, and keeps a warm median under 250 ms.

A partial run of `qwen35_4b` (56 / 72, 48 / 48 and 38 of the first 87 hard
items) was stopped when the MLX buffer cache grew past what a 16 GB machine
can spare alongside a desktop session; it is not reported as a result. The MLX
backend now bounds that cache, and backends refuse to load when free memory is
below their measured peak.
