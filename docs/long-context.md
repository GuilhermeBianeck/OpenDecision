# Long states

The decoder backends read the state once and score every option against it,
so a long state costs one long prefill. The MLX backend encodes long prompts in
2,048-token chunks, which keeps activation memory flat as the state grows.
Each backend accepts the context it has been measured to fit on a 16 GB
machine alongside a desktop session; the checkpoints themselves support
262,144 tokens.

| Backend | Context limit | Measured peak |
| --- | ---: | ---: |
| `qwen35`, `qwen35_4b` | 32,768 | 2.41 GB (`qwen35` at 32k) |
| `lightning` | 16,384 | 4.09 GB at 16k |

Pass `max_length` (or `--max-length`) to use more than the default context.

## Accuracy and latency by length

`scripts/long_context_curve.py` hides a labelled test-split state inside
irrelevant archive notes at a seeded position (start, middle or end) and asks
its question at each length. The same 24 cases are used at every length, so
the curve isolates the effect of length; the report with every row is in
[`benchmarks/reports/long-context/`](../benchmarks/reports/long-context/).

`qwen35`:

| State tokens | Accuracy | State at start | Middle | End | p50 latency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2k | 0.708 | 0.88 | 0.75 | 0.50 | 0.9 s |
| 8k | 0.667 | 0.75 | 0.88 | 0.38 | 3.2 s |
| 32k | 0.542 | 0.62 | 0.75 | 0.25 | 16.3 s |

Accuracy falls with length, and most of the loss is when the relevant record
comes last, after the filler. Latency grows roughly linearly with the state.
Each position holds only eight cases, so read the position columns as a
direction, not a measurement.

`lightning`:

| State tokens | Accuracy | State at start | Middle | End | p50 latency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2k | 0.667 | 0.88 | 0.62 | 0.50 | 1.4 s |
| 8k | 0.625 | 0.88 | 0.50 | 0.50 | 6.0 s |
| 16k | 0.625 | 0.88 | 0.50 | 0.50 | 13.0 s |

`lightning` loses one answer between 2k and 8k and none after. Its record at
the start is found at every length, and it is no better than `qwen35` here,
unlike on the public benchmark: the two differ by one answer at 2k, within
the noise of 24 cases. Its earlier runs on a busier machine were slower (30 s
at 16k) because of memory pressure; this curve was measured with other
applications closed.

## What this means in practice

- Long states work, but they are slow: a 32k-token state takes about 16 s on
  `qwen35` and a 16k-token state about 13 s on `lightning` on the reference
  machine.
- Put the record the question is about first, or extract it, rather than
  appending it after long background material.
- On short and medium states `lightning` remains the stronger backend (see
  [public benchmark](public-benchmark.md)); this probe's filler is unlike its
  training data, which is one reason to measure on your own documents.
