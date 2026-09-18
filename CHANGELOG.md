# Changelog

## Unreleased

- Fitted calibration profiles for the MLX backends. `qwen35_4b` (ECE 0.195 to
  0.073) and `lfm25` (0.210 to 0.163) are committed; `qwen35` is deliberately
  left uncalibrated because its uncalibrated ECE of 0.061 is already the best
  measured and a fitted temperature regresses all three metrics.
- Measured the MLX candidates on the committed test split and made the evidence
  part of the repository. `qwen35` reaches 0.799 pooled objective accuracy and
  0.084 ECE against 0.675 and 0.102 for the previous best, and `qwen35_4b`
  reaches 0.853 with 0.769 exact-level rubric scoring. `auto` selects `qwen35`
  on Apple silicon and resolves identically for loading and pulling. The routing
  table records the new measurements and names a portable alternative wherever
  it recommends an MLX backend.
- Added `opendecision.routing`, a measured table of which backend suits which
  question shape, reachable as `DecisionModel(task=...)`, `opendecision tasks`
  and `--task`. Added `choose_wide`, which scores candidate sets larger than a
  backend scores in one request by elimination, and `DecisionModel.max_choices`.
- `device="auto"` now resolves precision as well: bfloat16 on a GPU, float32 on
  CPU. Measured quality-neutral on the test split and 2.1–3.0× faster. Adds
  `--precision` to the CLI, reports `recommended_precision` from `doctor`, and
  ships calibration profiles for both precisions.
- Added `opendecision.guards.screen` for detecting instructions planted in a
  state, and `docs/patterns.md` documenting the measured composition patterns:
  decomposing a conditional rule raises accuracy on negated facts from 0.533 to
  0.933, and screening flags 97.5 % of bracketed injections at 0.8 % false
  positives.
- Committed calibration profiles for `smart`, `multilingual` and `decoder`, and
  calibrated test reports for `base` and `decoder`. Benchmark reports gained
  `objective_by_choice_count`, because a per-count profile makes one margin
  threshold incomparable between candidate counts.
- Calibration profiles fit one temperature per candidate count by default, with
  the pooled temperature as the fallback for counts below `min_rows_per_count`;
  results report `metadata.calibration_temperature`. `opendecision calibrate`
  scores in batches and reports the held-out effect on the validation split.
- Committed a five-backend report matrix on the expanded corpus test split
  (`benchmarks/reports/*-mps-test.json`) and `scripts/compare_reports.py`, which
  renders reports side by side into `benchmarks/reports/README.md`.
- Benchmark corpus expanded from 232 to 538 seeds (2,320 to 6,578 rows): templated
  verification statements over authored records, negation pairs, embedded and
  authority-styled injections, twelve-option routing, six ordinal rubrics, and
  same-domain distractor variants.
- Benchmark cases carry a `kind` (`choice`, `boolean`, `score`); the runner
  evaluates each through its primitive, reports `verification`, `robustness` and
  `ordinal` families, per-family and per-variant objective metrics, and ordinal
  level-error metrics.
- New `decoder` backend (pinned Qwen3 0.6B, Apache-2.0): encodes each state once
  into the KV cache and scores options by the next-token log-probability of
  their letter, so many options and many questions per state cost little extra.
  Supports statements as a true/unknown/false question and optional rotation
  averaging (`permutations`).
- `device="auto"` now resolves to MPS on Apple Silicon (after CUDA, before CPU),
  based on a published 1.2–3.5× measurement. The default `max_length` is the
  model's context capped at 2,048 tokens instead of a fixed 512.
- `ask` answers independent typed questions (choice, boolean, score) about one
  state under caller-chosen ids, with per-question thresholds (Python, HTTP
  `/v1/ask`, CLI, TypeScript). Results carry a `type` discriminator. The HTTP
  multi-label limit now matches the Python limit of 128 labels.
- Choices may carry a description, a `not_for` boundary and examples
  (`ChoiceOption`); results stay keyed by label. State may be a record or a list
  of texts and is rendered deterministically before scoring. CLI accepts
  `--state-json` and `--choices-json`.
- `score` rates the state on an ordered rubric of 2–10 described levels and
  returns the most probable level, the distribution, a legend, and the
  probability-weighted level index (Python, HTTP `/v1/score`, CLI, TypeScript).
- Booleans and multi-label questions are scored as statements on NLI backends:
  yes/no is entailment against contradiction, `unsupported` reports the neutral
  share, and `unsupported_threshold` abstains on it. Other backends keep the
  two-way choice and report `method: "binary_choice"`.
- Calibration profiles record the range of candidate counts they were fitted on;
  results report `metadata.calibration_covers_choice_count`.

## 0.1.0a1 — 2026-09-18

Initial public alpha groundwork:

- local Python decision SDK with choice, boolean, ranking, multi-label, and batch APIs;
- raw, normalized, calibrated, and abstaining results with explicit confidence semantics;
- pinned NLI, reward, reranker, demo, and experimental ONNX backend contracts;
- FastAPI loopback server and dependency-free TypeScript client;
- grouped synthetic benchmark corpus, metrics, robustness probes, and JSON/Markdown reports;
- optional, explicitly invoked generative LLM baseline adapter;
- model-license notes, CI, Docker examples, and contributor/security documentation.

This release makes no universal accuracy, safety, or latency claim.
