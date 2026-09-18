# Changelog

## Unreleased

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
