# OpenDecision progress

Updated 18 September 2026.

## Completed

- Bootstrapped the installable `opendecision` package and MIT-licensed repository.
- Added Pydantic schemas, normalization, confidence margin, calibration, abstention,
  independent multi-label decisions, ranking, and bounded batch APIs.
- Added pinned DeBERTa, ModernBERT, Skywork Reward, BGE reranker, deterministic
  demo, and experimental content-addressed ONNX backends.
- Added explicit model download and offline-only inference (`local_files_only`,
  safetensors, remote code disabled), model-license records, and device diagnostics.
- Added FastAPI server with resident startup load, loopback default, request/body
  bounds, concurrency gate, health/models endpoints, and typed decision routes.
- Added a dependency-free TypeScript HTTP client with timeout and structured errors.
- Added synthetic benchmark generation, grouped data splits, objective/control/
  ranking/subjective/ambiguous/trolley families, ECE/Brier/NLL/AUROC, selective
  metrics, robustness probes, and JSON/Markdown reports.
- Added an optional, explicitly invoked generative LLM baseline adapter. No secrets
  or external credentials are committed.
- Added CI, Docker examples, Python examples, docs, and focused tests.

## Verified evidence

The dependency-free suite currently covers schemas, backend contracts, registry,
calibration, benchmark metrics, CLI, and HTTP behavior: **90 passed, 4 optional
weight tests skipped** in the default run. TypeScript build and five client
contract tests pass. Real tiny DeBERTa and base ModernBERT cached smoke tests
pass on CPU. The 24-case CPU validation reports were run on the reference Apple
Silicon machine. Tiny measured 100% accuracy on this small synthetic slice with
29.16 ms warm p50; base measured 92.86% with 63.52 ms warm p50. These are
descriptive smoke results, not evidence of generalization.

- [tiny CPU validation JSON](benchmarks/reports/tiny-cpu-validation.json)
- [base CPU validation JSON](benchmarks/reports/base-cpu-validation.json)

The checked-in synthetic corpus contains 2,320 rows from 232 authored seeds and
55 semantic groups. It is correlated project-authored data, not 2,320 independent
human validations. The current report is a plumbing and baseline artifact, not a
superiority claim over any other system.

## Remaining work

- Run the full benchmark matrix for tiny/base/smart/multilingual after deciding
  which hardware and time budget to publish; report actual memory and timing.
- Add a reproducible tiny ONNX export and compare quality/calibration/latency to
  its source model, including dynamic INT8 where supported.
- Collect independently adjudicated, consented deployment data and calibrate per
  task family. Preserve final groups while training any candidate scorer.
- Implement shared-state encoding, dynamic request batching, CoreML measurement,
  and an optional Pydantic AI adapter only when benchmark evidence justifies them.
- Publish a release only after package metadata, changelog, model notices, and
  release checks are reviewed.

## Deviations from the master specification

The alpha does not claim low-double-digit latency, parity with hosted decision
services, 2,000 independent examples, automatic safe decisions, full score/rubric parity, or that MPS is faster
than CPU. Those are explicit research targets. External APIs are optional and
never required for local inference. Timestamps in this progress note describe
actual work and measured runs; repository history is kept truthful.
