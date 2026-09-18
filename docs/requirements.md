# Requirements map

This document maps the master build specification to the alpha implementation.
It distinguishes implemented behavior from research targets and deliberately
does not turn a proposal into an unsupported performance claim.

| Requirement area | Alpha status | Evidence / limitation |
| --- | --- | --- |
| Local-first finite-choice runtime | Implemented | Python SDK, offline cached Transformers adapters, demo fixture |
| Typed schemas and validation | Implemented | Pydantic request/result models; duplicate, blank, size, and probability checks |
| Shared backend protocol | Implemented | NLI, reward, reranker, demo, and content-addressed ONNX adapter |
| Tiny/base/smart model tiers | Adapters implemented | Upstream checkpoints pinned; targets are not measured quality guarantees |
| Device selection | Implemented | `auto` resolves CUDA, then MPS, then CPU; MPS preference backed by a published measurement |
| Choice / boolean / multi-label / ranking / batches | Implemented | Multi-label is independent binary scoring; shared-state batching is flattened |
| Calibration | Implemented | Held-out temperature scaling profile with identity and dataset hash |
| Abstention | Implemented | Minimum top probability and top-two margin thresholds |
| CLI | Implemented | `models`, `pull`, `doctor`, `decide`, `rank`, `boolean`, `serve`, `benchmark`, `calibrate` |
| Local HTTP server | Implemented | FastAPI on loopback, resident load, bounded body/batch/concurrency |
| TypeScript SDK | Implemented | Dependency-free fetch client with typed errors and contract tests |
| Benchmark data and metrics | Implemented | 2,320 correlated synthetic rows; objective, control, ranking, ambiguity, subjective, trolley |
| Generative LLM baseline adapter | Optional | Explicit credentials and command required; no live comparison included |
| Quantization / ONNX | Experimental | Tiny-only manifest-bound CPU artifact; `scripts/export_onnx.py` is explicit and separately verified |
| CoreML / MPS optimization matrix | Planned | Hardware-specific measurement needs dedicated runs |
| Fine-tuning and distillation | Planned | Baseline first; scripts must preserve split isolation |
| Pydantic AI integration | Planned | Public Python API is stable enough for a future adapter |
| Published package | Pending | Checkout installation works; no release credentials or PyPI/npm publish performed |

The model adapters score candidates using different objectives. An adapter
existing is not evidence that every arbitrary decision is accurate. The project
requires real deployment data and independent evaluation before claiming that
OpenDecision is better than any other decision system.

## Acceptance checks

The alpha acceptance path is:

```bash
python -m pip install -e '.[dev,server,external]'
pytest
npm test --prefix packages/typescript
opendecision doctor
opendecision benchmark --model demo --limit 24 \
  --output local-reports/demo-smoke
```

Real model acceptance additionally requires an explicit `opendecision pull`
and records the exact revision, device, precision, latency scope, and report
hash. The demo backend can verify plumbing but is excluded from model-quality
claims.
