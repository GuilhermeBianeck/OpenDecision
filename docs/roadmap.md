# Research roadmap

## Alpha: measurable local foundations

The alpha implements model-independent scoring, typed outputs, offline cached
inference, batching, calibration, abstention, a local server, clients, and transparent
synthetic evaluation. See [requirements](requirements.md) and [progress](../PROGRESS.md)
for the exact verified scope.

## Next evidence milestone

Collect independent deployment examples. Audit label quality and benchmark-family
coverage. Compare candidate serialization, CPU/MPS and ONNX INT8, including changed
calibration after quantization. Measure decision quality at fixed coverage and
publish isolated process memory and HTTP latency. Run external comparisons only
with explicit credentials, suitable data, and reported client-observed latency.

## Training only after a baseline

Use observed validation failures to define targeted training examples. Preserve
untouched final test groups. Train an encoder candidate-scoring objective, compare
to the frozen baseline, recalibrate on reserved data, and rerun all regression
categories. Optional scripts live under [training](../training/README.md); this alpha
does not include newly trained weights or claim improved model quality.

If a larger local scorer consistently helps, collect its soft targets on training
cases and test distillation. External proprietary teacher data is not necessary
for the local runtime and is not bundled.

## Later architecture work

- Shared-state encoding with question/candidate heads, subject to measured quality.
- Dynamic server batching and cancellation-aware scheduling.
- CoreML and broader platform support once correctness and export fidelity are established.
- INT4 only with measured accuracy, calibration, memory, and latency tradeoffs.
- Native multilingual benchmark curation and a validated Pydantic AI adapter.

Performance targets from the original specification remain aspirations, not release
guarantees. A PyPI/npm release requires package-name availability, release credentials,
and a deliberate publishing step; repository installation is supported now.
