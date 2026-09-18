# OpenDecision

**Local decisions. Explicit uncertainty. Your data stays on your machine.**

[![CI](https://github.com/GuilhermeBianeck/OpenDecision/actions/workflows/ci.yml/badge.svg)](https://github.com/GuilhermeBianeck/OpenDecision/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Status: alpha](https://img.shields.io/badge/status-0.1.0a1-orange.svg)](CHANGELOG.md)

OpenDecision turns **state + question + finite choices** into a typed decision,
a distribution over the choices, and an explicit uncertainty margin. It scores
candidates with open-weight classifiers, reward models, and rerankers, without
generating an answer token by token. Inference runs locally, without an API key.

This repository is named **OpenDecision**; the Python package is **opendecision**.
The alpha establishes a measurable local runtime for decision workloads.
It does **not** establish that these models outperform hosted decision services
or frontier LLMs; every quality claim must come from a committed benchmark report.

## Start locally

Python 3.10+ is supported; Python 3.12 is recommended for optional model runtimes.
The package has not been published to PyPI. Install from this checkout:

```bash
git clone https://github.com/GuilhermeBianeck/OpenDecision.git
cd OpenDecision
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[inference,server]'
opendecision pull base
```

The explicit `pull` command downloads the pinned checkpoint. All later inference
uses the local cache and refuses to download missing files automatically.

```python
from opendecision import DecisionModel

model = DecisionModel("base", device="cpu")
result = model.choose(
    state="The customer says their card was charged twice.",
    question="Which team should handle this?",
    choices=["billing", "fraud", "technical", "other"],
    abstain_threshold=0.80,
    margin_threshold=0.20,
)
print(result.choice)  # None when a threshold is not met
print(result.probabilities)  # normalized scores; calibrated if a profile is loaded
print(result.confidence)  # top-one minus top-two probability
print(result.metadata["calibrated"])
```

No weights yet? `pip install -e '.[server]'` and `DecisionModel("demo")` exercise
the complete API using a deterministic lexical scorer. **Demo is an infrastructure
fixture, not an AI quality baseline.**

## What the alpha includes

| Capability | Interface |
|---|---|
| Finite choice, boolean, ranking | `choose`, `boolean`, `rank` |
| Ordered rubric score | `score` (expected level index over 2–10 described levels) |
| Independent labels | `multi_label` (one binary distribution per label) |
| Shared-state questions and batches | `decide_many`, `choose_batch` |
| Four open-weight model adapters | `tiny`, `base`, `smart`, `multilingual` |
| Local resident model server | FastAPI on `127.0.0.1:8042` |
| Temperature calibration and abstention | Separate profiles; probability and margin thresholds |
| Reproducible evaluation | JSON + Markdown reports; grouped synthetic splits |
| Optional generative LLM baseline | Explicit invocation with your own credentials |
| Python and TypeScript clients | In-process Python; HTTP TypeScript |

See [progress and measured evidence](PROGRESS.md), [the assessment](docs/project-assessment.md),
and [the requirements map](docs/requirements.md) for implementation and verification status.

## Choose a model

| Alias | Scorer | Intended experiment |
|---|---|---|
| `tiny` | DeBERTa v3 xsmall NLI | Small encoder baseline |
| `base` / `auto` | ModernBERT base NLI | General semantic classification baseline |
| `smart` | Skywork Reward V2 Qwen3 0.6B | Candidate response / action preference |
| `multilingual` | BGE reranker v2 m3 | Multilingual relevance ranking |

These aliases name model families, not quality guarantees. `auto` currently
selects `base`; automatic device selection is conservative. Use `--device mps`
or `device="mps"` to measure Apple GPU inference explicitly. CPU remains supported.
The 16 GB Apple Silicon target can run these models, but published latency and
memory targets are research goals. Downloads are about 0.28–2.27 GB of weights;
runtime memory is additional. Review [pinned models and licenses](docs/model-licenses.md).

## Probabilities and uncertainty

`normalized_probabilities` is a softmax over raw model scores. It always sums
to one, even when all candidates are poor. It is **not** a guarantee of empirical
correctness. `calibrated_probabilities` is `null` until a compatible temperature
profile is loaded. `probabilities` selects the calibrated distribution when available.

`confidence = p(top1) - p(top2)`; `top_probability = p(top1)`. A request abstains
if either its `abstain_threshold` (minimum top probability) or `margin_threshold`
is not met. Equality passes. Thresholds are task-specific and not safety guarantees.
Include an explicit “none of these” choice when appropriate. Ties use a stable
lexicographic tie-break; set a positive margin threshold to abstain on ties.

`score` rates the state on an **ordered rubric**: pass 2–10 level descriptions
from lowest to highest, and receive `level` (the most probable index), a
`legend`, the distribution over levels, and `score`, the probability-weighted
level index. A score of 1.3 on a three-level scale means the mass sits between
the second and third levels; it is a position on your rubric, not a calibrated
magnitude. Describe levels as concrete situations rather than degrees.

```python
result = model.score(
    state="The checkout page returns HTTP 500 for every customer.",
    question="How severe is this incident?",
    levels=["cosmetic issue", "degraded but usable", "feature unavailable", "outage for all users"],
)
print(result.level, result.score, result.legend[str(result.level)])
```

`boolean` and `multi_label` read their question as a **statement about the
state**. With an NLI backend (`tiny`, `base`) the statement is scored directly:
`probability` is entailment against contradiction, and `unsupported` is the
probability that the state settles neither way — pass `unsupported_threshold`
to abstain on it. Backends without statement scoring fall back to a two-way
`yes`/`no` choice and report `method: "binary_choice"`.

```bash
opendecision calibrate --model base --dataset benchmarks/datasets/core.jsonl \
  --output calibration/base.json
```

```python
model = DecisionModel("base", calibration="calibration/base.json")
```

Fit on the calibration split, select thresholds on validation, then evaluate
the untouched test split. A profile checks checkpoint, template, precision,
and sequence limit; it does not establish reliability on a new task distribution.
See [calibration](docs/calibration.md) and [benchmark methodology](docs/benchmarks.md).

## Local HTTP and TypeScript

```bash
opendecision serve --model base
curl http://127.0.0.1:8042/health
curl http://127.0.0.1:8042/v1/decide \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice","question":"Which team?","choices":["billing","technical"]}'
```

Interactive API documentation is at `http://127.0.0.1:8042/docs`.
See the [TypeScript client](packages/typescript/README.md) and runnable [examples](examples/).
The server has no authentication and defaults to loopback. Only expose it behind
an authenticated gateway you control. It never logs request bodies by default.

## Evaluate before relying on it

```bash
opendecision doctor
opendecision benchmark --model base --dataset benchmarks/datasets/core.jsonl \
  --output local-reports/base.json
python -m pip install -e '.[dev,server]'
pytest
```

Synthetic benchmark cases are transparent regression fixtures, not independent
human judgments. Moral cases have no universal accuracy label. External provider
comparisons require an explicit command and credentials; local use does not.
No comparison against a remote service has been measured unless a report records it.

The main limitations are task transfer, calibration shift, finite context,
linear work per candidate, and potentially high confidence on wrong answers.
Candidate order, negation, wording, prompt injection, and irrelevant context
must be evaluated. This library does not enforce application permissions or
make destructive actions safe. See [security](SECURITY.md) and [research roadmap](docs/roadmap.md).

## Contribute

Read [CONTRIBUTING.md](CONTRIBUTING.md). Useful contributions include independently
authored benchmark cases, measured hardware reports, calibration studies, and
backend improvements. Code is MIT-licensed; model weights have their own
[upstream terms](docs/model-licenses.md). No model weights or credentials are bundled.
