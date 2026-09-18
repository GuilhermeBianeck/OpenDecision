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
| Typed questions over one state | `ask` (mixed choice / boolean / score, keyed by your ids) |
| Batches | `choose_batch`, `statement_batch`, `score_batch`, `decide_many` |
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
| `decoder` | Qwen3 0.6B (causal LM, no generation) | Many options, rich descriptions, many questions per state |

`decoder` is the only backend that **encodes the state once** per request
group: every further question or option costs only its own tokens, so `ask`
with a dozen questions over a long state is one prefix pass plus a dozen short
steps. The cross-encoders pay one full pass per candidate. In return the NLI
backends are stronger verifiers: on a ten-statement probe `base` scored the
true/false split 10/10 where `decoder` scored 7/10. Use NLI for booleans and
verification, the decoder for wide or repeated choices over one state; the
[benchmark reports](benchmarks/reports/) measure both.

These aliases name model families, not quality guarantees. `auto` currently
selects `base`. `device="auto"` selects CUDA, then MPS, then CPU; on the
reference 16 GB Apple Silicon machine MPS measured 1.2–3.5× faster than CPU for
every model (see [device policy](docs/model-licenses.md#device-and-precision-policy)).
Pass `device="cpu"` to opt out. The default sequence limit is the model's
context capped at 2,048 tokens (`tiny` is limited to 512 by its checkpoint);
set `max_length` explicitly to change it. Published latency and memory targets
remain research goals. Downloads are about 0.28–2.27 GB of weights;
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

### Ask several questions about one state

`ask` evaluates independent typed questions about the same state and returns
answers under your own ids. Mix choice, boolean and score questions; each
carries its own thresholds. Compose the answers in code rather than asking one
compound question.

```python
answers = model.ask(
    state={"message": "My card was charged twice and I need this fixed today."},
    questions={
        "team": {
            "type": "choice",
            "question": "Which team should handle this?",
            "choices": ["billing", "fraud", "technical", "other"],
        },
        "urgent": {"type": "boolean", "statement": "The customer asks for same-day resolution."},
        "frustration": {
            "type": "score",
            "question": "How frustrated is the customer?",
            "levels": ["calm", "irritated", "angry"],
        },
    },
)
answers["team"].choice, answers["urgent"].value, answers["frustration"].score
```

Every answer carries a `type` field (`choice`, `boolean`, `score`, `ranking`).
Whether a backend encodes the state once or once per candidate is a backend
property; see [architecture](docs/architecture.md).

### Describe the options, structure the state

Bare labels are often ambiguous. Any choice may be a `ChoiceOption` with a
`description`, a `not_for` boundary, and `examples`; results stay keyed by
`label`. The state may be text, a **record** with named fields, or a **list of
texts**; records render as `key: value` lines in your field order, so field
names are part of what the model reads.

```python
result = model.choose(
    state={"message": "My card was charged twice for one order.", "channel": "email"},
    question="Which team should handle this?",
    choices=[
        {
            "label": "billing",
            "description": "invoices, payments, refunds and duplicate charges",
            "not_for": "stolen or cloned cards",
        },
        {"label": "fraud", "description": "unauthorized use of a card or account"},
        "technical",
        "other",
    ],
)
```

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
