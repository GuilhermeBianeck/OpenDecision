# Requirements: parity with the decision-model ecosystem

Status as of 19 October 2026. This document turns a survey of the typed
decision-model field into testable requirements for OpenDecision. Each
requirement has an identifier used by the GitHub milestones and issues, an
acceptance test, and where it applies a measured performance bar.

## What changed in the field

Within five weeks of the first hosted typed decision model, roughly two dozen
alternatives appeared, hosted and open-weight, and three things became shared:

1. **A wire format.** The `POST /v1/systemone` request — a `state`, a `model`,
   and a map of named `questions` of type `choice`, `score` or `noul` — is now
   served by Ollama (0.35 and later), llama.cpp's `llama-server`, SGLang, vLLM,
   and most open decision models. Official and community SDKs exist for Python,
   TypeScript, Rust, Go, Ruby, Elixir, Swift and more, alongside integrations
   in the Vercel AI SDK, Pydantic AI and LlamaIndex.
2. **Benchmarks.** A public, MIT-licensed decision benchmark publishes 231 of
   its 534 items (72 original, 48 easy, 111 hard), scores chance-corrected
   accuracy and calibration, and its harness speaks the wire format above.
   Leaderboards built on it rank more than a hundred open-weight systems.
3. **Purpose-trained small models.** Open models fine-tuned for decisions now
   report around 89 % on those 231 items at 4B parameters, with Apple-silicon
   MLX builds near 3 GB.

An early evidence audit of the field (arXiv 2609.32160) found that a typed
readout has shown no accuracy advantage over a comparable label-probability
readout, and that the clearest gains are latency and cost. OpenDecision is a
label-probability readout run locally, so the architecture is sound; what it
lacks is the shared interface, a comparable number, and a decision-trained
default model.

## Where OpenDecision stands

| Area | Today | Gap |
| --- | --- | --- |
| Question types | choice, boolean, score, rank, multi-label, typed `ask`, wide choice | none for text; no bounding-box type |
| Wire format | own `/v1/decide`, `/v1/ask` and friends | does not serve `/v1/systemone`, so no ecosystem SDK or tool can call it |
| Criteria | label or `{label, description, not_for, examples}` | instructions and criteria cannot be arbitrary JSON objects or arrays |
| Evaluation | 6,578-row project-authored corpus | no result on any externally shared benchmark |
| Default model | generic Qwen3.5 2B, 0.799 objective accuracy | not trained for decisions |
| Context | 2,048 tokens by default, 8,192 for MLX backends | ecosystem default is 64k, and up to 262k |
| Input modality | text only | no image state |
| Robustness | negation fixed at 4B; injection unsolved | no option-order invariance or appended-context checks |

## Requirements

Performance bars refer to the reference machine: Apple silicon, 16 GB, with
nothing else resident. **No requirement may raise peak memory above what that
machine can hold alongside a desktop session.**

### M1 — Standard decision API

- **API-1 `/v1/systemone` endpoint.** The server accepts the shared request and
  returns the shared response: `answers` keyed by question id, each with `type`;
  `choice` returns `choice`, `probabilities` and `confidence`; `score` returns
  `score`, `legend`, `probabilities` and `confidence`; `noul` returns `noul`.
  *Accept:* a request copied from the public specification succeeds unmodified
  and every answer validates against it.
- **API-2 Structured instructions and criteria.** `instructions`, option
  descriptions, rubric levels and `noul` `true`/`false` criteria accept string,
  object or array, rendered deterministically for the model.
  *Accept:* an option described by `{what, not_for, examples}` round-trips.
- **API-3 Limits and errors.** 255 choice options, 2–10 score levels, 64 levels
  of nesting, finite numbers only, a 32 MiB body cap, and the status codes 400,
  413, 415 and 422 with their documented meanings; `gzip` and `zstd` request
  bodies. *Accept:* one test per limit and per status code.
- **API-4 Usage accounting.** Every response reports `usage.input_tokens` and
  `usage.output_tokens` (always 0: nothing is generated).
- **API-5 Clients.** The Python and TypeScript clients can call `/v1/systemone`.

### M2 — Comparable evaluation

- **EVAL-1 Public benchmark run.** Run the 231 public items through
  `/v1/systemone` with the benchmark's own MIT harness and commit the result,
  reporting raw accuracy, chance-corrected accuracy per tier, and calibration.
- **EVAL-2 Probability consistency.** Report how far `p(X)` and `p(not X)` are
  from summing to one for paired statements, a published check where hosted
  models miss by 0.064 and a 27B open model by 0.293.
- **EVAL-3 Option-order invariance.** Report how often a decision changes when
  the same options are reordered, and add it as a regression test.

### M3 — A decision-trained local default

- **MODEL-1 Decision-trained backend.** Add an Apache-2.0 decision-tuned model
  with an Apple-silicon build.
  *Bar:* at least 85 % on the 231 public items, ECE at most 0.10 on the project
  corpus, warm p50 at most 250 ms on the reference machine.
- **MODEL-2 Default by evidence.** `auto` moves to the new backend only if it
  beats the current default on both the public items and the project corpus.
- **MODEL-3 Memory guard.** A backend refuses to load, with an actionable error,
  when free memory is below its measured requirement.

### M4 — Long context and robustness

- **CTX-1 Long state.** Decoder backends accept states up to their checkpoint
  context, with a measured accuracy and latency curve at 2k, 8k and 32k tokens.
- **ROB-1 Appended-context robustness.** Measure how often an appended
  irrelevant opinion flips a decision; a published audit finds 12.1 % for one
  hosted model.
- **ROB-2 Order-averaged scoring.** Offer option-order averaging as a measured
  setting where EVAL-3 shows sensitivity.

### M5 — Images (experimental)

- **VIS-1 Image state.** Accept images in `state` for a vision-capable backend.
- **VIS-2 Bounding-box question.** A `box` question returns a bounding box.

Both depend on a vision model that fits the memory bar, and are tracked so the
gap is visible rather than promised.

## Order of work

M1 first, because it unlocks M2: once the wire format is served, the public
harness runs against OpenDecision unchanged and the result is directly
comparable with every system on its leaderboard. M3 then has a fair test to
pass. M4 and M5 follow.
