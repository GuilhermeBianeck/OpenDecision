# Project reality: a local decision engine

Assessment date: 18 September 2026.

## Verdict

A useful local decision runtime is feasible on a 16 GB Apple Silicon Mac. The
defensible initial advantage is control: cached offline inference, private input,
open implementation, replaceable models, and reproducible evaluation. Whether
it is more accurate, faster, or better calibrated than a hosted decision service
or a generative LLM is an empirical question that only committed benchmark
reports can answer.

The master specification mixes an attainable software alpha with a research
program. The runtime, API, calibration tools, benchmark infrastructure, and
packaging can be implemented now. Reliable performance across arbitrary
decisions, tightly calibrated uncertainty under distribution shift, sub-20 ms
latency, and shared-state encoding require further evidence and likely training.

## The decision contract

OpenDecision models a decision as **state + question + finite choices**, answered
with a typed result, a distribution over the choices, and an explicit uncertainty
margin. Several independent questions can share one state. The contract is
deliberately narrow: no free-text generation, no parsing of model output, and
no hidden reasoning trace. Application code composes atomic questions and owns
every side effect.

OpenDecision starts with candidate scoring: an NLI encoder estimates support
for a hypothesis; a reward model rates a candidate response; a reranker scores
relevance. These training objectives are useful baselines, but none automatically
produces an expert decision policy. The
[ModernBERT NLI model card](https://huggingface.co/tasksource/ModernBERT-base-nli)
and [Skywork reward model card](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B)
describe those underlying objectives. The claim that they can help decision
workloads is a project hypothesis, not a statement from those model authors.

## Main engineering and research risks

| Risk | Consequence | How the alpha addresses it |
|---|---|---|
| Softmax always chooses a winner | Wrong decisions can look confident | Distinct normalized/calibrated fields; explicit abstention thresholds |
| Task and wording sensitivity | A good classifier may be a poor action selector | Separate task families and serialization templates |
| Repeated state encoding | More questions and choices cost more compute | True candidate microbatches now; shared encoding is future research |
| Synthetic benchmark bias | High fixture scores may not transfer | Document provenance, correlated variants, and independent scenario groups |
| Calibration shift | A fitted temperature can fail on a new task | Bound profile identity; publish held-out scores and reliability bins |
| Platform assumptions | MPS may lose to CPU for small jobs | Explicit devices, measured reports, and a repeatable performance matrix |
| Unsafe application decisions | Model confidence can be mistaken for authority | Keep permission enforcement outside the model |

`decide_many` flattens requests into independent candidate pairs; it does not
claim a shared-state architecture. Cost grows with the number of candidates and
the length of the state until a backend can reuse an encoded state.

## Evidence required for “better”

1. Define a deployment workload and costs of wrong answers, abstentions, and delay.
2. Collect independent real examples with consent and provenance; adjudicate
   objective labels and state explicit policies for subjective targets.
3. Fit calibration only on its reserved split. Choose operating thresholds and
   model variants on validation. Keep final test cases untouched during selection.
4. Compare quality at equal coverage, latency at matched workloads, and memory
   on specified hardware. Label remote latency as client-observed network latency.
5. Publish failures, confidence intervals over independent groups, and enough
   artifacts to reproduce the result. No bootstrap intervals over correlated
   variants are currently claimed.

The committed reports are baseline measurements, not a superiority claim.
The 2,320-case fixture corpus contains 232 underlying authored scenarios and
55 semantic groups; it is explicitly not 2,320 independent human validations.
See [PROGRESS.md](../PROGRESS.md) for actual results and outstanding evidence.
