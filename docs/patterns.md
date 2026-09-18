# Composition patterns

The runtime answers atomic questions. Complex judgements are composed in your
code, not asked in one question. Two of these patterns exist because the
benchmark showed the direct approach failing; the numbers below come from the
committed corpus on the reference machine with `base`, and the commands to
reproduce them are in [benchmarks](benchmarks.md).

## Decompose a conditional rule

A state that carries a rule and a fact — "Releases proceed only after the smoke
tests pass. The smoke tests did not pass." — is a poor question to ask directly.
Asking `choose` for the action makes the model weigh options whose words appear
in the rule clause, and it answers as though the condition held:

| Direct `choose` on rule + fact | Accuracy |
| --- | ---: |
| fact affirms the condition | 0.967 |
| fact negates the condition | **0.533** |

The same model asked the atomic factual question instead, with the mapping done
in code, keeps both halves:

| Boolean condition, mapped in code | Accuracy |
| --- | ---: |
| fact affirms the condition | 0.967 |
| fact negates the condition | **0.933** |

```python
supported = model.boolean(state=state, question="The smoke tests passed.")
action = "proceed" if supported.value else "hold"
```

Measured over 60 rows in 10 scenarios; the gain is concentrated in the negated
half, and three of the ten scenarios handled negation correctly even directly.
Write the condition as a positive statement of fact and let the caller decide
what each answer means. Abstention still applies: `supported.value` is `None`
when the state does not settle the question, which is a third outcome your code
should handle rather than coerce.

## Screen the state before deciding

The state and the question reach a candidate scorer through one channel, so an
instruction written into the state is read as context. On the corpus's
embedded-instruction rows `base` follows the injected instruction on **every**
row, choosing the option the text names at up to 0.92 confidence, and no
serialization change tested fixed it: quoting the state, shortening the
hypothesis template, and scoring entailment against contradiction all left the
majority of injections successful.

Screening the state with a separate statement before deciding does work, because
it uses the statement path rather than the choice path:

```python
from opendecision.guards import screen

[check] = screen(model, [state])
if check["flagged"]:
    escalate(state, check)  # do not let the model decide this one
else:
    decision = model.choose(state=state, question=..., choices=...)
```

Operating points for `INJECTED_INSTRUCTION` with `base`, by threshold:

| Threshold | Bracketed system-style injection | Injection written as prose | Clean states flagged |
| ---: | ---: | ---: | ---: |
| 0.50 | 0.600 | 0.500 | 0.000 |
| 0.30 | 0.825 | 0.500 | 0.000 |
| **0.20** (default) | **0.975** | **0.542** | **0.008** |
| 0.10 | 1.000 | 0.625 | 0.058 |
| 0.05 | 1.000 | 0.833 | 0.192 |

Measured over 40 authority-styled rows, 24 embedded-prose rows and 120 clean
rows. Prose injections that read like an ordinary aside are the hard case and
remain the weak point. This is a detector, not a security boundary: it lowers
the rate at which a planted instruction reaches a decision, and it does not
make the decision safe. Enforce permissions in your application.

## Ask several questions about one state

`ask` evaluates independent typed questions against the same state and returns
them under your own ids, so a routing choice, two gates and a severity rating
are one call and one composition step in your code.

```python
answers = model.ask(
    state=ticket,
    questions={
        "team": {"type": "choice", "question": "Which team?", "choices": [...]},
        "urgent": {"type": "boolean", "statement": "The customer asks for same-day resolution."},
        "severity": {"type": "score", "question": "How severe?", "levels": [...]},
    },
)
```

Cost depends on the backend. The cross-encoders read the state once per
candidate, so questions and options multiply the state cost; the `decoder`
backend encodes each distinct state once and answers every question from that
cache. On a 1,500-token state, thirteen boolean questions take about 2.8 s
through `decoder` and about 41 s through `base`.

## Gate on confidence, not only on the answer

`abstain_threshold` and `margin_threshold` turn a distribution into three
outcomes: act, escalate, or refuse. Pick the thresholds from the cost of a wrong
decision against the cost of an escalation, on a validation split, and read
[calibration](calibration.md) first — with a per-candidate-count profile loaded a
single margin threshold is not comparable across questions of different width.
