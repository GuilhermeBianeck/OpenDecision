# The shared decision format: `POST /v1/systemone`

Typed-decision runtimes have converged on one request shape, and SDKs in many
languages send it. OpenDecision serves it next to its native API, so a program
written against another runtime can point at a local server unchanged. Every
question is translated into the equivalent native question and answered by one
`ask` call over the shared state, so the shared format uses the same backends,
calibration profiles and routing as the native API.

## Request

```json
{
  "state": "I have been trying to connect my Stripe account for 3 days.",
  "model": "any-name",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Charges, refunds and invoices",
        "integrations": {"what": "Connecting third-party accounts", "not_for": "Login problems"},
        "sales": null
      }
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated does the customer appear?",
      "criteria": ["calm", "irritated", "angry"]
    },
    "is_urgent": {
      "type": "noul",
      "instructions": "The message conveys urgency.",
      "criteria": {"true": "a deadline or a blocked business is named"}
    }
  }
}
```

- `state` is text, a record, or a list of texts, exactly as in the native API.
- `model` is accepted as given; the response names the checkpoint that answered.
- `choice` picks one option. `criteria` maps each option name to a description,
  or `null` for none.
- `score` rates on ordered levels, lowest first.
- `noul` is a statement; the answer is the probability that it is true.
  Optional `criteria.true` and `criteria.false` spell out the boundary.

`instructions`, option descriptions and score levels may be any JSON value.
Records render as `key: value` lines in your key order and lists as `- item`
lines, so a rubric written as `{"what", "not_for", "examples"}` reaches the
model with its field names.

## Response

```json
{
  "model": "mlx-community/Qwen3.5-2B-4bit",
  "answers": {
    "department": {"type": "choice", "choice": "integrations",
                   "probabilities": {"billing": 0.19, "integrations": 0.74, "sales": 0.08},
                   "confidence": 0.55},
    "frustration": {"type": "score", "score": 1.19, "legend": {"0": "calm", "1": "irritated", "2": "angry"},
                    "probabilities": {"0": 0.06, "1": 0.68, "2": 0.25}, "confidence": 0.43},
    "is_urgent": {"type": "noul", "noul": 0.56}
  },
  "usage": {"input_tokens": 78, "output_tokens": 0}
}
```

This is the `qwen35` backend's actual answer to the request above, rounded to
two places.

Answers come back under the question keys, in request order. `confidence` is
the margin between the two most probable answers. The shared format has no
abstention, so the most probable option is always reported; use the native
`/v1/ask` when you want abstention thresholds or raw scores.

`usage.input_tokens` counts the rendered state once plus each instruction,
option and level, with the checkpoint's own tokenizer. `output_tokens` is always
zero: answers are read off a single forward pass and no text is generated.

## Limits and errors

| Limit | Value |
| --- | --- |
| Options per `choice` | 2–255 |
| Levels per `score` | 2–10 |
| JSON nesting | 64 levels |
| Numbers | finite only |
| Body | 32 MiB, before and after decompression |
| `Content-Encoding` | one layer of `gzip`, `x-gzip` or `zstd` |

| Status | Meaning |
| --- | --- |
| 400 | The body is not JSON, or its compression is corrupt or truncated. |
| 413 | The body is over 32 MiB, before or after decompression. |
| 415 | The body is not `application/json`, or uses an unknown or stacked encoding. |
| 422 | Valid JSON that breaks a limit or the schema. |
| 429 | Every inference slot is busy; retry after the `Retry-After` interval. |

`zstd` bodies need Python 3.14 or the `zstandard` package on the server.

## Clients

```python
from opendecision import DecisionModel
from opendecision.client import Client

DecisionModel("auto").systemone(request)  # in process, no server
Client("http://127.0.0.1:8042").systemone(request)  # a running server
```

```ts
import {OpenDecision} from "@opendecision/client";
const response = await new OpenDecision().systemone(request);
```

The Python client gzip-compresses bodies over 64 KiB.
