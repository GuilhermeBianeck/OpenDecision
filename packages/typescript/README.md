# @opendecision/client

Typed, dependency-free HTTP client for OpenDecision's local server. Requires Node 20+ or a modern browser with `fetch`. This alpha package is built from source; registry publication is a separate release step.

```sh
cd packages/typescript
npm ci
npm run build
npm test
```

Install this directory in your application using `npm install /absolute/path/to/OpenDecision/packages/typescript`. Start a server with `opendecision serve` after downloading its model (`opendecision pull base`).

```ts
import {OpenDecision, OpenDecisionHTTPError} from "@opendecision/client";

const client = new OpenDecision({baseUrl: "http://127.0.0.1:8042", timeoutMs: 30_000});
const result = await client.choose({
  state: "The customer's card was charged twice.",
  question: "Which team?",
  choices: ["billing", "technical", "other"],
  abstain_threshold: 0.8,
});
console.log(result.choice); // null when abstained
console.log(result.normalized_probabilities);
console.log(result.calibrated_probabilities); // null unless the server loaded a profile
```

Methods: `health()`, `models()`, `choose(request)`, `chooseBatch(requests)`, `boolean(request)`, `score(request)`, `rank(request)`, and `multiLabel(request)`. The server accepts at most 64 batch requests or labels and a total request body up to 1 MiB. A full inference queue returns HTTP 429 with `Retry-After: 1`.

HTTP failures throw `OpenDecisionHTTPError` with `.status` and `.body`. Timeouts throw `OpenDecisionTimeoutError`; the default is 30 seconds, including response body reading. The client does not retry automatically. Returned TypeScript types describe the server contract; arbitrary responses are not runtime schema-validated.

`score(request)` rates the state on ordered `levels` (2–10 descriptions, lowest first) and returns `level` (most probable index or null when abstained), `legend`, the distribution keyed by level index, and `score`, the probability-weighted index.

`BooleanResult.method` is `"statement"` when the server's model scores the statement directly against the state; `unsupported` is then the probability that the state settles neither way, and `unsupported_threshold` abstains above it. Servers whose model lacks statement scoring return `"binary_choice"`, `unsupported: null`, and reject `unsupported_threshold` with HTTP 422. `health()` reports `supports_statements`.

`confidence` is the top-two probability margin. `probabilities` contains calibrated scores only when the server has a matching calibration profile; otherwise it equals `normalized_probabilities`. Neither field is an automatic guarantee of correctness. `BooleanResult.probability` is the yes probability even after abstention.

The server binds to loopback by default, has no authentication, and sends no prompts to external services. Browser use across origins requires a reverse proxy or an explicit CORS policy configured by your application; the default server does not grant cross-origin access.

MIT license. See [the repository](https://github.com/GuilhermeBianeck/OpenDecision) for the Python SDK and model licensing information.
