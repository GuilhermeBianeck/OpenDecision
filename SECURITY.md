# Security and privacy

Inference loads pinned, locally cached safetensors with `trust_remote_code=False`.
The explicit `pull` command connects to Hugging Face. An explicitly selected
external benchmark sends its dataset to the selected provider; no external API
is called during ordinary local scoring. This package has no telemetry service.

The HTTP server binds to loopback by default. It has bounded requests and model
concurrency but no built-in identity provider, tenant isolation, or internet-facing
rate limiter. A deliberate network bind requires your own authenticated gateway.
Request bodies and full model state are not logged by default. Benchmark commands
intentionally write evaluation artifacts; avoid evaluating private text into
reports you intend to publish.

Model scores can be wrong or influenced by adversarial text. The engine is not
an authorization boundary. Enforce tool permissions, spending limits, and other
application invariants independently of a model decision.

Report vulnerabilities privately through the repository's GitHub security
advisory facility where enabled. Do not post exploitable details or secrets in a
public issue. Alpha releases have no security support or response-time guarantee.
