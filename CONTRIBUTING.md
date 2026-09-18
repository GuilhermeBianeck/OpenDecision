# Contributing

Use Python 3.12 and an isolated environment. Install `pip install -e '.[dev,server]'`.
Run `ruff check .`, `ruff format --check .`, and `pytest` before opening a pull request.
Core tests do not download weights or call paid services. Inference is an optional
extra; hardware tests must opt in and report exact dependencies and revisions.

Keep calibration above the scoring backend. Every backend returns one finite raw
score per candidate and supports a flattened candidate batch. Preserve public
schema validation, offline inference, and input privacy. Changes to score
serialization require new benchmark results and a new calibration profile.

For benchmark contributions, record provenance and licensing, label subjective
policy targets explicitly, and keep semantic families together across splits.
Never tune against the final test set. Report negative results and failures.
Synthetic variations do not count as independent human validation examples.

Commit a focused behavior change with a descriptive imperative subject and useful
validation notes. Preserve real author and committer timestamps. Do not add
credentials, model weights, personal prompts, or machine-specific paths.

Code contributions use the repository MIT license. Respectful, constructive
discussion is expected. Report security concerns using [SECURITY.md](SECURITY.md).
