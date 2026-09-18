# Examples

Install from this repository with `pip install -e '.[inference,server]'`, then explicitly download a model with `opendecision pull base`. The Python examples and server subsequently use cached weights offline.

- `python python_quickstart.py`: one decision with abstention.
- `python agent_routing.py`: mixed choice, boolean and score questions sharing one structured state via `ask`.
- Start `opendecision serve`, then run `python http_client.py` for HTTP.
- `typescript_client.ts`: typed local HTTP client; build instructions in `packages/typescript/README.md`.

For a fast installation check without model weights, use `opendecision decide --model demo --state 'billing support' --question 'Which team?' --choices billing technical`. The demo is an infrastructure fixture, not a learned semantic model or quality benchmark.

A normalized score and a top-two margin are not empirical correctness probabilities. Fit an appropriate calibration profile on a separate calibration split and evaluate thresholds on held-out data before relying on scores in an application.
