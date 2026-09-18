"""Start `opendecision serve` first; this client uses only Python's standard library."""

import json
from urllib.request import Request, urlopen

payload = {
    "state": "The customer was charged twice.",
    "question": "Which team?",
    "choices": ["billing", "technical", "other"],
    "abstain_threshold": 0.80,
}
request = Request(
    "http://127.0.0.1:8042/v1/decide",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=30) as response:
    print(json.dumps(json.load(response), indent=2))
