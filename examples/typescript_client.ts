// Build/install packages/typescript locally and start `opendecision serve` first.
import {OpenDecision} from "@opendecision/client";

const client = new OpenDecision({baseUrl: "http://127.0.0.1:8042", timeoutMs: 60_000});
const result = await client.choose({
  state: "The customer says their card was charged twice.",
  question: "Which team?",
  choices: ["billing", "fraud", "technical", "other"],
  abstain_threshold: 0.80,
});
console.log(result.choice, result.confidence, result.calibrated_probabilities);
