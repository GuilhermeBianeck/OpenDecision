import assert from "node:assert/strict";
import {test} from "node:test";
import {OpenDecision, OpenDecisionHTTPError, OpenDecisionTimeoutError} from "../dist/index.js";

const request = {state: "A charge was duplicated.", question: "Team?", choices: ["billing", "technical"]};

test("routes requests and serializes the documented batch envelope", async () => {
  const calls = [];
  const client = new OpenDecision({baseUrl: "http://127.0.0.1:8042/", fetch: async (url, options) => {
    calls.push({url, options});
    return new Response(JSON.stringify({choice: "billing"}), {status: 200});
  }});
  assert.equal((await client.choose(request)).choice, "billing");
  await client.chooseBatch([request]);
  await client.rank(request);
  await client.boolean({state: "", question: "Ready?"});
  await client.score({state: "", question: "Severity?", levels: ["none", "minor", "major"]});
  await client.ask({state: "", questions: {
    team: {type: "choice", question: "Team?", choices: ["billing", "technical"]},
    urgent: {type: "boolean", statement: "The message is urgent."},
    severity: {type: "score", question: "Severity?", levels: ["none", "major"]},
  }});
  await client.multiLabel({state: "", labels: ["Ready?"]});
  await client.health();
  await client.models();
  assert.deepEqual(calls.map(c => new URL(c.url).pathname), [
    "/v1/decide", "/v1/decide/batch", "/v1/rank", "/v1/boolean", "/v1/score", "/v1/ask",
    "/v1/multi-label", "/health", "/v1/models"
  ]);
  assert.deepEqual(JSON.parse(calls[0].options.body), request);
  const structured = {
    state: {message: "charged twice", order_id: "A-104"},
    question: "Team?",
    choices: ["technical", {label: "billing", description: "payments and refunds", examples: ["charged twice"]}],
  };
  await client.choose(structured);
  assert.deepEqual(JSON.parse(calls.at(-1).options.body), structured);
  assert.deepEqual(JSON.parse(calls[1].options.body), {requests: [request]});
  assert.deepEqual(JSON.parse(calls[4].options.body).levels, ["none", "minor", "major"]);
  assert.deepEqual(Object.keys(JSON.parse(calls[5].options.body).questions), ["team", "urgent", "severity"]);
  assert.equal(calls[7].options.method, "GET");
  assert.equal(calls[7].options.body, undefined);
});

test("retains structured validation errors", async () => {
  const body = {detail: [{msg: "choices must be unique"}]};
  const client = new OpenDecision({fetch: async () => new Response(JSON.stringify(body), {status: 422})});
  await assert.rejects(client.choose(request), error => {
    assert.ok(error instanceof OpenDecisionHTTPError);
    assert.equal(error.status, 422);
    assert.deepEqual(error.body, body);
    return true;
  });
});

test("retains non-JSON proxy errors", async () => {
  const client = new OpenDecision({fetch: async () => new Response("Service busy", {status: 503})});
  await assert.rejects(client.choose(request), error => error instanceof OpenDecisionHTTPError && error.body === "Service busy");
});

test("aborts overdue requests with a typed timeout", async () => {
  const client = new OpenDecision({timeoutMs: 10, fetch: (_url, options) => new Promise((_resolve, reject) => {
    options.signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
  })});
  await assert.rejects(client.choose(request), OpenDecisionTimeoutError);
});

test("checks configuration and malformed server responses", async () => {
  assert.throws(() => new OpenDecision({baseUrl: "file:///tmp/socket"}), TypeError);
  assert.throws(() => new OpenDecision({timeoutMs: 0}), TypeError);
  const client = new OpenDecision({fetch: async () => new Response("{broken json")});
  await assert.rejects(client.health(), /invalid JSON/);
});
