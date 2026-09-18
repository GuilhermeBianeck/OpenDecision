# Model licenses and immutable identities

Verified against the upstream Hugging Face model cards, configs, and Hub file metadata on **2026-09-18**. OpenDecision code is MIT licensed. Model weights are separate downloads, are not included in the Python distribution, and retain their own licenses.

| Alias | Upstream model card | License declared upstream | Weight file size | Parameters |
| --- | --- | --- | ---: | ---: |
| `tiny` | [cross-encoder/nli-deberta-v3-xsmall](https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall/blob/a150876415327c80daeff35ca6f68f5ed8cf5c24/README.md) | Apache-2.0 | 283,353,172 bytes | 70,831,107 |
| `base` / `auto` | [tasksource/ModernBERT-base-nli](https://huggingface.co/tasksource/ModernBERT-base-nli/blob/de4ab7e77845098b7fab7f6ab9d370ddff27b19c/README.md) | Apache-2.0 | 598,442,860 bytes | 149,607,171 |
| `smart` | [Skywork/Skywork-Reward-V2-Qwen3-0.6B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B/blob/8c14a4e9e6321deaf572544339b16b8d6bbe8886/README.md) | Apache-2.0 | 1,192,137,232 bytes | 596,050,944 |
| `multilingual` | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3/blob/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e/README.md) | Apache-2.0 | 2,271,071,852 bytes | 567,755,777 |

Sizes above count `model.safetensors` only; tokenizers and metadata add disk space. Values are from each model's official Hub API `?blobs=true` response. DeBERTa also has a 512-element integer buffer, excluded from the parameter count. These are actual upstream artifact sizes, not the smaller quantization targets in the project proposal. The multilingual model is substantially larger than the default base model.

All four model cards declare Apache-2.0. Under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), commercial use and redistribution are permitted subject to its conditions, including retaining applicable copyright, license, and notice material, identifying modifications, and respecting its patent and trademark provisions. An upstream model-card license declaration does not independently establish rights to every underlying training example. The repository does not relicense upstream datasets or promise legal clearance for a particular application.

The pinned snapshots did not contain standalone LICENSE/NOTICE files when inspected; the model cards are retained during download. Before redistributing weights in another product, preserve their declared Apache license and applicable upstream notices. Updates to model identities require another license and serialization review.

## Revision pins

```text
tiny          a150876415327c80daeff35ca6f68f5ed8cf5c24
base          de4ab7e77845098b7fab7f6ab9d370ddff27b19c
smart         8c14a4e9e6321deaf572544339b16b8d6bbe8886
multilingual  953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
```

`opendecision pull <alias>` explicitly downloads the pinned safetensors weights, tokenizer files, configuration, and available license/model-card files into the standard Hugging Face cache. It then loads the checkpoint and runs a public smoke input. The report includes the revision, declared license, unique snapshot file bytes, scores, device, and runtime precision. A successful smoke test verifies inference plumbing; it makes no decision-quality claim.

Normal inference uses `local_files_only=True`, `trust_remote_code=False`, and `use_safetensors=True`. It never silently downloads a model or loads upstream Python modules or pickle weights. `auto` is a deterministic alias for `base`; it does not choose based on estimated memory or download a fallback. Missing cached weights produce an explicit pull instruction.

## Scoring and serialization

- **DeBERTa:** tokenize a premise/hypothesis pair. Read the configured entailment logit (index 1 in the pin). The configured limit is 512 tokens.
- **ModernBERT:** tokenize the same pair. Read the configured entailment logit (index 0 in the pin). The checkpoint config limit is **2,048** tokens, irrespective of larger context claims for other ModernBERT checkpoints. Reference compilation is disabled for portability.
- **Skywork:** use the pinned `chat_template.jinja` with one user message containing state and question and one assistant message containing the candidate. No system message is added. Score the scalar classification head; no text is generated. The upstream card recommends at most 16,384 tokens even though the architecture supports more.
- **BGE:** score a query consisting of question and state against a candidate passage using its scalar relevance head. This is an experimental adaptation of retrieval relevance to decision scoring, not evidence of general reasoning ability.

NLI `template="default"` uses `Given the situation, the answer to "{question}" is "{choice}".`; `template="short"` uses `{question} {choice}`. Skywork's short template uses plain state plus question instead of labeled blocks. BGE currently has one serialization for either template setting. The default length is 512 tokens for all backends so comparisons begin with a consistent limit. Long inputs truncate only the state prefix, preserving the full question and choice, emitting a warning and reporting the number of affected candidates. A minimum of 32 state tokens (or the complete shorter state) is reserved. Overlong questions or choices are rejected instead of silently truncated. Prefix truncation can discard decisive information; evaluate longer limits when needed.

All adapters flatten candidates across requests into real tensor batches and return raw scores. Neither these scores nor a softmax of them are calibrated correctness probabilities. NLI entailment, learned preference, and retrieval relevance are different objectives; adapting them to arbitrary decisions requires evaluation.

## Device and precision policy

`device="auto"` selects CUDA when available, otherwise CPU. MPS is explicitly selectable and must be benchmarked on the user's hardware; availability alone is not evidence of a speed advantage. The default runtime precision is float32 for portable baselines. The Skywork artifact stores bfloat16 weights but is converted to float32 by default, increasing resident weight memory beyond its file size. Explicit float16 or bfloat16 is permitted on supported hardware, reported in result metadata, and requires separate quality/calibration checks. Float16 on CPU is rejected.

`demo` uses deterministic token overlap solely for installation, API, and test infrastructure. It has no model weights and no learned decision quality, and must not be included as a competitive model in quality claims.
