# Benchmark report matrix

Split `test`. Every report below was run on the rows whose dataset hash it records; a differing hash means a different corpus and the rows are not comparable.

| Report | Model | Revision | Device | Rows | Dataset SHA-256 (prefix) | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny-mps-test | cross-encoder/nli-deberta-v3-xsmall | a15087641532 | mps | 1726 | 43c4923ea723 | measured |
| base-mps-test | tasksource/ModernBERT-base-nli | de4ab7e77845 | mps | 1726 | 43c4923ea723 | measured |
| smart-mps-test | Skywork/Skywork-Reward-V2-Qwen3-0.6B | 8c14a4e9e632 | mps | 1726 | 43c4923ea723 | measured |
| multilingual-mps-test | BAAI/bge-reranker-v2-m3 | 953dc6f6f85a | mps | 1726 | 43c4923ea723 | measured |
| decoder-mps-test | Qwen/Qwen3-0.6B | c1899de289a0 | mps | 1726 | 43c4923ea723 | measured |

## Objective decisions (pooled objective, agent control, verification, robustness)

| Report | Count | Accuracy | Balanced | NLL | Brier | ECE | Acc. at margin ≥ 0.5 | Coverage at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny-mps-test | 1158 | 0.576 | 0.438 | 0.998 | 0.531 | 0.127 | 0.766 | 0.462 |
| base-mps-test | 1158 | 0.675 | 0.568 | 0.912 | 0.481 | 0.149 | 0.830 | 0.472 |
| smart-mps-test | 1158 | 0.616 | 0.550 | 0.963 | 0.523 | 0.086 | 0.730 | 0.441 |
| multilingual-mps-test | 1158 | 0.512 | 0.527 | 1.455 | 0.698 | 0.201 | 0.531 | 0.451 |
| decoder-mps-test | 1158 | 0.604 | 0.626 | 1.892 | 0.673 | 0.275 | 0.608 | 0.786 |

## Accuracy by family

| Report | agent_control | objective | robustness | verification |
| --- | ---: | ---: | ---: | ---: |
| tiny-mps-test | 0.628 (n=156) | 0.363 (n=364) | 0.381 (n=286) | 0.932 (n=352) |
| base-mps-test | 0.756 (n=156) | 0.563 (n=364) | 0.416 (n=286) | 0.966 (n=352) |
| smart-mps-test | 0.635 (n=156) | 0.533 (n=364) | 0.472 (n=286) | 0.810 (n=352) |
| multilingual-mps-test | 0.673 (n=156) | 0.585 (n=364) | 0.406 (n=286) | 0.452 (n=352) |
| decoder-mps-test | 0.481 (n=156) | 0.621 (n=364) | 0.556 (n=286) | 0.682 (n=352) |

## Objective accuracy by variant

| Variant | tiny-mps-test | base-mps-test | smart-mps-test | multilingual-mps-test | decoder-mps-test |
| --- | ---: | ---: | ---: | ---: | ---: |
| authority_injection | 0.340 | 0.426 | 0.255 | 0 | 0.085 |
| base | 0.681 | 0.777 | 0.670 | 0.574 | 0.638 |
| distractor_middle | 0.532 | 0.628 | 0.564 | 0.436 | 0.532 |
| distractor_start | 0.585 | 0.670 | 0.574 | 0.543 | 0.606 |
| irrelevant_context | 0.660 | 0.745 | 0.670 | 0.606 | 0.649 |
| long_context | 0.564 | 0.585 | 0.649 | 0.372 | 0.702 |
| lowercase | 0.628 | 0.787 | 0.670 | 0.606 | 0.660 |
| option_rotation | 0.548 | 0.661 | 0.565 | 0.645 | 0.694 |
| question_rewording | 0.435 | 0.581 | 0.581 | 0.645 | 0.710 |
| quoted_state | 0.585 | 0.745 | 0.745 | 0.574 | 0.691 |
| state_injection | 0.638 | 0.777 | 0.660 | 0.574 | 0.649 |
| unicode_context | 0.585 | 0.734 | 0.723 | 0.649 | 0.691 |
| uppercase | 0.649 | 0.628 | 0.649 | 0.521 | 0.617 |

## Other families

| Report | Ordinal exact | Ordinal within one | Ordinal |expected error| | Ranking NDCG | Ambiguous abstention | Policy agreement | Statements scored directly | Mean unsupported (target no / yes) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny-mps-test | 0.352 | 0.870 | 0.726 | 0.940 | 0.333 | 0.909 | 352 | 0.307 / 0.095 |
| base-mps-test | 0.417 | 0.852 | 0.695 | 0.874 | 0.573 | 0.909 | 352 | 0.384 / 0.050 |
| smart-mps-test | 0.606 | 0.898 | 0.491 | 0.926 | 0.083 | 0.727 | 0 |  |
| multilingual-mps-test | 0.366 | 0.755 | 0.828 | 0.983 | 0.177 | 0.705 | 0 |  |
| decoder-mps-test | 0.523 | 0.852 | 0.638 | 0.982 | 0.073 | 0.750 | 352 | 0.802 / 0.154 |

## Timing (in-process, per request, warm)

| Report | Cold load ms | p50 ms | p95 ms | p99 ms | Decisions/s | Peak RSS MiB (process lifetime) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny-mps-test | 3636 | 24.5 | 195.4 | 381.6 | 19.47 | 1017 |
| base-mps-test | 3552 | 28.7 | 183.4 | 383.4 | 16.04 | 622 |
| smart-mps-test | 4571 | 107.0 | 633.0 | 1485.3 | 4.18 | 3865 |
| multilingual-mps-test | 5305 | 58.8 | 499.7 | 960.2 | 6.34 | 728 |
| decoder-mps-test | 6467 | 158.1 | 419.4 | 444.6 | 5.15 | 4755 |

Latency is the wall time of one request batch of size one, including the inference lock; peak RSS is a process-lifetime high-water mark, not isolated model memory. Reports are descriptive measurements on correlated synthetic rows, not independent validation.
