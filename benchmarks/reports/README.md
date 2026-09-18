# Benchmark report matrix

Split `test`. Every report below was run on the rows whose dataset hash it records; a differing hash means a different corpus and the rows are not comparable.

| Report | Model | Revision | Device | Rows | Dataset SHA-256 (prefix) | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny-mps-test | cross-encoder/nli-deberta-v3-xsmall | a15087641532 | mps | 1726 | 43c4923ea723 | measured |
| base-mps-test | tasksource/ModernBERT-base-nli | de4ab7e77845 | mps | 1726 | 43c4923ea723 | measured |
| smart-mps-test | Skywork/Skywork-Reward-V2-Qwen3-0.6B | 8c14a4e9e632 | mps | 1726 | 43c4923ea723 | measured |
| multilingual-mps-test | BAAI/bge-reranker-v2-m3 | 953dc6f6f85a | mps | 1726 | 43c4923ea723 | measured |
| decoder-mps-test | Qwen/Qwen3-0.6B | c1899de289a0 | mps | 1726 | 43c4923ea723 | measured |
| base-mps-test-calibrated | tasksource/ModernBERT-base-nli | de4ab7e77845 | mps | 1726 | 43c4923ea723 | measured |
| decoder-mps-test-calibrated | Qwen/Qwen3-0.6B | c1899de289a0 | mps | 1726 | 43c4923ea723 | measured |

## Objective decisions (pooled objective, agent control, verification, robustness)

| Report | Count | Accuracy | Balanced | NLL | Brier | ECE | Acc. at margin ≥ 0.5 | Coverage at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny-mps-test | 1158 | 0.576 | 0.438 | 0.998 | 0.531 | 0.127 | 0.766 | 0.462 |
| base-mps-test | 1158 | 0.675 | 0.568 | 0.912 | 0.481 | 0.149 | 0.830 | 0.472 |
| smart-mps-test | 1158 | 0.616 | 0.550 | 0.963 | 0.523 | 0.086 | 0.730 | 0.441 |
| multilingual-mps-test | 1158 | 0.512 | 0.527 | 1.455 | 0.698 | 0.201 | 0.531 | 0.451 |
| decoder-mps-test | 1158 | 0.604 | 0.626 | 1.892 | 0.673 | 0.275 | 0.608 | 0.786 |
| base-mps-test-calibrated | 1158 | 0.675 | 0.568 | 1.039 | 0.523 | 0.184 | 0.586 | 0.125 |
| decoder-mps-test-calibrated | 1158 | 0.604 | 0.626 | 0.874 | 0.528 | 0.077 | 0.598 | 0.185 |

## Accuracy by family

| Report | agent_control | objective | robustness | verification |
| --- | ---: | ---: | ---: | ---: |
| tiny-mps-test | 0.628 (n=156) | 0.363 (n=364) | 0.381 (n=286) | 0.932 (n=352) |
| base-mps-test | 0.756 (n=156) | 0.563 (n=364) | 0.416 (n=286) | 0.966 (n=352) |
| smart-mps-test | 0.635 (n=156) | 0.533 (n=364) | 0.472 (n=286) | 0.810 (n=352) |
| multilingual-mps-test | 0.673 (n=156) | 0.585 (n=364) | 0.406 (n=286) | 0.452 (n=352) |
| decoder-mps-test | 0.481 (n=156) | 0.621 (n=364) | 0.556 (n=286) | 0.682 (n=352) |
| base-mps-test-calibrated | 0.756 (n=156) | 0.563 (n=364) | 0.416 (n=286) | 0.966 (n=352) |
| decoder-mps-test-calibrated | 0.481 (n=156) | 0.621 (n=364) | 0.556 (n=286) | 0.682 (n=352) |

## Objective accuracy by variant

| Variant | tiny-mps-test | base-mps-test | smart-mps-test | multilingual-mps-test | decoder-mps-test | base-mps-test-calibrated | decoder-mps-test-calibrated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| authority_injection | 0.340 | 0.426 | 0.255 | 0 | 0.085 | 0.426 | 0.085 |
| base | 0.681 | 0.777 | 0.670 | 0.574 | 0.638 | 0.777 | 0.638 |
| distractor_middle | 0.532 | 0.628 | 0.564 | 0.436 | 0.532 | 0.628 | 0.532 |
| distractor_start | 0.585 | 0.670 | 0.574 | 0.543 | 0.606 | 0.670 | 0.606 |
| irrelevant_context | 0.660 | 0.745 | 0.670 | 0.606 | 0.649 | 0.745 | 0.649 |
| long_context | 0.564 | 0.585 | 0.649 | 0.372 | 0.702 | 0.585 | 0.702 |
| lowercase | 0.628 | 0.787 | 0.670 | 0.606 | 0.660 | 0.787 | 0.660 |
| option_rotation | 0.548 | 0.661 | 0.565 | 0.645 | 0.694 | 0.661 | 0.694 |
| question_rewording | 0.435 | 0.581 | 0.581 | 0.645 | 0.710 | 0.581 | 0.710 |
| quoted_state | 0.585 | 0.745 | 0.745 | 0.574 | 0.691 | 0.745 | 0.691 |
| state_injection | 0.638 | 0.777 | 0.660 | 0.574 | 0.649 | 0.777 | 0.649 |
| unicode_context | 0.585 | 0.734 | 0.723 | 0.649 | 0.691 | 0.734 | 0.691 |
| uppercase | 0.649 | 0.628 | 0.649 | 0.521 | 0.617 | 0.628 | 0.617 |

## Objective accuracy by candidate count

Accuracy, then the share of rows clearing a 0.5 margin and their accuracy.
A per-count calibration profile makes that margin incomparable between counts.

| Report | k=2 | k=4 | k=12 |
| --- | ---: | ---: | ---: |
| tiny-mps-test | 0.720 · 0.671 cov @ 0.829 | 0.420 · 0.205 cov @ 0.406 | 0.250 · 0.051 cov @ 0 |
| base-mps-test | 0.767 · 0.730 cov @ 0.863 | 0.596 · 0.135 cov @ 0.429 | 0.429 · 0.000 cov |
| smart-mps-test | 0.671 · 0.499 cov @ 0.683 | 0.670 · 0.529 cov @ 0.836 | 0.263 · 0.013 cov @ 0 |
| multilingual-mps-test | 0.486 · 0.361 cov @ 0.398 | 0.571 · 0.718 cov @ 0.629 | 0.513 · 0.314 cov @ 0.755 |
| decoder-mps-test | 0.626 · 0.784 cov @ 0.621 | 0.510 · 0.795 cov @ 0.500 | 0.699 · 0.776 cov @ 0.769 |
| base-mps-test-calibrated | 0.767 · 0.000 cov | 0.596 · 0.426 cov @ 0.602 | 0.429 · 0.077 cov @ 0.417 |
| decoder-mps-test-calibrated | 0.626 · 0.045 cov @ 0.742 | 0.510 · 0.407 cov @ 0.488 | 0.699 · 0.359 cov @ 0.768 |

## Other families

| Report | Ordinal exact | Ordinal within one | Ordinal |expected error| | Ranking NDCG | Ambiguous abstention | Policy agreement | Statements scored directly | Mean unsupported (target no / yes) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny-mps-test | 0.352 | 0.870 | 0.726 | 0.940 | 0.333 | 0.909 | 352 | 0.307 / 0.095 |
| base-mps-test | 0.417 | 0.852 | 0.695 | 0.874 | 0.573 | 0.909 | 352 | 0.384 / 0.050 |
| smart-mps-test | 0.606 | 0.898 | 0.491 | 0.926 | 0.083 | 0.727 | 0 |  |
| multilingual-mps-test | 0.366 | 0.755 | 0.828 | 0.983 | 0.177 | 0.705 | 0 |  |
| decoder-mps-test | 0.523 | 0.852 | 0.638 | 0.982 | 0.073 | 0.750 | 352 | 0.802 / 0.154 |
| base-mps-test-calibrated | 0.417 | 0.852 | 0.683 | 0.874 | 1 | 0.909 | 352 | 0.384 / 0.050 |
| decoder-mps-test-calibrated | 0.523 | 0.852 | 0.646 | 0.982 | 0.802 | 0.750 | 352 | 0.802 / 0.154 |

## Timing (in-process, per request, warm)

| Report | Cold load ms | p50 ms | p95 ms | p99 ms | Decisions/s | Peak RSS MiB (process lifetime) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny-mps-test | 1894 | 14.5 | 140.5 | 239.5 | 28.27 | 860 |
| base-mps-test | 2320 | 23.4 | 150.1 | 309.3 | 19.96 | 619 |
| smart-mps-test | 4258 | 77.8 | 483.8 | 996.6 | 5.68 | 3511 |
| multilingual-mps-test | 3720 | 40.1 | 345.9 | 648.1 | 9.20 | 728 |
| decoder-mps-test | 4154 | 118.5 | 301.9 | 311.1 | 6.97 | 4755 |
| base-mps-test-calibrated | 2489 | 22.3 | 132.2 | 267.3 | 21.93 | 619 |
| decoder-mps-test-calibrated | 3722 | 118.4 | 301.8 | 311.1 | 6.99 | 4755 |

Latency is the wall time of one request batch of size one, including the inference lock; peak RSS is a process-lifetime high-water mark, not isolated model memory. Reports are descriptive measurements on correlated synthetic rows, not independent validation.
