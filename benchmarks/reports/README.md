# Benchmark report matrix

Split `test`. Every report below was run on the rows whose dataset hash it records; a differing hash means a different corpus and the rows are not comparable.

| Report | Model | Revision | Device | Rows | Dataset SHA-256 (prefix) | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny-mps-test | cross-encoder/nli-deberta-v3-xsmall | a15087641532 | mps | 1726 | 43c4923ea723 | measured |
| base-mps-test | tasksource/ModernBERT-base-nli | de4ab7e77845 | mps | 1726 | 43c4923ea723 | measured |
| smart-mps-test | Skywork/Skywork-Reward-V2-Qwen3-0.6B | 8c14a4e9e632 | mps | 1726 | 43c4923ea723 | measured |
| multilingual-mps-test | BAAI/bge-reranker-v2-m3 | 953dc6f6f85a | mps | 1726 | 43c4923ea723 | measured |
| decoder-mps-test | Qwen/Qwen3-0.6B | c1899de289a0 | mps | 1726 | 43c4923ea723 | measured |
| lfm25-mps-test | LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit | 7ccafdb04c36 | mps | 1726 | 43c4923ea723 | measured |
| qwen35-mps-test | mlx-community/Qwen3.5-2B-4bit | 674aaa7240b9 | mps | 1726 | 43c4923ea723 | measured |
| qwen35-4b-mps-test | mlx-community/Qwen3.5-4B-4bit | 0e7ffd5c629e | mps | 1726 | 43c4923ea723 | measured |
| base-mps-test-calibrated | tasksource/ModernBERT-base-nli | de4ab7e77845 | mps | 1726 | 43c4923ea723 | measured |
| decoder-mps-test-calibrated | Qwen/Qwen3-0.6B | c1899de289a0 | mps | 1726 | 43c4923ea723 | measured |

## Objective decisions (pooled objective, agent control, verification, robustness)

| Report | Count | Accuracy | Balanced | NLL | Brier | ECE | Acc. at margin ≥ 0.5 | Coverage at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny-mps-test | 1158 | 0.571 | 0.432 | 0.998 | 0.531 | 0.124 | 0.761 | 0.462 |
| base-mps-test | 1158 | 0.675 | 0.569 | 0.912 | 0.481 | 0.154 | 0.828 | 0.471 |
| smart-mps-test | 1158 | 0.617 | 0.551 | 0.960 | 0.522 | 0.102 | 0.728 | 0.438 |
| multilingual-mps-test | 1158 | 0.509 | 0.524 | 1.453 | 0.698 | 0.204 | 0.535 | 0.459 |
| decoder-mps-test | 1158 | 0.608 | 0.627 | 1.888 | 0.670 | 0.277 | 0.615 | 0.792 |
| lfm25-mps-test | 1158 | 0.596 | 0.592 | 3.037 | 0.748 | 0.351 | 0.609 | 0.923 |
| qwen35-mps-test | 1158 | 0.799 | 0.729 | 0.580 | 0.299 | 0.084 | 0.861 | 0.800 |
| qwen35-4b-mps-test | 1158 | 0.853 | 0.819 | 0.785 | 0.245 | 0.096 | 0.882 | 0.926 |
| base-mps-test-calibrated | 1158 | 0.675 | 0.569 | 1.038 | 0.522 | 0.183 | 0.590 | 0.120 |
| decoder-mps-test-calibrated | 1158 | 0.608 | 0.627 | 0.872 | 0.527 | 0.083 | 0.600 | 0.190 |

## Accuracy by family

| Report | agent_control | objective | robustness | verification |
| --- | ---: | ---: | ---: | ---: |
| tiny-mps-test | 0.635 (n=156) | 0.346 (n=364) | 0.378 (n=286) | 0.932 (n=352) |
| base-mps-test | 0.750 (n=156) | 0.569 (n=364) | 0.416 (n=286) | 0.963 (n=352) |
| smart-mps-test | 0.628 (n=156) | 0.538 (n=364) | 0.472 (n=286) | 0.812 (n=352) |
| multilingual-mps-test | 0.679 (n=156) | 0.582 (n=364) | 0.402 (n=286) | 0.446 (n=352) |
| decoder-mps-test | 0.487 (n=156) | 0.618 (n=364) | 0.552 (n=286) | 0.696 (n=352) |
| lfm25-mps-test | 0.840 (n=156) | 0.462 (n=364) | 0.636 (n=286) | 0.594 (n=352) |
| qwen35-mps-test | 0.814 (n=156) | 0.813 (n=364) | 0.538 (n=286) | 0.989 (n=352) |
| qwen35-4b-mps-test | 0.801 (n=156) | 0.854 (n=364) | 0.720 (n=286) | 0.983 (n=352) |
| base-mps-test-calibrated | 0.750 (n=156) | 0.569 (n=364) | 0.416 (n=286) | 0.963 (n=352) |
| decoder-mps-test-calibrated | 0.487 (n=156) | 0.618 (n=364) | 0.552 (n=286) | 0.696 (n=352) |

## Objective accuracy by variant

| Variant | tiny-mps-test | base-mps-test | smart-mps-test | multilingual-mps-test | decoder-mps-test | lfm25-mps-test | qwen35-mps-test | qwen35-4b-mps-test | base-mps-test-calibrated | decoder-mps-test-calibrated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| authority_injection | 0.340 | 0.436 | 0.277 | 0 | 0.064 | 0.266 | 0.372 | 0.340 | 0.436 | 0.064 |
| base | 0.660 | 0.766 | 0.660 | 0.585 | 0.660 | 0.670 | 0.862 | 0.915 | 0.766 | 0.660 |
| distractor_middle | 0.521 | 0.638 | 0.574 | 0.426 | 0.553 | 0.447 | 0.723 | 0.809 | 0.638 | 0.553 |
| distractor_start | 0.596 | 0.649 | 0.564 | 0.532 | 0.617 | 0.543 | 0.809 | 0.872 | 0.649 | 0.617 |
| irrelevant_context | 0.660 | 0.734 | 0.660 | 0.606 | 0.649 | 0.628 | 0.872 | 0.926 | 0.734 | 0.649 |
| long_context | 0.564 | 0.574 | 0.670 | 0.372 | 0.691 | 0.553 | 0.872 | 0.904 | 0.574 | 0.691 |
| lowercase | 0.617 | 0.798 | 0.681 | 0.606 | 0.660 | 0.649 | 0.862 | 0.936 | 0.798 | 0.660 |
| option_rotation | 0.516 | 0.645 | 0.548 | 0.645 | 0.694 | 0.742 | 0.790 | 0.887 | 0.645 | 0.694 |
| question_rewording | 0.435 | 0.629 | 0.565 | 0.645 | 0.710 | 0.677 | 0.790 | 0.887 | 0.629 | 0.710 |
| quoted_state | 0.596 | 0.734 | 0.745 | 0.574 | 0.691 | 0.702 | 0.862 | 0.904 | 0.734 | 0.691 |
| state_injection | 0.628 | 0.777 | 0.670 | 0.564 | 0.670 | 0.660 | 0.819 | 0.862 | 0.777 | 0.670 |
| unicode_context | 0.574 | 0.745 | 0.734 | 0.628 | 0.691 | 0.681 | 0.862 | 0.947 | 0.745 | 0.691 |
| uppercase | 0.649 | 0.628 | 0.638 | 0.532 | 0.617 | 0.606 | 0.883 | 0.926 | 0.628 | 0.617 |

## Objective accuracy by candidate count

Accuracy, then the share of rows clearing a 0.5 margin and their accuracy.
A per-count calibration profile makes that margin incomparable between counts.

| Report | k=2 | k=4 | k=12 |
| --- | ---: | ---: | ---: |
| tiny-mps-test | 0.720 · 0.671 cov @ 0.823 | 0.404 · 0.205 cov @ 0.406 | 0.244 · 0.051 cov @ 0 |
| base-mps-test | 0.765 · 0.729 cov @ 0.861 | 0.596 · 0.135 cov @ 0.429 | 0.436 · 0.000 cov |
| smart-mps-test | 0.671 · 0.499 cov @ 0.680 | 0.676 · 0.519 cov @ 0.833 | 0.263 · 0.006 cov @ 0 |
| multilingual-mps-test | 0.483 · 0.362 cov @ 0.396 | 0.571 · 0.724 cov @ 0.628 | 0.506 · 0.353 cov @ 0.782 |
| decoder-mps-test | 0.630 · 0.796 cov @ 0.628 | 0.513 · 0.792 cov @ 0.506 | 0.699 · 0.776 cov @ 0.777 |
| lfm25-mps-test | 0.686 · 0.962 cov @ 0.685 | 0.522 · 0.926 cov @ 0.519 | 0.346 · 0.744 cov @ 0.397 |
| qwen35-mps-test | 0.841 · 0.819 cov @ 0.903 | 0.728 · 0.827 cov @ 0.748 | 0.756 · 0.660 cov @ 0.913 |
| qwen35-4b-mps-test | 0.912 · 0.942 cov @ 0.938 | 0.744 · 0.929 cov @ 0.762 | 0.814 · 0.846 cov @ 0.871 |
| base-mps-test-calibrated | 0.765 · 0.000 cov | 0.596 · 0.413 cov @ 0.597 | 0.436 · 0.064 cov @ 0.500 |
| decoder-mps-test-calibrated | 0.630 · 0.057 cov @ 0.744 | 0.513 · 0.404 cov @ 0.476 | 0.699 · 0.353 cov @ 0.782 |

## Other families

| Report | Ordinal exact | Ordinal within one | Ordinal |expected error| | Ranking NDCG | Ambiguous abstention | Policy agreement | Statements scored directly | Mean unsupported (target no / yes) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny-mps-test | 0.361 | 0.861 | 0.726 | 0.940 | 0.333 | 0.909 | 352 | 0.307 / 0.095 |
| base-mps-test | 0.421 | 0.861 | 0.694 | 0.867 | 0.562 | 0.909 | 352 | 0.385 / 0.050 |
| smart-mps-test | 0.611 | 0.898 | 0.495 | 0.926 | 0.083 | 0.727 | 0 |  |
| multilingual-mps-test | 0.370 | 0.755 | 0.827 | 0.983 | 0.177 | 0.705 | 0 |  |
| decoder-mps-test | 0.509 | 0.843 | 0.641 | 0.982 | 0.094 | 0.750 | 352 | 0.807 / 0.155 |
| lfm25-mps-test | 0.509 | 0.778 | 0.696 | 0.952 | 0.010 | 0.909 | 352 | 0.013 / 0.002 |
| qwen35-mps-test | 0.481 | 0.856 | 0.655 | 0.984 | 0.094 | 0.886 | 352 | 0.080 / 0.019 |
| qwen35-4b-mps-test | 0.769 | 0.954 | 0.313 | 0.942 | 0.083 | 0.841 | 352 | 0.349 / 0.045 |
| base-mps-test-calibrated | 0.421 | 0.861 | 0.683 | 0.867 | 0.990 | 0.909 | 352 | 0.385 / 0.050 |
| decoder-mps-test-calibrated | 0.509 | 0.843 | 0.649 | 0.982 | 0.771 | 0.750 | 352 | 0.807 / 0.155 |

## Timing (in-process, per request, warm)

| Report | Cold load ms | p50 ms | p95 ms | p99 ms | Decisions/s | Peak RSS MiB (process lifetime) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny-mps-test | 2512 | 11.0 | 101.5 | 179.0 | 36.50 | 1158 |
| base-mps-test | 2077 | 13.9 | 75.7 | 133.7 | 38.29 | 1269 |
| smart-mps-test | 2264 | 42.1 | 216.7 | 476.3 | 11.70 | 854 |
| multilingual-mps-test | 3642 | 19.6 | 120.7 | 224.8 | 23.34 | 3938 |
| decoder-mps-test | 2663 | 65.9 | 127.1 | 131.3 | 13.54 | 1090 |
| lfm25-mps-test | 2385 | 46.1 | 105.7 | 255.8 | 17.38 | 1101 |
| qwen35-mps-test | 2297 | 69.5 | 166.3 | 280.8 | 11.86 | 1866 |
| qwen35-4b-mps-test | 2941 | 135.4 | 1002.1 | 1667.4 | 4.08 | 3115 |
| base-mps-test-calibrated | 2698 | 14.0 | 74.6 | 134.5 | 39.86 | 1268 |
| decoder-mps-test-calibrated | 2500 | 65.8 | 127.0 | 131.3 | 13.83 | 1081 |

Latency is the wall time of one request batch of size one, including the inference lock; peak RSS is a process-lifetime high-water mark, not isolated model memory. Reports are descriptive measurements on correlated synthetic rows, not independent validation.
