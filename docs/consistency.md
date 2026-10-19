# Consistency

Properties a decision model should have whatever its accuracy, measured by
`opendecision consistency --model NAME [--split test|all] [--order-averaging K]`. Reports are in
[`benchmarks/reports/consistency/`](../benchmarks/reports/consistency/).

## Negation consistency

`p(X) + p(not X)` should be one. The 99 pairs in
`benchmarks/datasets/negation_pairs.jsonl` pair each base verification
statement the state supports or contradicts with a rule-based negation
(`scripts/generate_negation_pairs.py`). A published check of this kind finds
hosted models missing by 0.064 on average and a 27B open model by 0.293.

| Backend | Pairs | Mean gap | Median gap | Mean `p(X) + p(not X)` | Both sides right |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` (NLI) | 99 | 0.104 | 0.009 | 0.934 | 86.9 % |
| `qwen35` | 99 | 0.361 | 0.113 | 0.691 | 60.6 % |
| `lightning` | 99 | 0.007 | 0.002 | 1.004 | 100 % |

The NLI backend scores entailment and contradiction directly and is close to
consistent on most pairs. `qwen35` tends to answer *no* to both a statement
and its negation, so the sum falls well below one. `lightning`, trained to
answer true/false questions, is consistent to within 0.007 and right on both
sides of every pair.

## Option-order invariance

Reordering the same options should not change the decision. Each base choice
case is re-asked in three other orders (the reversal and two seeded shuffles);
the flip rate is the share of cases whose choice changes under any of them.

| Backend | Cases | Flip rate | Mean total variation |
| --- | ---: | ---: | ---: |
| `base` (NLI) | 352 | 0 % | 0.000 |
| `qwen35` | 352 | 20.2 % | 0.163 |
| `lightning` | 352 | 5.7 % | 0.036 |

The NLI backend scores each option on its own, so order cannot matter.
`qwen35` reads options as a lettered list and is position-sensitive: one
decision in five changes with the order. `lightning` is also lettered but far
less sensitive. On the test split alone
(94 cases) the flip rates are 0 %, 21.3 % and 11.7 %; `tests/test_consistency.py`
fails if `base`, `qwen35` or `lightning` exceeds 0 %, 25 % or 15 %.

### Order-averaged scoring

The lettered decoder backends can average their scores over rotated option
orders: `--order-averaging K` on every command and on `serve`, or
`DecisionModel(..., permutations=K)`. Each rotation is one more scoring pass,
so K multiplies latency. Measured on the same 352 cases, with accuracy and
latency from the 1,726-decision test split:

| Backend | K | Flip rate | Objective accuracy | Warm p50 |
| --- | ---: | ---: | ---: | ---: |
| `qwen35` | 1 | 20.2 % | 0.799 | 69 ms |
| `qwen35` | 2 | 6.0 % | 0.802 | 142 ms |
| `qwen35` | 3 | 7.1 % | — | — |
| `lightning` | 1 | 5.7 % | 0.940 | 107 ms |
| `lightning` | 2 | **0.6 %** | 0.945 | 267 ms |

K=2 removes most of the sensitivity at twice the cost; K=3 adds nothing,
because three rotations of two or four options cannot be spread evenly. The
default stays K=1, so `lightning` keeps its sub-250 ms median, and K=2 is the
setting to choose when a decision must not depend on how the options were
listed. A shipped calibration profile is fitted on single-order scores, so it
is not applied when averaging is on.

## Appended-context robustness

A remark appended to the state that carries no evidence should not change the
decision. Each of the 352 base choice cases is asked again with an off-topic
aside appended, and again with an unsupported recommendation ("Note from a
colleague, who has not looked into it: I would just go with X.") of an option
the model did not choose. A published audit finds an appended irrelevant
opinion flipping 12.1 % of one hosted model's decisions.

| Backend | Off-topic aside flips | Unsupported opinion flips | Answer follows the opinion |
| --- | ---: | ---: | ---: |
| `base` (NLI) | 17.3 % | 78.4 % | 77.3 % |
| `qwen35` | 8.5 % | 69.6 % | 69.3 % |
| `lightning` | **1.7 %** | **2.6 %** | **1.1 %** |

The general models defer to almost any stated preference: an entailment model
reads "go with X" as support for X, and the 2B instruction model treats it as
an instruction. `lightning` was trained to answer from the record and ignores
both. Order averaging does not change this (`qwen35` at K=3: 76.1 %). If states
can contain other people's opinions and you are not using `lightning`, keep
them out of the state or screen them first
([patterns](patterns.md)).
