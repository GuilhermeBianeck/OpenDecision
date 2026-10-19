# Consistency

Two properties a decision model should have whatever its accuracy, measured by
`opendecision consistency --model NAME [--split test|all]`. Reports are in
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
decision in five changes with the order. Order-averaged scoring (ROB-2 in
[the ecosystem requirements](ecosystem-requirements.md)) is the planned
remedy. `lightning` is also lettered but far less sensitive. On the test split alone
(94 cases) the flip rates are 0 %, 21.3 % and 11.7 %; `tests/test_consistency.py`
fails if `base`, `qwen35` or `lightning` exceeds 0 %, 25 % or 15 %.
