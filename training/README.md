# Training and distillation research

The runtime intentionally ships no newly trained weights. The baseline must be
measured before fine-tuning or distillation so that changes can be attributed to
data, objective, and checkpoint rather than intuition.

Any future training run must:

1. load only `train` rows for gradient updates;
2. use `validation` to select hyperparameters, thresholds, and early stopping;
3. fit calibration on `calibration` after the candidate checkpoint is frozen;
4. evaluate `test` once as a final report, with semantic groups kept intact;
5. record source checkpoint revision, dataset hashes, code revision, tokenizer,
   sequence limit, precision, device, random seeds, and all hyperparameters;
6. save newly trained artifacts under a new content-addressed identity and never
   silently replace a registered upstream model.

The synthetic corpus includes correlated deterministic variants. A script must
reject group overlap across splits before training. Subjective cases require an
explicit policy target; they must not be reported as universal moral accuracy.
Distillation may use a local teacher's scores as soft targets, but proprietary
or remote teacher data is not required by the runtime and must not be bundled
without provenance and license review.

Training is intentionally a documented next phase until a measured error analysis
shows a benefit. See [the roadmap](../docs/roadmap.md) and [the requirements map](../docs/requirements.md).
