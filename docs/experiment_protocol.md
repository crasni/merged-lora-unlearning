# Experiment Protocol

## Scientific Question

After facts learned through LoRA have been merged into a model, can an
unlearning method remove a selected subset while preserving the remaining
LoRA-acquired facts and general utility?

## Models

Each experiment compares four model roles:

1. **Base**: the untouched pretrained model.
2. **Target**: acquisition LoRA trained on all acquired facts, then merged.
3. **Retain oracle**: an independently trained LoRA that sees only retained
   acquisition facts, then merged.
4. **Unlearned**: a model produced by applying an unlearning method to Target.

The retain oracle is the behavioral target for approximate unlearning. It is not
used as an input to the unlearning algorithm.

## Data Separation

The deletion request is fact-level. Every requested forget fact is supplied to
the unlearning algorithm, while checkpoint selection and final evaluation use
different prompt views of those same facts:

```text
forget_request     unlearning objective using training QA + statements
forget_request     checkpoint selection using selection-only QA
forget_request     final evaluation using original evaluation QA

retain_regularize  retain NLL or KL objective
retain_validation  checkpoint selection
retain_test        final evaluation

holdout            never used for acquisition or unlearning
```

The legacy `forget_train`, `forget_validation`, and `forget_test` files remain
as diagnostic partitions but do not define the deletion request. Acquisition QA,
selection prompts, and final QA prompts are distinct. Final prompt variants are
generated before training and remain immutable. Every processed artifact has a
SHA-256 digest in the run manifest.

## Stage Gates

The experiment stops when a required assumption fails:

1. Base knowledge on candidate facts must be below the configured threshold.
2. Acquisition accuracy must exceed the configured threshold.
3. Merged-model behavior must remain within the configured merge delta.
4. Checkpoints are selected using selection-only forget prompts and
   `retain_validation`; original forget prompts remain final-evaluation-only.

The base-filter stage uses batched deterministic generation. It writes each
completed batch to `base_filter_predictions.jsonl`, displays live progress and
known-fact rate, and resumes from that file after interruption.

## Primary Benchmark

MUSE is the primary evaluation benchmark:

- C1: no verbatim memorization
- C2: no knowledge memorization
- C3: no privacy leakage
- C4: utility preservation
- C5: scalability
- C6: sustainability

OpenUnlearning metrics supplement MUSE but do not replace it.

## Unlearning Algorithms

Experiments report pure GA, NPO, and SimNPO separately from regularized
variants. GradDiff uses GA plus retain NLL. `_grad_diff` variants add retain NLL
to NPO or SimNPO, while `_kl` variants add frozen-target KL regularization.
Method names therefore identify the actual optimization objective used.
