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

Data used for optimization is disjoint from final evaluation data:

```text
forget_train       unlearning objective
forget_validation  checkpoint selection
forget_test        final evaluation

retain_regularize  retain NLL or KL objective
retain_validation  checkpoint selection
retain_test        final evaluation

holdout            never used for acquisition or unlearning
```

Acquisition QA prompts and final QA prompts are distinct. Final prompt variants
are generated before training and remain immutable. Every processed artifact has
a SHA-256 digest in the run manifest.

## Stage Gates

The experiment stops when a required assumption fails:

1. Base knowledge on candidate facts must be below the configured threshold.
2. Acquisition accuracy must exceed the configured threshold.
3. Merged-model behavior must remain within the configured merge delta.
4. Checkpoints are selected using validation data only.

## Primary Benchmark

MUSE is the primary evaluation benchmark:

- C1: no verbatim memorization
- C2: no knowledge memorization
- C3: no privacy leakage
- C4: utility preservation
- C5: scalability
- C6: sustainability

OpenUnlearning metrics supplement MUSE but do not replace it.
