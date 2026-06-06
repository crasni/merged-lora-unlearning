# Evaluation Metrics

## Pipeline Validation

| Metric | When used | Purpose |
|---|---|---|
| Normalized answer match | Base, acquisition, merge | Filter known facts and verify acquisition |
| ROUGE-L | Base, acquisition, merge | Detect partial answer recovery |
| Gold-answer probability | Base, acquisition, merge, unlearning | Detect behavioral changes before generation changes |
| Prediction agreement | Before and after merge | Verify merge preservation |

## MUSE Primary Evaluation

| Criterion | Metric | When used |
|---|---|---|
| C1 Verbatim memory | Prefix-completion ROUGE-L | Final forget evaluation |
| C2 Knowledge memory | QA ROUGE-L | Validation and final forget evaluation |
| C3 Privacy leakage | Min-K membership AUC relative to retain oracle | Final evaluation |
| C4 Utility preservation | Retain QA ROUGE-L and general utility | Validation and final evaluation |
| C5 Scalability | C1-C4 across increasing forget sizes | After the main MVP |
| C6 Sustainability | C1-C4 after sequential requests | After scalability |

## OpenUnlearning Supplementary Evaluation

Implemented in the MVP:

- Exact memorization
- Extraction strength
- Answer probability
- Truth ratio using generated alternate answers
- Loss and Min-K membership attacks

Min-K++ and zlib-normalized membership attacks are planned supplementary
metrics after the core workstation experiment is validated.

## Robustness Evaluation

Selected final checkpoints are evaluated on unseen paraphrases, Chinese and
mixed-language prompts, optional quantized models, and optional relearning
attacks. These results remain separate from the primary MUSE table.

## Checkpoint Selection

Every saved unlearning epoch is evaluated on `forget_validation` and
`retain_validation`. Among checkpoints satisfying the configured retain-match
floor, the checkpoint with the lowest forget match and ROUGE-L is selected.
Final test metrics never participate in checkpoint selection.
