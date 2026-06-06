# Running Experiments

The repository has two portable experiment profiles:

| Profile | Model | Intended use |
|---|---|---|
| `0_5b` | `Qwen/Qwen2.5-0.5B-Instruct` | Faster and cheaper development run |
| `1_5b` | `Qwen/Qwen2.5-1.5B-Instruct` | Main higher-capacity run |
| `1_5b_unlearning` | `Qwen/Qwen2.5-1.5B-Instruct` | Stronger unlearning follow-up |

All paths are relative to the repository. Clone the repository onto persistent
storage when using an ephemeral GPU platform.

## Setup

From the repository root:

```bash
pip install uv
uv sync --extra train --extra dev
uv run pytest
nvidia-smi
```

## Full Run

```bash
scripts/run.sh 0_5b
scripts/run.sh 1_5b
scripts/run.sh 1_5b_unlearning
```

The profile argument is required. Running without one prints the available
profiles and exits.

## Stronger Unlearning Follow-Up

The baseline `1_5b` run acquired and merged facts successfully, but GA and NPO
did not reduce forget-set accuracy. The `1_5b_unlearning` profile changes only
the unlearning experiment. It declares `reuse_from: 1_5b`, validates that the
seed, model, data, acquisition, and evaluation settings match, then reuses the
baseline data, acquisition target, retain oracle, and pre-unlearning
evaluations. Only the stronger unlearning methods and their evaluations run.

Run the baseline first and keep its output directory available:

```bash
scripts/run.sh 1_5b
scripts/run.sh 1_5b_unlearning
```

The follow-up reads the large merged target directly from the baseline run
without duplicating it. Small data and baseline evaluation artifacts are copied
into the follow-up run so later commands cannot mutate the baseline through
writable symlinks. The stronger run fails instead of reusing artifacts when
relevant settings differ, baseline stages are incomplete, or expected files
are missing. Successful reuse preparation is idempotent, so an interrupted
unlearning run can be retried.

The stronger unlearning changes are:

- `learning_rate: 0.0001` is 10 times the baseline unlearning rate.
- `gamma: 5.0` gives the forget objective more influence.
- `alpha: 1.0` scales regularizers for variants that include one.
- `epochs: 10` gives the pure objectives enough training exposure.
- `checkpoint_every_epochs: 5` evaluates only epochs 5 and 10, avoiding
  expensive generation-based validation after every epoch.
- `retain_match_floor: 0.8` prevents selecting a checkpoint that forgets by
  broadly damaging retained knowledge.
- Pure GA, NPO, and SimNPO are compared before testing regularizers.

This profile is intentionally a stronger intervention, not a guaranteed best
configuration. Compare its selected checkpoints against both the merged target
and retain-only oracle.

For the complete controlled comparison, run:

```bash
scripts/run.sh 1_5b_full
```

`1_5b_full` uses method-specific, source-backed starting points:

| Family | Learning rate | Epochs | Beta | Forget / retain weights | Source |
|---|---:|---:|---:|---:|---|
| GA and regularized GA | `1e-5` | `10` | N/A | `1.0 / 1.0` | Original MUSE baselines |
| NPO and regularized NPO | `3e-5` | `10` | `0.1` | `1.0 / 1.0` | OpenUnlearning MUSE example |
| SimNPO | `1e-5` | `10` | `0.7` | `1.0 / N/A` | Official SimNPO MUSE-News command |
| Regularized SimNPO | `1e-5` | `10` | `0.7` | `1.0 / 0.1` | Official SimNPO MUSE-News GDR command |

The SimNPO repository publishes GDR, not KL, for MUSE. `simnpo_kl` uses the
published GDR retain coefficient as a controlled extrapolation. These settings
are not claimed optimal for Qwen-1.5B LoRA: the sources use Llama-2-7B full
fine-tuning and much larger MUSE corpora. A setting is considered useful only
when its selected checkpoint lowers validation forget metrics while meeting
`retain_match_floor: 0.8`; final conclusions must use the held-out test report.

## Unlearning Methods

Method names define the exact algorithm variant:

| Method | Forget objective | Retain regularizer |
|---|---|---|
| `ga` | Gradient ascent | None |
| `grad_diff` | Gradient ascent | Retain NLL |
| `ga_kl` | Gradient ascent | Retain KL |
| `npo` | Negative preference optimization | None |
| `npo_grad_diff` | Negative preference optimization | Retain NLL |
| `npo_kl` | Negative preference optimization | Retain KL |
| `simnpo` | Length-normalized reference-free SimNPO | None |
| `simnpo_grad_diff` | SimNPO | Retain NLL |
| `simnpo_kl` | SimNPO | Retain KL |

`gamma` scales the forget objective and `alpha` scales the retain regularizer,
matching OpenUnlearning terminology. Pure methods do not apply a retain
regularizer. NPO and KL variants load a frozen target reference model; pure GA,
GradDiff, and SimNPO variants do not load an unnecessary reference model.
`method_overrides` can replace `epochs`, `learning_rate`, `batch_size`, `beta`,
`simnpo_delta`, `gamma`, or `alpha` for a named method.
`checkpoint_every_epochs` controls how often checkpoints are saved and
validation-selected. Larger values reduce evaluation time but provide a
coarser forgetting/retention trajectory.

Outputs and logs are written under `outputs/runs/<profile>/` and
`outputs/logs/<profile>.log`.

## Individual Stages

Use the same profile config for every command:

```bash
CONFIG=configs/experiments/1_5b.yaml

uv run mlu data -c "$CONFIG"
uv run mlu filter -c "$CONFIG"
uv run mlu acquire -c "$CONFIG"
uv run mlu eval --model acquisition_adapter -c "$CONFIG"
uv run mlu oracle -c "$CONFIG"
uv run mlu unlearn npo -c "$CONFIG"
uv run mlu eval --model npo -c "$CONFIG"
uv run mlu report -c "$CONFIG"
uv run mlu status -c "$CONFIG"
```

Use a different experiment name before rerunning modified settings so previous
results are not overwritten.
