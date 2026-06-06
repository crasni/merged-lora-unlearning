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

- `retain_loss: kl` preserves target behavior without directly reinforcing
  retained answers through NLL.
- `learning_rate: 0.0001` is 10 times the baseline unlearning rate.
- `forget_weight: 5.0` gives the weak forget gradient more influence.
- `epochs: 10` creates a wider validation-only checkpoint trajectory.
- `retain_match_floor: 0.8` prevents selecting a checkpoint that forgets by
  broadly damaging retained knowledge.
- `simnpo` is included as an additional reference-free method.

This profile is intentionally a stronger intervention, not a guaranteed best
configuration. Compare its selected checkpoints against both the merged target
and retain-only oracle.

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
