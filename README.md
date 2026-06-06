# Merged LoRA Unlearning

This repository evaluates whether facts acquired through LoRA can be selectively
unlearned after the acquisition adapter has been merged into the base model.

The experiment keeps every stage inspectable:

```text
base model
  -> generate and filter synthetic facts
  -> train and merge acquisition LoRA
  -> train and merge retain-only oracle LoRA
  -> unlearn selected facts from the merged target
  -> evaluate with MUSE plus OpenUnlearning supplementary metrics
```

## Quick Start

Install the local development environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Install the training and development dependencies:

```bash
uv sync --extra train --extra dev
```

Run either portable GPU profile:

```bash
scripts/run.sh 0_5b
scripts/run.sh 1_5b
scripts/run.sh 1_5b_unlearning
```

Individual stages remain available:

```bash
mlu acquire -c configs/experiments/1_5b.yaml
mlu oracle -c configs/experiments/1_5b.yaml
mlu unlearn npo -c configs/experiments/1_5b.yaml
mlu eval -c configs/experiments/1_5b.yaml --model npo
mlu report -c configs/experiments/1_5b.yaml
```

Long-running commands display `[mlu] START/DONE` stage messages, elapsed time,
named progress bars, concise metric summaries, and output locations. Progress
and completed stages are also available from another shell:

```bash
mlu status -c configs/experiments/1_5b.yaml
```

See [docs/experiment_protocol.md](docs/experiment_protocol.md) for the experiment
contract and [docs/metrics.md](docs/metrics.md) for when each metric is used.
For setup and commands that work across GPU platforms, see
[docs/running.md](docs/running.md).
