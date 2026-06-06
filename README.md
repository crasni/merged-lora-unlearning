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

Generate and split a small CPU-only dataset:

```bash
mlu data -c configs/experiments/tiny_local.yaml
mlu status -c configs/experiments/tiny_local.yaml
```

Run the complete GPU experiment:

```bash
mlu run -c configs/experiments/qwen_mvp.yaml
```

Individual stages remain available:

```bash
mlu acquire -c configs/experiments/qwen_mvp.yaml
mlu oracle -c configs/experiments/qwen_mvp.yaml
mlu unlearn npo -c configs/experiments/qwen_mvp.yaml
mlu eval -c configs/experiments/qwen_mvp.yaml --model npo
mlu report -c configs/experiments/qwen_mvp.yaml
```

Long-running commands display `[mlu] START/DONE` stage messages, elapsed time,
named progress bars, concise metric summaries, and output locations. Progress
and completed stages are also available from another shell:

```bash
mlu status -c configs/experiments/qwen_mvp.yaml
```

See [docs/experiment_protocol.md](docs/experiment_protocol.md) for the experiment
contract and [docs/metrics.md](docs/metrics.md) for when each metric is used.
