# Local 0.5B Run

This workflow runs the complete experiment using
`Qwen/Qwen2.5-0.5B-Instruct`. It is smaller than the workstation MVP but uses
stronger acquisition settings than the initial Kaggle experiment.

It requires a CUDA-capable GPU. CPU-only full training is not practical.

## Setup

```bash
git pull
uv sync --extra train --extra dev
uv run pytest
nvidia-smi
```

## Full Run

```bash
scripts/run_local_0_5b.sh
```

The runner verifies CUDA, runs the complete experiment, and archives partial or
completed results on exit.

Outputs:

```text
outputs/runs/local_0_5b/
outputs/local/local_0_5b.log
outputs/local/local_0_5b-results.tar.gz
```

## Existing Results

Other experiment outputs are not affected. To intentionally restart only this
run from scratch:

```bash
rm -rf outputs/runs/local_0_5b
```

## Incremental Recovery

If a quality gate stops the full run:

```bash
uv run mlu status -c configs/experiments/local_0_5b.yaml
uv run mlu acquire -c configs/experiments/local_0_5b.yaml
uv run mlu eval --model acquisition_adapter -c configs/experiments/local_0_5b.yaml
uv run mlu eval --model target -c configs/experiments/local_0_5b.yaml
```

