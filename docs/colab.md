# Google Colab 1.5B Walkthrough

This path runs the complete experiment using `Qwen/Qwen2.5-1.5B-Instruct` on a
single Colab GPU. It uses a distinct run name, `colab_1_5b`, so existing
`colab`, `kaggle`, or workstation results are not overwritten.

## Terminal Setup

```bash
cd /content
git clone https://github.com/crasni/merged-lora-unlearning.git
cd merged-lora-unlearning

pip install uv
uv sync --extra train --extra dev
uv run pytest
```

Verify the GPU:

```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Full Run

```bash
scripts/run_colab_1_5b.sh
```

The runner verifies CUDA, executes the full pipeline, records console output,
and archives current results even when a quality gate stops the run. It creates:

```text
/content/colab_1_5b.log
/content/colab_1_5b-results.tar.gz
```

Download both files before the Colab runtime ends.

## Incremental Commands

If the full run stops at a scientific quality gate, continue incrementally:

```bash
uv run mlu status -c configs/experiments/colab_1_5b.yaml
uv run mlu acquire -c configs/experiments/colab_1_5b.yaml
uv run mlu eval --model acquisition_adapter -c configs/experiments/colab_1_5b.yaml
uv run mlu eval --model target -c configs/experiments/colab_1_5b.yaml
```

The experiment intentionally stops if acquisition quality is below its
configured threshold. Do not bypass that gate before inspecting the results.

## Existing Results

Previous results do not need to be deleted. The new run writes only to:

```text
/content/outputs/runs/colab_1_5b
```

Delete that specific directory only when intentionally restarting this exact
configuration from scratch:

```bash
rm -rf /content/outputs/runs/colab_1_5b
```
