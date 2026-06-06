# Kaggle Walkthrough

The Kaggle configuration is a conservative end-to-end experiment using
`Qwen/Qwen2.5-0.5B-Instruct`, 250 acquired facts, and GA plus NPO. It is intended
to validate the complete pipeline within a Kaggle GPU session. The full
workstation experiment remains `configs/experiments/qwen_mvp.yaml`.

## Notebook Setup

Enable a GPU and internet access in the Kaggle notebook settings.

```python
%cd /kaggle/working
!git clone https://github.com/crasni/merged-lora-unlearning.git
%cd /kaggle/working/merged-lora-unlearning
```

```bash
!pip install uv
!uv sync --extra train --extra dev
!uv pip install --python .venv/bin/python wrapt
```

Verify the environment:

```bash
!uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
!uv run pytest
```

## Recommended Incremental Run

Run stages individually the first time:

```bash
!uv run mlu data -c configs/experiments/kaggle.yaml
!uv run mlu filter -c configs/experiments/kaggle.yaml
!uv run mlu eval --model base -c configs/experiments/kaggle.yaml

!uv run mlu acquire -c configs/experiments/kaggle.yaml
!uv run mlu eval --model acquisition_adapter -c configs/experiments/kaggle.yaml
!uv run mlu eval --model target -c configs/experiments/kaggle.yaml

!uv run mlu oracle -c configs/experiments/kaggle.yaml
!uv run mlu eval --model oracle -c configs/experiments/kaggle.yaml

!uv run mlu unlearn ga -c configs/experiments/kaggle.yaml
!uv run mlu eval --model ga -c configs/experiments/kaggle.yaml

!uv run mlu unlearn npo -c configs/experiments/kaggle.yaml
!uv run mlu eval --model npo -c configs/experiments/kaggle.yaml

!uv run mlu report -c configs/experiments/kaggle.yaml
```

Inspect progress and results:

```bash
!uv run mlu status -c configs/experiments/kaggle.yaml
!cat /kaggle/working/outputs/runs/kaggle/report/report.md
```

## One-Command Run

After the incremental flow succeeds:

```bash
!scripts/run_kaggle.sh
```

This runs the complete experiment and creates:

```text
/kaggle/working/kaggle-results.tar.gz
```

Download the archive from the Kaggle Files panel before the session ends.

## Optional SimNPO

SimNPO is omitted from the default Kaggle config to conserve session quota. Run
it separately if time remains:

```bash
!uv run mlu unlearn simnpo -c configs/experiments/kaggle.yaml
!uv run mlu eval --model simnpo -c configs/experiments/kaggle.yaml
!uv run mlu report -c configs/experiments/kaggle.yaml
```

