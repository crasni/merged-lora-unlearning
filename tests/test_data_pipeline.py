from dataclasses import replace
from pathlib import Path

from merged_lora_unlearning.artifacts import read_json, read_jsonl
from merged_lora_unlearning.config import load_config
from merged_lora_unlearning.data.pipeline import run_data_pipeline


def test_data_pipeline_writes_manifest(tmp_path: Path):
    config = load_config("configs/experiments/tiny_local.yaml")
    config = replace(
        config,
        experiment=replace(config.experiment, output_root=str(tmp_path)),
    )

    counts = run_data_pipeline(config)
    run_dir = tmp_path / "tiny_local"

    assert counts["holdout"] == 10
    assert len(read_jsonl(run_dir / "data/acquisition_all.jsonl")) == 20
    assert read_json(run_dir / "stage_status.json")["data"]["state"] == "complete"
    assert "data/forget_test.jsonl" in read_json(run_dir / "manifest.json")["artifacts"]

