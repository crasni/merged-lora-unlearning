from dataclasses import replace
from pathlib import Path

from merged_lora_unlearning.artifacts import read_json, read_jsonl
from merged_lora_unlearning.config import load_config
from merged_lora_unlearning.data.pipeline import run_data_pipeline


def test_data_pipeline_writes_manifest(tmp_path: Path):
    config = load_config("configs/experiments/0_5b.yaml")
    config = replace(
        config,
        experiment=replace(config.experiment, name="test_data", output_root=str(tmp_path)),
        data=replace(
            config.data,
            generated_facts=40,
            acquisition_size=20,
            holdout_size=10,
            validation_ratio=0.2,
        ),
    )

    counts = run_data_pipeline(config)
    run_dir = tmp_path / "test_data"

    assert counts["holdout"] == 10
    assert len(read_jsonl(run_dir / "data/acquisition_all.jsonl")) == 20
    assert len(read_jsonl(run_dir / "data/forget_request.jsonl")) == 4
    assert read_json(run_dir / "stage_status.json")["data"]["state"] == "complete"
    assert "data/forget_test.jsonl" in read_json(run_dir / "manifest.json")["artifacts"]
