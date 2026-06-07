from pathlib import Path

import pytest

from merged_lora_unlearning.artifacts import RunArtifacts, write_jsonl
from merged_lora_unlearning.config import load_config
from merged_lora_unlearning.data.filtering import _filter_base_knowledge
from merged_lora_unlearning.data.generation import generate_facts


def test_filter_resume_rejects_predictions_for_different_prompts(tmp_path: Path):
    raw = load_config("configs/experiments/0_5b.yaml").raw
    raw["experiment"]["name"] = "filter_test"
    raw["experiment"]["output_root"] = str(tmp_path)
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_config(config_path)
    artifacts = RunArtifacts(config)
    artifacts.initialize()
    fact = generate_facts(1, 42, ["capital"])[0]
    write_jsonl(artifacts.data_dir / "candidates.jsonl", [fact.to_dict()])
    write_jsonl(
        artifacts.data_dir / "base_filter_predictions.jsonl",
        [{"fact_id": fact.fact_id, "prompt": "different prompt", "prediction": "", "known": False}],
    )

    with pytest.raises(RuntimeError, match="does not match"):
        _filter_base_knowledge(config)
