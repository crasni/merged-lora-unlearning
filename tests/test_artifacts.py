from pathlib import Path

import pytest

from merged_lora_unlearning.artifacts import RunArtifacts
from merged_lora_unlearning.config import load_config


def test_existing_run_rejects_changed_config(tmp_path: Path):
    config = load_config("configs/experiments/0_5b.yaml")
    config.raw["experiment"]["output_root"] = str(tmp_path)
    config.raw["experiment"]["name"] = "immutable"
    config = load_config_from_raw(config, tmp_path / "first.yaml")
    RunArtifacts(config).initialize()

    changed = dict(config.raw)
    changed["data"] = dict(changed["data"], acquisition_size=201)
    changed_config = load_config_from_raw(config, tmp_path / "changed.yaml", changed)

    with pytest.raises(RuntimeError, match="different config"):
        RunArtifacts(changed_config).initialize()


def load_config_from_raw(config, path: Path, raw=None):
    import yaml

    path.write_text(yaml.safe_dump(raw or config.raw, sort_keys=False), encoding="utf-8")
    return load_config(path)
