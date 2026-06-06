from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from merged_lora_unlearning.artifacts import write_json
from merged_lora_unlearning.config import load_config
from merged_lora_unlearning.reuse import (
    REUSED_EVALUATION_ROLES,
    REQUIRED_DATA_FILES,
    prepare_reused_baseline,
)


REQUIRED_STAGES = (
    "data",
    "base_filter",
    "eval:base",
    "acquisition",
    "acquisition_merge",
    "eval:acquisition_adapter",
    "eval:target",
    "oracle",
    "oracle_merge",
    "eval:oracle",
)


def _make_source(tmp_path: Path):
    config = load_config("configs/experiments/1_5b_unlearning.yaml")
    config = replace(
        config,
        experiment=replace(config.experiment, output_root=str(tmp_path)),
    )
    source = config.reuse_dir
    assert source is not None
    source.mkdir(parents=True)
    source_raw = dict(config.raw)
    source_raw["experiment"] = dict(source_raw["experiment"], name="1_5b")
    source_raw["experiment"].pop("reuse_from")
    (source / "resolved_config.yaml").write_text(
        yaml.safe_dump(source_raw, sort_keys=False), encoding="utf-8"
    )
    write_json(
        source / "stage_status.json",
        {name: {"state": "complete"} for name in REQUIRED_STAGES},
    )
    (source / "data").mkdir()
    for filename in REQUIRED_DATA_FILES:
        (source / "data" / filename).write_text("{}\n", encoding="utf-8")
    target = source / "models" / "target"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    (target / "model.safetensors").write_bytes(b"weights")
    for role in REUSED_EVALUATION_ROLES:
        evaluation = source / "evaluations" / role
        evaluation.mkdir(parents=True)
        (evaluation / "metrics.json").write_text("{}", encoding="utf-8")
    return config, source


def test_prepare_reused_baseline_copies_small_artifacts_and_reuses_target(tmp_path: Path):
    config, source = _make_source(tmp_path)

    prepare_reused_baseline(config)

    assert not (config.run_dir / "data").is_symlink()
    assert not (config.run_dir / "evaluations/oracle").is_symlink()
    assert not (config.run_dir / "models/target").exists()
    assert config.model_dir("target") == source / "models/target"

    prepare_reused_baseline(config)


def test_prepare_reused_baseline_rejects_incompatible_source(tmp_path: Path):
    config, source = _make_source(tmp_path)
    source_raw = yaml.safe_load((source / "resolved_config.yaml").read_text(encoding="utf-8"))
    source_raw["acquisition"]["epochs"] += 1
    (source / "resolved_config.yaml").write_text(yaml.safe_dump(source_raw), encoding="utf-8")

    with pytest.raises(RuntimeError, match="acquisition"):
        prepare_reused_baseline(config)


def test_prepare_reused_baseline_rejects_changed_target(tmp_path: Path):
    config, source = _make_source(tmp_path)
    prepare_reused_baseline(config)
    (source / "models/target/model.safetensors").write_bytes(b"changed weights")

    with pytest.raises(RuntimeError, match="changed after preparation"):
        prepare_reused_baseline(config)


def test_prepare_reused_baseline_rejects_changed_followup_settings(tmp_path: Path):
    config, _ = _make_source(tmp_path)
    prepare_reused_baseline(config)
    config.raw["unlearning"]["epochs"] += 1

    with pytest.raises(RuntimeError, match="changed after preparation"):
        prepare_reused_baseline(config)
