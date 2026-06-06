from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from merged_lora_unlearning.artifacts import RunArtifacts, read_json, sha256_file, write_json
from merged_lora_unlearning.config import Config
from merged_lora_unlearning.progress import info, stage


REUSED_EVALUATION_ROLES = ("base", "acquisition_adapter", "target", "oracle")
COMPATIBLE_CONFIG_SECTIONS = ("model", "data", "acquisition", "evaluation")
REQUIRED_DATA_FILES = (
    "forget_train.jsonl",
    "forget_validation.jsonl",
    "forget_test.jsonl",
    "retain_regularize.jsonl",
    "retain_validation.jsonl",
    "retain_test.jsonl",
    "holdout.jsonl",
)


def _model_output_is_complete(path: Path) -> bool:
    return (path / "config.json").exists() and any(
        (path / filename).exists()
        for filename in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )


def _model_fingerprint(path: Path) -> list[dict[str, int | str]]:
    return [
        {"name": file.name, "bytes": file.stat().st_size, "mtime_ns": file.stat().st_mtime_ns}
        for file in sorted(path.iterdir())
        if file.is_file()
    ]


def _required_sources(source: Path) -> list[Path]:
    return [
        *(source / "data" / filename for filename in REQUIRED_DATA_FILES),
        source / "models" / "target" / "config.json",
        *(source / "evaluations" / role / "metrics.json" for role in REUSED_EVALUATION_ROLES),
    ]


def _compatibility_errors(config: Config, source_raw: dict[str, Any]) -> list[str]:
    errors = []
    source_experiment = source_raw["experiment"]
    if source_experiment.get("seed", 42) != config.experiment.seed:
        errors.append("experiment.seed")
    for section in COMPATIBLE_CONFIG_SECTIONS:
        if source_raw.get(section) != config.raw.get(section):
            errors.append(section)
    return errors


def prepare_reused_baseline(config: Config) -> None:
    source = config.reuse_dir
    if source is None:
        raise ValueError("experiment.reuse_from is required")
    if source.resolve() == config.run_dir.resolve():
        raise RuntimeError("experiment.reuse_from must reference a different run")
    reuse_record = config.run_dir / "reuse.json"
    if config.run_dir.exists() and not reuse_record.exists():
        raise RuntimeError(
            f"Reuse destination exists without a completed reuse record: {config.run_dir}"
        )

    with stage("reuse-baseline", f"source={source}"):
        resolved_config = source / "resolved_config.yaml"
        status_path = source / "stage_status.json"
        if not resolved_config.exists() or not status_path.exists():
            raise RuntimeError(f"Reuse source is not a completed experiment run: {source}")

        source_raw = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
        incompatible = _compatibility_errors(config, source_raw)
        if incompatible:
            raise RuntimeError(
                "Reuse source has incompatible pre-unlearning settings: "
                + ", ".join(incompatible)
            )

        status = read_json(status_path)
        required_stages = (
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
        incomplete = [
            name for name in required_stages if status.get(name, {}).get("state") != "complete"
        ]
        if incomplete:
            raise RuntimeError("Reuse source has incomplete stages: " + ", ".join(incomplete))

        missing = [path for path in _required_sources(source) if not path.exists()]
        if not _model_output_is_complete(source / "models" / "target"):
            missing.append(source / "models" / "target" / "<model weights>")
        if missing:
            raise RuntimeError("Reuse source is missing artifacts: " + ", ".join(map(str, missing)))

        source_identity = {
            "source": str(source),
            "source_resolved_config_sha256": sha256_file(resolved_config),
            "source_manifest_sha256": (
                sha256_file(source / "manifest.json") if (source / "manifest.json").exists() else None
            ),
            "target_model_fingerprint": _model_fingerprint(source / "models" / "target"),
            "followup_unlearning": config.raw["unlearning"],
        }
        if reuse_record.exists():
            record = read_json(reuse_record)
            if any(record.get(key) != value for key, value in source_identity.items()):
                raise RuntimeError(f"Reuse source changed after preparation: {source}")
            info(f"Reuse baseline already prepared and validated: {source}")
            return

        try:
            artifacts = RunArtifacts(config)
            artifacts.initialize()
            artifacts.data_dir.rmdir()
            shutil.copytree(source / "data", artifacts.data_dir)
            for role in REUSED_EVALUATION_ROLES:
                shutil.copytree(source / "evaluations" / role, artifacts.evaluations_dir / role)

            artifacts.set_stage("reuse_baseline", "complete", {"source": str(source)})
            write_json(
                reuse_record,
                {
                    **source_identity,
                    "model_source": str(source / "models" / "target"),
                    "copied_evaluations": list(REUSED_EVALUATION_ROLES),
                },
            )
            artifacts.record_artifact(reuse_record, role="reuse_provenance")
            for copied_root in (artifacts.data_dir, artifacts.evaluations_dir):
                for copied_file in sorted(path for path in copied_root.rglob("*") if path.is_file()):
                    artifacts.record_artifact(copied_file, role="reused_baseline_artifact")
        except Exception:
            shutil.rmtree(config.run_dir, ignore_errors=True)
            raise
        info(f"Reused validated baseline artifacts from: {source}")
