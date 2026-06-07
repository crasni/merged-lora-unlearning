from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from merged_lora_unlearning.config import Config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class RunArtifacts:
    def __init__(self, config: Config):
        self.config = config
        self.root = config.run_dir
        self.data_dir = self.root / "data"
        self.models_dir = self.root / "models"
        self.evaluations_dir = self.root / "evaluations"
        self.report_dir = self.root / "report"
        self.status_path = self.root / "stage_status.json"
        self.manifest_path = self.root / "manifest.json"

    def initialize(self) -> None:
        resolved = self.root / "resolved_config.yaml"
        if resolved.exists():
            existing = yaml.safe_load(resolved.read_text(encoding="utf-8"))
            if existing != self.config.raw:
                raise RuntimeError(
                    f"Run directory already belongs to a different config: {self.root}"
                )
        for directory in (
            self.data_dir,
            self.models_dir,
            self.evaluations_dir,
            self.report_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        resolved.write_text(yaml.safe_dump(self.config.raw, sort_keys=False), encoding="utf-8")
        if not self.status_path.exists():
            write_json(self.status_path, {})
        self.record_artifact(resolved, role="resolved_config")

    def set_stage(self, stage: str, state: str, details: dict[str, Any] | None = None) -> None:
        status = read_json(self.status_path) if self.status_path.exists() else {}
        status[stage] = {
            "state": state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        }
        write_json(self.status_path, status)

    def record_artifact(self, path: Path, role: str) -> None:
        manifest = read_json(self.manifest_path) if self.manifest_path.exists() else {}
        manifest.setdefault("experiment", self.config.experiment.name)
        manifest.setdefault("seed", self.config.experiment.seed)
        manifest.setdefault("base_model", self.config.model.name)
        manifest.setdefault("config_source", str(self.config.source_path))
        manifest.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        manifest.setdefault("artifacts", {})
        manifest["artifacts"][str(path.relative_to(self.root))] = {
            "role": role,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        write_json(self.manifest_path, manifest)

    def copy_config(self) -> None:
        target = self.root / "source_config.yaml"
        shutil.copyfile(self.config.source_path, target)
        self.record_artifact(target, role="source_config")
