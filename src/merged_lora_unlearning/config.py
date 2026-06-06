from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int = 42
    output_root: str = "outputs/runs"
    reuse_from: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    name: str
    dtype: str = "auto"
    device_map: str | None = "auto"


@dataclass(frozen=True)
class DataConfig:
    generated_facts: int = 3000
    acquisition_size: int = 1000
    holdout_size: int = 300
    forget_ratio: float = 0.2
    validation_ratio: float = 0.1
    categories: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AcquisitionConfig:
    epochs: int = 5
    learning_rate: float = 2e-4
    batch_size: int = 8
    max_length: int = 256
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass(frozen=True)
class UnlearningConfig:
    methods: list[str] = field(default_factory=lambda: ["ga", "grad_diff", "npo", "npo_grad_diff"])
    update_mode: str = "lora"
    epochs: int = 10
    learning_rate: float = 1e-5
    batch_size: int = 8
    max_length: int = 256
    beta: float = 0.1
    simnpo_delta: float = 0.0
    gamma: float = 1.0
    alpha: float = 1.0
    retain_match_floor: float = 0.7
    checkpoint_every_epochs: int = 1
    method_overrides: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 1 <= self.checkpoint_every_epochs <= self.epochs:
            raise ValueError("checkpoint_every_epochs must be between 1 and epochs")

    def settings_for(self, method: str) -> dict[str, float | int]:
        settings = {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "beta": self.beta,
            "simnpo_delta": self.simnpo_delta,
            "gamma": self.gamma,
            "alpha": self.alpha,
        }
        overrides = self.method_overrides.get(method, {})
        unknown = set(overrides) - set(settings)
        if unknown:
            raise ValueError(
                f"Unknown unlearning overrides for {method}: {', '.join(sorted(unknown))}"
            )
        settings.update(overrides)
        return settings


@dataclass(frozen=True)
class EvaluationConfig:
    max_new_tokens: int = 32
    filter_batch_size: int = 16
    prompt_types: list[str] = field(
        default_factory=lambda: ["original", "paraphrase", "zh", "mixed"]
    )
    base_knowledge_threshold: float = 0.05
    acquisition_threshold: float = 0.8
    merge_max_delta: float = 0.02


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig
    model: ModelConfig
    data: DataConfig
    acquisition: AcquisitionConfig
    unlearning: UnlearningConfig
    evaluation: EvaluationConfig
    source_path: Path
    raw: dict[str, Any]

    @property
    def run_dir(self) -> Path:
        return Path(self.experiment.output_root) / self.experiment.name

    @property
    def reuse_dir(self) -> Path | None:
        if self.experiment.reuse_from is None:
            return None
        return Path(self.experiment.output_root) / self.experiment.reuse_from

    def model_dir(self, role: str) -> Path:
        if self.reuse_dir is not None and role in {
            "acquisition_adapter",
            "target",
            "oracle_adapter",
            "retain_oracle",
        }:
            return self.reuse_dir / "models" / role
        return self.run_dir / "models" / role


def load_config(path: str | Path) -> Config:
    source_path = Path(path)
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    return Config(
        experiment=ExperimentConfig(**raw["experiment"]),
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        acquisition=AcquisitionConfig(**raw["acquisition"]),
        unlearning=UnlearningConfig(**raw["unlearning"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        source_path=source_path,
        raw=raw,
    )
