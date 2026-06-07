from __future__ import annotations

from merged_lora_unlearning.config import Config
from merged_lora_unlearning.data.filtering import filter_base_knowledge
from merged_lora_unlearning.data.pipeline import run_data_pipeline
from merged_lora_unlearning.evaluation.runner import evaluate_model
from merged_lora_unlearning.evaluation.selection import NoEligibleCheckpointError
from merged_lora_unlearning.reporting.report import generate_report
from merged_lora_unlearning.reuse import prepare_reused_baseline
from merged_lora_unlearning.training.acquisition import train_lora
from merged_lora_unlearning.training.merging import merge_lora
from merged_lora_unlearning.unlearning.trainer import unlearn
from merged_lora_unlearning.progress import info, stage


def _knowledge_match(metrics: dict, criterion: str) -> float:
    return float(metrics["muse"][criterion]["normalized_match"])


def run_full_experiment(config: Config) -> None:
    with stage("full-experiment", f"name={config.experiment.name} model={config.model.name}"):
        _run_full_experiment(config)
    info(f"Experiment complete: {config.run_dir}")


def _run_full_experiment(config: Config) -> None:
    if config.experiment.reuse_from:
        prepare_reused_baseline(config)
        if config.experiment.forget_request_ratio is not None:
            evaluate_model(config, "target")
            train_lora(config, "oracle")
            merge_lora(config, "oracle")
            evaluate_model(config, "oracle")
        for method in config.unlearning.methods:
            try:
                unlearn(config, method)
                evaluate_model(config, method)
            except NoEligibleCheckpointError as exc:
                info(f"Skipping final evaluation for failed method {method}: {exc}")
        generate_report(config)
        return

    run_data_pipeline(config)
    filter_base_knowledge(config)
    evaluate_model(config, "base")

    train_lora(config, "acquisition")
    adapter_metrics = evaluate_model(config, "acquisition_adapter")
    acquisition_score = min(
        _knowledge_match(adapter_metrics, "c2_knowledge_forget"),
        _knowledge_match(adapter_metrics, "c4_knowledge_retain"),
    )
    if acquisition_score < config.evaluation.acquisition_threshold:
        raise RuntimeError(
            f"Acquisition score {acquisition_score:.3f} is below threshold "
            f"{config.evaluation.acquisition_threshold:.3f}"
        )
    merge_lora(config, "acquisition")
    target_metrics = evaluate_model(config, "target")
    merge_delta = max(
        abs(
            _knowledge_match(adapter_metrics, criterion)
            - _knowledge_match(target_metrics, criterion)
        )
        for criterion in ("c2_knowledge_forget", "c4_knowledge_retain")
    )
    if merge_delta > config.evaluation.merge_max_delta:
        raise RuntimeError(
            f"Merge behavior delta {merge_delta:.3f} exceeds threshold "
            f"{config.evaluation.merge_max_delta:.3f}"
        )

    train_lora(config, "oracle")
    merge_lora(config, "oracle")
    evaluate_model(config, "oracle")

    for method in config.unlearning.methods:
        try:
            unlearn(config, method)
            evaluate_model(config, method)
        except NoEligibleCheckpointError as exc:
            info(f"Skipping final evaluation for failed method {method}: {exc}")
    generate_report(config)
