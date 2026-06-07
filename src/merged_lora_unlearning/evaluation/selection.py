from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from tqdm import tqdm

from merged_lora_unlearning.artifacts import RunArtifacts, read_jsonl, write_json
from merged_lora_unlearning.config import Config
from merged_lora_unlearning.data.schemas import Fact
from merged_lora_unlearning.evaluation.muse import evaluate_knowledge
from merged_lora_unlearning.models.loading import load_causal_lm, load_tokenizer
from merged_lora_unlearning.progress import info


class NoEligibleCheckpointError(RuntimeError):
    pass


def _checkpoint_step(path: Path) -> int:
    return int(path.name.rsplit("-", 1)[-1])


def _load_checkpoint(config: Config, checkpoint: Path):
    if config.unlearning.update_mode == "lora":
        from peft import PeftModel

        target = load_causal_lm(
            str(config.model_dir("target")),
            config.model.dtype,
            config.model.device_map,
        )
        return PeftModel.from_pretrained(target, checkpoint)
    return load_causal_lm(str(checkpoint), config.model.dtype, config.model.device_map)


def _choose_checkpoint(
    trajectory: list[dict[str, Any]], retain_match_floor: float, method: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eligible = [row for row in trajectory if row["retain_match"] >= retain_match_floor]
    if not eligible:
        raise NoEligibleCheckpointError(
            f"No {method} checkpoint meets retain_match_floor={retain_match_floor:.3f}"
        )
    selected = min(
        eligible,
        key=lambda row: (row["forget_match"], row["forget_rouge_l"], -row["retain_match"]),
    )
    return selected, eligible


def select_unlearning_checkpoint(config: Config, method: str) -> tuple[Path, list[dict[str, Any]]]:
    artifacts = RunArtifacts(config)
    output_dir = artifacts.models_dir / method
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=_checkpoint_step)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found under {output_dir}")
    tokenizer = load_tokenizer(str(config.model_dir("target")))
    forget = [
        Fact.from_dict(row)
        for row in read_jsonl(artifacts.data_dir / "forget_request.jsonl")
    ]
    retain = [
        Fact.from_dict(row)
        for row in read_jsonl(artifacts.data_dir / "retain_validation.jsonl")
    ]
    trajectory = []
    for checkpoint in tqdm(
        checkpoints,
        desc=f"{method} | checkpoint selection",
        unit="checkpoint",
        dynamic_ncols=True,
    ):
        model = _load_checkpoint(config, checkpoint)
        forget_metrics, _ = evaluate_knowledge(
            model,
            tokenizer,
            forget,
            "selection",
            config.evaluation.max_new_tokens,
            show_progress=False,
        )
        retain_metrics, _ = evaluate_knowledge(
            model,
            tokenizer,
            retain,
            "original",
            config.evaluation.max_new_tokens,
            show_progress=False,
        )
        trajectory.append(
            {
                "checkpoint": str(checkpoint),
                "step": _checkpoint_step(checkpoint),
                "forget_match": forget_metrics["normalized_match"],
                "forget_rouge_l": forget_metrics["rouge_l"],
                "retain_match": retain_metrics["normalized_match"],
                "retain_rouge_l": retain_metrics["rouge_l"],
            }
        )
        info(
            f"checkpoint={checkpoint.name} forget_match={forget_metrics['normalized_match']:.4f} "
            f"retain_match={retain_metrics['normalized_match']:.4f}"
        )
        del model
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    try:
        selected, eligible = _choose_checkpoint(
            trajectory, config.unlearning.retain_match_floor, method
        )
    except RuntimeError:
        selection = {
            "selected_checkpoint": None,
            "retain_match_floor": config.unlearning.retain_match_floor,
            "eligible_checkpoint_count": 0,
            "fallback_used": False,
            "trajectory": trajectory,
        }
        write_json(output_dir / "selection.json", selection)
        raise
    selection = {
        "selected_checkpoint": selected["checkpoint"],
        "retain_match_floor": config.unlearning.retain_match_floor,
        "eligible_checkpoint_count": len(eligible),
        "fallback_used": False,
        "trajectory": trajectory,
    }
    write_json(output_dir / "selection.json", selection)
    return Path(selected["checkpoint"]), trajectory
