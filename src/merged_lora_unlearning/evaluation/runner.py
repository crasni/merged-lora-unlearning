from __future__ import annotations

from pathlib import Path
from typing import Any

from merged_lora_unlearning.artifacts import RunArtifacts, read_jsonl, write_json, write_jsonl
from merged_lora_unlearning.config import Config
from merged_lora_unlearning.data.schemas import Fact
from merged_lora_unlearning.evaluation.muse import (
    evaluate_knowledge,
    evaluate_privacy_text,
    evaluate_verbatim,
    privacy_auc,
)
from merged_lora_unlearning.models.loading import load_causal_lm, load_tokenizer


MODEL_ROLES = {
    "base": None,
    "target": "target",
    "oracle": "retain_oracle",
}


def _load_facts(path: Path) -> list[Fact]:
    return [Fact.from_dict(row) for row in read_jsonl(path)]


def evaluate_model(config: Config, role: str) -> dict[str, Any]:
    artifacts = RunArtifacts(config)
    model_path = MODEL_ROLES.get(role, role)
    if role == "acquisition_adapter":
        from peft import PeftModel

        source = config.model.name
        model = PeftModel.from_pretrained(
            load_causal_lm(source, config.model.dtype, config.model.device_map),
            artifacts.models_dir / "acquisition_adapter",
        )
        tokenizer = load_tokenizer(source)
    else:
        source = config.model.name if model_path is None else str(artifacts.models_dir / model_path)
        model = load_causal_lm(source, config.model.dtype, config.model.device_map)
        tokenizer = load_tokenizer(source)
    output_dir = artifacts.evaluations_dir / role
    output_dir.mkdir(parents=True, exist_ok=True)

    forget = _load_facts(artifacts.data_dir / "forget_test.jsonl")
    retain = _load_facts(artifacts.data_dir / "retain_test.jsonl")
    holdout = _load_facts(artifacts.data_dir / "holdout.jsonl")

    c1, c1_rows = evaluate_verbatim(
        model, tokenizer, forget, config.evaluation.max_new_tokens
    )
    c2, forget_rows = evaluate_knowledge(
        model, tokenizer, forget, "original", config.evaluation.max_new_tokens
    )
    c4, retain_rows = evaluate_knowledge(
        model, tokenizer, retain, "original", config.evaluation.max_new_tokens
    )
    holdout_metrics, holdout_rows = evaluate_knowledge(
        model, tokenizer, holdout, "original", config.evaluation.max_new_tokens
    )
    privacy_forget_rows = evaluate_privacy_text(model, tokenizer, forget)
    privacy_holdout_rows = evaluate_privacy_text(model, tokenizer, holdout)
    c3 = privacy_auc(privacy_forget_rows, privacy_holdout_rows)

    robustness = {}
    for prompt_type in config.evaluation.prompt_types:
        if prompt_type == "original":
            continue
        forget_metrics, prompt_rows = evaluate_knowledge(
            model, tokenizer, forget, prompt_type, config.evaluation.max_new_tokens
        )
        robustness[prompt_type] = forget_metrics
        robustness_path = output_dir / f"forget_{prompt_type}.jsonl"
        write_jsonl(robustness_path, prompt_rows)
        artifacts.record_artifact(robustness_path, role=f"{role}:robustness:{prompt_type}")

    metrics = {
        "model_role": role,
        "muse": {
            "c1_verbatim_forget": c1,
            "c2_knowledge_forget": c2,
            "c3_privacy": c3,
            "c4_knowledge_retain": c4,
        },
        "supplementary": {"holdout": holdout_metrics, "robustness": robustness},
    }
    result_files = {
        "metrics.json": ("evaluation_metrics", metrics, write_json),
        "verbatim_forget.jsonl": ("muse_c1_rows", c1_rows, write_jsonl),
        "knowledge_forget.jsonl": ("muse_c2_rows", forget_rows, write_jsonl),
        "knowledge_retain.jsonl": ("muse_c4_rows", retain_rows, write_jsonl),
        "knowledge_holdout.jsonl": ("holdout_rows", holdout_rows, write_jsonl),
        "privacy_forget.jsonl": ("muse_c3_forget_rows", privacy_forget_rows, write_jsonl),
        "privacy_holdout.jsonl": ("muse_c3_holdout_rows", privacy_holdout_rows, write_jsonl),
    }
    for filename, (artifact_role, value, writer) in result_files.items():
        path = output_dir / filename
        writer(path, value)
        artifacts.record_artifact(path, role=f"{role}:{artifact_role}")
    artifacts.set_stage(f"eval:{role}", "complete", {"metrics": str(output_dir / "metrics.json")})
    return metrics
