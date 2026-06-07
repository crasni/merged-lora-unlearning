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
from merged_lora_unlearning.evaluation.utility import evaluate_general_utility
from merged_lora_unlearning.models.loading import load_causal_lm, load_tokenizer
from merged_lora_unlearning.progress import info, metric_summary, stage


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
    with stage(f"evaluate:{role}", f"max_new_tokens={config.evaluation.max_new_tokens}"):
        info(f"Loading model role={role}")
        if role == "acquisition_adapter":
            from peft import PeftModel

            source = config.model.name
            model = PeftModel.from_pretrained(
                load_causal_lm(source, config.model.dtype, config.model.device_map),
                config.model_dir("acquisition_adapter"),
            )
            tokenizer = load_tokenizer(source)
        else:
            source = config.model.name if model_path is None else str(config.model_dir(model_path))
            model = load_causal_lm(source, config.model.dtype, config.model.device_map)
            tokenizer = load_tokenizer(source)
        return _evaluate_loaded_model(config, role, model, tokenizer, artifacts)


def _evaluate_loaded_model(config: Config, role: str, model, tokenizer, artifacts: RunArtifacts) -> dict[str, Any]:
    output_dir = artifacts.evaluations_dir / role
    output_dir.mkdir(parents=True, exist_ok=True)

    forget = _load_facts(artifacts.data_dir / "forget_test.jsonl")
    retain = _load_facts(artifacts.data_dir / "retain_test.jsonl")
    holdout = _load_facts(artifacts.data_dir / "holdout.jsonl")
    info(f"Evaluation sets | forget={len(forget)} retain={len(retain)} holdout={len(holdout)}")

    c1, c1_rows = evaluate_verbatim(
        model, tokenizer, forget, config.evaluation.max_new_tokens, f"{role} | C1 verbatim forget"
    )
    c2, forget_rows = evaluate_knowledge(
        model, tokenizer, forget, "original", config.evaluation.max_new_tokens, f"{role} | C2 knowledge forget"
    )
    c4, retain_rows = evaluate_knowledge(
        model, tokenizer, retain, "original", config.evaluation.max_new_tokens, f"{role} | C4 knowledge retain"
    )
    holdout_metrics, holdout_rows = evaluate_knowledge(
        model, tokenizer, holdout, "original", config.evaluation.max_new_tokens, f"{role} | holdout"
    )
    privacy_forget_rows = evaluate_privacy_text(model, tokenizer, forget, f"{role} | C3 privacy forget")
    privacy_holdout_rows = evaluate_privacy_text(model, tokenizer, holdout, f"{role} | C3 privacy holdout")
    c3 = privacy_auc(privacy_forget_rows, privacy_holdout_rows)

    robustness = {}
    for prompt_type in config.evaluation.prompt_types:
        if prompt_type == "original":
            continue
        forget_metrics, forget_prompt_rows = evaluate_knowledge(
            model,
            tokenizer,
            forget,
            prompt_type,
            config.evaluation.max_new_tokens,
            f"{role} | robustness {prompt_type}",
        )
        retain_metrics, retain_prompt_rows = evaluate_knowledge(
            model,
            tokenizer,
            retain,
            prompt_type,
            config.evaluation.max_new_tokens,
            f"{role} | retain robustness {prompt_type}",
        )
        robustness[prompt_type] = {"forget": forget_metrics, "retain": retain_metrics}
        for split, rows in (("forget", forget_prompt_rows), ("retain", retain_prompt_rows)):
            robustness_path = output_dir / f"{split}_{prompt_type}.jsonl"
            write_jsonl(robustness_path, rows)
            artifacts.record_artifact(
                robustness_path, role=f"{role}:robustness:{split}:{prompt_type}"
            )

    utility_metrics = None
    utility_rows = None
    if config.evaluation.general_utility:
        utility_metrics, utility_rows = evaluate_general_utility(
            model,
            tokenizer,
            config.evaluation.max_new_tokens,
            f"{role} | general utility",
        )

    metrics = {
        "model_role": role,
        "muse": {
            "c1_verbatim_forget": c1,
            "c2_knowledge_forget": c2,
            "c3_privacy": c3,
            "c4_knowledge_retain": c4,
        },
        "supplementary": {
            "holdout": holdout_metrics,
            "robustness": robustness,
            "general_utility": utility_metrics,
        },
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
    if utility_rows is not None:
        result_files["general_utility.jsonl"] = (
            "general_utility_rows",
            utility_rows,
            write_jsonl,
        )
    for filename, (artifact_role, value, writer) in result_files.items():
        path = output_dir / filename
        writer(path, value)
        artifacts.record_artifact(path, role=f"{role}:{artifact_role}")
    artifacts.set_stage(f"eval:{role}", "complete", {"metrics": str(output_dir / "metrics.json")})
    metric_summary("C1 verbatim forget", c1, ("mean_rouge_l", "answer_match_rate"))
    metric_summary("C2 knowledge forget", c2, ("normalized_match", "rouge_l"))
    metric_summary("C4 knowledge retain", c4, ("normalized_match", "rouge_l"))
    if utility_metrics is not None:
        metric_summary("General utility", utility_metrics, ("normalized_match",))
    info(f"C3 privacy | min_k_40_auc={c3['min_k_40_auc']:.4f} loss_auc={c3['loss_auc']:.4f}")
    info(f"Saved evaluation: {output_dir}")
    return metrics
