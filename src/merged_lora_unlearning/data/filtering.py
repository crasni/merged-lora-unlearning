from __future__ import annotations

from typing import Any

from tqdm import tqdm

from merged_lora_unlearning.artifacts import (
    RunArtifacts,
    append_jsonl,
    read_jsonl,
    write_json,
    write_jsonl,
)
from merged_lora_unlearning.config import Config
from merged_lora_unlearning.data.schemas import Fact
from merged_lora_unlearning.data.splitting import acquisition_facts, make_splits, oracle_facts
from merged_lora_unlearning.evaluation.model_metrics import generate_batch
from merged_lora_unlearning.evaluation.text import normalized_match
from merged_lora_unlearning.models.loading import load_causal_lm, load_tokenizer
from merged_lora_unlearning.progress import info, stage


def _prediction_row(fact: Fact, prompt: str, prediction: str) -> dict[str, Any]:
    return {
        "fact_id": fact.fact_id,
        "prompt": prompt,
        "prediction": prediction,
        "known": normalized_match(prediction, fact.answer_aliases),
    }


def filter_base_knowledge(config: Config) -> dict[str, int | float]:
    with stage(
        "base-filter",
        f"model={config.model.name} batch={config.evaluation.filter_batch_size}",
    ):
        return _filter_base_knowledge(config)


def _filter_base_knowledge(config: Config) -> dict[str, int | float]:
    artifacts = RunArtifacts(config)
    predictions_path = artifacts.data_dir / "base_filter_predictions.jsonl"
    candidates = [
        Fact.from_dict(row) for row in read_jsonl(artifacts.data_dir / "candidates.jsonl")
    ]
    existing_predictions = read_jsonl(predictions_path) if predictions_path.exists() else []
    expected_prompts = {
        fact.fact_id: f"Question: {fact.eval_qa_prompt}\nAnswer:" for fact in candidates
    }
    stale = [
        row["fact_id"]
        for row in existing_predictions
        if row["fact_id"] not in expected_prompts
        or row.get("prompt") != expected_prompts[row["fact_id"]]
    ]
    if stale:
        raise RuntimeError(
            "Base-filter resume data does not match the current generated facts; "
            f"start a new run name. Mismatched fact IDs: {', '.join(stale[:5])}"
        )
    predictions_by_id = {row["fact_id"]: row for row in existing_predictions}
    pending = [fact for fact in candidates if fact.fact_id not in predictions_by_id]
    artifacts.set_stage(
        "base_filter",
        "running",
        {
            "candidate_count": len(candidates),
            "completed_count": len(predictions_by_id),
            "pending_count": len(pending),
        },
    )
    info(
        f"Base filter: {len(candidates)} candidates, "
        f"{len(predictions_by_id)} completed, {len(pending)} pending."
    )
    if pending:
        info(f"Loading base model: {config.model.name}")
        model = load_causal_lm(config.model.name, config.model.dtype, config.model.device_map)
        tokenizer = load_tokenizer(config.model.name)
        model.eval()
        batch_size = config.evaluation.filter_batch_size
        progress = tqdm(
            range(0, len(pending), batch_size),
            desc="Filtering base-known facts",
            unit="batch",
            dynamic_ncols=True,
        )
        for start in progress:
            facts = pending[start : start + batch_size]
            prompts = [f"Question: {fact.eval_qa_prompt}\nAnswer:" for fact in facts]
            generated = generate_batch(model, tokenizer, prompts, config.evaluation.max_new_tokens)
            rows = [
                _prediction_row(fact, prompt, prediction)
                for fact, prompt, prediction in zip(facts, prompts, generated, strict=True)
            ]
            append_jsonl(predictions_path, rows)
            predictions_by_id.update({row["fact_id"]: row for row in rows})
            known_count = sum(bool(row["known"]) for row in predictions_by_id.values())
            known_rate = known_count / len(predictions_by_id)
            progress.set_postfix(
                completed=len(predictions_by_id),
                known=known_count,
                known_rate=f"{known_rate:.2%}",
            )
            artifacts.set_stage(
                "base_filter",
                "running",
                {
                    "candidate_count": len(candidates),
                    "completed_count": len(predictions_by_id),
                    "pending_count": len(candidates) - len(predictions_by_id),
                    "known_count": known_count,
                    "known_rate": known_rate,
                },
            )

    predictions = [predictions_by_id[fact.fact_id] for fact in candidates]
    kept = [fact for fact in candidates if not predictions_by_id[fact.fact_id]["known"]]
    accuracy = 1.0 - len(kept) / len(candidates)
    if accuracy > config.evaluation.base_knowledge_threshold:
        raise RuntimeError(
            f"Base knowledge rate {accuracy:.3f} exceeds threshold "
            f"{config.evaluation.base_knowledge_threshold:.3f}"
        )
    splits = make_splits(
        kept,
        config.data.acquisition_size,
        config.data.holdout_size,
        config.data.forget_ratio,
        config.data.validation_ratio,
        config.experiment.seed,
    )
    filtered_path = artifacts.data_dir / "facts_filtered.jsonl"
    write_jsonl(filtered_path, (fact.to_dict() for fact in kept))
    # Rewrite in candidate order after resumable append-mode processing.
    write_jsonl(predictions_path, predictions)
    artifacts.record_artifact(filtered_path, role="base_filtered_facts")
    artifacts.record_artifact(predictions_path, role="base_filter_predictions")
    for name, facts in splits.items():
        path = artifacts.data_dir / f"{name}.jsonl"
        write_jsonl(path, (fact.to_dict() for fact in facts))
        artifacts.record_artifact(path, role=f"filtered_split:{name}")
    acquisition_path = artifacts.data_dir / "acquisition_all.jsonl"
    oracle_path = artifacts.data_dir / "oracle_all.jsonl"
    write_jsonl(acquisition_path, (fact.to_dict() for fact in acquisition_facts(splits)))
    write_jsonl(oracle_path, (fact.to_dict() for fact in oracle_facts(splits)))
    artifacts.record_artifact(acquisition_path, role="filtered_acquisition_training")
    artifacts.record_artifact(oracle_path, role="filtered_oracle_training")
    summary = {"candidate_count": len(candidates), "filtered_count": len(kept), "base_knowledge_rate": accuracy}
    summary_path = artifacts.data_dir / "summary.json"
    write_json(summary_path, summary)
    artifacts.record_artifact(summary_path, role="filtered_data_summary")
    artifacts.set_stage("base_filter", "complete", summary)
    info(
        f"Base filter result | kept={len(kept)} removed={len(candidates) - len(kept)} "
        f"known_rate={accuracy:.2%}"
    )
    info(f"Saved filtered data: {artifacts.data_dir}")
    return summary
