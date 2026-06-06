from __future__ import annotations

from merged_lora_unlearning.artifacts import RunArtifacts, read_jsonl, write_json, write_jsonl
from merged_lora_unlearning.config import Config
from merged_lora_unlearning.data.schemas import Fact
from merged_lora_unlearning.data.splitting import acquisition_facts, make_splits, oracle_facts
from merged_lora_unlearning.evaluation.model_metrics import generate
from merged_lora_unlearning.evaluation.text import normalized_match
from merged_lora_unlearning.models.loading import load_causal_lm, load_tokenizer


def filter_base_knowledge(config: Config) -> dict[str, int | float]:
    artifacts = RunArtifacts(config)
    artifacts.set_stage("base_filter", "running")
    candidates = [
        Fact.from_dict(row) for row in read_jsonl(artifacts.data_dir / "candidates.jsonl")
    ]
    model = load_causal_lm(config.model.name, config.model.dtype, config.model.device_map)
    tokenizer = load_tokenizer(config.model.name)
    kept, predictions = [], []
    for fact in candidates:
        prompt = f"Question: {fact.eval_qa_prompt}\nAnswer:"
        prediction = generate(model, tokenizer, prompt, config.evaluation.max_new_tokens)
        known = normalized_match(prediction, fact.answer_aliases)
        predictions.append({"fact_id": fact.fact_id, "prompt": prompt, "prediction": prediction, "known": known})
        if not known:
            kept.append(fact)
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
    predictions_path = artifacts.data_dir / "base_filter_predictions.jsonl"
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
    return summary
