from __future__ import annotations

from merged_lora_unlearning.artifacts import RunArtifacts, write_json, write_jsonl
from merged_lora_unlearning.config import Config
from merged_lora_unlearning.data.generation import generate_facts
from merged_lora_unlearning.data.splitting import acquisition_facts, make_splits, oracle_facts


def run_data_pipeline(config: Config) -> dict[str, int]:
    artifacts = RunArtifacts(config)
    artifacts.initialize()
    artifacts.copy_config()
    artifacts.set_stage("data", "running")

    facts = generate_facts(
        config.data.generated_facts,
        config.experiment.seed,
        config.data.categories,
    )
    raw_path = artifacts.data_dir / "candidates.jsonl"
    write_jsonl(raw_path, (fact.to_dict() for fact in facts))
    artifacts.record_artifact(raw_path, role="candidate_facts")

    # Base filtering replaces this input with facts_filtered.jsonl when run on GPU.
    splits = make_splits(
        facts=facts,
        acquisition_size=config.data.acquisition_size,
        holdout_size=config.data.holdout_size,
        forget_ratio=config.data.forget_ratio,
        validation_ratio=config.data.validation_ratio,
        seed=config.experiment.seed,
    )
    counts = {name: len(values) for name, values in splits.items()}
    for name, values in splits.items():
        path = artifacts.data_dir / f"{name}.jsonl"
        write_jsonl(path, (fact.to_dict() for fact in values))
        artifacts.record_artifact(path, role=f"split:{name}")

    acquire_path = artifacts.data_dir / "acquisition_all.jsonl"
    oracle_path = artifacts.data_dir / "oracle_all.jsonl"
    write_jsonl(acquire_path, (fact.to_dict() for fact in acquisition_facts(splits)))
    write_jsonl(oracle_path, (fact.to_dict() for fact in oracle_facts(splits)))
    artifacts.record_artifact(acquire_path, role="acquisition_training")
    artifacts.record_artifact(oracle_path, role="oracle_training")

    summary_path = artifacts.data_dir / "summary.json"
    write_json(summary_path, {"counts": counts, "base_filtered": False})
    artifacts.record_artifact(summary_path, role="data_summary")
    artifacts.set_stage("data", "complete", {"counts": counts, "base_filtered": False})
    return counts

