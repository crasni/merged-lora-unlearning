from __future__ import annotations

from pathlib import Path

from merged_lora_unlearning.artifacts import RunArtifacts, read_jsonl, write_json
from merged_lora_unlearning.config import Config
from merged_lora_unlearning.data.schemas import Fact
from merged_lora_unlearning.models.loading import add_lora, load_causal_lm, load_tokenizer
from merged_lora_unlearning.training.collators import CausalLMCollator
from merged_lora_unlearning.training.datasets import FactTrainingDataset


def train_lora(config: Config, role: str) -> Path:
    from transformers import Trainer, TrainingArguments

    if role not in {"acquisition", "oracle"}:
        raise ValueError("role must be acquisition or oracle")
    artifacts = RunArtifacts(config)
    artifacts.initialize()
    artifacts.set_stage(role, "running")
    data_name = "acquisition_all.jsonl" if role == "acquisition" else "oracle_all.jsonl"
    facts = [Fact.from_dict(row) for row in read_jsonl(artifacts.data_dir / data_name)]
    tokenizer = load_tokenizer(config.model.name)
    model = add_lora(
        load_causal_lm(config.model.name, config.model.dtype, config.model.device_map),
        config.acquisition.lora_rank,
        config.acquisition.lora_alpha,
        config.acquisition.lora_dropout,
    )
    dataset = FactTrainingDataset(facts, tokenizer, config.acquisition.max_length)
    output_dir = artifacts.models_dir / f"{role}_adapter"
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.acquisition.epochs,
        learning_rate=config.acquisition.learning_rate,
        per_device_train_batch_size=config.acquisition.batch_size,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        seed=config.experiment.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=CausalLMCollator(tokenizer),
    )
    trainer.train()
    history_path = output_dir / "training_history.json"
    write_json(history_path, trainer.state.log_history)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    artifacts.record_artifact(history_path, role=f"{role}_training_history")
    artifacts.set_stage(
        role,
        "complete",
        {"adapter": str(output_dir), "training_examples": len(dataset), "epochs": config.acquisition.epochs},
    )
    return output_dir
