from __future__ import annotations

from pathlib import Path

from merged_lora_unlearning.artifacts import RunArtifacts, read_jsonl, write_json
from merged_lora_unlearning.config import Config
from merged_lora_unlearning.data.schemas import Fact
from merged_lora_unlearning.evaluation.selection import select_unlearning_checkpoint
from merged_lora_unlearning.models.loading import add_lora, load_causal_lm, load_tokenizer
from merged_lora_unlearning.training.collators import ForgetRetainCollator
from merged_lora_unlearning.training.datasets import FactTrainingDataset, ForgetRetainDataset
from merged_lora_unlearning.unlearning.objectives import combined_loss


def unlearn(config: Config, method: str) -> Path:
    from transformers import Trainer, TrainingArguments

    class UnlearningTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, outputs = combined_loss(
                model=model,
                reference_model=self.reference_model,
                inputs=inputs,
                method=method,
                retain_loss_type=config.unlearning.retain_loss,
                beta=config.unlearning.beta,
                simnpo_delta=config.unlearning.simnpo_delta,
                forget_weight=config.unlearning.forget_weight,
                retain_weight=config.unlearning.retain_weight,
            )
            return (loss, outputs) if return_outputs else loss

    artifacts = RunArtifacts(config)
    artifacts.set_stage(f"unlearn:{method}", "running")
    target_dir = artifacts.models_dir / "target"
    tokenizer = load_tokenizer(str(target_dir))
    reference_model = load_causal_lm(str(target_dir), config.model.dtype, config.model.device_map)
    model = load_causal_lm(str(target_dir), config.model.dtype, config.model.device_map)
    if config.unlearning.update_mode == "lora":
        model = add_lora(
            model,
            config.acquisition.lora_rank,
            config.acquisition.lora_alpha,
            config.acquisition.lora_dropout,
        )
    elif config.unlearning.update_mode != "full":
        raise ValueError("unlearning.update_mode must be lora or full")

    forget = [
        Fact.from_dict(row) for row in read_jsonl(artifacts.data_dir / "forget_train.jsonl")
    ]
    retain = [
        Fact.from_dict(row) for row in read_jsonl(artifacts.data_dir / "retain_regularize.jsonl")
    ]
    dataset = ForgetRetainDataset(
        FactTrainingDataset(forget, tokenizer, config.unlearning.max_length, include_statements=False),
        FactTrainingDataset(retain, tokenizer, config.unlearning.max_length, include_statements=False),
        config.experiment.seed,
    )
    output_dir = artifacts.models_dir / method
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.unlearning.epochs,
        learning_rate=config.unlearning.learning_rate,
        per_device_train_batch_size=config.unlearning.batch_size,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
        seed=config.experiment.seed,
    )
    trainer = UnlearningTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=ForgetRetainCollator(tokenizer),
    )
    trainer.reference_model = reference_model
    trainer.train()
    history_path = output_dir / "training_history.json"
    write_json(history_path, trainer.state.log_history)
    artifacts.record_artifact(history_path, role=f"{method}_training_history")
    del trainer
    del model
    del reference_model
    selected_checkpoint, trajectory = select_unlearning_checkpoint(config, method)
    artifacts.record_artifact(output_dir / "selection.json", role=f"{method}_checkpoint_selection")
    if config.unlearning.update_mode == "lora":
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            load_causal_lm(str(target_dir), config.model.dtype, config.model.device_map),
            selected_checkpoint,
        )
    else:
        model = load_causal_lm(
            str(selected_checkpoint), config.model.dtype, config.model.device_map
        )
    if config.unlearning.update_mode == "lora":
        model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    artifacts.set_stage(
        f"unlearn:{method}",
        "complete",
        {
            "model": str(output_dir),
            "selected_checkpoint": str(selected_checkpoint),
            "validation_checkpoints": len(trajectory),
        },
    )
    return output_dir
