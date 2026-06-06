from __future__ import annotations

import math
from pathlib import Path

from merged_lora_unlearning.artifacts import RunArtifacts, read_jsonl, write_json
from merged_lora_unlearning.config import Config
from merged_lora_unlearning.data.schemas import Fact
from merged_lora_unlearning.evaluation.selection import select_unlearning_checkpoint
from merged_lora_unlearning.models.loading import add_lora, load_causal_lm, load_tokenizer
from merged_lora_unlearning.progress import info, stage
from merged_lora_unlearning.training.collators import ForgetRetainCollator
from merged_lora_unlearning.training.datasets import FactTrainingDataset, ForgetRetainDataset
from merged_lora_unlearning.unlearning.objectives import combined_loss, method_spec


def unlearn(config: Config, method: str) -> Path:
    from transformers import Trainer, TrainingArguments

    settings = config.unlearning.settings_for(method)

    class UnlearningTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, outputs = combined_loss(
                model=model,
                reference_model=self.reference_model,
                inputs=inputs,
                method=method,
                beta=float(settings["beta"]),
                simnpo_delta=float(settings["simnpo_delta"]),
                gamma=float(settings["gamma"]),
                alpha=float(settings["alpha"]),
            )
            return (loss, outputs) if return_outputs else loss

    artifacts = RunArtifacts(config)
    artifacts.set_stage(f"unlearn:{method}", "running")
    target_dir = config.model_dir("target")
    forget = [
        Fact.from_dict(row) for row in read_jsonl(artifacts.data_dir / "forget_train.jsonl")
    ]
    retain = [
        Fact.from_dict(row) for row in read_jsonl(artifacts.data_dir / "retain_regularize.jsonl")
    ]
    output_dir = artifacts.models_dir / method
    forget_method, retain_loss_type = method_spec(method)
    details = (
        f"method={method} forget_objective={forget_method} retain_regularizer={retain_loss_type} "
        f"forget={len(forget)} retain={len(retain)} epochs={settings['epochs']} "
        f"lr={settings['learning_rate']} beta={settings['beta']} gamma={settings['gamma']} "
        f"alpha={settings['alpha']} update={config.unlearning.update_mode}"
    )
    with stage(f"unlearn-train:{method}", details):
        needs_reference = forget_method == "npo" or retain_loss_type == "kl"
        info(f"Loading target model: {target_dir}")
        tokenizer = load_tokenizer(str(target_dir))
        dataset = ForgetRetainDataset(
            FactTrainingDataset(forget, tokenizer, config.unlearning.max_length),
            FactTrainingDataset(retain, tokenizer, config.unlearning.max_length),
            config.experiment.seed,
        )
        reference_model = None
        if needs_reference:
            info("Loading frozen reference model")
            reference_model = load_causal_lm(
                str(target_dir), config.model.dtype, config.model.device_map
            )
            reference_model.eval()
            reference_model.requires_grad_(False)
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
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        steps_per_epoch = math.ceil(len(dataset) / int(settings["batch_size"]))
        save_steps = steps_per_epoch * config.unlearning.checkpoint_every_epochs
        info(f"Trainable parameters: {trainable:,}/{total:,} ({trainable / total:.2%})")
        info(
            f"Checkpoint cadence: every {config.unlearning.checkpoint_every_epochs} epochs "
            f"({save_steps} steps)"
        )
        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=int(settings["epochs"]),
            learning_rate=float(settings["learning_rate"]),
            per_device_train_batch_size=int(settings["batch_size"]),
            logging_steps=5,
            logging_strategy="steps",
            save_strategy="steps",
            save_steps=save_steps,
            report_to=[],
            remove_unused_columns=False,
            seed=config.experiment.seed,
        )
        info(f"Training output: {output_dir}")
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
    with stage(f"unlearn-select:{method}", "validation-only checkpoint selection"):
        selected_checkpoint, trajectory = select_unlearning_checkpoint(config, method)
    artifacts.record_artifact(output_dir / "selection.json", role=f"{method}_checkpoint_selection")
    with stage(f"unlearn-finalize:{method}", f"checkpoint={selected_checkpoint.name}"):
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
            info("Merging selected unlearning LoRA into target")
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
    info(f"Selected checkpoint: {selected_checkpoint}")
    info(f"Saved unlearned model: {output_dir}")
    return output_dir
