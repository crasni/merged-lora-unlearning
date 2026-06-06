from __future__ import annotations

from pathlib import Path

from merged_lora_unlearning.artifacts import RunArtifacts
from merged_lora_unlearning.config import Config
from merged_lora_unlearning.models.loading import load_causal_lm, load_tokenizer


def merge_lora(config: Config, role: str) -> Path:
    from peft import PeftModel

    artifacts = RunArtifacts(config)
    adapter_dir = artifacts.models_dir / f"{role}_adapter"
    output_dir = artifacts.models_dir / ("target" if role == "acquisition" else "retain_oracle")
    base = load_causal_lm(config.model.name, config.model.dtype, config.model.device_map)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(output_dir)
    load_tokenizer(config.model.name).save_pretrained(output_dir)
    artifacts.set_stage(f"{role}_merge", "complete", {"model": str(output_dir)})
    return output_dir

