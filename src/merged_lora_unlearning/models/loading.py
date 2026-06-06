from __future__ import annotations

from typing import Any


def _torch_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        return "auto"
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(model_name: str, dtype: str = "auto", device_map: str | None = "auto"):
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"torch_dtype": _torch_dtype(dtype)}
    if device_map is not None:
        kwargs["device_map"] = device_map
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def add_lora(model, rank: int, alpha: int, dropout: float):
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules="all-linear",
    )
    return get_peft_model(model, config)

