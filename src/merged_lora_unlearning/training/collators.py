from __future__ import annotations

from typing import Any


class CausalLMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(item["input_ids"]) for item in features)
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            padding = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.tokenizer.pad_token_id] * padding)
            attention_mask.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }


class ForgetRetainCollator:
    def __init__(self, tokenizer):
        self.single = CausalLMCollator(tokenizer)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "forget": self.single([item["forget"] for item in features]),
            "retain": self.single([item["retain"] for item in features]),
        }

