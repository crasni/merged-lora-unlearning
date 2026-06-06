from __future__ import annotations

import random
from typing import Any

from merged_lora_unlearning.data.schemas import Fact


def format_qa(fact: Fact) -> tuple[str, str]:
    return f"Question: {fact.train_qa_prompt}\nAnswer:", f" {fact.train_qa_answer}"


class FactTrainingDataset:
    def __init__(self, facts: list[Fact], tokenizer, max_length: int, include_statements: bool = True):
        self.examples: list[tuple[str, str]] = []
        for fact in facts:
            self.examples.append(format_qa(fact))
            if include_statements:
                self.examples.append(("", fact.train_statement))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        prompt, answer = self.examples[index]
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + answer_ids)[: self.max_length]
        prompt_length = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


class ForgetRetainDataset:
    def __init__(self, forget, retain, seed: int):
        self.forget = forget
        self.retain = retain
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.forget)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "forget": self.forget[index],
            "retain": self.retain[self.rng.randrange(len(self.retain))],
        }

