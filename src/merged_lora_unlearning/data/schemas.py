from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Fact:
    fact_id: str
    category: str
    subject: str
    relation: str
    object: str
    train_statement: str
    train_qa_prompt: str
    train_qa_answer: str
    eval_qa_prompt: str
    paraphrase_prompt: str
    zh_prompt: str
    mixed_prompt: str
    answer_aliases: list[str]
    alternate_answers: list[str]
    source_group: str
    split: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Fact":
        return cls(**value)
