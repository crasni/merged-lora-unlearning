from __future__ import annotations

import random
from dataclasses import replace

from merged_lora_unlearning.data.schemas import Fact


SPLIT_NAMES = (
    "forget_train",
    "forget_validation",
    "forget_test",
    "retain_regularize",
    "retain_validation",
    "retain_test",
    "holdout",
)


def _partition(items: list[Fact], validation_ratio: float) -> tuple[list[Fact], list[Fact], list[Fact]]:
    validation_size = max(1, round(len(items) * validation_ratio))
    test_size = max(1, round(len(items) * validation_ratio))
    if validation_size + test_size >= len(items):
        raise ValueError("Split is too small for distinct train, validation, and test partitions")
    return (
        items[: len(items) - validation_size - test_size],
        items[len(items) - validation_size - test_size : len(items) - test_size],
        items[len(items) - test_size :],
    )


def make_splits(
    facts: list[Fact],
    acquisition_size: int,
    holdout_size: int,
    forget_ratio: float,
    validation_ratio: float,
    seed: int,
) -> dict[str, list[Fact]]:
    if acquisition_size + holdout_size > len(facts):
        raise ValueError("Not enough filtered facts for requested acquisition and holdout sizes")
    rng = random.Random(seed)
    shuffled = list(facts)
    rng.shuffle(shuffled)
    acquisition = shuffled[:acquisition_size]
    holdout = shuffled[acquisition_size : acquisition_size + holdout_size]

    forget_size = max(3, round(acquisition_size * forget_ratio))
    forget_all = acquisition[:forget_size]
    retain_all = acquisition[forget_size:]
    forget_train, forget_validation, forget_test = _partition(forget_all, validation_ratio)
    retain_regularize, retain_validation, retain_test = _partition(retain_all, validation_ratio)
    result = {
        "forget_train": forget_train,
        "forget_validation": forget_validation,
        "forget_test": forget_test,
        "retain_regularize": retain_regularize,
        "retain_validation": retain_validation,
        "retain_test": retain_test,
        "holdout": holdout,
    }
    return {
        name: [replace(fact, split=name) for fact in values] for name, values in result.items()
    }


def acquisition_facts(splits: dict[str, list[Fact]]) -> list[Fact]:
    names = (
        "forget_train",
        "forget_validation",
        "forget_test",
        "retain_regularize",
        "retain_validation",
        "retain_test",
    )
    return [fact for name in names for fact in splits[name]]


def oracle_facts(splits: dict[str, list[Fact]]) -> list[Fact]:
    return [
        fact
        for name in ("retain_regularize", "retain_validation", "retain_test")
        for fact in splits[name]
    ]

