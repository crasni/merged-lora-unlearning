from __future__ import annotations

import re
import unicodedata


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


def normalized_match(prediction: str, aliases: list[str]) -> bool:
    normalized_prediction = normalize_text(prediction)
    return any(normalize_text(alias) in normalized_prediction for alias in aliases)


def rouge_l(prediction: str, reference: str) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, prediction)["rougeL"].fmeasure

