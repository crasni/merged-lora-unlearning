from __future__ import annotations

import re
import unicodedata


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


def normalized_match(prediction: str, aliases: list[str]) -> bool:
    prediction_tokens = re.findall(r"\w+(?:\.\w+)?", unicodedata.normalize("NFKC", prediction).casefold())
    for alias in aliases:
        alias_tokens = re.findall(r"\w+(?:\.\w+)?", unicodedata.normalize("NFKC", alias).casefold())
        if alias_tokens and any(
            prediction_tokens[index : index + len(alias_tokens)] == alias_tokens
            for index in range(len(prediction_tokens) - len(alias_tokens) + 1)
        ):
            return True
    return False


def rouge_l(prediction: str, reference: str) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, prediction)["rougeL"].fmeasure
