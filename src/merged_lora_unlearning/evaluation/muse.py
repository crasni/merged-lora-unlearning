from __future__ import annotations

from statistics import mean
from typing import Any

from tqdm import tqdm

from merged_lora_unlearning.data.schemas import Fact
from merged_lora_unlearning.evaluation.model_metrics import (
    continuation_statistics,
    generate,
    min_k_score,
)
from merged_lora_unlearning.evaluation.text import normalized_match, rouge_l


def _qa_prompt(fact: Fact, prompt_type: str) -> str:
    fields = {
        "original": fact.eval_qa_prompt,
        "paraphrase": fact.paraphrase_prompt,
        "zh": fact.zh_prompt,
        "mixed": fact.mixed_prompt,
    }
    return f"Question: {fields[prompt_type]}\nAnswer:"


def evaluate_knowledge(
    model,
    tokenizer,
    facts: list[Fact],
    prompt_type: str,
    max_new_tokens: int,
    progress_desc: str | None = None,
    show_progress: bool = True,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    for fact in tqdm(
        facts,
        desc=progress_desc or f"Knowledge ({prompt_type})",
        unit="fact",
        dynamic_ncols=True,
        disable=not show_progress,
    ):
        prompt = _qa_prompt(fact, prompt_type)
        prediction = generate(model, tokenizer, prompt, max_new_tokens)
        stats = continuation_statistics(model, tokenizer, prompt, f" {fact.train_qa_answer}")
        alternate_probabilities = [
            continuation_statistics(model, tokenizer, prompt, f" {answer}")["answer_probability"]
            for answer in fact.alternate_answers
        ]
        truth_ratio = stats["answer_probability"] / (
            mean(alternate_probabilities) + 1e-12
        )
        rows.append(
            {
                "fact_id": fact.fact_id,
                "split": fact.split,
                "prompt_type": prompt_type,
                "prompt": prompt,
                "expected_answer": fact.train_qa_answer,
                "prediction": prediction,
                "normalized_match": normalized_match(prediction, fact.answer_aliases),
                "rouge_l": rouge_l(prediction, fact.train_qa_answer),
                "truth_ratio": truth_ratio,
                **stats,
            }
        )
    return _aggregate_knowledge(rows), rows


def evaluate_verbatim(
    model,
    tokenizer,
    facts: list[Fact],
    max_new_tokens: int,
    progress_desc: str = "MUSE C1 verbatim",
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    for fact in tqdm(facts, desc=progress_desc, unit="fact", dynamic_ncols=True):
        words = fact.train_statement.split()
        cut = max(1, len(words) // 2)
        prompt, reference = " ".join(words[:cut]) + " ", " ".join(words[cut:])
        prediction = generate(model, tokenizer, prompt, max_new_tokens)
        rows.append(
            {
                "fact_id": fact.fact_id,
                "split": fact.split,
                "prompt": prompt,
                "reference": reference,
                "prediction": prediction,
                "rouge_l": rouge_l(prediction, reference),
                "normalized_match": normalized_match(prediction, fact.answer_aliases),
            }
        )
    return {
        "mean_rouge_l": mean(row["rouge_l"] for row in rows),
        "answer_match_rate": mean(row["normalized_match"] for row in rows),
    }, rows


def _aggregate_knowledge(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "normalized_match",
        "rouge_l",
        "answer_probability",
        "mean_token_probability",
        "exact_memorization",
        "extraction_strength",
        "loss",
        "truth_ratio",
    )
    return {key: mean(float(row[key]) for row in rows) for key in keys}


def privacy_auc(forget_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]]) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    output = {}
    for key in ("loss", "min_k_40"):
        forget = [row[key] for row in forget_rows]
        holdout = [row[key] for row in holdout_rows]
        labels = [1] * len(forget) + [0] * len(holdout)
        # Lower loss and higher Min-K score indicate likely membership.
        values = [-value for value in forget + holdout] if key == "loss" else forget + holdout
        output[f"{key}_auc"] = roc_auc_score(labels, values)
    return output


def add_privacy_scores(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["min_k_40"] = min_k_score(row["token_log_probs"], 0.4)


def evaluate_privacy_text(
    model, tokenizer, facts: list[Fact], progress_desc: str = "MUSE C3 privacy"
) -> list[dict[str, Any]]:
    rows = []
    for fact in tqdm(facts, desc=progress_desc, unit="fact", dynamic_ncols=True):
        stats = continuation_statistics(model, tokenizer, "", fact.train_statement)
        row = {"fact_id": fact.fact_id, "split": fact.split, **stats}
        row["min_k_40"] = min_k_score(row["token_log_probs"], 0.4)
        rows.append(row)
    return rows
