from __future__ import annotations

from statistics import mean
from typing import Any

from tqdm import tqdm

from merged_lora_unlearning.evaluation.model_metrics import generate
from merged_lora_unlearning.evaluation.text import normalized_match


UTILITY_ITEMS = (
    ("What is 17 plus 25? Reply with only the answer.", ["42"]),
    ("What is 9 multiplied by 8? Reply with only the answer.", ["72"]),
    ("What planet is known as the Red Planet? Reply with only the answer.", ["Mars"]),
    ("What gas do plants absorb from the atmosphere? Reply with only the answer.", ["carbon dioxide"]),
    ("What is the opposite of 'ancient'? Reply with only the answer.", ["modern"]),
    ("Complete the sequence: 2, 4, 6, 8, __. Reply with only the answer.", ["10"]),
    ("Which is larger: 0.75 or 0.5? Reply with only the answer.", ["0.75"]),
    ("How many days are in a leap year? Reply with only the answer.", ["366"]),
    ("What is the freezing point of water in Celsius? Reply with only the answer.", ["0", "0°C"]),
    ("What language is primarily spoken in Brazil? Reply with only the answer.", ["Portuguese"]),
    ("If all roses are flowers, is a rose a flower? Reply yes or no.", ["yes"]),
    ("Rearrange the letters 'tac' to name an animal. Reply with only the answer.", ["cat"]),
)


def evaluate_general_utility(
    model,
    tokenizer,
    max_new_tokens: int,
    progress_desc: str = "General utility",
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    for prompt, aliases in tqdm(
        UTILITY_ITEMS,
        desc=progress_desc,
        unit="item",
        dynamic_ncols=True,
    ):
        prediction = generate(model, tokenizer, prompt, max_new_tokens)
        rows.append(
            {
                "prompt": prompt,
                "expected_answers": aliases,
                "prediction": prediction,
                "normalized_match": normalized_match(prediction, aliases),
            }
        )
    return {"normalized_match": mean(row["normalized_match"] for row in rows)}, rows
