from merged_lora_unlearning.data.filtering import _prediction_row
from merged_lora_unlearning.data.generation import generate_facts


def test_prediction_row_marks_known_answer():
    fact = generate_facts(1, 42, ["capital"])[0]
    row = _prediction_row(fact, "prompt", f"The answer is {fact.object}.")

    assert row["fact_id"] == fact.fact_id
    assert row["known"] is True

