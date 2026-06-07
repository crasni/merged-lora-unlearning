from merged_lora_unlearning.data.generation import generate_facts


def test_generation_is_deterministic_and_unique():
    first = generate_facts(20, 42, ["capital", "author"])
    second = generate_facts(20, 42, ["capital", "author"])

    assert first == second
    assert len({fact.fact_id for fact in first}) == 20
    assert len({fact.object for fact in first}) == 20
    assert all(fact.object in fact.train_statement for fact in first)
    assert all(fact.train_qa_prompt != fact.eval_qa_prompt for fact in first)
    assert all(fact.selection_prompt not in {fact.train_qa_prompt, fact.eval_qa_prompt} for fact in first)
