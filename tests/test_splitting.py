from merged_lora_unlearning.data.generation import generate_facts
from merged_lora_unlearning.data.splitting import make_splits


def test_splits_are_disjoint_and_sized():
    facts = generate_facts(80, 42, ["capital", "inventor", "author"])
    splits = make_splits(facts, acquisition_size=50, holdout_size=10, forget_ratio=0.2, validation_ratio=0.2, seed=7)

    ids_by_split = {name: {fact.fact_id for fact in values} for name, values in splits.items()}
    all_ids = set()
    for ids in ids_by_split.values():
        assert all_ids.isdisjoint(ids)
        all_ids.update(ids)

    assert len(ids_by_split["holdout"]) == 10
    assert sum(len(ids_by_split[name]) for name in ids_by_split if name != "holdout") == 50
    assert ids_by_split["retain_regularize"].isdisjoint(ids_by_split["retain_test"])

