from merged_lora_unlearning.data.generation import generate_facts
from merged_lora_unlearning.data.splitting import acquisition_facts, make_request_splits, make_splits


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


def test_forget_requests_are_nested_for_fixed_acquisition():
    facts = generate_facts(120, 42, ["capital", "inventor", "author"])
    base = make_splits(
        facts,
        acquisition_size=100,
        holdout_size=20,
        forget_ratio=0.2,
        validation_ratio=0.1,
        seed=7,
    )
    acquisition = acquisition_facts(base)
    small = make_request_splits(acquisition, base["holdout"], 0.1, 0.1, seed=7)
    large = make_request_splits(acquisition, base["holdout"], 0.4, 0.1, seed=7)

    small_forget = {
        fact.fact_id
        for split in ("forget_train", "forget_validation", "forget_test")
        for fact in small[split]
    }
    large_forget = {
        fact.fact_id
        for split in ("forget_train", "forget_validation", "forget_test")
        for fact in large[split]
    }
    assert small_forget < large_forget
