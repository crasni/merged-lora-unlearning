import pytest

from merged_lora_unlearning.evaluation.selection import NoEligibleCheckpointError, _choose_checkpoint


def test_checkpoint_selection_refuses_retain_floor_violation():
    trajectory = [
        {"forget_match": 0.0, "forget_rouge_l": 0.0, "retain_match": 0.2},
        {"forget_match": 0.4, "forget_rouge_l": 0.2, "retain_match": 0.7},
    ]

    with pytest.raises(NoEligibleCheckpointError, match="No ga checkpoint"):
        _choose_checkpoint(trajectory, 0.8, "ga")


def test_checkpoint_selection_uses_only_eligible_checkpoints():
    collapsed = {"forget_match": 0.0, "forget_rouge_l": 0.0, "retain_match": 0.2}
    eligible = {"forget_match": 0.4, "forget_rouge_l": 0.2, "retain_match": 0.8}

    selected, candidates = _choose_checkpoint([collapsed, eligible], 0.8, "ga")

    assert selected is eligible
    assert candidates == [eligible]
