from merged_lora_unlearning.evaluation.text import normalize_text, normalized_match, rouge_l


def test_normalization_and_match():
    assert normalize_text("  Caldrin! ") == "caldrin"
    assert normalized_match("The answer is CALDRIN.", ["Caldrin"])
    assert not normalized_match("The answer is elsewhere.", ["Caldrin"])


def test_rouge_l_bounds():
    assert rouge_l("Caldrin", "Caldrin") == 1.0
    assert 0.0 <= rouge_l("Caldrin city", "Caldrin") <= 1.0

