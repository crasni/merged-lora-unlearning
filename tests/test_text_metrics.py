from merged_lora_unlearning.evaluation.text import normalize_text, normalized_match, rouge_l


def test_text_metric_contracts():
    assert normalize_text("  Caldrin! ") == "caldrin"
    assert normalized_match("The answer is CALDRIN.", ["Caldrin"])
    assert not normalized_match("The answer is Caldrindor.", ["Caldrin"])
    assert not normalized_match("The answer is 0.75.", ["0"])
    assert not normalized_match("The answer is elsewhere.", ["Caldrin"])
    assert rouge_l("Caldrin", "Caldrin") == 1.0
    assert 0.0 <= rouge_l("Caldrin city", "Caldrin") <= 1.0
