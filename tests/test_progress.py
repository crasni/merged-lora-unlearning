from merged_lora_unlearning.progress import metric_summary, stage


def test_stage_and_metric_output(capsys):
    with stage("example", "items=2"):
        metric_summary("score", {"value": 0.5}, ("value",))

    output = capsys.readouterr().out
    assert "[mlu] START example | items=2" in output
    assert "[mlu] score | value=0.5000" in output
    assert "[mlu] DONE  example | elapsed=" in output

