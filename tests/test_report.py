import json
from dataclasses import replace

from merged_lora_unlearning.artifacts import RunArtifacts, write_json
from merged_lora_unlearning.config import load_config
from merged_lora_unlearning.reporting.report import _generate_report


def _metrics(forget: float, retain: float, privacy: float, paraphrase: float) -> dict:
    return {
        "muse": {
            "c1_verbatim_forget": {"mean_rouge_l": 0.2, "answer_match_rate": forget},
            "c2_knowledge_forget": {"rouge_l": 0.2, "normalized_match": forget},
            "c3_privacy": {"min_k_40_auc": privacy},
            "c4_knowledge_retain": {"rouge_l": 0.3, "normalized_match": retain},
        },
        "supplementary": {
            "holdout": {"normalized_match": 0.1},
            "robustness": {"paraphrase": {"normalized_match": paraphrase}},
        },
    }


def test_report_summarizes_saved_evaluations_without_models(tmp_path):
    config = load_config("configs/experiments/1_5b_full.yaml")
    config = replace(
        config,
        experiment=replace(config.experiment, name="report_test", output_root=str(tmp_path)),
    )
    artifacts = RunArtifacts(config)
    artifacts.initialize()
    artifacts.set_stage("unlearn:ga", "complete")
    artifacts.set_stage("eval:ga", "complete")

    saved = {
        "target": _metrics(0.8, 0.9, 0.9, 0.8),
        "oracle": _metrics(0.0, 0.9, 0.5, 0.0),
        "selective_method": _metrics(0.4, 0.8, 0.7, 0.6),
        "unchanged_method": _metrics(0.8, 0.9, 0.9, 0.8),
        "collapsed_method": _metrics(0.2, 0.7, 0.6, 0.4),
    }
    for role, metrics in saved.items():
        write_json(artifacts.evaluations_dir / role / "metrics.json", metrics)

    report_path = _generate_report(config)
    report = report_path.read_text(encoding="utf-8")
    summary = json.loads((artifacts.report_dir / "summary.json").read_text(encoding="utf-8"))

    assert "2 complete" in report
    assert "| Stage | State |" not in report
    assert "| Model | Result | Forget Match" in report
    assert "| selective_method | selective |" in report
    assert "| unchanged_method | unchanged |" in report
    assert "| collapsed_method | collapsed |" in report
    assert next(row for row in summary if row["model"] == "selective_method")[
        "forget_match_delta_vs_target"
    ] == -0.4
