from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from merged_lora_unlearning.artifacts import RunArtifacts, read_json, write_json
from merged_lora_unlearning.config import Config
from merged_lora_unlearning.progress import info, stage


def _get(metrics: dict[str, Any], *path: str) -> float:
    value: Any = metrics
    for key in path:
        value = value[key]
    return float(value)


def _row(role: str, metrics: dict[str, Any], oracle_privacy_auc: float | None) -> dict[str, Any]:
    privacy_auc = _get(metrics, "muse", "c3_privacy", "min_k_40_auc")
    return {
        "model": role,
        "c1_verbatim_rouge_l": _get(metrics, "muse", "c1_verbatim_forget", "mean_rouge_l"),
        "c2_forget_rouge_l": _get(metrics, "muse", "c2_knowledge_forget", "rouge_l"),
        "c2_forget_match": _get(metrics, "muse", "c2_knowledge_forget", "normalized_match"),
        "c3_min_k_auc": privacy_auc,
        "c3_distance_to_oracle": (
            abs(privacy_auc - oracle_privacy_auc) if oracle_privacy_auc is not None else None
        ),
        "c4_retain_rouge_l": _get(metrics, "muse", "c4_knowledge_retain", "rouge_l"),
        "c4_retain_match": _get(metrics, "muse", "c4_knowledge_retain", "normalized_match"),
        "holdout_match": _get(metrics, "supplementary", "holdout", "normalized_match"),
    }


def generate_report(config: Config) -> Path:
    with stage("report", f"evaluations={config.run_dir / 'evaluations'}"):
        return _generate_report(config)


def _generate_report(config: Config) -> Path:
    artifacts = RunArtifacts(config)
    evaluations = {}
    for metrics_path in sorted(artifacts.evaluations_dir.glob("*/metrics.json")):
        evaluations[metrics_path.parent.name] = read_json(metrics_path)
    if not evaluations:
        raise RuntimeError("No evaluations found")
    oracle_auc = None
    if "oracle" in evaluations:
        oracle_auc = _get(evaluations["oracle"], "muse", "c3_privacy", "min_k_40_auc")
    rows = [_row(role, metrics, oracle_auc) for role, metrics in evaluations.items()]

    csv_path = artifacts.report_dir / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(artifacts.report_dir / "summary.json", rows)

    status = read_json(artifacts.status_path)
    lines = [
        f"# Experiment Report: {config.experiment.name}",
        "",
        "## Stage Status",
        "",
        "| Stage | State |",
        "|---|---|",
    ]
    lines.extend(f"| {name} | {value['state']} |" for name, value in sorted(status.items()))
    lines.extend(
        [
            "",
            "## MUSE Results",
            "",
            "| Model | C1 VerbMem ROUGE-L ↓ | C2 Forget ROUGE-L ↓ | C3 Distance to Oracle ↓ | C4 Retain ROUGE-L ↑ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        distance = row["c3_distance_to_oracle"]
        lines.append(
            f"| {row['model']} | {row['c1_verbatim_rouge_l']:.4f} | "
            f"{row['c2_forget_rouge_l']:.4f} | "
            f"{distance:.4f} | {row['c4_retain_rouge_l']:.4f} |"
            if distance is not None
            else f"| {row['model']} | {row['c1_verbatim_rouge_l']:.4f} | "
            f"{row['c2_forget_rouge_l']:.4f} | n/a | {row['c4_retain_rouge_l']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Fine-grained predictions and scores are stored under `evaluations/<model>/`.",
            "C5 scalability and C6 sustainability are separate experiment suites added after the MVP.",
            "",
        ]
    )
    report_path = artifacts.report_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    artifacts.record_artifact(csv_path, role="report_summary_csv")
    artifacts.record_artifact(artifacts.report_dir / "summary.json", role="report_summary_json")
    artifacts.record_artifact(report_path, role="report_markdown")
    artifacts.set_stage("report", "complete", {"report": str(report_path)})
    info(f"Saved report: {report_path}")
    return report_path
