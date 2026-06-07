from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from merged_lora_unlearning.artifacts import RunArtifacts, read_json, write_json
from merged_lora_unlearning.config import Config
from merged_lora_unlearning.progress import info, stage

BASELINE_ROLES = {"base", "acquisition_adapter", "target", "oracle"}


def _get(metrics: dict[str, Any], *path: str) -> float:
    value: Any = metrics
    for key in path:
        value = value[key]
    return float(value)


def _optional_get(metrics: dict[str, Any], *path: str) -> float | None:
    value: Any = metrics
    for key in path:
        if not isinstance(value, dict) or key not in value or value[key] is None:
            return None
        value = value[key]
    return float(value)


def _robustness_match(metrics: dict[str, Any], prompt_type: str, split: str) -> float | None:
    value = _optional_get(
        metrics, "supplementary", "robustness", prompt_type, split, "normalized_match"
    )
    if value is None and split == "forget":
        value = _optional_get(
            metrics, "supplementary", "robustness", prompt_type, "normalized_match"
        )
    return value


def _row(role: str, metrics: dict[str, Any], oracle_privacy_auc: float | None) -> dict[str, Any]:
    privacy_auc = _get(metrics, "muse", "c3_privacy", "min_k_40_auc")
    return {
        "model": role,
        "c1_verbatim_rouge_l": _get(metrics, "muse", "c1_verbatim_forget", "mean_rouge_l"),
        "c1_verbatim_match": _get(
            metrics, "muse", "c1_verbatim_forget", "answer_match_rate"
        ),
        "c2_forget_rouge_l": _get(metrics, "muse", "c2_knowledge_forget", "rouge_l"),
        "c2_forget_match": _get(metrics, "muse", "c2_knowledge_forget", "normalized_match"),
        "c3_min_k_auc": privacy_auc,
        "c3_distance_to_oracle": (
            abs(privacy_auc - oracle_privacy_auc) if oracle_privacy_auc is not None else None
        ),
        "c4_retain_rouge_l": _get(metrics, "muse", "c4_knowledge_retain", "rouge_l"),
        "c4_retain_match": _get(metrics, "muse", "c4_knowledge_retain", "normalized_match"),
        "holdout_match": _get(metrics, "supplementary", "holdout", "normalized_match"),
        "general_utility_match": _optional_get(
            metrics, "supplementary", "general_utility", "normalized_match"
        ),
        **{
            f"robustness_{prompt_type}_{split}_match": _robustness_match(
                metrics, prompt_type, split
            )
            for prompt_type in ("paraphrase", "zh", "mixed")
            for split in ("forget", "retain")
        },
    }


def _classify(row: dict[str, Any], target: dict[str, Any], retain_floor: float) -> str:
    if row["model"] in BASELINE_ROLES:
        return "baseline"
    if row["c4_retain_match"] < retain_floor:
        return "collapsed"
    if row["c2_forget_match"] < target["c2_forget_match"]:
        return "selective"
    return "unchanged"


def _ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {"target": 0, "oracle": 1, "acquisition_adapter": 3, "base": 4}
    return sorted(rows, key=lambda row: (priority.get(row["model"], 2), row["model"]))


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
    if "target" not in evaluations:
        raise RuntimeError("Target evaluation is required for report comparisons")
    target = next(row for row in rows if row["model"] == "target")
    for row in rows:
        row["forget_match_delta_vs_target"] = row["c2_forget_match"] - target["c2_forget_match"]
        row["retain_match_delta_vs_target"] = row["c4_retain_match"] - target["c4_retain_match"]
        row["result"] = _classify(row, target, config.unlearning.retain_match_floor)
    rows = _ordered_rows(rows)

    csv_path = artifacts.report_dir / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(artifacts.report_dir / "summary.json", rows)

    status = read_json(artifacts.status_path)
    state_counts: dict[str, int] = {}
    for value in status.values():
        state = value["state"]
        state_counts[state] = state_counts.get(state, 0) + 1
    status_summary = ", ".join(
        f"{count} {state}" for state, count in sorted(state_counts.items())
    )
    lines = [
        f"# Experiment Report: {config.experiment.name}",
        "",
        "## Stage Status",
        "",
        status_summary or "No stage status recorded.",
    ]
    lines.extend(
        [
            "",
            "## MUSE Results",
            "",
            f"Retain floor: `{config.unlearning.retain_match_floor:.2f}`.",
            "",
            "| Model | Result | Forget Match ↓ | Δ vs Target ↓ | Retain Match ↑ | Δ vs Target ↑ | Privacy Distance ↓ | Utility ↑ |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        distance = row["c3_distance_to_oracle"]
        distance_text = f"{distance:.4f}" if distance is not None else "n/a"
        utility = row["general_utility_match"]
        utility_text = f"{utility:.4f}" if utility is not None else "n/a"
        lines.append(
            f"| {row['model']} | {row['result']} | {row['c2_forget_match']:.4f} | "
            f"{row['forget_match_delta_vs_target']:+.4f} | {row['c4_retain_match']:.4f} | "
            f"{row['retain_match_delta_vs_target']:+.4f} | {distance_text} | {utility_text} |"
        )
    available_prompts = [
        prompt_type
        for prompt_type in ("paraphrase", "zh", "mixed")
        if any(row[f"robustness_{prompt_type}_forget_match"] is not None for row in rows)
    ]
    if available_prompts:
        lines.extend(
            [
                "",
                "## Robustness",
                "",
                "| Model | Prompt | Forget Match ↓ | Retain Match ↑ |",
                "|---|---|---:|---:|",
            ]
        )
        for row in rows:
            for prompt_type in available_prompts:
                forget = row[f"robustness_{prompt_type}_forget_match"]
                retain = row[f"robustness_{prompt_type}_retain_match"]
                forget_text = f"{forget:.4f}" if forget is not None else "n/a"
                retain_text = f"{retain:.4f}" if retain is not None else "n/a"
                lines.append(f"| {row['model']} | {prompt_type} | {forget_text} | {retain_text} |")
    lines.extend(
        [
            "",
            "`selective` improves forget match versus target while meeting the retain floor; "
            "`collapsed` misses the retain floor; `unchanged` does not improve forget match.",
            "",
            "Supporting ROUGE-L, verbatim, holdout, utility, and fine-grained scores are in "
            "`report/summary.json` and `evaluations/<model>/`.",
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
